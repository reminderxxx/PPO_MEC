#!/usr/bin/env python3
"""Freeze one unsigned G14R22-B public non-holdout background acceptance job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_g14r22b_background_job import HOST_VERSION, canonical_sha256, file_sha256
from scripts.run_g14r22a_real_non_holdout_acceptance import build_package


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=checkout, text=True).strip()


def freeze_python_executable(path: Path) -> tuple[Path, Path]:
    """Preserve the invoked venv path while also freezing its resolved target."""
    if not path.is_absolute():
        raise ValueError("frozen Python executable path must be absolute")
    lexical = Path(os.path.abspath(str(path)))
    invoked = Path(os.path.abspath(sys.executable))
    if invoked != lexical:
        raise ValueError("builder interpreter must exactly equal the frozen Python executable")
    if not lexical.is_file():
        raise ValueError("frozen Python executable does not exist")
    return lexical, lexical.resolve(strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--scientific-work-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkout = args.executor_checkout.resolve()
    python, resolved_python = freeze_python_executable(args.python_executable)
    package_root = args.package_root.resolve()
    work_root = args.scientific_work_root.resolve()
    job_root = args.job_root.resolve()
    if package_root.exists() or work_root.exists() or job_root.exists():
        raise ValueError("package, scientific work, and job roots must all be absent")
    if git(checkout, "rev-parse", "HEAD") != args.executor_commit:
        raise ValueError("executor commit mismatch")
    if git(checkout, "status", "--porcelain"):
        raise ValueError("executor checkout must be clean")
    tree = git(checkout, "rev-parse", "HEAD^{tree}")
    package_root.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat()
    request: dict[str, Any] = {
        "request_version": "g14r22b_public_non_holdout_background_acceptance_v1",
        "created_at": created_at,
        "status": "UNSIGNED_TEST_ONLY",
        "grant_signed": False,
        "token_issued": False,
        "holdout_opened": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "formal_rollout_rerun_allowed": False,
        "automatic_retry_count": 0,
        "acceptance_scope": {
            "split": "formal_public_non_holdout",
            "agents": ["reactive_lru", "sa_ghmappo"],
            "seeds": [7],
            "capacities": ["constrained_288mb", "medium_576mb", "relaxed_864mb"],
            "expected_benchmark_rows": 216,
            "expected_statistics_rows": 6,
            "purpose": "capacity identity producer/CSV/statistics closure only",
        },
        "executor": {
            "checkout": str(checkout),
            "commit": args.executor_commit,
            "git_tree": tree,
            "python_executable": str(python),
            "python_resolved_executable": str(resolved_python),
        },
    }
    request["request_canonical_sha256"] = canonical_sha256(request)
    request_path = package_root / "background_acceptance_request_unsigned.json"
    write_json(request_path, request)
    scientific_package = build_package(
        executor_checkout=checkout,
        executor_commit=args.executor_commit,
        work_root=work_root,
    )
    scientific_package_path = package_root / "scientific_command_package.json"
    write_json(scientific_package_path, scientific_package)
    command = [
        str(python),
        str(checkout / "scripts/run_g14r22a_real_non_holdout_acceptance.py"),
        "--executor-checkout", str(checkout),
        "--executor-commit", args.executor_commit,
        "--work-root", str(work_root),
        "--prepared-command-package-path", str(scientific_package_path),
        "--background-request-path", str(request_path),
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(checkout),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    job_package: dict[str, Any] = {
        "host_version": HOST_VERSION,
        "job_id": "g14r22b_real_non_holdout_capacity_identity_v1",
        "executor_checkout": str(checkout),
        "executor_commit": args.executor_commit,
        "executor_git_tree": tree,
        "cwd": str(checkout),
        "python_executable": str(python),
        "python_resolved_executable": str(resolved_python),
        "environment": environment,
        "command": command,
        "request": {"path": str(request_path), "sha256": file_sha256(request_path)},
        "scientific_command_package": {
            "path": str(scientific_package_path),
            "sha256": file_sha256(scientific_package_path),
        },
        "terminal_contract": {
            "scientific_receipt_path": str(work_root / "acceptance_receipt.json"),
            "integrity_manifest_path": str(
                work_root / "public_entry_output/artifact_integrity_manifest.json"
            ),
            "integrity_root": str(work_root / "public_entry_output/published"),
            "expected_receipt_bindings": {
                "executor_commit": args.executor_commit,
                "executor_git_tree": tree,
                "command_package_sha256": file_sha256(scientific_package_path),
                "background_request_sha256": file_sha256(request_path),
            },
        },
        "scientific_work_root": str(work_root),
        "job_root": str(job_root),
        "automatic_retry_count": 0,
        "holdout_opened": False,
    }
    job_package["job_package_sha256"] = canonical_sha256(job_package)
    job_package_path = package_root / "background_job_package.json"
    write_json(job_package_path, job_package)
    receipt = {
        "freeze_version": "g14r22b_background_acceptance_freeze_v1",
        "created_at": created_at,
        "request": {"path": str(request_path), "sha256": file_sha256(request_path)},
        "scientific_command_package": {
            "path": str(scientific_package_path),
            "sha256": file_sha256(scientific_package_path),
        },
        "background_job_package": {
            "path": str(job_package_path),
            "sha256": file_sha256(job_package_path),
            "canonical_sha256": job_package["job_package_sha256"],
        },
        "executor_commit": args.executor_commit,
        "executor_git_tree": tree,
        "automatic_retry_count": 0,
        "holdout_opened": False,
        "launch_command": [
            str(python),
            str(checkout / "scripts/run_g14r22b_background_job.py"),
            "launch",
            "--job-package", str(job_package_path),
            "--job-root", str(job_root),
        ],
    }
    write_json(package_root / "freeze_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
