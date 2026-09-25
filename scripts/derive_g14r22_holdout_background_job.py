#!/usr/bin/env python3
"""Derive one formal G14R22-D background job from an exact new authorization.

The unsigned request and scientific package remain byte-for-byte unchanged.
The token is read only for hash verification and is never copied or serialized.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_g14r22b_background_job import (  # noqa: E402
    HOST_VERSION,
    canonical_sha256,
    file_sha256,
    validate_job_package,
)
from src.evaluators.dedicated_holdout_execution import (  # noqa: E402
    read_json,
    validate_command_package,
    validate_unsigned_request,
    verify_grant,
)


DERIVATION_VERSION = "g14r22d_authorized_background_derivation_v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=checkout, text=True).strip()


def _reference(path: Path, *, include_canonical: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if include_canonical:
        value["canonical_sha256"] = canonical_sha256(read_json(path))
    return value


def derive(
    *, request_path: Path, command_package_path: Path, grant_path: Path,
    token_path: Path, executor_checkout: Path, python_executable: Path,
    derived_root: Path, job_root: Path, isolated_fixture: bool = False,
) -> dict[str, Any]:
    paths = [request_path, command_package_path, grant_path, token_path]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("request/package/grant/token references must be regular non-symlink files")
    if derived_root.exists() or job_root.exists():
        raise ValueError("derived root and unique job root must both be absent")
    if not executor_checkout.is_absolute() or executor_checkout.is_symlink():
        raise ValueError("executor checkout must be an absolute non-symlink path")
    if not python_executable.is_absolute() or not python_executable.is_file():
        raise ValueError("frozen Python executable must be an existing absolute file")
    if Path(os.path.abspath(sys.executable)) != Path(os.path.abspath(python_executable)):
        raise ValueError("derivation must run with the frozen Python executable")
    request_before = _reference(request_path, include_canonical=True)
    package_before = _reference(command_package_path, include_canonical=True)
    request = read_json(request_path)
    package = read_json(command_package_path)
    grant = read_json(grant_path)
    token = token_path.read_bytes()
    validate_command_package(package, isolated_fixture=isolated_fixture)
    validate_unsigned_request(
        request, package, isolated_fixture=isolated_fixture
    )
    authorization = verify_grant(
        request,
        package,
        grant,
        token,
        boundary="background_derivation",
    )
    head = _git(executor_checkout, "rev-parse", "HEAD")
    tree = _git(executor_checkout, "rev-parse", "HEAD^{tree}")
    if head != package.get("executor_commit"):
        raise ValueError("derived executor commit differs from the scientific package")
    if package.get("executor_git_tree") not in (None, tree):
        raise ValueError("derived executor tree differs from the scientific package")
    if _git(executor_checkout, "status", "--porcelain"):
        raise ValueError("executor checkout must be clean during derivation")
    formal_output_root = Path(str(package.get("output_root", "")))
    if formal_output_root.exists() or formal_output_root.is_symlink():
        raise ValueError("formal one-time output root must be absent during derivation")

    request_ref = _reference(request_path, include_canonical=True)
    package_ref = _reference(command_package_path, include_canonical=True)
    grant_ref = _reference(grant_path, include_canonical=True)
    token_ref = _reference(token_path)
    mode = "formal_holdout_fixture" if isolated_fixture else "formal_holdout"
    check = "fixture-execute" if isolated_fixture else "execute"
    command = [
        str(python_executable),
        str(executor_checkout / "scripts/run_dedicated_public_holdout.py"),
        "--request-path", str(request_path),
        "--command-package-path", str(command_package_path),
        "--grant-path", str(grant_path),
        "--one-time-token-file", str(token_path),
        "--check", check,
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(executor_checkout),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    job_package: dict[str, Any] = {
        "host_version": HOST_VERSION,
        "job_id": f"g14r22d-{request['request_sha256'][:16]}",
        "execution_mode": mode,
        "isolated_fixture_non_scientific": bool(isolated_fixture),
        "executor_checkout": str(executor_checkout),
        "executor_commit": head,
        "executor_git_tree": tree,
        "cwd": str(executor_checkout),
        "python_executable": str(python_executable),
        "python_resolved_executable": str(python_executable.resolve(strict=True)),
        "environment": environment,
        "command": command,
        "request": request_ref,
        "scientific_command_package": package_ref,
        "grant": grant_ref,
        "one_time_token": token_ref,
        "authorization_validation": authorization,
        "terminal_contract": {
            "scientific_receipt_path": str(formal_output_root / "execution_receipt.json"),
            "opening_receipt_path": str(formal_output_root / "opening_receipt.json"),
            "integrity_manifest_path": str(formal_output_root / "artifact_integrity_manifest.json"),
            "integrity_root": str(formal_output_root / "published"),
            "expected_receipt_bindings": {
                "request_sha256": request["request_sha256"],
                "command_package_sha256": package["command_package_sha256"],
                "executor_commit": head,
                "output_root": str(formal_output_root),
            },
        },
        "formal_output_root": str(formal_output_root),
        "job_root": str(job_root),
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "resume_allowed": False,
        "reopen_allowed": False,
        "automatic_lock_cleanup_allowed": False,
        "holdout_opened": False,
    }
    job_package["job_package_sha256"] = canonical_sha256(job_package)
    derived_root.mkdir(parents=True, exist_ok=False)
    job_package_path = derived_root / "background_job_package.json"
    _write_json(job_package_path, job_package)
    validate_job_package(job_package_path, require_clean_checkout=True)
    request_after = _reference(request_path, include_canonical=True)
    package_after = _reference(command_package_path, include_canonical=True)
    if request_after != request_before or package_after != package_before:
        raise ValueError("unsigned request/package changed during derivation")
    receipt = {
        "derivation_version": DERIVATION_VERSION,
        "status": "AUTHORIZED_DERIVED_MATERIAL_READY_FOR_SINGLE_LAUNCH",
        "execution_mode": mode,
        "source_request": request_ref,
        "source_scientific_command_package": package_ref,
        "grant_reference": grant_ref,
        "one_time_token_reference": token_ref,
        "token_content_serialized": False,
        "authorization_validation": authorization,
        "executor_commit": head,
        "executor_git_tree": tree,
        "formal_output_root": str(formal_output_root),
        "unique_job_root": str(job_root),
        "background_job_package": _reference(job_package_path, include_canonical=True),
        "unsigned_sources_unchanged": True,
        "automatic_retry_count": 0,
        "launch_command": [
            str(python_executable),
            str(executor_checkout / "scripts/run_g14r22b_background_job.py"),
            "launch",
            "--job-package", str(job_package_path),
            "--job-root", str(job_root),
        ],
    }
    _write_json(derived_root / "derivation_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--command-package-path", type=Path, required=True)
    parser.add_argument("--grant-path", type=Path, required=True)
    parser.add_argument("--one-time-token-file", type=Path, required=True)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--isolated-fixture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = derive(
        request_path=args.request_path.resolve(),
        command_package_path=args.command_package_path.resolve(),
        grant_path=args.grant_path.resolve(),
        token_path=args.one_time_token_file.resolve(),
        executor_checkout=args.executor_checkout.resolve(),
        python_executable=Path(os.path.abspath(args.python_executable)),
        derived_root=args.derived_root.resolve(),
        job_root=args.job_root.resolve(),
        isolated_fixture=args.isolated_fixture,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
