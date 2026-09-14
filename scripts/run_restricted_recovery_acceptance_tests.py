"""Run one acceptance test scope and write a commit-bound JUnit receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_restricted_recovery_acceptance_artifacts import junit_cases
from src.runtime.evaluation_only_execution import file_sha256


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git_value(checkout: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", revision], text=True
    ).strip()


def clean(checkout: Path) -> bool:
    value = subprocess.check_output(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "README.md",
            "docs",
            "scripts",
            "src",
            "tests",
            "configs",
        ],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).strip()
    return not value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--executor-checkout", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--junit-path", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    checkout = Path(args.executor_checkout).resolve(strict=True)
    junit_path = Path(args.junit_path).resolve()
    receipt_path = Path(args.receipt_path).resolve()
    if junit_path.exists() or receipt_path.exists():
        raise ValueError("JUnit and receipt paths must be create-only")
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    before_commit = git_value(checkout, "HEAD")
    before_tree = git_value(checkout, "HEAD^{tree}")
    clean_before = clean(checkout)
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        raise ValueError("pytest selector arguments are required")
    command = [
        str(Path(args.python_executable).resolve(strict=True)),
        "-B",
        "-m",
        "pytest",
        *pytest_args,
        f"--junitxml={junit_path}",
    ]
    started_at = now_iso()
    completed = subprocess.run(command, cwd=checkout, env=dict(os.environ), check=False)
    completed_at = now_iso()
    after_commit = git_value(checkout, "HEAD")
    after_tree = git_value(checkout, "HEAD^{tree}")
    clean_after = clean(checkout)
    summary, cases = junit_cases(junit_path)
    receipt = {
        "restricted_recovery_test_receipt_version": "1.0.0",
        "label": args.label,
        "started_at": started_at,
        "completed_at": completed_at,
        "executor_checkout": str(checkout),
        "executor_commit": before_commit,
        "executor_git_tree": before_tree,
        "executor_identity_unchanged": before_commit == after_commit and before_tree == after_tree,
        "checkout_clean_before": clean_before,
        "checkout_clean_after": clean_after,
        "python_executable": command[0],
        "pythonpath_overlay": os.environ.get("PYTHONPATH"),
        "pytest_args": pytest_args,
        "command": command,
        "return_code": completed.returncode,
        "junit_absolute_path": str(junit_path),
        "junit_sha256": file_sha256(junit_path),
        "junit_summary": summary,
        "skips": [case for case in cases if case["status"] == "skipped"],
    }
    data = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with receipt_path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
