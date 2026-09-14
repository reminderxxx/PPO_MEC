"""Prepare or validate the unsigned, two-cell G14R20-I5 recovery request.

This command never creates a recovery run, grant, ledger, lock, staging area,
or scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.restricted_recovery import (
    RestrictedRecoveryError,
    build_restricted_recovery_request,
    validate_restricted_recovery_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "validate"), required=True)
    parser.add_argument("--original-request-path")
    parser.add_argument("--original-project-grant-path")
    parser.add_argument("--original-run-root")
    parser.add_argument("--recovery-execution-id")
    parser.add_argument("--recovery-root")
    parser.add_argument("--executor-checkout")
    parser.add_argument("--executor-commit")
    parser.add_argument("--python-executable")
    parser.add_argument("--output-path", required=True)
    return parser


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise RestrictedRecoveryError(
            "prepare requires arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing)
        )


def _write_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_path)
    if args.action == "prepare":
        _required(
            args,
            "original_request_path",
            "original_project_grant_path",
            "original_run_root",
            "recovery_execution_id",
            "recovery_root",
            "executor_checkout",
            "executor_commit",
            "python_executable",
        )
        request = build_restricted_recovery_request(
            original_request_path=args.original_request_path,
            original_project_grant_path=args.original_project_grant_path,
            original_run_root=args.original_run_root,
            recovery_execution_id=args.recovery_execution_id,
            recovery_root=args.recovery_root,
            executor_checkout=args.executor_checkout,
            executor_commit=args.executor_commit,
            python_executable=args.python_executable,
        )
        # The builder has just performed the full live source/executor audit.
        validate_restricted_recovery_request(request, check_live=False)
        _write_create_only(output, request)
        print(
            json.dumps(
                {
                    "status": request["status"],
                    "authorization_request_sha256": request["authorization_request_sha256"],
                    "recovery_grant_issued": False,
                    "real_recovery_started": False,
                    "holdout_opened": False,
                },
                indent=2,
            )
        )
        return
    if output.is_symlink() or not output.is_file():
        raise RestrictedRecoveryError("validate requires an existing regular request file")
    request = json.loads(output.read_text(encoding="utf-8-sig"))
    result = validate_restricted_recovery_request(request, check_live=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
