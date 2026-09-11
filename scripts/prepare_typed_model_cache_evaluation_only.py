"""Prepare or validate the G14R20-I evaluation-only authorization request.

This entry point is preparation-only.  It never creates the requested run root,
ledger, lock, staging directory, grant, or evaluation output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.evaluation_only_execution import (
    build_evaluation_execution_contract,
    build_model_source_reference,
    canonical_sha256,
    validate_execution_contract,
    validate_command_matrix_parsers,
    validate_model_source_reference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "validate"), required=True)
    parser.add_argument("--disposition-eligibility-path")
    parser.add_argument("--source-run-root")
    parser.add_argument("--scientific-checkout")
    parser.add_argument("--executor-checkout")
    parser.add_argument("--executor-commit")
    parser.add_argument("--python-executable")
    parser.add_argument("--evaluation-run-id")
    parser.add_argument("--evaluation-run-root")
    parser.add_argument("--output-path", required=True)
    return parser


def _write_create_only(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"preparation output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"missing preparation arguments: {missing}")


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_path).resolve()
    if args.action == "validate":
        package = json.loads(output.read_text(encoding="utf-8-sig"))
        source_audit = validate_model_source_reference(package["model_source_reference"])
        execution_audit = validate_execution_contract(package["evaluation_execution_contract"])
        parser_audit = validate_command_matrix_parsers(package["evaluation_execution_contract"])
        if package.get("authorization_request_sha256") != canonical_sha256(
            {key: value for key, value in package.items() if key != "authorization_request_sha256"}
        ):
            raise ValueError("authorization request package hash mismatch")
        print(json.dumps({"status": "pass", "source": source_audit, "execution": execution_audit, "parsers": parser_audit}, indent=2))
        return
    _required(
        args,
        "disposition_eligibility_path",
        "source_run_root",
        "scientific_checkout",
        "executor_checkout",
        "executor_commit",
        "python_executable",
        "evaluation_run_id",
        "evaluation_run_root",
    )
    if Path(args.evaluation_run_root).exists():
        raise ValueError("preparation requires an absent fresh evaluation run root")
    source = build_model_source_reference(
        disposition_eligibility_path=args.disposition_eligibility_path,
        source_run_root=args.source_run_root,
        scientific_checkout=args.scientific_checkout,
    )
    execution = build_evaluation_execution_contract(
        source_reference=source,
        evaluation_run_id=args.evaluation_run_id,
        evaluation_run_root=args.evaluation_run_root,
        executor_checkout=args.executor_checkout,
        executor_commit=args.executor_commit,
        python_executable=args.python_executable,
    )
    package = {
        "evaluation_only_authorization_request_version": "1.0.0",
        "status": "READY_FOR_EVALUATION_ONLY_AUTHORIZATION",
        "model_source_reference": source,
        "evaluation_execution_contract": execution,
        "requested_authority": list(execution["phases"]),
        "grant_issued": False,
        "formal_execution_authorized": False,
        "formal_execution_started": False,
        "holdout_opened": False,
        "run_root_created": False,
        "ledger_created": False,
        "lock_created": False,
        "staging_created": False,
    }
    package["authorization_request_sha256"] = canonical_sha256(package)
    _write_create_only(output, package)
    print(
        json.dumps(
            {
                "status": package["status"],
                "output_path": str(output),
                "authorization_request_sha256": package["authorization_request_sha256"],
                "formal_execution_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
