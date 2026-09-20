"""Prepare or validate an unsigned G14E07 request; never create its run root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_typed_model_cache_evaluation_only import _write_create_only
from src.runtime.evaluation_only_execution import (
    build_evaluation_execution_contract, build_model_source_reference,
    canonical_sha256, validate_command_matrix_parsers,
    validate_execution_contract, validate_model_source_reference,
)
from src.runtime.post_ablation_execution import HANDOFF_PATH, POST_PHASES, audit_handoff


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--action", choices=("prepare", "validate"), required=True)
    value.add_argument("--disposition-eligibility-path")
    value.add_argument("--source-run-root")
    value.add_argument("--scientific-checkout")
    value.add_argument("--executor-checkout")
    value.add_argument("--executor-commit")
    value.add_argument("--python-executable")
    value.add_argument("--evaluation-run-id")
    value.add_argument("--evaluation-run-root")
    value.add_argument("--handoff-path", default=str(HANDOFF_PATH))
    value.add_argument("--output-path", required=True)
    return value


def validate_package(package: dict, *, live: bool = True, require_absent_root: bool = True) -> dict:
    if package.get("authorization_request_sha256") != canonical_sha256({
        key: item for key, item in package.items() if key != "authorization_request_sha256"
    }):
        raise ValueError("post-ablation request hash mismatch")
    if (package.get("status") != "READY_FOR_POST_ABLATION_EVALUATION_AUTHORIZATION"
            or package.get("requested_authority") != list(POST_PHASES)
            or any(package.get(key) is not False for key in (
                "formal_grant_issued", "formal_execution_started", "holdout_opened",
                "run_root_created", "training_executed", "selection_executed",
                "prior_cells_copied", "prior_cells_redispatched",
            ))):
        raise ValueError("post-ablation request scope drift")
    source = package["model_source_reference"]
    execution = package["evaluation_execution_contract"]
    handoff = package["post_ablation_handoff_reference"]
    if handoff != execution.get("post_ablation_handoff_reference"):
        raise ValueError("request/contract handoff mismatch")
    if live:
        validate_model_source_reference(source)
    validate_execution_contract(execution, model_source_reference=source, check_live=live)
    validate_command_matrix_parsers(execution, model_source_reference=source, check_live=live)
    root = Path(execution["evaluation_run_root"])
    if live and require_absent_root and root.exists():
        raise ValueError("unsigned preparation requires absent create-only run root")
    return {"status": "pass", "remaining_phases": list(POST_PHASES),
            "remaining_command_counts": {phase: len(execution["command_plans"][phase]["commands"])
                                         for phase in POST_PHASES},
            "handoff_sha256": handoff["sha256"]}


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output_path).absolute()
    if args.action == "validate":
        package = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(validate_package(package), indent=2))
        return
    needed = ("disposition_eligibility_path", "source_run_root", "scientific_checkout",
              "executor_checkout", "executor_commit", "python_executable",
              "evaluation_run_id", "evaluation_run_root")
    missing = [name for name in needed if not getattr(args, name)]
    if missing:
        raise ValueError(f"missing preparation arguments: {missing}")
    if Path(args.evaluation_run_root).exists():
        raise ValueError("new run root already exists")
    handoff = audit_handoff(args.handoff_path)
    reference = {"path": str(HANDOFF_PATH), "sha256": handoff["handoff_sha256"]}
    source = build_model_source_reference(
        disposition_eligibility_path=args.disposition_eligibility_path,
        source_run_root=args.source_run_root,
        scientific_checkout=args.scientific_checkout,
    )
    execution = build_evaluation_execution_contract(
        source_reference=source, evaluation_run_id=args.evaluation_run_id,
        evaluation_run_root=args.evaluation_run_root,
        executor_checkout=args.executor_checkout, executor_commit=args.executor_commit,
        python_executable=args.python_executable,
        post_ablation_handoff_reference=reference,
    )
    package = {
        "post_ablation_request_version": "1.0.0",
        "status": "READY_FOR_POST_ABLATION_EVALUATION_AUTHORIZATION",
        "model_source_reference": source,
        "evaluation_execution_contract": execution,
        "post_ablation_handoff_reference": reference,
        "requested_authority": list(POST_PHASES),
        "formal_grant_issued": False,
        "formal_execution_started": False,
        "holdout_opened": False,
        "run_root_created": False,
        "training_executed": False,
        "selection_executed": False,
        "prior_cells_copied": False,
        "prior_cells_redispatched": False,
    }
    package["authorization_request_sha256"] = canonical_sha256(package)
    validate_package(package)
    _write_create_only(output, package)
    print(json.dumps({"status": package["status"], "output_path": str(output),
                      "authorization_request_sha256": package["authorization_request_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
