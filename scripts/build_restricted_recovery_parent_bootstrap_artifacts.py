"""Finalize G14R20-I5-C evidence without executing a recovery cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_restricted_recovery_acceptance_artifacts import (
    compare_protected_snapshots,
    junit_cases,
    write_create_only,
)
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ORIGINAL_RUN_ROOT,
    audit_original_recovery_source,
    validate_restricted_recovery_request,
)


I5B_ROOT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/analysis/"
    "g14r20_i5_b_python_binding_20260916"
)


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON is not an object: {path}")
    return value


def _ref(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--target-junit", required=True)
    parser.add_argument("--public-junit", required=True)
    parser.add_argument("--adjacent-junit", required=True)
    parser.add_argument("--full-junit", required=True)
    parser.add_argument("--validation-receipt", required=True)
    args = parser.parse_args()

    artifact = Path(args.artifact_root).absolute()
    request_path = artifact / "unsigned_recovery_request.json"
    request = _read(request_path)
    validate_restricted_recovery_request(request, check_live=True)
    execution = request["recovery_execution"]
    recovery_root = Path(execution["recovery_root"])
    if recovery_root.exists():
        raise ValueError("new recovery root must remain absent")

    i5b_request = I5B_ROOT / "unsigned_recovery_request.json"
    i5b_grant = I5B_ROOT / "restricted_recovery_grant.json"
    i5b_log = I5B_ROOT / "execute_cell_1.log"
    i5b_rc = I5B_ROOT / "execute_cell_1.returncode"
    if i5b_rc.read_text(encoding="utf-8").strip() != "1":
        raise ValueError("I5-B return code evidence drift")
    log_text = i5b_log.read_text(encoding="utf-8")
    if "recovery parent is not a writable real directory" not in log_text:
        raise ValueError("I5-B failure log no longer proves the parent failure")
    old_request = _read(i5b_request)
    old_root = Path(old_request["recovery_execution"]["recovery_root"])
    old_lock = Path(old_request["recovery_execution"]["lock_path"])
    if old_root.exists() or old_lock.exists():
        raise ValueError("I5-B recovery root/lock unexpectedly exists")

    source = audit_original_recovery_source()
    start = _read(artifact / "protection_start.json")
    end = _read(artifact / "protection_end.json")
    protection = compare_protected_snapshots(start, end)
    if protection["status"] != "pass" or len(start.get("files", [])) != 7:
        raise ValueError("seven-file protection comparison failed")

    junit_paths = {
        "target": Path(args.target_junit).absolute(),
        "public_two_process": Path(args.public_junit).absolute(),
        "adjacent": Path(args.adjacent_junit).absolute(),
        "full_repository": Path(args.full_junit).absolute(),
    }
    tests = {}
    for label, path in junit_paths.items():
        summary, cases = junit_cases(path)
        if summary["tests"] <= 0 or summary["failures"] or summary["errors"]:
            raise ValueError(f"test evidence failed: {label}")
        tests[label] = {
            "junit": _ref(path),
            "summary": summary,
            "skipped_nodeids": [row["nodeid"] for row in cases if row["status"] == "skipped"],
        }
    validation_receipt_path = Path(args.validation_receipt).absolute()
    validation_receipt = _read(validation_receipt_path)
    if validation_receipt.get("status") != "pass":
        raise ValueError("validation receipt did not pass")

    root_cause = {
        "version": "1.0.0",
        "goal": "G14R20-I5-C",
        "failure_classification": [
            "PRE_TRANSACTION_BOOTSTRAP_FAILURE",
            "RECOVERY_PARENT_CONTRACT_NOT_CLOSED",
        ],
        "i5_b_request": _ref(i5b_request),
        "i5_b_grant": _ref(i5b_grant),
        "i5_b_execute_log": _ref(i5b_log),
        "i5_b_return_code": 1,
        "facts": {
            "single_writer_required_existing_writable_real_parent": True,
            "request_builder_closed_parent_precondition": False,
            "live_validator_closed_parent_precondition": False,
            "qualify_closed_parent_precondition": False,
            "old_public_synthetic_fixture_precreated_scope": True,
            "failure_before_transaction": True,
            "failure_before_lock": True,
            "failure_before_recovery_root": True,
            "failure_before_ledger": True,
            "failure_before_staging": True,
            "failure_before_scientific_dispatch": True,
            "model_data_algorithm_hyperparameter_performance_related": False,
        },
        "i5_b_retry_resume_salvage_reexecution_allowed": False,
        "historical_start_evidence": "unavailable",
        "historical_protection_verdict": "UNVERIFIED",
    }
    write_create_only(artifact / "root_cause_audit.json", root_cause)

    acceptance = {
        "version": "1.0.0",
        "goal": "G14R20-I5-C",
        "status": "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "authorization_request_sha256": request["authorization_request_sha256"],
        "new_recovery_execution_id": execution["recovery_execution_id"],
        "new_recovery_root": str(recovery_root),
        "new_recovery_root_exists": False,
        "parent_bootstrap_contract": execution["parent_bootstrap_contract"],
        "tests": tests,
        "validation_receipt": _ref(validation_receipt_path),
        "protected_seven_files": {
            "start": _ref(artifact / "protection_start.json"),
            "end": _ref(artifact / "protection_end.json"),
            "comparison": protection,
        },
        "immutable_source": {
            "external_committed_cell_count": len(source["external_committed_cells"]),
            "source_model_count": len(request["source_models"]),
            "phase_ledger_prefix": source["phase_ledger_prefix"],
            "cell_ledger_prefix": source["cell_ledger_prefix"],
            "held_lock": source["held_lock"],
        },
        "historical_start_evidence": "unavailable",
        "historical_protection_verdict": "UNVERIFIED",
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
        "scientific_dispatch_count": 0,
        "model_load_count": 0,
        "performance_result_count": 0,
    }
    write_create_only(artifact / "parent_bootstrap_acceptance.json", acceptance)

    inventory = []
    for path in sorted(artifact.iterdir()):
        if path.name == "final_artifact_integrity.json" or not path.is_file():
            continue
        inventory.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    integrity = {
        "version": "1.0.0",
        "artifact_root": str(artifact),
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "file_count": len(inventory),
        "files": inventory,
        "inventory_sha256": canonical_sha256(inventory),
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    write_create_only(artifact / "final_artifact_integrity.json", integrity)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
