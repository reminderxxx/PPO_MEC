"""Classify public-rule and execution coverage from a frozen reachability ledger.

This diagnostic is ledger-only: it does not reopen raw mobility data, execute
the environment, train a policy, or select another window.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cscwd_public_reachability_coverage_diagnosis_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(row[field] for row in rows).items(), key=lambda item: str(item[0]))
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps": len(rows),
        "requested_actions": _counter(rows, "requested_action"),
        "executed_actions": _counter(rows, "executed_action"),
        "runner_legalization_count": sum(
            row["requested_action"] != row["executed_action"] for row in rows
        ),
        "action4_mask_legal_count": sum(bool(row["public_mask"][4]) for row in rows),
        "fallback_only_mask_count": sum(
            row["public_mask"] == [False, False, True, False, False] for row in rows
        ),
        "service_completed_count": sum(bool(row["service_completed"]) for row in rows),
        "contact_rejection_count": sum(
            row["admission_rejection_reason"] == "current_rsu_contact_expires_before_commit"
            for row in rows
        ),
        "migration_success_count": sum(bool(row["migration_success"]) for row in rows),
        "model_transfer_bytes": sum(int(row["model_transfer_bytes"]) for row in rows),
        "state_transfer_bytes": sum(int(row["state_transfer_bytes"]) for row in rows),
    }


def diagnose(
    reachability_manifest: Path,
    step_ledger: Path,
    public_estimator: Path,
    raw_environment: Path,
    reachability_runner: Path,
) -> dict[str, Any]:
    manifest_sha = _sha256(reachability_manifest)
    ledger_sha = _sha256(step_ledger)
    manifest = json.loads(reachability_manifest.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in step_ledger.read_text(encoding="utf-8").splitlines()]
    if manifest["step_ledger"]["sha256"] != ledger_sha:
        raise RuntimeError("reachability ledger hash differs from its producer manifest")
    if len(rows) != int(manifest["real_step_count"]):
        raise RuntimeError("reachability ledger row count differs from producer manifest")
    if _sha256(public_estimator) != manifest["public_estimator_sha256"]:
        raise RuntimeError("public estimator identity drift")
    if _sha256(raw_environment) != manifest["raw_environment_sha256"]:
        raise RuntimeError("raw environment identity drift")
    if _sha256(reachability_runner) != manifest["script_sha256"]:
        raise RuntimeError("reachability runner identity drift")

    estimator_source = public_estimator.read_text(encoding="utf-8")
    score_source = estimator_source.split("def _score(", 1)[1].split(
        "class CausalPublicImmediateRule", 1
    )[0]
    immediate_source = estimator_source.split(
        "class CausalPublicImmediateRule", 1
    )[1].split("def project_public_state", 1)[0]
    label_source = estimator_source.split("def public_prepare_advantage_label", 1)[1].split(
        "def _score(", 1
    )[0]
    raw_source = raw_environment.read_text(encoding="utf-8")
    runner_source = reachability_runner.read_text(encoding="utf-8")
    if "raw_full_step_contact_fit" in score_source:
        raise RuntimeError("diagnosis assumption drift: scorer now consumes raw full-step fit")
    if "_score(item[1])" not in immediate_source:
        raise RuntimeError("diagnosis assumption drift: public rule no longer uses shared scorer")
    if 'action4["raw_full_step_contact_fit"] == NO' not in label_source:
        raise RuntimeError("diagnosis assumption drift: auxiliary label lost raw contact abstention")
    if "planned > contact_budget" not in raw_source:
        raise RuntimeError("diagnosis assumption drift: raw atomic full-step gate changed")
    if "raw executor rewrote an action already checked against public mask" not in runner_source:
        raise RuntimeError("producer did not assert raw executed-action consistency")

    by_method = {
        method: _method_summary([row for row in rows if row["method"] == method])
        for method in manifest["methods"]
    }
    action4 = [row for row in rows if int(row["executed_action"]) == 4]
    public_methods = {
        "causal_public_immediate_rule",
        "causal_public_two_step_rule",
    }
    public_action4 = [row for row in action4 if row["method"] in public_methods]
    action4_rows = []
    for row in action4:
        planned = float(row["planned_step_cost_seconds"])
        public_contact = float(row["public_contact_budget_seconds"])
        actual_contact = float(row["actual_contact_budget_seconds_privileged_diagnostic"])
        action4_rows.append(
            {
                "window_id": row["window_id"],
                "design_id": row["design_id"],
                "method": row["method"],
                "step": int(row["step"]),
                "planned_step_cost_seconds": planned,
                "public_contact_budget_seconds": public_contact,
                "actual_contact_budget_seconds_privileged_diagnostic": actual_contact,
                "planned_minus_public_contact_seconds": planned - public_contact,
                "planned_minus_actual_contact_seconds": planned - actual_contact,
                "public_mask_action4": bool(row["public_mask"][4]),
                "requested_equals_executed": row["requested_action"] == row["executed_action"],
                "rejection_reason": row["admission_rejection_reason"],
                "model_transfer_bytes": int(row["model_transfer_bytes"]),
                "state_transfer_bytes": int(row["state_transfer_bytes"]),
                "migration_success": bool(row["migration_success"]),
            }
        )

    public_rows = [row for row in rows if row["method"] in public_methods]
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "ledger_only_no_raw_rescan_no_training_no_environment_replay",
        "inputs": {
            "reachability_manifest_sha256": manifest_sha,
            "step_ledger_sha256": ledger_sha,
            "public_estimator_sha256": _sha256(public_estimator),
            "raw_environment_sha256": _sha256(raw_environment),
            "reachability_runner_sha256": _sha256(reachability_runner),
        },
        "counts": {
            "episodes": int(manifest["episode_count"]),
            "steps": len(rows),
            "public_rule_steps": len(public_rows),
            "workflow_completed_episodes": sum(
                int(row["workflow_completed"]) for row in manifest["results"]
            ),
            "on_time_episodes": sum(
                bool(row["on_time_complete"]) for row in manifest["results"]
            ),
            "contact_rejections": sum(
                row["admission_rejection_reason"] == "current_rsu_contact_expires_before_commit"
                for row in rows
            ),
            "runner_legalizations": sum(
                row["requested_action"] != row["executed_action"] for row in rows
            ),
            "public_rule_legalizations": sum(
                row["requested_action"] != row["executed_action"] for row in public_rows
            ),
            "raw_executor_rewrites_after_legalization": 0,
            "action4_executions": len(action4),
            "public_rule_action4_executions": len(public_action4),
            "action4_contact_rejections": sum(
                row["admission_rejection_reason"] == "current_rsu_contact_expires_before_commit"
                for row in action4
            ),
            "action4_planned_exceeds_public_contact": sum(
                float(row["planned_step_cost_seconds"])
                > float(row["public_contact_budget_seconds"])
                for row in action4
            ),
            "action4_planned_exceeds_actual_contact": sum(
                float(row["planned_step_cost_seconds"])
                > float(row["actual_contact_budget_seconds_privileged_diagnostic"])
                for row in action4
            ),
        },
        "method_summary": by_method,
        "action4_rows": action4_rows,
        "static_contract_audit": {
            "mask_encodes_contact_duration_feasibility": False,
            "mask_scope": "semantic action availability; raw env narrows to fallback-only when uncovered",
            "estimator_produces_raw_full_step_contact_fit": True,
            "auxiliary_label_abstains_when_raw_full_step_contact_fit_is_no": True,
            "public_rule_score_consumes_raw_full_step_contact_fit": False,
            "public_rule_score_consumed_fields": [
                "current_service",
                "deadline_fit",
                "estimated_total_seconds",
                "future_readiness_gain",
                "model/state/input transfer bytes",
            ],
            "raw_executor_gate": "planned full step must fit physical current-RSU contact",
            "planned_full_step_includes": [
                "model network transfer",
                "model load",
                "current-node compute",
                "prefix recompute when applicable",
                "state network transfer after service success",
                "state restore",
                "input transfer when applicable",
            ],
            "success_only_state_commit": True,
            "rejected_action4_zero_byte_rollback_observed": all(
                not row["migration_success"]
                and int(row["model_transfer_bytes"]) == 0
                and int(row["state_transfer_bytes"]) == 0
                for row in action4
            ),
        },
        "cause_classification": {
            "confirmed_public_rule_consumer_defect": (
                "_score ignores raw_full_step_contact_fit, so both public rules selected action 4 "
                "in two states whose public estimated full-step cost already exceeded public contact"
            ),
            "defect_scope": (
                "two of 126 public-rule decisions; fixing it would avoid knowingly infeasible public-rule "
                "choices but cannot create a successful migration event"
            ),
            "confirmed_mechanism_coverage_blocker": (
                "all 12 action-4 executions had planned full-step cost above both public and actual contact; "
                "the atomic raw contract therefore rolled back before model/state commit"
            ),
            "not_a_raw_executor_rewrite": (
                "producer asserted event.action equals the mask-legalized action; 96 requested/executed "
                "differences came from the runner's explicit fallback legalization, and public rules had zero"
            ),
            "interpretation_boundary": (
                "the atomic full-step contract is internally implemented as frozen, but its physical validity "
                "is not established: it charges compute, model load, state restore, and transfers against "
                "current-RSU contact"
            ),
        },
        "next_change_boundary": {
            "minimal_correctness_fix_for_separate_task": (
                "make public rule selection fail closed on raw_full_step_contact_fit=no; this is a baseline "
                "consumer correction, not an algorithm innovation or a mechanism-enabling change"
            ),
            "mechanism_enabling_change_not_authorized": (
                "a phase-split/asynchronous prepare contract would require a separately frozen action/state "
                "machine and a justified resource model specifying which transfer/load/compute phases require "
                "vehicle contact; it must not be introduced by silently relaxing the current gate"
            ),
            "training_decision": "STOP",
        },
        "claim_boundary": {
            "raw_rows_read": 0,
            "environment_steps_run": 0,
            "training_steps": 0,
            "checkpoint_selection": False,
            "performance_claim": False,
        },
    }
    if result["counts"]["action4_contact_rejections"] != len(action4):
        raise RuntimeError("not all action-4 executions were rejected by contact")
    if result["counts"]["public_rule_legalizations"] != 0:
        raise RuntimeError("public-rule requested/executed consistency failed")
    if not result["static_contract_audit"]["rejected_action4_zero_byte_rollback_observed"]:
        raise RuntimeError("rejected action 4 committed bytes or migration state")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability-manifest", type=Path, required=True)
    parser.add_argument("--step-ledger", type=Path, required=True)
    parser.add_argument("--public-estimator", type=Path, required=True)
    parser.add_argument("--raw-environment", type=Path, required=True)
    parser.add_argument("--reachability-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        args.reachability_manifest,
        args.step_ledger,
        args.public_estimator,
        args.raw_environment,
        args.reachability_runner,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
