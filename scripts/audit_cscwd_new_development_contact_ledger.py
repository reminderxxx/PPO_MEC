"""Zero-step contact diagnosis of the frozen raw development reachability ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST_SHA = "481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c"
EXPECTED_LEDGER_SHA = "679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb"
CONTACT_REASON = "current_rsu_contact_expires_before_commit"
EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attempt_identity(row: dict) -> tuple:
    return (
        row["window_id"], row["design_id"], row["step"],
        row["executed_action"], row["clock_seconds_before"],
        row["planned_step_cost_seconds"],
    )


def diagnose(manifest_path: Path) -> dict:
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("reachability manifest differs from frozen input")
    manifest = json.loads(manifest_path.read_text())
    ledger_path = manifest_path.parent / manifest["step_ledger"]["path"]
    if manifest["step_ledger"]["sha256"] != EXPECTED_LEDGER_SHA or sha256(ledger_path) != EXPECTED_LEDGER_SHA:
        raise RuntimeError("step ledger differs from frozen input")
    rows = [json.loads(line) for line in ledger_path.open()]
    contact = [row for row in rows if row["admission_rejection_reason"] == CONTACT_REASON]
    action4 = [row for row in rows if row["executed_action"] == 4]
    if (manifest["episode_count"], manifest["real_step_count"], len(rows), len(contact), len(action4)) != (40, 320, 320, 68, 12):
        raise RuntimeError("frozen episode/step/contact/action4 counts changed")
    if any(row["admission_rejection_reason"] != CONTACT_REASON for row in action4):
        raise RuntimeError("action4 rejection pattern changed")

    failures = []
    for row in contact:
        planned = float(row["planned_step_cost_seconds"])
        actual = float(row["actual_contact_budget_seconds_privileged_diagnostic"])
        public = row["public_contact_budget_seconds"]
        public = None if public is None else float(public)
        trace = float(row["trace_remaining_seconds_privileged_diagnostic"])
        if not (planned > actual + EPS and planned <= trace + EPS):
            raise RuntimeError("contact rejection is not isolated from trace end")
        failures.append({
            "window_id": row["window_id"], "design_id": row["design_id"],
            "method": row["method"], "step": row["step"],
            "clock_seconds_before": row["clock_seconds_before"],
            "node_id_before": row["node_id_before"],
            "action": row["executed_action"],
            "planned_step_cost_seconds": planned,
            "public_contact_budget_seconds": public,
            "actual_contact_budget_seconds_privileged_diagnostic": actual,
            "trace_remaining_seconds_privileged_diagnostic": trace,
            "planned_minus_actual_contact_seconds": planned - actual,
            "planned_minus_public_contact_seconds": None if public is None else planned - public,
            "public_predicts_full_step_fit": public is not None and planned <= public + EPS,
            "actual_full_step_fit": False,
            "actual_start_contact_positive": actual > EPS,
            "pre_step_model_transfer_bytes": row["model_transfer_bytes"],
            "pre_step_state_transfer_bytes": row["state_transfer_bytes"],
        })

    per_action = {}
    for action in (0, 3, 4):
        subset = [row for row in failures if row["action"] == action]
        per_action[str(action)] = {
            "rejection_episodes": len(subset),
            "unique_attempts": len({attempt_identity(row) for row in contact if row["executed_action"] == action}),
            "public_predicts_fit_but_actual_rejects": sum(row["public_predicts_full_step_fit"] for row in subset),
            "planned_minus_actual_contact_min_seconds": min(row["planned_minus_actual_contact_seconds"] for row in subset),
            "planned_minus_actual_contact_max_seconds": max(row["planned_minus_actual_contact_seconds"] for row in subset),
        }
    per_method = {
        method: {
            "contact_rejections": sum(row["method"] == method for row in failures),
            "action4_rejections": sum(row["method"] == method and row["action"] == 4 for row in failures),
        }
        for method in manifest["methods"]
    }
    action4_rows = [row for row in failures if row["action"] == 4]
    return {
        "schema_version": "cscwd_new_development_contact_ledger_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_reachability_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "input_step_ledger_sha256": EXPECTED_LEDGER_SHA,
        "scope": "same_40_episodes_zero_extra_environment_steps_no_trace_reload",
        "counts": {
            "episodes": 40, "real_steps_reused": 320, "extra_environment_steps": 0,
            "contact_rejection_rows": len(failures),
            "unique_contact_attempts": len({attempt_identity(row) for row in contact}),
            "action4_rows": len(action4_rows),
            "unique_action4_attempts": len({attempt_identity(row) for row in action4}),
            "public_predicts_fit_but_actual_rejects": sum(row["public_predicts_full_step_fit"] for row in failures),
            "contact_rejections_with_positive_start_contact": sum(row["actual_start_contact_positive"] for row in failures),
            "contact_rejections_with_trace_sufficient": len(failures),
        },
        "per_action": per_action,
        "per_method": per_method,
        "per_window": dict(Counter(row["window_id"] for row in failures)),
        "action4_contact_ranges_seconds": {
            "public_min": min(float(row["public_contact_budget_seconds"]) for row in action4_rows),
            "public_max": max(float(row["public_contact_budget_seconds"]) for row in action4_rows),
            "actual_min": min(row["actual_contact_budget_seconds_privileged_diagnostic"] for row in action4_rows),
            "actual_max": max(row["actual_contact_budget_seconds_privileged_diagnostic"] for row in action4_rows),
            "planned_min": min(row["planned_step_cost_seconds"] for row in action4_rows),
            "planned_max": max(row["planned_step_cost_seconds"] for row in action4_rows),
        },
        "contact_rejection_rows": failures,
        "unverified": ["per_phase_prepare_vs_compute_fit", "current_and_target_rsu_geometry",
                       "native_preview_migration_status", "partial_transfer_semantics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.reachability_manifest)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    path = output / "contact_ledger_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(path), "sha256": sha256(path), "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
