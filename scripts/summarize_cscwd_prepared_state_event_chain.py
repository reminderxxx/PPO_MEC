"""Summarize the frozen replay, separating fallback from prepared-state reuse."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_prepared_state_event_chain import OUTPUT, VIEWS, METHODS, _csv, _write_csv
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json


def _true(value: str) -> bool:
    if value not in {"True", "False"}:
        raise RuntimeError(f"invalid bool: {value}")
    return value == "True"


def main() -> None:
    manifest_path = OUTPUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "diagnostic_summary.json" in manifest["output_files"]:
        raise RuntimeError("summary already exists")
    episodes = _csv(OUTPUT / "episode_diagnosis_rows.csv")
    events = _csv(OUTPUT / "event_chain_rows.csv")
    raw_handoffs = _csv(OUTPUT / "handoff_chain_rows.csv")
    handoffs = []
    for row in raw_handoffs:
        corrected = dict(row)
        if row["arrival_action"] == "2":
            corrected["interpreted_state_outcome"] = "vehicle_fallback_state_not_required"
        elif _true(row["arrival_state_ready"]):
            corrected["interpreted_state_outcome"] = "prepared_prefix_reused"
        elif not _true(row["arrival_service_completed"]):
            corrected["interpreted_state_outcome"] = "state_missing_current_service_failed"
        else:
            corrected["interpreted_state_outcome"] = "state_missing_recomputed"
        handoffs.append(corrected)
    summary = {"schema_version": "cscwd_prepared_state_event_chain_summary_v1",
               "source_run_id": "cscwd_causal_prepared_state_visibility_matched_20261010_v1",
               "denominator_note": "episode counts include repeated windows/seeds; handoff rows are service attempts, including retries",
               "views": {}}
    for view in VIEWS:
        summary["views"][view] = {}
        for method in METHODS:
            ep = [row for row in episodes if row["checkpoint_view"] == view and row["method"] == method]
            ev = [row for row in events if row["checkpoint_view"] == view and row["method"] == method]
            ha = [row for row in handoffs if row["checkpoint_view"] == view and row["method"] == method]
            a4 = [row for row in ev if row["action"] == "4"]
            failures = [row for row in ev if not _true(row["service_completed"])]
            first = [row for row in ep if row["first_failure_cause"]]
            if len(ep) != 100 or sum(int(row["service_failures"]) for row in ep) != len(failures):
                raise RuntimeError(f"denominator mismatch: {view} {method}")
            summary["views"][view][method] = {
                "episodes": len(ep), "behavior_steps": len(ev),
                "workflow_completed_episodes": sum(_true(row["workflow_completed"]) for row in ep),
                "on_time_workflow_completed_episodes": sum(_true(row["on_time_workflow_completed"]) for row in ep),
                "service_failure_episodes": len(first), "service_failure_events": len(failures),
                "first_failure_causes": dict(Counter(row["first_failure_cause"] for row in first)),
                "first_failure_actions": dict(Counter(row["first_failure_action"] for row in first)),
                "first_failure_by_split": dict(Counter(row["split"] for row in first)),
                "first_failure_by_region": dict(Counter(row["source_segment_id"] for row in first)),
                "first_failure_by_seed": dict(Counter(row["seed"] for row in first)),
                "action4_attempts": len(a4), "action4_state_commits": sum(_true(row["migration_success"]) for row in a4),
                "action4_current_service_failures": sum(not _true(row["service_completed"]) for row in a4),
                "action4_preview_feasible": sum(_true(row["target_prepare_feasible_preview"]) for row in a4),
                "action4_predicted_target_matched_future_handoff_posthoc": sum(_true(row["forecast_matches_first_handoff_posthoc"]) for row in a4),
                "handoff_service_attempts": len(ha),
                "handoff_interpreted_outcomes": dict(Counter(row["interpreted_state_outcome"] for row in ha)),
                "handoff_nonfallback_break_causes": dict(Counter(row["state_break_cause"] for row in ha if row["arrival_action"] != "2")),
                "total_recompute_seconds": sum(float(row["recompute_seconds"]) for row in ep),
                "total_transfer_mb": sum(float(row["total_transfer_mb"]) for row in ep),
            }
    handoff_path = OUTPUT / "handoff_interpretation_rows.csv"
    summary_path = OUTPUT / "diagnostic_summary.json"
    _write_csv(handoff_path, handoffs)
    _write_json(summary_path, summary)
    manifest["output_files"][handoff_path.name] = _sha256(handoff_path)
    manifest["output_files"][summary_path.name] = _sha256(summary_path)
    _write_json(manifest_path, manifest)
    print(json.dumps({"handoff_attempts": len(handoffs), "views": list(summary["views"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
