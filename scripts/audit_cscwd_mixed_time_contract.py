"""Read-only decision-time/clock audit of the frozen action branches."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json

SOURCE = ROOT / "artifacts/analysis/cscwd_service_feasible_action_branches_20261010_v1"
OUTPUT = ROOT / "artifacts/analysis/cscwd_conditional_abstention_audit_20261010_v2"
EXAMPLES = (
    "selected|regression|sa_ghmappo|17|regression_00|2",
    "selected|frozen_check|sa_ghmappo|29|frozen_check_00|5",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-8)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("create-only audit output already exists")
    manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    export = json.loads((SOURCE / "snapshot_export_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status"), manifest.get("gate_status"), manifest.get("source_science_commit")) != (
        "complete", "MIXED", "f46ec72b15f534ac44768a83ef6316c1cfcb6b58"
    ):
        raise RuntimeError("source scientific identity drift")
    for name, digest in manifest["files"].items():
        if _sha256(SOURCE / name) != digest:
            raise RuntimeError(f"branch artifact hash drift: {name}")
    for entry in export["state_files"].values():
        if _sha256(SOURCE / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"full state snapshot hash drift: {entry['path']}")
    mapping = _rows(SOURCE / "source_to_state_rows.csv")
    summary = _rows(SOURCE / "branch_summary_rows.csv")
    trajectory = _rows(SOURCE / "branch_trajectory_rows.csv")
    by_branch: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in trajectory:
        by_branch[(row["branch_key"], int(row["branch_action"]))].append(row)
    by_key = {row["branch_key"]: row for row in mapping}
    checked = 0
    first_step_durations = defaultdict(list)
    for row in summary:
        key, action = row["branch_key"], int(row["action"])
        if row["status"] != "COMPLETE":
            continue
        source = by_key[key]
        snapshot = json.loads((SOURCE / "state_snapshots" / f"{source['full_state_sha256']}.json").read_text(encoding="utf-8"))
        if snapshot["config"]["mobility_progression"] != "decision_step_index":
            raise RuntimeError("unexpected mobility contract")
        sequence = snapshot["instance"]["rsu_sequence"]
        clock = float(snapshot["clock_seconds"])
        expected_step = int(snapshot["step_index"])
        for index, event in enumerate(by_branch[(key, action)]):
            step = int(event["step_index"])
            if step != expected_step + index:
                raise RuntimeError("decision step did not increment by one")
            if event["current_rsu"] != str(sequence[min(step, len(sequence) - 1)]):
                raise RuntimeError("RSU differs from decision-index sequence")
            cost = float(event["step_cost_seconds"])
            after = float(event["clock_seconds_after"])
            if not _close(after, clock + cost):
                raise RuntimeError("modeled clock cost not additive")
            if index == 0:
                first_step_durations[action].append(cost)
            clock = after
            checked += 1
        if not _close(clock - float(snapshot["clock_seconds"]), float(row["elapsed_seconds"])):
            raise RuntimeError("branch elapsed metric not equal to clock delta")
        if (clock > float(snapshot["instance"]["deadline_seconds"])) != (row["ending_deadline_missed"] == "True"):
            raise RuntimeError("deadline flag inconsistent with modeled clock")
    if checked != 463 or len(summary) != 90:
        raise RuntimeError("branch audit denominator drift")
    examples = []
    for source_id in EXAMPLES:
        source = next(row for row in mapping if row["source_id"] == source_id)
        snapshot = json.loads((SOURCE / "state_snapshots" / f"{source['full_state_sha256']}.json").read_text(encoding="utf-8"))
        first = {}
        for action in (0, 2, 4):
            events = by_branch[(source["branch_key"], action)]
            event = events[0]
            next_rsu = str(snapshot["instance"]["rsu_sequence"][min(int(event["step_index"]) + 1, len(snapshot["instance"]["rsu_sequence"]) - 1)])
            first[str(action)] = {"step_cost_seconds": float(event["step_cost_seconds"]),
                                  "service_completed": event["service_completed"] == "True",
                                  "node_progressed": event["node_progressed"] == "True",
                                  "model_stage_events": json.loads(event["target_model_stage_events"]),
                                  "state_transfer_status": event["state_transfer_status"],
                                  "state_committed": event["state_commit"] == "True",
                                  "next_decision_rsu": next_rsu,
                                  "next_recorded_rsu": events[1]["current_rsu"] if len(events) > 1 else ""}
            if len(events) > 1 and first[str(action)]["next_recorded_rsu"] != next_rsu:
                raise RuntimeError("next RSU audit mismatch")
        examples.append({"source_id": source_id, "source_interval": snapshot["instance"]["source_interval"],
                         "mobility_sequence_origin": "synthetic_block_from_trace_handoff_pressure",
                         "decision_step_seconds": snapshot["config"]["mobility_abstraction"]["decision_step_seconds"],
                         "failed_service_seconds": snapshot["config"]["objective"]["failed_service_seconds"],
                         "start_clock_seconds": snapshot["clock_seconds"],
                         "deadline_seconds": snapshot["instance"]["deadline_seconds"],
                         "first_actions": first})
    receipt = {"schema_version": "cscwd_mixed_time_contract_audit_v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "PASS",
               "interface_profile": "calibrated_workflow_interface_v4_prepared_state_prefix",
               "mobility_progression": "decision_step_index",
               "source_science_commit": "f46ec72b15f534ac44768a83ef6316c1cfcb6b58",
               "source_analysis_manifest_sha256": _sha256(SOURCE / "analysis_manifest.json"),
               "source_snapshot_export_manifest_sha256": _sha256(SOURCE / "snapshot_export_manifest.json"),
               "branch_files_verified": len(manifest["files"]),
               "full_states_verified": len(export["state_files"]),
               "branch_slots": len(summary), "trajectory_steps_checked": checked,
               "first_step_cost_range_by_action": {str(action): [min(costs), max(costs)]
                                                   for action, costs in sorted(first_step_durations.items())},
               "examples": examples,
               "time_contract_result": "decision_step_index_and_modeled_clock_are_separate_by_frozen_design",
               "extra_replay_env_steps": 0}
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "time_contract_receipt.json", receipt)
    print(json.dumps({"status": "PASS_CONTRACT_WITH_MODEL_LIMITATION", "steps": checked,
                      "replay_steps": 0, "examples": len(examples)}))


if __name__ == "__main__":
    main()
