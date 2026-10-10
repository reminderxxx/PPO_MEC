"""Create split/region/seed audit strata from the completed event replay."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_prepared_state_event_chain import OUTPUT, _csv, _write_csv
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json


def main() -> None:
    manifest_path = OUTPUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = OUTPUT / "split_region_seed_rows.csv"
    if output_path.exists():
        raise RuntimeError("stratified audit already exists")
    episodes = _csv(OUTPUT / "episode_diagnosis_rows.csv")
    events = _csv(OUTPUT / "event_chain_rows.csv")
    handoffs = _csv(OUTPUT / "handoff_interpretation_rows.csv")
    groups: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(lambda: {"episodes": [], "events": [], "handoffs": []})
    episode_to_group = {}
    for row in episodes:
        group = (row["checkpoint_view"], row["split"], row["source_segment_id"], row["method"], row["seed"])
        episode = (row["checkpoint_view"], row["split"], row["method"], row["seed"], row["design_id"])
        episode_to_group[episode] = group
        groups[group]["episodes"].append(row)
    for kind, rows in (("events", events), ("handoffs", handoffs)):
        for row in rows:
            episode = (row["checkpoint_view"], row["split"], row["method"], row["seed"], row["design_id"])
            groups[episode_to_group[episode]][kind].append(row)
    output = []
    for (view, split, region, method, seed), parts in sorted(groups.items()):
        ep, ev, ha = parts["episodes"], parts["events"], parts["handoffs"]
        a4 = [row for row in ev if row["action"] == "4"]
        nonfallback = [row for row in ha if row["arrival_action"] != "2"]
        output.append({"checkpoint_view": view, "split": split, "source_segment_id": region,
                       "method": method, "seed": seed, "episodes": len(ep), "behavior_steps": len(ev),
                       "completed_episodes": sum(row["workflow_completed"] == "True" for row in ep),
                       "on_time_completed_episodes": sum(row["on_time_workflow_completed"] == "True" for row in ep),
                       "service_failure_episodes": sum(bool(row["first_failure_cause"]) for row in ep),
                       "service_failure_events": sum(int(row["service_failures"]) for row in ep),
                       "first_failure_action4_episodes": sum(row["first_failure_action"] == "4" for row in ep),
                       "action4_attempts": len(a4), "action4_commits": sum(row["migration_success"] == "True" for row in a4),
                       "action4_current_service_failures": sum(row["service_completed"] == "False" for row in a4),
                       "action4_forecast_matches_future_handoff_posthoc": sum(row["forecast_matches_first_handoff_posthoc"] == "True" for row in a4),
                       "handoff_attempts": len(ha), "vehicle_fallback_attempts": len(ha) - len(nonfallback),
                       "nonfallback_prepared_reuse": sum(row["interpreted_state_outcome"] == "prepared_prefix_reused" for row in nonfallback),
                       "nonfallback_recompute": sum(row["interpreted_state_outcome"] == "state_missing_recomputed" for row in nonfallback),
                       "nonfallback_current_service_failed": sum(row["interpreted_state_outcome"] == "state_missing_current_service_failed" for row in nonfallback),
                       "recompute_seconds": sum(float(row["recompute_seconds"]) for row in ep),
                       "total_transfer_mb": sum(float(row["total_transfer_mb"]) for row in ep)})
    if sum(row["episodes"] for row in output) != 800 or sum(row["behavior_steps"] for row in output) != 5632:
        raise RuntimeError("strata do not partition the replay")
    _write_csv(output_path, output)
    manifest["output_files"][output_path.name] = _sha256(output_path)
    manifest["strata_count"] = len(output)
    _write_json(manifest_path, manifest)
    print(json.dumps({"strata": len(output), "episodes": 800, "steps": 5632}))


if __name__ == "__main__":
    main()
