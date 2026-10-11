"""Read-only recomputation of a complete CSCWD public-region v3 artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EPS = 1e-6
RUN_ID = "cscwd_phase_v2_public_prepare_regions_20261011_v3"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path, schema: str) -> dict:
    record = read(path)
    assert record["schema_version"] == schema
    expected = hashlib.sha256(json.dumps(record["payload"], sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()
    assert record["payload_sha256"] == expected, path
    return record["payload"]


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=EPS)


def recompute_event(row: dict) -> None:
    assert row["public_action_mask_before"][row["executed_action"]]
    phases = row["phase_events"]
    assert close(sum(p["elapsed_seconds"] for p in phases), row["step_cost_seconds"])
    assert row["backhaul_transfer_bytes"] == row["model_transfer_bytes"] + row["state_transfer_bytes"]
    assert close(sum(p["elapsed_seconds"] for p in phases if p["phase"] == "prefix_recompute"),
                 row["recompute_seconds"])
    for phase in phases:
        assert close(phase["end_seconds"] - phase["start_seconds"], phase["elapsed_seconds"])


def recompute_metrics(events: list[dict], remaining_deadline: float) -> dict:
    last = events[-1]
    completed = bool(last["terminated"])
    elapsed = sum(row["step_cost_seconds"] for row in events)
    sums = {key: sum(row[key] for row in events) for key in (
        "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
        "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds", "reward")}
    return {"workflow_completed": completed,
            "on_time": completed and elapsed <= remaining_deadline + EPS,
            "service_failures": sum(not row["service_completed"] for row in events),
            "migration_successes": sum(row["migration_success"] for row in events),
            "migration_attempts": sum(row["executed_action"] == 4 for row in events),
            "total_transfer_bytes": (sums["input_transfer_bytes"] + sums["model_transfer_bytes"]
                                     + sums["state_transfer_bytes"]),
            "elapsed_seconds": elapsed,
            "censored_at_continuation_cap": not (last["terminated"] or last["truncated"]),
            **sums}


def recompute_migration(events: list[dict]) -> dict:
    first = events[0]
    transfers = [phase for phase in first["phase_events"] if phase["phase"] == "state_transfer"]
    target = str(transfers[0]["location"]).split("_to_")[-1] if transfers else None
    used = False
    recomputed = False
    if first["migration_success"] and target:
        for event in events[1:]:
            if any(phase["phase"] == "current_node_compute" and phase["location"] == target
                   for phase in event["phase_events"]):
                used = True
                recomputed = any(phase["phase"] == "prefix_recompute" for phase in event["phase_events"])
                break
    return {"first_migration_status": first["migration_status"],
            "first_migration_success": first["migration_success"],
            "first_target_rsu_id_after_execution": target,
            "actual_target_departed_before_delivery": first["migration_status"] == "target_departed_before_delivery",
            "later_target_compute_observed": used,
            "later_target_compute_without_recompute": used and not recomputed}


def classify(a4: dict, a3: dict, use: dict) -> str:
    if a4["censored_at_continuation_cap"] or a3["censored_at_continuation_cap"]:
        return "UNCERTAIN_CENSORED"
    nonharm = (int(a4["workflow_completed"]) >= int(a3["workflow_completed"])
               and int(a4["on_time"]) >= int(a3["on_time"])
               and a4["service_failures"] <= a3["service_failures"])
    if not nonharm:
        return "SERVICE_HARM"
    strict = (int(a4["workflow_completed"]) > int(a3["workflow_completed"])
              or int(a4["on_time"]) > int(a3["on_time"])
              or a4["service_failures"] < a3["service_failures"])
    if strict:
        return "POSITIVE_SERVICE_WITH_COST_ACCOUNTED"
    if use["later_target_compute_without_recompute"] and a4["recompute_seconds"] < a3["recompute_seconds"] - EPS:
        return "POSITIVE_MECHANISM_WITH_RESOURCE_TRADEOFF"
    if (a4["recompute_seconds"] >= a3["recompute_seconds"] - EPS
            and a4["elapsed_seconds"] >= a3["elapsed_seconds"] - EPS
            and a4["total_transfer_bytes"] > a3["total_transfer_bytes"]):
        return "NO_BENEFIT_COSTLY"
    return "MIXED_OR_UNCERTAIN"


def audit(root: Path) -> dict:
    completion, summary, inventory, paired = (read(root / name) for name in (
        "completion_receipt.json", "summary.json", "selection_inventory.json", "paired_regions.json"))
    assert completion["status"] == summary["status"] == "COMPLETE"
    assert completion["run_id"] == summary["run_id"] == RUN_ID
    assert completion["summary_sha256"] == digest(root / "summary.json")
    assert completion["integrity_sha256"] == digest(root / "artifact_integrity.json")
    assert not (root / "failure_receipt.json").exists()
    manifest = read(root / "artifact_integrity.json")["files"]
    listed = {item["path"] for item in manifest}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert listed == actual - {"artifact_integrity.json", "completion_receipt.json"}
    for item in manifest:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], path
    assert len(inventory) == 24 and sum(item["decision"] == "selected" for item in inventory) == 11
    identity = payload(root / "identity_inventory.json", "cscwd_public_full_state_v3_1")
    identities = identity["rows"]
    assert len(identities) == 11 and len({row["start_id"] for row in identities}) == 11
    by_safe = {row["start_id_safe"]: row for row in identities}
    by_id = {row["start_id"]: row for row in identities}
    public_full: defaultdict[str, set[str]] = defaultdict(set)
    for row in identities:
        public_full[row["public_observation_sha256"]].add(row["full_simulation_sha256"])
        assert row["public_observation_group_id"] == row["public_observation_sha256"]
    for row in identities:
        assert row["public_group_distinct_full_state_count"] == len(public_full[row["public_observation_sha256"]])
        assert row["same_public_different_full"] == (len(public_full[row["public_observation_sha256"]]) > 1)
    prefix_steps = 0
    for row in identities:
        record = payload(root / "prefixes" / f"{row['start_id_safe']}.json", "cscwd_public_full_state_v3_1")
        assert record["identity"] == {key: value for key, value in row.items()
                                       if key not in ("full_duplicate_of", "public_observation_group_id",
                                                      "public_group_distinct_full_state_count", "same_public_different_full")}
        events = record["events"]
        assert len(events) == row["prefix_env_steps"] == len(row["prefix_actions"])
        for index, event in enumerate(events):
            recompute_event(event)
            assert event["executed_action"] == row["prefix_actions"][index]
            assert event["step_from_reset"] == index
        prefix_steps += len(events)
    assert prefix_steps == identity["prefix_env_steps"] == summary["prefix_environment_step_count"]

    branches = {}
    branch_events = 0
    for path in sorted((root / "branches").glob("*.json")):
        branch = payload(path, "cscwd_region_branch_v3_1")
        row = by_safe[branch["start_id_safe"]]
        assert row["full_duplicate_of"] is None
        assert branch["start_id"] == row["start_id"]
        assert branch["input_identity"] == {key: row[key] for key in (
            "identity_version", "public_observation_sha256", "full_simulation_sha256")}
        assert branch["first_action"] in row["branch_actions"]
        assert path.name == f"{row['start_id_safe']}_action{branch['first_action']}.json"
        events = branch["events"]
        assert 1 <= len(events) <= 8
        for index, event in enumerate(events):
            recompute_event(event)
            assert event["start_id"] == row["start_id"] and event["condition_id"] == row["condition_id"]
            assert event["step_from_reset"] == len(row["prefix_actions"]) + index
            if index == 0:
                assert event["executed_action"] == branch["first_action"]
                assert event["public_state_sha256_before"] == row["public_observation_sha256"]
            else:
                expected = 3 if event["public_action_mask_before"][3] else 2
                assert event["executed_action"] == expected
        expected = recompute_metrics(events, row["public_features"]["remaining_deadline_seconds"])
        for key, value in expected.items():
            observed = branch["metrics"][key]
            assert close(observed, value) if isinstance(value, float) else observed == value, (path, key)
        use = recompute_migration(events)
        assert branch["posthoc_migration"] == use
        key = (row["start_id"], branch["first_action"])
        assert key not in branches
        branches[key] = {"metrics": expected, "migration": use}
        branch_events += len(events)
    assert len(branches) == summary["completed_branch_count"] == completion["completed_branch_count"] == 27
    assert prefix_steps + branch_events == summary["environment_step_count"] == completion["environment_step_count"]
    assert summary["environment_step_count"] <= completion["max_env_steps_including_prefix_replay"]

    counts = Counter()
    rows = []
    paired_by_id = {row["start_id"]: row for row in paired}
    assert set(paired_by_id) == set(by_id)
    for row in identities:
        start_id = row["start_id"]
        representative = row["full_duplicate_of"] or start_id
        a3, a4 = branches[(representative, 3)], branches[(representative, 4)]
        label = classify(a4["metrics"], a3["metrics"], a4["migration"])
        observed = paired_by_id[start_id]
        assert observed["action4_region"] == label
        assert observed["action4_posthoc_migration"] == a4["migration"]
        contrasts = observed["action4_minus_action3"]
        for key, value in contrasts.items():
            assert close(value, a4["metrics"][key] - a3["metrics"][key]), (start_id, key)
        counts[label] += 1
        rows.append({"start_id": start_id, "condition_id": row["condition_id"],
                     "prefix_actions": row["prefix_actions"],
                     "regression_exposed": row["regression_exposed"],
                     "public_observation_group_id": row["public_observation_group_id"],
                     "full_duplicate_of": row["full_duplicate_of"],
                     "region": label, "action3": a3["metrics"], "action4": a4["metrics"],
                     "action4_minus_action3": contrasts,
                     "action4_posthoc_migration": a4["migration"]})
    assert dict(counts) == summary["region_counts_by_cell"]
    return {"status": "PASS", "source_run_id": RUN_ID,
            "source_summary_sha256": digest(root / "summary.json"),
            "source_integrity_sha256": digest(root / "artifact_integrity.json"),
            "verified_manifest_files": len(manifest), "verified_prefix_files": len(identities),
            "verified_branch_files": len(branches), "prefix_environment_steps": prefix_steps,
            "branch_environment_steps": branch_events, "total_environment_steps": prefix_steps + branch_events,
            "public_observation_group_count": len(public_full),
            "same_public_different_full_group_count": sum(len(group) > 1 for group in public_full.values()),
            "distinct_full_state_count": len({row["full_simulation_sha256"] for row in identities}),
            "region_counts": dict(sorted(counts.items())), "paired_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output_root.resolve()
    if output.exists():
        raise FileExistsError("create-only audit root already exists")
    result = audit(root)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "paired_rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
