"""Contract tests for outcome-blind public prepare region coverage."""

from __future__ import annotations

import json

import pytest

from scripts.confirm_cscwd_phase_v2_public_prepare_regions import (
    CONDITIONS, enumerate_public_starts, posthoc_migration_use, region,
)


def behavior_rows() -> list[dict[str, str]]:
    rows = []
    for condition in CONDITIONS:
        actions = (0, 4) if condition == "ample_cache" else (0, 1, 4)
        for seed, first in enumerate(actions, start=1):
            for step, action in enumerate((first, 3)):
                rows.append({
                    "source_segment_id": "us_101", "condition_id": condition,
                    "arm": "sa_candidate" if first != 0 else "ppo",
                    "view": "selected", "seed": str(seed),
                    "window_id": "frozen_us101", "method": "sa_ghmappo" if first != 0 else "ppo",
                    "step": str(step), "executed_action": str(action),
                    "action_mask": json.dumps([True, True, True, True, True]),
                    "time_profile": "raw_ngsim_event_time_v2",
                    "migration_status": "poison_outcome_must_not_be_used",
                    "reward": "999999",
                })
    return rows


def metrics(**overrides) -> dict:
    result = {"workflow_completed": True, "on_time": True, "service_failures": 0,
              "recompute_seconds": 4.0, "elapsed_seconds": 10.0,
              "total_transfer_bytes": 100, "censored_at_continuation_cap": False}
    result.update(overrides)
    return result


def test_outcome_blind_full_shallow_inventory() -> None:
    rows = behavior_rows()
    selected, inventory = enumerate_public_starts(rows, [])
    assert len(selected) == 11
    assert len(inventory) == 24
    assert sum(item["regression_exposed"] for item in selected) == 3
    assert sum(len(item["branch_actions"]) for item in selected) <= 36
    assert all(item["source_segment_id"] == "us_101" for item in selected)
    assert next(item for item in inventory if item["source_segment_id"] == "us_101"
                and item["condition_id"] == "ample_cache" and item["prefix_actions"] == [1])[
                    "decision"] == "excluded_not_observed_in_frozen_behavior"
    for row in rows:
        row["migration_status"] = "different_posthoc_outcome"
        row["reward"] = "-999999"
    changed, _ = enumerate_public_starts(rows, [])
    assert [(item["condition_id"], item["prefix_actions"], item["branch_actions"])
            for item in changed] == [(item["condition_id"], item["prefix_actions"], item["branch_actions"])
                                  for item in selected]


def test_mask_drift_fails_before_selection() -> None:
    rows = behavior_rows()
    rows[0]["action_mask"] = json.dumps([True, True, True, False, True])
    with pytest.raises(RuntimeError, match="inconsistent action masks"):
        enumerate_public_starts(rows, [])


def test_positive_requires_delivery_use_and_recompute_gain() -> None:
    ledger = [
        {"migration_status": "delivered", "migration_success": True,
         "phase_events": [{"phase": "state_transfer", "location": "rsu_0_to_rsu_1"}]},
        {"phase_events": [{"phase": "current_node_compute", "location": "rsu_1"}]},
    ]
    use = posthoc_migration_use(ledger)
    assert use["later_target_compute_without_recompute"]
    assert region(metrics(recompute_seconds=0, total_transfer_bytes=200), metrics(), use) == (
        "POSITIVE_MECHANISM_WITH_RESOURCE_TRADEOFF")
    assert region(metrics(recompute_seconds=4, total_transfer_bytes=200, elapsed_seconds=11),
                  metrics(), use) == "NO_BENEFIT_COSTLY"
    assert region(metrics(recompute_seconds=0, censored_at_continuation_cap=True),
                  metrics(), use) == "UNCERTAIN_CENSORED"


def test_first_step_departure_is_posthoc_only() -> None:
    use = posthoc_migration_use([{"migration_status": "target_departed_before_delivery",
                                  "migration_success": False,
                                  "phase_events": [{"phase": "state_transfer", "location": "rsu_1_to_rsu_2"}]}])
    assert use["actual_target_departed_before_delivery"]
    assert not use["later_target_compute_without_recompute"]
