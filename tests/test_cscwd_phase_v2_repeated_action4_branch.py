"""Small contract tests for the bounded repeated-action branch diagnostic."""

from __future__ import annotations

import pytest

from scripts.confirm_cscwd_phase_v2_repeated_action4 import (
    CONDITIONS, SOURCE_WINDOW, branch_metrics, choose_continuation,
    event_row, locally_dominated, select_witnesses,
)


def pilot_rows() -> list[dict[str, str]]:
    rows = []
    repeated_by_condition = {"default": {7, 29, 61}, "adverse_link": {7, 29, 61},
                             "ample_cache": {7, 17, 29, 43, 61}}
    for condition in CONDITIONS:
        for seed in (7, 17, 29, 43, 61):
            repeated = seed in repeated_by_condition[condition]
            for step in range(2):
                rows.append({
                    "arm": "sa_candidate", "view": "selected", "time_profile": "raw_ngsim_event_time_v2",
                    "source_segment_id": "us_101", "window_id": SOURCE_WINDOW,
                    "condition_id": condition, "seed": str(seed), "step": str(step),
                    "executed_action": "4" if repeated else "3",
                    "migration_status": ("delivered" if step == 0 else "target_departed_before_delivery")
                    if repeated else "not_requested",
                    "migration_success": str(bool(repeated and step == 0)),
                })
    return rows


def test_witness_grid_and_no_replacement() -> None:
    selected = select_witnesses(pilot_rows())
    assert [len(selected[key]) for key in CONDITIONS] == [3, 3, 5]
    changed = pilot_rows()
    changed[0]["migration_status"] = "not_delivered"
    with pytest.raises(RuntimeError, match="unregistered failed action4"):
        select_witnesses(changed)


def test_public_continuation_uses_only_mask() -> None:
    assert choose_continuation([True, True, True, True, True]) == 3
    assert choose_continuation([False, False, True, False, False]) == 2
    with pytest.raises(RuntimeError, match="no legal action"):
        choose_continuation([False] * 5)


def test_branch_cost_and_service_dominance() -> None:
    row = {
        "input_transfer_bytes": 10, "model_transfer_bytes": 20, "state_transfer_bytes": 3,
        "backhaul_transfer_bytes": 23, "partial_transfer_bytes": 0,
        "service_completed": True, "migration_success": False, "executed_action": 4,
        "recompute_seconds": 1.5, "step_cost_seconds": 2.0, "reward": 3.0,
    }
    action4 = branch_metrics([row], {"workflow_completed": True, "on_time_workflow_completed": 1}, censored=False)
    smaller = dict(row, state_transfer_bytes=0, backhaul_transfer_bytes=20, executed_action=3)
    action3 = branch_metrics([smaller], {"workflow_completed": True, "on_time_workflow_completed": 1}, censored=False)
    assert action4["total_transfer_bytes"] == 33
    assert locally_dominated(action4, action3)
    assert not locally_dominated(action4, dict(action3, service_failures=1))
    assert not locally_dominated(action4, dict(action3, censored_at_continuation_cap=True))
    with pytest.raises(RuntimeError, match="conservation"):
        branch_metrics([dict(row, backhaul_transfer_bytes=22)],
                       {"workflow_completed": True}, censored=False)


def test_phase_ledger_rejects_clock_or_byte_drift() -> None:
    info = {"semantic_state": {"calibrated_context": {}}, "action_mask": [True] * 5}
    event = {
        "action": 4, "service_completed": True, "migration_success": False,
        "migration_status": "target_departed_before_delivery", "failure_reason": None,
        "input_transfer_bytes": 10, "model_transfer_bytes": 20, "state_transfer_bytes": 3,
        "backhaul_transfer_bytes": 23, "partial_transfer_bytes": 0,
        "step_cost_seconds": 1.0, "reward": 2.0,
        "phase_events": [{"phase": "input_upload", "elapsed_seconds": 1.0}],
    }
    assert event_row("default", "action4", 1, info, event, False, False)["state_transfer_bytes"] == 3
    with pytest.raises(RuntimeError, match="phase/step"):
        event_row("default", "action4", 1, info, dict(event, step_cost_seconds=2.0), False, False)
