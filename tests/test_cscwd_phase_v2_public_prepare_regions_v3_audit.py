"""Synthetic accounting checks for the independent v3 artifact reader."""

from __future__ import annotations

import pytest

from scripts.audit_cscwd_phase_v2_public_prepare_regions_v3 import (
    classify, recompute_event, recompute_metrics,
)


def event(**changes):
    row = {
        "public_action_mask_before": [True] * 5, "executed_action": 4,
        "phase_events": [{"phase": "state_transfer", "start_seconds": 0.0,
                          "end_seconds": 2.0, "elapsed_seconds": 2.0}],
        "step_cost_seconds": 2.0,
        "input_transfer_bytes": 10, "model_transfer_bytes": 20,
        "state_transfer_bytes": 5, "backhaul_transfer_bytes": 25,
        "partial_transfer_bytes": 0, "recompute_seconds": 0.0,
        "reward": 3.0, "service_completed": True,
        "migration_success": True, "terminated": True, "truncated": False,
    }
    row.update(changes)
    return row


def test_independent_event_and_metric_reducer() -> None:
    row = event()
    recompute_event(row)
    metrics = recompute_metrics([row], remaining_deadline=3.0)
    assert metrics["workflow_completed"] and metrics["on_time"]
    assert metrics["total_transfer_bytes"] == 35
    assert metrics["migration_successes"] == 1
    assert metrics["migration_attempts"] == 1
    assert not metrics["censored_at_continuation_cap"]
    with pytest.raises(AssertionError):
        recompute_event(event(backhaul_transfer_bytes=24))


def test_censor_precedes_positive_and_costly_regions() -> None:
    base = {"workflow_completed": True, "on_time": True,
            "service_failures": 0, "recompute_seconds": 4.0,
            "elapsed_seconds": 10.0, "total_transfer_bytes": 100,
            "censored_at_continuation_cap": False}
    a3 = dict(base)
    a4 = {**base, "recompute_seconds": 0.0, "elapsed_seconds": 6.0,
          "total_transfer_bytes": 200}
    use = {"later_target_compute_without_recompute": True}
    assert classify(a4, a3, use) == "POSITIVE_MECHANISM_WITH_RESOURCE_TRADEOFF"
    assert classify({**a4, "censored_at_continuation_cap": True}, a3, use) == "UNCERTAIN_CENSORED"
    assert classify({**base, "elapsed_seconds": 11.0, "total_transfer_bytes": 200},
                    a3, use) == "NO_BENEFIT_COSTLY"
