"""Focused checks for the read-only pilot evidence reconstruction."""

import json

import pytest

from scripts.audit_cscwd_phase_v2_pilot_independent import (
    OBJECTIVE, V2, annotate_events, recompute_episode,
)


def _one_node_episode():
    transfer_penalty = -OBJECTIVE["transfer_gib_weight"] * 100 / (1024**3)
    reward = 2.0 + 4.0 - 0.08 + transfer_penalty
    row = {
        "time_profile": V2, "completed_nodes": "1", "node_count": "1",
        "service_failures": "0", "migration_successes": "0", "migration_attempts": "0",
        "workflow_completion_rate": "1", "on_time_workflow_completion_rate": "1",
        "late_workflow_completion_rate": "0", "unfinished_rate": "0",
        "service_failure_rate": "0", "completed_sample_elapsed_seconds": "1.1",
        "elapsed_seconds_for_failed_or_truncated": "", "total_transfer_mb": "0.0001",
        "input_transfer_bytes": "100", "model_transfer_bytes": "0",
        "state_transfer_bytes": "0", "backhaul_transfer_bytes": "0",
        "partial_transfer_bytes": "0", "recompute_seconds": "0",
        "phase_elapsed_seconds": json.dumps({"input_upload": 0.1, "current_node_compute": 0.9}),
        "reward": str(reward), "reward_node_completion": "2",
        "reward_workflow_completion": "4", "reward_time_penalty": "-0.08",
        "reward_transfer_penalty": str(transfer_penalty),
        "reward_failure_penalty": "0", "reward_deadline_penalty": "0",
        **{f"action_{index}": "1" if index == 0 else "0" for index in range(5)},
    }
    phases = [
        {"phase": "input_upload", "location": "vehicle_to_rsu_0",
         "elapsed_seconds": 0.1, "transferred_bytes": 100},
        {"phase": "current_node_compute", "location": "rsu_0",
         "elapsed_seconds": 0.9, "transferred_bytes": 0},
    ]
    step = {
        "step": "0", "requested_action": "0", "executed_action": "0",
        "action_mask": "[true, false, true, true, false]",
        "service_completed": "True", "migration_success": "False",
        "migration_status": "not_requested", "failure_reason": "",
        "step_cost_seconds": "1.0", "reward": str(reward),
        "input_transfer_bytes": "100", "model_transfer_bytes": "0",
        "state_transfer_bytes": "0", "backhaul_transfer_bytes": "0",
        "partial_transfer_bytes": "0", "phase_events": json.dumps(phases),
        "terminated": "True", "truncated": "False",
    }
    return row, [step]


def test_reconstructs_original_reward_and_binary_failure_denominator():
    row, steps = _one_node_episode()
    actual = recompute_episode(row, steps)
    assert actual["workflow_completed"] == 1
    assert actual["on_time"] == 1
    assert actual["service_failure_episode"] == 0
    assert actual["total_transfer_bytes"] == 100


def test_rejects_changed_transfer_accounting():
    row, steps = _one_node_episode()
    row["total_transfer_mb"] = "0"
    with pytest.raises(RuntimeError, match="total MB"):
        recompute_episode(row, steps)


def test_event_attribution_marks_observed_use_without_claiming_counterfactual():
    first = {
        "step": 0, "action": 4, "service_completed": True,
        "migration_success": True, "migration_status": "delivered",
        "failure_reason": "", "input_transfer_bytes": 0,
        "model_transfer_bytes": 100, "state_transfer_bytes": 10,
        "partial_transfer_bytes": 0, "step_cost_seconds": 1.0, "reward": 0.0,
        "phases": [
            {"phase": "model_prepare_network", "location": "repository_to_rsu_1",
             "transferred_bytes": 100, "elapsed_seconds": 0.2},
            {"phase": "state_transfer", "location": "rsu_0_to_rsu_1",
             "transferred_bytes": 10, "elapsed_seconds": 0.1},
        ],
    }
    second = {
        **first, "step": 1, "action": 3, "model_transfer_bytes": 0,
        "state_transfer_bytes": 0, "migration_success": False,
        "migration_status": "not_requested",
        "phases": [{"phase": "current_node_compute", "location": "rsu_1",
                    "transferred_bytes": 0, "elapsed_seconds": 0.5}],
    }
    labels, counts = annotate_events(
        ("sa_candidate", "selected", "7", V2, "default", "us_101", "window"),
        {"events": [first, second]},
    )
    assert labels[0]["model_observed_use"] == "later_compute_at_prepared_rsu_observed"
    assert labels[0]["post_delivery_observation"] == "later_target_compute_without_recompute"
    assert counts["state:later_target_compute_without_recompute:bytes"] == 10
