from __future__ import annotations

from scripts.finalize_calibrated_workflow_event_aux_abstention import (
    CANDIDATE,
    _summary_row,
)


def test_summary_separates_failure_episodes_from_attempts() -> None:
    rows = [
        {
            "service_failure_rate": "1.0",
            "service_failures": "3",
            "workflow_completion_rate": "1.0",
            "on_time_workflow_completion_rate": "0.0",
            "reward": "1.0",
            "action_0": "2",
            "action_1": "0",
            "action_2": "0",
            "action_3": "1",
            "action_4": "0",
            "current_missing_action_0": "2",
            "current_missing_action_1": "0",
            "current_missing_action_2": "0",
            "current_missing_action_3": "0",
            "current_missing_action_4": "0",
            "model_prepare_mb": "4",
            "state_transfer_mb": "1",
            "input_transfer_mb": "2",
            "total_transfer_mb": "7",
            "recompute_seconds": "5",
            "completed_sample_elapsed_seconds": "9",
        },
        {
            "service_failure_rate": "0.0",
            "service_failures": "0",
            "workflow_completion_rate": "1.0",
            "on_time_workflow_completion_rate": "1.0",
            "reward": "3.0",
            "action_0": "1",
            "action_1": "0",
            "action_2": "1",
            "action_3": "1",
            "action_4": "0",
            "current_missing_action_0": "1",
            "current_missing_action_1": "0",
            "current_missing_action_2": "1",
            "current_missing_action_3": "0",
            "current_missing_action_4": "0",
            "model_prepare_mb": "2",
            "state_transfer_mb": "0",
            "input_transfer_mb": "1",
            "total_transfer_mb": "3",
            "recompute_seconds": "1",
            "completed_sample_elapsed_seconds": "3",
        },
    ]
    summary = _summary_row(
        rows,
        arm=CANDIDATE,
        view="selected",
        split="combined",
        seed="combined",
    )
    assert summary["episode_n"] == 2
    assert summary["failure_episode_n"] == 1
    assert summary["failure_attempt_n"] == 3
    assert summary["mean_on_time_workflow_completion_rate"] == 0.5
    assert summary["sum_action_0"] == 3.0
