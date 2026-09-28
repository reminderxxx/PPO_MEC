from __future__ import annotations

from scripts.analyze_mechanism_failure_causes import classify_episode
from scripts.evaluate_mechanism_frozen_checkpoints import evaluation_unit_id


def _episode(*, service_success: bool, adapter_hit: bool = True) -> dict:
    return {
        "formal_request_execution_audit": {
            "workflow_completed_under_exogenous_execution": service_success,
            "right_censored": False,
            "external_request_denominator": 1,
            "request_alignment_status": "pass",
            "transfer_mb_per_request": 2.0,
            "workflow_continuity_rate": float(service_success),
            "end_to_end_workflow_delay": 1.0 if service_success else None,
            "request_exposure_fingerprint": "fingerprint",
        },
        "formal_request_exposure": {"requests": [{}]},
        "cache_event_trace": [
            {
                "event_type": "request",
                "service_success": service_success,
                "base_model_hit": True,
                "adapter_hit": adapter_hit,
                "joint_model_hit": adapter_hit,
                "workflow_state_ready": True,
                "stall_occurred": False,
                "capacity_rejection_reason": None,
            }
        ],
        "step_trace": [
            {
                "action_id": 3,
                "handoff_event_count": 0,
                "handoff_ready": False,
                "action_invalid": False,
                "action_precondition_valid": True,
                "cache_target_alignment_mismatch": False,
            }
        ],
        "agent_action_diagnostics": {"action_projection_count": 0},
        "handoff_summary": {},
    }


def test_failure_classifier_separates_dependency_miss_and_denominator() -> None:
    row = classify_episode(_episode(service_success=False, adapter_hit=False))
    assert row["primary_failure_category"] == "cache_dependency_miss"
    assert row["request_failure_count"] == 1
    assert row["adapter_miss_failure_count"] == 1
    assert row["denominator_consistent"] is True


def test_failure_classifier_keeps_conditional_delay_semantics() -> None:
    completed = classify_episode(_episode(service_success=True))
    failed = classify_episode(_episode(service_success=False, adapter_hit=False))
    assert completed["end_to_end_workflow_delay_available"] is True
    assert failed["end_to_end_workflow_delay_available"] is False


def test_evaluation_unit_identity_is_controller_neutral() -> None:
    assert evaluation_unit_id("window_1", "workflow_1") == (
        "mechanism_frozen_checkpoint_evaluation_v1/window_1/workflow_1"
    )
