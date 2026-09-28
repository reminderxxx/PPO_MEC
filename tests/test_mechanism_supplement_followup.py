from __future__ import annotations

from scripts.analyze_mechanism_failure_causes import classify_episode
from scripts.evaluate_mechanism_frozen_checkpoints import (
    _aggregate,
    _normalise_settings,
    evaluation_unit_id,
)
from src.agents.mappo_agent import MAPPOAgent


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


def test_frozen_evaluation_settings_keep_alias_and_guard_mode() -> None:
    settings = _normalise_settings(
        {
            "settings": [
                {
                    "setting_id": "candidate_guard_off",
                    "agent_name": "sa_ghmappo",
                    "controller_role": "attribution",
                    "guard_mode": "disabled_at_inference",
                    "agent_config_overrides": {
                        "cache_warm_start_guard_enabled": False,
                    },
                }
            ]
        }
    )
    assert settings[0]["setting_id"] == "candidate_guard_off"
    assert settings[0]["agent_name"] == "sa_ghmappo"
    assert settings[0]["guard_mode"] == "disabled_at_inference"


def test_frozen_evaluation_aggregate_reports_guard_action_share() -> None:
    row = {
        "controller": "sa_ghmappo",
        "controller_role": "candidate",
        "guard_mode": "enabled_current_only",
        "total_steps": 5,
        "cache_warm_start_guard_count": 3,
        "guard_action_delta_count": 2,
        "final_action_counts": {"0": 3, "2": 2},
        "workflow_completed": True,
        "workflow_continuity_rate": 0.8,
        "handoff_failure_rate": 0.0,
        "full_service_ready_request_rate": 0.8,
        "transfer_mb_per_request": 2.0,
        "backhaul_traffic_cost": 10.0,
        "adapter_state_migration_overhead": 1.0,
        "request_failure_count": 1,
        "right_censored": False,
        "end_to_end_workflow_delay": 5.0,
    }
    aggregate = _aggregate("candidate_guard_on", [row])
    assert aggregate["cache_warm_start_guard_rate"] == 0.6
    assert aggregate["effective_policy_action_share"] == 0.4
    assert aggregate["final_action_counts"] == {"0": 3, "2": 2}


def test_symmetric_current_service_readiness_guard_is_configurable_for_mappo() -> None:
    agent = MAPPOAgent(
        cache_warm_start_guard_enabled=True,
        cache_warm_start_guard_current_only=True,
    )
    semantic_state = {
        "primary_vehicle_id": "vehicle_0",
        "vehicles": [
            {"vehicle_id": "vehicle_0", "associated_rsu_id": "rsu_0"}
        ],
        "current_workflow_node": {"required_adapter": "adapter_a"},
        "predictions": {
            "predicted_first_handoff_rsu_by_vehicle": {"vehicle_0": "rsu_1"}
        },
        "rsus": [
            {"rsu_id": "rsu_0", "cached_adapter_ids": []},
            {"rsu_id": "rsu_1", "cached_adapter_ids": []},
        ],
    }
    selected = {"slow": 0, "fast": 1, "event": 1}
    result = agent._apply_cache_warm_start_guard_to_actions(
        semantic_state=semantic_state,
        selected_actions=selected,
    )
    assert result["guarded"] is True
    assert selected["slow"] == 1
    assert selected["event"] == 0

    semantic_state["rsus"][0]["cached_adapter_ids"] = ["adapter_a"]
    selected = {"slow": 0, "fast": 1, "event": 1}
    result = agent._apply_cache_warm_start_guard_to_actions(
        semantic_state=semantic_state,
        selected_actions=selected,
    )
    assert result["guarded"] is False
    assert result["reason"] == "current_adapter_ready_current_only"
    assert selected == {"slow": 0, "fast": 1, "event": 1}
