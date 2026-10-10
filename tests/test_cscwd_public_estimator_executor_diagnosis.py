"""Regression witnesses for public-estimator/executor phase conformance."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.agents.causal_public_action_estimator import (
    NO,
    UNKNOWN,
    YES,
    estimate_public_actions,
)
from src.envs.core.calibrated_continuous_workflow_env import (
    PREPARED_STATE_PREFIX_PROFILE,
)
from src.envs.specs.action_schema import ActionMaskBuilder
from tests.test_calibrated_workflow_prepared_state_prefix import _env, _fixture
from tests.test_causal_public_action_estimator import _state


def _public_estimator_state(env, config: dict) -> tuple[dict, list[bool]]:
    info = env._info()
    state = deepcopy(info["semantic_state"])
    context = state["calibrated_context"]
    context["time_contract"] = {
        "schema_version": "diagnostic_only",
        "clock_seconds": env.clock_seconds,
        "deadline_seconds": 100.0,
        "remaining_deadline_seconds": 100.0 - env.clock_seconds,
        "contact_scope": "current_rsu",
    }
    context["vehicle_fallback_seconds"] = float(
        config["vehicle"]["fallback_seconds"]
    )
    context["failed_service_seconds"] = float(
        config["objective"]["failed_service_seconds"]
    )
    return state, list(info["action_mask"])


def test_native_action4_services_current_node_before_state_commit() -> None:
    config, instance = _fixture()
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    current = env._current_rsu_id()
    node = env._current_node()
    bundle = env._bundle_ids(str(node["required_adapter"]))
    env.caches[current].residents = list(bundle)
    env.caches[current].last_used = {item: 0 for item in bundle}
    completed_before = env.metrics["completed_nodes"]

    _, _, _, _, info = env.step(4)
    transition = info["transition"]
    assert transition["action"] == 4
    assert transition["service_completed"] is True
    assert transition["migration_success"] is True
    assert transition["state_transfer"]["status"] == (
        "prepared_after_current_node_completion"
    )
    assert env.metrics["completed_nodes"] == completed_before + 1


def test_unknown_target_prepare_fails_closed_without_partial_total() -> None:
    config, instance = _fixture()
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    current = env._current_rsu_id()
    node = env._current_node()
    bundle = env._bundle_ids(str(node["required_adapter"]))
    env.caches[current].residents = list(bundle)
    env.caches[current].last_used = {item: 0 for item in bundle}
    state, mask = _public_estimator_state(env, config)

    estimate = estimate_public_actions(state, mask)["actions"]["4"]
    assert estimate["target_prepare"] == UNKNOWN
    assert "requires_private_eviction_order" in estimate["unknown_reasons"]
    assert estimate["cost_status"] == "unknown_required_phase"
    assert estimate["estimated_conditional_seconds"] is None
    assert estimate["estimated_total_seconds"] is None
    assert estimate["deadline_fit"] == UNKNOWN
    assert estimate["phase_status"]["target_model_prepare"] == UNKNOWN
    assert estimate["actual_executed_seconds"] is None
    assert estimate["actual_execution_cost_status"] == "unavailable_online"


def test_failed_current_service_skips_estimated_state_commit() -> None:
    state = _state(current_ready=False)
    estimate = estimate_public_actions(state)["actions"]["4"]
    assert estimate["current_service"] == "no"
    assert estimate["state_transfer_bytes"] == 0
    assert estimate["target_state_commit"] == NO
    assert estimate["phase_status"]["target_state_commit"] == (
        "skipped_current_service_failed"
    )
    # Model staging (0.62 s) remains committed, followed by the 2 s failure;
    # state network/restore is not charged on the failed-service branch.
    assert estimate["estimated_total_seconds"] == pytest.approx(2.62)

    config, instance = _fixture()
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    current = env._current_rsu_id()
    env.caches[current].residents = []
    env.caches[current].last_used = {}
    _, _, _, _, info = env.step(4)
    actual = info["transition"]
    assert actual["service_completed"] is False
    assert actual["migration_success"] is False
    assert actual["model_transfer_bytes"] > 0
    assert actual["state_transfer_bytes"] == 0


@pytest.mark.parametrize("current_ready", [True, False])
def test_known_action4_phase_estimate_matches_native_consumer(
    current_ready: bool,
) -> None:
    config, instance = _fixture()
    instance = deepcopy(instance)
    instance["link_profile"]["actual_mbps"] = instance["link_profile"][
        "estimated_mbps"
    ]
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    current = env._current_rsu_id()
    target = env._predicted_handoff_target()
    assert target is not None
    node = env._current_node()
    bundle = env._bundle_ids(str(node["required_adapter"]))
    env.caches[current].residents = list(bundle) if current_ready else []
    env.caches[current].last_used = {
        item: 0 for item in env.caches[current].residents
    }
    env.caches[target].residents = []
    env.caches[target].last_used = {}
    state, mask = _public_estimator_state(env, config)

    estimate = estimate_public_actions(state, mask)["actions"]["4"]
    _, _, _, _, info = env.step(4)
    actual = info["transition"]
    assert estimate["cost_status"] == "known_conditional_estimate"
    assert estimate["estimated_total_seconds"] == pytest.approx(
        actual["step_cost_seconds"]
    )
    assert estimate["model_transfer_bytes"] == actual["model_transfer_bytes"]
    assert estimate["state_transfer_bytes"] == actual["state_transfer_bytes"]
    assert estimate["current_service"] == (YES if current_ready else NO)
    assert estimate["target_state_commit"] == (
        YES if current_ready else NO
    )


def test_prepare_contact_and_raw_full_step_contact_are_distinct() -> None:
    state = _state()
    # action4 prepare is 0.81 s while the full conditional step is 2.81 s.
    state["calibrated_context"]["contact_budget_seconds"] = 0.9
    estimate = estimate_public_actions(state)["actions"]["4"]
    assert estimate["target_prepare_contact_fit"] == YES
    assert estimate["estimated_total_seconds"] == pytest.approx(2.81)
    assert estimate["raw_full_step_contact_fit"] == NO
    assert estimate["raw_trace_fit"] == UNKNOWN
    assert estimate["raw_execution_fit"] == NO


def test_hidden_trace_end_does_not_enter_online_estimate() -> None:
    left = _state()
    right = deepcopy(left)
    left["calibrated_context"]["time_contract"][
        "trace_remaining_seconds"
    ] = 100.0
    right["calibrated_context"]["time_contract"][
        "trace_remaining_seconds"
    ] = 0.0
    left_estimate = estimate_public_actions(left)
    right_estimate = estimate_public_actions(right)
    assert left_estimate == right_estimate
    assert left_estimate["actions"]["4"]["raw_trace_fit"] == UNKNOWN


def test_mask_has_no_deadline_contact_or_current_bundle_guard() -> None:
    state = _state(current_ready=False, remaining=0.0)
    state["calibrated_context"]["contact_budget_seconds"] = 0.0
    mask_info = ActionMaskBuilder().build_mask_info(state)
    assert mask_info["mask"] == [True, True, True, True, True]
    assert mask_info["semantic_preconditions"]["distinct_handoff_target"]
    # Missing current bundle, zero contact and zero deadline do not create a
    # global action-4 guard; only semantic target availability controls it.
    assert "4" not in mask_info["invalid_reasons"]
