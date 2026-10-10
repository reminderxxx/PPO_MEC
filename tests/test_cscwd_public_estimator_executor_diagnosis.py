"""Read-only witnesses for public-estimator/executor contract differences."""

from __future__ import annotations

from copy import deepcopy

from src.agents.causal_public_action_estimator import (
    UNKNOWN,
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


def test_unknown_target_prepare_is_reported_with_a_known_partial_total() -> None:
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
    # Diagnostic witness: the scalar excludes the unknown target phase instead
    # of becoming unknown, so it is not a full action-cost estimate.
    assert estimate["estimated_total_seconds"] is not None

    _, _, _, _, info = env.step(4)
    actual = info["transition"]
    assert actual["service_completed"] is True
    assert actual["step_cost_seconds"] > estimate["estimated_total_seconds"]


def test_failed_current_service_does_not_commit_estimated_state_transfer() -> None:
    state = _state(current_ready=False)
    estimate = estimate_public_actions(state)["actions"]["4"]
    assert estimate["current_service"] == "no"
    # Diagnostic witness: current estimator includes the conditional state
    # phase even though native execution commits it only after service success.
    assert estimate["state_transfer_bytes"] == 2_000_000
    assert estimate["estimated_total_seconds"] == 2.81

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


def test_mask_has_no_deadline_contact_or_current_bundle_guard() -> None:
    state = _state(current_ready=False, remaining=0.0)
    state["calibrated_context"]["contact_budget_seconds"] = 0.0
    mask_info = ActionMaskBuilder().build_mask_info(state)
    assert mask_info["mask"] == [True, True, True, True, True]
    assert mask_info["semantic_preconditions"]["distinct_handoff_target"]
    # Missing current bundle, zero contact and zero deadline do not create a
    # global action-4 guard; only semantic target availability controls it.
    assert "4" not in mask_info["invalid_reasons"]
