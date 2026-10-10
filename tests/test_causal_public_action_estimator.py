"""Contracts for the causal public calibrated-workflow action estimator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from src.agents.causal_public_action_estimator import (
    CausalPublicImmediateRule,
    CausalPublicTwoStepRule,
    NO,
    SCHEMA_VERSION,
    UNKNOWN,
    YES,
    estimate_public_actions,
    public_prepare_advantage_label,
)


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _state(*, current_ready: bool = True, remaining: float = 60.0) -> dict:
    current_residents = ["base:a", "adapter:x"] if current_ready else ["base:a"]
    return {
        "interface_profile": "raw_ngsim_event_time_v1",
        "primary_vehicle_id": "veh0",
        "vehicles": [{"vehicle_id": "veh0", "associated_rsu_id": "rsu_0"}],
        "rsus": [
            {
                "rsu_id": "rsu_0",
                "typed_resident_object_ids": current_residents,
                "cache_capacity": 100_000_000,
                "cache_used_bytes": 30_000_000 if current_ready else 20_000_000,
            },
            {
                "rsu_id": "rsu_1",
                "typed_resident_object_ids": ["base:a"],
                "cache_capacity": 100_000_000,
                "cache_used_bytes": 20_000_000,
            },
        ],
        "current_workflow_node": {
            "node_id": "n0",
            "required_adapter": "x",
            "input_bytes": 1_000_000,
            "state_bytes": 2_000_000,
            "compute_seconds": 2.0,
            "successors": ["n1"],
        },
        "workflow": {
            "workflow_id": "w0",
            "nodes": [
                {
                    "node_id": "n0",
                    "required_adapter": "x",
                    "input_bytes": 1_000_000,
                    "state_bytes": 2_000_000,
                    "compute_seconds": 2.0,
                    "successors": ["n1"],
                },
                {
                    "node_id": "n1",
                    "required_adapter": "x",
                    "input_bytes": 500_000,
                    "state_bytes": 1_000_000,
                    "compute_seconds": 1.0,
                    "successors": [],
                },
            ],
            "edges": [["n0", "n1"]],
            "execution_order": ["n0", "n1"],
            "completed_node_ids": [],
            "current_node_id": "n0",
            "is_completed": False,
        },
        "predictions": {
            "predicted_next_rsu_by_vehicle": {"veh0": "rsu_1"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh0": "rsu_1"},
            "next_rsu_sequence": {"veh0": ["rsu_1", "rsu_1"]},
            "prediction_confidence_by_vehicle": {"veh0": 0.8},
            "prediction_uncertainty_by_vehicle": {"veh0": 0.2},
        },
        "calibrated_context": {
            "contact_budget_seconds": 10.0,
            "state_bytes": 2_000_000,
            "input_bytes": 1_000_000,
            "compute_seconds": 2.0,
            "required_bundle_ids": ["base:a", "adapter:x"],
            "object_catalog": {
                "base:a": {
                    "resident_bytes": 20_000_000,
                    "transfer_bytes": 10_000_000,
                    "load_seconds": 0.3,
                },
                "adapter:x": {
                    "resident_bytes": 10_000_000,
                    "transfer_bytes": 5_000_000,
                    "load_seconds": 0.2,
                },
            },
            "adapter_to_bundle": {"x": ["base:a", "adapter:x"]},
            "link": {"estimated_mbps": 100.0, "fixed_seconds": 0.02},
            "measured_time_seconds": {"state_restore_overhead": 0.01},
            "vehicle_fallback_seconds": 8.0,
            "failed_service_seconds": 2.0,
            "prepared_state_prefix": {
                "current": {
                    "known": True,
                    "exists": False,
                    "valid": False,
                    "missing_completed_fraction": 0.0,
                },
                "predicted_target": {
                    "known": True,
                    "exists": False,
                    "valid": False,
                    "missing_completed_fraction": 0.0,
                },
            },
            "time_contract": {
                "schema_version": "raw_ngsim_event_time_v1",
                "clock_seconds": 0.0,
                "deadline_seconds": remaining,
                "remaining_deadline_seconds": remaining,
                "trace_remaining_seconds": 2.2,
                "trace_remaining_policy_visible": False,
                "contact_scope": "current_rsu",
            },
        },
    }


def test_public_estimate_is_pure_and_ignores_private_future_fields() -> None:
    state = _state()
    before = _hash(state)
    estimate = estimate_public_actions(state)
    assert estimate["schema_version"] == "causal_public_action_estimator_v2"
    assert estimate["schema_version"] == SCHEMA_VERSION
    assert _hash(state) == before
    assert estimate["actions"]["4"]["current_service"] == YES
    assert estimate["actions"]["4"]["target_prepare"] == YES
    assert estimate["actions"]["4"]["deadline_fit"] == YES
    assert estimate["actions"]["2"]["estimated_total_seconds"] == 10.1

    changed = deepcopy(state)
    changed["actual_future_route"] = ["secret_a", "secret_b"]
    changed["future_source_positions"] = [[1, 2], [3, 4]]
    changed["calibrated_context"]["time_contract"]["trace_remaining_seconds"] = 0.0
    assert estimate_public_actions(changed) == estimate


def test_prepare_serve_and_abstain_contract_has_no_tuned_threshold() -> None:
    positive = public_prepare_advantage_label(_state())
    assert positive["decision"] == "prepare"
    assert positive["event_target"] == 1
    assert positive["supervision_weight"] == 1.0

    current_missing = public_prepare_advantage_label(_state(current_ready=False))
    assert current_missing["decision"] == "serve"
    assert current_missing["reason"] == "action4_fails_current_service"

    deadline = public_prepare_advantage_label(_state(remaining=2.0))
    assert deadline["decision"] == "serve"
    assert deadline["reason"] == "prepare_misses_deadline_while_service_fits"

    unknown = _state()
    del unknown["calibrated_context"]["time_contract"]
    label = public_prepare_advantage_label(unknown)
    assert label["decision"] == "abstain"
    assert label["supervision_weight"] == 0.0


def test_private_eviction_order_is_unknown_not_fabricated_infeasible() -> None:
    state = _state()
    target = state["rsus"][1]
    target["cache_used_bytes"] = target["cache_capacity"]
    estimate = estimate_public_actions(state)
    assert estimate["target_bundle"]["admission"] == UNKNOWN
    assert estimate["actions"]["4"]["target_prepare"] == UNKNOWN
    assert estimate["actions"]["4"]["estimated_total_seconds"] is None
    assert estimate["actions"]["4"]["deadline_fit"] == UNKNOWN
    assert public_prepare_advantage_label(state)["decision"] == "abstain"


def test_raw_full_step_contact_failure_abstains_without_future_truth() -> None:
    state = _state()
    state["calibrated_context"]["contact_budget_seconds"] = 0.9
    label = public_prepare_advantage_label(state)
    action4 = label["estimate"]["actions"]["4"]
    assert action4["target_prepare_contact_fit"] == YES
    assert action4["raw_full_step_contact_fit"] == NO
    assert action4["raw_trace_fit"] == UNKNOWN
    assert label["decision"] == "abstain"
    assert label["reason"] == "raw_full_step_contact_insufficient"


class _PublicOnlySource:
    def __init__(self, info: dict) -> None:
        self.info = info

    def _info(self):
        return deepcopy(self.info)

    def clone(self):  # pragma: no cover - called only on contract violation
        raise AssertionError("public rule must not clone")

    def clone_for_decision_model(self):  # pragma: no cover
        raise AssertionError("public rule must not preview")

    def step(self, _action):  # pragma: no cover
        raise AssertionError("public rule must not step")


def test_public_rules_are_deterministic_and_do_not_preview_environment() -> None:
    state = _state()
    info = {"semantic_state": state, "action_mask": [True] * 5}
    source = _PublicOnlySource(info)
    for rule in (CausalPublicImmediateRule(), CausalPublicTwoStepRule()):
        first = rule.select_action(source)
        second = rule.select_action(source)
        assert first == second
        assert first in range(5)
    assert _hash(state) == _hash(_state())


def test_current_service_and_prepare_costs_are_action_symmetric() -> None:
    estimate = estimate_public_actions(_state())
    assert estimate["actions"]["0"]["current_service"] == YES
    assert estimate["actions"]["3"]["current_service"] == YES
    assert estimate["actions"]["4"]["current_service"] == YES
    assert estimate["actions"]["2"]["model_transfer_bytes"] == 0
    assert estimate["actions"]["2"]["input_transfer_bytes"] == 1_000_000
    assert estimate["actions"]["1"]["model_transfer_bytes"] == 5_000_000
    assert estimate["actions"]["4"]["model_transfer_bytes"] == 5_000_000
    assert estimate["actions"]["4"]["state_transfer_bytes"] == 2_000_000
    assert estimate["actions"]["1"]["target_state_commit"] == "not_applicable"
    assert estimate["actions"]["4"]["target_state_commit"] == YES
    assert NO != YES
