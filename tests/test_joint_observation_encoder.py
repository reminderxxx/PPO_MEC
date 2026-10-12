"""Opt-in policy inputs consume v2 state without leaking critic observations."""

from copy import deepcopy
import math

import pytest
import torch

from src.encoders.fusion_encoder import FlatSemanticEncoder, SurrogateFusionEncoder
from src.encoders.joint_observation_features import joint_actor_features, joint_critic_features


def _state():
    node = {"node_id": "n", "node_name": "node", "required_base_model": "b",
            "required_adapter": "a", "input_size": 1, "output_size": 1,
            "predecessors": [], "successors": []}
    rsu = {"rsu_id": "r", "position_x": 0.0, "position_y": 0.0, "coverage_radius": 10,
           "cached_adapter_ids": [], "loading_adapter_ids": ["a"], "ready_base_model_ids": ["b"],
           "capacity_bytes": 150, "occupied_bytes": 130, "reserved_bytes": 30,
           "free_bytes": 20, "loading_remaining_frames": {"adapter:a": 3},
           "loading_object_ids": ["adapter:a"], "shared_state_available": True}
    return {"time_index": 0, "primary_vehicle_id": "v", "vehicles": [{"vehicle_id": "v",
            "associated_rsu_id": "r", "position_x": 0.0, "position_y": 0.0,
            "speed": 1.0, "base_model_id": "b"}], "rsus": [rsu],
            "predictions": {}, "workflow": {"nodes": [node], "edges": [],
            "execution_order": ["n"], "completed_node_ids": [], "current_node_id": "n"},
            "current_workflow_node": node, "current_node_work_units_required": 2,
            "current_node_work_units_progress": 1, "current_node_service_steps_remaining": 1,
            "vehicle_service_steps_remaining": 4, "current_node_parents_ready": True,
            "current_adapter_ready": False, "current_state_ready": True,
            "execution_state_version": 3, "execution_state_payload_bytes": 100,
            "state_copies_by_rsu": {}, "state_transfers_by_rsu": {},
            "shared_state": {"available": True, "age_frames": 2, "transmitted_bytes": 300},
            "centralized_critic_state": {"joint_workflows": {"v": {"remaining_nodes": 2,
            "remaining_work_units": 1, "work_progress": 1, "state_version": 3,
            "parents_ready": True, "execution_location": "r"}}, "rsus": [deepcopy(rsu)],
            "associations": {"v": "r"}}}


@pytest.mark.parametrize("encoder_class", [FlatSemanticEncoder, SurrogateFusionEncoder])
def test_true_joint_state_changes_only_central_critic_context(encoder_class):
    torch.manual_seed(12)
    encoder = encoder_class(joint_observation_enabled=True)
    state = _state()
    before = encoder(state)
    changed = deepcopy(state)
    changed["centralized_critic_state"]["joint_workflows"]["v"]["remaining_work_units"] = 20
    changed["centralized_critic_state"]["rsus"][0]["occupied_bytes"] = 0
    after = encoder(changed)
    for key in ("shared_embedding", "slow_context", "fast_context", "event_context"):
        assert torch.equal(before[key], after[key]), key
    assert not torch.allclose(before["centralized_critic_context"], after["centralized_critic_context"])


@pytest.mark.parametrize("encoder_class", [FlatSemanticEncoder, SurrogateFusionEncoder])
def test_cache_readiness_progress_and_shared_age_change_actual_actor_embedding(encoder_class):
    torch.manual_seed(12)
    encoder = encoder_class(joint_observation_enabled=True)
    baseline = encoder(_state())["shared_embedding"]
    for field, value in (("current_adapter_ready", True), ("vehicle_service_steps_remaining", 40),
                         ("execution_state_version", 20)):
        state = _state()
        state[field] = value
        assert not torch.allclose(baseline, encoder(state)["shared_embedding"]), field
    state = _state()
    state["rsus"][0]["reserved_bytes"] = 100
    assert not torch.allclose(baseline, encoder(state)["shared_embedding"])
    state = _state()
    state["shared_state"]["age_frames"] = 30
    output = encoder(state)
    assert not torch.allclose(baseline, output["shared_embedding"])
    output["shared_embedding"].sum().backward()
    assert encoder._joint_actor_projection[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("encoder_class", [FlatSemanticEncoder, SurrogateFusionEncoder])
def test_flag_off_preserves_legacy_state_keys_and_outputs(encoder_class):
    torch.manual_seed(91)
    legacy = encoder_class()
    torch.manual_seed(91)
    explicit = encoder_class(joint_observation_enabled=False)
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    assert not any("joint_" in key for key in legacy.state_dict())
    assert torch.equal(legacy(_state())["shared_embedding"], explicit(_state())["shared_embedding"])
    torch.manual_seed(91)
    enabled = encoder_class(joint_observation_enabled=True)
    restored = encoder_class(joint_observation_enabled=True)
    restored.load_state_dict(enabled.state_dict(), strict=True)
    assert torch.equal(enabled(_state())["shared_embedding"], restored(_state())["shared_embedding"])


def test_feature_helpers_bound_nonfinite_numeric_fields_and_ignore_private_state_for_actor():
    state = _state()
    state["execution_state_payload_bytes"] = math.nan
    state["vehicle_service_steps_remaining"] = math.inf
    state["shared_state"]["age_frames"] = None
    state["rsus"][0]["capacity_bytes"] = math.inf
    state["rsus"][0]["reserved_bytes"] = -10
    state["centralized_critic_state"]["joint_workflows"]["v"]["remaining_work_units"] = math.nan
    for helper in (joint_actor_features, joint_critic_features):
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in helper(state))
    local = deepcopy(state)
    local.pop("centralized_critic_state")
    assert joint_actor_features(local) == joint_actor_features(state)


@pytest.mark.parametrize("encoder_class", [FlatSemanticEncoder, SurrogateFusionEncoder])
def test_current_broadcast_size_does_not_leak_remote_cache_before_delivery(encoder_class):
    from src.envs.core.joint_causal_observation import SharedStateChannel

    torch.manual_seed(4)
    encoder = encoder_class(joint_observation_enabled=True)
    outputs, visible = [], []
    for adapters in ([], ["private_remote_adapter"]):
        state = _state()
        current = [state["rsus"][0], {"rsu_id": "remote", "cached_adapter_ids": adapters}]
        channel = SharedStateChannel(delay_frames=2)
        snapshot = channel.snapshot(current, 0)
        state["rsus"] = channel.actor_rsus(current, "r", snapshot)
        state["shared_state"] = snapshot["metadata"]
        visible.append(joint_actor_features(state))
        outputs.append(encoder(state)["shared_embedding"])
    assert visible[0] == visible[1]
    assert torch.equal(outputs[0], outputs[1])
