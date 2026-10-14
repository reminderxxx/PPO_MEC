"""Physical accounting regressions for the opt-in joint environment revision."""
from __future__ import annotations

import pytest

from src.envs.core.multi_request_joint_control_env import MultiRequestJointControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode


def _env(positions=None, *, input_size=1, output_size=1, node_count=3,
         state_frames=1, state_bytes=1024, count=1, **kwargs):
    positions = positions or [0.0] * 30
    workflows = {}
    for index in range(count):
        nodes = [WorkflowNode(f"n{step}", f"n{step}", "base", "adapter", input_size,
                              output_size, predecessors=[f"n{step-1}"] if step else [])
                 for step in range(node_count)]
        workflows[f"v{index}"] = WorkflowGraphState(
            f"wf{index}", nodes, [(f"n{s}", f"n{s+1}") for s in range(node_count - 1)],
            [node.node_id for node in nodes], current_node_id="n0")
    return MultiRequestJointControlEnv(
        environment_revision="v2",
        vehicle_adapter_ids=kwargs.pop("vehicle_adapter_ids", {vid: ["adapter"] for vid in workflows}),
        frames=[
            {"time_index": frame * 10, "source_segment_id": "s", "vehicles": [
                VehicleState(vid, x, 0.0, 1.0, "base") for vid in workflows]}
            for frame, x in enumerate(positions)], vehicle_workflows=workflows,
        rsu_states=[RSUState("a", 0, 0, 100), RSUState("b", 10, 0, 100)],
        costs=ReplayCosts(handoff_state_frames=state_frames),
        state_bytes_per_link_frame=state_bytes,
        model_cache_config={"capacity_bytes": 20, "base_size_bytes": 8,
                            "adapter_size_bytes": 2, "transfer_bytes_per_frame": 8}, **kwargs)


def _warm(env, rsu="a"):
    env.model_cache.begin_frame()
    env.model_cache.request_adapter(rsu, "adapter", "base", available_link_slots=1)
    env.model_cache.advance()
    env.model_cache.begin_frame()
    env.model_cache.advance()
    env._sync_cache_view()


def test_strict_cold_cache_mask_is_shared_and_opt_in():
    strict = _env(strict_cold_cache_mask=True)
    cold = strict.decision_states()["v0"]
    assert cold["action_contract"] == "semantic_discrete_5_per_request_v2_cold_cache_mask"
    assert cold["action_mask"][0] and not cold["action_mask"][3]
    with pytest.raises(ValueError, match="illegal semantic action"):
        strict.step_actions({"v0": 3})
    _warm(strict)
    assert strict.decision_states()["v0"]["action_mask"][3]
    assert _env().decision_states()["v0"]["action_mask"][3]


def test_workload_controls_compute_and_progress_is_visible():
    env = _env(input_size=3072)
    _warm(env)
    _, _, _, first = env.step_actions({"v0": 3})
    assert first["events"][0]["status"] == "rsu_compute"
    state = env.decision_states()["v0"]["semantic_state"]
    assert state["current_node_service_steps_required"] == 3
    assert state["current_node_service_steps_remaining"] == 2
    assert state["current_node_work_units_progress"] == 1
    env.step_actions({"v0": 3})
    _, _, _, third = env.step_actions({"v0": 3})
    assert third["completed_nodes"] == 1


def test_state_prepare_completion_is_charged_and_stale_copy_cannot_execute():
    env = _env([0, 3, 6, 6, 6, 6], node_count=5)
    _warm(env, "a")
    _warm(env, "b")
    env.step_actions({"v0": 3})
    _, _, _, prepared = env.step_actions({"v0": 4})
    event = prepared["events"][0]
    assert event["state_action_status"] == "state_prepared"
    assert event["state_transfer_bytes"] == 2
    assert prepared["reward_components"]["state_transfer"] == -0.01
    assert env.state_copies[("v0", "b")]["version"] == 1
    assert env.state_version["v0"] == 2
    assert not env.decision_states()["v0"]["semantic_state"]["current_state_ready"]
    _, _, _, arrival = env.step_actions({"v0": 3})
    assert arrival["events"][0]["status"] == "state_restore"
    assert arrival["completed_nodes"] == 0
    assert arrival["events"][0]["state_transfer_bytes"] == 2  # Only the new prefix delta.
    _, _, _, resumed = env.step_actions({"v0": 3})
    assert resumed["completed_nodes"] == 1


def test_handoff_waits_full_payload_and_charges_terminal_transfer_frame():
    env = _env([0, 6, 6, 6, 6, 6], output_size=4, state_bytes=2)
    _warm(env, "a")
    _warm(env, "b")
    env.step_actions({"v0": 3})
    transfers = []
    for _ in range(3):
        _, _, _, info = env.step_actions({"v0": 3})
        assert info["completed_nodes"] == 0
        assert info["reward_components"]["state_transfer"] == -0.01
        transfers.append(info["events"][0]["state_transfer_bytes"])
    assert transfers == [2, 2, 1]
    assert env.summary()["state_transfer_bytes"] == 5
    assert env.summary()["state_transfer_link_frames"] == 3
    assert env.step_actions({"v0": 3})[3]["completed_nodes"] == 1


def test_fallback_preserves_work_but_requires_state_reception():
    env = _env(input_size=2048)
    _warm(env)
    env.step_actions({"v0": 3})
    _, _, _, restore = env.step_actions({"v0": 2})
    assert restore["events"][0]["status"] == "state_restore"
    assert env.work_progress["v0"] == 1
    for _ in range(3):
        assert env.step_actions({"v0": 2})[3]["completed_nodes"] == 0
    assert env.step_actions({"v0": 2})[3]["completed_nodes"] == 1


def test_dag_parent_availability_prevents_out_of_order_completion():
    env = _env()
    env.workflows["v0"].current_node_id = "n1"
    _warm(env)
    _, _, _, result = env.step_actions({"v0": 3})
    assert result["events"][0]["status"] == "parents_not_ready"
    assert result["completed_nodes"] == 0


def test_shared_observation_is_delayed_and_joint_context_is_critic_only():
    env = _env(count=2)
    _warm(env, "b")
    states = env.decision_states()
    row = states["v0"]
    assert "joint_workflows" not in row["semantic_state"]
    assert len(row["centralized_critic_state"]["joint_workflows"]) == 2
    remote = next(r for r in row["semantic_state"]["rsus"] if r["rsu_id"] == "b")
    assert "cached_adapter_ids" not in remote
    assert "transmitted_bytes" not in row["semantic_state"]["shared_state"]
    assert "total_transmitted_bytes" not in row["semantic_state"]["shared_state"]
    transmitted = env.shared_state_channel.last_result["metadata"]["transmitted_bytes"]
    _, _, _, result = env.step_actions({"v0": 2, "v1": 2})
    assert result["reward_components"]["shared_state"] == pytest.approx(-0.001 * transmitted / 1024)
    assert env.summary()["shared_state_transmitted_bytes"] == transmitted
    remote = next(r for r in env.decision_states()["v0"]["semantic_state"]["rsus"] if r["rsu_id"] == "b")
    assert remote["cached_adapter_ids"] == ["adapter"]
    assert remote["shared_state_age_frames"] == 1


def test_reset_clears_predictor_and_execution_versions():
    env = _env([0, 3, 6, 6])
    env.step_actions({"v0": 2})
    env.decision_states()
    env.reset()
    state = env.decision_states()["v0"]["semantic_state"]
    assert state["execution_state_version"] == 0
    assert state["predictions"]["prediction_as_of_frame"] == 0
    metadata = env.shared_state_channel.last_result["metadata"]
    assert metadata["total_transmitted_bytes"] == metadata["transmitted_bytes"]
    assert "total_transmitted_bytes" not in state["shared_state"]


def test_model_loading_and_state_share_link_without_double_allocation():
    env = _env([0, 6, 6, 6, 6], node_count=4)
    _warm(env)
    env.step_actions({"v0": 3})
    _, _, _, admission = env.step_actions({"v0": 0})
    assert admission["model_transfer_link_frames"] == 1
    assert admission["events"][0]["state_transfer_link_frames"] == 0
    assert admission["events"][0]["status"] == "state_link_queue"
    _, _, _, loading = env.step_actions({"v0": 3})
    assert loading["model_transfer_link_frames"] == 1
    assert loading["events"][0]["state_transfer_link_frames"] == 0
    assert env.step_actions({"v0": 3})[3]["events"][0]["state_transfer_link_frames"] == 1


def test_return_to_previous_rsu_reuses_saved_prefix_without_permanent_ready_flag():
    env = _env([0, 6, 6, 0, 0, 0], node_count=5)
    _warm(env, "a")
    _warm(env, "b")
    env.step_actions({"v0": 3})  # a: version 1, two bytes.
    env.step_actions({"v0": 3})  # b: copy version 1.
    env.step_actions({"v0": 3})  # b: version 2, four bytes.
    _, _, _, returned = env.step_actions({"v0": 3})
    event = returned["events"][0]
    assert event["status"] == "state_restore"
    assert event["state_transfer_bytes"] == 2
    assert not event["prepared_on_arrival"]
    assert env.step_actions({"v0": 3})[3]["completed_nodes"] == 1


def test_partial_transfer_costs_remain_accounted_at_episode_end():
    env = _env([0, 6, 6], output_size=10, state_bytes=2)
    _warm(env, "a")
    _warm(env, "b")
    env.step_actions({"v0": 3})
    env.step_actions({"v0": 3})
    _, _, done, _ = env.step_actions({"v0": 3})
    assert done
    summary = env.summary()
    assert summary["state_transfer_bytes"] == 4
    assert summary["state_transfer_link_frames"] == 2
    assert summary["pending_state_transfer_count"] == 1
    assert summary["pending_state_transfer_bytes"] == 4
    assert summary["state_prepared_count"] == 0


def test_per_frame_receipts_reconstruct_reward_and_transfer_ledgers():
    env = _env([0, 0, 0, 0, 6, 6, 6, 6, 6], count=2, output_size=5)
    rewards = []
    while True:
        actions = {vid: 0 for vid in env.decision_states()}
        _, reward, done, info = env.step_actions(actions)
        rewards.append(reward)
        receipts = [row["frame_cost_receipt"] for row in info["events"] if "frame_cost_receipt" in row]
        assert len(receipts) == 1
        assert sum(receipts[0]["reward_components"].values()) == pytest.approx(reward)
        if done:
            break
    receipts = [row["frame_cost_receipt"] for row in env.events if "frame_cost_receipt" in row]
    assert len(receipts) == env.frame_index
    assert len({row["frame_index"] for row in receipts}) == env.frame_index
    assert sum(sum(row["reward_components"].values()) for row in receipts) == pytest.approx(sum(rewards))
    summary = env.summary()
    assert sum(row["model_transfer_bytes"] for row in receipts) == sum(
        cache["transferred_bytes"] for cache in summary["model_cache"].values())
    assert sum(row["model_transfer_link_frames"] for row in receipts) == summary["model_transfer_link_frames"]
    assert sum(row["shared_state_transmitted_bytes"] for row in receipts) == summary["shared_state_transmitted_bytes"]
    assert sum(row["state_transfer_bytes"] for row in receipts) == summary["state_transfer_bytes"]
    assert sum(row["state_transfer_link_frames"] for row in receipts) == summary["state_transfer_link_frames"]
    assert summary["state_transfer_bytes"] > 0


def test_model_cache_unknown_configuration_is_rejected():
    env = _env()
    env.model_cache_config["capcity_bytes"] = 10
    with pytest.raises(ValueError, match="unknown model_cache_config fields"):
        env.reset()


def test_synthetic_profile_units_are_explicitly_converted():
    env = _env(input_size=1024, output_size=3, workflow_profile_unit_bytes=2)
    _warm(env)
    assert env.decision_states()["v0"]["semantic_state"]["current_node_service_steps_required"] == 2
    env.step_actions({"v0": 3})
    env.step_actions({"v0": 3})
    assert env.state_payload_bytes["v0"] == 8  # Two progress deltas plus 3*2 output bytes.
    assert env.configuration()["workflow_profile_unit_bytes"] == 2
    assert env.configuration()["workflow_size_source"] == "synthetic_profile_units_not_measured_bytes"


def test_undelivered_remote_cache_does_not_leak_through_actor_wire_sizes():
    empty, cached = _env(), _env()
    _warm(cached, "b")
    empty_row, cached_row = empty.decision_states()["v0"], cached.decision_states()["v0"]
    assert empty_row["semantic_state"] == cached_row["semantic_state"]
    assert empty_row["action_mask"] == cached_row["action_mask"]
    assert empty_row["centralized_critic_state"] != cached_row["centralized_critic_state"]
    assert empty._frame_shared_state_bytes != cached._frame_shared_state_bytes


def test_delayed_remote_cache_update_does_not_leak_before_delivery():
    empty, cached = _env(), _env()
    for env in (empty, cached):
        env.step_actions({"v0": 2})
    _warm(cached, "b")
    empty_row, cached_row = empty.decision_states()["v0"], cached.decision_states()["v0"]
    assert empty_row["semantic_state"]["shared_state"]["available"]
    assert empty_row["semantic_state"] == cached_row["semantic_state"]
    assert empty_row["action_mask"] == cached_row["action_mask"]
    assert empty_row["centralized_critic_state"] != cached_row["centralized_critic_state"]
    assert empty._frame_shared_state_bytes != cached._frame_shared_state_bytes
    for env in (empty, cached):
        env.step_actions({"v0": 2})
    assert empty.decision_states()["v0"]["semantic_state"]["rsus"] != cached.decision_states()["v0"]["semantic_state"]["rsus"]


def test_undeclared_vehicle_adapter_cannot_bypass_model_cache():
    env = _env(vehicle_adapter_ids={})
    info = env.decision_states()["v0"]
    assert not info["semantic_state"]["vehicle_adapter_ready"]
    assert not info["action_mask"][2]
    with pytest.raises(ValueError, match="illegal semantic action"):
        env.step_actions({"v0": 2})
    assert env.configuration()["vehicle_adapter_ids"] == {"v0": []}
    assert env.configuration()["vehicle_adapter_policy"] == "declared_only"


def test_declared_vehicle_adapter_allows_execution_only_with_matching_base():
    env = _env(vehicle_adapter_ids={"v0": ["adapter"]})
    assert env.decision_states()["v0"]["action_mask"][2]
    assert env.step_actions({"v0": 2})[3]["events"][0]["status"] == "vehicle_compute"
    env.frames[env.frame_index]["vehicles"][0].base_model_id = "incompatible"
    info = env.decision_states()["v0"]
    assert info["semantic_state"]["vehicle_adapter_ready"]
    assert not info["action_mask"][2]
    with pytest.raises(ValueError, match="illegal semantic action"):
        env.step_actions({"v0": 2})


def test_legacy_vehicle_adapter_assumption_requires_explicit_policy():
    env = _env(vehicle_adapter_ids={}, vehicle_adapter_policy="assumed_available")
    assert env.decision_states()["v0"]["action_mask"][2]
    assert env.step_actions({"v0": 2})[3]["events"][0]["status"] == "vehicle_compute"
    config = env.configuration()
    assert config["vehicle_adapter_policy"] == "assumed_available"
    assert config["fallback_adapter_contract"] == "explicit_legacy_adapter_available_assumption"


@pytest.mark.parametrize("declarations, message", [
    ({"missing_vehicle": ["adapter"]}, "unknown vehicle_adapter_ids vehicles"),
    ({"v0": ["missing_adapter"]}, "unknown vehicle adapter IDs"),
    ({"v0": "adapter"}, "must be a list of adapter IDs"),
])
def test_invalid_vehicle_adapter_declarations_fail_early(declarations, message):
    with pytest.raises(ValueError, match=message):
        _env(vehicle_adapter_ids=declarations)


def test_adapter_list_declares_uniform_static_preload_for_every_vehicle():
    env = _env(count=2, vehicle_adapter_ids=["adapter"])
    states = env.decision_states()
    assert all(info["action_mask"][2] for info in states.values())
    assert env.configuration()["vehicle_adapter_ids"] == {"v0": ["adapter"], "v1": ["adapter"]}
    assert env.configuration()["vehicle_adapter_declaration"] == ["adapter"]
    empty = _env(count=2, vehicle_adapter_ids=[])
    assert all(not info["action_mask"][2] for info in empty.decision_states().values())
