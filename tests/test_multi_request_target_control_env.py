"""Counterfactual controls must change service while conserving physical costs."""
from copy import deepcopy

import pytest

from src.envs.core.multi_request_target_control_env import JointRequestControl, MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode


def make_env(*, count=2, nodes=2, frames=50, positions=None, input_size=4, output_size=2, bases=None, **kwargs):
    workflows = {}
    for i in range(count):
        workflow_nodes = [WorkflowNode(f"n{j}", f"n{j}", bases[j] if bases else "base", f"adapter{j if bases else j % 2}",
            input_size, output_size, predecessors=[f"n{j-1}"] if j else []) for j in range(nodes)]
        workflows[f"v{i}"] = WorkflowGraphState(f"wf{i}", workflow_nodes,
            [(f"n{j}", f"n{j+1}") for j in range(nodes-1)],
            [node.node_id for node in workflow_nodes], current_node_id="n0")
    positions = positions or [0.0] * frames
    return MultiRequestTargetControlEnv(
        frames=[{"time_index": j * 10, "source_segment_id": "s", "vehicles": [
            VehicleState(vid, x, 0, 1, "base") for vid in workflows]} for j, x in enumerate(positions)],
        vehicle_workflows=workflows, rsu_states=[RSUState("a", 0, 0, 20), RSUState("b", 10, 0, 20)],
        costs=ReplayCosts(), workload_bytes_per_unit=2,
        model_cache_config={"capacity_bytes": 20, "base_size_bytes": 8,
                            "adapter_size_bytes": 2, "transfer_bytes_per_frame": 8}, **kwargs)


def warm(env, target="a", adapter="adapter0"):
    env.model_cache.begin_frame()
    env.model_cache.request_adapter(target, adapter, "base", available_link_slots=1)
    env.model_cache.advance()
    env.model_cache.begin_frame()
    env.model_cache.advance()
    env._sync_cache_view()


def run_rule(env, split):
    reward = 0.0
    while True:
        decisions = env.decision_states()
        commands = {}
        for vid, decision in decisions.items():
            target = "b" if split and int(vid[1:]) % 2 else "a"
            commands[vid] = JointRequestControl(target, target,
                decision["semantic_state"]["current_workflow_node"]["node_id"])
        _, value, done, _ = env.step_controls(commands)
        reward += value
        if done:
            return reward, env.summary()


def test_placement_changes_completion_time_reward_and_target_trace():
    current = make_env(count=4, nodes=3)
    distributed = make_env(count=4, nodes=3)
    reward_a, summary_a = run_rule(current, False)
    reward_b, summary_b = run_rule(distributed, True)
    assert summary_a["completed_requests"] == summary_b["completed_requests"] == 4
    assert summary_b["steps"] < summary_a["steps"] and reward_b > reward_a
    assert {e["target_rsu_id"] for e in distributed.events} == {"a", "b"}
    assert summary_b["input_transfer_bytes"] == 4 * 3 * 4
    assert summary_b["output_transfer_bytes"] == 4 * 3 * 2


def test_v3_rejects_old_actions_and_illegal_joint_control_without_mutation():
    env = make_env(count=1)
    with pytest.raises(ValueError, match="five-action"):
        env.step_actions({"v0": 0})
    before = deepcopy(env.model_cache.snapshot("a"))
    with pytest.raises(ValueError, match="reachable"):
        env.step_controls({"v0": JointRequestControl("unknown")})
    with pytest.raises(ValueError, match="known DAG"):
        env.step_controls({"v0": JointRequestControl("a", "a", "unknown")})
    assert env.frame_index == 0 and env.events == [] and env.model_cache.snapshot("a") == before
    state = env.decision_states()["v0"]
    assert env.environment_revision == "v3"
    assert "action_mask" not in state and state["action_contract"] == "joint_parameterized_control_v3"
    assert state["semantic_state"]["observation_contract"] == "joint_request_observation_v3"


def test_explicit_base_and_future_adapter_admission_and_safe_eviction():
    env = make_env(count=1, nodes=2)
    env.step_controls({"v0": JointRequestControl(cache_target="b", cache_node_id="n1", cache_kind="base_model")})
    assert env.model_cache.snapshot("b")["ready_base_model_ids"] == ["base"]
    assert not env.model_cache.is_ready("b", "adapter1", "base")
    env.step_controls({"v0": JointRequestControl(cache_target="b", cache_node_id="n1")})
    assert env.model_cache.is_ready("b", "adapter1", "base")
    assert env.workflows["v0"].current_node_id == "n0"
    with pytest.raises(ValueError, match="orphan"):
        env.step_controls({"v0": JointRequestControl(cache_target="b", evict_object_ids=("base:base",))})
    result = env.step_controls({"v0": JointRequestControl(cache_target="b", evict_object_ids=("adapter:adapter1", "base:base"))})[3]
    assert result["events"][0]["frame_cost_receipt"]["explicit_evictions_by_rsu"]["b"]["evicted_bytes"] == 10
    assert env.model_cache.snapshot("b")["occupied_bytes"] == 0


def test_multiframe_input_output_and_delivered_completion_receipts():
    env = make_env(count=1, nodes=1, input_size=5, output_size=5,
                   input_bytes_per_link_frame=2, output_bytes_per_link_frame=2)
    warm(env)
    statuses, reward = [], 0.0
    while True:
        _, value, done, info = env.step_controls({"v0": JointRequestControl("a")})
        event = info["events"][0]
        statuses.append(event["status"])
        assert event["input_transfer_link_frames"] + event["output_transfer_link_frames"] <= 1
        assert sum(event["frame_cost_receipt"]["reward_components"].values()) == value
        if event["status"] == "output_ready":
            assert not env.workflows["v0"].is_completed and info["completed_nodes"] == 0
        reward += value
        if done:
            break
    assert statuses[:3] == ["input_transfer", "input_transfer", "input_ready"]
    assert statuses[-3:] == ["output_transfer", "output_transfer", "node_completed"]
    assert env.summary()["input_transfer_bytes"] == env.summary()["output_transfer_bytes"] == 5
    assert env.summary()["input_transfer_link_frames"] == env.summary()["output_transfer_link_frames"] == 3
    assert sum(sum(e["frame_cost_receipt"]["reward_components"].values()) for e in env.events) == reward


def test_prepared_state_preserves_work_and_reactive_transfer_has_real_cost():
    prepared, reactive = make_env(count=1, input_size=12), make_env(count=1, input_size=12)
    for env in (prepared, reactive):
        warm(env, "a")
        warm(env, "b")
        env.step_controls({"v0": JointRequestControl("a")})  # Input delivery.
        env.step_controls({"v0": JointRequestControl("a")})  # Partial work.
    prepared.step_controls({"v0": JointRequestControl(prepare_target="b")})
    reactive.step_controls({"v0": JointRequestControl()})
    p = prepared.step_controls({"v0": JointRequestControl("b")})[3]["events"][0]
    r = reactive.step_controls({"v0": JointRequestControl("b")})[3]["events"][0]
    assert p["status"] == "input_ready" and p["prepared_on_arrival"]
    assert r["status"] == "state_restore" and r["state_transfer_bytes"] > 0
    assert prepared.work_progress["v0"] == reactive.work_progress["v0"] == 1


def test_simultaneous_prepare_and_compute_makes_old_copy_stale():
    env = make_env(count=1, input_size=12)
    warm(env, "a")
    warm(env, "b")
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a", prepare_target="b")})
    assert not env._state_ready("v0", "b")
    result = env.step_controls({"v0": JointRequestControl("b")})[3]
    assert result["events"][0]["status"] == "state_restore"


def test_compute_priority_changes_request_progress_with_same_capacity():
    env = make_env(count=2, input_size=12)
    warm(env)
    env.step_controls({vid: JointRequestControl("a") for vid in env.workflows})
    env.step_controls({vid: JointRequestControl("a") for vid in env.workflows})
    before = dict(env.work_progress)
    result = env.step_controls({"v0": JointRequestControl("a", priority=0),
                                "v1": JointRequestControl("a", priority=3)})[3]
    assert env.work_progress["v1"] - before["v1"] == 1
    assert env.work_progress["v0"] == before["v0"]
    assert [e["status"] for e in result["events"]] == ["rsu_compute", "compute_queue"]


def test_input_model_and_state_cannot_overallocate_one_link():
    env = make_env(count=1)
    for _ in range(4):
        _, _, _, result = env.step_controls({"v0": JointRequestControl("a", "a", "n0")})
        receipt = result["events"][0]["frame_cost_receipt"]
        assert sum(receipt[field] for field in ("model_transfer_link_frames", "state_transfer_link_frames",
                                               "input_transfer_link_frames", "output_transfer_link_frames")) <= 1


def test_reset_clears_io_and_horizon_does_not_count_unreturned_outputs():
    env = make_env(count=1, nodes=1, frames=3, input_size=2)
    warm(env)
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a")})
    assert not env.workflows["v0"].is_completed
    env.reset()
    assert env.input_received == set() and env.output_pending == {} and not env.io_totals


def test_output_delivery_requires_coverage_and_local_compute_requires_adapter():
    env = make_env(count=1, nodes=1, input_size=2, positions=[0, 0, 100, 100])
    warm(env)
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a")})
    event = env.step_controls({"v0": JointRequestControl()})[3]["events"][0]
    assert event["status"] == "output_out_of_coverage" and not env.workflows["v0"].is_completed
    with pytest.raises(ValueError, match="declared"):
        env.step_controls({"v0": JointRequestControl("vehicle")})


def test_cross_base_execution_requires_both_base_and_adapter_at_target():
    env = make_env(count=1, nodes=2, bases=["base", "other"])
    _, summary = run_rule(env, False)
    assert summary["completed_requests"] == 1
    assert env.model_cache.is_ready("a", "adapter1", "other")
    assert not env.model_cache.is_ready("a", "adapter1", "base")
    assert set(env.model_cache.snapshot("a")["ready_base_model_ids"]) == {"base", "other"}


def test_v2_learning_policy_fails_closed_on_new_observation_contract():
    from src.agents.registry import build_agent
    env = make_env(count=1)
    agent = build_agent("mappo", joint_observation_enabled=True, actor_objective="masked_env_action_v1")
    with pytest.raises(ValueError, match="joint_request_observation_v2"):
        agent.act([], env.decision_states()["v0"])


def test_byte_cache_feature_includes_base_and_inflight_reservations():
    from src.agents.cache_offload_agent import _occupancy
    assert _occupancy({"capacity_bytes": 4096, "occupied_bytes": 3072,
                       "adapter_cache_capacity": 2, "cached_adapter_ids": ["a", "b"]}) == 0.75
    assert _occupancy({"capacity_bytes": 4096, "occupied_bytes": 4096,
                       "cached_adapter_ids": []}) == 1.0
    assert _occupancy({"adapter_cache_capacity": 2, "cached_adapter_ids": ["a"]}) == 0.5


def test_duplicate_joint_eviction_is_rejected_before_any_request_mutation():
    env = make_env(count=2)
    warm(env)
    before = deepcopy(env.model_cache.snapshot("a"))
    with pytest.raises(ValueError, match="duplicate"):
        env.step_controls({vid: JointRequestControl(cache_target="a", evict_object_ids=("adapter:adapter0",))
                           for vid in env.workflows})
    assert env.frame_index == 0 and env.model_cache.snapshot("a") == before


def test_cache_feature_checkpoint_keeps_legacy_units_and_records_new_units(tmp_path):
    import torch
    from src.agents.cache_offload_agent import CacheOffloadDRLAgent, _build_cache_feature_tensor
    agent = CacheOffloadDRLAgent()
    path = tmp_path / "current.pt"
    agent.save(str(path))
    payload = torch.load(path, weights_only=False)
    assert payload["config"]["cache_feature_contract"] == "typed_byte_occupancy_v2"
    restored = CacheOffloadDRLAgent()
    restored.load(str(path))
    assert restored._network.feature_contract == "typed_byte_occupancy_v2"
    payload["config"].pop("cache_feature_contract")
    old = tmp_path / "historical.pt"
    torch.save(payload, old)
    restored.load(str(old))
    assert restored._network.feature_contract == "legacy_adapter_slots_v1"
    state = {"vehicles": [{"vehicle_id": "v", "associated_rsu_id": "a"}],
        "rsus": [{"rsu_id": "a", "capacity_bytes": 4096, "occupied_bytes": 3072,
                  "adapter_cache_capacity": 2, "cached_adapter_ids": ["x", "y"]}]}
    assert _build_cache_feature_tensor(state)[0] == 0.75
    assert _build_cache_feature_tensor(state, restored._network.feature_contract)[0] == 1
    restored.save(str(path))
    assert torch.load(path, weights_only=False)["config"]["cache_feature_contract"] == "legacy_adapter_slots_v1"


def test_declared_vehicle_executes_without_edge_upload_and_switch_requires_state():
    local = make_env(count=1, nodes=1, input_size=2, vehicle_adapter_ids=["adapter0"])
    for index in range(4):
        _, _, done, _ = local.step_controls({"v0": JointRequestControl("vehicle")})
        assert done == (index == 3)
    assert local.summary()["completed_requests"] == 1
    assert local.summary()["input_transfer_bytes"] == 0
    edge = make_env(count=1, input_size=12, vehicle_adapter_ids=["adapter0"])
    warm(edge)
    edge.step_controls({"v0": JointRequestControl("a")})
    edge.step_controls({"v0": JointRequestControl("a")})
    event = edge.step_controls({"v0": JointRequestControl("vehicle")})[3]["events"][0]
    assert event["status"] == "state_restore" and event["state_transfer_link_frames"] == 1
    assert edge.work_progress["v0"] == 1


def test_pending_result_can_follow_vehicle_via_paid_versioned_edge_forwarding():
    env = make_env(count=1, nodes=1, input_size=2, positions=[0, 0, 29, 29, 29])
    warm(env)
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a")})
    assert "v0" in env.output_pending and not env.workflows["v0"].is_completed
    restored = env.step_controls({"v0": JointRequestControl("b")})[3]["events"][0]
    assert restored["status"] == "state_restore" and restored["state_transfer_bytes"] > 0
    assert restored["output_transfer_bytes"] == 0
    delivered = env.step_controls({"v0": JointRequestControl("b")})[3]["events"][0]
    assert delivered["status"] == "node_completed" and delivered["output_transfer_bytes"] == 2
    assert delivered["output_source_rsu_id"] == "b"
    assert env.workflows["v0"].is_completed and env.execution_location["v0"] == "b"


def test_waiting_control_still_records_the_actual_output_downlink_source():
    env = make_env(count=1, nodes=1, input_size=2)
    warm(env)
    env.step_controls({"v0": JointRequestControl("a")})
    env.step_controls({"v0": JointRequestControl("a")})
    event = env.step_controls({"v0": JointRequestControl()})[3]["events"][0]
    assert event["target_rsu_id"] is None and event["output_source_rsu_id"] == "a"
    assert event["status"] == "node_completed" and event["output_transfer_link_frames"] == 1
