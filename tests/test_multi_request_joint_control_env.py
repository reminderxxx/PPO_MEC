from __future__ import annotations

import pytest

from src.envs.core.multi_request_joint_control_env import MultiRequestJointControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode


def _workflow(index: int, node_count: int = 1) -> WorkflowGraphState:
    nodes = [WorkflowNode(f"n{index}_{step}", f"n{step}", "base", "adapter", 1, 1)
             for step in range(node_count)]
    return WorkflowGraphState(f"wf{index}", nodes, [], [node.node_id for node in nodes],
                              current_node_id=nodes[0].node_id)


def _env(positions: list[float], count: int = 1, *, fetch_frames: int = 0,
         node_count: int = 1) -> MultiRequestJointControlEnv:
    ids = [f"v{index}" for index in range(count)]
    frames = [{"time_index": step, "source_segment_id": "segment",
               "vehicles": [VehicleState(vehicle_id, position, 0.0, 1.0, "base")
                            for vehicle_id in ids]}
              for step, position in enumerate(positions)]
    return MultiRequestJointControlEnv(
        frames=frames,
        vehicle_workflows={vehicle_id: _workflow(index, node_count)
                           for index, vehicle_id in enumerate(ids)},
        rsu_states=[RSUState("a", 0.0, 0.0, 100.0), RSUState("b", 10.0, 0.0, 100.0)],
        costs=ReplayCosts(compute_slots_per_rsu=1, adapter_slots_per_rsu=2,
                          link_slots_per_rsu=1, adapter_fetch_frames=fetch_frames),
    )


def test_cache_is_action_controlled_and_rsu_compute_is_shared() -> None:
    env = _env([0.0] * 12, count=8)
    _, _, _, first = env.step_actions({f"v{index}": 3 for index in range(8)})
    assert first["completed_nodes"] == 0
    assert env.summary()["cache_fetch_count"] == 0
    assert all(event["status"] == "adapter_not_ready" for event in first["events"])
    _, _, _, second = env.step_actions({f"v{index}": 0 for index in range(8)})
    assert second["completed_nodes"] == 1
    assert env.summary()["cache_fetch_count"] == 1
    assert sum(event["status"] == "compute_queue" for event in second["events"]) == 7


def test_vehicle_fallback_has_four_frame_cost_and_illegal_actions_fail() -> None:
    env = _env([0.0] * 5)
    with pytest.raises(ValueError, match="one action"):
        env.step_actions({})
    for _ in range(3):
        _, _, done, info = env.step_actions({"v0": 2})
        assert not done and info["completed_nodes"] == 0
    _, _, done, info = env.step_actions({"v0": 2})
    assert done and info["completed_nodes"] == 1
    assert env.summary()["vehicle_compute_frames"] == 4


def test_prediction_uses_past_positions_and_prepare_uses_shared_link() -> None:
    env = _env([0.0, 3.0, 4.0, 5.0], node_count=4)
    assert not env.decision_states()["v0"]["action_mask"][4]
    env.step_actions({"v0": 0})
    second = env.decision_states()["v0"]
    assert second["semantic_state"]["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["v0"] == "b"
    assert second["action_mask"][4]
    _, _, _, info = env.step_actions({"v0": 4})
    assert info["events"][0]["cache_action_status"] == "adapter_fetch"
    assert info["events"][0]["state_action_status"] == "link_queue"
    env.step_actions({"v0": 4})
    assert env.summary()["state_prepared_count"] == 1


def test_unprepared_association_handoff_is_counted() -> None:
    env = _env([0.0, 6.0, 6.0], node_count=3)
    env.step_actions({"v0": 0})
    _, _, _, info = env.step_actions({"v0": 3})
    event = info["events"][0]
    assert event["association_handoff"]
    assert event["handoff_failed"]
    assert event["status"] == "cold_restart"
    assert env.summary()["association_handoff_count"] == 1
