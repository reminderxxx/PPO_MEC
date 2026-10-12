from __future__ import annotations

import pytest

from src.envs.core.multi_request_replay_env import (
    MultiRequestReplayEnv,
    ReplayCosts,
    select_continuous_window,
)
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode


def _vehicle(vehicle_id: str) -> VehicleState:
    return VehicleState(vehicle_id, 0.0, 0.0, 1.0, "base")


def _workflow(index: int) -> WorkflowGraphState:
    node = WorkflowNode(f"n{index}", f"n{index}", "base", "adapter", 1, 1)
    return WorkflowGraphState(
        workflow_id=f"workflow_{index}",
        nodes=[node],
        edges=[],
        execution_order=[node.node_id],
        current_node_id=node.node_id,
    )


def test_eight_requests_share_one_compute_slot_and_each_finishes_once() -> None:
    ids = [f"v{index}" for index in range(8)]
    frames = [
        {"time_index": step, "source_segment_id": "segment", "vehicles": [_vehicle(item) for item in ids]}
        for step in range(10)
    ]
    env = MultiRequestReplayEnv(
        frames=frames,
        vehicle_workflows={item: _workflow(index) for index, item in enumerate(ids)},
        rsu_states=[RSUState("rsu", 0.0, 0.0, 10.0)],
        costs=ReplayCosts(adapter_fetch_frames=0),
    )
    env.reset()
    while True:
        _, _, done, info = env.step()
        assert info["completed_nodes"] <= 1
        if done:
            break
    assert env.summary()["completed_requests"] == 8
    assert env.summary()["completed_nodes"] == 8
    assert len({event["vehicle_id"] for event in env.events}) == 8


def test_window_requires_all_vehicles_to_persist_in_one_segment() -> None:
    frames = [
        {"time_index": step, "source_segment_id": "one", "vehicles": [_vehicle(f"v{i}") for i in range(8)]}
        for step in range(3)
    ]
    selected, ids, offset = select_continuous_window(frames, vehicle_count=8, window_length=3)
    assert offset == 0
    assert len(ids) == 8
    assert len(selected) == 3
    frames[2]["source_segment_id"] = "two"
    with pytest.raises(RuntimeError, match="persistent vehicles"):
        select_continuous_window(frames, vehicle_count=8, window_length=3)


def test_window_excludes_duplicate_vehicle_records() -> None:
    frames = [
        {"time_index": step, "source_segment_id": "one", "vehicles": [_vehicle(f"v{i}") for i in range(8)]}
        for step in range(2)
    ]
    frames[1]["vehicles"].append(_vehicle("v0"))
    with pytest.raises(RuntimeError, match="persistent vehicles"):
        select_continuous_window(frames, vehicle_count=8, window_length=2)


def test_uncovered_target_is_rejected() -> None:
    frames = [
        {"time_index": step, "source_segment_id": "one", "vehicles": [_vehicle("v0")]}
        for step in range(2)
    ]
    env = MultiRequestReplayEnv(
        frames=frames,
        vehicle_workflows={"v0": _workflow(0)},
        rsu_states=[RSUState("rsu", 0.0, 0.0, 10.0)],
    )
    with pytest.raises(ValueError, match="uncovered RSU"):
        env.step({"v0": "unknown"})


def test_action_can_select_an_alternate_covered_rsu() -> None:
    frames = [
        {"time_index": step, "source_segment_id": "one", "vehicles": [_vehicle("v0")]}
        for step in range(2)
    ]
    env = MultiRequestReplayEnv(
        frames=frames,
        vehicle_workflows={"v0": _workflow(0)},
        rsu_states=[
            RSUState("rsu_a", 0.0, 0.0, 10.0),
            RSUState("rsu_b", 1.0, 0.0, 10.0),
        ],
        costs=ReplayCosts(adapter_fetch_frames=0),
    )
    _, _, done, info = env.step({"v0": "rsu_b"})
    assert done
    assert info["events"][0]["associated_rsu_id"] == "rsu_a"
    assert info["events"][0]["target_rsu_id"] == "rsu_b"
    assert env.summary()["completed_requests"] == 1


def test_adapter_transfers_share_link_across_frames() -> None:
    frames = [
        {"time_index": step, "source_segment_id": "one", "vehicles": [_vehicle("v0"), _vehicle("v1")]}
        for step in range(4)
    ]
    first = _workflow(0)
    second = _workflow(1)
    second.nodes[0].required_adapter = "other_adapter"
    env = MultiRequestReplayEnv(
        frames=frames,
        vehicle_workflows={"v0": first, "v1": second},
        rsu_states=[RSUState("rsu", 0.0, 0.0, 10.0)],
        costs=ReplayCosts(adapter_fetch_frames=2, adapter_slots_per_rsu=2),
    )
    _, _, _, first_frame = env.step()
    assert {event["status"] for event in first_frame["events"]} == {"adapter_fetch", "link_queue"}
    _, _, _, second_frame = env.step()
    assert not any(event["status"] == "adapter_fetch" for event in second_frame["events"])
