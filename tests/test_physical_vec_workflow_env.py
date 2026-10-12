"""End-to-end checks for the opt-in physical workflow environment."""

from __future__ import annotations

import pytest

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.physical_vec_workflow_env import PhysicalVecWorkflowCoreEnv
from src.envs.specs import ControlAction, RSUState, WorkflowGraphState, WorkflowNode
from src.metrics.physical_event_metrics import reduce_physical_events
from src.runtime.physical_cost_profile import CostParameter, PhysicalCostProfile, ReplayIdentity


def _graph(adapter: str = "adapter_perception", size: int = 4) -> WorkflowGraphState:
    node = WorkflowNode("n1", "task", "veh_base_v1", adapter, size, size)
    return WorkflowGraphState("physical_test", [node], [], ["n1"], current_node_id="n1")


def _frames(handoff_at: int | None = None) -> list[dict]:
    return [
        {
            "time_index": step,
            "vehicles": [{
                "vehicle_id": "v1", "position_x": 100.0 if handoff_at is not None and step >= handoff_at else 0.0,
                "position_y": 0.0, "speed": 10.0, "base_model_id": "veh_base_v1",
                "active_workflow_id": "physical_test",
            }],
        }
        for step in range(25)
    ]


def _control(action_id: int) -> ControlAction:
    return ControlAction(metadata={"action_id": action_id})


def test_cache_load_waits_across_steps_then_output_and_delivery_complete() -> None:
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames()),
        workflow_state=_graph("adapter_fusion"),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0)],
        max_steps=20, backhaul_bytes_per_s=50_000_000.0,
        wireless_bytes_per_s=1_000_000_000.0,
        compute_rate_work_units_per_s=100.0,
        cache_capacity_bytes_per_rsu=400_000_000.0,
    )
    state, _ = env.reset()
    assert state["physical_state"]["nodes"][0]["status"] in {"waiting_input", "waiting_predecessor"}
    state, _, terminated, _, first = env.step(_control(0))
    assert not terminated
    assert state["physical_state"]["nodes"][0]["status"] == "waiting_model"
    assert any(item["transfer_kind"] == "model_load" for item in first["physical_events"]
               if item["event_type"] == "transfer_started")
    for _ in range(19):
        state, _, terminated, truncated, info = env.step(_control(3))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert state["physical_state"]["nodes"][0]["status"] == "completed"
    assert info["workflow_completion_time_s"] > 1.0
    assert reduce_physical_events(env.export_physical_events()).availability == "available"
    assert info["physical_event_trace"] == env.export_physical_events()


def test_prediction_exposes_uncalibrated_distribution_eta_and_confidence() -> None:
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames(handoff_at=3)),
        workflow_state=_graph(),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0), RSUState("rsu_b", 100.0, 0.0, 30.0)],
    )
    state, _ = env.reset()
    predictions = state["predictions"]
    assert predictions["prediction_probability_semantics"] == "uncalibrated_horizon_vote_share"
    assert predictions["prediction_evidence"] == "current_and_past_frames_only"
    assert 0.0 <= predictions["prediction_confidence_by_vehicle"]["v1"] <= 1.0
    assert 0.0 <= predictions["prediction_uncertainty_by_vehicle"]["v1"] <= 1.0
    probabilities = predictions["next_rsu_candidate_probability_by_vehicle"]["v1"]
    if probabilities:
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    eta = predictions["predicted_handoff_eta_s_by_vehicle"]["v1"]
    assert eta is None or eta > 0.0


def test_running_old_rsu_continues_through_handoff_and_forwards_result() -> None:
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames(handoff_at=3)),
        workflow_state=_graph(size=4),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0), RSUState("rsu_b", 100.0, 0.0, 30.0)],
        max_steps=20, backhaul_bytes_per_s=1_000_000.0,
        wireless_bytes_per_s=1_000_000_000.0,
        compute_rate_work_units_per_s=1.0,
        cache_capacity_bytes_per_rsu=400_000_000.0,
    )
    env.reset()
    saw_handoff_running = False
    for _ in range(20):
        state, _, terminated, truncated, info = env.step(_control(3))
        node = state["physical_state"]["nodes"][0]
        if state["vehicles"][0]["associated_rsu_id"] == "rsu_b" and node["status"] == "running":
            saw_handoff_running = True
            assert node["deployment_rsu_id"] == "rsu_a"
        if terminated or truncated:
            break
    assert saw_handoff_running
    assert terminated and not truncated
    receipts = state["physical_state"]["handoff_receipts"]
    assert any(item["kind"] == "forward_result" and item["source_rsu_id"] == "rsu_a"
               and item["target_rsu_id"] == "rsu_b" and item["size_bytes"] == 4
               for item in receipts)
    assert info["physical_event_summary"]["availability"] == "available"
    assert all(item["time_s"] <= info["workflow_completion_time_s"] for item in env.export_physical_events())


def test_exact_identity_replay_output_opens_only_after_compute_completion() -> None:
    input_value = {"sensor": [1, 2]}
    generation = {"temperature": 0}
    identity = ReplayIdentity.from_components(
        model_identity="veh_base_v1:adapter_perception",
        input_value=input_value, generation_config=generation,
    )
    parameter = CostParameter(0.2, "synthetic", "fixed-test")
    profile = PhysicalCostProfile(
        "fixed-test", identity, parameter, parameter, parameter, parameter,
    )
    kwargs = {
        "mobility_provider": ReplayProvider(_frames()),
        "workflow_state": _graph(),
        "rsu_states": [RSUState("rsu_a", 0.0, 0.0, 30.0)],
        "cost_profile": profile,
        "replay_input_value": input_value,
        "generation_config": generation,
        "replay_output": {"result": "fixed"},
    }
    env = PhysicalVecWorkflowCoreEnv(**kwargs)
    with pytest.raises(ValueError, match="unavailable before simulated completion"):
        env.read_replay_output("n1", 0.0, identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        env.read_replay_output(
            "n1", 0.0,
            ReplayIdentity.from_components(
                model_identity="wrong-model", input_value=input_value,
                generation_config=generation,
            ),
        )
    _, _, terminated, _, info = env.step(_control(3))
    assert terminated
    assert env.read_replay_output("n1", info["workflow_completion_time_s"], identity) == {"result": "fixed"}
    with pytest.raises(ValueError, match="identity mismatch"):
        PhysicalVecWorkflowCoreEnv(**{**kwargs, "generation_config": {"temperature": 1}})


def test_handoff_during_wireless_delivery_fails_instead_of_claiming_success() -> None:
    node = WorkflowNode("n1", "task", "veh_base_v1", "adapter_perception", 1, 4)
    graph = WorkflowGraphState("delivery_handoff", [node], [], ["n1"], current_node_id="n1")
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames(handoff_at=2)),
        workflow_state=graph,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0), RSUState("rsu_b", 100.0, 0.0, 30.0)],
        max_steps=10, wireless_bytes_per_s=1.0,
        compute_rate_work_units_per_s=100.0,
        cache_capacity_bytes_per_rsu=400_000_000.0,
    )
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(_control(3))
        if terminated or truncated:
            break
    assert not terminated and truncated
    assert not info["service_success"]
    assert info["workflow_completion_time_s"] is None
    assert any(item["event_type"] == "delivery_failed" for item in env.export_physical_events())
    assert reduce_physical_events(env.export_physical_events()).availability == "unavailable"


def test_two_sink_outputs_are_both_delivered_before_request_completion() -> None:
    nodes = [
        WorkflowNode("n1", "first", "veh_base_v1", "adapter_perception", 2, 3),
        WorkflowNode("n2", "second", "veh_base_v1", "adapter_tracking", 2, 5),
    ]
    graph = WorkflowGraphState("two_sinks", nodes, [], ["n1", "n2"], current_node_id="n1")
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames()), workflow_state=graph,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0)], max_steps=12,
        wireless_bytes_per_s=100.0, compute_rate_work_units_per_s=100.0,
        cache_capacity_bytes_per_rsu=400_000_000.0,
    )
    env.reset()
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(_control(3))
        if terminated or truncated:
            break
    assert terminated and not truncated
    trace = env.export_physical_events()
    assert len([item for item in trace if item["event_type"] == "delivery_completed"]) == 2
    assert len([item for item in trace if item["event_type"] == "request_completed"]) == 1
    assert info["physical_event_summary"]["availability"] == "available"


def test_prefetch_remaining_in_flight_does_not_invalidate_workflow_terminal() -> None:
    frames = _frames()
    for frame in frames:
        frame["vehicles"][0]["speed"] = 100.0
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(frames), workflow_state=_graph(),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 45.0), RSUState("rsu_b", 65.0, 0.0, 45.0)],
        backhaul_bytes_per_s=100.0, compute_rate_work_units_per_s=1000.0,
        cache_capacity_bytes_per_rsu=400_000_000.0,
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(_control(1))
    assert terminated and not truncated
    assert info["workflow_physical_metrics"]["availability"] == "available"
    assert info["physical_event_summary"]["availability"] == "unavailable"
    assert any(item["request_id"].startswith("physical_test:v1:prepare:")
               for item in info["physical_event_summary"]["requests"])


def test_millisecond_frames_advance_within_one_control_step_and_stop_at_trace_end() -> None:
    frames = [
        {"time_index": 1000 + offset, "vehicles": [{
            "vehicle_id": "v1", "position_x": 0.0, "position_y": 0.0,
            "speed": 10.0, "base_model_id": "veh_base_v1",
            "active_workflow_id": "physical_test",
        }]}
        for offset in (0, 100, 200, 300)
    ]
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(frames, time_unit="ms"),
        workflow_state=_graph(size=1000),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0)],
        control_step_s=0.25, max_steps=20,
        compute_rate_work_units_per_s=1.0,
    )
    env.reset()
    state, _, terminated, truncated, _ = env.step(_control(3))
    assert not terminated and not truncated
    assert state["time_index"] == 1200
    assert state["physical_state"]["time_s"] == pytest.approx(0.25)
    state, _, terminated, truncated, info = env.step(_control(3))
    assert not terminated and truncated
    assert state["time_index"] == 1300
    assert state["physical_state"]["time_s"] == pytest.approx(0.3)
    assert info["failure_reason"] == "mobility_trace_exhausted"
    assert len([event for event in env.export_physical_events()
                if event["event_type"] == "request_failed"]) == 1


def test_primary_departure_fails_without_reassigning_workflow_to_another_vehicle() -> None:
    frames = [
        {"time_index": time, "vehicles": [
            *([{
                "vehicle_id": "v1", "position_x": 0.0, "position_y": 0.0,
                "speed": 10.0, "base_model_id": "veh_base_v1",
                "active_workflow_id": "physical_test",
            }] if time < 1200 else []),
            {"vehicle_id": "v2", "position_x": 1.0, "position_y": 0.0,
             "speed": 10.0, "base_model_id": "veh_base_v1",
             "active_workflow_id": "physical_test"},
        ]}
        for time in (1000, 1100, 1200, 1300)
    ]
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(frames, time_unit="ms"),
        workflow_state=_graph(size=1000),
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 30.0)],
        max_steps=20, compute_rate_work_units_per_s=1.0,
    )
    env.reset()
    state, _, terminated, truncated, info = env.step(_control(3))
    assert not terminated and truncated
    assert state["primary_vehicle_id"] == "v1"
    assert state["time_index"] == 1200
    assert state["physical_state"]["time_s"] == pytest.approx(0.2)
    assert info["failure_reason"] == "primary_vehicle_departed"
    assert not info["service_success"]
    assert info["workflow_physical_metrics"]["availability"] == "unavailable"
