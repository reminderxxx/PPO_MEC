"""End-to-end fixed synthetic acceptance for the new physical environment."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scripts.run_physical_env_acceptance import _frames, run_acceptance
from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.physical_vec_workflow_env import PhysicalVecWorkflowCoreEnv
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.runtime.physical_cost_profile import CostParameter, PhysicalCostProfile, ReplayIdentity


def test_fixed_physical_environment_acceptance(tmp_path) -> None:
    output_path = tmp_path / "physical_acceptance.json"
    result = run_acceptance(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["scenario_sha256"] == result["scenario_sha256"]
    assert all(written["checks"].values())
    assert written["scenario"]["parameter_evidence"] == "synthetic"
    assert written["event_ledger"]


def _replay_profile(identity: ReplayIdentity) -> PhysicalCostProfile:
    parameter = lambda seconds: CostParameter(seconds, "synthetic", "single-node integration fixture")
    return PhysicalCostProfile(
        profile_id="single-node-replay-fixture",
        replay_identity=identity,
        cold_load=parameter(1.0),
        hot_call=parameter(1.0),
        adapter_switch=parameter(0.0),
        state_restore=parameter(0.0),
    )


def _replay_env(profile: PhysicalCostProfile) -> PhysicalVecWorkflowCoreEnv:
    node = WorkflowNode("n0", "perception", "veh_base_v1", "adapter_perception", 16, 24)
    return PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(_frames()),
        workflow_state=WorkflowGraphState("replay_wf", [node], [], ["n0"], current_node_id="n0"),
        rsu_states=[RSUState("rsu-0", 0.0, 0.0, 60.0), RSUState("rsu-1", 100.0, 0.0, 60.0)],
        max_steps=20,
        control_step_s=1.0,
        backhaul_bytes_per_s=100_000_000.0,
        wireless_bytes_per_s=100_000_000.0,
        compute_slots_per_rsu=1,
        compute_rate_work_units_per_s=32.0,
        cache_capacity_bytes_per_rsu=400_000_000,
        cost_profile=profile,
        replay_input_value={"prompt": "fixture input"},
        generation_config={"temperature": 0, "max_tokens": 8},
        replay_output={"text": "fixture result"},
    )


def test_cost_replay_identity_and_output_gate_are_bound_to_environment_compute() -> None:
    identity = ReplayIdentity.from_components(
        model_identity="veh_base_v1:adapter_perception",
        input_value={"prompt": "fixture input"},
        generation_config={"temperature": 0, "max_tokens": 8},
    )
    for wrong in (
        replace(identity, model_identity="wrong-model"),
        replace(identity, input_identity="wrong-input"),
        replace(identity, generation_identity="wrong-generation"),
    ):
        with pytest.raises(ValueError, match="identity mismatch"):
            _replay_env(_replay_profile(wrong))

    env = _replay_env(_replay_profile(identity))
    wrapped = GymVecEnv(core_env=env)
    wrapped.reset(seed=17)
    with pytest.raises(ValueError, match="unavailable"):
        env.read_replay_output("n0", time_s=100, identity=identity)
    completion = None
    for _ in range(20):
        _, _, terminated, truncated, _ = wrapped.step(0)
        completed = [row for row in env.export_physical_events()
                     if row["event_type"] == "compute_completed" and row["node_id"] == "n0"]
        if completed:
            completion = completed[0]["time_s"]
            break
        with pytest.raises(ValueError, match="unavailable"):
            env.read_replay_output("n0", time_s=100, identity=identity)
        assert not terminated and not truncated
    assert completion is not None
    with pytest.raises(ValueError, match="unavailable"):
        env.read_replay_output("n0", time_s=completion - 1e-6, identity=identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        env.read_replay_output("n0", time_s=completion, identity=replace(identity, input_identity="wrong"))
    assert env.read_replay_output("n0", time_s=completion, identity=identity) == {"text": "fixture result"}
