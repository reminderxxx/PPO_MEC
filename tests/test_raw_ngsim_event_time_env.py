"""Physical admission and causal public-state checks for raw NGSIM replay."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv, RawVehicleTrace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workload():
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    manifest = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    row = deepcopy(next(item for item in manifest["instances"] if item["design_id"] == "dev_01"))
    return config, row


def trace(speed_metres_per_second: float = 1.0, n: int = 24) -> RawVehicleTrace:
    times = tuple(round(i * 0.1, 9) for i in range(n))
    return RawVehicleTrace(times, tuple((0.0, speed_metres_per_second * time) for time in times), "fixture_vehicle")


def test_admitted_service_advances_elapsed_and_keeps_byte_account(workload):
    config, row = workload
    node = next(item for item in row["nodes"] if item["node_id"] == row["execution_order"][0])
    node["compute_seconds"] = 0.2
    row["initial_residents"]["rsu_0"] = list(config["adapter_to_bundle"][node["required_adapter"]])
    env = RawNGSIMEventTimeEnv(config, row, trace())
    before = env.clock_seconds
    _, _, _, _, info = env.step(3)
    event = info["transition"]
    assert event["service_completed"] is True
    assert event["step_cost_seconds"] <= env._physical_contact_budget_seconds() + 0.2
    assert env.clock_seconds == pytest.approx(before + event["step_cost_seconds"])
    assert env.metrics["completed_nodes"] == 1
    assert env.metrics["model_transfer_bytes"] + env.metrics["state_transfer_bytes"] + env.metrics["input_transfer_bytes"] == event["model_transfer_bytes"] + event["state_transfer_bytes"] + event["input_transfer_bytes"]


def test_contact_rejection_does_not_commit_cache_or_accelerate(workload):
    config, row = workload
    node = next(item for item in row["nodes"] if item["node_id"] == row["execution_order"][0])
    node["compute_seconds"] = 0.5
    row["initial_residents"]["rsu_0"] = list(config["adapter_to_bundle"][node["required_adapter"]])
    env = RawNGSIMEventTimeEnv(config, row, trace(30.0))
    cache_before = deepcopy(env.caches)
    before = env.clock_seconds
    _, _, terminated, truncated, info = env.step(3)
    assert info["transition"]["admission_rejection_reason"] == "current_rsu_contact_expires_before_commit"
    assert not terminated
    assert env.clock_seconds > before
    assert env.clock_seconds - before <= config["objective"]["failed_service_seconds"]
    assert env.metrics["completed_nodes"] == 0
    assert env.metrics["service_failures"] == 1
    assert env.metrics["model_transfer_bytes"] == env.metrics["state_transfer_bytes"] == env.metrics["input_transfer_bytes"] == 0
    assert env.caches == cache_before
    assert truncated == (env.clock_seconds >= env.trace.times_seconds[-1])


def test_trace_end_truncates_without_terminal_success(workload):
    config, row = workload
    config["vehicle"]["fallback_seconds"] = 8.0
    env = RawNGSIMEventTimeEnv(config, row, trace())
    _, _, terminated, truncated, info = env.step(2)
    assert not terminated and not truncated
    assert info["transition"]["admission_rejection_reason"] == "trace_end_before_service_commit"
    _, _, terminated, truncated, _ = env.step(2)
    assert not terminated and truncated
    assert env.clock_seconds == pytest.approx(2.3)
    assert env.metrics["workflow_completed"] == 0


def test_rejected_cache_fill_commits_no_model_bytes(workload):
    config, row = workload
    row["initial_residents"]["rsu_0"] = []
    env = RawNGSIMEventTimeEnv(config, row, trace(30.0))
    cache_before = deepcopy(env.caches)
    _, _, _, _, info = env.step(0)
    assert info["transition"]["admission_rejection_reason"] is not None
    assert env.caches == cache_before
    assert env.metrics["completed_nodes"] == 0
    assert env.metrics["model_transfer_bytes"] == 0
    assert env.metrics["transfer_seconds"] == 0


def test_admitted_cache_fill_conserves_time_components(workload):
    config, row = workload
    row["initial_residents"]["rsu_0"] = []
    env = RawNGSIMEventTimeEnv(config, row, trace(0.0, n=1000))
    before = env.clock_seconds
    _, _, _, _, info = env.step(0)
    event = info["transition"]
    assert event["service_completed"] is True
    assert event["model_transfer_bytes"] > 0
    assert event["step_cost_seconds"] == pytest.approx(
        event["transfer_seconds"] + event["service_operation_seconds"] + event["recompute_seconds"]
    )
    assert env.clock_seconds - before == pytest.approx(event["step_cost_seconds"])
    assert env.metrics["model_transfer_bytes"] == event["model_transfer_bytes"]


def test_future_trajectory_and_end_do_not_change_public_state(workload):
    config, row = workload
    short = trace(2.0)
    future = tuple((100.0, -100.0) for _ in range(2, 30))
    altered = RawVehicleTrace(tuple(round(i * 0.1, 9) for i in range(30)), short.xy_metres[:2] + future, "fixture_vehicle")
    first = RawNGSIMEventTimeEnv(config, row, short)
    second = RawNGSIMEventTimeEnv(config, row, altered)
    public_first = first._info()
    public_second = second._info()
    assert public_first == public_second
    assert "trace_remaining_seconds" not in public_first["semantic_state"]["calibrated_context"]["time_contract"]
    assert first._physical_contact_budget_seconds() != second._physical_contact_budget_seconds()
