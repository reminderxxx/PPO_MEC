"""Actual reset/step checks for the opt-in serial raw phase contract."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env


ROOT = Path(__file__).resolve().parents[1]
RAW_V1_SHA = "30666d7c874717d8464a64702c811116e96b931a5db6b57287c6c11d2dcc6575"


@pytest.fixture
def setup():
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    contract = json.loads((ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text())
    workload = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instances = {row["design_id"]: row for row in workload["instances"]}
    return config, contract, instances


def trace(speed: float = 7.4, count: int = 1200, suffix: str = "forward") -> RawVehicleTrace:
    times = tuple(round(i * 0.1, 9) for i in range(count))
    positions = [(0.0, speed * time) for time in times]
    if suffix == "reverse":
        positions[2:] = [(0.0, -speed * time) for time in times[2:]]
    return RawVehicleTrace(times, tuple(positions), "fixture_vehicle")


def env(setup, design: str = "dev_01", *, vehicle_trace=None, condition=0) -> RawNGSIMPhaseV2Env:
    config, contract, instances = setup
    return RawNGSIMPhaseV2Env(
        config, deepcopy(instances[design]), vehicle_trace or trace(),
        phase_contract=contract["physical_assumptions"],
        condition=contract["conditions"][condition],
    )


def phases(event):
    return {row["phase"]: row for row in event["phase_events"]}


def test_legacy_file_and_public_suffix_contract(setup):
    assert sha256((ROOT / "src/envs/core/raw_ngsim_event_time_env.py").read_bytes()).hexdigest() == RAW_V1_SHA
    first = env(setup, vehicle_trace=trace(suffix="forward"))
    second = env(setup, vehicle_trace=trace(suffix="reverse"))
    assert first._info() == second._info()
    state = first._info()["semantic_state"]
    assert state["time_profile"] == "raw_ngsim_event_time_v2"
    assert state["calibrated_context"]["phase_contract"]["backhaul"]["estimated_mbps"] == 1000.0
    assert "actual_contact_budget_seconds" not in json.dumps(first._info())
    with pytest.raises(RuntimeError, match="hidden-future"):
        first.clone_for_decision_model()


def test_input_upload_partial_blocks_compute_and_accounts_bytes(setup):
    instance = env(setup)
    instance.instance["link_profile"]["actual_mbps"] = 0.2
    cache_before = deepcopy(instance.caches)
    _, _, terminated, truncated, info = instance.step(4)
    event = info["transition"]
    assert not terminated and not truncated
    assert not event["service_completed"]
    assert event["failure_reason"] == "current_contact_expired"
    assert 0 <= event["input_transfer_bytes"] < instance.nodes[0]["input_bytes"]
    assert event["input_transfer_bytes"] == event["partial_transfer_bytes"]
    assert event["model_transfer_bytes"] == event["state_transfer_bytes"] == 0
    assert instance.caches == cache_before
    assert instance.execution_owner_rsu_id is None
    assert instance.completed == []
    assert list(phases(event)) == ["input_upload"]


def test_compute_may_cross_source_contact_then_deliver_at_target(setup):
    instance = env(setup)
    start = instance.clock_seconds
    contact = instance._physical_contact_budget_seconds()
    _, _, terminated, truncated, info = instance.step(4)
    event = info["transition"]
    p = phases(event)
    assert not terminated and not truncated
    assert event["service_completed"] and event["migration_success"]
    assert event["migration_status"] == "delivered"
    assert p["input_upload"]["end_seconds"] <= start + contact + 1e-9
    assert p["current_node_compute"]["end_seconds"] > start + contact
    assert p["state_transfer"]["start_seconds"] >= p["current_node_compute"]["end_seconds"]
    assert p["target_restore"]["start_seconds"] >= p["state_transfer"]["end_seconds"]
    assert event["execution_owner_rsu_id"] == "rsu_0"
    assert event["state_transfer_bytes"] > 0
    assert instance.prepared_state[event["predicted_target_rsu_id"]]["completed_node_ids"] == [event["node_id"]]
    assert event["backhaul_transfer_bytes"] == event["model_transfer_bytes"] + event["state_transfer_bytes"]
    assert event["step_cost_seconds"] == pytest.approx(sum(row["elapsed_seconds"] for row in event["phase_events"]))
    assert instance.metrics["completed_nodes"] == 1


def test_state_backhaul_timeout_retains_node_and_partial_cost(setup):
    instance = env(setup, "regression_04")
    node = instance._current_node()
    instance.caches["rsu_0"].residents = instance._bundle_ids(str(node["required_adapter"]))
    _, _, _, _, info = instance.step(4)
    event = info["transition"]
    assert event["service_completed"]
    assert not event["migration_success"]
    assert event["migration_status"] == "state_transfer_phase_timeout"
    assert event["state_transfer_bytes"] > 0
    assert event["partial_transfer_bytes"] >= event["state_transfer_bytes"]
    assert instance.completed == [event["node_id"]]
    assert not instance.prepared_state
    assert phases(event)["state_transfer"]["status"] == "phase_timeout"


def test_target_prediction_error_does_not_create_state_or_repeat_node(setup):
    forward = env(setup, vehicle_trace=trace(suffix="forward"))
    reverse = env(setup, vehicle_trace=trace(suffix="reverse"))
    assert forward._info() == reverse._info()
    _, _, _, _, forward_info = forward.step(4)
    _, _, _, _, reverse_info = reverse.step(4)
    good = forward_info["transition"]
    bad = reverse_info["transition"]
    assert good["migration_success"]
    assert bad["service_completed"] and not bad["migration_success"]
    assert bad["migration_status"] in {"target_delivery_timeout", "target_departed_before_delivery"}
    assert reverse.completed == [bad["node_id"]]
    assert not reverse.prepared_state
    assert reverse.metrics["duplicate_execution_count"] == 0


def test_target_departed_before_state_delivery_is_real_failure(setup):
    instance = env(setup, vehicle_trace=trace(speed=20.0))
    assert instance._info()["action_mask"][4]
    _, _, _, _, info = instance.step(4)
    event = info["transition"]
    assert event["service_completed"] and not event["migration_success"]
    assert event["migration_status"] == "target_departed_before_delivery"
    assert event["state_transfer_bytes"] > 0
    assert not instance.prepared_state


def test_delivery_pending_at_trace_end_truncates_without_success(setup):
    instance = env(setup, vehicle_trace=trace(speed=1.0, count=30))
    assert instance._info()["action_mask"][4]
    _, _, terminated, truncated, info = instance.step(4)
    event = info["transition"]
    assert not terminated and truncated
    assert event["service_completed"] and not event["migration_success"]
    assert event["migration_status"] == "trace_end_pending_delivery"
    assert instance.pending_delivery is not None
    assert instance.metrics["pending_state_count"] == 1
    assert instance.clock_seconds == pytest.approx(instance.trace.times_seconds[-1])


def test_partial_model_transfer_has_cost_without_cache_commit(setup):
    instance = env(setup)
    instance.phase_condition["backhaul_actual_mbps"] = 1.0
    target = instance._predicted_handoff_target()
    assert target is not None
    before = deepcopy(instance.caches[target])
    _, _, _, _, info = instance.step(4)
    event = info["transition"]
    assert phases(event)["model_prepare_network"]["status"] == "phase_timeout"
    assert event["model_transfer_bytes"] > 0
    assert event["partial_transfer_bytes"] >= event["model_transfer_bytes"]
    assert instance.caches[target] == before
    assert not event["migration_success"]


def test_trace_end_truncates_incomplete_compute_without_state(setup):
    instance = env(setup, vehicle_trace=trace(count=10))
    _, _, terminated, truncated, info = instance.step(4)
    event = info["transition"]
    assert not terminated and truncated
    assert not event["service_completed"] and not event["migration_success"]
    assert event["failure_reason"] == "trace_end_during_compute"
    assert not instance.completed and not instance.prepared_state
    assert instance.clock_seconds == pytest.approx(instance.trace.times_seconds[-1])
    assert sum(row["elapsed_seconds"] for row in event["phase_events"]) == pytest.approx(event["step_cost_seconds"])


def test_fallback_and_ample_cache_keep_common_five_action_mask(setup):
    instance = env(setup, condition=2)
    assert len(instance._info()["action_mask"]) == 5
    node = instance._current_node()
    for cache in instance.caches.values():
        assert set(instance._bundle_ids(str(node["required_adapter"]))).issubset(set(cache.residents))
    _, _, _, _, info = instance.step(2)
    assert info["transition"]["action"] == 2
    assert info["transition"]["migration_status"] == "not_requested"
    assert instance.metrics["completed_nodes"] == 1
