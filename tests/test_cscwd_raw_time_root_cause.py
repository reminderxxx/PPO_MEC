"""Independent witnesses for the frozen raw-time diagnosis; no source data."""

import json
from pathlib import Path

import pytest

from scripts.diagnose_cscwd_raw_time_root_cause import (
    future_clone_probe,
    gate,
    native_phase,
    prospective_prepare,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv, RawVehicleTrace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dev_workload():
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    manifest = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    row = next(item for item in manifest["instances"] if item["design_id"] == "dev_01")
    return config, row


def _trace(speed: float = 12.0) -> RawVehicleTrace:
    times = tuple(round(i / 10, 9) for i in range(24))
    return RawVehicleTrace(times, tuple((0.0, speed * t) for t in times), "synthetic")


def test_prepare_phase_can_fit_while_full_step_gate_rejects(dev_workload):
    config, row = dev_workload
    env = RawNGSIMEventTimeEnv(config, row, _trace())
    contact = env._physical_contact_budget_seconds()
    assert contact == pytest.approx((10.0 - 1.2) / 12.0)
    for action in (1, 4):
        assert action in env.valid_actions()
        prepare = prospective_prepare(env, action)
        native = native_phase(env, action, estimated=False)
        admission = gate(env, action, native["total_seconds"])
        assert prepare["native_prepare_contact_test_seconds"] < contact
        assert native["compute_seconds"] > 0
        assert admission["contact_exceeded"] is True


def test_decision_clone_uses_private_future_contact(dev_workload):
    config, row = dev_workload
    witness = future_clone_probe(config, row, _trace())
    assert witness["public_state_equal"] is True
    assert witness["slow_preview_completed"] is True
    assert witness["fast_preview_completed"] is False
    assert witness["fast_preview_rejection"] == "current_rsu_contact_expires_before_commit"


def test_network_rate_is_decimal_megabits_per_second():
    from src.envs.core.calibrated_continuous_workflow_env import _network_seconds

    assert _network_seconds(1_000_000, 1_000.0, 0.02) == pytest.approx(0.028)
    assert _network_seconds(0, 1_000.0, 0.02) == 0.0
