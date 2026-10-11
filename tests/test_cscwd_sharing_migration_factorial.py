"""Synthetic contract checks; no raw NGSIM interval or sealed result is read."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    protocol = json.loads(
        (ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text()
    )
    workload = json.loads(
        (ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text()
    )
    instance = next(row for row in workload["instances"] if row["design_id"] == "dev_01")
    times = tuple(round(index * 0.1, 9) for index in range(1200))
    trace = RawVehicleTrace(times, tuple((0.0, 7.4 * time) for time in times), "synthetic")
    return config, protocol, instance, trace


def make_env(inputs, *, sharing=True, migration=True, instance=None):
    config, protocol, default_instance, trace = inputs
    return RawNGSIMPhaseV2Env(
        config, deepcopy(instance or default_instance), trace,
        phase_contract=protocol["physical_assumptions"],
        condition=protocol["conditions"][0],
        factorial_profile={
            "base_sharing_enabled": sharing,
            "state_migration_enabled": migration,
        },
    )


def test_private_copies_preserve_task_and_charge_duplicate_bytes(inputs):
    original = deepcopy(inputs[2])
    original["cache_capacity_bytes"] = 3_000_000_000
    shared = make_env(inputs, instance=original)
    private = make_env(inputs, sharing=False, instance=original)
    assert shared.instance["nodes"] == private.instance["nodes"] == original["nodes"]
    assert shared.instance["execution_order"] == private.instance["execution_order"]
    assert shared._info()["action_mask"] == private._info()["action_mask"]
    assert private.config["measured_identity"] == shared.config["measured_identity"]
    assert private._bundle_ready("rsu_0", "helmet_shared")
    assert private._resident_bytes(private.caches["rsu_0"]) == shared._resident_bytes(shared.caches["rsu_0"])

    shared_event = shared._admit_bundle("rsu_0", "alpr")
    private_event = private._admit_bundle("rsu_0", "alpr")
    base_bytes = int(shared.config["object_catalog"]["base:family_a"]["transfer_bytes"])
    assert shared_event["committed"] and private_event["committed"]
    assert private_event["transfer_bytes"] - shared_event["transfer_bytes"] == base_bytes
    assert private._resident_bytes(private.caches["rsu_0"]) - shared._resident_bytes(shared.caches["rsu_0"]) == base_bytes
    assert private._bundle_ready("rsu_0", "helmet_shared")
    assert private._bundle_ready("rsu_0", "alpr")
    assert private._object_catalog["base:family_a::private::alpr"]["model_function_identity"] == (
        shared.config["measured_identity"]["base_model"]
    )
    assert "base:family_a" not in private._object_catalog
    assert original["initial_residents"] == inputs[2]["initial_residents"]


def test_migration_off_keeps_action_four_and_pays_recompute(inputs):
    enabled = make_env(inputs)
    disabled = make_env(inputs, migration=False)
    assert enabled._info()["action_mask"] == disabled._info()["action_mask"]
    assert enabled._info()["action_mask"][4]
    _, _, _, _, on_info = enabled.step(4)
    _, _, _, _, off_info = disabled.step(4)
    on, off = on_info["transition"], off_info["transition"]
    assert on["service_completed"] and off["service_completed"]
    assert on["migration_success"] and on["state_transfer_bytes"] > 0
    assert not off["migration_success"] and off["migration_status"] == "disabled_by_factor"
    assert off["state_transfer_bytes"] == off["state_restore_seconds"] == 0
    assert off["model_transfer_bytes"] == on["model_transfer_bytes"] > 0
    assert off["cache_events"] == on["cache_events"]
    assert off["phase_events"][:3] == on["phase_events"][:3]
    assert disabled.metrics["migration_attempts"] == 0
    assert disabled.metrics["migration_failures"] == 0
    assert disabled.prepared_state == {}
    _, _, _, _, on_next = enabled.step(0)
    _, _, _, _, off_next = disabled.step(0)
    assert on_next["transition"]["recompute_seconds"] == 0
    assert off_next["transition"]["recompute_seconds"] > 0
    assert enabled.completed == disabled.completed
    assert disabled.metrics["duplicate_execution_count"] == 0


def test_migration_off_retains_target_prepare_failure(inputs):
    disabled = make_env(inputs, migration=False)
    disabled.phase_condition["backhaul_actual_mbps"] = 0.01
    _, _, _, _, info = disabled.step(4)
    event = info["transition"]
    assert event["service_completed"]
    assert event["migration_status"] == "target_model_phase_timeout"
    assert event["cache_events"][0]["status"] == "phase_timeout"
    assert event["model_transfer_bytes"] == event["partial_transfer_bytes"] > 0
    assert event["state_transfer_bytes"] == 0
    assert disabled.metrics["migration_attempts"] == 0
    assert disabled.metrics["migration_failures"] == 0


def test_factorial_profile_rejects_unpaired_or_function_changed_inputs(inputs):
    config, protocol, instance, trace = inputs
    bad = deepcopy(instance)
    bad["initial_residents"]["rsu_0"] = ["base:family_a"]
    with pytest.raises(ValueError, match="unpaired initial resident"):
        make_env(inputs, sharing=False, instance=bad)
    bad_config = deepcopy(config)
    bad_config["measured_identity"].pop("base_model")
    with pytest.raises(ValueError, match="fixed base model function identity"):
        RawNGSIMPhaseV2Env(
            bad_config, instance, trace,
            phase_contract=protocol["physical_assumptions"],
            condition=protocol["conditions"][0],
            factorial_profile={"base_sharing_enabled": False, "state_migration_enabled": True},
        )
    with pytest.raises(ValueError, match="exactly two explicit Boolean"):
        RawNGSIMPhaseV2Env(
            config, instance, trace,
            phase_contract=protocol["physical_assumptions"],
            condition=protocol["conditions"][0],
            factorial_profile={"base_sharing_enabled": False},
        )


def test_private_initial_duplicates_do_not_bypass_capacity(inputs):
    instance = deepcopy(inputs[2])
    instance["initial_residents"]["rsu_0"].append("adapter:alpr")
    shared = make_env(inputs, instance=instance)
    assert shared._bundle_ready("rsu_0", "helmet_shared")
    assert shared._bundle_ready("rsu_0", "alpr")
    with pytest.raises(RuntimeError, match="typed cache capacity exceeded"):
        make_env(inputs, sharing=False, instance=instance)


def test_four_synthetic_arms_and_default_path(inputs):
    config, protocol, instance, trace = inputs
    default = RawNGSIMPhaseV2Env(
        config, instance, trace,
        phase_contract=protocol["physical_assumptions"],
        condition=protocol["conditions"][0],
    )
    active = make_env(inputs)
    assert default.factorial_profile is None
    assert "factorial_profile" not in default._info()["semantic_state"]["calibrated_context"]
    assert default._info()["action_mask"] == active._info()["action_mask"]
    _, _, _, _, default_info = default.step(4)
    _, _, _, _, active_info = active.step(4)
    default_event = default_info["transition"]
    active_event = dict(active_info["transition"])
    active_event.pop("factorial_profile")
    assert default_event == active_event

    common_mask = active._info()["action_mask"]
    for sharing in (False, True):
        for migration in (False, True):
            environment = make_env(inputs, sharing=sharing, migration=migration)
            assert environment._info()["action_mask"] == common_mask
            assert environment._info()["action_mask"][4]
            _, _, _, _, info = environment.step(4)
            event = info["transition"]
            assert event["factorial_profile"]["base_sharing_enabled"] is sharing
            assert event["factorial_profile"]["state_migration_enabled"] is migration
            assert event["service_completed"]
            assert event["step_cost_seconds"] == pytest.approx(
                sum(row["elapsed_seconds"] for row in event["phase_events"])
            )
            assert event["backhaul_transfer_bytes"] == (
                event["model_transfer_bytes"] + event["state_transfer_bytes"]
            )
            environment._validate_all_caches()
            if not migration:
                assert event["migration_status"] == "disabled_by_factor"
                assert event["state_transfer_bytes"] == 0
