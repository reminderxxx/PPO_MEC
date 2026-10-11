"""Read-only witness for the phase-v2 public-observation identifiability limit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from src.agents.causal_public_action_estimator import (
    CausalPublicImmediateRule,
    CausalPublicTwoStepRule,
    estimate_public_actions,
    public_prepare_advantage_label,
)
from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env


ROOT = Path(__file__).resolve().parents[1]


def _trace(*, suffix: str) -> RawVehicleTrace:
    speed = 7.4
    times = tuple(round(index * 0.1, 9) for index in range(1200))
    positions = [(0.0, speed * time) for time in times]
    if suffix == "reverse":
        positions[2:] = [(0.0, -speed * time) for time in times[2:]]
    return RawVehicleTrace(times, tuple(positions), f"fixture_{suffix}")


def _env(*, suffix: str) -> RawNGSIMPhaseV2Env:
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    contract = json.loads(
        (ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text()
    )
    workload = json.loads(
        (
            ROOT
            / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
        ).read_text()
    )
    instance = next(row for row in workload["instances"] if row["design_id"] == "dev_01")
    return RawNGSIMPhaseV2Env(
        config,
        deepcopy(instance),
        _trace(suffix=suffix),
        phase_contract=contract["physical_assumptions"],
        condition=contract["conditions"][0],
    )


def _local_outcome_key(event: dict) -> tuple[int, int, float, int]:
    """Declared diagnostic ordering, not a full-horizon policy objective."""

    total_bytes = sum(
        int(event[name])
        for name in (
            "model_transfer_bytes",
            "state_transfer_bytes",
            "input_transfer_bytes",
        )
    )
    return (
        int(event["service_completed"]),
        int(event["migration_success"]),
        -float(event["step_cost_seconds"]),
        -total_bytes,
    )


def test_same_public_prefix_can_require_opposite_local_actions() -> None:
    forward_action3 = _env(suffix="forward")
    forward_action4 = _env(suffix="forward")
    reverse_action3 = _env(suffix="reverse")
    reverse_action4 = _env(suffix="reverse")

    infos = [
        environment._info()
        for environment in (
            forward_action3,
            forward_action4,
            reverse_action3,
            reverse_action4,
        )
    ]
    assert all(info == infos[0] for info in infos[1:])
    public_info = infos[0]
    state = public_info["semantic_state"]
    assert state["predictions"]["causal_provenance"]["confidence_calibrated"] is False
    assert state["predictions"]["causal_provenance"]["actual_future_used"] is False
    assert not {
        "actual_future_route",
        "actual_target_entry_seconds",
        "actual_contact_budget_seconds",
    }.intersection(state)

    estimates = [
        estimate_public_actions(info["semantic_state"], info["action_mask"])
        for info in infos
    ]
    labels = [
        public_prepare_advantage_label(info["semantic_state"], info["action_mask"])
        for info in infos
    ]
    assert all(estimate == estimates[0] for estimate in estimates[1:])
    assert all(label == labels[0] for label in labels[1:])
    assert labels[0]["decision"] == "prepare"
    assert labels[0]["supervision_weight"] == 1.0
    assert estimates[0]["actions"]["4"]["predicted_target_delivery"] == "yes"
    assert estimates[0]["actions"]["4"]["raw_trace_fit"] == "unknown"

    immediate_actions = [
        CausalPublicImmediateRule().select_action_from_info(info) for info in infos
    ]
    two_step_actions = [
        CausalPublicTwoStepRule().select_action_from_info(info) for info in infos
    ]
    assert len(set(immediate_actions)) == len(set(two_step_actions)) == 1

    forward_event3 = forward_action3.step(3)[4]["transition"]
    forward_event4 = forward_action4.step(4)[4]["transition"]
    reverse_event3 = reverse_action3.step(3)[4]["transition"]
    reverse_event4 = reverse_action4.step(4)[4]["transition"]

    assert all(
        event["service_completed"]
        for event in (forward_event3, forward_event4, reverse_event3, reverse_event4)
    )
    assert forward_event4["migration_success"] is True
    assert reverse_event4["migration_success"] is False
    assert reverse_event4["migration_status"] in {
        "target_delivery_timeout",
        "target_departed_before_delivery",
    }
    assert _local_outcome_key(forward_event4) > _local_outcome_key(forward_event3)
    assert _local_outcome_key(reverse_event3) > _local_outcome_key(reverse_event4)
