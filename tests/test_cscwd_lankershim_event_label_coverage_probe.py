"""Pure contracts for the bounded Lankershim label-coverage probe."""

from pathlib import Path

from scripts.probe_cscwd_lankershim_event_label_coverage import (
    _event_label_diagnostics,
    _load_protocol,
    _select_requested_action,
)
from tests.test_causal_public_action_estimator import _state


def test_probe_protocol_is_bounded_nontraining_and_result_blind() -> None:
    protocol = _load_protocol(Path(
        "configs/experiment/cscwd_lankershim_event_label_coverage_probe_v1.json"
    ).resolve())
    assert protocol["environment_step_budget"] == 96
    assert protocol["source_segment_id"] == "lankershim"
    assert protocol["selection_is_result_blind"] is True
    assert protocol["training_enabled"] is False
    assert protocol["optimizer_steps"] == 0
    assert protocol["checkpoint_reads"] == 0
    assert protocol["formal_or_holdout_reads"] == 0
    assert protocol["automatic_retry"] is False


def test_action_cycle_selection_is_deterministic_and_mask_safe() -> None:
    assert _select_requested_action(4, [True] * 5) == (4, False)
    assert _select_requested_action(4, [False, False, True, True, False]) == (
        2,
        True,
    )


def test_label_diagnostics_are_observational_and_deterministic() -> None:
    state = _state()
    before = state.copy()
    first = _event_label_diagnostics(state, [True] * 5)
    second = _event_label_diagnostics(state, [True] * 5)

    assert state == before
    assert first == second
    assert first["event_label"] == "prepare"
    assert first["event_label_reason"] == (
        "service_safe_feasible_future_readiness_gain"
    )
    assert first["event_supervision_weight"] == 1.0
    assert len(first["semantic_state_sha256"]) == 64
    assert first["action4_legal"] is True
