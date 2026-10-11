"""Contracts for positive-prepare calibration-provenance abstention."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from scripts.run_cscwd_phase_v2_causal_public_pilot import (
    _load,
    _training_signal,
    _validate_protocol,
)
from src.agents.causal_public_action_estimator import (
    CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
    EVENT_LABEL_SCHEMA_VERSION,
    public_prepare_advantage_label,
)
from tests.test_calibrated_workflow_prepared_state_prefix import _fixture
from tests.test_causal_public_action_estimator import _state


def _config(*, gated: bool) -> dict:
    config, _ = _fixture()
    return {
        **config,
        "mechanism_aux_current_service_feasibility_gate_enabled": False,
        "mechanism_aux_missing_current_event_abstention_enabled": False,
        "mechanism_aux_causal_public_prepare_advantage_enabled": True,
        "mechanism_aux_positive_prepare_calibration_abstention_enabled": gated,
    }


def _agent(*, gated: bool):
    return _build_learned(
        "sa_ghmappo",
        7,
        _config(gated=gated),
        popart_enabled=False,
    ).agent


def _prepare_state(calibration_value=...):
    state = _state()
    if calibration_value is not ...:
        state["predictions"]["causal_provenance"] = {
            "source": "public_fixture",
            "actual_future_used": False,
            "confidence_calibrated": calibration_value,
        }
    return state


def _outputs() -> dict[str, torch.Tensor]:
    return {
        "slow_logits": torch.zeros(3, requires_grad=True),
        "fast_logits": torch.zeros(2, requires_grad=True),
        "event_logits": torch.zeros(2, requires_grad=True),
    }


def _gradients(agent, state: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = _outputs()
    loss = agent._compute_auxiliary_loss([state], [outputs])
    loss.backward()
    return loss.detach(), {
        name: tensor.grad.detach().clone() for name, tensor in outputs.items()
    }


@pytest.mark.parametrize("calibration_value", [..., False, "true", 1])
def test_missing_false_and_nonliteral_true_abstain(calibration_value) -> None:
    state = _prepare_state(calibration_value)
    original = public_prepare_advantage_label(state)
    gated = public_prepare_advantage_label(
        state,
        positive_prepare_calibration_abstention_enabled=True,
    )
    assert original["schema_version"] == EVENT_LABEL_SCHEMA_VERSION
    assert original["decision"] == "prepare"
    assert original["supervision_weight"] == 1.0
    assert gated["schema_version"] == (
        CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION
    )
    assert gated["decision"] == "abstain"
    assert gated["event_target"] == 0
    assert gated["event_soft_target"] == 0.0
    assert gated["supervision_weight"] == 0.0
    assert gated["reason"] == (
        "positive_prepare_prediction_confidence_not_calibrated"
    )


def test_literal_true_preserves_positive_prepare_and_serve_is_unchanged() -> None:
    prepared = public_prepare_advantage_label(
        _prepare_state(True),
        positive_prepare_calibration_abstention_enabled=True,
    )
    assert prepared["decision"] == "prepare"
    assert prepared["event_target"] == 1
    assert prepared["supervision_weight"] == 1.0

    current_missing = _state(current_ready=False)
    served = public_prepare_advantage_label(
        current_missing,
        positive_prepare_calibration_abstention_enabled=True,
    )
    assert served["decision"] == "serve"
    assert served["event_target"] == 0
    assert served["supervision_weight"] == 1.0
    assert served["reason"] == "action4_fails_current_service"


def test_gate_removes_only_event_gradient_for_uncalibrated_prepare() -> None:
    original = _agent(gated=False)
    gated = _agent(gated=True)
    state = _prepare_state(False)
    original_loss, original_gradients = _gradients(original, state)
    gated_loss, gated_gradients = _gradients(gated, state)
    assert gated_loss < original_loss
    assert torch.equal(
        original_gradients["slow_logits"], gated_gradients["slow_logits"]
    )
    assert torch.equal(
        original_gradients["fast_logits"], gated_gradients["fast_logits"]
    )
    assert torch.count_nonzero(original_gradients["event_logits"]) > 0
    assert torch.count_nonzero(gated_gradients["event_logits"]) == 0

    _, original_true_gradients = _gradients(original, _prepare_state(True))
    _, gated_true_gradients = _gradients(gated, _prepare_state(True))
    for name in original_true_gradients:
        assert torch.equal(original_true_gradients[name], gated_true_gradients[name])


def test_builder_checkpoint_identity_and_default_compatibility(tmp_path: Path) -> None:
    original = _agent(gated=False)
    gated = _agent(gated=True)
    assert (
        original._mechanism_aux_positive_prepare_calibration_abstention_enabled
        is False
    )
    assert gated._mechanism_aux_positive_prepare_calibration_abstention_enabled is True
    assert original._checkpoint_config()["mechanism_aux_event_target_semantics"] == (
        EVENT_LABEL_SCHEMA_VERSION
    )
    assert gated._checkpoint_config()["mechanism_aux_event_target_semantics"] == (
        CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION
    )

    original_path = tmp_path / "original.pt"
    gated_path = tmp_path / "gated.pt"
    original.save(str(original_path))
    gated.save(str(gated_path))
    original.load(str(original_path))
    gated.load(str(gated_path))
    with pytest.raises(ValueError, match="calibration-abstention"):
        original.load(str(gated_path))
    with pytest.raises(ValueError, match="calibration-abstention"):
        gated.load(str(original_path))

    invalid = deepcopy(_config(gated=True))
    invalid["mechanism_aux_causal_public_prepare_advantage_enabled"] = False
    with pytest.raises(ValueError, match="requires causal public prepare advantage"):
        _build_learned("sa_ghmappo", 7, invalid, popart_enabled=False)


def test_training_signal_log_uses_the_cell_semantics() -> None:
    state = _prepare_state(False)
    row = {
        "env_info": {"transition": {}},
        "action_info": {"env_action_probs": [0.2] * 5},
        "decision_info": {"semantic_state": state, "action_mask": [1] * 5},
        "training_episode_index": 1,
        "rollout_segment_index": 0,
        "action": 4,
        "reward": 0.0,
        "value": 0.0,
        "advantage": 0.0,
        "return": 0.0,
        "log_prob": 0.0,
        "bootstrap_value": 0.0,
        "segment_terminated": False,
        "segment_truncated": False,
        "batch_boundary": False,
    }
    original = _training_signal(
        row, arm="sa_public_v3", method="sa_ghmappo", seed=7,
        update_index=1, global_step=1,
    )
    gated = _training_signal(
        row, arm="sa_calibration_abstention", method="sa_ghmappo", seed=7,
        update_index=1, global_step=1,
        positive_prepare_calibration_abstention_enabled=True,
    )
    assert original["event_label"] == "prepare"
    assert original["event_supervision_weight"] == 1.0
    assert gated["event_label"] == "abstain"
    assert gated["event_supervision_weight"] == 0.0
    assert gated["positive_prepare_calibration_abstention_enabled"] is True


def test_completed_v1_protocol_is_rejected_after_v4_semantics_fix() -> None:
    path = Path(
        "configs/experiment/"
        "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_v1.json"
    )
    protocol = _load(path)
    with pytest.raises(RuntimeError, match="single-variable contract drift"):
        _validate_protocol(
            protocol, require_clean=False, require_authorized=False
        )
    assert protocol["authorization_scope"] == (
        "one_create_only_development_run_no_retry"
    )
