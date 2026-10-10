"""Frozen contracts for the causal public prepare-advantage auxiliary label."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from src.agents.causal_public_action_estimator import EVENT_LABEL_SCHEMA_VERSION
from tests.test_calibrated_workflow_prepared_state_prefix import _fixture
from tests.test_causal_public_action_estimator import _state


def _agents():
    config, _ = _fixture()
    legacy_config = {
        **config,
        "mechanism_aux_current_service_feasibility_gate_enabled": False,
        "mechanism_aux_missing_current_event_abstention_enabled": False,
        "mechanism_aux_causal_public_prepare_advantage_enabled": False,
    }
    candidate_config = {
        **legacy_config,
        "mechanism_aux_causal_public_prepare_advantage_enabled": True,
    }
    return (
        _build_learned("sa_ghmappo", 7, legacy_config, popart_enabled=False),
        _build_learned("sa_ghmappo", 7, candidate_config, popart_enabled=False),
        legacy_config,
    )


def _outputs() -> dict[str, torch.Tensor]:
    return {
        "slow_logits": torch.zeros(3, requires_grad=True),
        "fast_logits": torch.zeros(2, requires_grad=True),
        "event_logits": torch.zeros(2, requires_grad=True),
    }


def _gradients(agent, state: dict):
    outputs = _outputs()
    loss = agent.agent._compute_auxiliary_loss([state], [outputs])
    loss.backward()
    return loss.detach(), {
        key: value.grad.detach().clone()
        for key, value in outputs.items()
    }


def test_candidate_changes_only_event_auxiliary_direction_on_prepare_sample() -> None:
    legacy, candidate, _ = _agents()
    state = _state()
    legacy_targets = legacy.agent._build_mechanism_targets(state)
    legacy_event = legacy.agent._event_auxiliary_target(state, legacy_targets)
    candidate_event = candidate.agent._event_auxiliary_target(state, legacy_targets)
    assert legacy_event["event_target"] == 0
    assert candidate_event == {
        "event_target": 1,
        "event_soft_target": 1.0,
        "supervision_weight": 1.0,
        "decision": "prepare",
        "reason": "service_safe_feasible_future_readiness_gain",
    }

    _, old_gradients = _gradients(legacy, state)
    _, new_gradients = _gradients(candidate, state)
    assert torch.equal(old_gradients["slow_logits"], new_gradients["slow_logits"])
    assert torch.equal(old_gradients["fast_logits"], new_gradients["fast_logits"])
    assert old_gradients["event_logits"][1] > old_gradients["event_logits"][0]
    assert new_gradients["event_logits"][1] < new_gradients["event_logits"][0]


def test_unknown_sample_abstains_from_event_ce_and_temporal_only() -> None:
    legacy, candidate, _ = _agents()
    state = _state()
    del state["calibrated_context"]["time_contract"]
    old_loss, old_gradients = _gradients(legacy, state)
    new_loss, new_gradients = _gradients(candidate, state)
    assert new_loss < old_loss
    assert torch.equal(old_gradients["slow_logits"], new_gradients["slow_logits"])
    assert torch.equal(old_gradients["fast_logits"], new_gradients["fast_logits"])
    assert torch.count_nonzero(old_gradients["event_logits"]) > 0
    assert torch.count_nonzero(new_gradients["event_logits"]) == 0
    assert candidate.agent._event_auxiliary_supervision_weight(state) == 0.0


def test_default_disabled_and_baseline_method_identity_are_unchanged() -> None:
    legacy, candidate, config = _agents()
    for key, value in legacy.agent._network.state_dict().items():
        assert torch.equal(value, candidate.agent._network.state_dict()[key])
    default = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    state = _state()
    target = default.agent._build_mechanism_targets(state)
    assert default.agent._event_auxiliary_target(state, target)["decision"] == "legacy"

    for method in ("ppo", "mappo"):
        baseline = _build_learned(method, 7, config, popart_enabled=False)
        changed = _build_learned(
            method,
            7,
            {
                **config,
                "mechanism_aux_causal_public_prepare_advantage_enabled": True,
            },
            popart_enabled=False,
        )
        for key, value in baseline.agent._network.state_dict().items():
            assert torch.equal(value, changed.agent._network.state_dict()[key])


def test_checkpoint_semantics_and_mutual_exclusion_are_explicit(
    tmp_path: Path,
) -> None:
    legacy, candidate, config = _agents()
    legacy_path = tmp_path / "legacy.pt"
    candidate_path = tmp_path / "candidate.pt"
    legacy.agent.save(str(legacy_path))
    candidate.agent.save(str(candidate_path))
    payload = torch.load(candidate_path, map_location="cpu")
    assert payload["config"]["mechanism_aux_event_target_semantics"] == (
        EVENT_LABEL_SCHEMA_VERSION
    )
    legacy.agent.load(str(legacy_path))
    candidate.agent.load(str(candidate_path))
    with pytest.raises(ValueError, match="causal public prepare-advantage"):
        legacy.agent.load(str(candidate_path))
    with pytest.raises(ValueError, match="causal public prepare-advantage"):
        candidate.agent.load(str(legacy_path))

    stale_path = tmp_path / "candidate_v1.pt"
    payload["config"]["mechanism_aux_event_target_semantics"] = (
        "causal_public_prepare_advantage_v1"
    )
    torch.save(payload, stale_path)
    with pytest.raises(ValueError, match="event-target checkpoint label"):
        candidate.agent.load(str(stale_path))

    invalid = deepcopy(config)
    invalid["mechanism_aux_missing_current_event_abstention_enabled"] = True
    invalid["mechanism_aux_causal_public_prepare_advantage_enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_learned("sa_ghmappo", 7, invalid, popart_enabled=False)
