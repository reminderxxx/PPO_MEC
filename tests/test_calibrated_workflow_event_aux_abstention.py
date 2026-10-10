"""Current-missing abstention for only the event auxiliary supervision."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from scripts.run_calibrated_workflow_value_normalization_ab import (
    _collect_exact_update_batch,
)
from src.envs.core.calibrated_continuous_workflow_env import (
    PREPARED_STATE_PREFIX_PROFILE,
)
from tests.test_calibrated_workflow_prepared_state_prefix import (
    OLD_PROFILE,
    _env,
    _fixture,
)


def _agents(config: dict):
    legacy_config = {
        **config,
        "mechanism_aux_current_service_feasibility_gate_enabled": False,
        "mechanism_aux_missing_current_event_abstention_enabled": False,
    }
    candidate_config = {
        **config,
        "mechanism_aux_current_service_feasibility_gate_enabled": False,
        "mechanism_aux_missing_current_event_abstention_enabled": True,
    }
    return (
        _build_learned("sa_ghmappo", 7, legacy_config, popart_enabled=False),
        _build_learned("sa_ghmappo", 7, candidate_config, popart_enabled=False),
    )


def _state_after_steady_actions(config: dict, instance: dict, count: int) -> dict:
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    for _ in range(count):
        assert 3 in env.valid_actions()
        env.step(3)
    return env._info()["semantic_state"]


def _outputs() -> dict[str, torch.Tensor]:
    return {
        "slow_logits": torch.zeros(3, requires_grad=True),
        "fast_logits": torch.zeros(2, requires_grad=True),
        "event_logits": torch.zeros(2, requires_grad=True),
    }


def _loss_and_gradients(agent, states: list[dict]):
    outputs = [_outputs() for _ in states]
    loss = agent.agent._compute_auxiliary_loss(states, outputs)
    loss.backward()
    gradients = [
        {
            name: output[name].grad.detach().clone()
            for name in ("slow_logits", "fast_logits", "event_logits")
        }
        for output in outputs
    ]
    return loss.detach(), gradients


def test_ready_sample_preserves_all_auxiliary_gradients_exactly() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    legacy.agent._temporal_consistency_coef = 0.7
    candidate.agent._temporal_consistency_coef = 0.7
    ready = _state_after_steady_actions(config, instance, 2)

    old_loss, old_gradients = _loss_and_gradients(legacy, [ready])
    new_loss, new_gradients = _loss_and_gradients(candidate, [ready])
    assert torch.equal(old_loss, new_loss)
    for name in ("slow_logits", "fast_logits", "event_logits"):
        assert torch.equal(old_gradients[0][name], new_gradients[0][name])
    assert candidate.agent._event_auxiliary_supervision_weight(ready) == 1.0


def test_missing_sample_masks_event_ce_and_temporal_margin_but_not_other_losses() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    legacy.agent._temporal_consistency_coef = 0.7
    candidate.agent._temporal_consistency_coef = 0.7
    missing = _state_after_steady_actions(config, instance, 4)

    old_loss, old_gradients = _loss_and_gradients(legacy, [missing])
    new_loss, new_gradients = _loss_and_gradients(candidate, [missing])
    assert new_loss < old_loss
    assert torch.equal(old_gradients[0]["slow_logits"], new_gradients[0]["slow_logits"])
    assert torch.equal(old_gradients[0]["fast_logits"], new_gradients[0]["fast_logits"])
    assert torch.count_nonzero(old_gradients[0]["event_logits"]) > 0
    assert torch.count_nonzero(new_gradients[0]["event_logits"]) == 0
    assert candidate.agent._event_auxiliary_supervision_weight(missing) == 0.0

    event_logits = torch.zeros(2, requires_grad=True)
    ppo_loss = -Categorical(logits=event_logits).log_prob(torch.tensor(1))
    ppo_loss.backward()
    assert torch.count_nonzero(event_logits.grad) > 0


def test_mixed_and_all_missing_batches_keep_original_denominator_without_nan() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    legacy.agent._temporal_consistency_coef = 0.7
    candidate.agent._temporal_consistency_coef = 0.7
    ready = _state_after_steady_actions(config, instance, 2)
    missing = _state_after_steady_actions(config, instance, 4)

    old_loss, _ = _loss_and_gradients(legacy, [ready, missing])
    new_loss, gradients = _loss_and_gradients(candidate, [ready, missing])
    target = legacy.agent._build_mechanism_targets(missing)
    missing_event_term = float(target["confidence_weight"]) * (
        legacy.agent._auxiliary_event_weight * nn.functional.cross_entropy(
            torch.zeros(1, 2), torch.tensor([target["event_target"]])
        )
        + legacy.agent._temporal_consistency_coef * nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(1), torch.tensor([float(target["event_soft_target"])])
        )
    )
    assert old_loss - new_loss == pytest.approx(float(missing_event_term) / 2.0)
    assert torch.count_nonzero(gradients[0]["event_logits"]) > 0
    assert torch.count_nonzero(gradients[1]["event_logits"]) == 0
    assert candidate.agent._auxiliary_event_supervision_stats([ready, missing]) == {
        "eligible_count": 2,
        "supervised_count": 1,
        "abstained_count": 1,
        "supervised_fraction": 0.5,
    }

    all_missing_loss, all_missing_gradients = _loss_and_gradients(
        candidate, [missing, deepcopy(missing)]
    )
    assert torch.isfinite(all_missing_loss)
    assert all(
        torch.count_nonzero(item["event_logits"]) == 0
        for item in all_missing_gradients
    )
    assert candidate.agent._auxiliary_event_supervision_stats([missing, missing])[
        "supervised_fraction"
    ] == 0.0


def test_abstention_uses_only_current_public_readiness_and_requires_v4() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    _, candidate = _agents(config)
    missing = _state_after_steady_actions(config, instance, 4)
    changed_future = deepcopy(missing)
    changed_future["predictions"] = {
        "predicted_next_rsu_by_vehicle": {"veh0": "unseen_future"},
        "predicted_first_handoff_rsu_by_vehicle": {"veh0": "unseen_future"},
        "next_rsu_sequence": {"veh0": ["unseen_future"]},
    }
    assert candidate.agent._event_auxiliary_supervision_weight(missing) == 0.0
    assert candidate.agent._event_auxiliary_supervision_weight(changed_future) == 0.0

    old_profile = deepcopy(missing)
    old_profile["interface_profile"] = OLD_PROFILE
    with pytest.raises(ValueError, match="requires the v4 public observation profile"):
        candidate.agent._event_auxiliary_supervision_weight(old_profile)


def test_checkpoint_semantics_and_mutually_exclusive_flags(tmp_path: Path) -> None:
    config, _ = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    legacy_path = tmp_path / "legacy.pt"
    candidate_path = tmp_path / "candidate.pt"
    legacy.agent.save(str(legacy_path))
    candidate.agent.save(str(candidate_path))
    payload = torch.load(candidate_path, map_location="cpu")
    assert payload["config"]["mechanism_aux_event_target_semantics"] == (
        "missing_current_event_abstention_v1"
    )
    legacy.agent.load(str(legacy_path))
    candidate.agent.load(str(candidate_path))
    with pytest.raises(ValueError, match="event-supervision abstention checkpoint mismatch"):
        legacy.agent.load(str(candidate_path))
    with pytest.raises(ValueError, match="event-supervision abstention checkpoint mismatch"):
        candidate.agent.load(str(legacy_path))

    invalid = {
        **config,
        "mechanism_aux_current_service_feasibility_gate_enabled": True,
        "mechanism_aux_missing_current_event_abstention_enabled": True,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_learned("sa_ghmappo", 7, invalid, popart_enabled=False)


def test_optimizer_receipt_discloses_fixed_denominator_supervision_fraction() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    _, candidate = _agents(config)
    rows, _, _, _ = _collect_exact_update_batch(
        agent=candidate,
        config=config,
        train_instances=[instance],
        order=[0],
        rng=random.Random(7),
        state=None,
        episodes_started=0,
        transition_count=8,
        step_cap=24,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert all(
        row["log_prob"]
        == row["action_info"]["env_action_log_prob"]
        for row in rows
    )
    update = candidate.learn(rows)
    assert update["optimizer_step_count"] == 4
    assert update["event_aux_supervision_eligible_count"] == 32
    assert update["event_aux_supervision_supervised_count"] == 16
    assert update["event_aux_supervision_abstained_count"] == 16
    assert update["event_aux_supervision_supervised_fraction"] == 0.5
    assert all(
        row["event_aux_supervision_eligible_count"] == 8
        and row["event_aux_supervision_supervised_count"] == 4
        and row["event_aux_supervision_abstained_count"] == 4
        and row["event_aux_supervision_supervised_fraction"] == 0.5
        for row in update["optimizer_step_records"]
    )


@pytest.mark.parametrize("method", ["ppo", "mappo", "dt_handoff_drl"])
def test_candidate_flag_is_sa_only_and_preserves_baseline_identity(
    method: str,
    tmp_path: Path,
) -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy_config = {
        **config,
        "mechanism_aux_missing_current_event_abstention_enabled": False,
    }
    candidate_config = {
        **config,
        "mechanism_aux_missing_current_event_abstention_enabled": True,
    }
    legacy = _build_learned(method, 7, legacy_config, popart_enabled=False)
    candidate = _build_learned(method, 7, candidate_config, popart_enabled=False)
    assert legacy.agent._mechanism_aux_missing_current_event_abstention_enabled is False
    assert candidate.agent._mechanism_aux_missing_current_event_abstention_enabled is False
    assert legacy.agent._network.state_dict().keys() == candidate.agent._network.state_dict().keys()
    for key, value in legacy.agent._network.state_dict().items():
        assert torch.equal(value, candidate.agent._network.state_dict()[key])

    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    observation, info = env.reset()
    _, old_info = legacy.act(observation, info)
    _, new_info = candidate.act(observation, info)
    assert old_info["env_action_probs"] == new_info["env_action_probs"]
    assert old_info["action_mask"] == new_info["action_mask"]

    checkpoint = tmp_path / f"{method}.pt"
    legacy.agent.save(str(checkpoint))
    candidate.agent.load(str(checkpoint))
