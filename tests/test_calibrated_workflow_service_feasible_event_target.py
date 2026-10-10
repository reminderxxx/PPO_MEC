"""Target-only service-feasibility gate for the SA event auxiliary label."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from src.envs.core.calibrated_continuous_workflow_env import PREPARED_STATE_PREFIX_PROFILE
from tests.test_calibrated_workflow_prepared_state_prefix import OLD_PROFILE, _env, _fixture


def _agents(config: dict):
    legacy_config = {**config, "mechanism_aux_current_service_feasibility_gate_enabled": False}
    candidate_config = {**config, "mechanism_aux_current_service_feasibility_gate_enabled": True}
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


def test_gate_changes_only_event_target_when_current_complete_bundle_is_missing() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    legacy.agent._mechanism_confidence_floor = 1.0
    candidate.agent._mechanism_confidence_floor = 1.0
    semantic = _state_after_steady_actions(config, instance, 4)
    old = legacy.agent._build_mechanism_targets(semantic)
    new = candidate.agent._build_mechanism_targets(semantic)
    assert old["event_target"] == 1 and old["event_soft_target"] > 0.0
    assert new["event_target"] == 0 and new["event_soft_target"] == 0.0
    assert {key: value for key, value in old.items() if key not in {"event_target", "event_soft_target"}} == {
        key: value for key, value in new.items() if key not in {"event_target", "event_soft_target"}
    }


def test_current_ready_positive_and_default_disabled_behavior_are_exact() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    semantic = _state_after_steady_actions(config, instance, 2)
    expected = legacy.agent._build_mechanism_targets(semantic)
    assert expected["event_target"] == 1 and expected["event_soft_target"] > 0.0
    assert candidate.agent._build_mechanism_targets(semantic) == expected
    default = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    assert default.agent._build_mechanism_targets(semantic) == expected
    for key, value in legacy.agent._network.state_dict().items():
        assert torch.equal(value, candidate.agent._network.state_dict()[key])


def test_gate_requires_v4_public_bundle_contract() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    _, candidate = _agents(config)
    semantic = _state_after_steady_actions(config, instance, 4)
    semantic["interface_profile"] = OLD_PROFILE
    with pytest.raises(ValueError, match="requires the v4 public observation profile"):
        candidate.agent._build_mechanism_targets(semantic)


def test_auxiliary_event_gradient_direction_changes_but_slow_fast_do_not() -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    semantic = _state_after_steady_actions(config, instance, 4)

    def gradients(agent):
        outputs = {
            "slow_logits": torch.zeros(3, requires_grad=True),
            "fast_logits": torch.zeros(2, requires_grad=True),
            "event_logits": torch.zeros(2, requires_grad=True),
        }
        loss = agent.agent._compute_auxiliary_loss([semantic], [outputs])
        loss.backward()
        return tuple(outputs[name].grad.detach().clone() for name in ("slow_logits", "fast_logits", "event_logits"))

    old_slow, old_fast, old_event = gradients(legacy)
    new_slow, new_fast, new_event = gradients(candidate)
    assert torch.equal(old_slow, new_slow)
    assert torch.equal(old_fast, new_fast)
    assert old_event[1] < old_event[0]
    assert new_event[1] > new_event[0]


def test_checkpoint_target_semantics_are_explicit_and_cross_load_is_rejected(tmp_path: Path) -> None:
    config, _ = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    legacy, candidate = _agents(config)
    old_path, new_path = tmp_path / "legacy.pt", tmp_path / "candidate.pt"
    legacy.agent.save(str(old_path))
    candidate.agent.save(str(new_path))
    old_payload = torch.load(old_path, map_location="cpu")
    new_payload = torch.load(new_path, map_location="cpu")
    assert old_payload["config"]["mechanism_aux_event_target_semantics"] == "legacy_target_timing_v1"
    assert new_payload["config"]["mechanism_aux_event_target_semantics"] == "current_complete_bundle_ready_v1"
    legacy.agent.load(str(old_path))
    candidate.agent.load(str(new_path))
    with pytest.raises(ValueError, match="event-target semantics checkpoint mismatch"):
        legacy.agent.load(str(new_path))
    with pytest.raises(ValueError, match="event-target semantics checkpoint mismatch"):
        candidate.agent.load(str(old_path))
