from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent, _load_json
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
)
from src.trainers.popart import ScalarPopArt


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT_DIR / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
MANIFEST = ROOT_DIR / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"


def _dev_state() -> tuple[dict, list[float], dict]:
    config, _ = _load_experiment_config(BASE_CONFIG)
    manifest = _load_json(MANIFEST)
    instance = deepcopy(next(row for row in manifest["instances"] if row["split"] == "dev"))
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    return config, observation, info


def test_popart_preserves_denormalized_output_and_adam_moments() -> None:
    torch.manual_seed(7)
    layer = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    inputs = torch.randn(5, 3)
    layer(inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    before = ScalarPopArt.linear_output(inputs, layer).detach().clone()
    old_exp_avg = optimizer.state[layer.weight]["exp_avg"].detach().clone()

    popart = ScalarPopArt(min_std=1.0)
    update = popart.update(
        np.full(60, 100.0, dtype=np.float64),
        output_layer=layer,
        optimizer=optimizer,
    )
    after = popart.denormalize_tensor(
        ScalarPopArt.linear_output(inputs, layer)
    ).detach()

    torch.testing.assert_close(after, before, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        optimizer.state[layer.weight]["exp_avg"],
        old_exp_avg * update.affine_scale,
        atol=1e-8,
        rtol=1e-6,
    )
    assert popart.mean == pytest.approx(100.0)
    assert popart.std == pytest.approx(1.0)


def test_popart_handles_near_zero_and_extreme_finite_targets() -> None:
    layer = nn.Linear(2, 1)
    popart = ScalarPopArt(min_std=1.0, max_abs_target=1.0e20)
    popart.update(
        [3.0, 3.0 + 1e-12, 3.0 - 1e-12],
        output_layer=layer,
        optimizer=None,
    )
    assert popart.std == pytest.approx(1.0)
    popart.update(
        [1.0e12, -1.0e12],
        output_layer=layer,
        optimizer=None,
    )
    assert np.isfinite(popart.mean)
    assert np.isfinite(popart.std)
    assert torch.isfinite(layer.weight).all()
    assert torch.isfinite(layer.bias).all()


@pytest.mark.parametrize("targets", [[float("nan")], [float("inf")], [1.0e21]])
def test_popart_rejects_nonfinite_or_out_of_bound_targets(targets: list[float]) -> None:
    popart = ScalarPopArt(min_std=1.0, max_abs_target=1.0e20)
    with pytest.raises(ValueError):
        popart.update(targets, output_layer=nn.Linear(2, 1), optimizer=None)


def test_popart_agent_preserves_actor_auxiliary_and_denormalized_value() -> None:
    config, observation, info = _dev_state()
    agent = _build_agent(
        "sa_ghmappo",
        7,
        config,
        agent_overrides={
            "deterministic_action": True,
            "value_normalization_enabled": True,
            "popart_min_std": 1.0,
        },
    )
    semantic_state = agent._extract_semantic_state(info)
    before = agent._forward_policy(semantic_state, run_metadata=info.get("run_metadata"))
    before_logits = {
        key: before[key].detach().clone()
        for key in ("slow_logits", "fast_logits", "event_logits")
    }
    before_value = before["value"].detach().clone()
    before_aux = agent._compute_auxiliary_loss(
        batch_states=[semantic_state], batch_outputs=[before]
    ).detach()
    action_before, _ = agent.act(observation, info)

    agent._popart.update(
        np.linspace(10.0, 110.0, 60, dtype=np.float64),
        output_layer=agent._value_output_layer(),
        optimizer=agent._optimizer,
    )
    after = agent._forward_policy(semantic_state, run_metadata=info.get("run_metadata"))
    after_aux = agent._compute_auxiliary_loss(
        batch_states=[semantic_state], batch_outputs=[after]
    ).detach()

    for key, logits in before_logits.items():
        torch.testing.assert_close(after[key], logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(after["value"], before_value, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(after_aux, before_aux, atol=0.0, rtol=0.0)
    action_after, _ = agent.act(observation, info)
    assert action_before == action_after


def test_popart_checkpoint_round_trip_and_disabled_compatibility(tmp_path: Path) -> None:
    config, _, info = _dev_state()
    enabled = _build_agent(
        "ppo",
        17,
        config,
        agent_overrides={"value_normalization_enabled": True, "popart_min_std": 1.0},
    )
    enabled._popart.update(
        np.linspace(-20.0, 80.0, 60, dtype=np.float64),
        output_layer=enabled._value_output_layer(),
        optimizer=enabled._optimizer,
    )
    enabled_path = tmp_path / "enabled.pt"
    enabled.save(str(enabled_path))
    restored = _build_agent(
        "ppo",
        17,
        config,
        agent_overrides={"value_normalization_enabled": True, "popart_min_std": 1.0},
    )
    restored.load(str(enabled_path))
    semantic_state = enabled._extract_semantic_state(info)
    expected = enabled._forward_policy(semantic_state)["value"]
    actual = restored._forward_policy(semantic_state)["value"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)
    assert restored._popart.state_dict() == enabled._popart.state_dict()

    disabled = _build_agent("ppo", 29, config)
    disabled_path = tmp_path / "disabled.pt"
    disabled.save(str(disabled_path))
    disabled_restored = _build_agent("ppo", 29, config)
    disabled_restored.load(str(disabled_path))
    assert disabled_restored._popart is None
    enabled_from_old = _build_agent(
        "ppo",
        29,
        config,
        agent_overrides={"value_normalization_enabled": True},
    )
    with pytest.raises(ValueError, match="requires PopArt checkpoint state"):
        enabled_from_old.load(str(disabled_path))
