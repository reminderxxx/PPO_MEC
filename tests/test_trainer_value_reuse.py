"""Rollout collection must reuse actor values without dropping bootstrap calls."""

from __future__ import annotations

import pytest

from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
from src.trainers.on_policy_trainer import OnPolicyTrainer


class _EpisodeEnv:
    def __init__(self, *, truncated: bool) -> None:
        self.truncated = truncated
        self.index = 0

    def reset(self):
        self.index = 0
        return [0.0], {}

    def step(self, action):
        self.index += 1
        done = self.index == 3
        return [float(self.index)], 1.0, done and not self.truncated, done and self.truncated, {}


class _ValueAgent:
    def __init__(self, *, report_value: bool, value: float) -> None:
        self.report_value = report_value
        self.value = value
        self.calls = []

    def act(self, observation, info):
        return 0, {"value": self.value} if self.report_value else {}

    def evaluate_value(self, observation, info):
        self.calls.append(observation[0])
        return self.value


@pytest.mark.parametrize("trainer_type", [OnPolicyTrainer, MARLOnPolicyTrainer])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("value", [0.0, 2.5])
def test_reusing_actor_values_preserves_rollout_and_bootstrap(trainer_type, truncated, value):
    collected = []
    agents = []
    for report_value in [True, False]:
        agent = _ValueAgent(report_value=report_value, value=value)
        trainer = trainer_type(
            env=_EpisodeEnv(truncated=truncated), agent=agent,
            recorder=EpisodeRecorder(), max_steps=4,
        )
        options = {"collect_model_targets": False} if trainer_type is MARLOnPolicyTrainer else {}
        summary, rows = trainer.collect_episode(**options)
        agents.append(agent)
        collected.append((summary, rows))

    reused_rows = collected[0][1]
    fallback_rows = collected[1][1]
    for reused, fallback in zip(reused_rows, fallback_rows):
        for key in ["action", "reward", "value", "next_value", "advantage", "return"]:
            assert reused[key] == fallback[key]
    assert len(reused_rows) == len(fallback_rows) == 3
    assert len(agents[1].calls) - len(agents[0].calls) == 3
    if trainer_type is OnPolicyTrainer:
        assert agents[0].calls == ([3.0] if truncated else [])
    else:
        assert agents[0].calls == ([1.0, 2.0, 3.0, 3.0] if truncated else [1.0, 2.0])
    assert collected[0][0]["trainer_info"]["bootstrap_value"] == (value if truncated else 0.0)
