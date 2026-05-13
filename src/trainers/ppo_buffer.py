"""PPO rollout buffer 与 GAE 计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PPORolloutStep:
    """单步 rollout 记录。"""

    observation: list[float]
    action: int
    reward: float
    terminated: bool
    truncated: bool
    log_prob: float
    value: float
    next_observation: list[float]
    action_info: dict[str, Any]
    decision_info: dict[str, Any]
    env_info: dict[str, Any]
    advantage: float = 0.0
    return_value: float = 0.0


class PPORolloutBuffer:
    """收集 on-policy trajectory，并计算 GAE / return。"""

    def __init__(self) -> None:
        self._steps: list[PPORolloutStep] = []

    def add_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: float,
        value: float,
        next_observation: Any,
        action_info: dict[str, Any],
        decision_info: dict[str, Any],
        env_info: dict[str, Any],
    ) -> None:
        self._steps.append(
            PPORolloutStep(
                observation=np.asarray(observation, dtype=np.float32).reshape(-1).tolist(),
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                log_prob=float(log_prob),
                value=float(value),
                next_observation=np.asarray(next_observation, dtype=np.float32).reshape(-1).tolist(),
                action_info=dict(action_info),
                decision_info=dict(decision_info),
                env_info=dict(env_info),
            )
        )

    def finalize(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        if not self._steps:
            return

        advantages = [0.0 for _ in self._steps]
        gae = 0.0
        for index in reversed(range(len(self._steps))):
            step = self._steps[index]
            if index == len(self._steps) - 1:
                next_value = float(last_value)
            else:
                next_value = float(self._steps[index + 1].value)

            next_non_terminal = 0.0 if step.terminated else 1.0
            delta = step.reward + gamma * next_value * next_non_terminal - step.value
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[index] = gae

        for index, advantage in enumerate(advantages):
            self._steps[index].advantage = float(advantage)
            self._steps[index].return_value = float(advantage + self._steps[index].value)

    def to_training_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "observation": step.observation,
                "action": step.action,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "log_prob": step.log_prob,
                "value": step.value,
                "advantage": step.advantage,
                "return": step.return_value,
                "next_observation": step.next_observation,
                "action_info": dict(step.action_info),
                "decision_info": dict(step.decision_info),
                "env_info": dict(step.env_info),
            }
            for step in self._steps
        ]

    def __len__(self) -> int:
        return len(self._steps)
