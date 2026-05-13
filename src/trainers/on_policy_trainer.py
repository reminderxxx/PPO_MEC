"""最小 on-policy trainer。"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.metrics.recorder import EpisodeRecorder
from src.trainers.base_trainer import BaseTrainer
from src.trainers.ppo_buffer import PPORolloutBuffer


class OnPolicyTrainer(BaseTrainer):
    """驱动 env-wrapper-agent 跑单个 episode，并支持 PPO rollout/GAE。"""

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        recorder: EpisodeRecorder,
        max_steps: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self._env = env
        self._agent = agent
        self._recorder = recorder
        self._max_steps = max_steps
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)

    def collect_episode(
        self,
        run_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._recorder.start_episode(run_metadata=run_metadata)
        observation, info = self._env.reset()
        buffer = PPORolloutBuffer()
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated and step_count < self._max_steps:
            decision_info = dict(info)
            action, action_info = self._agent.act(observation, info)
            next_observation, reward, terminated, truncated, next_info = self._env.step(action)
            estimated_value = float(action_info.get("value", self._estimate_value(observation, info)))
            buffer.add_step(
                observation=observation,
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                log_prob=float(action_info.get("log_prob", 0.0)),
                value=estimated_value,
                next_observation=next_observation,
                action_info=action_info,
                decision_info=decision_info,
                env_info=next_info,
            )
            observation = next_observation
            info = next_info
            step_count += 1

        last_value = 0.0 if terminated else self._estimate_value(observation, info)
        buffer.finalize(last_value=last_value, gamma=self._gamma, gae_lambda=self._gae_lambda)
        rollout = buffer.to_training_rows()
        summary = self._recorder.build_summary()
        summary["trainer_info"] = {
            "trainer_name": "on_policy_trainer",
            "max_steps": self._max_steps,
            "gamma": self._gamma,
            "gae_lambda": self._gae_lambda,
            "collected_steps": len(rollout),
            "bootstrap_value": round(float(last_value), 6),
        }
        return summary, rollout

    def run_episode(
        self,
        run_metadata: dict[str, Any] | None = None,
        learn: bool = True,
    ) -> dict[str, Any]:
        summary, rollout = self.collect_episode(run_metadata=run_metadata)
        if learn:
            learn_info = self._agent.learn(rollout)
        else:
            learn_info = {
                "agent_name": self._agent.agent_name,
                "policy_update_skipped": True,
                "reason": "evaluation_only",
                "collected_steps": len(rollout),
            }
        summary["agent_info"] = {
            "agent_name": self._agent.agent_name,
            "learn_info": learn_info,
        }
        return summary

    def _estimate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        if hasattr(self._agent, "evaluate_value") and callable(getattr(self._agent, "evaluate_value")):
            try:
                return float(self._agent.evaluate_value(observation, info))
            except TypeError:
                return float(self._agent.evaluate_value(observation))
        return 0.0
