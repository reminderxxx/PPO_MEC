"""多头 / 多控制器 on-policy trainer。"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.metrics.recorder import EpisodeRecorder
from src.trainers.base_trainer import BaseTrainer
from src.trainers.ppo_buffer import PPORolloutBuffer


class MARLOnPolicyTrainer(BaseTrainer):
    """保持 env 接口不变，但为多头协同策略保留 decision_info。"""

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
            decision_info["run_metadata"] = dict(run_metadata or {})
            action, action_info = self._agent.act(observation, info)
            next_observation, reward, terminated, truncated, next_info = self._env.step(int(action))
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
        summary["agent_action_diagnostics"] = self._summarize_agent_action_diagnostics(rollout)
        summary["trainer_info"] = {
            "trainer_name": "marl_on_policy_trainer",
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

    def _summarize_agent_action_diagnostics(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        total_steps = max(len(rollout), 1)
        guard_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("guard_triggered", False))
        ]
        target_mismatch_rows = [
            row
            for row in guard_rows
            if bool(row.get("action_info", {}).get("continuity_guard", {}).get("target_mismatch", False))
        ]
        prefetch_to_prepare_rows = [
            row
            for row in guard_rows
            if int(row.get("action_info", {}).get("original_action", -1)) == 1
            and int(row.get("action_info", {}).get("guarded_action", -1)) == 4
        ]
        hard_override_rows = [
            row
            for row in guard_rows
            if bool(row.get("action_info", {}).get("continuity_guard", {}).get("hard_override_applied", False))
        ]
        backhaul_guard_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("backhaul_guard", {}).get("guarded", False))
        ]
        cache_warm_guard_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("cache_warm_start_guard", {}).get("guarded", False))
        ]
        action_projection_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("action_projection_applied", False))
        ]
        guard_action_delta_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("guard_action_delta", False))
        ]
        invalid_action_attempt_count = sum(
            int(row.get("action_info", {}).get("invalid_action_attempt_count", 0) or 0)
            for row in rollout
        )

        def metric_mean(field_name: str) -> float:
            values = [
                float(row.get("env_info", {}).get("metrics_protocol", {}).get(field_name, 0.0) or 0.0)
                for row in rollout
            ]
            return round(float(sum(values)) / float(max(len(values), 1)), 6)

        return {
            "total_steps": len(rollout),
            "continuity_guard_trigger_count": len(guard_rows),
            "continuity_guard_trigger_rate": round(float(len(guard_rows)) / float(total_steps), 6),
            "target_mismatch_guard_count": len(target_mismatch_rows),
            "guard_prefetch_to_prepare_count": len(prefetch_to_prepare_rows),
            "guard_hard_override_count": len(hard_override_rows),
            "action_projection_count": len(action_projection_rows),
            "action_projection_rate": round(float(len(action_projection_rows)) / float(total_steps), 6),
            "invalid_action_attempt_count": invalid_action_attempt_count,
            "invalid_action_attempt_rate": round(float(invalid_action_attempt_count) / float(total_steps), 6),
            "guard_action_delta_count": len(guard_action_delta_rows),
            "guard_action_delta_rate": round(float(len(guard_action_delta_rows)) / float(total_steps), 6),
            "backhaul_guard_count": len(backhaul_guard_rows),
            "backhaul_guard_rate": round(float(len(backhaul_guard_rows)) / float(total_steps), 6),
            "cache_warm_start_guard_count": len(cache_warm_guard_rows),
            "cache_warm_start_guard_rate": round(float(len(cache_warm_guard_rows)) / float(total_steps), 6),
            "dag_frontier_size_mean": metric_mean("dag_frontier_size"),
            "dag_critical_path_pressure_mean": metric_mean("dag_critical_path_pressure"),
            "dag_current_node_dependency_pressure_mean": metric_mean("dag_current_node_dependency_pressure"),
            "dag_remaining_nodes_mean": metric_mean("dag_remaining_nodes"),
        }
