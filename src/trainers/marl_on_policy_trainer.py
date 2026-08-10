"""多头 / 多控制器 on-policy trainer。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
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
        collect_model_targets: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._recorder.start_episode(run_metadata=run_metadata)
        observation, info = self._env.reset()
        buffer = PPORolloutBuffer()
        terminated = False
        truncated = False
        step_count = 0
        algorithm_memory = self._initial_algorithm_memory()
        online_planner_enabled = bool(
            getattr(self._agent, "_env_action_model_online_planner_enabled", False)
        )
        teacher_distillation_enabled = bool(
            getattr(
                self._agent,
                "_env_action_model_teacher_distillation_enabled",
                False,
            )
            and collect_model_targets
        )
        training_planner_enabled = bool(
            getattr(
                self._agent,
                "_env_action_model_training_planner_enabled",
                False,
            )
            and collect_model_targets
        )

        while not terminated and not truncated and step_count < self._max_steps:
            policy_info = dict(info)
            policy_info["run_metadata"] = dict(run_metadata or {})
            policy_info["algorithm_memory"] = deepcopy(algorithm_memory)
            decision_info = dict(policy_info)
            action, action_info = self._agent.act(observation, policy_info)
            learned_transition_model_rollout = None
            if bool(
                getattr(
                    self._agent,
                    "_learned_transition_model_planner_enabled",
                    False,
                )
            ):
                learned_transition_model_rollout = (
                    self._agent.predict_learned_transition_targets(
                        observation=observation,
                        action_info=action_info,
                    )
                )
            if learned_transition_model_rollout:
                planned_action, planner_stats = (
                    self._agent.select_env_action_from_model_targets(
                        action_info=action_info,
                        rollout_info=learned_transition_model_rollout,
                    )
                )
                if int(planned_action) != int(action):
                    action_info = self._agent.relabel_action_info_for_env_action(
                        action_info=action_info,
                        decision_info=decision_info,
                        env_action=int(planned_action),
                        planner_stats=planner_stats,
                    )
                    action = int(planned_action)
                else:
                    action_info = dict(action_info)
                    action_info["online_counterfactual_planner"] = planner_stats
                action_info["env_action_model_rollout"] = learned_transition_model_rollout
                action_info["learned_transition_model_planner"] = planner_stats
            env_action_model_rollout = None
            if (
                collect_model_targets
                or online_planner_enabled
                or teacher_distillation_enabled
                or training_planner_enabled
            ):
                env_action_model_rollout = (
                    self._collect_env_action_counterfactual_targets(
                        observation=observation,
                        action_info=action_info,
                        algorithm_memory=algorithm_memory,
                        decision_info=decision_info,
                        run_metadata=run_metadata,
                    )
                )
            if env_action_model_rollout and (
                online_planner_enabled or training_planner_enabled
            ):
                planned_action, planner_stats = (
                    self._agent.select_env_action_from_model_targets(
                        action_info=action_info,
                        rollout_info=env_action_model_rollout,
                        training_only=(
                            training_planner_enabled and not online_planner_enabled
                        ),
                    )
                )
                if int(planned_action) != int(action):
                    action_info = self._agent.relabel_action_info_for_env_action(
                        action_info=action_info,
                        decision_info=decision_info,
                        env_action=int(planned_action),
                        planner_stats=planner_stats,
                    )
                else:
                    action_info = dict(action_info)
                    action_info["online_counterfactual_planner"] = planner_stats
                action = int(planned_action)
            if (
                env_action_model_rollout
                and teacher_distillation_enabled
                and not online_planner_enabled
            ):
                _, teacher_planner_stats = (
                    self._agent.select_env_action_from_model_targets(
                        action_info=action_info,
                        rollout_info=env_action_model_rollout,
                        teacher_only=True,
                    )
                )
                action_info = dict(action_info)
                action_info["counterfactual_teacher_planner"] = (
                    teacher_planner_stats
                )
            if env_action_model_rollout:
                action_info = dict(action_info)
                action_info["env_action_model_rollout"] = env_action_model_rollout
            counterfactual_model_rollout = None
            if collect_model_targets:
                counterfactual_model_rollout = (
                    self._collect_option_counterfactual_targets(
                        action_info=action_info,
                        run_metadata=run_metadata,
                    )
                )
            if counterfactual_model_rollout:
                action_info = dict(action_info)
                option_gate_info = dict(action_info.get("option_gate", {}))
                option_gate_info["counterfactual_model_rollout"] = (
                    counterfactual_model_rollout
                )
                action_info["option_gate"] = option_gate_info
            next_observation, reward, terminated, truncated, next_info = self._env.step(int(action))
            estimated_value = float(action_info.get("value", self._estimate_value(observation, policy_info)))
            next_estimated_value = (
                0.0
                if terminated
                else self._estimate_value(next_observation, next_info)
            )
            buffer.add_step(
                observation=observation,
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                log_prob=float(action_info.get("log_prob", 0.0)),
                value=estimated_value,
                next_observation=next_observation,
                next_value=next_estimated_value,
                action_info=action_info,
                decision_info=decision_info,
                env_info=next_info,
            )
            algorithm_memory = self._advance_algorithm_memory(
                algorithm_memory,
                action=int(action),
                reward=float(reward),
                decision_info=decision_info,
                next_info=next_info,
            )
            observation = next_observation
            info = next_info
            step_count += 1

        bootstrap_info = dict(info)
        bootstrap_info["run_metadata"] = dict(run_metadata or {})
        bootstrap_info["algorithm_memory"] = deepcopy(algorithm_memory)
        last_value = 0.0 if terminated else self._estimate_value(observation, bootstrap_info)
        buffer.finalize(last_value=last_value, gamma=self._gamma, gae_lambda=self._gae_lambda)
        rollout = buffer.to_training_rows()
        summary = self._recorder.build_summary()
        summary["agent_action_diagnostics"] = self._summarize_agent_action_diagnostics(rollout)
        counterfactual_model_query_count = sum(
            int(
                row.get("action_info", {})
                .get("option_gate", {})
                .get("counterfactual_model_rollout", {})
                .get("unique_model_query_count", 0)
                or 0
            )
            + int(
                row.get("action_info", {})
                .get("env_action_model_rollout", {})
                .get("unique_model_query_count", 0)
                or 0
            )
            for row in rollout
        )
        counterfactual_model_transition_count = sum(
            int(
                row.get("action_info", {})
                .get("option_gate", {})
                .get("counterfactual_model_rollout", {})
                .get("model_transition_count", 0)
                or 0
            )
            + int(
                row.get("action_info", {})
                .get("env_action_model_rollout", {})
                .get("model_transition_count", 0)
                or 0
            )
            for row in rollout
        )
        summary["trainer_info"] = {
            "trainer_name": "marl_on_policy_trainer",
            "max_steps": self._max_steps,
            "gamma": self._gamma,
            "gae_lambda": self._gae_lambda,
            "collected_steps": len(rollout),
            "bootstrap_value": round(float(last_value), 6),
            "counterfactual_model_query_count": counterfactual_model_query_count,
            "counterfactual_model_transition_count": (
                counterfactual_model_transition_count
            ),
        }
        return summary, rollout

    def _collect_env_action_counterfactual_targets(
        self,
        *,
        observation: Any,
        action_info: dict[str, Any],
        algorithm_memory: dict[str, Any],
        decision_info: dict[str, Any],
        run_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not bool(action_info.get("env_action_model_rollout_enabled", False)):
            return {}
        action_mask = list(action_info.get("action_mask", []))
        valid_actions = [
            action_id
            for action_id in range(5)
            if not action_mask
            or (action_id < len(action_mask) and bool(action_mask[action_id]))
        ]
        if not valid_actions:
            return {}

        beam_search_payload: dict[str, Any] = {}
        beam_search_transition_count = 0
        beam_search_enabled = bool(
            action_info.get("env_action_model_beam_search_enabled", False)
        )
        beam_search_context_only = bool(
            action_info.get(
                "env_action_model_beam_search_context_only",
                True,
            )
        )
        beam_search_eta = int(
            action_info.get("predicted_first_non_current_eta", 0) or 0
        )
        beam_search_min_eta = max(
            int(action_info.get("env_action_model_beam_search_min_eta", 0)),
            0,
        )
        beam_search_max_eta = max(
            int(action_info.get("env_action_model_beam_search_max_eta", 999)),
            beam_search_min_eta,
        )
        beam_search_context_active = bool(
            self._algorithm_memory_has_recovery_signal(algorithm_memory)
            or (
                action_info.get("raw_handoff_candidate", False)
                and beam_search_min_eta
                <= beam_search_eta
                <= beam_search_max_eta
            )
        )
        if (
            beam_search_enabled
            and (
                not beam_search_context_only
                or beam_search_context_active
            )
        ):
            (
                beam_search_payload,
                beam_search_transition_count,
            ) = self._collect_env_action_beam_search_targets(
                valid_actions=valid_actions,
                algorithm_memory=algorithm_memory,
                decision_info=decision_info,
                run_metadata=run_metadata,
                horizon=max(
                    int(
                        action_info.get(
                            "env_action_model_beam_search_horizon",
                            4,
                        )
                    ),
                    1,
                ),
                beam_width=max(
                    int(
                        action_info.get(
                            "env_action_model_beam_search_width",
                            2,
                        )
                    ),
                    1,
                ),
            )

        rollout_horizons_raw = action_info.get(
            "env_action_model_rollout_horizons",
            [action_info.get("env_action_model_rollout_horizon", 1)],
        )
        if not isinstance(rollout_horizons_raw, (list, tuple)):
            rollout_horizons_raw = [rollout_horizons_raw]
        rollout_horizons = sorted(
            {
                max(int(horizon or 1), 1)
                for horizon in rollout_horizons_raw
            }
        )
        rollout_horizon = max(rollout_horizons)
        action_targets: dict[int, float] = {}
        action_rewards: dict[int, float] = {}
        action_transition_samples: list[dict[str, Any]] = []
        action_mechanism_targets: dict[int, float] = {}
        action_targets_by_horizon: dict[int, dict[int, float]] = {
            horizon: {}
            for horizon in rollout_horizons
        }
        action_mechanism_targets_by_horizon: dict[int, dict[int, float]] = {
            horizon: {}
            for horizon in rollout_horizons
        }
        action_resource_costs_by_horizon: dict[int, dict[int, float]] = {
            horizon: {}
            for horizon in rollout_horizons
        }
        imagination_replay_enabled = bool(
            action_info.get(
                "env_action_model_imagination_replay_enabled",
                False,
            )
        )
        imagination_recovery_only = bool(
            action_info.get(
                "env_action_model_imagination_replay_recovery_only",
                False,
            )
        )
        imagination_depths = {
            max(int(depth), 1)
            for depth in action_info.get(
                "env_action_model_imagination_replay_depths",
                [],
            )
            if int(depth) < rollout_horizon
        }
        imagination_horizons = sorted(
            {
                max(int(horizon), 1)
                for horizon in action_info.get(
                    "env_action_model_imagination_replay_horizons",
                    [1],
                )
            }
        )
        projection_info = dict(action_info.get("action_projection", {}))
        root_probs = projection_info.get("masked_env_action_probs", [])
        if isinstance(root_probs, list) and len(root_probs) >= 5:
            dominant_action = max(
                valid_actions,
                key=lambda action_id: float(root_probs[action_id]),
            )
        else:
            dominant_action = int(
                action_info.get("final_env_action", valid_actions[0])
            )
        imagination_branch_actions = self._select_imagination_branch_actions(
            valid_actions=valid_actions,
            root_probs=root_probs,
            dominant_action=dominant_action,
            branch_mode=str(
                action_info.get(
                    "env_action_model_imagination_replay_branch_mode",
                    "dominant",
                )
            ),
            branch_top_k=max(
                int(
                    action_info.get(
                        "env_action_model_imagination_replay_branch_top_k",
                        1,
                    )
                ),
                1,
            ),
        )
        imagined_recovery_samples: list[dict[str, Any]] = []
        imagined_model_query_count = 0
        model_transition_count = 0
        prefetch_validation_window = max(
            int(
                action_info.get(
                    "env_action_model_prefetch_validation_window",
                    6,
                )
                or 6
            ),
            1,
        )
        for initial_action in valid_actions:
            branch_env = deepcopy(self._env)
            if hasattr(branch_env, "_recorder"):
                branch_env._recorder = None
            branch_memory = deepcopy(algorithm_memory)
            branch_decision_info = deepcopy(decision_info)
            branch_action = initial_action
            discounted_return = 0.0
            discount = 1.0
            next_observation: Any = None
            next_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            first_reward = 0.0
            first_next_observation: Any = None
            mechanism_score = 0.0
            resource_cost = 0.0
            pending_prefetches: list[dict[str, int | str | None]] = []
            for branch_step in range(rollout_horizon):
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    next_info,
                ) = branch_env.step(branch_action)
                model_transition_count += 1
                if branch_step == 0:
                    first_reward = float(reward)
                    first_next_observation = deepcopy(next_observation)
                branch_metrics = next_info.get("metrics_protocol", {})
                if not isinstance(branch_metrics, dict):
                    branch_metrics = {}
                current_associated_rsu_id = branch_metrics.get(
                    "post_action_associated_rsu_id"
                )
                remaining_prefetches: list[dict[str, int | str | None]] = []
                for pending_prefetch in pending_prefetches:
                    source_step = int(pending_prefetch["source_step"])
                    step_gap = branch_step - source_step
                    if (
                        current_associated_rsu_id is not None
                        and current_associated_rsu_id
                        == pending_prefetch.get("target_rsu_id")
                        and step_gap >= 1
                        and step_gap <= prefetch_validation_window
                    ):
                        mechanism_score = 1.0
                        continue
                    if step_gap >= prefetch_validation_window:
                        continue
                    remaining_prefetches.append(pending_prefetch)
                pending_prefetches = remaining_prefetches
                if bool(
                    branch_metrics.get("mechanism_success_strict", False)
                    or branch_metrics.get("migration_prepare_realized", False)
                    or branch_metrics.get("handoff_ready", False)
                    or branch_metrics.get("handoff_ready_from_prepare", False)
                    or (
                        branch_step == 0
                        and int(branch_metrics.get("handoff_event_count", 0) or 0) > 0
                        and branch_metrics.get("migration_during_handoff", False)
                    )
                ):
                    mechanism_score = 1.0
                resource_cost += discount * (
                    max(
                        float(branch_metrics.get("backhaul_traffic_cost", 0.0) or 0.0),
                        0.0,
                    )
                    + max(
                        float(
                            branch_metrics.get(
                                "adapter_state_migration_overhead",
                                0.0,
                            )
                            or 0.0
                        ),
                        0.0,
                    )
                    * max(
                        float(
                            action_info.get(
                                "env_action_model_resource_cost_scale",
                                64.0,
                            )
                            or 64.0
                        ),
                        1e-6,
                    )
                )
                if (
                    branch_metrics.get("predictive_prefetch_requested", False)
                    and branch_metrics.get("cache_target_rsu_id") is not None
                ):
                    pending_prefetches.append(
                        {
                            "source_step": branch_step,
                            "target_rsu_id": branch_metrics.get(
                                "cache_target_rsu_id"
                            ),
                        }
                    )
                discounted_return += discount * float(reward)
                discount *= self._gamma
                branch_memory = self._advance_algorithm_memory(
                    branch_memory,
                    action=int(branch_action),
                    reward=float(reward),
                    decision_info=branch_decision_info,
                    next_info=next_info,
                )
                reached_horizon = branch_step + 1
                if reached_horizon in action_targets_by_horizon:
                    horizon_bootstrap = 0.0
                    if not terminated and not truncated:
                        horizon_bootstrap_info = dict(next_info)
                        horizon_bootstrap_info["run_metadata"] = dict(
                            run_metadata or {}
                        )
                        horizon_bootstrap_info["algorithm_memory"] = deepcopy(
                            branch_memory
                        )
                        horizon_bootstrap = self._estimate_value(
                            next_observation,
                            horizon_bootstrap_info,
                        )
                    action_targets_by_horizon[reached_horizon][initial_action] = (
                        float(discounted_return + discount * horizon_bootstrap)
                    )
                    action_mechanism_targets_by_horizon[reached_horizon][
                        initial_action
                    ] = mechanism_score
                    action_resource_costs_by_horizon[reached_horizon][
                        initial_action
                    ] = float(resource_cost)
                if terminated or truncated:
                    for horizon in rollout_horizons:
                        if horizon > reached_horizon:
                            action_targets_by_horizon[horizon][initial_action] = (
                                float(discounted_return)
                            )
                            action_mechanism_targets_by_horizon[horizon][
                                initial_action
                            ] = mechanism_score
                            action_resource_costs_by_horizon[horizon][
                                initial_action
                            ] = float(resource_cost)
                    break
                if branch_step + 1 < rollout_horizon:
                    branch_policy_info = dict(next_info)
                    branch_policy_info["run_metadata"] = dict(run_metadata or {})
                    branch_policy_info["deterministic_policy"] = True
                    branch_policy_info["algorithm_memory"] = deepcopy(
                        branch_memory
                    )
                    branch_decision_info = deepcopy(branch_policy_info)
                    branch_action, branch_action_info = self._agent.act(
                        next_observation,
                        branch_policy_info,
                    )
                    if (
                        imagination_replay_enabled
                        and initial_action in imagination_branch_actions
                        and reached_horizon in imagination_depths
                        and (
                            not imagination_recovery_only
                            or self._algorithm_memory_has_recovery_signal(
                                branch_memory
                            )
                        )
                    ):
                        (
                            imagined_sample,
                            imagined_transition_count,
                        ) = self._collect_imagined_env_action_sample(
                            branch_env=branch_env,
                            observation=next_observation,
                            decision_info=branch_policy_info,
                            algorithm_memory=branch_memory,
                            action_info=branch_action_info,
                            run_metadata=run_metadata,
                            imagination_depth=reached_horizon,
                            imagination_horizons=imagination_horizons,
                        )
                        if imagined_sample:
                            imagined_recovery_samples.append(
                                imagined_sample
                            )
                            imagined_model_query_count += int(
                                imagined_sample["action_info"][
                                    "env_action_model_rollout"
                                ].get("unique_model_query_count", 0)
                            )
                            model_transition_count += (
                                imagined_transition_count
                            )
            bootstrap_value = 0.0
            if not terminated and not truncated:
                bootstrap_info = dict(next_info)
                bootstrap_info["run_metadata"] = dict(run_metadata or {})
                bootstrap_info["algorithm_memory"] = deepcopy(branch_memory)
                bootstrap_value = self._estimate_value(
                    next_observation,
                    bootstrap_info,
                )
            action_rewards[initial_action] = first_reward
            if first_next_observation is not None:
                action_transition_samples.append(
                    {
                        "observation": deepcopy(observation),
                        "action": int(initial_action),
                        "reward": float(first_reward),
                        "next_observation": deepcopy(first_next_observation),
                        "next_value": float(bootstrap_value),
                        "terminated": bool(terminated),
                    }
                )
            action_targets[initial_action] = float(
                discounted_return + discount * bootstrap_value
            )
            action_mechanism_targets[initial_action] = mechanism_score

        return {
            "protocol": "digital_twin_multihorizon_env_action_td_v2",
            "mechanism_target_protocol": (
                "branch_replay_prefetch_validation_and_handoff_alignment_v2"
            ),
            "prefetch_validation_window": prefetch_validation_window,
            "rollout_horizon": rollout_horizon,
            "rollout_horizons": rollout_horizons,
            "action_td_targets": {
                str(action_id): round(target, 6)
                for action_id, target in action_targets.items()
            },
            "action_td_targets_by_horizon": {
                str(horizon): {
                    str(action_id): round(target, 6)
                    for action_id, target in targets.items()
                }
                for horizon, targets in action_targets_by_horizon.items()
            },
            "action_mechanism_targets": {
                str(action_id): round(target, 6)
                for action_id, target in action_mechanism_targets.items()
            },
            "action_mechanism_targets_by_horizon": {
                str(horizon): {
                    str(action_id): round(target, 6)
                    for action_id, target in targets.items()
                }
                for horizon, targets in action_mechanism_targets_by_horizon.items()
            },
            "action_resource_costs": {
                str(action_id): round(
                    action_resource_costs_by_horizon[rollout_horizon].get(
                        action_id,
                        0.0,
                    ),
                    6,
                )
                for action_id in valid_actions
            },
            "action_resource_costs_by_horizon": {
                str(horizon): {
                    str(action_id): round(cost, 6)
                    for action_id, cost in costs.items()
                }
                for horizon, costs in action_resource_costs_by_horizon.items()
            },
            "action_immediate_rewards": {
                str(action_id): round(reward, 6)
                for action_id, reward in action_rewards.items()
            },
            "counterfactual_transition_samples": action_transition_samples,
            "imagined_recovery_samples": imagined_recovery_samples,
            "imagined_recovery_sample_count": len(
                imagined_recovery_samples
            ),
            "imagination_branch_actions": sorted(
                imagination_branch_actions
            ),
            "imagined_model_query_count": imagined_model_query_count,
            **beam_search_payload,
            "unique_model_query_count": (
                len(action_targets)
                + imagined_model_query_count
                + int(beam_search_payload.get("beam_model_query_count", 0))
            ),
            "model_transition_count": (
                model_transition_count + beam_search_transition_count
            ),
        }

    @staticmethod
    def _select_imagination_branch_actions(
        *,
        valid_actions: list[int],
        root_probs: Any,
        dominant_action: int,
        branch_mode: str,
        branch_top_k: int,
    ) -> set[int]:
        if str(branch_mode).strip().lower() != "top_k":
            return {int(dominant_action)}
        if isinstance(root_probs, list) and len(root_probs) >= 5:
            ranked_actions = sorted(
                valid_actions,
                key=lambda action_id: float(root_probs[action_id]),
                reverse=True,
            )
        else:
            ranked_actions = [
                int(dominant_action),
                *[
                    action_id
                    for action_id in valid_actions
                    if action_id != int(dominant_action)
                ],
            ]
        return set(ranked_actions[: max(int(branch_top_k), 1)])

    def _collect_env_action_beam_search_targets(
        self,
        *,
        valid_actions: list[int],
        algorithm_memory: dict[str, Any],
        decision_info: dict[str, Any],
        run_metadata: dict[str, Any] | None,
        horizon: int,
        beam_width: int,
        source_env: Any | None = None,
    ) -> tuple[dict[str, Any], int]:
        targets: dict[int, float] = {}
        best_sequences: dict[int, list[int]] = {}
        transition_count = 0
        beam_root_env = self._env if source_env is None else source_env

        def node_score(node: dict[str, Any]) -> float:
            if bool(node["terminated"] or node["truncated"]):
                return float(node["discounted_return"])
            bootstrap_info = dict(node["next_info"])
            bootstrap_info["run_metadata"] = dict(run_metadata or {})
            bootstrap_info["algorithm_memory"] = deepcopy(node["memory"])
            bootstrap_value = self._estimate_value(
                node["observation"],
                bootstrap_info,
            )
            return float(
                node["discounted_return"]
                + node["discount"] * bootstrap_value
            )

        for initial_action in valid_actions:
            branch_env = deepcopy(beam_root_env)
            if hasattr(branch_env, "_recorder"):
                branch_env._recorder = None
            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = branch_env.step(int(initial_action))
            transition_count += 1
            branch_memory = self._advance_algorithm_memory(
                deepcopy(algorithm_memory),
                action=int(initial_action),
                reward=float(reward),
                decision_info=deepcopy(decision_info),
                next_info=next_info,
            )
            beam: list[dict[str, Any]] = [
                {
                    "env": branch_env,
                    "memory": branch_memory,
                    "observation": next_observation,
                    "next_info": dict(next_info),
                    "decision_info": dict(next_info),
                    "discounted_return": float(reward),
                    "discount": self._gamma,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "sequence": [int(initial_action)],
                }
            ]
            for _ in range(1, horizon):
                expanded: list[dict[str, Any]] = []
                for node in beam:
                    if bool(node["terminated"] or node["truncated"]):
                        expanded.append(node)
                        continue
                    branch_policy_info = dict(node["next_info"])
                    branch_policy_info["run_metadata"] = dict(
                        run_metadata or {}
                    )
                    branch_policy_info["deterministic_policy"] = True
                    branch_policy_info["algorithm_memory"] = deepcopy(
                        node["memory"]
                    )
                    _, branch_action_info = self._agent.act(
                        node["observation"],
                        branch_policy_info,
                    )
                    action_mask = list(
                        branch_action_info.get("action_mask", [])
                    )
                    branch_valid_actions = [
                        action_id
                        for action_id in range(5)
                        if not action_mask
                        or (
                            action_id < len(action_mask)
                            and bool(action_mask[action_id])
                        )
                    ]
                    for candidate_action in branch_valid_actions:
                        candidate_env = deepcopy(node["env"])
                        if hasattr(candidate_env, "_recorder"):
                            candidate_env._recorder = None
                        (
                            candidate_observation,
                            candidate_reward,
                            candidate_terminated,
                            candidate_truncated,
                            candidate_info,
                        ) = candidate_env.step(int(candidate_action))
                        transition_count += 1
                        candidate_memory = self._advance_algorithm_memory(
                            deepcopy(node["memory"]),
                            action=int(candidate_action),
                            reward=float(candidate_reward),
                            decision_info=branch_policy_info,
                            next_info=candidate_info,
                        )
                        expanded.append(
                            {
                                "env": candidate_env,
                                "memory": candidate_memory,
                                "observation": candidate_observation,
                                "next_info": dict(candidate_info),
                                "decision_info": dict(branch_policy_info),
                                "discounted_return": float(
                                    node["discounted_return"]
                                    + node["discount"]
                                    * float(candidate_reward)
                                ),
                                "discount": float(
                                    node["discount"] * self._gamma
                                ),
                                "terminated": bool(candidate_terminated),
                                "truncated": bool(candidate_truncated),
                                "sequence": [
                                    *node["sequence"],
                                    int(candidate_action),
                                ],
                            }
                        )
                beam = sorted(
                    expanded,
                    key=node_score,
                    reverse=True,
                )[:beam_width]
                if not beam or all(
                    bool(node["terminated"] or node["truncated"])
                    for node in beam
                ):
                    break
            if not beam:
                continue
            best_node = max(beam, key=node_score)
            targets[int(initial_action)] = node_score(best_node)
            best_sequences[int(initial_action)] = list(
                best_node["sequence"]
            )

        return (
            {
                "beam_search_protocol": "digital_twin_discrete_beam_search_v1",
                "beam_search_horizon": int(horizon),
                "beam_search_width": int(beam_width),
                "beam_action_td_targets": {
                    str(action_id): round(target, 6)
                    for action_id, target in targets.items()
                },
                "beam_best_sequences": {
                    str(action_id): sequence
                    for action_id, sequence in best_sequences.items()
                },
                "beam_model_query_count": int(transition_count),
            },
            transition_count,
        )

    @staticmethod
    def _algorithm_memory_has_recovery_signal(
        algorithm_memory: dict[str, Any],
    ) -> bool:
        return bool(
            int(algorithm_memory.get("failed_prepare_streak", 0) or 0) > 0
            or int(algorithm_memory.get("no_progress_streak", 0) or 0) > 0
            or bool(algorithm_memory.get("last_handoff_failed", False))
            or bool(algorithm_memory.get("last_stall", False))
        )

    def _collect_imagined_env_action_sample(
        self,
        *,
        branch_env: Any,
        observation: Any,
        decision_info: dict[str, Any],
        algorithm_memory: dict[str, Any],
        action_info: dict[str, Any],
        run_metadata: dict[str, Any] | None,
        imagination_depth: int,
        imagination_horizons: list[int],
    ) -> tuple[dict[str, Any], int]:
        action_mask = list(action_info.get("action_mask", []))
        valid_actions = [
            action_id
            for action_id in range(5)
            if not action_mask
            or (
                action_id < len(action_mask)
                and bool(action_mask[action_id])
            )
        ]
        if len(valid_actions) < 2:
            return {}, 0

        beam_search_payload: dict[str, Any] = {}
        beam_search_transition_count = 0
        if (
            bool(
                action_info.get(
                    "env_action_model_imagination_beam_search_enabled",
                    False,
                )
            )
            and self._algorithm_memory_has_recovery_signal(algorithm_memory)
        ):
            (
                beam_search_payload,
                beam_search_transition_count,
            ) = self._collect_env_action_beam_search_targets(
                valid_actions=valid_actions,
                algorithm_memory=algorithm_memory,
                decision_info=decision_info,
                run_metadata=run_metadata,
                horizon=max(
                    int(
                        action_info.get(
                            "env_action_model_beam_search_horizon",
                            4,
                        )
                    ),
                    1,
                ),
                beam_width=max(
                    int(
                        action_info.get(
                            "env_action_model_beam_search_width",
                            2,
                        )
                    ),
                    1,
                ),
                source_env=branch_env,
            )

        rollout_horizons = sorted(
            {
                max(int(horizon), 1)
                for horizon in imagination_horizons
            }
        )
        rollout_horizon = max(rollout_horizons)
        action_targets: dict[int, float] = {}
        action_targets_by_horizon: dict[int, dict[int, float]] = {
            horizon: {}
            for horizon in rollout_horizons
        }
        action_rewards: dict[int, float] = {}
        model_transition_count = 0
        for candidate_action in valid_actions:
            candidate_env = deepcopy(branch_env)
            if hasattr(candidate_env, "_recorder"):
                candidate_env._recorder = None
            candidate_memory = deepcopy(algorithm_memory)
            candidate_decision_info = deepcopy(decision_info)
            branch_action = candidate_action
            discounted_return = 0.0
            discount = 1.0
            next_observation = observation
            next_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            first_reward = 0.0
            for branch_step in range(rollout_horizon):
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    next_info,
                ) = candidate_env.step(branch_action)
                model_transition_count += 1
                if branch_step == 0:
                    first_reward = float(reward)
                discounted_return += discount * float(reward)
                discount *= self._gamma
                candidate_memory = self._advance_algorithm_memory(
                    candidate_memory,
                    action=int(branch_action),
                    reward=float(reward),
                    decision_info=candidate_decision_info,
                    next_info=next_info,
                )
                reached_horizon = branch_step + 1
                if reached_horizon in action_targets_by_horizon:
                    bootstrap_value = 0.0
                    if not terminated and not truncated:
                        bootstrap_info = dict(next_info)
                        bootstrap_info["run_metadata"] = dict(
                            run_metadata or {}
                        )
                        bootstrap_info["algorithm_memory"] = deepcopy(
                            candidate_memory
                        )
                        bootstrap_value = self._estimate_value(
                            next_observation,
                            bootstrap_info,
                        )
                    action_targets_by_horizon[reached_horizon][
                        candidate_action
                    ] = float(
                        discounted_return + discount * bootstrap_value
                    )
                if terminated or truncated:
                    for horizon in rollout_horizons:
                        if horizon > reached_horizon:
                            action_targets_by_horizon[horizon][
                                candidate_action
                            ] = float(discounted_return)
                    break
                if branch_step + 1 < rollout_horizon:
                    candidate_policy_info = dict(next_info)
                    candidate_policy_info["run_metadata"] = dict(
                        run_metadata or {}
                    )
                    candidate_policy_info["deterministic_policy"] = True
                    candidate_policy_info["algorithm_memory"] = deepcopy(
                        candidate_memory
                    )
                    candidate_decision_info = deepcopy(
                        candidate_policy_info
                    )
                    branch_action, _ = self._agent.act(
                        next_observation,
                        candidate_policy_info,
                    )
            action_rewards[candidate_action] = first_reward
            action_targets[candidate_action] = (
                action_targets_by_horizon[rollout_horizon][
                    candidate_action
                ]
            )

        imagined_rollout = {
            "protocol": "digital_twin_imagined_recovery_beam_td_v3",
            "rollout_horizon": rollout_horizon,
            "rollout_horizons": rollout_horizons,
            "imagination_depth": int(imagination_depth),
            "action_td_targets": {
                str(action_id): round(target, 6)
                for action_id, target in action_targets.items()
            },
            "action_td_targets_by_horizon": {
                str(horizon): {
                    str(action_id): round(target, 6)
                    for action_id, target in targets.items()
                }
                for horizon, targets in action_targets_by_horizon.items()
            },
            "action_immediate_rewards": {
                str(action_id): round(reward, 6)
                for action_id, reward in action_rewards.items()
            },
            **beam_search_payload,
            "unique_model_query_count": (
                len(action_targets)
                + int(beam_search_payload.get("beam_model_query_count", 0))
            ),
            "model_transition_count": (
                model_transition_count + beam_search_transition_count
            ),
        }
        return (
            {
                "decision_info": deepcopy(decision_info),
                "action_info": {
                    "action_projection": deepcopy(
                        action_info.get("action_projection", {})
                    ),
                    "env_action_model_rollout": imagined_rollout,
                },
                "imagination_depth": int(imagination_depth),
            },
            model_transition_count + beam_search_transition_count,
        )

    @staticmethod
    def _initial_algorithm_memory() -> dict[str, Any]:
        return {
            "step_index": 0,
            "last_action_id": -1,
            "same_action_streak": 0,
            "prepare_action_streak": 0,
            "failed_prepare_streak": 0,
            "no_progress_streak": 0,
            "last_reward": 0.0,
            "last_handoff_failed": False,
            "last_stall": False,
            "last_mechanism_success": False,
            "last_prefetch_expired": False,
            "last_cache_hit": False,
        }

    @staticmethod
    def _remaining_nodes(info: dict[str, Any]) -> int | None:
        semantic_state = info.get("semantic_state", {})
        if not isinstance(semantic_state, dict):
            return None
        current_node = semantic_state.get("current_workflow_node") or {}
        if isinstance(current_node, dict) and current_node.get("remaining_nodes") is not None:
            return int(current_node["remaining_nodes"])
        workflow = semantic_state.get("workflow") or {}
        if isinstance(workflow, dict):
            for key in ("remaining_nodes", "dag_remaining_nodes"):
                if workflow.get(key) is not None:
                    return int(workflow[key])
            execution_order = workflow.get("execution_order", [])
            completed_node_ids = workflow.get("completed_node_ids", [])
            if isinstance(execution_order, list) and isinstance(
                completed_node_ids,
                list,
            ):
                return max(
                    len(execution_order) - len(set(completed_node_ids)),
                    0,
                )
        if semantic_state.get("dag_remaining_nodes") is not None:
            return int(semantic_state["dag_remaining_nodes"])
        return None

    @classmethod
    def _advance_algorithm_memory(
        cls,
        memory: dict[str, Any],
        *,
        action: int,
        reward: float,
        decision_info: dict[str, Any],
        next_info: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = next_info.get("metrics_protocol", {})
        if not isinstance(metrics, dict):
            metrics = {}
        raw_previous_action = memory.get("last_action_id", -1)
        previous_action = int(
            -1 if raw_previous_action is None else raw_previous_action
        )
        same_action_streak = (
            int(memory.get("same_action_streak", 0) or 0) + 1
            if previous_action == int(action)
            else 1
        )
        prepare_action_streak = (
            int(memory.get("prepare_action_streak", 0) or 0) + 1
            if int(action) == 4
            else 0
        )
        current_remaining = cls._remaining_nodes(decision_info)
        next_remaining = cls._remaining_nodes(next_info)
        no_progress = bool(metrics.get("stall_occurred", False))
        if current_remaining is not None and next_remaining is not None:
            no_progress = no_progress or next_remaining >= current_remaining
        no_progress_streak = (
            int(memory.get("no_progress_streak", 0) or 0) + 1
            if no_progress
            else 0
        )
        mechanism_success = bool(
            metrics.get("mechanism_success_strict", False)
            or metrics.get("migration_prepare_realized", False)
            or metrics.get("handoff_ready_from_prepare", False)
        )
        prefetch_expired = str(
            metrics.get("predictive_prefetch_validation_state", "")
        ) == "expired_miss" or bool(metrics.get("prefetch_expired_miss", False))
        prepare_failed = bool(
            int(action) == 4
            and not mechanism_success
            and (
                no_progress
                or metrics.get("handoff_failed", False)
                or metrics.get("action_invalid", False)
                or prefetch_expired
            )
        )
        failed_prepare_streak = (
            int(memory.get("failed_prepare_streak", 0) or 0) + 1
            if prepare_failed
            else 0
        )
        return {
            "step_index": int(memory.get("step_index", 0) or 0) + 1,
            "last_action_id": int(action),
            "same_action_streak": same_action_streak,
            "prepare_action_streak": prepare_action_streak,
            "failed_prepare_streak": failed_prepare_streak,
            "no_progress_streak": no_progress_streak,
            "last_reward": float(reward),
            "last_handoff_failed": bool(metrics.get("handoff_failed", False)),
            "last_stall": bool(metrics.get("stall_occurred", False)),
            "last_mechanism_success": mechanism_success,
            "last_prefetch_expired": prefetch_expired,
            "last_cache_hit": bool(metrics.get("cache_hit", False)),
        }

    def _collect_option_counterfactual_targets(
        self,
        *,
        action_info: dict[str, Any],
        run_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        option_info = dict(action_info.get("option_gate", {}))
        if (
            not bool(option_info.get("enabled", False))
            or not bool(option_info.get("counterfactual_model_rollout_enabled", False))
        ):
            return {}
        option_actions_raw = option_info.get("option_actions", {})
        option_mask = list(option_info.get("option_mask", []))
        if not isinstance(option_actions_raw, dict):
            return {}

        option_actions: dict[int, int] = {}
        for raw_option, raw_action in option_actions_raw.items():
            try:
                option_index = int(raw_option)
                env_action = int(raw_action)
            except (TypeError, ValueError):
                continue
            if option_mask and (
                option_index >= len(option_mask)
                or not bool(option_mask[option_index])
            ):
                continue
            option_actions[option_index] = env_action
        if not option_actions:
            return {}

        action_targets: dict[int, float] = {}
        action_rewards: dict[int, float] = {}
        model_transition_count = 0
        rollout_horizon = max(
            int(option_info.get("counterfactual_model_rollout_horizon", 1) or 1),
            1,
        )
        for env_action in sorted(set(option_actions.values())):
            branch_env = deepcopy(self._env)
            if hasattr(branch_env, "_recorder"):
                branch_env._recorder = None
            branch_action = env_action
            discounted_return = 0.0
            discount = 1.0
            next_observation: Any = None
            next_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            first_reward = 0.0
            for branch_step in range(rollout_horizon):
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    next_info,
                ) = branch_env.step(branch_action)
                model_transition_count += 1
                if branch_step == 0:
                    first_reward = float(reward)
                discounted_return += discount * float(reward)
                discount *= self._gamma
                if terminated or truncated:
                    break
                if branch_step + 1 < rollout_horizon:
                    branch_policy_info = dict(next_info)
                    branch_policy_info["run_metadata"] = dict(run_metadata or {})
                    branch_policy_info["deterministic_policy"] = True
                    branch_action, _ = self._agent.act(
                        next_observation,
                        branch_policy_info,
                    )
            bootstrap_value = 0.0
            if not terminated and not truncated:
                bootstrap_info = dict(next_info)
                bootstrap_info["run_metadata"] = dict(run_metadata or {})
                bootstrap_value = self._estimate_value(
                    next_observation,
                    bootstrap_info,
                )
            action_rewards[env_action] = first_reward
            action_targets[env_action] = float(
                discounted_return + discount * bootstrap_value
            )

        return {
            "protocol": "digital_twin_multistep_option_td_v2",
            "rollout_horizon": rollout_horizon,
            "option_td_targets": {
                str(option_index): round(action_targets[env_action], 6)
                for option_index, env_action in option_actions.items()
            },
            "option_immediate_rewards": {
                str(option_index): round(action_rewards[env_action], 6)
                for option_index, env_action in option_actions.items()
            },
            "unique_model_query_count": len(action_targets),
            "model_transition_count": model_transition_count,
        }

    def run_episode(
        self,
        run_metadata: dict[str, Any] | None = None,
        learn: bool = True,
    ) -> dict[str, Any]:
        summary, rollout = self.collect_episode(
            run_metadata=run_metadata,
            collect_model_targets=learn,
        )
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
        prefetch_admission_guard_rows = [
            row
            for row in rollout
            if bool(
                row.get("action_info", {})
                .get("predictive_prefetch_admission_guard", {})
                .get("guarded", False)
            )
        ]
        coverage_recovery_final_guard_rows = [
            row
            for row in rollout
            if bool(
                row.get("action_info", {})
                .get("coverage_recovery_final_guard", {})
                .get("guarded", False)
            )
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
        option_gate_rows = [
            row
            for row in rollout
            if bool(row.get("action_info", {}).get("option_gate", {}).get("enabled", False))
        ]
        option_gate_applied_rows = [
            row
            for row in option_gate_rows
            if bool(row.get("action_info", {}).get("option_gate", {}).get("applied", False))
        ]
        option_label_counts = Counter(
            str(row.get("action_info", {}).get("option_gate", {}).get("option_label", "unknown"))
            for row in option_gate_rows
        )
        option_selection_reason_counts = Counter(
            str(row.get("action_info", {}).get("option_gate", {}).get("selection_reason", "unknown"))
            for row in option_gate_rows
        )
        planner_rows = [
            row
            for row in rollout
            if isinstance(
                row.get("action_info", {}).get("online_counterfactual_planner"),
                dict,
            )
        ]
        planner_enabled_rows = [
            row
            for row in planner_rows
            if bool(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("enabled", False)
            )
        ]
        planner_applied_rows = [
            row
            for row in planner_enabled_rows
            if bool(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("applied", False)
            )
        ]
        planner_candidate_counts = [
            float(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("candidate_count", 0)
                or 0
            )
            for row in planner_enabled_rows
        ]
        planner_score_margins = [
            float(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("score_margin", 0.0)
                or 0.0
            )
            for row in planner_enabled_rows
        ]
        planner_mechanism_success_counts = [
            float(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("mechanism_target_success_count", 0)
                or 0
            )
            for row in planner_enabled_rows
        ]
        planner_action_counts = Counter(
            str(
                row.get("action_info", {})
                .get("online_counterfactual_planner", {})
                .get("selected_action", row.get("action", 3))
            )
            for row in planner_enabled_rows
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
            "predictive_prefetch_admission_guard_count": len(prefetch_admission_guard_rows),
            "predictive_prefetch_admission_guard_rate": round(
                float(len(prefetch_admission_guard_rows)) / float(total_steps),
                6,
            ),
            "coverage_recovery_final_guard_count": len(coverage_recovery_final_guard_rows),
            "coverage_recovery_final_guard_rate": round(
                float(len(coverage_recovery_final_guard_rows)) / float(total_steps),
                6,
            ),
            "option_gate_enabled_count": len(option_gate_rows),
            "option_gate_enabled_rate": round(float(len(option_gate_rows)) / float(total_steps), 6),
            "option_gate_applied_count": len(option_gate_applied_rows),
            "option_gate_applied_rate": round(float(len(option_gate_applied_rows)) / float(total_steps), 6),
            "option_gate_popularity_safe_count": int(option_label_counts.get("popularity_safe", 0)),
            "option_gate_mechanism_prepare_count": int(option_label_counts.get("mechanism_prepare", 0)),
            "option_gate_no_rsu_local_count": int(option_label_counts.get("no_rsu_local", 0)),
            "option_gate_context_prior_count": int(option_selection_reason_counts.get("context_prior_margin", 0)),
            "option_gate_label_counts": dict(option_label_counts),
            "option_gate_selection_reason_counts": dict(option_selection_reason_counts),
            "online_planner_enabled_count": len(planner_enabled_rows),
            "online_planner_enabled_rate": round(
                float(len(planner_enabled_rows)) / float(total_steps),
                6,
            ),
            "online_planner_applied_count": len(planner_applied_rows),
            "online_planner_applied_rate": round(
                float(len(planner_applied_rows)) / float(total_steps),
                6,
            ),
            "online_planner_candidate_count_mean": round(
                float(sum(planner_candidate_counts))
                / float(max(len(planner_candidate_counts), 1)),
                6,
            ),
            "online_planner_score_margin_mean": round(
                float(sum(planner_score_margins))
                / float(max(len(planner_score_margins), 1)),
                6,
            ),
            "online_planner_mechanism_target_success_count": int(
                sum(planner_mechanism_success_counts)
            ),
            "online_planner_mechanism_target_success_rate": round(
                float(sum(planner_mechanism_success_counts))
                / float(max(sum(planner_candidate_counts), 1.0)),
                6,
            ),
            "online_planner_selected_action_counts": dict(planner_action_counts),
            "dag_frontier_size_mean": metric_mean("dag_frontier_size"),
            "dag_critical_path_pressure_mean": metric_mean("dag_critical_path_pressure"),
            "dag_current_node_dependency_pressure_mean": metric_mean("dag_current_node_dependency_pressure"),
            "dag_remaining_nodes_mean": metric_mean("dag_remaining_nodes"),
        }
