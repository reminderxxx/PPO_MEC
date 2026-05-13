"""Controller-level QMIX baseline for the current discrete VEC contract."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.agents.base_agent import BaseAgent
from src.encoders import FlatSemanticEncoder


CONTROLLER_ACTION_DIMS = {"slow": 3, "fast": 2, "event": 2}
CONTROLLER_NAMES = ("slow", "fast", "event")


def _aggregate_controller_actions(head_actions: dict[str, int]) -> tuple[int, str]:
    slow_action = int(head_actions.get("slow", 0))
    fast_action = int(head_actions.get("fast", 0))
    event_action = int(head_actions.get("event", 0))
    if event_action == 1:
        return 4, "event_head_prepare"
    if slow_action == 2:
        return 1, "slow_head_prefetch"
    if slow_action == 1:
        return 0, "slow_head_cache_fill"
    if fast_action == 1:
        return 2, "fast_head_vehicle_fallback"
    return 3, "fast_head_steady_offload"


def _canonical_heads_for_env_action(env_action: int) -> dict[str, int]:
    if int(env_action) == 0:
        return {"slow": 1, "fast": 0, "event": 0}
    if int(env_action) == 1:
        return {"slow": 2, "fast": 0, "event": 0}
    if int(env_action) == 2:
        return {"slow": 0, "fast": 1, "event": 0}
    if int(env_action) == 4:
        return {"slow": 0, "fast": 0, "event": 1}
    return {"slow": 0, "fast": 0, "event": 0}


def _head_action_labels(head_actions: dict[str, int]) -> dict[str, str]:
    slow_labels = ["no_cache_change", "current_rsu_cache_fill", "predictive_next_rsu_prefetch"]
    fast_labels = ["current_rsu_offload", "vehicle_fallback"]
    event_labels = ["keep", "handoff_prepare"]
    return {
        "slow": slow_labels[max(0, min(int(head_actions.get("slow", 0)), len(slow_labels) - 1))],
        "fast": fast_labels[max(0, min(int(head_actions.get("fast", 0)), len(fast_labels) - 1))],
        "event": event_labels[max(0, min(int(head_actions.get("event", 0)), len(event_labels) - 1))],
    }


class _QMIXNetwork(nn.Module):
    """Flat semantic encoder, per-controller Q heads, and monotonic mixer."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
        mixer_hidden_dim: int = 32,
        mixing: str = "qmix",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mixer_hidden_dim = int(mixer_hidden_dim)
        self.mixing = str(mixing)
        self.encoder = FlatSemanticEncoder(hidden_dim=self.hidden_dim)
        self.slow_head = self._make_head(self.hidden_dim, CONTROLLER_ACTION_DIMS["slow"], hidden_dims)
        self.fast_head = self._make_head(self.hidden_dim, CONTROLLER_ACTION_DIMS["fast"], hidden_dims)
        self.event_head = self._make_head(self.hidden_dim, CONTROLLER_ACTION_DIMS["event"], hidden_dims)
        if self.mixing == "qmix":
            self.hyper_w1 = nn.Linear(self.hidden_dim, len(CONTROLLER_NAMES) * self.mixer_hidden_dim)
            self.hyper_b1 = nn.Linear(self.hidden_dim, self.mixer_hidden_dim)
            self.hyper_w2 = nn.Linear(self.hidden_dim, self.mixer_hidden_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.mixer_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.mixer_hidden_dim, 1),
            )

    @staticmethod
    def _make_head(input_dim: int, output_dim: int, hidden_dims: tuple[int, int]) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward_single(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        encoded = self.encoder(semantic_state)
        return {
            "encoded": encoded,
            "slow_q": self.slow_head(encoded["slow_context"].unsqueeze(0)).squeeze(0),
            "fast_q": self.fast_head(encoded["fast_context"].unsqueeze(0)).squeeze(0),
            "event_q": self.event_head(encoded["event_context"].unsqueeze(0)).squeeze(0),
            "state_context": encoded["centralized_critic_context"],
            "critic_mode": "centralized_mixer",
            "critic_context_key": "centralized_critic_context",
            "mixer": self.mixing,
        }

    def mix(self, q_values: torch.Tensor, state_context: torch.Tensor) -> torch.Tensor:
        if self.mixing == "vdn":
            return torch.sum(q_values, dim=-1)
        batch_size = q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state_context)).view(batch_size, len(CONTROLLER_NAMES), self.mixer_hidden_dim)
        b1 = self.hyper_b1(state_context).view(batch_size, 1, self.mixer_hidden_dim)
        hidden = torch.nn.functional.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state_context)).view(batch_size, self.mixer_hidden_dim, 1)
        b2 = self.hyper_b2(state_context).view(batch_size, 1, 1)
        mixed = torch.bmm(hidden, w2) + b2
        return mixed.view(batch_size)


class QMIXAgent(BaseAgent):
    """Controller-level QMIX over cache, execution/offload, and handoff-event controllers."""

    observation_contract = "flat_semantic_multi_controller_ctde_v1"
    action_contract = "semantic_discrete_5_multi_controller_3head_qmix"
    support_level = "trainable"

    def __init__(
        self,
        *,
        agent_name: str = "qmix",
        policy_type: str = "qmix_policy",
        mixing: str = "qmix",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        train_epochs: int = 4,
        replay_capacity: int = 4096,
        min_replay_size: int = 16,
        target_update_interval: int = 4,
        epsilon_start: float = 0.25,
        epsilon_final: float = 0.02,
        epsilon_decay_updates: int = 32,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
        mixer_hidden_dim: int = 32,
        double_qmix: bool = True,
        max_grad_norm: float = 1.0,
        deterministic_action: bool = False,
        random_seed: int = 7,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        super().__init__(agent_name=agent_name)
        self.policy_type = policy_type
        self.mixing = str(mixing)
        self.double_qmix = bool(double_qmix)
        self._learning_rate = float(learning_rate)
        self._gamma = float(gamma)
        self._batch_size = int(batch_size)
        self._train_epochs = int(train_epochs)
        self._replay_capacity = int(replay_capacity)
        self._min_replay_size = int(min_replay_size)
        self._target_update_interval = max(int(target_update_interval), 1)
        self._epsilon_start = max(0.0, min(float(epsilon_start), 1.0))
        self._epsilon_final = max(0.0, min(float(epsilon_final), 1.0))
        self._epsilon_decay_updates = max(int(epsilon_decay_updates), 1)
        self._hidden_dim = int(hidden_dim)
        self._hidden_dims = tuple(hidden_dims)
        self._mixer_hidden_dim = int(mixer_hidden_dim)
        self._max_grad_norm = float(max_grad_norm)
        self._deterministic_action = bool(deterministic_action)
        self._device = torch.device(device)
        self._update_count = 0
        self._rng = random.Random(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self._network = _QMIXNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
            mixer_hidden_dim=self._mixer_hidden_dim,
            mixing=self.mixing,
        ).to(self._device)
        self._target_network = _QMIXNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
            mixer_hidden_dim=self._mixer_hidden_dim,
            mixing=self.mixing,
        ).to(self._device)
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_network.eval()
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self._replay: deque[dict[str, Any]] = deque(maxlen=self._replay_capacity)
        self.baseline_config = {
            "family": "qmix",
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "centralized_mixer": True,
            "mixer": self.mixing,
            "vehicle_or_rsu_agent_ctde": False,
            "paper_grade_independent_baseline": True,
            "excluded_sa_mechanisms": [
                "graph_encoder",
                "surrogate_prediction_features",
                "uncertainty_signal",
                "dag_dependency_aware_features",
                "mechanism_auxiliary_loss",
                "heuristic_imitation",
                "continuity_guard",
                "backhaul_guard",
                "cache_warm_start_guard",
            ],
        }

    def act(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = self._extract_semantic_state(info)
        action_mask = self._extract_action_mask(info)
        deterministic = bool(self._deterministic_action or (info or {}).get("deterministic_policy", False))
        epsilon = 0.0 if deterministic else self._current_epsilon()
        with torch.no_grad():
            output = self._network.forward_single(semantic_state)
            if not deterministic and self._rng.random() < epsilon:
                env_action = self._sample_valid_env_action(action_mask)
                head_actions = _canonical_heads_for_env_action(env_action)
                joint_q = self._joint_q_for_heads(self._network, output, head_actions)
                policy_mode = "epsilon_sample"
                aggregation_reason = f"epsilon_env_action_{env_action}"
            else:
                head_actions, env_action, aggregation_reason, joint_q = self._best_joint_action_from_output(
                    self._network,
                    output,
                    action_mask,
                )
                policy_mode = "greedy_joint_q"
        return env_action, {
            "policy_mode": policy_mode,
            "policy_type": self.policy_type,
            "encoder_mode": "flat_baseline",
            "critic_mode": "centralized_mixer",
            "critic_context_key": "centralized_critic_context",
            "mixer": self.mixing,
            "action_mask": list(action_mask) if action_mask is not None else None,
            "action_mask_applied": bool(self._action_mask_has_valid_action(action_mask)),
            "valid_action_count": self._valid_action_count(action_mask),
            "head_actions": dict(head_actions),
            "head_action_labels": _head_action_labels(head_actions),
            "aggregation_reason": aggregation_reason,
            "log_prob": 0.0,
            "value": round(float(joint_q.item()), 6),
            "joint_q_value": round(float(joint_q.item()), 6),
            "epsilon": round(float(epsilon), 6),
            "action_probs": {
                "slow": self._softmax_list(output["slow_q"]),
                "fast": self._softmax_list(output["fast_q"]),
                "event": self._softmax_list(output["event_q"]),
            },
            "q_values": {
                "slow": self._round_list(output["slow_q"]),
                "fast": self._round_list(output["fast_q"]),
                "event": self._round_list(output["event_q"]),
            },
        }

    def evaluate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        del observation
        semantic_state = self._extract_semantic_state(info)
        action_mask = self._extract_action_mask(info)
        with torch.no_grad():
            output = self._network.forward_single(semantic_state)
            _, _, _, joint_q = self._best_joint_action_from_output(self._network, output, action_mask)
        return float(joint_q.item())

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rollout:
            current_state = self._extract_semantic_state(row.get("decision_info"))
            action_info = row.get("action_info") or {}
            head_actions = action_info.get("head_actions")
            if not isinstance(head_actions, dict) or not all(name in head_actions for name in CONTROLLER_NAMES):
                head_actions = _canonical_heads_for_env_action(int(row.get("action", 0)))
            next_state = row.get("env_info", {}).get("semantic_state")
            self._replay.append(
                {
                    "state": current_state,
                    "head_actions": {name: int(head_actions.get(name, 0)) for name in CONTROLLER_NAMES},
                    "reward": float(row.get("reward", 0.0)),
                    "terminated": bool(row.get("terminated", False)),
                    "next_state": next_state if isinstance(next_state, dict) else None,
                    "action_mask": self._extract_action_mask(row.get("decision_info")),
                    "next_action_mask": self._extract_action_mask(row.get("env_info")),
                }
            )
        if len(self._replay) < max(1, self._min_replay_size):
            return {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "policy_update_skipped": True,
                "reason": "replay_warmup",
                "collected_steps": len(rollout),
                "replay_size": len(self._replay),
                "update_count": self._update_count,
            }

        batch_size = max(1, min(self._batch_size, len(self._replay)))
        loss_total = 0.0
        td_error_total = 0.0
        update_steps = 0
        for _ in range(max(self._train_epochs, 1)):
            batch = self._rng.sample(list(self._replay), batch_size)
            rewards = torch.as_tensor([float(item["reward"]) for item in batch], dtype=torch.float32, device=self._device)
            terminated = torch.as_tensor([float(item["terminated"]) for item in batch], dtype=torch.float32, device=self._device)
            selected_q_values: list[torch.Tensor] = []
            target_q_values: list[torch.Tensor] = []
            for item in batch:
                output = self._network.forward_single(item["state"])
                selected_q_values.append(self._joint_q_for_heads(self._network, output, item["head_actions"]))
                with torch.no_grad():
                    next_state = item.get("next_state")
                    if not isinstance(next_state, dict) or not next_state.get("current_workflow_node"):
                        target_q_values.append(torch.tensor(0.0, dtype=torch.float32, device=self._device))
                        continue
                    if self.double_qmix:
                        online_next = self._network.forward_single(next_state)
                        next_heads, _, _, _ = self._best_joint_action_from_output(
                            self._network,
                            online_next,
                            item.get("next_action_mask"),
                        )
                        target_next = self._target_network.forward_single(next_state)
                        target_q_values.append(self._joint_q_for_heads(self._target_network, target_next, next_heads))
                    else:
                        target_next = self._target_network.forward_single(next_state)
                        _, _, _, target_joint_q = self._best_joint_action_from_output(
                            self._target_network,
                            target_next,
                            item.get("next_action_mask"),
                        )
                        target_q_values.append(target_joint_q)
            selected_q = torch.stack(selected_q_values, dim=0)
            next_q = torch.stack(target_q_values, dim=0)
            targets = rewards + self._gamma * (1.0 - terminated) * next_q
            td_error = targets - selected_q
            loss = nn.functional.smooth_l1_loss(selected_q, targets)
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=self._max_grad_norm)
            self._optimizer.step()
            loss_total += float(loss.item())
            td_error_total += float(torch.mean(torch.abs(td_error)).item())
            update_steps += 1

        self._update_count += 1
        if self._update_count % self._target_update_interval == 0:
            self._target_network.load_state_dict(self._network.state_dict())
        denominator = max(update_steps, 1)
        return {
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "policy_update_skipped": False,
            "update_count": self._update_count,
            "collected_steps": len(rollout),
            "replay_size": len(self._replay),
            "batch_size": batch_size,
            "train_epochs": self._train_epochs,
            "double_qmix": self.double_qmix,
            "mixer": self.mixing,
            "epsilon": round(self._current_epsilon(), 6),
            "qmix_loss": round(loss_total / denominator, 6),
            "mean_abs_td_error": round(td_error_total / denominator, 6),
            "target_update_interval": self._target_update_interval,
        }

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "update_count": self._update_count,
            "config": self._checkpoint_config(),
            "network_state_dict": self._network.state_dict(),
            "target_network_state_dict": self._target_network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        torch.save(checkpoint, output_path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(Path(path), map_location=self._device)
        self._network.load_state_dict(checkpoint["network_state_dict"])
        target_state = checkpoint.get("target_network_state_dict") or checkpoint.get("network_state_dict")
        self._target_network.load_state_dict(target_state)
        if checkpoint.get("optimizer_state_dict") is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._update_count = int(checkpoint.get("update_count", 0))

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "mixing": self.mixing,
            "double_qmix": self.double_qmix,
            "learning_rate": self._learning_rate,
            "gamma": self._gamma,
            "batch_size": self._batch_size,
            "train_epochs": self._train_epochs,
            "replay_capacity": self._replay_capacity,
            "min_replay_size": self._min_replay_size,
            "target_update_interval": self._target_update_interval,
            "epsilon_start": self._epsilon_start,
            "epsilon_final": self._epsilon_final,
            "epsilon_decay_updates": self._epsilon_decay_updates,
            "hidden_dim": self._hidden_dim,
            "hidden_dims": self._hidden_dims,
            "mixer_hidden_dim": self._mixer_hidden_dim,
            "max_grad_norm": self._max_grad_norm,
            "observation_contract": self.observation_contract,
            "action_contract": self.action_contract,
        }

    def _current_epsilon(self) -> float:
        progress = min(float(self._update_count) / float(self._epsilon_decay_updates), 1.0)
        return self._epsilon_start + progress * (self._epsilon_final - self._epsilon_start)

    def _extract_semantic_state(self, info: dict[str, Any] | None) -> dict[str, Any]:
        semantic_state = (info or {}).get("semantic_state")
        if semantic_state is None:
            raise ValueError(f"{self.agent_name} requires info['semantic_state'] for flat semantic encoding.")
        return semantic_state

    def _extract_action_mask(self, info: dict[str, Any] | None) -> list[bool] | None:
        raw_mask = (info or {}).get("action_mask")
        if raw_mask is None or not isinstance(raw_mask, (list, tuple)):
            return None
        normalized = [bool(item) for item in raw_mask[:5]]
        if len(normalized) < 5:
            normalized.extend([True for _ in range(5 - len(normalized))])
        return normalized

    def _action_mask_has_valid_action(self, action_mask: list[bool] | None) -> bool:
        return bool(action_mask and any(bool(item) for item in action_mask))

    def _valid_action_count(self, action_mask: list[bool] | None) -> int:
        if action_mask is None:
            return 5
        return int(sum(1 for item in action_mask if bool(item)))

    def _is_env_action_valid(self, env_action: int, action_mask: list[bool] | None) -> bool:
        if not self._action_mask_has_valid_action(action_mask):
            return True
        assert action_mask is not None
        return bool(0 <= int(env_action) < len(action_mask) and action_mask[int(env_action)])

    def _sample_valid_env_action(self, action_mask: list[bool] | None) -> int:
        if not self._action_mask_has_valid_action(action_mask):
            return self._rng.randrange(5)
        assert action_mask is not None
        valid_actions = [index for index, is_valid in enumerate(action_mask) if bool(is_valid)]
        return int(self._rng.choice(valid_actions))

    def _candidate_head_actions(self) -> list[dict[str, int]]:
        return [
            {"slow": slow, "fast": fast, "event": event}
            for slow in range(CONTROLLER_ACTION_DIMS["slow"])
            for fast in range(CONTROLLER_ACTION_DIMS["fast"])
            for event in range(CONTROLLER_ACTION_DIMS["event"])
        ]

    def _joint_q_for_heads(
        self,
        network: _QMIXNetwork,
        output: dict[str, Any],
        head_actions: dict[str, int],
    ) -> torch.Tensor:
        controller_qs = torch.stack(
            [
                output["slow_q"][int(head_actions.get("slow", 0))],
                output["fast_q"][int(head_actions.get("fast", 0))],
                output["event_q"][int(head_actions.get("event", 0))],
            ],
            dim=0,
        ).unsqueeze(0)
        return network.mix(controller_qs, output["state_context"].unsqueeze(0)).squeeze(0)

    def _best_joint_action_from_output(
        self,
        network: _QMIXNetwork,
        output: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> tuple[dict[str, int], int, str, torch.Tensor]:
        best_heads: dict[str, int] | None = None
        best_env_action = 3
        best_reason = "fast_head_steady_offload"
        best_q: torch.Tensor | None = None
        for head_actions in self._candidate_head_actions():
            env_action, reason = _aggregate_controller_actions(head_actions)
            if not self._is_env_action_valid(env_action, action_mask):
                continue
            joint_q = self._joint_q_for_heads(network, output, head_actions)
            if best_q is None or float(joint_q.item()) > float(best_q.item()):
                best_heads = dict(head_actions)
                best_env_action = int(env_action)
                best_reason = reason
                best_q = joint_q
        if best_heads is None or best_q is None:
            fallback_action = self._sample_valid_env_action(action_mask)
            best_heads = _canonical_heads_for_env_action(fallback_action)
            best_env_action, best_reason = _aggregate_controller_actions(best_heads)
            best_q = self._joint_q_for_heads(network, output, best_heads)
        return best_heads, best_env_action, best_reason, best_q

    @staticmethod
    def _round_list(values: torch.Tensor) -> list[float]:
        return [round(float(item), 6) for item in values.detach().cpu().tolist()]

    @staticmethod
    def _softmax_list(values: torch.Tensor) -> list[float]:
        return [round(float(item), 6) for item in torch.softmax(values, dim=-1).detach().cpu().tolist()]
