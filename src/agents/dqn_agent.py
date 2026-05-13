"""DQN-family baselines over the shared semantic discrete action contract."""

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


class _QNetwork(nn.Module):
    """Flat semantic encoder followed by plain or dueling action-value heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
        action_dim: int = 5,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = bool(dueling)
        self.encoder = FlatSemanticEncoder(hidden_dim=hidden_dim)
        if self.dueling:
            self.feature_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_dims[1], 1)
            self.advantage_head = nn.Linear(hidden_dims[1], action_dim)
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], action_dim),
            )

    def forward_single(self, semantic_state: dict[str, Any]) -> torch.Tensor:
        encoded = self.encoder(semantic_state)
        if self.dueling:
            features = self.feature_head(encoded["shared_embedding"].unsqueeze(0))
            advantage = self.advantage_head(features)
            value = self.value_head(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(encoded["shared_embedding"].unsqueeze(0))
        return q_values.squeeze(0)


class DQNAgent(BaseAgent):
    """Experience-replay DQN baseline for the current five-way action space."""

    observation_contract = "flat_semantic_encoder_v1"
    action_contract = "semantic_discrete_5"
    support_level = "trainable"

    def __init__(
        self,
        *,
        agent_name: str = "dqn",
        policy_type: str = "dqn_policy",
        double_dqn: bool = False,
        dueling: bool = False,
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
        max_grad_norm: float = 1.0,
        deterministic_action: bool = False,
        random_seed: int = 7,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        super().__init__(agent_name=agent_name)
        self.policy_type = policy_type
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
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
        self._max_grad_norm = float(max_grad_norm)
        self._deterministic_action = bool(deterministic_action)
        self._device = torch.device(device)
        self._update_count = 0
        self._rng = random.Random(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self._network = _QNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
            action_dim=5,
            dueling=self.dueling,
        ).to(self._device)
        self._target_network = _QNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
            action_dim=5,
            dueling=self.dueling,
        ).to(self._device)
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_network.eval()
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self._replay: deque[dict[str, Any]] = deque(maxlen=self._replay_capacity)

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
            q_values = self._network.forward_single(semantic_state)
            masked_q_values = self._mask_values(q_values, action_mask)
            if not deterministic and self._rng.random() < epsilon:
                action = self._sample_valid_action(action_mask)
                policy_mode = "epsilon_sample"
            else:
                action = int(torch.argmax(masked_q_values, dim=-1).item())
                policy_mode = "greedy"
            log_probs = torch.log_softmax(masked_q_values, dim=-1)
            probs = torch.softmax(masked_q_values, dim=-1)
            selected_log_prob = float(log_probs[action].item())
            selected_q = float(q_values[action].item())
            max_q = float(torch.max(masked_q_values).item())
        return action, {
            "policy_mode": policy_mode,
            "policy_type": self.policy_type,
            "encoder_mode": "flat_baseline",
            "critic_mode": "q_value",
            "q_architecture": "dueling" if self.dueling else "plain",
            "action_mask": list(action_mask) if action_mask is not None else None,
            "action_mask_applied": bool(self._action_mask_has_valid_action(action_mask)),
            "valid_action_count": self._valid_action_count(action_mask),
            "head_actions": {"flat": int(action)},
            "head_action_labels": {"flat": f"flat_action_{int(action)}"},
            "aggregation_reason": "flat_q_argmax",
            "log_prob": round(selected_log_prob, 6),
            "value": round(max_q, 6),
            "q_value": round(selected_q, 6),
            "epsilon": round(float(epsilon), 6),
            "action_probs": {"flat": [round(float(item), 6) for item in probs.tolist()]},
            "q_values": [round(float(item), 6) for item in q_values.tolist()],
        }

    def evaluate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        del observation
        semantic_state = self._extract_semantic_state(info)
        action_mask = self._extract_action_mask(info)
        with torch.no_grad():
            q_values = self._network.forward_single(semantic_state)
            masked_q_values = self._mask_values(q_values, action_mask)
        return float(torch.max(masked_q_values).item())

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rollout:
            current_state = self._extract_semantic_state(row.get("decision_info"))
            next_state = row.get("env_info", {}).get("semantic_state")
            self._replay.append(
                {
                    "state": current_state,
                    "action": int(row.get("action", 0)),
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
            states = [item["state"] for item in batch]
            actions = torch.as_tensor([int(item["action"]) for item in batch], dtype=torch.long, device=self._device)
            rewards = torch.as_tensor([float(item["reward"]) for item in batch], dtype=torch.float32, device=self._device)
            terminated = torch.as_tensor([float(item["terminated"]) for item in batch], dtype=torch.float32, device=self._device)
            q_values = torch.stack([self._network.forward_single(state) for state in states], dim=0)
            selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            target_q_values: list[torch.Tensor] = []
            with torch.no_grad():
                for item in batch:
                    next_state = item.get("next_state")
                    if not isinstance(next_state, dict) or not next_state.get("current_workflow_node"):
                        target_q_values.append(torch.tensor(0.0, dtype=torch.float32, device=self._device))
                        continue
                    if self.double_dqn:
                        online_next = self._mask_values(
                            self._network.forward_single(next_state),
                            item.get("next_action_mask"),
                        )
                        next_action = int(torch.argmax(online_next, dim=-1).item())
                        target_next = self._target_network.forward_single(next_state)
                        target_q_values.append(target_next[next_action])
                    else:
                        target_next = self._mask_values(
                            self._target_network.forward_single(next_state),
                            item.get("next_action_mask"),
                        )
                        target_q_values.append(torch.max(target_next))
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
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
            "epsilon": round(self._current_epsilon(), 6),
            "q_loss": round(loss_total / denominator, 6),
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
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
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

    def _mask_values(self, values: torch.Tensor, action_mask: list[bool] | None) -> torch.Tensor:
        if not self._action_mask_has_valid_action(action_mask):
            return values
        assert action_mask is not None
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=values.device)
        if mask_tensor.numel() != values.shape[-1]:
            return values
        return values.masked_fill(~mask_tensor, -1.0e9)

    def _sample_valid_action(self, action_mask: list[bool] | None) -> int:
        if not self._action_mask_has_valid_action(action_mask):
            return self._rng.randrange(5)
        assert action_mask is not None
        valid_actions = [index for index, is_valid in enumerate(action_mask) if bool(is_valid)]
        return int(self._rng.choice(valid_actions))


class DDQNAgent(DQNAgent):
    """Double-DQN variant with the same observation/action contract."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("agent_name", None)
        kwargs.pop("policy_type", None)
        kwargs.pop("double_dqn", None)
        kwargs.pop("dueling", None)
        super().__init__(
            agent_name="ddqn",
            policy_type="ddqn_policy",
            double_dqn=True,
            dueling=False,
            **kwargs,
        )


class DuelingDQNAgent(DQNAgent):
    """Dueling-DQN variant with value and advantage streams."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("agent_name", None)
        kwargs.pop("policy_type", None)
        kwargs.pop("double_dqn", None)
        kwargs.pop("dueling", None)
        super().__init__(
            agent_name="dueling_dqn",
            policy_type="dueling_dqn_policy",
            double_dqn=False,
            dueling=True,
            **kwargs,
        )


class DuelingDDQNAgent(DQNAgent):
    """Dueling Double-DQN variant for the current discrete action contract."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("agent_name", None)
        kwargs.pop("policy_type", None)
        kwargs.pop("double_dqn", None)
        kwargs.pop("dueling", None)
        super().__init__(
            agent_name="dueling_ddqn",
            policy_type="dueling_ddqn_policy",
            double_dqn=True,
            dueling=True,
            **kwargs,
        )
