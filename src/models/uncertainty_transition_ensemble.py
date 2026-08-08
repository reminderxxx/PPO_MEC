"""Bootstrap ensemble for conservative action-conditioned transition/Q prediction.

The model is trained only from environment rollout rows. It predicts the
immediate reward, the normalized next observation, and a one-step TD target
``r_t + gamma * V(s_{t+1})`` supplied by the current MAPPO critic. The TD
head is used for action improvement while the transition head is retained for
diagnostics and future multi-step extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class EnsembleFitStats:
    sample_count: int
    train_count: int
    validation_count: int
    train_loss: float
    validation_rmse: float
    mean_predictive_std: float
    uncertainty_scale: float


class _TransitionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class UncertaintyTransitionEnsemble:
    """Small bootstrap ensemble with a stable checkpoint contract."""

    def __init__(
        self,
        *,
        observation_dim: int = 9,
        action_count: int = 5,
        ensemble_size: int = 5,
        hidden_dim: int = 64,
        learning_rate: float = 3e-3,
        fit_epochs: int = 4,
        max_samples: int = 4096,
        min_samples: int = 64,
        discount: float = 0.99,
        random_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        self.observation_dim = max(int(observation_dim), 1)
        self.action_count = max(int(action_count), 1)
        self.ensemble_size = max(int(ensemble_size), 2)
        self.hidden_dim = max(int(hidden_dim), 8)
        self.learning_rate = max(float(learning_rate), 1e-6)
        self.fit_epochs = max(int(fit_epochs), 1)
        self.min_samples = max(int(min_samples), 1)
        self.max_samples = max(int(max_samples), self.min_required_samples)
        self.discount = float(np.clip(float(discount), 0.0, 1.0))
        self.random_seed = int(random_seed)
        self.device = torch.device(device)
        self.input_dim = self.observation_dim + self.action_count
        self.output_dim = self.observation_dim + 2
        self._rng = np.random.default_rng(self.random_seed)
        self._models = nn.ModuleList(
            [
                _TransitionHead(self.input_dim, self.output_dim, self.hidden_dim)
                for _ in range(self.ensemble_size)
            ]
        ).to(self.device)
        self._optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self._models
        ]
        self.sample_count = 0
        self.update_count = 0
        self.uncertainty_scale = 1.0
        self.last_fit_stats: dict[str, Any] = {}
        self._replay_inputs: np.ndarray | None = None
        self._replay_targets: np.ndarray | None = None

    @property
    def min_required_samples(self) -> int:
        return max(self.min_samples, self.ensemble_size * 4)

    @property
    def ready(self) -> bool:
        return self.sample_count >= self.min_required_samples

    def fit(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        arrays = self._rows_to_arrays(rollout)
        if arrays is None:
            return {
                "enabled": True,
                "ready": self.ready,
                "sample_count": self.sample_count,
                "skipped": True,
                "reason": "insufficient_valid_rows",
            }
        inputs, targets = arrays
        if self._replay_inputs is None:
            replay_inputs = inputs
            replay_targets = targets
        else:
            replay_inputs = np.concatenate([self._replay_inputs, inputs], axis=0)
            replay_targets = np.concatenate([self._replay_targets, targets], axis=0)
        if replay_inputs.shape[0] > self.max_samples:
            selected = self._rng.choice(
                replay_inputs.shape[0], self.max_samples, replace=False
            )
            replay_inputs = replay_inputs[selected]
            replay_targets = replay_targets[selected]
        self._replay_inputs = np.asarray(replay_inputs, dtype=np.float32)
        self._replay_targets = np.asarray(replay_targets, dtype=np.float32)
        inputs = self._replay_inputs
        targets = self._replay_targets
        sample_count = int(inputs.shape[0])
        self.sample_count = sample_count
        validation_count = max(1, int(round(0.20 * sample_count))) if sample_count >= 10 else 0
        if validation_count:
            train_inputs = inputs[:-validation_count]
            train_targets = targets[:-validation_count]
            validation_inputs = inputs[-validation_count:]
            validation_targets = targets[-validation_count:]
        else:
            train_inputs = inputs
            train_targets = targets
            validation_inputs = inputs[:0]
            validation_targets = targets[:0]

        input_tensor = torch.as_tensor(train_inputs, dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(train_targets, dtype=torch.float32, device=self.device)
        member_losses: list[float] = []
        for member_index, (model, optimizer) in enumerate(zip(self._models, self._optimizers)):
            model.train()
            bootstrap_indices = self._rng.integers(
                0,
                max(len(train_inputs), 1),
                size=max(len(train_inputs), 1),
            )
            member_inputs = input_tensor[bootstrap_indices]
            member_targets = target_tensor[bootstrap_indices]
            for _ in range(self.fit_epochs):
                prediction = model(member_inputs)
                loss = nn.functional.smooth_l1_loss(prediction, member_targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            member_losses.append(float(loss.detach().cpu().item()))

        validation_rmse = 0.0
        mean_predictive_std = 0.0
        if validation_count:
            predictions = self._predict_array(validation_inputs)
            validation_mean = predictions.mean(axis=0)
            validation_rmse = float(np.sqrt(np.mean((validation_mean - validation_targets) ** 2)))
            mean_predictive_std = float(predictions.std(axis=0).mean())
            return_std = float(predictions[:, :, -1].std(axis=0).mean())
            return_error = float(
                np.abs(validation_mean[:, -1] - validation_targets[:, -1]).mean()
            )
            if return_std > 1e-6:
                self.uncertainty_scale = float(
                    np.clip(0.5 * self.uncertainty_scale + 0.5 * return_error / return_std, 0.5, 3.0)
                )
        self.update_count += 1
        stats = EnsembleFitStats(
            sample_count=sample_count,
            train_count=int(len(train_inputs)),
            validation_count=validation_count,
            train_loss=float(np.mean(member_losses)) if member_losses else 0.0,
            validation_rmse=validation_rmse,
            mean_predictive_std=mean_predictive_std,
            uncertainty_scale=float(self.uncertainty_scale),
        )
        self.last_fit_stats = {
            "enabled": True,
            "ready": self.ready,
            "skipped": False,
            "sample_count": stats.sample_count,
            "train_count": stats.train_count,
            "validation_count": stats.validation_count,
            "train_loss": round(stats.train_loss, 6),
            "validation_rmse": round(stats.validation_rmse, 6),
            "mean_predictive_std": round(stats.mean_predictive_std, 6),
            "uncertainty_scale": round(stats.uncertainty_scale, 6),
            "update_count": self.update_count,
        }
        return dict(self.last_fit_stats)

    def predict(self, observation: Any, action_ids: list[int]) -> dict[str, Any]:
        if not self.ready or not action_ids:
            return {"ready": False, "reason": "model_not_ready"}
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.size != self.observation_dim:
            return {"ready": False, "reason": "observation_dimension_mismatch"}
        inputs = []
        for action_id in action_ids:
            action_one_hot = np.zeros(self.action_count, dtype=np.float32)
            if 0 <= int(action_id) < self.action_count:
                action_one_hot[int(action_id)] = 1.0
            inputs.append(np.concatenate([observation_array, action_one_hot]))
        prediction_array = self._predict_array(np.asarray(inputs, dtype=np.float32))
        means = prediction_array.mean(axis=0)
        stds = prediction_array.std(axis=0)
        return {
            "ready": True,
            "action_ids": [int(action_id) for action_id in action_ids],
            "reward_mean": means[:, 0].astype(float).tolist(),
            "reward_std": (stds[:, 0] * self.uncertainty_scale).astype(float).tolist(),
            "next_observation_mean": means[:, 1 : 1 + self.observation_dim].astype(float).tolist(),
            "next_observation_std": stds[:, 1 : 1 + self.observation_dim].astype(float).tolist(),
            "td_target_mean": means[:, -1].astype(float).tolist(),
            "td_target_std": (stds[:, -1] * self.uncertainty_scale).astype(float).tolist(),
            "uncertainty_scale": float(self.uncertainty_scale),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "observation_dim": self.observation_dim,
                "action_count": self.action_count,
                "ensemble_size": self.ensemble_size,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "fit_epochs": self.fit_epochs,
                "max_samples": self.max_samples,
                "min_samples": self.min_samples,
                "discount": self.discount,
                "random_seed": self.random_seed,
            },
            "models": [model.state_dict() for model in self._models],
            "optimizers": [optimizer.state_dict() for optimizer in self._optimizers],
            "sample_count": self.sample_count,
            "update_count": self.update_count,
            "uncertainty_scale": self.uncertainty_scale,
            "last_fit_stats": dict(self.last_fit_stats),
            "replay_inputs": (
                self._replay_inputs.tolist()
                if self._replay_inputs is not None
                else None
            ),
            "replay_targets": (
                self._replay_targets.tolist()
                if self._replay_targets is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for model, model_state in zip(self._models, state.get("models", [])):
            model.load_state_dict(model_state)
        for optimizer, optimizer_state in zip(self._optimizers, state.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        self.sample_count = int(state.get("sample_count", 0))
        self.update_count = int(state.get("update_count", 0))
        self.uncertainty_scale = float(state.get("uncertainty_scale", 1.0))
        self.last_fit_stats = dict(state.get("last_fit_stats", {}))
        replay_inputs = state.get("replay_inputs")
        replay_targets = state.get("replay_targets")
        self._replay_inputs = (
            np.asarray(replay_inputs, dtype=np.float32)
            if replay_inputs is not None
            else None
        )
        self._replay_targets = (
            np.asarray(replay_targets, dtype=np.float32)
            if replay_targets is not None
            else None
        )

    def _predict_array(self, inputs: np.ndarray) -> np.ndarray:
        input_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        predictions: list[np.ndarray] = []
        for model in self._models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(input_tensor).cpu().numpy())
        return np.stack(predictions, axis=0)

    def _rows_to_arrays(
        self,
        rollout: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for row in rollout:
            try:
                observation = np.asarray(row.get("observation"), dtype=np.float32).reshape(-1)
                next_observation = np.asarray(row.get("next_observation"), dtype=np.float32).reshape(-1)
                action_id = int(row.get("action"))
                reward = float(row.get("reward"))
                next_value = float(row.get("next_value", 0.0))
                terminated = bool(row.get("terminated", False))
                td_target = reward + self.discount * next_value * (0.0 if terminated else 1.0)
            except (TypeError, ValueError):
                continue
            if (
                observation.size != self.observation_dim
                or next_observation.size != self.observation_dim
                or not 0 <= action_id < self.action_count
                or not np.isfinite(observation).all()
                or not np.isfinite(next_observation).all()
                or not np.isfinite([reward, next_value, td_target]).all()
            ):
                continue
            action_one_hot = np.zeros(self.action_count, dtype=np.float32)
            action_one_hot[action_id] = 1.0
            inputs.append(np.concatenate([observation, action_one_hot]))
            targets.append(np.concatenate([[reward], next_observation, [td_target]]))
        if len(inputs) < self.min_required_samples:
            return None
        return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)
