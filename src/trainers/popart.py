"""PopArt running target normalization for scalar value heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PopArtUpdate:
    old_mean: float
    old_std: float
    new_mean: float
    new_std: float
    sample_count: int
    affine_scale: float


class ScalarPopArt:
    """Cumulative running moments plus output-preserving affine compensation.

    The wrapped network continues to emit normalized scalar values.  Callers
    must denormalize those values before GAE/bootstrap and normalize return
    targets only for the critic loss.
    """

    def __init__(self, *, min_std: float = 1.0, max_abs_target: float = 1.0e20) -> None:
        if not math.isfinite(float(min_std)) or float(min_std) <= 0.0:
            raise ValueError("PopArt min_std must be finite and positive")
        if not math.isfinite(float(max_abs_target)) or float(max_abs_target) <= 0.0:
            raise ValueError("PopArt max_abs_target must be finite and positive")
        self.min_std = float(min_std)
        self.max_abs_target = float(max_abs_target)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def std(self) -> float:
        if self.count <= 0:
            return 1.0
        variance = max(self.m2 / float(self.count), 0.0)
        return max(math.sqrt(variance), self.min_std)

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - float(self.mean)) / float(self.std)

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * float(self.std) + float(self.mean)

    @staticmethod
    def linear_output(inputs: torch.Tensor, output_layer: nn.Linear) -> torch.Tensor:
        """Evaluate the compensated affine head in float64.

        PopArt can subtract a large running mean from a small raw value.  The
        float64 affine evaluation keeps the frozen 1e-6 denormalized prediction
        invariance gate meaningful while gradients still flow to float32 model
        parameters through the casts.
        """

        return F.linear(
            inputs.to(dtype=torch.float64),
            output_layer.weight.to(dtype=torch.float64),
            output_layer.bias.to(dtype=torch.float64),
        )

    def update(
        self,
        targets: np.ndarray | list[float],
        *,
        output_layer: nn.Linear,
        optimizer: torch.optim.Optimizer | None,
    ) -> PopArtUpdate:
        values = np.asarray(targets, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("PopArt requires at least one target")
        if not np.all(np.isfinite(values)):
            raise ValueError("PopArt targets must all be finite")
        if float(np.max(np.abs(values))) > self.max_abs_target:
            raise ValueError("PopArt target exceeds the frozen finite-value bound")
        if output_layer.out_features != 1:
            raise ValueError("PopArt requires a scalar linear value output layer")

        old_mean = float(self.mean)
        old_std = float(self.std)
        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(np.sum((values - batch_mean) ** 2, dtype=np.float64))
        if not all(math.isfinite(item) for item in (batch_mean, batch_m2)):
            raise ValueError("PopArt batch moments are non-finite")

        if self.count == 0:
            new_count = batch_count
            new_mean = batch_mean
            new_m2 = batch_m2
        else:
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / float(new_count)
            new_m2 = (
                self.m2
                + batch_m2
                + delta * delta * self.count * batch_count / float(new_count)
            )
        new_variance = max(new_m2 / float(new_count), 0.0)
        new_std = max(math.sqrt(new_variance), self.min_std)
        if not all(math.isfinite(item) for item in (new_mean, new_m2, new_std)):
            raise ValueError("PopArt updated moments are non-finite")

        scale = old_std / new_std
        with torch.no_grad():
            output_layer.weight.mul_(scale)
            output_layer.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)
        self._rescale_optimizer_state(
            optimizer=optimizer,
            parameters=(output_layer.weight, output_layer.bias),
            scale=scale,
        )
        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2
        return PopArtUpdate(
            old_mean=old_mean,
            old_std=old_std,
            new_mean=new_mean,
            new_std=new_std,
            sample_count=new_count,
            affine_scale=scale,
        )

    @staticmethod
    def _rescale_optimizer_state(
        *,
        optimizer: torch.optim.Optimizer | None,
        parameters: tuple[nn.Parameter, nn.Parameter],
        scale: float,
    ) -> None:
        if optimizer is None:
            return
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            exp_avg = state.get("exp_avg")
            if isinstance(exp_avg, torch.Tensor):
                exp_avg.mul_(scale)
            exp_avg_sq = state.get("exp_avg_sq")
            if isinstance(exp_avg_sq, torch.Tensor):
                exp_avg_sq.mul_(scale * scale)
            max_exp_avg_sq = state.get("max_exp_avg_sq")
            if isinstance(max_exp_avg_sq, torch.Tensor):
                max_exp_avg_sq.mul_(scale * scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "scalar_popart_v1",
            "count": int(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
            "min_std": float(self.min_std),
            "max_abs_target": float(self.max_abs_target),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("schema_version")) != "scalar_popart_v1":
            raise ValueError("unsupported PopArt state schema")
        count = int(state["count"])
        mean = float(state["mean"])
        m2 = float(state["m2"])
        min_std = float(state["min_std"])
        max_abs_target = float(state["max_abs_target"])
        if count < 0 or m2 < 0.0:
            raise ValueError("invalid PopArt count or second moment")
        if not all(
            math.isfinite(item) and item >= 0.0
            for item in (abs(mean), m2, min_std, max_abs_target)
        ):
            raise ValueError("non-finite PopArt state")
        if min_std <= 0.0 or max_abs_target <= 0.0:
            raise ValueError("invalid PopArt bounds")
        self.count = count
        self.mean = mean
        self.m2 = m2
        self.min_std = min_std
        self.max_abs_target = max_abs_target
