"""Basic PPO baseline over the shared semantic action contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent


class PPOAgent(PPOBaseAgent):
    """Single-agent PPO with a flat semantic encoder and independent critic."""

    observation_contract = "flat_semantic_encoder_v1"
    action_contract = "semantic_discrete_5"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="ppo",
            policy_type="ppo_policy",
            encoder_kind="flat",
            centralized_critic=False,
            hierarchical_conditioning=False,
            use_hierarchy=False,
            use_prediction_features=False,
            use_uncertainty_signal=False,
            use_dependency_aware=False,
            graph_continuity_critic_enabled=False,
            uncertainty_aware_event_scaling_enabled=False,
            uncertainty_aware_critic_enabled=False,
            event_head_enabled=False,
            adapter_prefetch_enabled=True,
            auxiliary_coef=0.0,
            event_entropy_coef_scale=1.0,
            event_entropy_credit_floor=0.0,
            event_policy_credit_floor=0.0,
            event_advantage_blend=1.0,
            event_logit_temperature=1.0,
            event_logit_temperature_final=1.0,
            event_temperature_decay_updates=0,
            event_logit_sharpening_final_scale=1.0,
            event_logit_sharpening_timing_gain=0.0,
            event_actor_loss_extra_gain=1.0,
            event_prepare_margin_boost=0.0,
            temporal_consistency_coef=0.0,
            **kwargs,
        )
        self.baseline_config = {
            "family": "ppo",
            "flat_policy": True,
            "graph_encoder": False,
            "hierarchy": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": False,
        }


class TraceDrivenMultiRequestPPO(nn.Module):
    """Exploratory shared actor for per-vehicle RSU choices in the replay env.

    This has a separate observation/action contract from PPOAgent and is not
    registered as a formal PPO_MEC baseline.
    """

    feature_names = (
        "distance_over_radius",
        "nearest_association",
        "adapter_cached",
        "adapter_loading",
        "associated_demand_over_eight",
        "cache_occupancy",
        "remaining_nodes_over_twenty",
        "last_service_here",
        "frame_fraction",
        "unfinished_fraction",
    )

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        width = len(self.feature_names)
        self.actor = nn.Sequential(
            nn.Linear(width, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(width, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    @classmethod
    def encode(
        cls, observation: dict[str, Any], *, window_length: int
    ) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
        """Encode only the current frame; mask uncovered RSU choices."""
        from math import dist

        rsu_ids = sorted(observation["rsus"])
        active_ids = [
            vehicle_id for vehicle_id, item in sorted(observation["vehicles"].items())
            if not item["completed"] and item["covered_rsu_ids"]
        ]
        unfinished = sum(not item["completed"] for item in observation["vehicles"].values())
        rows: list[list[list[float]]] = []
        masks: list[list[bool]] = []
        for vehicle_id in active_ids:
            vehicle = observation["vehicles"][vehicle_id]
            row = []
            mask = []
            for rsu_id in rsu_ids:
                rsu = observation["rsus"][rsu_id]
                covered = rsu_id in vehicle["covered_rsu_ids"]
                mask.append(covered)
                adapter_id = vehicle["required_adapter"]
                radius = max(float(rsu["coverage_radius"]), 1e-6)
                row.append([
                    min(dist(
                        (vehicle["position_x"], vehicle["position_y"]),
                        (rsu["position_x"], rsu["position_y"]),
                    ) / radius, 2.0),
                    float(vehicle["associated_rsu_id"] == rsu_id),
                    float(adapter_id in rsu["cached_adapter_ids"]),
                    float(adapter_id in rsu["loading_adapter_ids"]),
                    float(rsu["associated_vehicle_count"]) / 8.0,
                    float(len(rsu["cached_adapter_ids"]) + len(rsu["loading_adapter_ids"]))
                    / max(float(rsu["adapter_slots"]), 1.0),
                    min(float(vehicle["remaining_nodes"]) / 20.0, 1.0),
                    float(vehicle["last_service_rsu_id"] == rsu_id),
                    float(observation["frame_index"]) / max(window_length - 1, 1),
                    float(unfinished) / max(len(observation["vehicles"]), 1),
                ])
            rows.append(row)
            masks.append(mask)
        if not active_ids:
            return active_ids, rsu_ids, torch.empty((0, len(rsu_ids), len(cls.feature_names))), torch.empty((0, len(rsu_ids)), dtype=torch.bool)
        return (
            active_ids,
            rsu_ids,
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool),
        )

    def distribution_and_value(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        if features.ndim != 3 or mask.shape != features.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("features/mask must contain at least one legal RSU per request")
        logits = self.actor(features).squeeze(-1).masked_fill(~mask, -1.0e9)
        weights = mask.unsqueeze(-1).float()
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        values = self.critic(pooled).squeeze(-1)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(features, mask)
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values
