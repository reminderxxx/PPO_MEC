"""Model-cache-aware offloading PPO baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent
from src.encoders import FlatSemanticEncoder


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _resolve_primary_vehicle(semantic_state: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(semantic_state.get("vehicles", []))
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    if primary_vehicle_id:
        primary_vehicle_id = str(primary_vehicle_id)
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == primary_vehicle_id:
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


def _rsu_by_id(semantic_state: dict[str, Any], rsu_id: Any) -> dict[str, Any]:
    for rsu in semantic_state.get("rsus", []) or []:
        if str(rsu.get("rsu_id")) == str(rsu_id):
            return dict(rsu)
    return {}


def _occupancy(rsu: dict[str, Any]) -> float:
    capacity = max(float(rsu.get("cache_capacity", rsu.get("adapter_cache_capacity", 1.0)) or 1.0), 1.0)
    return _clamp01(float(len(rsu.get("cached_adapter_ids", []) or [])) / capacity)


def _free_ratio(rsu: dict[str, Any]) -> float:
    return 1.0 - _occupancy(rsu) if rsu else 0.0


def _adapter_ready(rsu: dict[str, Any], adapter_id: str | None) -> float:
    if not adapter_id:
        return 0.0
    return 1.0 if str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", [])} else 0.0


def _demand_score(predictions: dict[str, Any], rsu_id: Any) -> float:
    cache_demand = predictions.get("cache_demand", {}) if isinstance(predictions, dict) else {}
    demand_scores = cache_demand.get("demand_score_by_rsu", {}) if isinstance(cache_demand, dict) else {}
    value = demand_scores.get(rsu_id, 0.0) if isinstance(demand_scores, dict) else 0.0
    if isinstance(value, dict):
        numeric_values = []
        for item in value.values():
            try:
                numeric_values.append(float(item))
            except (TypeError, ValueError):
                continue
        return _clamp01(max(numeric_values) if numeric_values else 0.0)
    try:
        return _clamp01(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _build_cache_feature_tensor(semantic_state: dict[str, Any]) -> torch.Tensor:
    primary_vehicle = _resolve_primary_vehicle(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    predictions = semantic_state.get("predictions", {}) or {}
    next_sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) if isinstance(predictions, dict) else []
    predicted_next_rsu_id = (
        predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        if isinstance(predictions, dict)
        else None
    )
    if predicted_next_rsu_id is None and next_sequence:
        predicted_next_rsu_id = next_sequence[0]
    predicted_target_rsu_id = (
        predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        if isinstance(predictions, dict)
        else None
    )
    current_node = semantic_state.get("current_workflow_node") or {}
    required_adapter = current_node.get("required_adapter")
    current_rsu = _rsu_by_id(semantic_state, current_rsu_id)
    predicted_rsu = _rsu_by_id(semantic_state, predicted_next_rsu_id)
    target_rsu = _rsu_by_id(semantic_state, predicted_target_rsu_id)
    rsus = list(semantic_state.get("rsus", []) or [])
    occupancies = [_occupancy(rsu) for rsu in rsus]
    future_load = predictions.get("future_load", {}) if isinstance(predictions, dict) else {}
    current_load = float(future_load.get(current_rsu_id, 0.0) or 0.0) / 10.0
    predicted_load = float(future_load.get(predicted_next_rsu_id, 0.0) or 0.0) / 10.0
    target_load = float(future_load.get(predicted_target_rsu_id, 0.0) or 0.0) / 10.0
    features = [
        _occupancy(current_rsu),
        sum(occupancies) / max(len(occupancies), 1) if occupancies else 0.0,
        max(occupancies) if occupancies else 0.0,
        _adapter_ready(current_rsu, required_adapter),
        _adapter_ready(predicted_rsu, required_adapter),
        _adapter_ready(target_rsu, required_adapter),
        _free_ratio(current_rsu),
        _free_ratio(predicted_rsu),
        _free_ratio(target_rsu),
        current_load,
        predicted_load,
        target_load,
        _demand_score(predictions, current_rsu_id),
        max(_demand_score(predictions, predicted_next_rsu_id), _demand_score(predictions, predicted_target_rsu_id)),
    ]
    return torch.tensor([_clamp01(item) for item in features], dtype=torch.float32)


class _CacheOffloadPolicyNetwork(nn.Module):
    """Flat semantic actor-critic augmented with model-cache state."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.encoder = FlatSemanticEncoder(hidden_dim=hidden_dim)
        self.cache_projection = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.slow_norm = nn.LayerNorm(hidden_dim)
        self.fast_norm = nn.LayerNorm(hidden_dim)
        self.event_norm = nn.LayerNorm(hidden_dim)
        self.critic_norm = nn.LayerNorm(hidden_dim)
        self.slow_actor = self._make_head(hidden_dim, 3, hidden_dims)
        self.fast_actor = self._make_head(hidden_dim, 2, hidden_dims)
        self.event_actor = self._make_head(hidden_dim, 2, hidden_dims)
        self.central_critic = self._make_head(hidden_dim, 1, hidden_dims)

    @staticmethod
    def _make_head(input_dim: int, output_dim: int, hidden_dims: tuple[int, int]) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward_single(
        self,
        semantic_state: dict[str, Any],
        event_logit_temperature: float | None = None,
    ) -> dict[str, Any]:
        encoded = dict(self.encoder(semantic_state))
        cache_features = _build_cache_feature_tensor(semantic_state)
        cache_embedding = self.cache_projection(cache_features.unsqueeze(0)).squeeze(0)
        slow_context = self.slow_norm(encoded["slow_context"] + cache_embedding)
        fast_context = self.fast_norm(encoded["fast_context"] + 0.55 * cache_embedding)
        event_context = self.event_norm(encoded["event_context"] + 0.35 * cache_embedding)
        critic_context = self.critic_norm(encoded["centralized_critic_context"] + cache_embedding)
        temperature = max(float(event_logit_temperature if event_logit_temperature is not None else 1.0), 0.25)
        value = self.central_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
        encoded["encoder_mode"] = "cache_offload_domain_baseline"
        encoded["current_cache_occupancy"] = cache_features[0:1]
        encoded["current_adapter_ready"] = cache_features[3:4]
        encoded["future_adapter_ready"] = torch.max(cache_features[4:5], cache_features[5:6])
        return {
            "encoded": encoded,
            "slow_logits": self.slow_actor(slow_context.unsqueeze(0)).squeeze(0),
            "fast_logits": self.fast_actor(fast_context.unsqueeze(0)).squeeze(0),
            "event_logits": self.event_actor(event_context.unsqueeze(0)).squeeze(0) / temperature,
            "value": value,
            "critic_mode": "cache_offload_centralized_critic",
            "critic_context_key": "flat_semantic_plus_cache_scalars",
            "head_values": {"slow": value, "fast": value, "event": value},
        }


class CacheOffloadDRLAgent(PPOBaseAgent):
    """Model/adapter cache aware offloading learned baseline.

    This baseline receives model-cache occupancy, adapter readiness, cache
    demand, and future-load scalars. It is a domain comparator, not a copy of
    the SA-GHMAPPO graph/surrogate/guarded policy.
    """

    observation_contract = "flat_semantic_plus_cache_offload_v1"
    action_contract = "semantic_discrete_5_cache_offload_3head"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="cache_offload_drl",
            policy_type="cache_offload_drl_policy",
            encoder_kind="flat",
            centralized_critic=True,
            hierarchical_conditioning=False,
            use_hierarchy=True,
            use_prediction_features=False,
            use_uncertainty_signal=False,
            use_dependency_aware=False,
            graph_continuity_critic_enabled=False,
            uncertainty_aware_event_scaling_enabled=False,
            uncertainty_aware_critic_enabled=False,
            event_head_enabled=True,
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
            deterministic_temporal_smoothing_enabled=False,
            deterministic_high_prepare_override_enabled=False,
            predictive_prepare_hard_override_enabled=False,
            continuity_guard_enabled=False,
            handoff_target_alignment_guard_enabled=False,
            heuristic_imitation_coef=0.0,
            mechanism_aux_coef=0.0,
            mechanism_entropy_coef=0.0,
            backhaul_guard_enabled=False,
            cache_warm_start_guard_enabled=False,
            **kwargs,
        )
        self._network = _CacheOffloadPolicyNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self.baseline_config = {
            "family": "cache_offload_drl",
            "domain_focus": "model_adapter_cache_and_computation_offloading",
            "flat_policy": False,
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "cache_scalar_features": True,
            "graph_encoder": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": True,
            "ctde_scope": "controller_level_cache_execution_handoff",
            "paper_grade_independent_baseline": True,
            "reference_basis": "Joint service/model caching and computation offloading DRL literature.",
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

    def _checkpoint_config(self) -> dict[str, Any]:
        config = super()._checkpoint_config()
        config.update(
            {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "observation_contract": self.observation_contract,
                "action_contract": self.action_contract,
                "domain_feature_block": "model_cache_offload_scalars",
            }
        )
        return config
