"""多时间尺度 MARL 智能体公共组件。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.agents.base_agent import BaseAgent
from src.agents.popularity_cache_heuristic_agent import PopularityCacheHeuristicAgent
from src.encoders import FlatSemanticEncoder, SurrogateFusionEncoder
from src.encoders.fusion_encoder import (
    build_graph_continuity_critic_features,
    build_prediction_reliability_summary,
    compute_temporal_prepare_window_score,
)


控制头动作空间 = {
    "slow": 3,
    "fast": 2,
    "event": 2,
}

控制头动作语义 = {
    "slow": {0: "no_cache_change", 1: "current_rsu_cache_fill", 2: "predictive_next_rsu_prefetch"},
    "fast": {0: "current_rsu_offload", 1: "vehicle_fallback"},
    "event": {0: "keep", 1: "handoff_prepare"},
}


def _resolve_primary_vehicle_from_semantic_state(
    semantic_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    vehicles = list(semantic_state.get("vehicles", []))
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    first_vehicle = dict(vehicles[0]) if vehicles else {}
    resolved_vehicle = None
    lookup_fallback = False
    if primary_vehicle_id:
        primary_vehicle_id = str(primary_vehicle_id)
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == primary_vehicle_id:
                resolved_vehicle = dict(vehicle)
                break
        if resolved_vehicle is None and first_vehicle:
            lookup_fallback = True
    if resolved_vehicle is None:
        resolved_vehicle = dict(first_vehicle)
    first_vehicle_id = first_vehicle.get("vehicle_id")
    resolved_vehicle_id = resolved_vehicle.get("vehicle_id")
    primary_vehicle_present = bool(
        semantic_state.get("primary_vehicle_present", False)
        or (primary_vehicle_id and resolved_vehicle_id == primary_vehicle_id)
    )
    resolution_warning = ""
    if primary_vehicle_id and lookup_fallback:
        resolution_warning = "primary_vehicle_lookup_fallback_to_first"
    return resolved_vehicle, {
        "primary_vehicle_id": primary_vehicle_id,
        "primary_vehicle_present": primary_vehicle_present,
        "primary_vehicle_reordered_to_front": bool(semantic_state.get("primary_vehicle_reordered_to_front", False)),
        "first_vehicle_id": first_vehicle_id,
        "first_vehicle_matches_primary": bool(
            primary_vehicle_id and first_vehicle_id and str(first_vehicle_id) == str(primary_vehicle_id)
        ),
        "primary_vehicle_lookup_fallback": lookup_fallback,
        "primary_vehicle_resolution_warning": resolution_warning,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


class _多层感知机(nn.Module):
    """轻量 MLP。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class 分层策略网络(nn.Module):
    """共享 encoder + 多 actor heads + centralized / independent critic。"""

    def __init__(
        self,
        hidden_dim: int = 64,
        encoder_kind: str = "graph",
        use_hierarchy: bool = True,
        hierarchical_conditioning: bool = False,
        centralized_critic: bool = True,
        use_prediction_features: bool = True,
        use_uncertainty_signal: bool = True,
        use_dependency_aware: bool = True,
        prediction_feature_dim: int = 13,
        prediction_gate_min_leak: float = 0.0,
        graph_continuity_critic_enabled: bool = False,
        uncertainty_aware_critic_enabled: bool = False,
        event_logit_temperature: float = 1.0,
        hidden_dims: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_hierarchy = bool(use_hierarchy)
        self.hierarchical_conditioning = bool(hierarchical_conditioning)
        self.centralized_critic = bool(centralized_critic)
        self.encoder_kind = str(encoder_kind)
        self.event_logit_temperature = max(float(event_logit_temperature), 0.25)

        if self.encoder_kind == "flat":
            self.encoder = FlatSemanticEncoder(hidden_dim=self.hidden_dim)
        else:
            self.encoder = SurrogateFusionEncoder(
                hidden_dim=self.hidden_dim,
                use_prediction_features=use_prediction_features,
                use_uncertainty_signal=use_uncertainty_signal,
                use_dependency_aware=use_dependency_aware,
                prediction_feature_dim=prediction_feature_dim,
                prediction_gate_min_leak=prediction_gate_min_leak,
                graph_continuity_critic_enabled=graph_continuity_critic_enabled,
                uncertainty_aware_critic_enabled=uncertainty_aware_critic_enabled,
            )

        if self.use_hierarchy:
            self.slow_actor = _多层感知机(self.hidden_dim, 控制头动作空间["slow"], hidden_dims=hidden_dims)
            fast_input_dim = self.hidden_dim + (控制头动作空间["slow"] if self.hierarchical_conditioning else 0)
            event_input_dim = self.hidden_dim + (
                控制头动作空间["slow"] + 控制头动作空间["fast"] if self.hierarchical_conditioning else 0
            )
            self.fast_actor = _多层感知机(fast_input_dim, 控制头动作空间["fast"], hidden_dims=hidden_dims)
            self.event_actor = _多层感知机(event_input_dim, 控制头动作空间["event"], hidden_dims=hidden_dims)
            if self.centralized_critic:
                self.central_critic = _多层感知机(self.hidden_dim, 1, hidden_dims=hidden_dims)
            else:
                self.slow_critic = _多层感知机(self.hidden_dim, 1, hidden_dims=hidden_dims)
                self.fast_critic = _多层感知机(self.hidden_dim, 1, hidden_dims=hidden_dims)
                self.event_critic = _多层感知机(self.hidden_dim, 1, hidden_dims=hidden_dims)
        else:
            self.flat_actor = _多层感知机(self.hidden_dim, 5, hidden_dims=hidden_dims)
            self.flat_critic = _多层感知机(self.hidden_dim, 1, hidden_dims=hidden_dims)

    def forward_single(
        self,
        semantic_state: dict[str, Any],
        event_logit_temperature: float | None = None,
    ) -> dict[str, Any]:
        encoded = self.encoder(semantic_state)
        effective_event_temperature = max(
            float(self.event_logit_temperature if event_logit_temperature is None else event_logit_temperature),
            0.25,
        )
        if not self.use_hierarchy:
            flat_logits = self.flat_actor(encoded["shared_embedding"].unsqueeze(0)).squeeze(0)
            critic_context_key = "centralized_critic_context" if self.centralized_critic else "critic_context"
            critic_context = encoded.get(critic_context_key, encoded["critic_context"])
            value = self.flat_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
            return {
                "encoded": encoded,
                "flat_logits": flat_logits,
                "value": value,
                "critic_mode": "centralized" if self.centralized_critic else "independent",
                "critic_context_key": critic_context_key,
                "head_values": {},
            }

        slow_logits = self.slow_actor(encoded["slow_context"].unsqueeze(0)).squeeze(0)
        slow_probs = torch.softmax(slow_logits, dim=-1)
        fast_input = encoded["fast_context"]
        if self.hierarchical_conditioning:
            fast_input = torch.cat([fast_input, slow_probs], dim=-1)
        fast_logits = self.fast_actor(fast_input.unsqueeze(0)).squeeze(0)
        fast_probs = torch.softmax(fast_logits, dim=-1)
        event_input = encoded["event_context"]
        if self.hierarchical_conditioning:
            event_input = torch.cat([event_input, slow_probs, fast_probs], dim=-1)
        event_logits = self.event_actor(event_input.unsqueeze(0)).squeeze(0)
        event_logits = event_logits / effective_event_temperature

        if self.centralized_critic:
            critic_context_key = "centralized_critic_context"
            critic_context = encoded.get(critic_context_key, encoded["critic_context"])
            value = self.central_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
            head_values = {
                "slow": value,
                "fast": value,
                "event": value,
            }
        else:
            critic_context_key = "critic_context"
            slow_value = self.slow_critic(encoded["slow_context"].unsqueeze(0)).squeeze(0).squeeze(-1)
            fast_value = self.fast_critic(encoded["fast_context"].unsqueeze(0)).squeeze(0).squeeze(-1)
            event_value = self.event_critic(encoded["event_context"].unsqueeze(0)).squeeze(0).squeeze(-1)
            head_values = {
                "slow": slow_value,
                "fast": fast_value,
                "event": event_value,
            }
            value = torch.stack(list(head_values.values())).mean()

        return {
            "encoded": encoded,
            "slow_logits": slow_logits,
            "fast_logits": fast_logits,
            "event_logits": event_logits,
            "value": value,
            "head_values": head_values,
            "event_logit_temperature": effective_event_temperature,
            "critic_mode": "centralized" if self.centralized_critic else "independent",
            "critic_context_key": critic_context_key,
        }


def 聚合层级动作(
    head_actions: dict[str, int],
    use_hierarchy: bool,
    event_head_enabled: bool,
    adapter_prefetch_enabled: bool,
) -> tuple[int, str]:
    if not use_hierarchy:
        env_action = int(head_actions.get("flat", 3))
        return env_action, f"flat_action_{env_action}"
    slow_action = int(head_actions.get("slow", 0))
    fast_action = int(head_actions.get("fast", 0))
    event_action = int(head_actions.get("event", 0))
    if event_head_enabled and event_action == 1:
        return 4, "event_head_prepare"
    if adapter_prefetch_enabled and slow_action == 2:
        return 1, "slow_head_prefetch"
    if slow_action == 1:
        return 0, "slow_head_cache_fill"
    if fast_action == 1:
        return 2, "fast_head_vehicle_fallback"
    return 3, "fast_head_steady_offload"


class 分层PPO基类(BaseAgent):
    """共享 encoder 的多头 PPO / MAPPO / SA-GHMAPPO 基类。"""

    def __init__(
        self,
        agent_name: str,
        policy_type: str,
        encoder_kind: str,
        centralized_critic: bool,
        hierarchical_conditioning: bool,
        use_hierarchy: bool = True,
        use_prediction_features: bool = True,
        use_uncertainty_signal: bool = True,
        use_dependency_aware: bool = True,
        graph_continuity_critic_enabled: bool = False,
        uncertainty_aware_event_scaling_enabled: bool = False,
        uncertainty_aware_critic_enabled: bool = False,
        event_head_enabled: bool = True,
        adapter_prefetch_enabled: bool = True,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        auxiliary_coef: float = 0.0,
        head_credit_enabled: bool = False,
        head_credit_protocol: str = "aggregation_reason_weighted_ppo_v2",
        mechanism_logit_bias_strength: float = 0.0,
        mechanism_confidence_floor: float = 0.0,
        prediction_feature_dim: int = 13,
        prediction_gate_min_leak: float = 0.0,
        slow_entropy_coef_scale: float = 1.0,
        fast_entropy_coef_scale: float = 1.0,
        event_entropy_coef_scale: float = 1.0,
        slow_entropy_credit_floor: float = 0.0,
        fast_entropy_credit_floor: float = 0.0,
        event_entropy_credit_floor: float = 0.0,
        event_logit_temperature: float = 1.0,
        event_logit_temperature_final: float | None = None,
        event_temperature_decay_updates: int = 0,
        slow_policy_credit_floor: float = 0.0,
        fast_policy_credit_floor: float = 0.0,
        event_policy_credit_floor: float = 0.0,
        event_advantage_blend: float = 1.0,
        event_logit_sharpening_final_scale: float = 1.0,
        event_logit_sharpening_timing_gain: float = 0.0,
        event_actor_loss_extra_gain: float = 1.0,
        event_prepare_margin_boost: float = 0.0,
        temporal_consistency_coef: float = 0.0,
        temporal_prepare_lead_steps: float = 2.5,
        temporal_prepare_sigma: float = 1.25,
        temporal_prepare_activation_threshold: float = 0.35,
        deterministic_temporal_smoothing_enabled: bool = False,
        deterministic_temporal_smoothing_steps: int = 1,
        deterministic_event_borderline_prob: float = 0.43,
        deterministic_event_borderline_margin: float = -0.10,
        deterministic_temporal_urgency_floor: float = 0.35,
        deterministic_high_prepare_override_enabled: bool = True,
        deterministic_high_prepare_threshold: float = 0.55,
        deterministic_high_urgency_threshold: float = 0.50,
        deterministic_high_prepare_relaxed_margin: float = -0.12,
        predictive_prepare_hard_override_enabled: bool = False,
        predictive_prepare_hard_override_score_threshold: float = 0.55,
        predictive_prepare_hard_override_confidence_threshold: float = 0.70,
        continuity_guard_enabled: bool = False,
        handoff_target_alignment_guard_enabled: bool = False,
        continuity_guard_logit_penalty: float = 1.0,
        continuity_guard_prepare_boost: float = 1.25,
        continuity_guard_confidence_threshold: float = 0.45,
        continuity_guard_prepare_score_threshold: float = 0.30,
        continuity_guard_hard_override_enabled: bool = False,
        heuristic_imitation_coef: float = 0.0,
        heuristic_imitation_warmup_updates: int = 2,
        heuristic_imitation_decay: float = 0.5,
        mechanism_aux_coef: float = 0.0,
        mechanism_window_weight: float = 1.0,
        prepare_action_prior_weight: float = 0.5,
        mechanism_entropy_coef: float = 0.0,
        mechanism_retention_start_update: int = 0,
        mechanism_aux_coef_floor_after_update: float = 0.0,
        mechanism_window_weight_floor_after_update: float = 1.0,
        mechanism_entropy_floor_after_update: float = 0.0,
        mechanism_aux_current_cache_fill_enabled: bool = True,
        latency_fallback_bias_enabled: bool = False,
        latency_fallback_bias_strength: float = 0.0,
        latency_fallback_confidence_floor: float = 0.0,
        latency_fallback_slow_suppression_strength: float = 0.0,
        backhaul_guard_enabled: bool = False,
        backhaul_guard_max_reactive_fills_per_adapter: int = 1,
        cache_warm_start_guard_enabled: bool = False,
        cache_warm_start_guard_min_countdown: float = 1.5,
        cache_warm_start_guard_max_prefetch_countdown: float = 0.0,
        predictive_prefetch_admission_guard_enabled: bool = False,
        predictive_prefetch_admission_min_confidence: float = 0.55,
        predictive_prefetch_admission_require_distinct_next: bool = True,
        auxiliary_slow_weight: float = 1.0,
        auxiliary_fast_weight: float = 0.5,
        auxiliary_event_weight: float = 1.0,
        train_epochs: int = 6,
        target_kl: float = 0.0,
        kl_early_stop_enabled: bool = False,
        batch_size: int = 32,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
        deterministic_action: bool = False,
        random_seed: int = 7,
        device: str = "cpu",
    ) -> None:
        super().__init__(agent_name=agent_name)
        self.policy_type = policy_type
        self._encoder_kind = encoder_kind
        self._centralized_critic = bool(centralized_critic)
        self._hierarchical_conditioning = bool(hierarchical_conditioning)
        self._use_hierarchy = bool(use_hierarchy)
        self._use_prediction_features = bool(use_prediction_features)
        self._use_uncertainty_signal = bool(use_uncertainty_signal)
        self._use_dependency_aware = bool(use_dependency_aware)
        self._graph_continuity_critic_enabled = bool(graph_continuity_critic_enabled)
        self._uncertainty_aware_event_scaling_enabled = bool(uncertainty_aware_event_scaling_enabled)
        self._uncertainty_aware_critic_enabled = bool(uncertainty_aware_critic_enabled)
        self._event_head_enabled = bool(event_head_enabled)
        self._adapter_prefetch_enabled = bool(adapter_prefetch_enabled)
        self._learning_rate = float(learning_rate)
        self._clip_ratio = float(clip_ratio)
        self._entropy_coef = float(entropy_coef)
        self._value_coef = float(value_coef)
        self._auxiliary_coef = float(auxiliary_coef)
        self._head_credit_enabled = bool(head_credit_enabled)
        self._head_credit_protocol = str(head_credit_protocol or "aggregation_reason_weighted_ppo_v2")
        self._mechanism_logit_bias_strength = float(mechanism_logit_bias_strength)
        self._mechanism_confidence_floor = float(mechanism_confidence_floor)
        self._prediction_feature_dim = int(prediction_feature_dim)
        self._prediction_gate_min_leak = max(0.0, min(float(prediction_gate_min_leak), 1.0))
        self._slow_entropy_coef_scale = max(float(slow_entropy_coef_scale), 0.0)
        self._fast_entropy_coef_scale = max(float(fast_entropy_coef_scale), 0.0)
        self._event_entropy_coef_scale = max(float(event_entropy_coef_scale), 0.0)
        self._slow_entropy_credit_floor = max(0.0, min(float(slow_entropy_credit_floor), 1.0))
        self._fast_entropy_credit_floor = max(0.0, min(float(fast_entropy_credit_floor), 1.0))
        self._event_entropy_credit_floor = max(0.0, min(float(event_entropy_credit_floor), 1.0))
        self._event_logit_temperature = max(float(event_logit_temperature), 0.25)
        if event_logit_temperature_final is None:
            event_logit_temperature_final = min(self._event_logit_temperature, 1.0)
        self._event_logit_temperature_final = max(float(event_logit_temperature_final), 0.25)
        self._event_temperature_decay_updates = max(int(event_temperature_decay_updates), 0)
        self._slow_policy_credit_floor = max(0.0, min(float(slow_policy_credit_floor), 1.0))
        self._fast_policy_credit_floor = max(0.0, min(float(fast_policy_credit_floor), 1.0))
        self._event_policy_credit_floor = max(0.0, min(float(event_policy_credit_floor), 1.0))
        self._policy_credit_floor_by_head = {
            "slow": self._slow_policy_credit_floor,
            "fast": self._fast_policy_credit_floor,
            "event": self._event_policy_credit_floor,
        }
        self._entropy_credit_floor_by_head = {
            "slow": self._slow_entropy_credit_floor,
            "fast": self._fast_entropy_credit_floor,
            "event": self._event_entropy_credit_floor,
        }
        self._entropy_coef_scale_by_head = {
            "slow": self._slow_entropy_coef_scale,
            "fast": self._fast_entropy_coef_scale,
            "event": self._event_entropy_coef_scale,
        }
        self._event_advantage_blend = max(float(event_advantage_blend), 0.0)
        self._event_logit_sharpening_final_scale = max(float(event_logit_sharpening_final_scale), 1.0)
        self._event_logit_sharpening_timing_gain = max(float(event_logit_sharpening_timing_gain), 0.0)
        self._event_actor_loss_extra_gain = max(float(event_actor_loss_extra_gain), 1.0)
        self._event_prepare_margin_boost = max(float(event_prepare_margin_boost), 0.0)
        self._temporal_consistency_coef = max(float(temporal_consistency_coef), 0.0)
        self._temporal_prepare_lead_steps = max(float(temporal_prepare_lead_steps), 0.5)
        self._temporal_prepare_sigma = max(float(temporal_prepare_sigma), 0.25)
        self._temporal_prepare_activation_threshold = max(
            0.0,
            min(float(temporal_prepare_activation_threshold), 1.0),
        )
        self._deterministic_temporal_smoothing_enabled = bool(deterministic_temporal_smoothing_enabled)
        self._deterministic_temporal_smoothing_steps = max(int(deterministic_temporal_smoothing_steps), 1)
        self._deterministic_event_borderline_prob = max(
            0.0,
            min(float(deterministic_event_borderline_prob), 1.0),
        )
        self._deterministic_event_borderline_margin = float(deterministic_event_borderline_margin)
        self._deterministic_temporal_urgency_floor = max(
            0.0,
            min(float(deterministic_temporal_urgency_floor), 1.0),
        )
        self._deterministic_high_prepare_override_enabled = bool(deterministic_high_prepare_override_enabled)
        self._deterministic_high_prepare_threshold = max(
            0.0,
            min(float(deterministic_high_prepare_threshold), 1.0),
        )
        self._deterministic_high_urgency_threshold = max(
            0.0,
            min(float(deterministic_high_urgency_threshold), 1.0),
        )
        self._deterministic_high_prepare_relaxed_margin = float(deterministic_high_prepare_relaxed_margin)
        self._predictive_prepare_hard_override_enabled = bool(predictive_prepare_hard_override_enabled)
        self._predictive_prepare_hard_override_score_threshold = max(
            0.0,
            min(float(predictive_prepare_hard_override_score_threshold), 1.0),
        )
        self._predictive_prepare_hard_override_confidence_threshold = max(
            0.0,
            min(float(predictive_prepare_hard_override_confidence_threshold), 1.0),
        )
        self._continuity_guard_enabled = bool(continuity_guard_enabled)
        self._handoff_target_alignment_guard_enabled = bool(handoff_target_alignment_guard_enabled)
        self._continuity_guard_logit_penalty = max(float(continuity_guard_logit_penalty), 0.0)
        self._continuity_guard_prepare_boost = max(float(continuity_guard_prepare_boost), 0.0)
        self._continuity_guard_confidence_threshold = max(
            0.0,
            min(float(continuity_guard_confidence_threshold), 1.0),
        )
        self._continuity_guard_prepare_score_threshold = max(
            0.0,
            min(float(continuity_guard_prepare_score_threshold), 1.0),
        )
        self._continuity_guard_hard_override_enabled = bool(continuity_guard_hard_override_enabled)
        self._heuristic_imitation_coef = max(float(heuristic_imitation_coef), 0.0)
        self._heuristic_imitation_warmup_updates = max(int(heuristic_imitation_warmup_updates), 0)
        self._heuristic_imitation_decay = max(float(heuristic_imitation_decay), 0.0)
        self._mechanism_aux_coef = max(float(mechanism_aux_coef), 0.0)
        self._mechanism_window_weight = max(float(mechanism_window_weight), 1.0)
        self._prepare_action_prior_weight = max(float(prepare_action_prior_weight), 0.0)
        self._mechanism_entropy_coef = max(float(mechanism_entropy_coef), 0.0)
        self._mechanism_retention_start_update = max(int(mechanism_retention_start_update), 0)
        self._mechanism_aux_coef_floor_after_update = max(float(mechanism_aux_coef_floor_after_update), 0.0)
        self._mechanism_window_weight_floor_after_update = max(
            float(mechanism_window_weight_floor_after_update),
            1.0,
        )
        self._mechanism_entropy_floor_after_update = max(float(mechanism_entropy_floor_after_update), 0.0)
        self._mechanism_aux_current_cache_fill_enabled = bool(mechanism_aux_current_cache_fill_enabled)
        self._latency_fallback_bias_enabled = bool(latency_fallback_bias_enabled)
        self._latency_fallback_bias_strength = max(float(latency_fallback_bias_strength), 0.0)
        self._latency_fallback_confidence_floor = max(
            0.0,
            min(float(latency_fallback_confidence_floor), 1.0),
        )
        self._latency_fallback_slow_suppression_strength = max(
            float(latency_fallback_slow_suppression_strength),
            0.0,
        )
        self._backhaul_guard_enabled = bool(backhaul_guard_enabled)
        self._backhaul_guard_max_reactive_fills_per_adapter = max(
            int(backhaul_guard_max_reactive_fills_per_adapter),
            0,
        )
        self._cache_warm_start_guard_enabled = bool(cache_warm_start_guard_enabled)
        self._cache_warm_start_guard_min_countdown = max(
            float(cache_warm_start_guard_min_countdown),
            0.0,
        )
        self._cache_warm_start_guard_max_prefetch_countdown = max(
            float(cache_warm_start_guard_max_prefetch_countdown),
            0.0,
        )
        self._predictive_prefetch_admission_guard_enabled = bool(
            predictive_prefetch_admission_guard_enabled
        )
        self._predictive_prefetch_admission_min_confidence = max(
            0.0,
            min(float(predictive_prefetch_admission_min_confidence), 1.0),
        )
        self._predictive_prefetch_admission_require_distinct_next = bool(
            predictive_prefetch_admission_require_distinct_next
        )
        self._auxiliary_slow_weight = float(auxiliary_slow_weight)
        self._auxiliary_fast_weight = float(auxiliary_fast_weight)
        self._auxiliary_event_weight = float(auxiliary_event_weight)
        self._train_epochs = int(train_epochs)
        self._target_kl = float(target_kl)
        self._kl_early_stop_enabled = bool(kl_early_stop_enabled)
        self._batch_size = int(batch_size)
        self._max_grad_norm = float(max_grad_norm)
        self._hidden_dim = int(hidden_dim)
        self._hidden_dims = tuple(hidden_dims)
        self._deterministic_action = bool(deterministic_action)
        self._device = torch.device(device)
        self._update_count = 0
        self._deterministic_temporal_streak = 0
        self._last_deterministic_time_index: int | None = None
        self._backhaul_guard_seen_reactive_fills: dict[str, int] = {}
        self._backhaul_guard_last_time_index: int | None = None

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self._network = 分层策略网络(
            hidden_dim=self._hidden_dim,
            encoder_kind=self._encoder_kind,
            use_hierarchy=self._use_hierarchy,
            hierarchical_conditioning=self._hierarchical_conditioning,
            centralized_critic=self._centralized_critic,
            use_prediction_features=self._use_prediction_features,
            use_uncertainty_signal=self._use_uncertainty_signal,
            use_dependency_aware=self._use_dependency_aware,
            prediction_feature_dim=self._prediction_feature_dim,
            prediction_gate_min_leak=self._prediction_gate_min_leak,
            graph_continuity_critic_enabled=self._graph_continuity_critic_enabled,
            uncertainty_aware_critic_enabled=self._uncertainty_aware_critic_enabled,
            event_logit_temperature=self._event_logit_temperature,
            hidden_dims=self._hidden_dims,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = self._extract_semantic_state(info)
        action_mask = self._extract_action_mask(info)
        deterministic = bool(self._deterministic_action or (info or {}).get("deterministic_policy", False))
        with torch.no_grad():
            policy_output = self._forward_policy(semantic_state)
            (
                selected_actions,
                head_log_probs,
                head_entropies,
                action_prob_payload,
                action_projection_info,
            ) = self._sample_actions(
                policy_output,
                deterministic=deterministic,
                action_mask=action_mask,
            )
            projected_head_actions = dict(selected_actions)
            projected_env_action, projected_aggregation_reason = 聚合层级动作(
                head_actions=projected_head_actions,
                use_hierarchy=self._use_hierarchy,
                event_head_enabled=self._event_head_enabled,
                adapter_prefetch_enabled=self._adapter_prefetch_enabled,
            )
            guard_info = dict(policy_output.get("continuity_guard_info", {}))
            if self._should_hard_apply_continuity_guard(
                selected_actions=selected_actions,
                guard_info=guard_info,
            ):
                selected_actions = dict(selected_actions)
                selected_actions["slow"] = 0
                selected_actions["event"] = 1
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
                guard_info["hard_override_applied"] = True
            smoothing_info = self._apply_deterministic_temporal_smoothing(
                semantic_state=semantic_state,
                policy_output=policy_output,
                selected_actions=selected_actions,
                deterministic=deterministic,
            )
            cache_warm_guard_info = self._apply_cache_warm_start_guard_to_actions(
                semantic_state=semantic_state,
                selected_actions=selected_actions,
            )
            if cache_warm_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            prefetch_admission_guard_info = self._apply_predictive_prefetch_admission_guard_to_actions(
                semantic_state=semantic_state,
                selected_actions=selected_actions,
            )
            if prefetch_admission_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            backhaul_guard_info = self._apply_backhaul_guard_to_actions(
                semantic_state=semantic_state,
                selected_actions=selected_actions,
                cache_warm_guard_info=cache_warm_guard_info,
            )
            if backhaul_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            env_action, aggregation_reason = 聚合层级动作(
                head_actions=selected_actions,
                use_hierarchy=self._use_hierarchy,
                event_head_enabled=self._event_head_enabled,
                adapter_prefetch_enabled=self._adapter_prefetch_enabled,
            )
            if guard_info:
                guard_info["guarded_action"] = int(env_action)
            guard_action_delta = bool(
                int(projected_env_action) != int(env_action)
                or dict(projected_head_actions) != dict(selected_actions)
            )
            head_credit_weights = self._build_head_credit_weights(aggregation_reason=aggregation_reason)
            log_prob, entropy = self._combine_head_statistics(
                head_log_probs=head_log_probs,
                head_entropies=head_entropies,
                head_credit_weights=head_credit_weights,
            )
            value = float(policy_output["value"].item())
            prediction_gate = float(policy_output["encoded"].get("prediction_gate", torch.tensor([1.0], device=self._device)).flatten()[0].item())
            temporal_urgency = float(policy_output["encoded"].get("temporal_urgency", torch.tensor([0.0], device=self._device)).flatten()[0].item())
            prepare_window_score = float(policy_output["encoded"].get("prepare_window_score", torch.tensor([0.0], device=self._device)).flatten()[0].item())
            handoff_countdown_steps = float(policy_output["encoded"].get("handoff_countdown_steps", torch.tensor([0.0], device=self._device)).flatten()[0].item())
            active_event_logit_temperature = float(policy_output.get("event_logit_temperature", self._current_event_logit_temperature()))
            event_probs = action_prob_payload.get("event", [1.0, 0.0]) if self._use_hierarchy else [1.0, 0.0]
            event_prepare_prob = float(event_probs[1]) if len(event_probs) > 1 else 0.0
            event_margin = 0.0
            if self._use_hierarchy:
                event_margin = float((policy_output["event_logits"][1] - policy_output["event_logits"][0]).item())
            predicted_handoff_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
            prediction_target_diagnostics = self._build_prediction_target_diagnostics(
                semantic_state=semantic_state,
                temporal_urgency=temporal_urgency,
                predicted_handoff_target_valid=predicted_handoff_target_valid,
            )

        return env_action, {
            "policy_mode": "deterministic" if deterministic else "sample",
            "policy_type": self.policy_type,
            "encoder_mode": policy_output["encoded"].get("encoder_mode"),
            "critic_mode": policy_output.get("critic_mode", "centralized" if self._centralized_critic else "independent"),
            "critic_context_key": policy_output.get("critic_context_key", "critic_context"),
            "action_mask": list(action_mask) if action_mask is not None else None,
            "action_mask_applied": bool(self._action_mask_has_valid_action(action_mask)),
            "valid_action_count": self._valid_action_count(action_mask),
            "raw_head_actions": dict(action_projection_info.get("raw_head_actions", projected_head_actions)),
            "projected_head_actions": projected_head_actions,
            "head_actions": selected_actions,
            "head_action_labels": self._head_action_labels(selected_actions),
            "raw_env_action": int(action_projection_info.get("raw_env_action", projected_env_action)),
            "projected_env_action": int(projected_env_action),
            "final_env_action": int(env_action),
            "action_projection": action_projection_info,
            "action_projection_applied": bool(action_projection_info.get("projection_applied", False)),
            "invalid_action_attempt_count": int(action_projection_info.get("invalid_attempt_count", 0) or 0),
            "guard_action_delta": guard_action_delta,
            "projected_aggregation_reason": projected_aggregation_reason,
            "aggregation_reason": aggregation_reason,
            "log_prob": round(float(log_prob.item()), 6),
            "value": round(value, 6),
            "entropy": round(float(entropy.item()), 6),
            "head_log_probs": {
                head_name: round(float(head_value.item()), 6)
                for head_name, head_value in head_log_probs.items()
            },
            "action_probs": action_prob_payload,
            "prediction_gate": round(prediction_gate, 6),
            "temporal_urgency": round(temporal_urgency, 6),
            "prepare_window_score": round(prepare_window_score, 6),
            "handoff_countdown_steps": round(handoff_countdown_steps, 6),
            "event_logit_temperature": round(active_event_logit_temperature, 6),
            "event_sharpening_info": dict(policy_output.get("event_sharpening_info", {})),
            "event_prepare_prob": round(event_prepare_prob, 6),
            "event_margin": round(event_margin, 6),
            "predicted_handoff_target_valid": bool(predicted_handoff_target_valid),
            "raw_handoff_candidate": bool(prediction_target_diagnostics["raw_handoff_candidate"]),
            "prediction_confidence": round(float(prediction_target_diagnostics["prediction_confidence"]), 6),
            "prediction_uncertainty": round(float(prediction_target_diagnostics["prediction_uncertainty"]), 6),
            "urgency_support": round(float(prediction_target_diagnostics["urgency_support"]), 6),
            "prediction_gate_value": round(float(prediction_target_diagnostics["prediction_gate_value"]), 6),
            "gate_pass": bool(prediction_target_diagnostics["gate_pass"]),
            "prediction_invalid_reason": str(prediction_target_diagnostics["invalid_reason"]),
            "predictor_invoked": bool(prediction_target_diagnostics["predictor_invoked"]),
            "prediction_state_available": bool(prediction_target_diagnostics["prediction_state_available"]),
            "prediction_sequence_horizon": int(prediction_target_diagnostics["prediction_sequence_horizon"]),
            "next_rsu_non_null_count": int(prediction_target_diagnostics["next_rsu_non_null_count"]),
            "candidate_block_reason": str(prediction_target_diagnostics["candidate_block_reason"]),
            "primary_vehicle_id": prediction_target_diagnostics["primary_vehicle_id"],
            "primary_vehicle_present": bool(prediction_target_diagnostics["primary_vehicle_present"]),
            "primary_vehicle_reordered_to_front": bool(
                prediction_target_diagnostics["primary_vehicle_reordered_to_front"]
            ),
            "first_vehicle_id": prediction_target_diagnostics["first_vehicle_id"],
            "first_vehicle_matches_primary": bool(
                prediction_target_diagnostics["first_vehicle_matches_primary"]
            ),
            "primary_vehicle_lookup_fallback": bool(
                prediction_target_diagnostics["primary_vehicle_lookup_fallback"]
            ),
            "primary_vehicle_resolution_warning": str(
                prediction_target_diagnostics["primary_vehicle_resolution_warning"]
            ),
            "current_rsu_id": prediction_target_diagnostics["current_rsu_id"],
            "predicted_sequence_preview": list(prediction_target_diagnostics["predicted_sequence_preview"]),
            "predicted_sequence_all_null": bool(prediction_target_diagnostics["predicted_sequence_all_null"]),
            "predicted_sequence_all_current_rsu": bool(prediction_target_diagnostics["predicted_sequence_all_current_rsu"]),
            "predicted_sequence_contains_other_rsu": bool(prediction_target_diagnostics["predicted_sequence_contains_other_rsu"]),
            "predicted_first_non_current_rsu": prediction_target_diagnostics["predicted_first_non_current_rsu"],
            "predicted_first_non_current_eta": int(prediction_target_diagnostics["predicted_first_non_current_eta"]),
            "head_credit_protocol": self._head_credit_protocol,
            "head_credit_weights": head_credit_weights,
            "effective_head_credit_floors": {
                "policy": dict(self._policy_credit_floor_by_head),
                "entropy": dict(self._entropy_credit_floor_by_head),
                "entropy_scale": dict(self._entropy_coef_scale_by_head),
            },
            "deterministic_temporal_smoothing": smoothing_info,
            "cache_warm_start_guard": cache_warm_guard_info,
            "predictive_prefetch_admission_guard": prefetch_admission_guard_info,
            "backhaul_guard": backhaul_guard_info,
            "deterministic_event_prepare_overridden": bool(smoothing_info.get("override_triggered", False)),
            "deterministic_event_prepare_smoothed": bool(smoothing_info.get("borderline_triggered", False)),
            "guard_triggered": bool(guard_info.get("guard_triggered", False)),
            "continuity_guard": guard_info,
            "original_action": int(guard_info.get("original_action", env_action)),
            "guarded_action": int(guard_info.get("guarded_action", env_action)),
            "predicted_next_rsu_id": guard_info.get("predicted_next_rsu_id"),
            "predicted_handoff_target_rsu_id": guard_info.get("predicted_handoff_target_rsu_id"),
            "continuity_guard_reason": str(guard_info.get("reason", "not_triggered")),
        }

    def evaluate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        del observation
        semantic_state = self._extract_semantic_state(info)
        with torch.no_grad():
            policy_output = self._forward_policy(semantic_state)
        return float(policy_output["value"].item())

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        if not rollout:
            return {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "policy_update_skipped": True,
                "reason": "empty_rollout",
                "update_count": self._update_count,
            }

        imitation_rollout_stats = self._annotate_heuristic_imitation_targets(rollout)
        semantic_states = [self._extract_semantic_state(row.get("decision_info")) for row in rollout]
        action_masks = [self._extract_action_mask(row.get("decision_info")) for row in rollout]
        actions = [int(row["action"]) for row in rollout]
        returns = np.asarray([float(row.get("return", row.get("reward", 0.0))) for row in rollout], dtype=np.float32)
        advantages = np.asarray([float(row.get("advantage", 0.0)) for row in rollout], dtype=np.float32)
        values = np.asarray([float(row.get("value", 0.0)) for row in rollout], dtype=np.float32)
        old_log_probs = np.asarray([float(row.get("log_prob", 0.0)) for row in rollout], dtype=np.float32)
        retention_active = self._mechanism_retention_active_for_update()
        effective_mechanism_aux_coef = self._effective_mechanism_aux_coef()
        effective_mechanism_entropy_coef = self._effective_mechanism_entropy_coef()
        mechanism_guidance_annotations = [
            self._build_mechanism_guidance_annotation(semantic_state, row)
            for semantic_state, row in zip(semantic_states, rollout, strict=False)
        ]
        mechanism_transition_weights = np.asarray(
            [
                float(annotation.get("transition_weight", 1.0))
                for annotation in mechanism_guidance_annotations
            ],
            dtype=np.float32,
        )

        advantage_mean = float(advantages.mean()) if len(advantages) > 0 else 0.0
        advantage_std = float(advantages.std()) if len(advantages) > 0 else 0.0
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        normalized_advantages = normalized_advantages * mechanism_transition_weights
        event_advantages_raw = np.asarray(
            [
                float(row.get("event_advantage", row.get("advantage", 0.0)))
                for row in rollout
            ],
            dtype=np.float32,
        )
        event_advantage_mean = float(event_advantages_raw.mean()) if len(event_advantages_raw) > 0 else 0.0
        event_advantage_std = float(event_advantages_raw.std()) if len(event_advantages_raw) > 0 else 0.0
        uses_pre_normalized_event_advantage = any("event_advantage_normalized" in row for row in rollout)
        if uses_pre_normalized_event_advantage:
            normalized_event_advantages = np.asarray(
                [
                    float(row.get("event_advantage_normalized", 0.0))
                    for row in rollout
                ],
                dtype=np.float32,
            )
        else:
            normalized_event_advantages = (event_advantages_raw - event_advantage_mean) / (event_advantage_std + 1e-8)
        normalized_event_advantages = normalized_event_advantages * mechanism_transition_weights
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self._device)
        advantage_tensor = torch.as_tensor(normalized_advantages, dtype=torch.float32, device=self._device)
        event_advantage_tensor = torch.as_tensor(normalized_event_advantages, dtype=torch.float32, device=self._device)
        old_log_prob_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self._device)
        old_value_tensor = torch.as_tensor(values, dtype=torch.float32, device=self._device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self._device)

        head_action_tensors: dict[str, torch.Tensor] = {}
        if self._use_hierarchy:
            for head_name in ["slow", "fast", "event"]:
                head_action_tensors[head_name] = torch.as_tensor(
                    [
                        int(row.get("action_info", {}).get("head_actions", {}).get(head_name, 0))
                        for row in rollout
                    ],
                    dtype=torch.long,
                    device=self._device,
                )
            head_credit_tensors = {
                head_name: torch.as_tensor(
                    [
                        float(row.get("action_info", {}).get("head_credit_weights", {}).get(head_name, 1.0))
                        for row in rollout
                    ],
                    dtype=torch.float32,
                    device=self._device,
                )
                for head_name in ["slow", "fast", "event"]
            }
            old_head_log_prob_tensors = {
                head_name: torch.as_tensor(
                    [
                        float(row.get("action_info", {}).get("head_log_probs", {}).get(head_name, 0.0))
                        for row in rollout
                    ],
                    dtype=torch.float32,
                    device=self._device,
                )
                for head_name in ["slow", "fast", "event"]
            }

        batch_size = max(1, min(self._batch_size, len(rollout)))
        actor_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        approx_kl_total = 0.0
        clip_fraction_total = 0.0
        auxiliary_loss_total = 0.0
        heuristic_imitation_loss_total = 0.0
        mechanism_aux_loss_total = 0.0
        mechanism_entropy_bonus_total = 0.0
        effective_imitation_coef = self._effective_heuristic_imitation_coef()
        mechanism_guidance_rollout_stats = self._summarize_mechanism_guidance_annotations(
            mechanism_guidance_annotations,
            rollout,
        )
        update_steps = 0
        executed_epochs = 0
        early_stop_triggered = False

        for _ in range(self._train_epochs):
            permutation = torch.randperm(len(rollout), device=self._device)
            epoch_kl_values: list[float] = []
            for start_index in range(0, len(rollout), batch_size):
                batch_indices = permutation[start_index : start_index + batch_size]
                batch_index_list = batch_indices.detach().cpu().tolist()
                batch_states = [semantic_states[int(index)] for index in batch_index_list]
                batch_rows = [rollout[int(index)] for index in batch_index_list]
                batch_outputs = [self._forward_policy(state) for state in batch_states]
                if self._use_hierarchy:
                    batch_head_actions = {
                        head_name: tensor[batch_indices] for head_name, tensor in head_action_tensors.items()
                    }
                    batch_head_credits = {
                        head_name: tensor[batch_indices] for head_name, tensor in head_credit_tensors.items()
                    }
                    head_log_prob_outputs, head_entropy_outputs = self._compute_head_log_prob_and_entropy_tensors(
                        batch_outputs=batch_outputs,
                        head_action_tensors=batch_head_actions,
                    )
                    new_log_prob, entropy = self._compute_weighted_log_prob_and_entropy(
                        batch_outputs=batch_outputs,
                        head_action_tensors=batch_head_actions,
                        head_credit_tensors=batch_head_credits,
                        head_log_probs=head_log_prob_outputs,
                        head_entropies=head_entropy_outputs,
                    )
                    actor_loss = self._compute_hierarchical_actor_loss(
                        batch_states=batch_states,
                        head_log_probs=head_log_prob_outputs,
                        old_head_log_probs={
                            head_name: tensor[batch_indices]
                            for head_name, tensor in old_head_log_prob_tensors.items()
                        },
                        head_credit_tensors=batch_head_credits,
                        base_advantage=advantage_tensor[batch_indices],
                        event_advantage=event_advantage_tensor[batch_indices],
                    )
                else:
                    batch_action_masks = [action_masks[int(index)] for index in batch_index_list]
                    logits = torch.stack(
                        [
                            self._masked_flat_logits(output["flat_logits"], mask)
                            for output, mask in zip(batch_outputs, batch_action_masks, strict=False)
                        ],
                        dim=0,
                    )
                    distribution = Categorical(logits=logits)
                    new_log_prob = distribution.log_prob(action_tensor[batch_indices])
                    entropy = distribution.entropy().mean()
                    ratio = torch.exp(new_log_prob - old_log_prob_tensor[batch_indices])
                    surrogate_1 = ratio * advantage_tensor[batch_indices]
                    surrogate_2 = torch.clamp(
                        ratio,
                        1.0 - self._clip_ratio,
                        1.0 + self._clip_ratio,
                    ) * advantage_tensor[batch_indices]
                    actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_prediction = torch.stack([output["value"] for output in batch_outputs], dim=0)
                ratio = torch.exp(new_log_prob - old_log_prob_tensor[batch_indices])
                value_loss = torch.mean((return_tensor[batch_indices] - value_prediction) ** 2)
                auxiliary_loss = self._compute_auxiliary_loss(batch_states=batch_states, batch_outputs=batch_outputs)
                heuristic_imitation_loss = self._compute_heuristic_imitation_loss(
                    batch_outputs=batch_outputs,
                    batch_rows=batch_rows,
                )
                batch_annotations = [mechanism_guidance_annotations[int(index)] for index in batch_index_list]
                mechanism_aux_loss, mechanism_entropy_bonus = self._compute_mechanism_auxiliary_loss(
                    batch_outputs=batch_outputs,
                    batch_annotations=batch_annotations,
                )
                total_loss = (
                    actor_loss
                    + self._value_coef * value_loss
                    - self._entropy_coef * entropy
                    + self._auxiliary_coef * auxiliary_loss
                    + effective_imitation_coef * heuristic_imitation_loss
                    + effective_mechanism_aux_coef * mechanism_aux_loss
                    - effective_mechanism_entropy_coef * mechanism_entropy_bonus
                )

                self._optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=self._max_grad_norm)
                self._optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(old_log_prob_tensor[batch_indices] - new_log_prob)
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self._clip_ratio).float())

                actor_loss_total += float(actor_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                approx_kl_total += float(approx_kl.item())
                clip_fraction_total += float(clip_fraction.item())
                auxiliary_loss_total += float(auxiliary_loss.item())
                heuristic_imitation_loss_total += float(heuristic_imitation_loss.item())
                mechanism_aux_loss_total += float(mechanism_aux_loss.item())
                mechanism_entropy_bonus_total += float(mechanism_entropy_bonus.item())
                update_steps += 1
                epoch_kl_values.append(float(approx_kl.item()))
            executed_epochs += 1
            if self._kl_early_stop_enabled and self._target_kl > 0.0 and epoch_kl_values:
                epoch_kl_mean = float(sum(epoch_kl_values) / len(epoch_kl_values))
                if epoch_kl_mean >= self._target_kl:
                    early_stop_triggered = True
                    break

        self._update_count += 1
        denominator = max(update_steps, 1)
        explained_variance = 0.0
        if len(returns) > 1:
            return_variance = float(np.var(returns))
            if return_variance > 1e-8:
                explained_variance = 1.0 - float(np.var(returns - values) / return_variance)
        mechanism_prob_after_update = self._compute_mechanism_guided_action_prob_summary(
            semantic_states=semantic_states,
            annotations=mechanism_guidance_annotations,
        )

        return {
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "policy_update_skipped": False,
            "update_count": self._update_count,
            "collected_steps": len(rollout),
            "clip_ratio": self._clip_ratio,
            "entropy_coef": self._entropy_coef,
            "value_coef": self._value_coef,
            "auxiliary_coef": self._auxiliary_coef,
            "actor_loss": round(actor_loss_total / denominator, 6),
            "value_loss": round(value_loss_total / denominator, 6),
            "auxiliary_loss": round(auxiliary_loss_total / denominator, 6),
            "heuristic_imitation_coef": round(self._heuristic_imitation_coef, 6),
            "effective_heuristic_imitation_coef": round(effective_imitation_coef, 6),
            "heuristic_imitation_loss": round(heuristic_imitation_loss_total / denominator, 6),
            "heuristic_imitation_applied_count": int(imitation_rollout_stats["applied_count"]),
            "heuristic_imitation_match_count": int(imitation_rollout_stats["match_count"]),
            "heuristic_imitation_match_rate": round(float(imitation_rollout_stats["match_rate"]), 6),
            "mechanism_aux_coef": round(self._mechanism_aux_coef, 6),
            "effective_mechanism_aux_coef": round(effective_mechanism_aux_coef, 6),
            "mechanism_aux_loss_mean": round(mechanism_aux_loss_total / denominator, 6),
            "mechanism_entropy_coef": round(self._mechanism_entropy_coef, 6),
            "effective_mechanism_entropy_coef": round(effective_mechanism_entropy_coef, 6),
            "mechanism_head_entropy": round(mechanism_entropy_bonus_total / denominator, 6),
            "mechanism_retention_active": bool(retention_active),
            "mechanism_retention_start_update": int(self._mechanism_retention_start_update),
            "mechanism_aux_coef_floor_after_update": round(self._mechanism_aux_coef_floor_after_update, 6),
            "mechanism_window_weight_floor_after_update": round(
                self._mechanism_window_weight_floor_after_update,
                6,
            ),
            "mechanism_entropy_floor_after_update": round(self._mechanism_entropy_floor_after_update, 6),
            **mechanism_guidance_rollout_stats,
            **mechanism_prob_after_update,
            "policy_entropy": round(entropy_total / denominator, 6),
            "approx_kl": round(approx_kl_total / denominator, 6),
            "clip_fraction": round(clip_fraction_total / denominator, 6),
            "target_kl": round(self._target_kl, 6),
            "kl_early_stop_enabled": self._kl_early_stop_enabled,
            "early_stop_triggered": early_stop_triggered,
            "effective_train_epochs": executed_epochs,
            "advantage_mean_raw": round(advantage_mean, 6),
            "advantage_std_raw": round(advantage_std, 6),
            "event_advantage_mean_raw": round(event_advantage_mean, 6),
            "event_advantage_std_raw": round(event_advantage_std, 6),
            "value_mean": round(float(old_value_tensor.mean().item()), 6),
            "return_mean": round(float(return_tensor.mean().item()), 6),
            "explained_variance": round(explained_variance, 6),
            "learning_rate": round(float(self._optimizer.param_groups[0]["lr"]), 10),
            "head_action_usage": self._summarize_head_action_usage(rollout),
            "encoder_kind": self._encoder_kind,
            "use_hierarchy": self._use_hierarchy,
            "hierarchical_conditioning": self._hierarchical_conditioning,
            "graph_continuity_critic_enabled": self._graph_continuity_critic_enabled,
            "uncertainty_aware_event_scaling_enabled": self._uncertainty_aware_event_scaling_enabled,
            "uncertainty_aware_critic_enabled": self._uncertainty_aware_critic_enabled,
            "head_credit_enabled": self._head_credit_enabled,
            "head_credit_protocol": self._head_credit_protocol,
            "prediction_gate_min_leak": self._prediction_gate_min_leak,
            "slow_policy_credit_floor": self._slow_policy_credit_floor,
            "fast_policy_credit_floor": self._fast_policy_credit_floor,
            "event_policy_credit_floor": self._event_policy_credit_floor,
            "event_advantage_blend": self._event_advantage_blend,
            "slow_entropy_coef_scale": self._slow_entropy_coef_scale,
            "fast_entropy_coef_scale": self._fast_entropy_coef_scale,
            "event_entropy_coef_scale": self._event_entropy_coef_scale,
            "slow_entropy_credit_floor": self._slow_entropy_credit_floor,
            "fast_entropy_credit_floor": self._fast_entropy_credit_floor,
            "event_entropy_credit_floor": self._event_entropy_credit_floor,
            "event_logit_temperature": self._event_logit_temperature,
            "event_logit_temperature_final": self._event_logit_temperature_final,
            "event_temperature_decay_updates": self._event_temperature_decay_updates,
            "active_event_logit_temperature": self._current_event_logit_temperature(),
            "active_event_logit_sharpening_scale": self._current_event_logit_sharpening_scale(),
            "event_logit_sharpening_final_scale": self._event_logit_sharpening_final_scale,
            "event_logit_sharpening_timing_gain": self._event_logit_sharpening_timing_gain,
            "event_actor_loss_extra_gain": self._event_actor_loss_extra_gain,
            "event_prepare_margin_boost": self._event_prepare_margin_boost,
            "temporal_consistency_coef": self._temporal_consistency_coef,
            "temporal_prepare_lead_steps": self._temporal_prepare_lead_steps,
            "temporal_prepare_sigma": self._temporal_prepare_sigma,
            "temporal_prepare_activation_threshold": self._temporal_prepare_activation_threshold,
            "deterministic_high_prepare_override_enabled": self._deterministic_high_prepare_override_enabled,
            "deterministic_high_prepare_threshold": self._deterministic_high_prepare_threshold,
            "deterministic_high_urgency_threshold": self._deterministic_high_urgency_threshold,
            "deterministic_high_prepare_relaxed_margin": self._deterministic_high_prepare_relaxed_margin,
            "predictive_prepare_hard_override_enabled": self._predictive_prepare_hard_override_enabled,
            "predictive_prepare_hard_override_score_threshold": self._predictive_prepare_hard_override_score_threshold,
            "predictive_prepare_hard_override_confidence_threshold": self._predictive_prepare_hard_override_confidence_threshold,
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
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        torch.save(checkpoint, output_path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(Path(path), map_location=self._device)
        self._network.load_state_dict(checkpoint["network_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._update_count = int(checkpoint.get("update_count", 0))

    def _extract_semantic_state(self, info: dict[str, Any] | None) -> dict[str, Any]:
        semantic_state = (info or {}).get("semantic_state")
        if semantic_state is None:
            raise ValueError(f"{self.agent_name} 需要 info['semantic_state'] 才能做图结构编码。")
        return semantic_state

    def _extract_action_mask(self, info: dict[str, Any] | None) -> list[bool] | None:
        raw_mask = (info or {}).get("action_mask")
        if raw_mask is None:
            return None
        if not isinstance(raw_mask, (list, tuple)):
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
        action_id = int(env_action)
        return bool(0 <= action_id < len(action_mask or []) and (action_mask or [])[action_id])

    def _mask_logits(
        self,
        logits: torch.Tensor,
        mask: list[bool] | torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return logits
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.to(device=logits.device, dtype=torch.bool)
        else:
            mask_tensor = torch.as_tensor(list(mask), dtype=torch.bool, device=logits.device)
        if mask_tensor.numel() != logits.shape[-1] or not bool(mask_tensor.any().item()):
            return logits
        return logits.masked_fill(~mask_tensor, -1.0e9)

    def _masked_flat_logits(self, logits: torch.Tensor, action_mask: list[bool] | None) -> torch.Tensor:
        return self._mask_logits(logits, action_mask)

    def _best_valid_env_action_from_policy(
        self,
        policy_output: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> int:
        if not self._action_mask_has_valid_action(action_mask):
            return 3
        assert action_mask is not None
        if not self._use_hierarchy:
            masked_logits = self._masked_flat_logits(policy_output["flat_logits"], action_mask)
            return int(torch.argmax(masked_logits, dim=-1).item())
        masked_scores = self._masked_flat_logits(self._hierarchical_env_action_scores(policy_output), action_mask)
        return int(torch.argmax(masked_scores, dim=-1).item())

    def _hierarchical_env_action_scores(self, policy_output: dict[str, Any]) -> torch.Tensor:
        event_log_probs = torch.log_softmax(policy_output["event_logits"], dim=-1)
        slow_log_probs = torch.log_softmax(policy_output["slow_logits"], dim=-1)
        fast_log_probs = torch.log_softmax(policy_output["fast_logits"], dim=-1)
        return torch.stack(
            [
                event_log_probs[0] + slow_log_probs[1],
                event_log_probs[0] + slow_log_probs[2],
                event_log_probs[0] + slow_log_probs[0] + fast_log_probs[1],
                event_log_probs[0] + slow_log_probs[0] + fast_log_probs[0],
                event_log_probs[1],
            ],
            dim=0,
        )

    def _project_head_actions_to_valid_env_action(
        self,
        selected_actions: dict[str, int],
        policy_output: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> dict[str, int]:
        env_action, _ = 聚合层级动作(
            head_actions=selected_actions,
            use_hierarchy=self._use_hierarchy,
            event_head_enabled=self._event_head_enabled,
            adapter_prefetch_enabled=self._adapter_prefetch_enabled,
        )
        if self._is_env_action_valid(env_action, action_mask):
            return selected_actions
        valid_env_action = self._best_valid_env_action_from_policy(policy_output, action_mask)
        return self._head_targets_for_env_action(valid_env_action)

    def _forward_policy(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        policy_output = self._network.forward_single(
            semantic_state,
            event_logit_temperature=self._current_event_logit_temperature(),
        )
        return self._apply_policy_adjustments(policy_output, semantic_state)

    def _selected_action_statistics(
        self,
        policy_output: dict[str, Any],
        selected_actions: dict[str, int],
        action_mask: list[bool] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[float]]]:
        if not self._use_hierarchy:
            flat_action = torch.tensor(int(selected_actions.get("flat", 3)), dtype=torch.long, device=self._device)
            flat_logits = self._masked_flat_logits(policy_output["flat_logits"], action_mask)
            distribution = Categorical(logits=flat_logits)
            return (
                {"flat": distribution.log_prob(flat_action)},
                {"flat": distribution.entropy()},
                {"flat": [round(float(item), 6) for item in torch.softmax(flat_logits, dim=-1).tolist()]},
            )

        head_log_probs: dict[str, torch.Tensor] = {}
        head_entropies: dict[str, torch.Tensor] = {}
        action_prob_payload: dict[str, list[float]] = {}
        for head_name in ["slow", "fast", "event"]:
            logits = policy_output[f"{head_name}_logits"]
            action_tensor = torch.tensor(int(selected_actions.get(head_name, 0)), dtype=torch.long, device=self._device)
            distribution = Categorical(logits=logits)
            head_log_probs[head_name] = distribution.log_prob(action_tensor)
            head_entropies[head_name] = distribution.entropy()
            action_prob_payload[head_name] = [round(float(item), 6) for item in torch.softmax(logits, dim=-1).tolist()]
        return head_log_probs, head_entropies, action_prob_payload

    def _should_hard_apply_continuity_guard(
        self,
        *,
        selected_actions: dict[str, int],
        guard_info: dict[str, Any],
    ) -> bool:
        return bool(
            self._continuity_guard_hard_override_enabled
            and self._use_hierarchy
            and guard_info.get("guard_triggered", False)
            and int(selected_actions.get("slow", 0)) == 2
        )

    def _current_event_logit_temperature(self) -> float:
        if self._event_temperature_decay_updates <= 0:
            return self._event_logit_temperature
        schedule_progress = min(
            float(self._update_count) / float(max(self._event_temperature_decay_updates, 1)),
            1.0,
        )
        current_temperature = (
            self._event_logit_temperature
            + (self._event_logit_temperature_final - self._event_logit_temperature) * schedule_progress
        )
        return max(float(current_temperature), 0.25)

    def _current_event_logit_sharpening_scale(self) -> float:
        if self._event_logit_sharpening_final_scale <= 1.0:
            return 1.0
        if self._event_temperature_decay_updates <= 0:
            return self._event_logit_sharpening_final_scale
        schedule_progress = min(
            float(self._update_count) / float(max(self._event_temperature_decay_updates, 1)),
            1.0,
        )
        current_scale = 1.0 + (self._event_logit_sharpening_final_scale - 1.0) * schedule_progress
        return max(float(current_scale), 1.0)

    def _build_prediction_reliability_summary(self, semantic_state: dict[str, Any]) -> dict[str, float]:
        return build_prediction_reliability_summary(
            semantic_state,
            prediction_gate_min_leak=self._prediction_gate_min_leak,
        )

    def _build_event_scaling_summary(
        self,
        *,
        semantic_state: dict[str, Any],
        timing_features: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not (
            self._uncertainty_aware_event_scaling_enabled
            and self._use_prediction_features
            and self._use_uncertainty_signal
            and self._use_hierarchy
            and self._event_head_enabled
        ):
            return {
                "event_actor_weight_scale": 1.0,
                "event_margin_scale": 1.0,
                "event_sharpen_factor": 1.0,
                "event_aggressive_support": 1.0,
                "continuity_pressure_score": 1.0,
                "conditional_conservative_pressure": 0.0,
                "future_switch_evidence": 0.0,
                "path_pressure_score": 0.0,
                "target_adapter_support": 0.0,
            }
        reliability_summary = self._build_prediction_reliability_summary(semantic_state)
        reliability_timing_alignment = float(reliability_summary.get("reliability_timing_alignment", 0.0))
        conservative_prepare_pressure = float(reliability_summary.get("conservative_prepare_pressure", 0.0))
        if timing_features is None:
            timing_support = float(reliability_summary.get("timing_support", 0.0))
        else:
            timing_support = max(
                float(timing_features.get("prepare_window_score", 0.0)),
                float(timing_features.get("temporal_urgency", 0.0)),
            )
        continuity_features = build_graph_continuity_critic_features(
            semantic_state,
            prediction_gate_min_leak=self._prediction_gate_min_leak,
        )
        future_switch_evidence = _clamp01(
            0.45 * float(continuity_features.get("predicted_path_switch_ratio", 0.0))
            + 0.20 * float(continuity_features.get("future_unique_rsu_ratio", 0.0))
            + 0.20 * float(continuity_features.get("predicted_target_differs", 0.0))
            + 0.15 * float(continuity_features.get("predicted_next_differs", 0.0))
        )
        path_pressure_score = _clamp01(
            0.65 * float(continuity_features.get("critical_path_length_norm", 0.0))
            + 0.35 * float(continuity_features.get("frontier_width_ratio", 0.0))
        )
        target_adapter_support = _clamp01(
            max(
                float(continuity_features.get("target_has_adapter", 0.0))
                - float(continuity_features.get("current_has_adapter", 0.0)),
                0.0,
            )
        )
        continuity_pressure_score = _clamp01(
            0.50 * future_switch_evidence
            + 0.35 * path_pressure_score
            + 0.15 * target_adapter_support
        )
        event_aggressive_support = _clamp01(
            0.50 * reliability_timing_alignment
            + 0.30 * continuity_pressure_score
            + 0.20 * timing_support
        )
        conditional_conservative_pressure = _clamp01(
            conservative_prepare_pressure
            * (1.0 - 0.90 * continuity_pressure_score)
            * (0.65 + 0.35 * (1.0 - event_aggressive_support))
        )
        event_actor_weight_scale = max(
            0.8,
            min(
                float(
                    0.78
                    + 0.45 * event_aggressive_support
                    + 0.40 * continuity_pressure_score
                    - 0.22 * conditional_conservative_pressure
                ),
                1.35,
            ),
        )
        event_margin_scale = max(
            0.7,
            min(
                float(
                    0.72
                    + 0.35 * event_aggressive_support
                    + 0.55 * continuity_pressure_score
                    - 0.15 * conditional_conservative_pressure
                ),
                1.40,
            ),
        )
        event_sharpen_factor = max(
            0.55,
            min(
                float(
                    0.55
                    + 0.30 * reliability_timing_alignment
                    + 0.75 * continuity_pressure_score
                    - 0.15 * conditional_conservative_pressure
                ),
                1.25,
            ),
        )
        return {
            "event_actor_weight_scale": event_actor_weight_scale,
            "event_margin_scale": event_margin_scale,
            "event_sharpen_factor": event_sharpen_factor,
            "event_aggressive_support": event_aggressive_support,
            "continuity_pressure_score": continuity_pressure_score,
            "conditional_conservative_pressure": conditional_conservative_pressure,
            "future_switch_evidence": future_switch_evidence,
            "path_pressure_score": path_pressure_score,
            "target_adapter_support": target_adapter_support,
        }

    def _compute_event_reliability_scale(
        self,
        *,
        semantic_state: dict[str, Any],
        timing_features: dict[str, float] | None = None,
    ) -> float:
        scaling_summary = self._build_event_scaling_summary(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        return float(scaling_summary["event_actor_weight_scale"])

    def _build_event_reliability_scale_tensor(self, batch_states: list[dict[str, Any]]) -> torch.Tensor:
        if not batch_states:
            return torch.empty(0, dtype=torch.float32, device=self._device)
        scales = [
            self._compute_event_reliability_scale(semantic_state=state)
            for state in batch_states
        ]
        return torch.as_tensor(scales, dtype=torch.float32, device=self._device)

    def _compute_event_prepare_margin_boost(
        self,
        *,
        semantic_state: dict[str, Any],
        timing_features: dict[str, float],
    ) -> float:
        if (
            not self._use_hierarchy
            or not self._event_head_enabled
            or self._event_prepare_margin_boost <= 1e-8
        ):
            return 0.0
        if not self._semantic_state_has_valid_predicted_handoff_target(semantic_state):
            return 0.0
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        timing_support = max(
            float(timing_features.get("prepare_window_score", 0.0)),
            float(timing_features.get("temporal_urgency", 0.0)),
        )
        if timing_support < self._temporal_prepare_activation_threshold:
            return 0.0
        if self._use_uncertainty_signal:
            confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
            uncertainty = max(
                0.0,
                min(float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0)), 1.0),
            )
            urgency_support = 0.7 + 0.3 * max(0.0, min(float(timing_features.get("temporal_urgency", 0.0)), 1.0))
            gate_value = max(0.0, min(1.0, confidence * (1.0 - uncertainty) * urgency_support))
            diagnostic_gate_threshold = max(
                self._prediction_gate_min_leak if self._use_uncertainty_signal else 0.0,
                1e-6,
            )
            if gate_value < diagnostic_gate_threshold:
                return 0.0
        reliability_scale = self._compute_event_reliability_scale(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        scaling_summary = self._build_event_scaling_summary(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        normalized_support = (timing_support - self._temporal_prepare_activation_threshold) / max(
            1.0 - self._temporal_prepare_activation_threshold,
            1e-6,
        )
        normalized_support = max(0.0, min(float(normalized_support), 1.0))
        return (
            self._event_prepare_margin_boost
            * normalized_support
            * reliability_scale
            * float(scaling_summary["event_margin_scale"])
        )

    def _apply_deterministic_temporal_smoothing(
        self,
        *,
        semantic_state: dict[str, Any],
        policy_output: dict[str, Any],
        selected_actions: dict[str, int],
        deterministic: bool,
    ) -> dict[str, Any]:
        if not (
            deterministic
            and self._use_hierarchy
            and self._event_head_enabled
            and self._deterministic_temporal_smoothing_enabled
        ):
            self._deterministic_temporal_streak = 0
            return {
                "enabled": False,
                "forced_event_prepare": False,
                "override_triggered": False,
                "borderline_triggered": False,
            }
        current_time_index = int(semantic_state.get("time_index", 0) or 0)
        if self._last_deterministic_time_index is None or current_time_index <= self._last_deterministic_time_index:
            self._deterministic_temporal_streak = 0
        self._last_deterministic_time_index = current_time_index
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        event_probs = torch.softmax(policy_output["event_logits"], dim=-1)
        prepare_prob = float(event_probs[1].item())
        margin = float((policy_output["event_logits"][1] - policy_output["event_logits"][0]).item())
        prepare_window_score = float(timing_features.get("prepare_window_score", 0.0))
        temporal_urgency = float(timing_features.get("temporal_urgency", 0.0))
        temporal_score = max(prepare_window_score, temporal_urgency)
        predicted_handoff_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        prediction_confidence = 0.0
        if isinstance(predictions, dict) and vehicle_id:
            prediction_confidence = float(
                predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0) or 0.0
            )
        borderline = bool(
            selected_actions.get("event", 0) == 0
            and prepare_prob >= self._deterministic_event_borderline_prob
            and margin >= self._deterministic_event_borderline_margin
            and temporal_score >= self._deterministic_temporal_urgency_floor
        )
        forced_event_prepare = False
        override_triggered = False
        borderline_triggered = False
        predictive_prepare_override = bool(
            selected_actions.get("event", 0) == 0
            and self._predictive_prepare_hard_override_enabled
            and predicted_handoff_target_valid
            and temporal_score >= self._predictive_prepare_hard_override_score_threshold
            and prediction_confidence >= self._predictive_prepare_hard_override_confidence_threshold
        )
        high_prepare_override = bool(
            selected_actions.get("event", 0) == 0
            and self._deterministic_high_prepare_override_enabled
            and predicted_handoff_target_valid
            and prepare_window_score >= self._deterministic_high_prepare_threshold
            and temporal_urgency >= self._deterministic_high_urgency_threshold
            and margin >= self._deterministic_high_prepare_relaxed_margin
        )
        if selected_actions.get("event", 0) == 1:
            self._deterministic_temporal_streak = 0
        elif predictive_prepare_override or high_prepare_override:
            selected_actions["event"] = 1
            forced_event_prepare = True
            override_triggered = True
            self._deterministic_temporal_streak = 0
        elif borderline:
            self._deterministic_temporal_streak += 1
            if self._deterministic_temporal_streak >= self._deterministic_temporal_smoothing_steps:
                selected_actions["event"] = 1
                forced_event_prepare = True
                borderline_triggered = True
                self._deterministic_temporal_streak = 0
        else:
            self._deterministic_temporal_streak = 0
        return {
            "enabled": True,
            "forced_event_prepare": forced_event_prepare,
            "override_triggered": override_triggered,
            "borderline_triggered": borderline_triggered,
            "borderline": borderline,
            "predictive_prepare_override_eligible": predictive_prepare_override,
            "high_prepare_override_eligible": high_prepare_override,
            "predicted_handoff_target_valid": predicted_handoff_target_valid,
            "prediction_confidence": round(prediction_confidence, 6),
            "prepare_prob": round(prepare_prob, 6),
            "event_margin": round(margin, 6),
            "temporal_score": round(temporal_score, 6),
            "prepare_window_score": round(prepare_window_score, 6),
            "temporal_urgency": round(temporal_urgency, 6),
            "streak_after_step": int(self._deterministic_temporal_streak),
        }

    def _semantic_state_has_valid_predicted_handoff_target(self, semantic_state: dict[str, Any]) -> bool:
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predicted_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        return bool(predicted_target)

    def _semantic_state_has_raw_handoff_candidate(self, semantic_state: dict[str, Any]) -> bool:
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
        sequence_has_change = any(
            next_rsu_id is not None and (current_rsu_id is None or next_rsu_id != current_rsu_id)
            for next_rsu_id in next_sequence
        )
        next_hop_has_change = bool(
            predicted_next_rsu_id is not None and (current_rsu_id is None or predicted_next_rsu_id != current_rsu_id)
        )
        return bool(predicted_target or next_hop_has_change or sequence_has_change)

    def _build_prediction_target_diagnostics(
        self,
        *,
        semantic_state: dict[str, Any],
        temporal_urgency: float,
        predicted_handoff_target_valid: bool,
    ) -> dict[str, Any]:
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, primary_resolution = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        prediction_state_available = bool(isinstance(predictions, dict) and predictions)
        predictor_invoked = bool(prediction_state_available and str(predictions.get("predictor_name", "")))
        next_rsu_sequence_map = predictions.get("next_rsu_sequence", {}) if isinstance(predictions, dict) else {}
        vehicle_sequence = list(next_rsu_sequence_map.get(vehicle_id, [])) if isinstance(next_rsu_sequence_map, dict) else []
        next_rsu_non_null_count = sum(1 for rsu_id in vehicle_sequence if rsu_id is not None)
        non_null_sequence = [rsu_id for rsu_id in vehicle_sequence if rsu_id is not None]
        predicted_first_non_current_rsu: str | None = None
        predicted_first_non_current_eta = 0
        for step_index, rsu_id in enumerate(vehicle_sequence, start=1):
            if rsu_id is None:
                continue
            if primary_vehicle.get("associated_rsu_id") is None or rsu_id != primary_vehicle.get("associated_rsu_id"):
                predicted_first_non_current_rsu = str(rsu_id)
                predicted_first_non_current_eta = int(step_index)
                break
        predicted_sequence_all_null = bool(len(vehicle_sequence) > 0 and next_rsu_non_null_count <= 0)
        predicted_sequence_contains_other_rsu = bool(predicted_first_non_current_rsu)
        predicted_sequence_all_current_rsu = bool(
            len(non_null_sequence) > 0
            and not predicted_sequence_contains_other_rsu
        )
        predicted_sequence_preview = [
            None if rsu_id is None else str(rsu_id)
            for rsu_id in vehicle_sequence[:6]
        ]
        raw_handoff_candidate = self._semantic_state_has_raw_handoff_candidate(semantic_state)
        confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
        uncertainty = max(0.0, min(float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0)), 1.0))
        urgency_support = 1.0
        if self._use_uncertainty_signal:
            urgency_support = 0.7 + 0.3 * max(0.0, min(float(temporal_urgency), 1.0))
        if not self._use_prediction_features:
            raw_gate_value = 0.0
        elif not raw_handoff_candidate:
            raw_gate_value = 0.0
        elif not self._use_uncertainty_signal:
            raw_gate_value = 1.0
        else:
            raw_gate_value = max(0.0, min(1.0, confidence * (1.0 - uncertainty) * urgency_support))
        diagnostic_gate_threshold = self._prediction_gate_min_leak if self._use_uncertainty_signal else 0.0
        if not self._use_prediction_features:
            gate_pass = False
        elif not raw_handoff_candidate:
            gate_pass = False
        elif not self._use_uncertainty_signal:
            gate_pass = True
        else:
            gate_pass = bool(raw_gate_value >= max(diagnostic_gate_threshold, 1e-6))

        invalid_reason = "none"
        candidate_block_reason = "none"
        if not prediction_state_available or not predictor_invoked:
            candidate_block_reason = "missing_prediction_state"
        elif len(vehicle_sequence) <= 0:
            candidate_block_reason = "no_next_rsu"
        elif next_rsu_non_null_count <= 0:
            candidate_block_reason = "no_next_rsu"
        elif not raw_handoff_candidate:
            candidate_block_reason = "same_rsu"

        if not raw_handoff_candidate:
            invalid_reason = "no_candidate"
        elif candidate_block_reason == "missing_prediction_state":
            invalid_reason = "missing_prediction_state"
        elif gate_pass and not predicted_handoff_target_valid:
            invalid_reason = "valid_chain_lost"
        elif not gate_pass and self._use_prediction_features and self._use_uncertainty_signal:
            required_confidence = diagnostic_gate_threshold / max((1.0 - uncertainty) * urgency_support, 1e-6)
            max_allowed_uncertainty = 1.0 - (diagnostic_gate_threshold / max(confidence * urgency_support, 1e-6))
            if confidence + 1e-6 < required_confidence:
                invalid_reason = "low_confidence"
            elif uncertainty - 1e-6 > max_allowed_uncertainty:
                invalid_reason = "high_uncertainty"
            else:
                invalid_reason = "gate_below_threshold"

        return {
            "predictor_invoked": bool(predictor_invoked),
            "prediction_state_available": bool(prediction_state_available),
            "prediction_sequence_horizon": int(len(vehicle_sequence)),
            "next_rsu_non_null_count": int(next_rsu_non_null_count),
            "candidate_block_reason": candidate_block_reason,
            "primary_vehicle_id": primary_resolution["primary_vehicle_id"],
            "primary_vehicle_present": bool(primary_resolution["primary_vehicle_present"]),
            "primary_vehicle_reordered_to_front": bool(
                primary_resolution["primary_vehicle_reordered_to_front"]
            ),
            "first_vehicle_id": primary_resolution["first_vehicle_id"],
            "first_vehicle_matches_primary": bool(primary_resolution["first_vehicle_matches_primary"]),
            "primary_vehicle_lookup_fallback": bool(
                primary_resolution["primary_vehicle_lookup_fallback"]
            ),
            "primary_vehicle_resolution_warning": str(
                primary_resolution["primary_vehicle_resolution_warning"]
            ),
            "current_rsu_id": primary_vehicle.get("associated_rsu_id"),
            "predicted_sequence_preview": predicted_sequence_preview,
            "predicted_sequence_all_null": bool(predicted_sequence_all_null),
            "predicted_sequence_all_current_rsu": bool(predicted_sequence_all_current_rsu),
            "predicted_sequence_contains_other_rsu": bool(predicted_sequence_contains_other_rsu),
            "predicted_first_non_current_rsu": predicted_first_non_current_rsu,
            "predicted_first_non_current_eta": int(predicted_first_non_current_eta),
            "raw_handoff_candidate": bool(raw_handoff_candidate),
            "prediction_confidence": float(confidence),
            "prediction_uncertainty": float(uncertainty),
            "urgency_support": float(urgency_support),
            "prediction_gate_value": float(raw_gate_value),
            "gate_pass": bool(gate_pass),
            "invalid_reason": invalid_reason,
        }

    def _sample_actions(
        self,
        policy_output: dict[str, Any],
        deterministic: bool,
        action_mask: list[bool] | None = None,
    ) -> tuple[dict[str, int], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[float]], dict[str, Any]]:
        if not self._use_hierarchy:
            flat_logits = self._masked_flat_logits(policy_output["flat_logits"], action_mask)
            distribution = Categorical(logits=flat_logits)
            if deterministic:
                flat_action = torch.argmax(flat_logits, dim=-1)
            else:
                flat_action = distribution.sample()
            selected_actions = {"flat": int(flat_action.item())}
            env_action, aggregation_reason = 聚合层级动作(
                head_actions=selected_actions,
                use_hierarchy=self._use_hierarchy,
                event_head_enabled=self._event_head_enabled,
                adapter_prefetch_enabled=self._adapter_prefetch_enabled,
            )
            projection_info = self._build_action_projection_info(
                raw_actions=selected_actions,
                projected_actions=selected_actions,
                raw_env_action=env_action,
                raw_aggregation_reason=aggregation_reason,
                projected_env_action=env_action,
                projected_aggregation_reason=aggregation_reason,
                action_mask=action_mask,
            )
            return (
                selected_actions,
                {"flat": distribution.log_prob(flat_action)},
                {"flat": distribution.entropy()},
                {"flat": [round(float(item), 6) for item in torch.softmax(flat_logits, dim=-1).tolist()]},
                projection_info,
            )

        if self._action_mask_has_valid_action(action_mask):
            assert action_mask is not None
            env_scores = self._masked_flat_logits(
                self._hierarchical_env_action_scores(policy_output),
                action_mask,
            )
            distribution = Categorical(logits=env_scores)
            if deterministic:
                env_action_tensor = torch.argmax(env_scores, dim=-1)
            else:
                env_action_tensor = distribution.sample()
            env_action = int(env_action_tensor.item())
            selected_actions = self._head_targets_for_env_action(env_action)
            projected_env_action, projected_aggregation_reason = 聚合层级动作(
                head_actions=selected_actions,
                use_hierarchy=self._use_hierarchy,
                event_head_enabled=self._event_head_enabled,
                adapter_prefetch_enabled=self._adapter_prefetch_enabled,
            )
            head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                policy_output=policy_output,
                selected_actions=selected_actions,
                action_mask=action_mask,
            )
            projection_info = self._build_action_projection_info(
                raw_actions=selected_actions,
                projected_actions=selected_actions,
                raw_env_action=projected_env_action,
                raw_aggregation_reason=projected_aggregation_reason,
                projected_env_action=projected_env_action,
                projected_aggregation_reason=projected_aggregation_reason,
                action_mask=action_mask,
            )
            projection_info["masked_hierarchical_env_action_sampling"] = True
            projection_info["masked_env_action_log_prob"] = round(
                float(distribution.log_prob(env_action_tensor).item()),
                6,
            )
            projection_info["masked_env_action_probs"] = [
                round(float(item), 6)
                for item in torch.softmax(env_scores, dim=-1).tolist()
            ]
            return selected_actions, head_log_probs, head_entropies, action_prob_payload, projection_info

        selected_actions: dict[str, int] = {}
        head_log_probs: dict[str, torch.Tensor] = {}
        head_entropies: dict[str, torch.Tensor] = {}
        action_prob_payload: dict[str, list[float]] = {}
        for head_name in ["slow", "fast", "event"]:
            logits = policy_output[f"{head_name}_logits"]
            distribution = Categorical(logits=logits)
            if deterministic:
                action_tensor = torch.argmax(logits, dim=-1)
            else:
                action_tensor = distribution.sample()
            selected_actions[head_name] = int(action_tensor.item())
            head_log_probs[head_name] = distribution.log_prob(action_tensor)
            head_entropies[head_name] = distribution.entropy()
            action_prob_payload[head_name] = [round(float(item), 6) for item in torch.softmax(logits, dim=-1).tolist()]
        raw_actions = dict(selected_actions)
        raw_env_action, raw_aggregation_reason = 聚合层级动作(
            head_actions=raw_actions,
            use_hierarchy=self._use_hierarchy,
            event_head_enabled=self._event_head_enabled,
            adapter_prefetch_enabled=self._adapter_prefetch_enabled,
        )
        projected_actions = self._project_head_actions_to_valid_env_action(
            selected_actions=selected_actions,
            policy_output=policy_output,
            action_mask=action_mask,
        )
        projected_env_action, projected_aggregation_reason = 聚合层级动作(
            head_actions=projected_actions,
            use_hierarchy=self._use_hierarchy,
            event_head_enabled=self._event_head_enabled,
            adapter_prefetch_enabled=self._adapter_prefetch_enabled,
        )
        head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
            policy_output=policy_output,
            selected_actions=projected_actions,
            action_mask=action_mask,
        )
        projection_info = self._build_action_projection_info(
            raw_actions=raw_actions,
            projected_actions=projected_actions,
            raw_env_action=raw_env_action,
            raw_aggregation_reason=raw_aggregation_reason,
            projected_env_action=projected_env_action,
            projected_aggregation_reason=projected_aggregation_reason,
            action_mask=action_mask,
        )
        return projected_actions, head_log_probs, head_entropies, action_prob_payload, projection_info

    def _build_action_projection_info(
        self,
        *,
        raw_actions: dict[str, int],
        projected_actions: dict[str, int],
        raw_env_action: int,
        raw_aggregation_reason: str,
        projected_env_action: int,
        projected_aggregation_reason: str,
        action_mask: list[bool] | None,
    ) -> dict[str, Any]:
        raw_valid = self._is_env_action_valid(raw_env_action, action_mask)
        projected_valid = self._is_env_action_valid(projected_env_action, action_mask)
        return {
            "raw_head_actions": dict(raw_actions),
            "projected_head_actions": dict(projected_actions),
            "raw_env_action": int(raw_env_action),
            "projected_env_action": int(projected_env_action),
            "raw_aggregation_reason": raw_aggregation_reason,
            "projected_aggregation_reason": projected_aggregation_reason,
            "raw_env_action_valid": bool(raw_valid),
            "projected_env_action_valid": bool(projected_valid),
            "projection_applied": bool(
                not raw_valid
                or int(raw_env_action) != int(projected_env_action)
                or dict(raw_actions) != dict(projected_actions)
            ),
            "invalid_attempt_count": 0 if raw_valid else 1,
            "valid_action_count": self._valid_action_count(action_mask),
        }

    def _compute_head_log_prob_and_entropy_tensors(
        self,
        batch_outputs: list[dict[str, Any]],
        head_action_tensors: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        head_log_probs: dict[str, torch.Tensor] = {}
        head_entropies: dict[str, torch.Tensor] = {}
        for head_name in ["slow", "fast", "event"]:
            logits = torch.stack([output[f"{head_name}_logits"] for output in batch_outputs], dim=0)
            distribution = Categorical(logits=logits)
            head_log_probs[head_name] = distribution.log_prob(head_action_tensors[head_name])
            head_entropies[head_name] = distribution.entropy()
        return head_log_probs, head_entropies

    def _compute_weighted_log_prob_and_entropy(
        self,
        batch_outputs: list[dict[str, Any]],
        head_action_tensors: dict[str, torch.Tensor],
        head_credit_tensors: dict[str, torch.Tensor],
        head_log_probs: dict[str, torch.Tensor] | None = None,
        head_entropies: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head_log_probs is None or head_entropies is None:
            head_log_probs, head_entropies = self._compute_head_log_prob_and_entropy_tensors(
                batch_outputs=batch_outputs,
                head_action_tensors=head_action_tensors,
            )
        return self._aggregate_weighted_head_metrics(
            head_log_probs=head_log_probs,
            head_entropies=head_entropies,
            head_credit_tensors=head_credit_tensors,
        )

    def _aggregate_weighted_head_metrics(
        self,
        *,
        head_log_probs: dict[str, torch.Tensor],
        head_entropies: dict[str, torch.Tensor],
        head_credit_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_log_prob: torch.Tensor | None = None
        weighted_entropy_sum: torch.Tensor | None = None
        weight_sum: torch.Tensor | None = None
        for head_name in ["slow", "fast", "event"]:
            head_log_prob = head_log_probs[head_name]
            head_weight = self._resolve_actor_weight(head_name=head_name, base_weight=head_credit_tensors[head_name])
            weighted_log_prob = head_log_prob * head_weight
            joint_log_prob = weighted_log_prob if joint_log_prob is None else joint_log_prob + weighted_log_prob
            entropy_weight = self._resolve_entropy_weight(head_name=head_name, base_weight=head_credit_tensors[head_name])
            weighted_entropy = head_entropies[head_name] * entropy_weight
            weighted_entropy_sum = weighted_entropy if weighted_entropy_sum is None else weighted_entropy_sum + weighted_entropy
            weight_sum = entropy_weight if weight_sum is None else weight_sum + entropy_weight
        assert joint_log_prob is not None
        assert weighted_entropy_sum is not None
        assert weight_sum is not None
        entropy = (weighted_entropy_sum / torch.clamp(weight_sum, min=1e-6)).mean()
        return joint_log_prob, entropy

    def _compute_hierarchical_actor_loss(
        self,
        *,
        batch_states: list[dict[str, Any]],
        head_log_probs: dict[str, torch.Tensor],
        old_head_log_probs: dict[str, torch.Tensor],
        head_credit_tensors: dict[str, torch.Tensor],
        base_advantage: torch.Tensor,
        event_advantage: torch.Tensor,
    ) -> torch.Tensor:
        surrogate_sum: torch.Tensor | None = None
        weight_sum: torch.Tensor | None = None
        event_reliability_scales = self._build_event_reliability_scale_tensor(batch_states)
        for head_name in ["slow", "fast", "event"]:
            actor_weight = self._resolve_actor_weight(head_name=head_name, base_weight=head_credit_tensors[head_name])
            if head_name == "event" and len(event_reliability_scales) > 0:
                actor_weight = actor_weight * event_reliability_scales
            if head_name == "event" and self._event_actor_loss_extra_gain > 1.0:
                if len(event_reliability_scales) > 0:
                    actor_weight = actor_weight * (
                        1.0 + (self._event_actor_loss_extra_gain - 1.0) * event_reliability_scales
                    )
                else:
                    actor_weight = actor_weight * self._event_actor_loss_extra_gain
            head_advantage = event_advantage if head_name == "event" else base_advantage
            if head_name == "event" and self._event_advantage_blend < 1.0:
                head_advantage = (
                    self._event_advantage_blend * event_advantage
                    + (1.0 - self._event_advantage_blend) * base_advantage
                )
            ratio = torch.exp(head_log_probs[head_name] - old_head_log_probs[head_name])
            surrogate_1 = ratio * head_advantage
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - self._clip_ratio,
                1.0 + self._clip_ratio,
            ) * head_advantage
            head_surrogate = torch.min(surrogate_1, surrogate_2) * actor_weight
            surrogate_sum = head_surrogate if surrogate_sum is None else surrogate_sum + head_surrogate
            weight_sum = actor_weight if weight_sum is None else weight_sum + actor_weight
        assert surrogate_sum is not None
        assert weight_sum is not None
        return -(surrogate_sum / torch.clamp(weight_sum, min=1e-6)).mean()

    def _combine_head_statistics(
        self,
        head_log_probs: dict[str, torch.Tensor],
        head_entropies: dict[str, torch.Tensor],
        head_credit_weights: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_hierarchy:
            return head_log_probs["flat"], head_entropies["flat"]
        joint_log_prob: torch.Tensor | None = None
        weighted_entropy_sum: torch.Tensor | None = None
        total_weight = 0.0
        for head_name in ["slow", "fast", "event"]:
            weight = self._resolve_actor_weight(head_name=head_name, base_weight=float(head_credit_weights.get(head_name, 1.0)))
            weighted_log_prob = head_log_probs[head_name] * weight
            joint_log_prob = weighted_log_prob if joint_log_prob is None else joint_log_prob + weighted_log_prob
            entropy_weight = self._resolve_entropy_weight(head_name=head_name, base_weight=float(head_credit_weights.get(head_name, 1.0)))
            weighted_entropy = head_entropies[head_name] * entropy_weight
            weighted_entropy_sum = weighted_entropy if weighted_entropy_sum is None else weighted_entropy_sum + weighted_entropy
            total_weight += float(entropy_weight)
        assert joint_log_prob is not None
        assert weighted_entropy_sum is not None
        entropy = weighted_entropy_sum / max(total_weight, 1e-6)
        return joint_log_prob, entropy

    def _resolve_actor_weight(self, head_name: str, base_weight: torch.Tensor | float) -> torch.Tensor | float:
        floor = float(self._policy_credit_floor_by_head.get(head_name, 0.0))
        if floor <= 0.0:
            return base_weight
        if isinstance(base_weight, torch.Tensor):
            return torch.clamp(base_weight, min=floor)
        return max(float(base_weight), floor)

    def _resolve_entropy_weight(self, head_name: str, base_weight: torch.Tensor | float) -> torch.Tensor | float:
        floor = float(self._entropy_credit_floor_by_head.get(head_name, 0.0))
        scale = float(self._entropy_coef_scale_by_head.get(head_name, 1.0))
        if isinstance(base_weight, torch.Tensor):
            effective_weight = torch.clamp(base_weight, min=floor) if floor > 0.0 else base_weight
            return effective_weight * scale
        effective_weight = max(float(base_weight), floor)
        return effective_weight * scale

    def _build_head_credit_weights(self, aggregation_reason: str) -> dict[str, float]:
        if not self._use_hierarchy or not self._head_credit_enabled:
            return {"slow": 1.0, "fast": 1.0, "event": 1.0}
        if self._head_credit_protocol == "aggregation_reason_weighted_controller_ppo_v3":
            if aggregation_reason == "event_head_prepare":
                return {"slow": 0.3, "fast": 0.1, "event": 1.0}
            if aggregation_reason in {"slow_head_prefetch", "slow_head_cache_fill"}:
                return {"slow": 1.0, "fast": 0.2, "event": 0.15}
            if aggregation_reason in {"fast_head_vehicle_fallback", "fast_head_steady_offload"}:
                return {"slow": 0.3, "fast": 1.0, "event": 0.15}
            return {"slow": 0.35, "fast": 1.0, "event": 0.25}
        if aggregation_reason == "event_head_prepare":
            return {"slow": 0.2, "fast": 0.0, "event": 1.0}
        if aggregation_reason in {"slow_head_prefetch", "slow_head_cache_fill"}:
            return {"slow": 1.0, "fast": 0.15, "event": 0.05}
        if aggregation_reason in {"fast_head_vehicle_fallback", "fast_head_steady_offload"}:
            return {"slow": 0.15, "fast": 1.0, "event": 0.0}
        return {"slow": 0.3, "fast": 1.0, "event": 0.2}

    def _mechanism_retention_active_for_update(self) -> bool:
        if self._mechanism_retention_start_update <= 0:
            return False
        next_update_index = int(self._update_count) + 1
        return bool(next_update_index >= self._mechanism_retention_start_update)

    def _effective_mechanism_aux_coef(self) -> float:
        if not self._mechanism_retention_active_for_update():
            return self._mechanism_aux_coef
        return max(self._mechanism_aux_coef, self._mechanism_aux_coef_floor_after_update)

    def _effective_mechanism_window_weight(self) -> float:
        if not self._mechanism_retention_active_for_update():
            return self._mechanism_window_weight
        return max(self._mechanism_window_weight, self._mechanism_window_weight_floor_after_update)

    def _effective_mechanism_entropy_coef(self) -> float:
        if not self._mechanism_retention_active_for_update():
            return self._mechanism_entropy_coef
        return max(self._mechanism_entropy_coef, self._mechanism_entropy_floor_after_update)

    def _build_mechanism_guidance_annotation(
        self,
        semantic_state: dict[str, Any],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        action_info = dict(row.get("action_info", {}))
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_window_score = float(action_info.get("prepare_window_score", timing_features.get("prepare_window_score", 0.0)) or 0.0)
        temporal_urgency = float(action_info.get("temporal_urgency", timing_features.get("temporal_urgency", 0.0)) or 0.0)
        timing_active = bool(prepare_window_score >= self._temporal_prepare_activation_threshold)
        prediction_state_available = bool(action_info.get("prediction_state_available", False))
        raw_handoff_candidate = bool(action_info.get("raw_handoff_candidate", self._semantic_state_has_raw_handoff_candidate(semantic_state)))
        valid_handoff_target = bool(action_info.get("predicted_handoff_target_valid", self._semantic_state_has_valid_predicted_handoff_target(semantic_state)))
        next_rsu_non_null_count = int(action_info.get("next_rsu_non_null_count", 0) or 0)
        gate_pass = bool(action_info.get("gate_pass", False))
        rsus = list(semantic_state.get("rsus", []))
        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
        predicted_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_next_rsu_id), {})
        handoff_target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_handoff_target_rsu_id), {})
        current_cache_ready = bool(required_adapter and required_adapter in current_rsu.get("cached_adapter_ids", []))
        predicted_next_cache_ready = bool(required_adapter and required_adapter in predicted_rsu.get("cached_adapter_ids", []))
        handoff_target_cache_ready = bool(required_adapter and required_adapter in handoff_target_rsu.get("cached_adapter_ids", []))
        mechanism_action_legal = bool(self._use_hierarchy and self._event_head_enabled)
        prefetch_action_legal = bool(
            self._use_hierarchy
            and self._adapter_prefetch_enabled
            and predicted_next_rsu_id
            and predicted_next_rsu_id != current_rsu_id
        )
        prepare_action_legal = bool(mechanism_action_legal and valid_handoff_target)
        target_mismatch = bool(
            predicted_next_rsu_id
            and predicted_handoff_target_rsu_id
            and predicted_next_rsu_id != predicted_handoff_target_rsu_id
        )
        cache_ready = bool(current_cache_ready or predicted_next_cache_ready or handoff_target_cache_ready)
        prediction_usable = bool(
            prediction_state_available
            and (raw_handoff_candidate or next_rsu_non_null_count > 0 or prefetch_action_legal)
        )
        needs_event_guidance = bool(
            self._mechanism_aux_coef > 0.0
            and prepare_action_legal
            and timing_active
            and prediction_usable
            and (not cache_ready or target_mismatch or not gate_pass)
        )
        needs_prefetch_guidance = bool(
            self._mechanism_aux_coef > 0.0
            and prefetch_action_legal
            and timing_active
            and prediction_usable
            and not predicted_next_cache_ready
            and not target_mismatch
        )
        needs_guidance = bool(needs_event_guidance or needs_prefetch_guidance)
        slow_target = 0
        slow_weight = 0.0
        if needs_prefetch_guidance:
            slow_target = 2
            slow_weight = max(self._prepare_action_prior_weight, 0.0)
        elif self._mechanism_aux_current_cache_fill_enabled and required_adapter and not current_cache_ready:
            slow_target = 1
            slow_weight = 0.35 * max(self._prepare_action_prior_weight, 0.0)
        guidance_strength = _clamp01(
            0.45 * prepare_window_score
            + 0.25 * temporal_urgency
            + 0.20 * float(action_info.get("prediction_confidence", 0.0) or 0.0)
            + 0.10 * float(gate_pass)
        )
        guidance_strength = max(guidance_strength, 0.25 if needs_guidance else 0.0)
        transition_weight = self._effective_mechanism_window_weight() if needs_guidance else 1.0
        return {
            "apply": needs_guidance,
            "event_guidance": needs_event_guidance,
            "prefetch_guidance": needs_prefetch_guidance,
            "raw_handoff_candidate": raw_handoff_candidate,
            "valid_handoff_target": valid_handoff_target,
            "timing_active": timing_active,
            "prediction_state_available": prediction_state_available,
            "next_rsu_non_null_count": next_rsu_non_null_count,
            "gate_pass": gate_pass,
            "cache_ready": cache_ready,
            "current_cache_ready": current_cache_ready,
            "predicted_next_cache_ready": predicted_next_cache_ready,
            "handoff_target_cache_ready": handoff_target_cache_ready,
            "mechanism_action_legal": mechanism_action_legal,
            "prefetch_action_legal": prefetch_action_legal,
            "target_mismatch": target_mismatch,
            "event_target": 1,
            "slow_target": slow_target,
            "event_weight": guidance_strength if needs_event_guidance else 0.0,
            "slow_weight": slow_weight,
            "transition_weight": transition_weight,
            "event_prepare_prob_before": float(action_info.get("event_prepare_prob", 0.0) or 0.0),
            "event_entropy_before": float(action_info.get("head_entropies", {}).get("event", action_info.get("entropy", 0.0)) or 0.0)
            if isinstance(action_info.get("head_entropies", {}), dict)
            else float(action_info.get("entropy", 0.0) or 0.0),
            "prepare_window_score": prepare_window_score,
            "temporal_urgency": temporal_urgency,
        }

    def _summarize_mechanism_guidance_annotations(
        self,
        annotations: list[dict[str, Any]],
        rollout: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_count = max(len(annotations), 1)
        guided = [item for item in annotations if bool(item.get("apply", False))]
        guided_count = len(guided)
        mechanism_window_count = sum(
            1
            for row in rollout
            if str(row.get("decision_info", {}).get("run_metadata", {}).get("window_class", "")) == "mechanism_activating"
        )
        weighted_count = sum(1 for item in annotations if float(item.get("transition_weight", 1.0) or 1.0) > 1.0)
        def mean_field(field_name: str) -> float:
            if not guided:
                return 0.0
            return float(fmean(float(item.get(field_name, 0.0) or 0.0) for item in guided))

        return {
            "mechanism_window_count": int(mechanism_window_count),
            "mechanism_guided_action_count": int(guided_count),
            "mechanism_guided_transition_ratio": round(float(guided_count) / float(total_count), 6),
            "weighted_mechanism_transition_ratio": round(float(weighted_count) / float(total_count), 6),
            "mechanism_window_weight": round(self._mechanism_window_weight, 6),
            "effective_mechanism_window_weight": round(self._effective_mechanism_window_weight(), 6),
            "prepare_action_prior_weight": round(self._prepare_action_prior_weight, 6),
            "mechanism_guided_event_prepare_prob_before_update": round(mean_field("event_prepare_prob_before"), 6),
            "mechanism_guided_prepare_window_score_mean": round(mean_field("prepare_window_score"), 6),
            "mechanism_guided_temporal_urgency_mean": round(mean_field("temporal_urgency"), 6),
            "mechanism_guided_gate_pass_rate": round(mean_field("gate_pass"), 6),
            "mechanism_guided_cache_ready_rate": round(mean_field("cache_ready"), 6),
            "mechanism_guided_target_mismatch_rate": round(mean_field("target_mismatch"), 6),
            "mechanism_prepare_action_legal_count": int(
                sum(1 for item in annotations if bool(item.get("mechanism_action_legal", False)))
            ),
            "mechanism_prefetch_action_legal_count": int(
                sum(1 for item in annotations if bool(item.get("prefetch_action_legal", False)))
            ),
            "mechanism_event_guidance_count": int(
                sum(1 for item in annotations if bool(item.get("event_guidance", False)))
            ),
            "mechanism_prefetch_guidance_count": int(
                sum(1 for item in annotations if bool(item.get("prefetch_guidance", False)))
            ),
        }

    def _compute_mechanism_guided_action_prob_summary(
        self,
        *,
        semantic_states: list[dict[str, Any]],
        annotations: list[dict[str, Any]],
    ) -> dict[str, float]:
        guided_states = [
            semantic_state
            for semantic_state, annotation in zip(semantic_states, annotations, strict=False)
            if bool(annotation.get("apply", False))
        ]
        if not guided_states or not self._use_hierarchy:
            return {
                "mechanism_guided_event_prepare_prob_after_update": 0.0,
                "mechanism_guided_prefetch_prob_after_update": 0.0,
            }
        event_probs: list[float] = []
        prefetch_probs: list[float] = []
        with torch.no_grad():
            for semantic_state in guided_states:
                policy_output = self._forward_policy(semantic_state)
                event_prob = torch.softmax(policy_output["event_logits"], dim=-1)
                slow_prob = torch.softmax(policy_output["slow_logits"], dim=-1)
                event_probs.append(float(event_prob[1].item()) if event_prob.numel() > 1 else 0.0)
                prefetch_probs.append(float(slow_prob[2].item()) if slow_prob.numel() > 2 else 0.0)
        return {
            "mechanism_guided_event_prepare_prob_after_update": round(float(fmean(event_probs)), 6),
            "mechanism_guided_prefetch_prob_after_update": round(float(fmean(prefetch_probs)), 6),
        }

    def _compute_mechanism_auxiliary_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_annotations: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_hierarchy or self._mechanism_aux_coef <= 0.0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return zero, zero
        loss_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        for policy_output, annotation in zip(batch_outputs, batch_annotations, strict=False):
            if not bool(annotation.get("apply", False)):
                continue
            event_logits = policy_output["event_logits"]
            event_target = torch.tensor([int(annotation.get("event_target", 1))], dtype=torch.long, device=self._device)
            event_loss = nn.functional.cross_entropy(event_logits.unsqueeze(0), event_target)
            event_distribution = Categorical(logits=event_logits)
            weighted_loss = float(annotation.get("event_weight", 1.0)) * event_loss
            slow_weight = float(annotation.get("slow_weight", 0.0) or 0.0)
            if slow_weight > 1e-8:
                slow_target = torch.tensor([int(annotation.get("slow_target", 0))], dtype=torch.long, device=self._device)
                slow_loss = nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), slow_target)
                weighted_loss = weighted_loss + slow_weight * slow_loss
            loss_terms.append(weighted_loss)
            entropy_terms.append(event_distribution.entropy())
        if not loss_terms:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return zero, zero
        return torch.stack(loss_terms).mean(), torch.stack(entropy_terms).mean()

    def _apply_policy_adjustments(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        adjusted = self._apply_continuity_guard(policy_output, semantic_state)
        return self._apply_event_logit_sharpening(adjusted, semantic_state)

    def _apply_continuity_guard(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not self._use_hierarchy
            or not self._continuity_guard_enabled
            or not self._handoff_target_alignment_guard_enabled
        ):
            return policy_output

        guard_info = self._build_continuity_guard_info(policy_output, semantic_state)
        if not guard_info.get("guard_triggered", False):
            return policy_output

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        confidence = float(guard_info.get("prediction_confidence", 0.0))
        prepare_score = float(guard_info.get("prepare_window_score", 0.0))
        strength = max(confidence, prepare_score, 0.25)
        target_cache_ready = bool(
            guard_info.get("predicted_next_cache_ready", False)
            or guard_info.get("handoff_target_cache_ready", False)
        )
        prefetch_penalty = self._continuity_guard_logit_penalty * strength if target_cache_ready else 0.0
        prepare_boost = self._continuity_guard_prepare_boost * strength if target_cache_ready else 0.0
        slow_logits[2] = slow_logits[2] - prefetch_penalty
        event_logits[1] = event_logits[1] + prepare_boost
        event_logits[0] = event_logits[0] - 0.25 * prepare_boost
        adjusted["slow_logits"] = slow_logits
        adjusted["event_logits"] = event_logits
        guard_info.update(
            {
                "logit_prefetch_penalty": round(float(prefetch_penalty), 6),
                "logit_prepare_boost": round(float(prepare_boost), 6),
                "target_cache_ready_for_prepare": target_cache_ready,
            }
        )
        adjusted["continuity_guard_info"] = guard_info
        return adjusted

    def _apply_cache_warm_start_guard_to_actions(
        self,
        *,
        semantic_state: dict[str, Any],
        selected_actions: dict[str, int],
    ) -> dict[str, Any]:
        if not self._cache_warm_start_guard_enabled or not self._use_hierarchy:
            return {"enabled": False, "guarded": False, "reason": "disabled"}

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        if not required_adapter:
            return {"enabled": True, "guarded": False, "reason": "missing_required_adapter"}
        required_adapter = str(required_adapter)
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        if current_rsu_id is None:
            return {"enabled": True, "guarded": False, "reason": "missing_current_rsu"}

        rsu_map = {
            str(rsu.get("rsu_id")): rsu
            for rsu in semantic_state.get("rsus", [])
            if isinstance(rsu, dict)
        }
        current_rsu = rsu_map.get(str(current_rsu_id), {})
        current_cache_ready = required_adapter in {
            str(adapter_id)
            for adapter_id in current_rsu.get("cached_adapter_ids", [])
        }
        if not current_cache_ready:
            original = dict(selected_actions)
            selected_actions["slow"] = 1
            selected_actions["event"] = 0
            return {
                "enabled": True,
                "guarded": True,
                "reason": "current_adapter_not_warm_cache_first",
                "required_adapter": required_adapter,
                "current_rsu_id": current_rsu_id,
                "original_actions": original,
                "guarded_actions": dict(selected_actions),
            }

        predictions = semantic_state.get("predictions", {})
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predicted_target = None
        if isinstance(predictions, dict) and vehicle_id:
            predicted_target = (
                predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
                or predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
            )
            if predicted_target is None:
                sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
                for rsu_id in sequence if isinstance(sequence, list) else []:
                    if rsu_id is not None and str(rsu_id) != str(current_rsu_id):
                        predicted_target = rsu_id
                        break
        if predicted_target is None or str(predicted_target) == str(current_rsu_id):
            return {"enabled": True, "guarded": False, "reason": "no_distinct_predicted_target"}

        target_rsu = rsu_map.get(str(predicted_target), {})
        target_cache_ready = required_adapter in {
            str(adapter_id)
            for adapter_id in target_rsu.get("cached_adapter_ids", [])
        }
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        countdown_steps = float(timing_features.get("countdown_steps", 0.0) or 0.0)
        max_prefetch_countdown = self._cache_warm_start_guard_max_prefetch_countdown
        event_prepare_selected = int(selected_actions.get("event", 0)) == 1
        if (
            not target_cache_ready
            and event_prepare_selected
            and max_prefetch_countdown > 0.0
            and countdown_steps > max_prefetch_countdown
        ):
            return {
                "enabled": True,
                "guarded": False,
                "reason": "target_prefetch_deferred_until_freshness_window",
                "required_adapter": required_adapter,
                "current_rsu_id": current_rsu_id,
                "predicted_target_rsu_id": predicted_target,
                "current_cache_ready": True,
                "target_cache_ready": False,
                "handoff_countdown_steps": round(countdown_steps, 6),
                "min_countdown": self._cache_warm_start_guard_min_countdown,
                "max_prefetch_countdown": max_prefetch_countdown,
            }
        if (
            not target_cache_ready
            and countdown_steps >= self._cache_warm_start_guard_min_countdown
            and event_prepare_selected
        ):
            original = dict(selected_actions)
            selected_actions["slow"] = 2
            selected_actions["event"] = 0
            return {
                "enabled": True,
                "guarded": True,
                "reason": "target_adapter_not_warm_prefetch_first",
                "required_adapter": required_adapter,
                "current_rsu_id": current_rsu_id,
                "predicted_target_rsu_id": predicted_target,
                "handoff_countdown_steps": round(countdown_steps, 6),
                "min_countdown": self._cache_warm_start_guard_min_countdown,
                "max_prefetch_countdown": max_prefetch_countdown,
                "original_actions": original,
                "guarded_actions": dict(selected_actions),
            }
        return {
            "enabled": True,
            "guarded": False,
            "reason": "cache_warm_enough_or_prepare_imminent",
            "required_adapter": required_adapter,
            "current_cache_ready": True,
            "target_cache_ready": bool(target_cache_ready),
            "handoff_countdown_steps": round(countdown_steps, 6),
            "min_countdown": self._cache_warm_start_guard_min_countdown,
            "max_prefetch_countdown": max_prefetch_countdown,
        }

    def _apply_predictive_prefetch_admission_guard_to_actions(
        self,
        *,
        semantic_state: dict[str, Any],
        selected_actions: dict[str, int],
    ) -> dict[str, Any]:
        if not self._predictive_prefetch_admission_guard_enabled or not self._use_hierarchy:
            return {"enabled": False, "guarded": False, "reason": "disabled"}
        if int(selected_actions.get("event", 0)) == 1:
            return {"enabled": True, "guarded": False, "reason": "event_prepare_selected"}
        if int(selected_actions.get("slow", 0)) != 2:
            return {"enabled": True, "guarded": False, "reason": "not_predictive_prefetch_selected"}

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        if not required_adapter:
            return {"enabled": True, "guarded": False, "reason": "missing_required_adapter"}
        required_adapter = str(required_adapter)
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        if current_rsu_id is None or not vehicle_id:
            return {"enabled": True, "guarded": False, "reason": "missing_vehicle_or_current_rsu"}

        rsu_map = {
            str(rsu.get("rsu_id")): rsu
            for rsu in semantic_state.get("rsus", [])
            if isinstance(rsu, dict)
        }
        current_rsu = rsu_map.get(str(current_rsu_id), {})
        current_cache_ready = required_adapter in {
            str(adapter_id)
            for adapter_id in current_rsu.get("cached_adapter_ids", [])
        }
        if not current_cache_ready:
            return {"enabled": True, "guarded": False, "reason": "current_adapter_not_warm"}

        predictions = semantic_state.get("predictions", {})
        if not isinstance(predictions, dict):
            return {"enabled": True, "guarded": False, "reason": "missing_predictions"}
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = (
            predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
            or predictions.get("predicted_handoff_target_rsu_id_by_vehicle", {}).get(vehicle_id)
        )
        next_rsu_sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
        predicted_prefetch_target_rsu_id = predicted_next_rsu_id
        if predicted_prefetch_target_rsu_id is None and isinstance(next_rsu_sequence, list) and next_rsu_sequence:
            predicted_prefetch_target_rsu_id = next_rsu_sequence[0]
        if (
            predicted_prefetch_target_rsu_id is None
            or str(predicted_prefetch_target_rsu_id) == str(current_rsu_id)
        ):
            for candidate_rsu_id in next_rsu_sequence if isinstance(next_rsu_sequence, list) else []:
                if candidate_rsu_id is not None and str(candidate_rsu_id) != str(current_rsu_id):
                    predicted_prefetch_target_rsu_id = candidate_rsu_id
                    break
        if (
            predicted_prefetch_target_rsu_id is None
            or str(predicted_prefetch_target_rsu_id) == str(current_rsu_id)
        ):
            return {
                "enabled": True,
                "guarded": False,
                "reason": "missing_distinct_prefetch_target",
                "current_rsu_id": current_rsu_id,
                "predicted_next_rsu_id": predicted_next_rsu_id,
            }

        distinct_handoff_target = bool(
            predicted_handoff_target_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
        )
        if not distinct_handoff_target:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "missing_distinct_handoff_target_for_prepare",
                "current_rsu_id": current_rsu_id,
                "predicted_prefetch_target_rsu_id": predicted_prefetch_target_rsu_id,
            }

        target_rsu = rsu_map.get(str(predicted_prefetch_target_rsu_id), {})
        target_cache_ready = required_adapter in {
            str(adapter_id)
            for adapter_id in target_rsu.get("cached_adapter_ids", [])
        }
        if target_cache_ready:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "target_adapter_ready",
                "required_adapter": required_adapter,
                "predicted_prefetch_target_rsu_id": predicted_prefetch_target_rsu_id,
            }

        prediction_confidence = max(
            0.0,
            min(
                float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0) or 0.0),
                1.0,
            ),
        )
        predicted_next_distinct = bool(
            predicted_next_rsu_id is not None and str(predicted_next_rsu_id) != str(current_rsu_id)
        )
        predicted_next_aligned = bool(
            predicted_next_distinct
            and str(predicted_next_rsu_id) == str(predicted_prefetch_target_rsu_id)
        )
        handoff_target_aligned = bool(
            str(predicted_handoff_target_rsu_id) == str(predicted_prefetch_target_rsu_id)
        )
        alignment_ready = bool(
            (not self._predictive_prefetch_admission_require_distinct_next or predicted_next_aligned)
            and handoff_target_aligned
        )
        low_confidence = prediction_confidence < self._predictive_prefetch_admission_min_confidence
        if low_confidence and not alignment_ready:
            original = dict(selected_actions)
            selected_actions["slow"] = 0
            selected_actions["event"] = 1
            return {
                "enabled": True,
                "guarded": True,
                "reason": "low_confidence_unaligned_prefetch_deferred_to_prepare",
                "required_adapter": required_adapter,
                "current_rsu_id": current_rsu_id,
                "predicted_next_rsu_id": predicted_next_rsu_id,
                "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
                "predicted_prefetch_target_rsu_id": predicted_prefetch_target_rsu_id,
                "prediction_confidence": round(prediction_confidence, 6),
                "min_confidence": self._predictive_prefetch_admission_min_confidence,
                "predicted_next_aligned": predicted_next_aligned,
                "handoff_target_aligned": handoff_target_aligned,
                "require_distinct_next": self._predictive_prefetch_admission_require_distinct_next,
                "original_actions": original,
                "guarded_actions": dict(selected_actions),
            }
        return {
            "enabled": True,
            "guarded": False,
            "reason": "prefetch_admitted",
            "required_adapter": required_adapter,
            "current_rsu_id": current_rsu_id,
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "predicted_prefetch_target_rsu_id": predicted_prefetch_target_rsu_id,
            "prediction_confidence": round(prediction_confidence, 6),
            "min_confidence": self._predictive_prefetch_admission_min_confidence,
            "predicted_next_aligned": predicted_next_aligned,
            "handoff_target_aligned": handoff_target_aligned,
            "require_distinct_next": self._predictive_prefetch_admission_require_distinct_next,
        }

    def _apply_backhaul_guard_to_actions(
        self,
        *,
        semantic_state: dict[str, Any],
        selected_actions: dict[str, int],
        cache_warm_guard_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._backhaul_guard_enabled or not self._use_hierarchy:
            return {"enabled": False, "guarded": False, "reason": "disabled"}
        current_time_index = int(semantic_state.get("time_index", 0) or 0)
        if self._backhaul_guard_last_time_index is None or current_time_index <= self._backhaul_guard_last_time_index:
            self._backhaul_guard_seen_reactive_fills = {}
        self._backhaul_guard_last_time_index = current_time_index
        if int(selected_actions.get("event", 0)) == 1:
            return {"enabled": True, "guarded": False, "reason": "event_prepare_selected"}
        if int(selected_actions.get("slow", 0)) != 1:
            return {"enabled": True, "guarded": False, "reason": "not_reactive_cache_fill"}

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = str(current_node.get("required_adapter") or "")
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        current_rsu = next(
            (rsu for rsu in semantic_state.get("rsus", []) if str(rsu.get("rsu_id", "")) == str(current_rsu_id)),
            {},
        )
        current_cache_ready = bool(
            required_adapter
            and required_adapter in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )
        raw_handoff_candidate = self._semantic_state_has_raw_handoff_candidate(semantic_state)
        valid_handoff_target = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        if current_cache_ready:
            selected_actions["slow"] = 0
            return {
                "enabled": True,
                "guarded": True,
                "reason": "current_cache_ready",
                "required_adapter": required_adapter,
                "reactive_fill_count_before": 0,
            }
        if (
            cache_warm_guard_info
            and cache_warm_guard_info.get("guarded", False)
            and cache_warm_guard_info.get("reason") == "current_adapter_not_warm_cache_first"
        ):
            return {
                "enabled": True,
                "guarded": False,
                "reason": "cache_warm_guard_allows_current_fill",
                "required_adapter": required_adapter,
            }
        if raw_handoff_candidate or valid_handoff_target:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "prediction_signal_allows_cache_fill",
                "required_adapter": required_adapter,
            }
        guard_key = required_adapter or "unknown_adapter"
        fill_count = int(self._backhaul_guard_seen_reactive_fills.get(guard_key, 0))
        if fill_count >= self._backhaul_guard_max_reactive_fills_per_adapter:
            selected_actions["slow"] = 0
            return {
                "enabled": True,
                "guarded": True,
                "reason": "reactive_fill_budget_exhausted",
                "required_adapter": required_adapter,
                "reactive_fill_count_before": fill_count,
                "max_reactive_fills_per_adapter": self._backhaul_guard_max_reactive_fills_per_adapter,
            }
        self._backhaul_guard_seen_reactive_fills[guard_key] = fill_count + 1
        return {
            "enabled": True,
            "guarded": False,
            "reason": "reactive_fill_budget_available",
            "required_adapter": required_adapter,
            "reactive_fill_count_before": fill_count,
            "reactive_fill_count_after": fill_count + 1,
            "max_reactive_fills_per_adapter": self._backhaul_guard_max_reactive_fills_per_adapter,
        }

    def _build_continuity_guard_info(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predictions = semantic_state.get("predictions", {})
        current_node = semantic_state.get("current_workflow_node") or {}
        rsus = list(semantic_state.get("rsus", []))
        required_adapter = current_node.get("required_adapter")
        predicted_next_rsu_id = None
        predicted_handoff_target_rsu_id = None
        confidence = 0.0
        uncertainty = 1.0
        if isinstance(predictions, dict) and vehicle_id:
            predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
            next_sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
            if predicted_next_rsu_id is None and next_sequence:
                predicted_next_rsu_id = next_sequence[0]
            predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
            confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0) or 0.0)
            uncertainty = float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0) or 1.0)
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = float(timing_features.get("prepare_window_score", 0.0) or 0.0)
        temporal_urgency = float(timing_features.get("temporal_urgency", 0.0) or 0.0)
        original_action, original_head_actions = self._greedy_env_action_from_logits(policy_output)
        target_present = bool(predicted_handoff_target_rsu_id and predicted_handoff_target_rsu_id != current_rsu_id)
        target_mismatch = bool(
            target_present
            and predicted_next_rsu_id
            and str(predicted_next_rsu_id) != str(predicted_handoff_target_rsu_id)
        )
        predicted_next_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_next_rsu_id), {})
        handoff_target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_handoff_target_rsu_id), {})
        predicted_next_cache_ready = bool(
            required_adapter
            and required_adapter in predicted_next_rsu.get("cached_adapter_ids", [])
        )
        handoff_target_cache_ready = bool(
            required_adapter
            and required_adapter in handoff_target_rsu.get("cached_adapter_ids", [])
        )
        high_confidence = confidence >= self._continuity_guard_confidence_threshold
        handoff_imminent = (
            prepare_score >= self._continuity_guard_prepare_score_threshold
            or temporal_urgency >= self._deterministic_temporal_urgency_floor
        )
        guard_triggered = bool(target_present and (target_mismatch or high_confidence or handoff_imminent))
        reason = "not_triggered"
        if guard_triggered:
            if target_mismatch:
                reason = "predicted_next_target_mismatch_prefers_prepare"
            elif handoff_imminent:
                reason = "handoff_imminent_prefers_prepare"
            else:
                reason = "high_confidence_handoff_target_prefers_prepare"
        return {
            "guard_triggered": guard_triggered,
            "original_action": int(original_action),
            "original_head_actions": original_head_actions,
            "guarded_action": int(original_action),
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "current_rsu_id": current_rsu_id,
            "target_mismatch": target_mismatch,
            "required_adapter": required_adapter,
            "predicted_next_cache_ready": predicted_next_cache_ready,
            "handoff_target_cache_ready": handoff_target_cache_ready,
            "prediction_confidence": round(float(confidence), 6),
            "prediction_uncertainty": round(float(uncertainty), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "reason": reason,
            "hard_override_enabled": self._continuity_guard_hard_override_enabled,
            "hard_override_applied": False,
        }

    def _greedy_env_action_from_logits(self, policy_output: dict[str, Any]) -> tuple[int, dict[str, int]]:
        if not self._use_hierarchy:
            action = int(torch.argmax(policy_output["flat_logits"], dim=-1).item())
            return action, {"flat": action}
        head_actions = {
            "slow": int(torch.argmax(policy_output["slow_logits"], dim=-1).item()),
            "fast": int(torch.argmax(policy_output["fast_logits"], dim=-1).item()),
            "event": int(torch.argmax(policy_output["event_logits"], dim=-1).item()),
        }
        env_action, _ = 聚合层级动作(
            head_actions=head_actions,
            use_hierarchy=self._use_hierarchy,
            event_head_enabled=self._event_head_enabled,
            adapter_prefetch_enabled=self._adapter_prefetch_enabled,
        )
        return int(env_action), head_actions

    def _apply_event_logit_sharpening(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._use_hierarchy or self._event_logit_sharpening_final_scale <= 1.0:
            return policy_output
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        timing_support = max(
            float(timing_features.get("prepare_window_score", 0.0)),
            float(timing_features.get("temporal_urgency", 0.0)),
        )
        reliability_summary = self._build_prediction_reliability_summary(semantic_state)
        scaling_summary = self._build_event_scaling_summary(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        sharpen_factor = float(scaling_summary["event_sharpen_factor"])
        base_sharpen_growth = self._current_event_logit_sharpening_scale() - 1.0
        sharpen_scale = 1.0 + base_sharpen_growth * (
            1.0 + self._event_logit_sharpening_timing_gain * timing_support
        ) * sharpen_factor
        margin_boost = self._compute_event_prepare_margin_boost(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        if sharpen_scale <= 1.0 + 1e-8 and margin_boost <= 1e-8:
            return policy_output
        adjusted = dict(policy_output)
        event_logits = adjusted["event_logits"].clone()
        if sharpen_scale > 1.0 + 1e-8:
            event_center = event_logits.mean()
            event_logits = (event_logits - event_center) * sharpen_scale + event_center
        if margin_boost > 1e-8:
            event_logits[1] = event_logits[1] + margin_boost
            event_logits[0] = event_logits[0] - 0.25 * margin_boost
        adjusted["event_logits"] = event_logits
        adjusted["event_sharpening_info"] = {
            "sharpen_scale": round(float(sharpen_scale), 6),
            "timing_support": round(float(timing_support), 6),
            "sharpen_factor": round(float(sharpen_factor), 6),
            "prediction_reliability": round(float(reliability_summary.get("prediction_reliability", 0.0)), 6),
            "event_aggressive_support": round(float(scaling_summary.get("event_aggressive_support", 0.0)), 6),
            "continuity_pressure_score": round(float(scaling_summary.get("continuity_pressure_score", 0.0)), 6),
            "conditional_conservative_pressure": round(
                float(scaling_summary.get("conditional_conservative_pressure", 0.0)),
                6,
            ),
            "margin_boost": round(float(margin_boost), 6),
        }
        return adjusted

    def _compute_auxiliary_loss(
        self,
        batch_states: list[dict[str, Any]],
        batch_outputs: list[dict[str, Any]],
    ) -> torch.Tensor:
        del batch_states
        del batch_outputs
        return torch.tensor(0.0, dtype=torch.float32, device=self._device)

    def _effective_heuristic_imitation_coef(self) -> float:
        if self._heuristic_imitation_coef <= 0.0:
            return 0.0
        if self._update_count < self._heuristic_imitation_warmup_updates:
            return self._heuristic_imitation_coef
        decay_steps = self._update_count - self._heuristic_imitation_warmup_updates + 1
        return float(self._heuristic_imitation_coef * (self._heuristic_imitation_decay ** decay_steps))

    def _annotate_heuristic_imitation_targets(self, rollout: list[dict[str, Any]]) -> dict[str, float | int]:
        applied_count = 0
        match_count = 0
        if self._heuristic_imitation_coef <= 0.0 or not rollout:
            return {"applied_count": 0, "match_count": 0, "match_rate": 0.0}
        teacher = PopularityCacheHeuristicAgent()
        for row in rollout:
            row["imitation_applied"] = False
            decision_info = dict(row.get("decision_info", {}))
            semantic_state = self._extract_semantic_state(decision_info)
            run_metadata = dict(decision_info.get("run_metadata", {}))
            if not self._should_apply_heuristic_imitation(semantic_state, run_metadata):
                continue
            teacher_action, teacher_info = teacher.act(
                None,
                {
                    "semantic_state": semantic_state,
                    "action_mask": decision_info.get("action_mask"),
                },
            )
            student_action = int(row.get("action", -1))
            row["teacher_action"] = int(teacher_action)
            row["student_action"] = student_action
            row["teacher_reason"] = str(teacher_info.get("heuristic_reason", "unknown"))
            row["imitation_applied"] = True
            row["imitation_head_targets"] = self._head_targets_for_env_action(int(teacher_action))
            applied_count += 1
            if student_action == int(teacher_action):
                match_count += 1
        match_rate = float(match_count) / float(applied_count) if applied_count else 0.0
        return {"applied_count": applied_count, "match_count": match_count, "match_rate": match_rate}

    def _should_apply_heuristic_imitation(
        self,
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> bool:
        if self._heuristic_imitation_coef <= 0.0:
            return False
        current_node = semantic_state.get("current_workflow_node") or {}
        if not current_node:
            return False
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        required_adapter = current_node.get("required_adapter")
        predictions = semantic_state.get("predictions", {})
        predicted_handoff_target = None
        if isinstance(predictions, dict) and vehicle_id:
            predicted_handoff_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = next(
            (rsu for rsu in semantic_state.get("rsus", []) if str(rsu.get("rsu_id", "")) == str(current_rsu_id)),
            {},
        )
        current_adapter_missing = bool(
            required_adapter
            and current_rsu_id
            and str(required_adapter) not in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )
        handoff_signal = bool(predicted_handoff_target and predicted_handoff_target != current_rsu_id)
        mechanism_window = str(run_metadata.get("window_class", "")) == "mechanism_activating"
        return bool((mechanism_window or handoff_signal) and (current_adapter_missing or handoff_signal))

    def _head_targets_for_env_action(self, action: int) -> dict[str, int]:
        if int(action) == 0:
            return {"slow": 1, "fast": 0, "event": 0}
        if int(action) == 1:
            return {"slow": 2, "fast": 0, "event": 0}
        if int(action) == 2:
            return {"slow": 0, "fast": 1, "event": 0}
        if int(action) == 4:
            return {"slow": 0, "fast": 0, "event": 1}
        return {"slow": 0, "fast": 0, "event": 0}

    def _compute_heuristic_imitation_loss(
        self,
        batch_outputs: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        if self._heuristic_imitation_coef <= 0.0:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        loss_terms: list[torch.Tensor] = []
        for policy_output, row in zip(batch_outputs, batch_rows, strict=False):
            if not bool(row.get("imitation_applied", False)):
                continue
            teacher_action = int(row.get("teacher_action", 3))
            if not self._use_hierarchy:
                target = torch.tensor([teacher_action], dtype=torch.long, device=self._device)
                loss_terms.append(nn.functional.cross_entropy(policy_output["flat_logits"].unsqueeze(0), target))
                continue
            head_targets = dict(row.get("imitation_head_targets", self._head_targets_for_env_action(teacher_action)))
            if teacher_action == 4:
                target = torch.tensor([int(head_targets.get("event", 1))], dtype=torch.long, device=self._device)
                loss_terms.append(nn.functional.cross_entropy(policy_output["event_logits"].unsqueeze(0), target))
            elif teacher_action in {0, 1}:
                target = torch.tensor([int(head_targets.get("slow", 0))], dtype=torch.long, device=self._device)
                loss_terms.append(nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), target))
            elif teacher_action == 2:
                target = torch.tensor([int(head_targets.get("fast", 1))], dtype=torch.long, device=self._device)
                loss_terms.append(nn.functional.cross_entropy(policy_output["fast_logits"].unsqueeze(0), target))
            else:
                event_target = torch.tensor([0], dtype=torch.long, device=self._device)
                slow_target = torch.tensor([0], dtype=torch.long, device=self._device)
                fast_target = torch.tensor([0], dtype=torch.long, device=self._device)
                loss_terms.append(
                    0.5 * nn.functional.cross_entropy(policy_output["event_logits"].unsqueeze(0), event_target)
                    + 0.25 * nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), slow_target)
                    + 0.25 * nn.functional.cross_entropy(policy_output["fast_logits"].unsqueeze(0), fast_target)
                )
        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

    def apply_stability_controls(
        self,
        *,
        learning_rate_scale: float = 1.0,
        clip_ratio_scale: float = 1.0,
        entropy_coef_scale: float = 1.0,
        auxiliary_coef_scale: float = 1.0,
        slow_weight_scale: float = 1.0,
        event_weight_scale: float = 1.0,
        mechanism_bias_delta: float = 0.0,
        max_auxiliary_coef: float | None = None,
        max_mechanism_logit_bias_strength: float | None = None,
    ) -> dict[str, float]:
        self._learning_rate = max(self._learning_rate * float(learning_rate_scale), 1e-6)
        for group in self._optimizer.param_groups:
            group["lr"] = self._learning_rate
        self._clip_ratio = max(self._clip_ratio * float(clip_ratio_scale), 0.02)
        self._entropy_coef = max(self._entropy_coef * float(entropy_coef_scale), 0.0)
        self._auxiliary_coef = max(self._auxiliary_coef * float(auxiliary_coef_scale), 0.0)
        if max_auxiliary_coef is not None:
            self._auxiliary_coef = min(self._auxiliary_coef, float(max_auxiliary_coef))
        self._auxiliary_slow_weight = max(self._auxiliary_slow_weight * float(slow_weight_scale), 0.0)
        self._auxiliary_event_weight = max(self._auxiliary_event_weight * float(event_weight_scale), 0.0)
        self._mechanism_logit_bias_strength = max(self._mechanism_logit_bias_strength + float(mechanism_bias_delta), 0.0)
        if max_mechanism_logit_bias_strength is not None:
            self._mechanism_logit_bias_strength = min(
                self._mechanism_logit_bias_strength,
                float(max_mechanism_logit_bias_strength),
            )
        return {
            "learning_rate": round(self._learning_rate, 10),
            "clip_ratio": round(self._clip_ratio, 6),
            "entropy_coef": round(self._entropy_coef, 6),
            "auxiliary_coef": round(self._auxiliary_coef, 6),
            "auxiliary_slow_weight": round(self._auxiliary_slow_weight, 6),
            "auxiliary_event_weight": round(self._auxiliary_event_weight, 6),
            "mechanism_logit_bias_strength": round(self._mechanism_logit_bias_strength, 6),
        }

    def _head_action_labels(self, selected_actions: dict[str, int]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for head_name, action_id in selected_actions.items():
            if head_name == "flat":
                labels[head_name] = f"env_action_{action_id}"
            else:
                labels[head_name] = 控制头动作语义[head_name].get(action_id, "unknown")
        return labels

    def _summarize_head_action_usage(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._use_hierarchy:
            counts = {str(action_id): 0 for action_id in range(5)}
            for row in rollout:
                counts[str(int(row["action"]))] += 1
            return {"flat": counts}
        summary: dict[str, dict[str, int]] = {}
        for head_name, action_map in 控制头动作语义.items():
            summary[head_name] = {action_name: 0 for action_name in action_map.values()}
        for row in rollout:
            head_actions = row.get("action_info", {}).get("head_actions", {})
            for head_name, action_id in head_actions.items():
                action_name = 控制头动作语义.get(head_name, {}).get(int(action_id), "unknown")
                summary.setdefault(head_name, {})
                summary[head_name][action_name] = summary[head_name].get(action_name, 0) + 1
        return summary

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "encoder_kind": self._encoder_kind,
            "centralized_critic": self._centralized_critic,
            "hierarchical_conditioning": self._hierarchical_conditioning,
            "use_hierarchy": self._use_hierarchy,
            "use_prediction_features": self._use_prediction_features,
            "use_uncertainty_signal": self._use_uncertainty_signal,
            "use_dependency_aware": self._use_dependency_aware,
            "graph_continuity_critic_enabled": self._graph_continuity_critic_enabled,
            "uncertainty_aware_event_scaling_enabled": self._uncertainty_aware_event_scaling_enabled,
            "uncertainty_aware_critic_enabled": self._uncertainty_aware_critic_enabled,
            "event_head_enabled": self._event_head_enabled,
            "adapter_prefetch_enabled": self._adapter_prefetch_enabled,
            "head_credit_enabled": self._head_credit_enabled,
            "head_credit_protocol": self._head_credit_protocol,
            "mechanism_logit_bias_strength": self._mechanism_logit_bias_strength,
            "mechanism_confidence_floor": self._mechanism_confidence_floor,
            "prediction_feature_dim": self._prediction_feature_dim,
            "prediction_gate_min_leak": self._prediction_gate_min_leak,
            "slow_policy_credit_floor": self._slow_policy_credit_floor,
            "fast_policy_credit_floor": self._fast_policy_credit_floor,
            "event_policy_credit_floor": self._event_policy_credit_floor,
            "event_advantage_blend": self._event_advantage_blend,
            "slow_entropy_coef_scale": self._slow_entropy_coef_scale,
            "fast_entropy_coef_scale": self._fast_entropy_coef_scale,
            "event_entropy_coef_scale": self._event_entropy_coef_scale,
            "slow_entropy_credit_floor": self._slow_entropy_credit_floor,
            "fast_entropy_credit_floor": self._fast_entropy_credit_floor,
            "event_entropy_credit_floor": self._event_entropy_credit_floor,
            "event_logit_temperature": self._event_logit_temperature,
            "event_logit_temperature_final": self._event_logit_temperature_final,
            "event_temperature_decay_updates": self._event_temperature_decay_updates,
            "event_logit_sharpening_final_scale": self._event_logit_sharpening_final_scale,
            "event_logit_sharpening_timing_gain": self._event_logit_sharpening_timing_gain,
            "event_actor_loss_extra_gain": self._event_actor_loss_extra_gain,
            "event_prepare_margin_boost": self._event_prepare_margin_boost,
            "temporal_consistency_coef": self._temporal_consistency_coef,
            "temporal_prepare_lead_steps": self._temporal_prepare_lead_steps,
            "temporal_prepare_sigma": self._temporal_prepare_sigma,
            "temporal_prepare_activation_threshold": self._temporal_prepare_activation_threshold,
            "deterministic_temporal_smoothing_enabled": self._deterministic_temporal_smoothing_enabled,
            "deterministic_temporal_smoothing_steps": self._deterministic_temporal_smoothing_steps,
            "deterministic_event_borderline_prob": self._deterministic_event_borderline_prob,
            "deterministic_event_borderline_margin": self._deterministic_event_borderline_margin,
            "deterministic_temporal_urgency_floor": self._deterministic_temporal_urgency_floor,
            "deterministic_high_prepare_override_enabled": self._deterministic_high_prepare_override_enabled,
            "deterministic_high_prepare_threshold": self._deterministic_high_prepare_threshold,
            "deterministic_high_urgency_threshold": self._deterministic_high_urgency_threshold,
            "deterministic_high_prepare_relaxed_margin": self._deterministic_high_prepare_relaxed_margin,
            "predictive_prepare_hard_override_enabled": self._predictive_prepare_hard_override_enabled,
            "predictive_prepare_hard_override_score_threshold": self._predictive_prepare_hard_override_score_threshold,
            "predictive_prepare_hard_override_confidence_threshold": self._predictive_prepare_hard_override_confidence_threshold,
            "continuity_guard_enabled": self._continuity_guard_enabled,
            "handoff_target_alignment_guard_enabled": self._handoff_target_alignment_guard_enabled,
            "continuity_guard_logit_penalty": self._continuity_guard_logit_penalty,
            "continuity_guard_prepare_boost": self._continuity_guard_prepare_boost,
            "continuity_guard_confidence_threshold": self._continuity_guard_confidence_threshold,
            "continuity_guard_prepare_score_threshold": self._continuity_guard_prepare_score_threshold,
            "continuity_guard_hard_override_enabled": self._continuity_guard_hard_override_enabled,
            "heuristic_imitation_coef": self._heuristic_imitation_coef,
            "heuristic_imitation_warmup_updates": self._heuristic_imitation_warmup_updates,
            "heuristic_imitation_decay": self._heuristic_imitation_decay,
            "mechanism_aux_coef": self._mechanism_aux_coef,
            "mechanism_window_weight": self._mechanism_window_weight,
            "prepare_action_prior_weight": self._prepare_action_prior_weight,
            "mechanism_entropy_coef": self._mechanism_entropy_coef,
            "mechanism_retention_start_update": self._mechanism_retention_start_update,
            "mechanism_aux_coef_floor_after_update": self._mechanism_aux_coef_floor_after_update,
            "mechanism_window_weight_floor_after_update": self._mechanism_window_weight_floor_after_update,
            "mechanism_entropy_floor_after_update": self._mechanism_entropy_floor_after_update,
            "mechanism_aux_current_cache_fill_enabled": self._mechanism_aux_current_cache_fill_enabled,
            "latency_fallback_bias_enabled": self._latency_fallback_bias_enabled,
            "latency_fallback_bias_strength": self._latency_fallback_bias_strength,
            "latency_fallback_confidence_floor": self._latency_fallback_confidence_floor,
            "latency_fallback_slow_suppression_strength": self._latency_fallback_slow_suppression_strength,
            "backhaul_guard_enabled": self._backhaul_guard_enabled,
            "backhaul_guard_max_reactive_fills_per_adapter": self._backhaul_guard_max_reactive_fills_per_adapter,
            "cache_warm_start_guard_enabled": self._cache_warm_start_guard_enabled,
            "cache_warm_start_guard_min_countdown": self._cache_warm_start_guard_min_countdown,
            "cache_warm_start_guard_max_prefetch_countdown": self._cache_warm_start_guard_max_prefetch_countdown,
            "predictive_prefetch_admission_guard_enabled": self._predictive_prefetch_admission_guard_enabled,
            "predictive_prefetch_admission_min_confidence": self._predictive_prefetch_admission_min_confidence,
            "predictive_prefetch_admission_require_distinct_next": self._predictive_prefetch_admission_require_distinct_next,
            "auxiliary_slow_weight": self._auxiliary_slow_weight,
            "auxiliary_fast_weight": self._auxiliary_fast_weight,
            "auxiliary_event_weight": self._auxiliary_event_weight,
            "learning_rate": self._learning_rate,
            "clip_ratio": self._clip_ratio,
            "entropy_coef": self._entropy_coef,
            "value_coef": self._value_coef,
            "auxiliary_coef": self._auxiliary_coef,
            "train_epochs": self._train_epochs,
            "target_kl": self._target_kl,
            "kl_early_stop_enabled": self._kl_early_stop_enabled,
            "batch_size": self._batch_size,
            "max_grad_norm": self._max_grad_norm,
            "hidden_dim": self._hidden_dim,
            "hidden_dims": list(self._hidden_dims),
            "deterministic_action": self._deterministic_action,
        }

class SAGHMAPPOBaseAgent(分层PPO基类):
    """? surrogate 控制头动作语义控制头动作语义??"""

    def _compute_auxiliary_loss(
        self,
        batch_states: list[dict[str, Any]],
        batch_outputs: list[dict[str, Any]],
    ) -> torch.Tensor:
        if not self._use_hierarchy:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        loss_terms: list[torch.Tensor] = []
        for semantic_state, policy_output in zip(batch_states, batch_outputs, strict=False):
            pseudo_targets = self._build_mechanism_targets(semantic_state)
            confidence = float(pseudo_targets["confidence_weight"])
            if confidence <= 1e-6:
                continue
            slow_target = torch.tensor([pseudo_targets["slow_target"]], dtype=torch.long, device=self._device)
            fast_target = torch.tensor([pseudo_targets["fast_target"]], dtype=torch.long, device=self._device)
            event_target = torch.tensor([pseudo_targets["event_target"]], dtype=torch.long, device=self._device)
            slow_loss = nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), slow_target)
            fast_loss = nn.functional.cross_entropy(policy_output["fast_logits"].unsqueeze(0), fast_target)
            event_loss = nn.functional.cross_entropy(policy_output["event_logits"].unsqueeze(0), event_target)
            temporal_consistency_loss = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            if self._temporal_consistency_coef > 0.0:
                prepare_margin = (policy_output["event_logits"][1] - policy_output["event_logits"][0]).unsqueeze(0)
                soft_event_target = torch.tensor(
                    [float(pseudo_targets.get("event_soft_target", 0.0))],
                    dtype=torch.float32,
                    device=self._device,
                )
                temporal_consistency_loss = nn.functional.binary_cross_entropy_with_logits(
                    prepare_margin,
                    soft_event_target,
                )
            weighted_loss = (
                self._auxiliary_slow_weight * slow_loss
                + self._auxiliary_fast_weight * fast_loss
                + self._auxiliary_event_weight * event_loss
                + self._temporal_consistency_coef * temporal_consistency_loss
            )
            loss_terms.append(weighted_loss * confidence)
        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

    def _apply_policy_adjustments(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        adjusted = dict(policy_output)
        if self._use_hierarchy and self._mechanism_logit_bias_strength > 0.0:
            pseudo_targets = self._build_mechanism_targets(semantic_state)
            confidence = max(float(pseudo_targets["confidence_weight"]), self._mechanism_confidence_floor)
            if confidence > 1e-6:
                slow_logits = adjusted["slow_logits"].clone()
                fast_logits = adjusted["fast_logits"].clone()
                event_logits = adjusted["event_logits"].clone()
                bias_scale = self._mechanism_logit_bias_strength * confidence
                slow_target = int(pseudo_targets["slow_target"])
                fast_target = int(pseudo_targets.get("fast_target", 0))
                event_target = int(pseudo_targets["event_target"])
                event_soft_target = float(pseudo_targets.get("event_soft_target", 0.0))
                if slow_target in {1, 2}:
                    slow_logits[slow_target] = slow_logits[slow_target] + bias_scale
                if (
                    self._latency_fallback_bias_enabled
                    and fast_target == 1
                    and self._latency_fallback_bias_strength > 0.0
                ):
                    fast_logits[1] = fast_logits[1] + self._latency_fallback_bias_strength * confidence
                latency_fallback_candidate = bool(
                    float(pseudo_targets.get("latency_fallback_candidate", 0.0) or 0.0) > 0.0
                )
                if (
                    self._latency_fallback_bias_enabled
                    and latency_fallback_candidate
                    and self._latency_fallback_slow_suppression_strength > 0.0
                ):
                    suppression = self._latency_fallback_slow_suppression_strength * confidence
                    slow_logits[1] = slow_logits[1] - suppression
                    slow_logits[2] = slow_logits[2] - suppression
                    event_logits[1] = event_logits[1] - suppression
                if event_soft_target > 1e-6:
                    event_logits[1] = event_logits[1] + 1.25 * bias_scale * event_soft_target
                adjusted["slow_logits"] = slow_logits
                adjusted["fast_logits"] = fast_logits
                adjusted["event_logits"] = event_logits
                adjusted["mechanism_bias_info"] = {
                    "bias_scale": round(float(bias_scale), 6),
                    "slow_target": slow_target,
                    "fast_target": fast_target,
                    "event_target": event_target,
                    "event_soft_target": round(event_soft_target, 6),
                    "confidence": round(float(confidence), 6),
                    "latency_fallback_candidate": latency_fallback_candidate,
                }
        adjusted = self._apply_continuity_guard(adjusted, semantic_state)
        return self._apply_event_logit_sharpening(adjusted, semantic_state)

    def _build_mechanism_targets(self, semantic_state: dict[str, Any]) -> dict[str, float | int]:
        rsus = semantic_state.get("rsus", [])
        current_node = semantic_state.get("current_workflow_node") or {}
        predictions = semantic_state.get("predictions", {})
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        required_adapter = current_node.get("required_adapter")
        confidence_weight = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )

        current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
        predicted_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_next_rsu_id), {})
        handoff_target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_handoff_target_rsu_id), {})

        slow_target = 0
        if self._adapter_prefetch_enabled and predicted_next_rsu_id and predicted_next_rsu_id != current_rsu_id:
            if required_adapter not in predicted_rsu.get("cached_adapter_ids", []):
                slow_target = 2
        elif (
            self._mechanism_aux_current_cache_fill_enabled
            and required_adapter
            and required_adapter not in current_rsu.get("cached_adapter_ids", [])
        ):
            slow_target = 1

        current_adapter_ready = bool(required_adapter and required_adapter in current_rsu.get("cached_adapter_ids", []))
        predicted_next_differs = bool(predicted_next_rsu_id and predicted_next_rsu_id != current_rsu_id)
        predicted_target_differs = bool(
            predicted_handoff_target_rsu_id
            and predicted_handoff_target_rsu_id != current_rsu_id
        )
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
        sequence_contains_other_rsu = any(
            rsu_id is not None and rsu_id != current_rsu_id
            for rsu_id in next_sequence
        )
        fast_target = 0 if current_rsu_id is not None else 1
        latency_fallback_candidate = bool(
            self._latency_fallback_bias_enabled
            and current_rsu_id is not None
            and current_adapter_ready
            and not predicted_next_differs
            and not predicted_target_differs
            and not sequence_contains_other_rsu
        )
        if latency_fallback_candidate:
            fast_target = 1
        event_target = 0
        event_soft_target = 0.0
        if self._event_head_enabled and predicted_handoff_target_rsu_id and predicted_handoff_target_rsu_id != current_rsu_id:
            if required_adapter in handoff_target_rsu.get("cached_adapter_ids", []):
                event_soft_target = float(timing_features["prepare_window_score"])
                if event_soft_target >= self._temporal_prepare_activation_threshold:
                    event_target = 1
        if slow_target in {1, 2} or event_soft_target > 1e-6:
            confidence_floor = self._mechanism_confidence_floor * (
                0.5
                + 0.5 * max(float(timing_features["temporal_urgency"]), event_soft_target)
            )
            confidence_weight = max(confidence_weight, confidence_floor)
        if latency_fallback_candidate:
            confidence_weight = max(confidence_weight, self._latency_fallback_confidence_floor)

        return {
            "slow_target": slow_target,
            "fast_target": fast_target,
            "event_target": event_target,
            "confidence_weight": confidence_weight,
            "event_soft_target": event_soft_target,
            "latency_fallback_candidate": float(latency_fallback_candidate),
            "temporal_urgency": float(timing_features["temporal_urgency"]),
            "prepare_window_score": float(timing_features["prepare_window_score"]),
            "handoff_countdown_steps": float(timing_features["countdown_steps"]),
        }


class JSONCheckpointMixin:
    """用于启发式或非 torch 智能体的最小 JSON checkpoint。"""

    def _save_json_checkpoint(self, path: str, payload: dict[str, Any]) -> None:
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
