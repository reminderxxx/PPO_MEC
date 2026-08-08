"""多时间尺度 MARL 智能体公共组件。"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
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
from src.models.uncertainty_transition_ensemble import (
    UncertaintyTransitionEnsemble,
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

OPTION_GATE_LABELS = {
    0: "accept_mappo",
    1: "popularity_safe",
    2: "no_rsu_local",
    3: "mechanism_prepare",
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


def _rsu_by_id_from_semantic_state(semantic_state: dict[str, Any], rsu_id: Any) -> dict[str, Any]:
    for rsu in semantic_state.get("rsus", []) or []:
        if str(rsu.get("rsu_id")) == str(rsu_id):
            return dict(rsu)
    return {}


def _build_digital_twin_handoff_feature_tensor(semantic_state: dict[str, Any]) -> torch.Tensor:
    primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    predictions = semantic_state.get("predictions", {}) or {}
    next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
    predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
    if predicted_next_rsu_id is None and next_sequence:
        predicted_next_rsu_id = next_sequence[0]
    predicted_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    confidence = _clamp01(float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0) or 0.0))
    uncertainty = _clamp01(float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0) or 1.0))
    dwell_time = float(predictions.get("dwell_time", {}).get(vehicle_id, 0.0) or 0.0)

    horizon = max(len(next_sequence), 1)
    non_current_eta = horizon + 1
    non_current_count = 0
    unique_future_rsus: set[str] = set()
    switch_count = 0
    previous_rsu_id = current_rsu_id
    for index, rsu_id in enumerate(next_sequence, start=1):
        if rsu_id is None:
            continue
        unique_future_rsus.add(str(rsu_id))
        if str(rsu_id) != str(current_rsu_id):
            non_current_count += 1
            non_current_eta = min(non_current_eta, index)
        if previous_rsu_id is not None and str(rsu_id) != str(previous_rsu_id):
            switch_count += 1
        previous_rsu_id = rsu_id

    current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
    coverage_radius = max(float(current_rsu.get("coverage_radius", 0.0) or 0.0), 1.0)
    dx = float(primary_vehicle.get("position_x", 0.0) or 0.0) - float(
        current_rsu.get("position_x", primary_vehicle.get("position_x", 0.0)) or 0.0
    )
    dy = float(primary_vehicle.get("position_y", 0.0) or 0.0) - float(
        current_rsu.get("position_y", primary_vehicle.get("position_y", 0.0)) or 0.0
    )
    distance = math.sqrt(dx * dx + dy * dy) if current_rsu else 0.0
    boundary_urgency = 1.0 - _clamp01(max(coverage_radius - distance, 0.0) / coverage_radius) if current_rsu else 0.0

    current_node = semantic_state.get("current_workflow_node") or {}
    service_steps_remaining = float(semantic_state.get("current_node_service_steps_remaining", 0.0) or 0.0)
    service_pressure = _clamp01(service_steps_remaining / float(max(non_current_eta, 1)))
    future_load = predictions.get("future_load", {}) if isinstance(predictions, dict) else {}
    current_load = float(future_load.get(current_rsu_id, 0.0) or 0.0)
    target_load = float(future_load.get(predicted_target_rsu_id, 0.0) or 0.0)
    predicted_load = float(future_load.get(predicted_next_rsu_id, 0.0) or 0.0)
    load_pressure = _clamp01(max(target_load, predicted_load, current_load) / 10.0)
    load_gap_pressure = _clamp01(max(target_load - current_load, predicted_load - current_load, 0.0) / 10.0)

    has_prediction = bool(predicted_next_rsu_id or predicted_target_rsu_id or next_sequence)
    target_differs = bool(predicted_target_rsu_id and str(predicted_target_rsu_id) != str(current_rsu_id))
    next_differs = bool(predicted_next_rsu_id and str(predicted_next_rsu_id) != str(current_rsu_id))
    features = [
        1.0 if has_prediction else 0.0,
        1.0 if next_differs else 0.0,
        1.0 if target_differs else 0.0,
        confidence,
        uncertainty,
        _clamp01(dwell_time / 20.0),
        _clamp01(float(horizon) / 10.0),
        _clamp01(float(switch_count) / float(horizon)),
        _clamp01(float(len(unique_future_rsus)) / max(float(len(semantic_state.get("rsus", []) or [])), 1.0)),
        _clamp01(float(non_current_count) / float(horizon)),
        _clamp01(float(non_current_eta) / float(horizon + 1)),
        boundary_urgency,
        service_pressure,
        load_gap_pressure if current_node else load_pressure,
    ]
    return torch.tensor([_clamp01(item) for item in features], dtype=torch.float32)


def _build_outcome_memory_feature_tensor(
    semantic_state: dict[str, Any],
) -> torch.Tensor:
    memory = semantic_state.get("algorithm_memory", {})
    if not isinstance(memory, dict):
        memory = {}
    raw_last_action_id = memory.get("last_action_id", -1)
    last_action_id = int(
        -1 if raw_last_action_id is None else raw_last_action_id
    )
    last_action_one_hot = [
        1.0 if last_action_id == action_id else 0.0
        for action_id in range(5)
    ]
    features = [
        1.0 if int(memory.get("step_index", 0) or 0) > 0 else 0.0,
        *last_action_one_hot,
        _clamp01(float(memory.get("same_action_streak", 0) or 0) / 8.0),
        _clamp01(float(memory.get("prepare_action_streak", 0) or 0) / 8.0),
        _clamp01(float(memory.get("failed_prepare_streak", 0) or 0) / 8.0),
        _clamp01(float(memory.get("no_progress_streak", 0) or 0) / 8.0),
        math.tanh(float(memory.get("last_reward", 0.0) or 0.0) / 4.0),
        1.0 if bool(memory.get("last_handoff_failed", False)) else 0.0,
        1.0 if bool(memory.get("last_stall", False)) else 0.0,
        1.0 if bool(memory.get("last_mechanism_success", False)) else 0.0,
        1.0 if bool(memory.get("last_prefetch_expired", False)) else 0.0,
        1.0 if bool(memory.get("last_cache_hit", False)) else 0.0,
    ]
    return torch.tensor(features, dtype=torch.float32)


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
        option_gate_enabled: bool = False,
        option_gate_count: int = 4,
        option_counterfactual_critic_enabled: bool = False,
        env_action_model_critic_enabled: bool = False,
        digital_twin_handoff_fusion_enabled: bool = False,
        digital_twin_handoff_slow_scale: float = 0.35,
        digital_twin_handoff_fast_scale: float = 0.45,
        digital_twin_handoff_event_scale: float = 0.85,
        digital_twin_handoff_critic_scale: float = 0.70,
        digital_twin_planning_residual_enabled: bool = False,
        digital_twin_planning_residual_scale: float = 1.0,
        outcome_memory_fusion_enabled: bool = False,
        outcome_memory_actor_scale: float = 0.80,
        outcome_memory_critic_scale: float = 0.70,
        outcome_recovery_residual_enabled: bool = False,
        outcome_recovery_residual_scale: float = 1.0,
        outcome_context_residual_enabled: bool = False,
        hidden_dims: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_hierarchy = bool(use_hierarchy)
        self.hierarchical_conditioning = bool(hierarchical_conditioning)
        self.centralized_critic = bool(centralized_critic)
        self.encoder_kind = str(encoder_kind)
        self.event_logit_temperature = max(float(event_logit_temperature), 0.25)
        self.option_gate_enabled = bool(option_gate_enabled)
        self.option_gate_count = max(int(option_gate_count), 1)
        self.option_counterfactual_critic_enabled = bool(
            option_counterfactual_critic_enabled
        )
        self.env_action_model_critic_enabled = bool(
            env_action_model_critic_enabled
        )
        self.digital_twin_handoff_fusion_enabled = bool(digital_twin_handoff_fusion_enabled)
        self.digital_twin_handoff_slow_scale = max(float(digital_twin_handoff_slow_scale), 0.0)
        self.digital_twin_handoff_fast_scale = max(float(digital_twin_handoff_fast_scale), 0.0)
        self.digital_twin_handoff_event_scale = max(float(digital_twin_handoff_event_scale), 0.0)
        self.digital_twin_handoff_critic_scale = max(float(digital_twin_handoff_critic_scale), 0.0)
        self.digital_twin_planning_residual_enabled = bool(
            digital_twin_planning_residual_enabled
        )
        self.digital_twin_planning_residual_scale = max(
            float(digital_twin_planning_residual_scale),
            0.0,
        )
        self.outcome_memory_fusion_enabled = bool(outcome_memory_fusion_enabled)
        self.outcome_memory_actor_scale = max(float(outcome_memory_actor_scale), 0.0)
        self.outcome_memory_critic_scale = max(float(outcome_memory_critic_scale), 0.0)
        self.outcome_recovery_residual_enabled = bool(
            outcome_recovery_residual_enabled
        )
        self.outcome_recovery_residual_scale = max(
            float(outcome_recovery_residual_scale),
            0.0,
        )
        self.outcome_context_residual_enabled = bool(
            outcome_context_residual_enabled
        )

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
        if self.option_gate_enabled:
            self.option_actor = _多层感知机(self.hidden_dim, self.option_gate_count, hidden_dims=hidden_dims)
            if self.option_counterfactual_critic_enabled:
                self.option_critic = _多层感知机(
                    self.hidden_dim,
                    self.option_gate_count,
                    hidden_dims=hidden_dims,
                )
        if self.env_action_model_critic_enabled:
            self.env_action_critic = _多层感知机(
                self.hidden_dim,
                5,
                hidden_dims=hidden_dims,
            )
        if self.digital_twin_handoff_fusion_enabled:
            self.dt_handoff_projection = nn.Sequential(
                nn.Linear(14, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.dt_slow_norm = nn.LayerNorm(self.hidden_dim)
            self.dt_fast_norm = nn.LayerNorm(self.hidden_dim)
            self.dt_event_norm = nn.LayerNorm(self.hidden_dim)
            self.dt_critic_norm = nn.LayerNorm(self.hidden_dim)
        if self.digital_twin_planning_residual_enabled:
            planning_hidden_dim = max(int(hidden_dims[0]), 16)
            self.digital_twin_planning_adapter = nn.Sequential(
                nn.Linear(self.hidden_dim, planning_hidden_dim),
                nn.Tanh(),
                nn.Linear(planning_hidden_dim, 5),
            )
            nn.init.zeros_(self.digital_twin_planning_adapter[-1].weight)
            nn.init.zeros_(self.digital_twin_planning_adapter[-1].bias)
        if self.outcome_memory_fusion_enabled:
            self.outcome_memory_projection = nn.Sequential(
                nn.Linear(16, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.outcome_memory_shared_norm = nn.LayerNorm(self.hidden_dim)
            self.outcome_memory_slow_norm = nn.LayerNorm(self.hidden_dim)
            self.outcome_memory_fast_norm = nn.LayerNorm(self.hidden_dim)
            self.outcome_memory_event_norm = nn.LayerNorm(self.hidden_dim)
            self.outcome_memory_critic_norm = nn.LayerNorm(self.hidden_dim)
        if self.outcome_recovery_residual_enabled:
            residual_hidden_dim = max(int(hidden_dims[0]), 16)
            self.outcome_recovery_adapter = nn.Sequential(
                nn.Linear(self.hidden_dim, residual_hidden_dim),
                nn.Tanh(),
                nn.Linear(residual_hidden_dim, 5),
            )
            nn.init.zeros_(self.outcome_recovery_adapter[-1].weight)
            nn.init.zeros_(self.outcome_recovery_adapter[-1].bias)
        if self.outcome_context_residual_enabled:
            context_hidden_dim = max(int(hidden_dims[0]), 16)
            self.outcome_context_residual_adapter = nn.Sequential(
                nn.Linear(self.hidden_dim + 16 + 14, context_hidden_dim),
                nn.Tanh(),
                nn.Linear(context_hidden_dim, 5),
            )
            nn.init.zeros_(self.outcome_context_residual_adapter[-1].weight)
            nn.init.zeros_(self.outcome_context_residual_adapter[-1].bias)

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
        dt_features: torch.Tensor | None = None
        if (
            self.digital_twin_handoff_fusion_enabled
            or self.digital_twin_planning_residual_enabled
            or self.outcome_context_residual_enabled
        ):
            dt_features = _build_digital_twin_handoff_feature_tensor(semantic_state).to(
                device=encoded["shared_embedding"].device,
                dtype=encoded["shared_embedding"].dtype,
            )
        if self.digital_twin_handoff_fusion_enabled:
            assert dt_features is not None
            dt_embedding = self.dt_handoff_projection(dt_features.unsqueeze(0)).squeeze(0)
            encoded = dict(encoded)
            encoded["slow_context"] = self.dt_slow_norm(
                encoded["slow_context"] + self.digital_twin_handoff_slow_scale * dt_embedding
            )
            encoded["fast_context"] = self.dt_fast_norm(
                encoded["fast_context"] + self.digital_twin_handoff_fast_scale * dt_embedding
            )
            encoded["event_context"] = self.dt_event_norm(
                encoded["event_context"] + self.digital_twin_handoff_event_scale * dt_embedding
            )
            for critic_key in ["critic_context", "centralized_critic_context"]:
                if critic_key in encoded:
                    encoded[critic_key] = self.dt_critic_norm(
                        encoded[critic_key] + self.digital_twin_handoff_critic_scale * dt_embedding
                    )
            encoded["digital_twin_handoff_fusion_enabled"] = torch.tensor([1.0], dtype=dt_features.dtype, device=dt_features.device)
            encoded["digital_twin_handoff_target_differs"] = dt_features[2:3]
            encoded["digital_twin_handoff_boundary_urgency"] = dt_features[11:12]
            encoded["digital_twin_handoff_service_pressure"] = dt_features[12:13]
        memory_features: torch.Tensor | None = None
        if (
            self.outcome_memory_fusion_enabled
            or self.outcome_recovery_residual_enabled
            or self.outcome_context_residual_enabled
        ):
            memory_features = _build_outcome_memory_feature_tensor(
                semantic_state
            ).to(
                device=encoded["shared_embedding"].device,
                dtype=encoded["shared_embedding"].dtype,
            )
        if self.outcome_memory_fusion_enabled:
            assert memory_features is not None
            memory_embedding = self.outcome_memory_projection(
                memory_features.unsqueeze(0)
            ).squeeze(0)
            encoded = dict(encoded)
            encoded["shared_embedding"] = self.outcome_memory_shared_norm(
                encoded["shared_embedding"]
                + self.outcome_memory_actor_scale * memory_embedding
            )
            encoded["slow_context"] = self.outcome_memory_slow_norm(
                encoded["slow_context"]
                + self.outcome_memory_actor_scale * memory_embedding
            )
            encoded["fast_context"] = self.outcome_memory_fast_norm(
                encoded["fast_context"]
                + self.outcome_memory_actor_scale * memory_embedding
            )
            encoded["event_context"] = self.outcome_memory_event_norm(
                encoded["event_context"]
                + self.outcome_memory_actor_scale * memory_embedding
            )
            for critic_key in ("critic_context", "centralized_critic_context"):
                if critic_key in encoded:
                    encoded[critic_key] = self.outcome_memory_critic_norm(
                        encoded[critic_key]
                        + self.outcome_memory_critic_scale * memory_embedding
                    )
            encoded["outcome_memory_fusion_enabled"] = torch.tensor(
                [1.0],
                dtype=memory_features.dtype,
                device=memory_features.device,
            )
            encoded["outcome_memory_same_action_streak"] = memory_features[6:7]
            encoded["outcome_memory_failed_prepare_streak"] = memory_features[8:9]
            encoded["outcome_memory_no_progress_streak"] = memory_features[9:10]
        recovery_residual_bias: torch.Tensor | None = None
        recovery_gate = torch.tensor(
            0.0,
            dtype=encoded["shared_embedding"].dtype,
            device=encoded["shared_embedding"].device,
        )
        if (
            self.outcome_recovery_residual_enabled
            or self.outcome_context_residual_enabled
        ):
            assert memory_features is not None
            recovery_gate = torch.stack(
                [
                    memory_features[8],
                    memory_features[9],
                    memory_features[11],
                    memory_features[12],
                ]
            ).max()
        if self.outcome_recovery_residual_enabled:
            recovery_residual_bias = (
                self.outcome_recovery_residual_scale
                * recovery_gate
                * self.outcome_recovery_adapter(
                    encoded["shared_embedding"].unsqueeze(0)
                ).squeeze(0)
            )
        context_residual_bias: torch.Tensor | None = None
        if self.outcome_context_residual_enabled:
            assert memory_features is not None
            assert dt_features is not None
            context_residual_bias = (
                self.outcome_recovery_residual_scale
                * recovery_gate
                * self.outcome_context_residual_adapter(
                    torch.cat(
                        [
                            encoded["shared_embedding"],
                            memory_features,
                            dt_features,
                        ],
                        dim=0,
                    ).unsqueeze(0)
                ).squeeze(0)
            )
        planning_residual_bias: torch.Tensor | None = None
        planning_gate = torch.tensor(
            0.0,
            dtype=encoded["shared_embedding"].dtype,
            device=encoded["shared_embedding"].device,
        )
        if self.digital_twin_planning_residual_enabled:
            assert dt_features is not None
            planning_gate = torch.maximum(dt_features[1], dt_features[2])
            planning_residual_bias = (
                self.digital_twin_planning_residual_scale
                * planning_gate
                * self.digital_twin_planning_adapter(
                    encoded["shared_embedding"].unsqueeze(0)
                ).squeeze(0)
            )
        combined_residual_bias: torch.Tensor | None = None
        for residual_bias in (
            recovery_residual_bias,
            context_residual_bias,
            planning_residual_bias,
        ):
            if residual_bias is None:
                continue
            combined_residual_bias = (
                residual_bias
                if combined_residual_bias is None
                else combined_residual_bias + residual_bias
            )
        if not self.use_hierarchy:
            flat_logits = self.flat_actor(encoded["shared_embedding"].unsqueeze(0)).squeeze(0)
            if combined_residual_bias is not None:
                flat_logits = flat_logits + combined_residual_bias
            critic_context_key = "centralized_critic_context" if self.centralized_critic else "critic_context"
            critic_context = encoded.get(critic_context_key, encoded["critic_context"])
            value = self.flat_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
            output = {
                "encoded": encoded,
                "flat_logits": flat_logits,
                "value": value,
                "critic_mode": "centralized" if self.centralized_critic else "independent",
                "critic_context_key": critic_context_key,
                "head_values": {},
                "outcome_recovery_residual_gate": recovery_gate,
                "digital_twin_planning_residual_gate": planning_gate,
                "outcome_recovery_residual_bias": recovery_residual_bias,
                "outcome_context_residual_bias": context_residual_bias,
                "digital_twin_planning_residual_bias": planning_residual_bias,
            }
            if self.option_gate_enabled:
                output["option_logits"] = self.option_actor(encoded["shared_embedding"].unsqueeze(0)).squeeze(0)
                if self.option_counterfactual_critic_enabled:
                    output["option_q_values"] = self.option_critic(
                        encoded["shared_embedding"].unsqueeze(0)
                    ).squeeze(0)
            if self.env_action_model_critic_enabled:
                output["env_action_q_values"] = self.env_action_critic(
                    encoded["shared_embedding"].unsqueeze(0)
                ).squeeze(0)
            return output

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

        output = {
            "encoded": encoded,
            "slow_logits": slow_logits,
            "fast_logits": fast_logits,
            "event_logits": event_logits,
            "value": value,
            "head_values": head_values,
            "event_logit_temperature": effective_event_temperature,
            "critic_mode": "centralized" if self.centralized_critic else "independent",
            "critic_context_key": critic_context_key,
            "outcome_recovery_residual_gate": recovery_gate,
            "digital_twin_planning_residual_gate": planning_gate,
            "outcome_recovery_residual_bias": recovery_residual_bias,
            "outcome_context_residual_bias": context_residual_bias,
            "digital_twin_planning_residual_bias": planning_residual_bias,
        }
        if combined_residual_bias is not None:
            output["env_action_logits_bias"] = combined_residual_bias
        if self.option_gate_enabled:
            output["option_logits"] = self.option_actor(encoded["shared_embedding"].unsqueeze(0)).squeeze(0)
            if self.option_counterfactual_critic_enabled:
                output["option_q_values"] = self.option_critic(
                    encoded["shared_embedding"].unsqueeze(0)
                ).squeeze(0)
        if self.env_action_model_critic_enabled:
            output["env_action_q_values"] = self.env_action_critic(
                encoded["shared_embedding"].unsqueeze(0)
            ).squeeze(0)
        return output


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
        conservative_imitation_enabled: bool = False,
        conservative_imitation_reward_quantile: float = 0.35,
        conservative_imitation_min_weight: float = 0.10,
        conservative_imitation_max_weight: float = 1.60,
        conservative_imitation_shortfall_coef: float = 0.90,
        conservative_imitation_failure_coef: float = 0.65,
        conservative_imitation_mismatch_coef: float = 0.25,
        conservative_imitation_success_decay: float = 0.30,
        mechanism_aux_coef: float = 0.0,
        mechanism_window_weight: float = 1.0,
        prepare_action_prior_weight: float = 0.5,
        mechanism_entropy_coef: float = 0.0,
        mechanism_retention_start_update: int = 0,
        mechanism_aux_coef_floor_after_update: float = 0.0,
        mechanism_window_weight_floor_after_update: float = 1.0,
        mechanism_entropy_floor_after_update: float = 0.0,
        mechanism_aux_current_cache_fill_enabled: bool = True,
        retrospective_handoff_aux_enabled: bool = False,
        retrospective_handoff_aux_max_eta: float = 6.0,
        retrospective_handoff_aux_min_score: float = 0.08,
        retrospective_handoff_aux_prepare_weight: float = 0.70,
        retrospective_handoff_aux_transition_weight: float = 1.45,
        mechanism_credit_prd_enabled: bool = False,
        mechanism_credit_policy_coef: float = 0.0,
        mechanism_credit_event_coef: float = 0.0,
        mechanism_credit_option_coef: float = 0.0,
        mechanism_credit_clip: float = 2.0,
        mechanism_credit_success_bonus: float = 1.0,
        mechanism_credit_prepare_bonus: float = 0.45,
        mechanism_credit_ready_bonus: float = 0.75,
        mechanism_credit_prefetch_hit_bonus: float = 0.55,
        mechanism_credit_miss_penalty: float = 0.55,
        mechanism_credit_false_positive_penalty: float = 0.35,
        mechanism_credit_min_context: float = 0.15,
        mechanism_focal_aux_enabled: bool = False,
        mechanism_focal_gamma: float = 1.5,
        digital_twin_handoff_fusion_enabled: bool = False,
        digital_twin_handoff_slow_scale: float = 0.35,
        digital_twin_handoff_fast_scale: float = 0.45,
        digital_twin_handoff_event_scale: float = 0.85,
        digital_twin_handoff_critic_scale: float = 0.70,
        digital_twin_planning_residual_enabled: bool = False,
        digital_twin_planning_residual_scale: float = 1.0,
        outcome_memory_fusion_enabled: bool = False,
        outcome_memory_actor_scale: float = 0.80,
        outcome_memory_critic_scale: float = 0.70,
        outcome_recovery_residual_enabled: bool = False,
        outcome_recovery_residual_scale: float = 1.0,
        outcome_context_residual_enabled: bool = False,
        digital_twin_policy_prior_enabled: bool = False,
        digital_twin_policy_prior_logit_bias: float = 0.0,
        digital_twin_policy_prior_event_scale: float = 1.0,
        digital_twin_policy_prior_slow_scale: float = 0.75,
        digital_twin_policy_prior_fast_scale: float = 0.25,
        digital_twin_policy_prior_prepare_threshold: float = 0.20,
        digital_twin_policy_prior_prefetch_threshold: float = 0.22,
        digital_twin_policy_prior_confidence_floor: float = 0.12,
        digital_twin_policy_prior_distill_coef: float = 0.0,
        digital_twin_policy_prior_distill_warmup_updates: int = 0,
        digital_twin_policy_prior_distill_decay: float = 0.90,
        digital_twin_policy_prior_advantage_weight: float = 0.35,
        digital_twin_policy_prior_max_weight: float = 2.0,
        digital_twin_policy_prior_pacing_enabled: bool = False,
        digital_twin_policy_prior_pacing_threshold: float = 0.55,
        digital_twin_policy_prior_pacing_fast_scale: float = 1.0,
        digital_twin_policy_prior_pacing_event_suppression: float = 0.45,
        digital_twin_policy_prior_pacing_slow_suppression: float = 0.35,
        digital_twin_policy_prior_pacing_short_dag_threshold: int = 9,
        digital_twin_policy_prior_env_action_bias_enabled: bool = False,
        digital_twin_policy_prior_env_action_logit_bias: float = 0.0,
        digital_twin_policy_prior_continuation_threshold: float = 0.55,
        digital_twin_policy_prior_continuation_prepare_scale: float = 1.0,
        digital_twin_policy_prior_continuation_wait_scale: float = 0.75,
        digital_twin_policy_prior_continuation_steady_suppression: float = 0.30,
        digital_twin_policy_prior_adaptive_wait_enabled: bool = False,
        digital_twin_policy_prior_wait_ready_threshold: float = 0.55,
        digital_twin_policy_prior_wait_timing_ceiling: float = 0.38,
        digital_twin_policy_prior_wait_cache_ready_scale: float = 1.15,
        digital_twin_policy_prior_prepare_not_ready_scale: float = 1.0,
        env_action_ppo_enabled: bool = False,
        env_action_ppo_coef: float = 0.0,
        env_action_ppo_advantage_blend: float = 0.65,
        env_action_ppo_teacher_coef: float = 0.0,
        env_action_ppo_mechanism_focus: float = 0.0,
        env_action_sparse_recovery_focus: float = 0.0,
        env_action_risk_adjusted_recovery_coef: float = 0.0,
        env_action_risk_adjusted_recovery_floor: float = 0.30,
        env_action_adapter_miss_counterfactual_coef: float = 0.0,
        cache_feasibility_prior_enabled: bool = False,
        cache_feasibility_cache_fill_bias: float = 0.0,
        cache_feasibility_steady_penalty: float = 0.0,
        cache_feasibility_prepare_penalty: float = 0.0,
        cache_feasibility_prefetch_penalty: float = 0.0,
        cache_feasibility_current_miss_prepare_penalty: float = 0.0,
        cache_feasibility_current_miss_prefetch_penalty: float = 0.0,
        cache_feasibility_min_context: float = 0.0,
        handoff_alignment_barrier_enabled: bool = False,
        handoff_alignment_barrier_prepare_penalty: float = 0.0,
        handoff_alignment_barrier_prefetch_penalty: float = 0.0,
        handoff_alignment_barrier_current_fill_bias: float = 0.0,
        handoff_alignment_barrier_target_mismatch_penalty: float = 0.0,
        handoff_alignment_barrier_late_eta_penalty: float = 0.0,
        handoff_alignment_barrier_min_context: float = 0.0,
        sparse_handoff_recovery_prior_enabled: bool = False,
        sparse_handoff_recovery_prefetch_bias: float = 0.0,
        sparse_handoff_recovery_prepare_bias: float = 0.0,
        sparse_handoff_recovery_current_fill_bias: float = 0.0,
        sparse_handoff_recovery_steady_bias: float = 0.0,
        sparse_handoff_recovery_local_penalty: float = 0.0,
        sparse_handoff_recovery_min_context: float = 0.0,
        sparse_handoff_recovery_max_eta: int = 16,
        sparse_handoff_realization_credit_enabled: bool = False,
        sparse_handoff_realization_success_bonus: float = 0.0,
        sparse_handoff_realization_ready_bonus: float = 0.0,
        sparse_handoff_realization_prefetch_bonus: float = 0.0,
        sparse_handoff_realization_failed_prepare_penalty: float = 0.0,
        sparse_handoff_realization_local_penalty: float = 0.0,
        sparse_handoff_realization_min_context: float = 0.0,
        sparse_handoff_option_prior_enabled: bool = False,
        sparse_handoff_option_prepare_bias: float = 0.0,
        sparse_handoff_option_popularity_penalty: float = 0.0,
        sparse_handoff_option_local_penalty: float = 0.0,
        sparse_handoff_option_min_context: float = 0.0,
        sparse_handoff_option_max_eta: int = 16,
        env_action_ppo_max_weight: float = 2.50,
        env_action_ppo_ratio_barrier_coef: float = 0.0,
        env_action_ppo_ratio_barrier_margin: float = 0.35,
        env_action_counterfactual_margin_enabled: bool = False,
        env_action_counterfactual_margin_coef: float = 0.0,
        env_action_counterfactual_margin_min_gap: float = 0.05,
        env_action_counterfactual_margin_max_weight: float = 2.0,
        env_action_counterfactual_margin_advantage_gate: float = 0.0,
        env_action_counterfactual_margin_advantage_blend: float = 0.5,
        argmax_margin_regularization_enabled: bool = False,
        argmax_margin_coef: float = 0.0,
        argmax_margin_min_gap: float = 0.25,
        argmax_margin_max_weight: float = 4.0,
        argmax_margin_tail_risk_threshold: float = 0.10,
        argmax_margin_mechanism_penalty_scale: float = 1.0,
        event_prd_advantage_enabled: bool = False,
        event_prd_advantage_coef: float = 0.0,
        event_prd_advantage_clip: float = 2.0,
        delayed_mechanism_credit_enabled: bool = False,
        delayed_mechanism_credit_policy_coef: float = 0.0,
        delayed_mechanism_credit_event_coef: float = 0.0,
        delayed_mechanism_credit_horizon: int = 4,
        delayed_mechanism_credit_decay: float = 0.70,
        delayed_mechanism_credit_clip: float = 2.0,
        delayed_mechanism_credit_ready_bonus: float = 1.20,
        delayed_mechanism_credit_success_bonus: float = 0.80,
        delayed_mechanism_credit_failure_penalty: float = 0.90,
        delayed_mechanism_credit_missed_prepare_scale: float = 0.55,
        delayed_mechanism_credit_stale_penalty: float = 0.35,
        delayed_mechanism_credit_context_gate: float = 0.25,
        delayed_mechanism_credit_strict_opportunity_enabled: bool = False,
        opportunity_constrained_policy_enabled: bool = False,
        opportunity_constrained_min_context: float = 0.52,
        opportunity_constrained_low_context: float = 0.28,
        opportunity_constrained_prepare_penalty: float = 0.0,
        opportunity_constrained_prefetch_penalty: float = 0.0,
        opportunity_constrained_prepare_bias: float = 0.0,
        opportunity_constrained_prefetch_bias: float = 0.0,
        opportunity_constrained_current_bias: float = 0.0,
        opportunity_constrained_local_bias: float = 0.0,
        opportunity_constrained_no_rsu_service_bias: float = 0.0,
        opportunity_constrained_no_rsu_local_penalty: float = 0.0,
        opportunity_constrained_no_rsu_prepare_bias: float = 0.0,
        opportunity_constrained_no_rsu_prepare_min_context: float = 0.12,
        opportunity_constrained_confidence_floor: float = 0.18,
        opportunity_constrained_uncertainty_ceiling: float = 0.78,
        opportunity_constrained_reliability_floor: float = 0.28,
        net_advantage_prepare_gate_enabled: bool = False,
        net_advantage_prepare_gate_bias: float = 0.0,
        net_advantage_prepare_gate_min_score: float = 0.48,
        net_advantage_prepare_gate_margin: float = 0.14,
        net_advantage_prepare_gate_prefetch_scale: float = 0.35,
        net_advantage_prepare_gate_current_scale: float = 0.55,
        net_advantage_prepare_gate_service_fill_scale: float = 0.0,
        net_advantage_prepare_gate_local_penalty_scale: float = 0.0,
        net_advantage_prepare_gate_cost_scale: float = 0.45,
        net_advantage_prepare_gate_policy_coef: float = 0.0,
        net_advantage_prepare_gate_event_coef: float = 0.0,
        net_advantage_prepare_gate_clip: float = 1.5,
        coverage_recovery_gate_bias_scale: float = 0.70,
        coverage_recovery_gate_min_scale: float = 0.0,
        coverage_recovery_gate_fallback_suppression_scale: float = 1.10,
        coverage_recovery_gate_fast_suppression_scale: float = 0.68,
        coverage_recovery_gate_current_suppression_scale: float = 0.20,
        coverage_recovery_gate_prepare_credit: float = 0.52,
        coverage_recovery_gate_fallback_penalty: float = 0.76,
        coverage_recovery_guard_enabled: bool = False,
        coverage_recovery_final_guard_enabled: bool = False,
        coverage_recovery_final_guard_min_scale: float = 0.20,
        coverage_recovery_final_guard_min_confidence: float = 0.35,
        coverage_recovery_target_memory_option_credit: float = 0.0,
        coverage_recovery_target_memory_option_penalty: float = 0.0,
        service_completion_gate_enabled: bool = False,
        service_completion_gate_bias: float = 0.0,
        service_completion_gate_remaining_nodes_threshold: int = 2,
        service_completion_gate_event_suppression_scale: float = 0.65,
        service_completion_gate_prefetch_suppression_scale: float = 0.45,
        service_completion_gate_fallback_suppression_scale: float = 0.85,
        service_completion_gate_policy_coef: float = 0.0,
        service_completion_gate_event_coef: float = 0.0,
        service_completion_gate_clip: float = 1.5,
        backhaul_aware_policy_enabled: bool = False,
        backhaul_aware_service_fill_bias: float = 0.0,
        backhaul_aware_redundant_fill_penalty: float = 0.0,
        backhaul_aware_no_signal_prefetch_penalty: float = 0.0,
        backhaul_aware_no_signal_prepare_penalty: float = 0.0,
        backhaul_aware_steady_bias: float = 0.0,
        backhaul_aware_service_pressure_floor: float = 0.35,
        advantage_weighted_behavior_regularization_enabled: bool = False,
        advantage_weighted_behavior_coef: float = 0.0,
        advantage_weighted_behavior_positive_coef: float = 1.0,
        advantage_weighted_behavior_negative_coef: float = 0.85,
        advantage_weighted_behavior_temperature: float = 0.75,
        advantage_weighted_behavior_max_weight: float = 2.0,
        advantage_weighted_behavior_positive_gate: float = 0.10,
        advantage_weighted_behavior_negative_gate: float = 0.05,
        advantage_weighted_behavior_mechanism_scale: float = 1.25,
        latency_fallback_bias_enabled: bool = False,
        latency_fallback_bias_strength: float = 0.0,
        latency_fallback_confidence_floor: float = 0.0,
        latency_fallback_slow_suppression_strength: float = 0.0,
        steady_rsu_bias_enabled: bool = False,
        steady_rsu_bias_strength: float = 0.0,
        steady_rsu_confidence_floor: float = 0.0,
        backhaul_guard_enabled: bool = False,
        backhaul_guard_max_reactive_fills_per_adapter: int = 1,
        cache_warm_start_guard_enabled: bool = False,
        cache_warm_start_guard_min_countdown: float = 1.5,
        cache_warm_start_guard_max_prefetch_countdown: float = 0.0,
        predictive_prefetch_admission_guard_enabled: bool = False,
        predictive_prefetch_admission_min_confidence: float = 0.55,
        predictive_prefetch_admission_require_distinct_next: bool = True,
        idle_popularity_fallback_enabled: bool = False,
        idle_popularity_fallback_only_vehicle_fallback: bool = True,
        idle_popularity_prefetch_threshold: int = 2,
        idle_popularity_no_rsu_local_fallback_enabled: bool = False,
        idle_popularity_no_rsu_local_requires_low_context: bool = True,
        idle_popularity_no_rsu_service_continuity_enabled: bool = False,
        idle_popularity_no_rsu_any_action_override_enabled: bool = False,
        option_gate_enabled: bool = False,
        option_gate_count: int = 4,
        option_gate_loss_coef: float = 0.35,
        option_gate_entropy_coef: float = 0.001,
        option_gate_prior_coef: float = 0.0,
        option_gate_prior_warmup_updates: int = 0,
        option_gate_prior_decay: float = 0.85,
        option_gate_prior_logit_bias: float = 0.0,
        option_gate_log_prob_weight: float = 1.0,
        option_gate_context_prior_enabled: bool = False,
        option_gate_deterministic_prior_margin: float = 0.0,
        option_gate_idle_prior_enabled: bool = False,
        option_gate_idle_recovery_mechanism_prior_enabled: bool = False,
        option_gate_idle_recovery_min_context: float = 0.10,
        option_gate_mechanism_preserve_enabled: bool = True,
        option_gate_prd_enabled: bool = False,
        option_gate_prd_coef: float = 0.0,
        option_gate_prd_clip: float = 2.0,
        option_gate_counterfactual_prd_enabled: bool = False,
        option_gate_counterfactual_coef: float = 0.0,
        option_gate_counterfactual_clip: float = 1.5,
        option_counterfactual_critic_enabled: bool = False,
        option_counterfactual_value_coef: float = 0.5,
        option_counterfactual_advantage_coef: float = 1.0,
        option_counterfactual_advantage_clip: float = 2.0,
        option_counterfactual_warmup_updates: int = 2,
        option_counterfactual_tail_weight: float = 0.0,
        option_counterfactual_policy_improvement_enabled: bool | None = None,
        option_counterfactual_policy_improvement_coef: float = 1.0,
        option_counterfactual_policy_improvement_clip: float = 2.0,
        option_counterfactual_policy_improvement_deterministic_only: bool = True,
        option_counterfactual_model_rollout_enabled: bool = False,
        option_counterfactual_model_rollout_horizon: int = 1,
        env_action_model_critic_enabled: bool = False,
        env_action_model_critic_value_coef: float = 0.5,
        env_action_model_critic_advantage_coef: float = 1.0,
        env_action_model_critic_policy_improvement_coef: float = 2.0,
        env_action_model_critic_advantage_clip: float = 2.0,
        env_action_model_critic_warmup_updates: int = 2,
        env_action_model_rollout_enabled: bool = False,
        env_action_model_rollout_horizon: int = 4,
        env_action_model_rollout_horizons: tuple[int, ...] | list[int] | None = None,
        env_action_model_imagination_replay_enabled: bool = False,
        env_action_model_imagination_replay_depths: (
            tuple[int, ...] | list[int] | None
        ) = None,
        env_action_model_imagination_replay_horizons: (
            tuple[int, ...] | list[int] | None
        ) = None,
        env_action_model_imagination_replay_recovery_only: bool = False,
        env_action_model_imagination_beam_search_enabled: bool = False,
        env_action_model_imagination_replay_branch_mode: str = "dominant",
        env_action_model_imagination_replay_branch_top_k: int = 1,
        env_action_model_policy_improvement_enabled: bool = False,
        env_action_model_policy_improvement_coef: float = 0.5,
        env_action_model_policy_improvement_temperature: float = 2.0,
        env_action_model_policy_improvement_robust_horizons_enabled: bool = False,
        env_action_model_policy_improvement_horizon_risk_coef: float = 0.75,
        env_action_model_policy_improvement_horizon_aggregation_mode: str = "mean_std",
        env_action_model_policy_improvement_horizon_lambda: float = 0.90,
        env_action_model_policy_improvement_adaptive_kl_enabled: bool = False,
        env_action_model_policy_improvement_target_kl: float = 0.03,
        env_action_model_policy_improvement_regret_adaptive_kl_enabled: bool = False,
        env_action_model_policy_improvement_max_target_kl: float = 0.35,
        env_action_model_policy_improvement_regret_priority_coef: float = 0.0,
        env_action_model_policy_improvement_tail_distillation_enabled: bool = False,
        env_action_model_policy_improvement_tail_quantile: float = 0.75,
        env_action_model_policy_improvement_tail_min_regret: float = 0.50,
        env_action_model_policy_improvement_tail_epochs: int = 0,
        env_action_model_policy_improvement_tail_coef: float = 1.0,
        env_action_model_policy_improvement_tail_max_policy_kl: float = 0.0,
        env_action_model_policy_improvement_tail_recovery_only: bool = False,
        env_action_model_policy_improvement_tail_adapter_only: bool = False,
        env_action_model_policy_improvement_tail_beam_only: bool = False,
        env_action_model_policy_improvement_tail_planning_adapter_only: bool = False,
        env_action_model_policy_improvement_tail_residual_optimizer_enabled: bool = False,
        env_action_model_policy_improvement_tail_residual_learning_rate: float = 0.01,
        env_action_model_policy_improvement_tail_residual_backtrack_factor: float = 0.5,
        env_action_model_policy_improvement_tail_residual_min_learning_rate: float = 1e-5,
        env_action_model_policy_improvement_tail_residual_max_backtracks: int = 4,
        env_action_model_policy_improvement_tail_logit_projection_enabled: bool = False,
        env_action_model_policy_improvement_tail_target_balance_enabled: bool = False,
        env_action_model_policy_improvement_tail_target_balance_power: float = 0.5,
        env_action_model_policy_improvement_tail_target_balance_max_weight: float = 4.0,
        learned_transition_model_enabled: bool = False,
        learned_transition_model_planner_enabled: bool = False,
        learned_transition_model_ensemble_size: int = 5,
        learned_transition_model_hidden_dim: int = 64,
        learned_transition_model_learning_rate: float = 3e-3,
        learned_transition_model_fit_epochs: int = 4,
        learned_transition_model_max_samples: int = 4096,
        learned_transition_model_min_samples: int = 64,
        learned_transition_model_discount: float = 0.99,
        learned_transition_model_risk_coef: float = 0.8,
        learned_transition_model_exploration_coef: float = 0.0,
        learned_transition_model_policy_coef: float = 1.0,
        learned_transition_model_policy_prior_coef: float = 0.15,
        learned_transition_model_min_margin: float = 0.02,
        learned_transition_model_warmup_updates: int = 1,
        env_action_model_online_planner_enabled: bool = False,
        env_action_model_online_planner_coef: float = 1.0,
        env_action_model_online_planner_mechanism_coef: float = 0.0,
        env_action_model_online_planner_policy_prior_coef: float = 0.15,
        env_action_model_online_planner_min_margin: float = 0.0,
        env_action_model_online_planner_prefer_beam_targets: bool = False,
        env_action_model_resource_constraint_enabled: bool = False,
        env_action_model_resource_cost_coef: float = 0.0,
        env_action_model_resource_cost_scale: float = 64.0,
        env_action_model_adaptive_horizon_enabled: bool = False,
        env_action_model_adaptive_horizon_temperature: float = 1.0,
        env_action_model_beam_search_enabled: bool = False,
        env_action_model_beam_search_horizon: int = 4,
        env_action_model_beam_search_width: int = 2,
        env_action_model_beam_search_context_only: bool = True,
        env_action_model_beam_search_min_eta: int = 0,
        env_action_model_beam_search_max_eta: int = 999,
        env_action_model_policy_improvement_prefer_beam_targets: bool = False,
        counterfactual_teacher_prd_enabled: bool = False,
        counterfactual_teacher_event_coef: float = 0.0,
        counterfactual_teacher_option_coef: float = 0.0,
        counterfactual_teacher_clip: float = 2.0,
        counterfactual_teacher_mechanism_bonus: float = 0.0,
        counterfactual_teacher_missed_prepare_penalty: float = 0.0,
        counterfactual_teacher_local_bonus: float = 0.0,
        counterfactual_teacher_current_rsu_penalty: float = 0.0,
        counterfactual_teacher_invalid_mechanism_penalty: float = 0.0,
        service_continuity_teacher_enabled: bool = False,
        service_continuity_current_bonus: float = 0.0,
        service_continuity_prepare_bonus: float = 0.0,
        service_continuity_local_penalty: float = 0.0,
        service_continuity_min_prepare_context: float = 0.20,
        tail_risk_prd_enabled: bool = False,
        tail_risk_policy_coef: float = 0.0,
        tail_risk_event_coef: float = 0.0,
        tail_risk_option_coef: float = 0.0,
        tail_risk_clip: float = 2.0,
        tail_risk_quantile: float = 0.20,
        tail_risk_reward_shortfall_coef: float = 0.0,
        tail_risk_service_coef: float = 0.0,
        tail_risk_continuity_coef: float = 0.0,
        tail_risk_handoff_failure_coef: float = 0.0,
        tail_risk_failed_mechanism_coef: float = 0.0,
        tail_risk_redundant_mechanism_coef: float = 0.0,
        tail_risk_success_credit: float = 0.0,
        opportunity_prd_enabled: bool = False,
        opportunity_policy_coef: float = 0.0,
        opportunity_event_coef: float = 0.0,
        opportunity_option_coef: float = 0.0,
        opportunity_clip: float = 1.6,
        opportunity_reward_quantile: float = 0.60,
        opportunity_reward_surplus_coef: float = 0.0,
        opportunity_service_success_coef: float = 0.0,
        opportunity_cache_hit_coef: float = 0.0,
        opportunity_continuity_coef: float = 0.0,
        opportunity_current_rsu_efficiency_coef: float = 0.0,
        opportunity_local_fallback_coef: float = 0.0,
        opportunity_backhaul_penalty_coef: float = 0.0,
        opportunity_delay_penalty_coef: float = 0.0,
        opportunity_failed_service_penalty_coef: float = 0.0,
        opportunity_mechanism_success_bonus: float = 0.0,
        handoff_risk_prd_enabled: bool = False,
        handoff_risk_event_coef: float = 0.0,
        handoff_risk_option_coef: float = 0.0,
        handoff_risk_clip: float = 1.5,
        handoff_risk_failure_penalty: float = 1.0,
        handoff_risk_ready_bonus: float = 0.75,
        handoff_risk_prepare_bonus: float = 0.25,
        handoff_risk_unprepared_penalty: float = 0.35,
        handoff_risk_confidence_threshold: float = 0.55,
        handoff_risk_cost_dual_enabled: bool = False,
        handoff_risk_cost_dual_lr: float = 0.0,
        handoff_risk_cost_target: float = 0.0,
        handoff_risk_cost_dual_max: float = 1.5,
        handoff_risk_cost_dual_initial: float = 0.0,
        idle_execution_prd_enabled: bool = False,
        idle_execution_policy_coef: float = 0.0,
        idle_execution_option_coef: float = 0.0,
        idle_execution_clip: float = 1.5,
        idle_execution_current_rsu_delay_coef: float = 0.35,
        idle_execution_local_bonus: float = 0.25,
        idle_execution_mechanism_penalty: float = 0.30,
        idle_execution_timing_threshold: float = 0.28,
        idle_execution_mechanism_preserve_bonus: float = 0.18,
        net_utility_prd_enabled: bool = False,
        net_utility_backhaul_coef: float = 0.0,
        net_utility_migration_coef: float = 0.0,
        net_utility_expired_prefetch_coef: float = 0.0,
        net_utility_idle_prefetch_penalty: float = 0.0,
        net_utility_failed_mechanism_penalty: float = 0.0,
        net_utility_failed_mechanism_backhaul_coef: float = 0.0,
        net_utility_mechanism_window_failed_penalty_scale: float = 1.0,
        net_utility_success_bonus: float = 0.0,
        net_utility_backhaul_normalizer: float = 64.0,
        net_utility_cost_dual_enabled: bool = False,
        net_utility_cost_dual_lr: float = 0.0,
        net_utility_cost_target: float = 0.0,
        net_utility_cost_dual_max: float = 2.0,
        net_utility_cost_dual_initial: float = 0.0,
        net_utility_option_termination_enabled: bool = False,
        net_utility_option_termination_conservative_enabled: bool = False,
        net_utility_option_termination_max_timing_support: float = 0.20,
        dag_aware_option_termination_enabled: bool = False,
        dag_aware_option_min_critical_path: int = 6,
        dag_aware_option_short_workflow_max_nodes: int = 12,
        dag_aware_option_branching_successors: int = 3,
        dag_aware_idle_prefetch_confidence_floor: float = 0.65,
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
        self._conservative_imitation_enabled = bool(conservative_imitation_enabled)
        self._conservative_imitation_reward_quantile = max(
            0.0,
            min(float(conservative_imitation_reward_quantile), 1.0),
        )
        self._conservative_imitation_min_weight = max(float(conservative_imitation_min_weight), 0.0)
        self._conservative_imitation_max_weight = max(
            float(conservative_imitation_max_weight),
            self._conservative_imitation_min_weight,
        )
        self._conservative_imitation_shortfall_coef = max(
            float(conservative_imitation_shortfall_coef),
            0.0,
        )
        self._conservative_imitation_failure_coef = max(float(conservative_imitation_failure_coef), 0.0)
        self._conservative_imitation_mismatch_coef = max(float(conservative_imitation_mismatch_coef), 0.0)
        self._conservative_imitation_success_decay = max(
            0.0,
            min(float(conservative_imitation_success_decay), 1.0),
        )
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
        self._retrospective_handoff_aux_enabled = bool(retrospective_handoff_aux_enabled)
        self._retrospective_handoff_aux_max_eta = max(float(retrospective_handoff_aux_max_eta), 1.0)
        self._retrospective_handoff_aux_min_score = max(
            0.0,
            min(float(retrospective_handoff_aux_min_score), 1.0),
        )
        self._retrospective_handoff_aux_prepare_weight = max(
            float(retrospective_handoff_aux_prepare_weight),
            0.0,
        )
        self._retrospective_handoff_aux_transition_weight = max(
            float(retrospective_handoff_aux_transition_weight),
            1.0,
        )
        self._mechanism_credit_prd_enabled = bool(mechanism_credit_prd_enabled)
        self._mechanism_credit_policy_coef = max(float(mechanism_credit_policy_coef), 0.0)
        self._mechanism_credit_event_coef = max(float(mechanism_credit_event_coef), 0.0)
        self._mechanism_credit_option_coef = max(float(mechanism_credit_option_coef), 0.0)
        self._mechanism_credit_clip = max(float(mechanism_credit_clip), 0.0)
        self._mechanism_credit_success_bonus = max(float(mechanism_credit_success_bonus), 0.0)
        self._mechanism_credit_prepare_bonus = max(float(mechanism_credit_prepare_bonus), 0.0)
        self._mechanism_credit_ready_bonus = max(float(mechanism_credit_ready_bonus), 0.0)
        self._mechanism_credit_prefetch_hit_bonus = max(float(mechanism_credit_prefetch_hit_bonus), 0.0)
        self._mechanism_credit_miss_penalty = max(float(mechanism_credit_miss_penalty), 0.0)
        self._mechanism_credit_false_positive_penalty = max(float(mechanism_credit_false_positive_penalty), 0.0)
        self._mechanism_credit_min_context = max(
            0.0,
            min(float(mechanism_credit_min_context), 1.0),
        )
        self._mechanism_focal_aux_enabled = bool(mechanism_focal_aux_enabled)
        self._mechanism_focal_gamma = max(float(mechanism_focal_gamma), 0.0)
        self._digital_twin_handoff_fusion_enabled = bool(digital_twin_handoff_fusion_enabled)
        self._digital_twin_handoff_slow_scale = max(float(digital_twin_handoff_slow_scale), 0.0)
        self._digital_twin_handoff_fast_scale = max(float(digital_twin_handoff_fast_scale), 0.0)
        self._digital_twin_handoff_event_scale = max(float(digital_twin_handoff_event_scale), 0.0)
        self._digital_twin_handoff_critic_scale = max(float(digital_twin_handoff_critic_scale), 0.0)
        self._digital_twin_planning_residual_enabled = bool(
            digital_twin_planning_residual_enabled
        )
        self._digital_twin_planning_residual_scale = max(
            float(digital_twin_planning_residual_scale),
            0.0,
        )
        self._outcome_memory_fusion_enabled = bool(
            outcome_memory_fusion_enabled
        )
        self._outcome_memory_actor_scale = max(
            float(outcome_memory_actor_scale),
            0.0,
        )
        self._outcome_memory_critic_scale = max(
            float(outcome_memory_critic_scale),
            0.0,
        )
        self._outcome_recovery_residual_enabled = bool(
            outcome_recovery_residual_enabled
        )
        self._outcome_recovery_residual_scale = max(
            float(outcome_recovery_residual_scale),
            0.0,
        )
        self._outcome_context_residual_enabled = bool(
            outcome_context_residual_enabled
        )
        self._digital_twin_policy_prior_enabled = bool(digital_twin_policy_prior_enabled)
        self._digital_twin_policy_prior_logit_bias = max(float(digital_twin_policy_prior_logit_bias), 0.0)
        self._digital_twin_policy_prior_event_scale = max(float(digital_twin_policy_prior_event_scale), 0.0)
        self._digital_twin_policy_prior_slow_scale = max(float(digital_twin_policy_prior_slow_scale), 0.0)
        self._digital_twin_policy_prior_fast_scale = max(float(digital_twin_policy_prior_fast_scale), 0.0)
        self._digital_twin_policy_prior_prepare_threshold = max(
            0.0,
            min(float(digital_twin_policy_prior_prepare_threshold), 1.0),
        )
        self._digital_twin_policy_prior_prefetch_threshold = max(
            0.0,
            min(float(digital_twin_policy_prior_prefetch_threshold), 1.0),
        )
        self._digital_twin_policy_prior_confidence_floor = max(
            0.0,
            min(float(digital_twin_policy_prior_confidence_floor), 1.0),
        )
        self._digital_twin_policy_prior_distill_coef = max(
            float(digital_twin_policy_prior_distill_coef),
            0.0,
        )
        self._digital_twin_policy_prior_distill_warmup_updates = max(
            int(digital_twin_policy_prior_distill_warmup_updates),
            0,
        )
        self._digital_twin_policy_prior_distill_decay = max(
            float(digital_twin_policy_prior_distill_decay),
            0.0,
        )
        self._digital_twin_policy_prior_advantage_weight = max(
            float(digital_twin_policy_prior_advantage_weight),
            0.0,
        )
        self._digital_twin_policy_prior_max_weight = max(
            float(digital_twin_policy_prior_max_weight),
            0.0,
        )
        self._digital_twin_policy_prior_pacing_enabled = bool(digital_twin_policy_prior_pacing_enabled)
        self._digital_twin_policy_prior_pacing_threshold = max(
            0.0,
            min(float(digital_twin_policy_prior_pacing_threshold), 1.0),
        )
        self._digital_twin_policy_prior_pacing_fast_scale = max(
            float(digital_twin_policy_prior_pacing_fast_scale),
            0.0,
        )
        self._digital_twin_policy_prior_pacing_event_suppression = max(
            float(digital_twin_policy_prior_pacing_event_suppression),
            0.0,
        )
        self._digital_twin_policy_prior_pacing_slow_suppression = max(
            float(digital_twin_policy_prior_pacing_slow_suppression),
            0.0,
        )
        self._digital_twin_policy_prior_pacing_short_dag_threshold = max(
            int(digital_twin_policy_prior_pacing_short_dag_threshold),
            1,
        )
        self._digital_twin_policy_prior_env_action_bias_enabled = bool(
            digital_twin_policy_prior_env_action_bias_enabled
        )
        self._digital_twin_policy_prior_env_action_logit_bias = max(
            float(digital_twin_policy_prior_env_action_logit_bias),
            0.0,
        )
        self._digital_twin_policy_prior_continuation_threshold = max(
            0.0,
            min(float(digital_twin_policy_prior_continuation_threshold), 1.0),
        )
        self._digital_twin_policy_prior_continuation_prepare_scale = max(
            float(digital_twin_policy_prior_continuation_prepare_scale),
            0.0,
        )
        self._digital_twin_policy_prior_continuation_wait_scale = max(
            float(digital_twin_policy_prior_continuation_wait_scale),
            0.0,
        )
        self._digital_twin_policy_prior_continuation_steady_suppression = max(
            float(digital_twin_policy_prior_continuation_steady_suppression),
            0.0,
        )
        self._digital_twin_policy_prior_adaptive_wait_enabled = bool(
            digital_twin_policy_prior_adaptive_wait_enabled
        )
        self._digital_twin_policy_prior_wait_ready_threshold = max(
            0.0,
            min(float(digital_twin_policy_prior_wait_ready_threshold), 1.0),
        )
        self._digital_twin_policy_prior_wait_timing_ceiling = max(
            0.0,
            min(float(digital_twin_policy_prior_wait_timing_ceiling), 1.0),
        )
        self._digital_twin_policy_prior_wait_cache_ready_scale = max(
            float(digital_twin_policy_prior_wait_cache_ready_scale),
            0.0,
        )
        self._digital_twin_policy_prior_prepare_not_ready_scale = max(
            float(digital_twin_policy_prior_prepare_not_ready_scale),
            0.0,
        )
        self._env_action_ppo_enabled = bool(env_action_ppo_enabled)
        self._env_action_ppo_coef = max(float(env_action_ppo_coef), 0.0)
        self._env_action_ppo_advantage_blend = max(
            0.0,
            min(float(env_action_ppo_advantage_blend), 1.0),
        )
        self._env_action_ppo_teacher_coef = max(float(env_action_ppo_teacher_coef), 0.0)
        self._env_action_ppo_mechanism_focus = max(float(env_action_ppo_mechanism_focus), 0.0)
        self._env_action_sparse_recovery_focus = max(float(env_action_sparse_recovery_focus), 0.0)
        self._env_action_risk_adjusted_recovery_coef = max(
            float(env_action_risk_adjusted_recovery_coef),
            0.0,
        )
        self._env_action_risk_adjusted_recovery_floor = max(
            0.0,
            min(float(env_action_risk_adjusted_recovery_floor), 1.0),
        )
        self._env_action_adapter_miss_counterfactual_coef = max(
            float(env_action_adapter_miss_counterfactual_coef),
            0.0,
        )
        self._cache_feasibility_prior_enabled = bool(cache_feasibility_prior_enabled)
        self._cache_feasibility_cache_fill_bias = max(float(cache_feasibility_cache_fill_bias), 0.0)
        self._cache_feasibility_steady_penalty = max(float(cache_feasibility_steady_penalty), 0.0)
        self._cache_feasibility_prepare_penalty = max(float(cache_feasibility_prepare_penalty), 0.0)
        self._cache_feasibility_prefetch_penalty = max(float(cache_feasibility_prefetch_penalty), 0.0)
        self._cache_feasibility_current_miss_prepare_penalty = max(
            float(cache_feasibility_current_miss_prepare_penalty),
            0.0,
        )
        self._cache_feasibility_current_miss_prefetch_penalty = max(
            float(cache_feasibility_current_miss_prefetch_penalty),
            0.0,
        )
        self._cache_feasibility_min_context = max(
            0.0,
            min(float(cache_feasibility_min_context), 1.0),
        )
        self._handoff_alignment_barrier_enabled = bool(handoff_alignment_barrier_enabled)
        self._handoff_alignment_barrier_prepare_penalty = max(
            float(handoff_alignment_barrier_prepare_penalty),
            0.0,
        )
        self._handoff_alignment_barrier_prefetch_penalty = max(
            float(handoff_alignment_barrier_prefetch_penalty),
            0.0,
        )
        self._handoff_alignment_barrier_current_fill_bias = max(
            float(handoff_alignment_barrier_current_fill_bias),
            0.0,
        )
        self._handoff_alignment_barrier_target_mismatch_penalty = max(
            float(handoff_alignment_barrier_target_mismatch_penalty),
            0.0,
        )
        self._handoff_alignment_barrier_late_eta_penalty = max(
            float(handoff_alignment_barrier_late_eta_penalty),
            0.0,
        )
        self._handoff_alignment_barrier_min_context = max(
            0.0,
            min(float(handoff_alignment_barrier_min_context), 1.0),
        )
        self._sparse_handoff_recovery_prior_enabled = bool(
            sparse_handoff_recovery_prior_enabled
        )
        self._sparse_handoff_recovery_prefetch_bias = max(
            float(sparse_handoff_recovery_prefetch_bias),
            0.0,
        )
        self._sparse_handoff_recovery_prepare_bias = max(
            float(sparse_handoff_recovery_prepare_bias),
            0.0,
        )
        self._sparse_handoff_recovery_current_fill_bias = max(
            float(sparse_handoff_recovery_current_fill_bias),
            0.0,
        )
        self._sparse_handoff_recovery_steady_bias = max(
            float(sparse_handoff_recovery_steady_bias),
            0.0,
        )
        self._sparse_handoff_recovery_local_penalty = max(
            float(sparse_handoff_recovery_local_penalty),
            0.0,
        )
        self._sparse_handoff_recovery_min_context = max(
            0.0,
            min(float(sparse_handoff_recovery_min_context), 1.0),
        )
        self._sparse_handoff_recovery_max_eta = max(
            int(sparse_handoff_recovery_max_eta),
            1,
        )
        self._sparse_handoff_realization_credit_enabled = bool(
            sparse_handoff_realization_credit_enabled
        )
        self._sparse_handoff_realization_success_bonus = max(
            float(sparse_handoff_realization_success_bonus),
            0.0,
        )
        self._sparse_handoff_realization_ready_bonus = max(
            float(sparse_handoff_realization_ready_bonus),
            0.0,
        )
        self._sparse_handoff_realization_prefetch_bonus = max(
            float(sparse_handoff_realization_prefetch_bonus),
            0.0,
        )
        self._sparse_handoff_realization_failed_prepare_penalty = max(
            float(sparse_handoff_realization_failed_prepare_penalty),
            0.0,
        )
        self._sparse_handoff_realization_local_penalty = max(
            float(sparse_handoff_realization_local_penalty),
            0.0,
        )
        self._sparse_handoff_realization_min_context = max(
            0.0,
            min(float(sparse_handoff_realization_min_context), 1.0),
        )
        self._sparse_handoff_option_prior_enabled = bool(
            sparse_handoff_option_prior_enabled
        )
        self._sparse_handoff_option_prepare_bias = max(
            float(sparse_handoff_option_prepare_bias),
            0.0,
        )
        self._sparse_handoff_option_popularity_penalty = max(
            float(sparse_handoff_option_popularity_penalty),
            0.0,
        )
        self._sparse_handoff_option_local_penalty = max(
            float(sparse_handoff_option_local_penalty),
            0.0,
        )
        self._sparse_handoff_option_min_context = max(
            0.0,
            min(float(sparse_handoff_option_min_context), 1.0),
        )
        self._sparse_handoff_option_max_eta = max(
            int(sparse_handoff_option_max_eta),
            1,
        )
        self._env_action_ppo_max_weight = max(float(env_action_ppo_max_weight), 0.0)
        self._env_action_ppo_ratio_barrier_coef = max(float(env_action_ppo_ratio_barrier_coef), 0.0)
        self._env_action_ppo_ratio_barrier_margin = max(
            0.0,
            min(float(env_action_ppo_ratio_barrier_margin), 0.95),
        )
        self._env_action_counterfactual_margin_enabled = bool(env_action_counterfactual_margin_enabled)
        self._env_action_counterfactual_margin_coef = max(float(env_action_counterfactual_margin_coef), 0.0)
        self._env_action_counterfactual_margin_min_gap = max(float(env_action_counterfactual_margin_min_gap), 0.0)
        self._env_action_counterfactual_margin_max_weight = max(float(env_action_counterfactual_margin_max_weight), 0.0)
        self._env_action_counterfactual_margin_advantage_gate = max(
            float(env_action_counterfactual_margin_advantage_gate),
            0.0,
        )
        self._env_action_counterfactual_margin_advantage_blend = max(
            0.0,
            min(float(env_action_counterfactual_margin_advantage_blend), 1.0),
        )
        self._argmax_margin_regularization_enabled = bool(argmax_margin_regularization_enabled)
        self._argmax_margin_coef = max(float(argmax_margin_coef), 0.0)
        self._argmax_margin_min_gap = max(float(argmax_margin_min_gap), 0.0)
        self._argmax_margin_max_weight = max(float(argmax_margin_max_weight), 0.0)
        self._argmax_margin_tail_risk_threshold = max(
            0.0,
            min(float(argmax_margin_tail_risk_threshold), 1.0),
        )
        self._argmax_margin_mechanism_penalty_scale = max(
            float(argmax_margin_mechanism_penalty_scale),
            0.0,
        )
        self._event_prd_advantage_enabled = bool(event_prd_advantage_enabled)
        self._event_prd_advantage_coef = max(float(event_prd_advantage_coef), 0.0)
        self._event_prd_advantage_clip = max(float(event_prd_advantage_clip), 0.0)
        self._delayed_mechanism_credit_enabled = bool(delayed_mechanism_credit_enabled)
        self._delayed_mechanism_credit_policy_coef = max(float(delayed_mechanism_credit_policy_coef), 0.0)
        self._delayed_mechanism_credit_event_coef = max(float(delayed_mechanism_credit_event_coef), 0.0)
        self._delayed_mechanism_credit_horizon = max(int(delayed_mechanism_credit_horizon), 1)
        self._delayed_mechanism_credit_decay = max(0.0, min(float(delayed_mechanism_credit_decay), 1.0))
        self._delayed_mechanism_credit_clip = max(float(delayed_mechanism_credit_clip), 0.0)
        self._delayed_mechanism_credit_ready_bonus = max(float(delayed_mechanism_credit_ready_bonus), 0.0)
        self._delayed_mechanism_credit_success_bonus = max(float(delayed_mechanism_credit_success_bonus), 0.0)
        self._delayed_mechanism_credit_failure_penalty = max(float(delayed_mechanism_credit_failure_penalty), 0.0)
        self._delayed_mechanism_credit_missed_prepare_scale = max(
            float(delayed_mechanism_credit_missed_prepare_scale),
            0.0,
        )
        self._delayed_mechanism_credit_stale_penalty = max(float(delayed_mechanism_credit_stale_penalty), 0.0)
        self._delayed_mechanism_credit_context_gate = max(
            0.0,
            min(float(delayed_mechanism_credit_context_gate), 1.0),
        )
        self._delayed_mechanism_credit_strict_opportunity_enabled = bool(
            delayed_mechanism_credit_strict_opportunity_enabled
        )
        self._opportunity_constrained_policy_enabled = bool(opportunity_constrained_policy_enabled)
        self._opportunity_constrained_min_context = max(
            0.0,
            min(float(opportunity_constrained_min_context), 1.0),
        )
        self._opportunity_constrained_low_context = max(
            0.0,
            min(float(opportunity_constrained_low_context), self._opportunity_constrained_min_context),
        )
        self._opportunity_constrained_prepare_penalty = max(
            float(opportunity_constrained_prepare_penalty),
            0.0,
        )
        self._opportunity_constrained_prefetch_penalty = max(
            float(opportunity_constrained_prefetch_penalty),
            0.0,
        )
        self._opportunity_constrained_prepare_bias = max(float(opportunity_constrained_prepare_bias), 0.0)
        self._opportunity_constrained_prefetch_bias = max(float(opportunity_constrained_prefetch_bias), 0.0)
        self._opportunity_constrained_current_bias = max(float(opportunity_constrained_current_bias), 0.0)
        self._opportunity_constrained_local_bias = max(float(opportunity_constrained_local_bias), 0.0)
        self._opportunity_constrained_no_rsu_service_bias = max(
            float(opportunity_constrained_no_rsu_service_bias),
            0.0,
        )
        self._opportunity_constrained_no_rsu_local_penalty = max(
            float(opportunity_constrained_no_rsu_local_penalty),
            0.0,
        )
        self._opportunity_constrained_no_rsu_prepare_bias = max(
            float(opportunity_constrained_no_rsu_prepare_bias),
            0.0,
        )
        self._opportunity_constrained_no_rsu_prepare_min_context = max(
            0.0,
            min(float(opportunity_constrained_no_rsu_prepare_min_context), 1.0),
        )
        self._opportunity_constrained_confidence_floor = max(
            0.0,
            min(float(opportunity_constrained_confidence_floor), 1.0),
        )
        self._opportunity_constrained_uncertainty_ceiling = max(
            0.0,
            min(float(opportunity_constrained_uncertainty_ceiling), 1.0),
        )
        self._opportunity_constrained_reliability_floor = max(
            0.0,
            min(float(opportunity_constrained_reliability_floor), 1.0),
        )
        self._net_advantage_prepare_gate_enabled = bool(net_advantage_prepare_gate_enabled)
        self._net_advantage_prepare_gate_bias = max(float(net_advantage_prepare_gate_bias), 0.0)
        self._net_advantage_prepare_gate_min_score = max(
            0.0,
            min(float(net_advantage_prepare_gate_min_score), 1.0),
        )
        self._net_advantage_prepare_gate_margin = max(float(net_advantage_prepare_gate_margin), 0.0)
        self._net_advantage_prepare_gate_prefetch_scale = max(
            float(net_advantage_prepare_gate_prefetch_scale),
            0.0,
        )
        self._net_advantage_prepare_gate_current_scale = max(
            float(net_advantage_prepare_gate_current_scale),
            0.0,
        )
        self._net_advantage_prepare_gate_service_fill_scale = max(
            float(net_advantage_prepare_gate_service_fill_scale),
            0.0,
        )
        self._net_advantage_prepare_gate_local_penalty_scale = max(
            float(net_advantage_prepare_gate_local_penalty_scale),
            0.0,
        )
        self._net_advantage_prepare_gate_cost_scale = max(
            float(net_advantage_prepare_gate_cost_scale),
            0.0,
        )
        self._net_advantage_prepare_gate_policy_coef = max(
            float(net_advantage_prepare_gate_policy_coef),
            0.0,
        )
        self._net_advantage_prepare_gate_event_coef = max(
            float(net_advantage_prepare_gate_event_coef),
            0.0,
        )
        self._net_advantage_prepare_gate_clip = max(float(net_advantage_prepare_gate_clip), 0.0)
        self._coverage_recovery_gate_bias_scale = max(
            float(coverage_recovery_gate_bias_scale),
            0.0,
        )
        self._coverage_recovery_gate_min_scale = max(
            0.0,
            min(float(coverage_recovery_gate_min_scale), 1.0),
        )
        self._coverage_recovery_gate_fallback_suppression_scale = max(
            float(coverage_recovery_gate_fallback_suppression_scale),
            0.0,
        )
        self._coverage_recovery_gate_fast_suppression_scale = max(
            float(coverage_recovery_gate_fast_suppression_scale),
            0.0,
        )
        self._coverage_recovery_gate_current_suppression_scale = max(
            float(coverage_recovery_gate_current_suppression_scale),
            0.0,
        )
        self._coverage_recovery_gate_prepare_credit = max(
            float(coverage_recovery_gate_prepare_credit),
            0.0,
        )
        self._coverage_recovery_gate_fallback_penalty = max(
            float(coverage_recovery_gate_fallback_penalty),
            0.0,
        )
        self._coverage_recovery_guard_enabled = bool(coverage_recovery_guard_enabled)
        self._coverage_recovery_final_guard_enabled = bool(coverage_recovery_final_guard_enabled)
        self._coverage_recovery_final_guard_min_scale = max(
            0.0,
            min(float(coverage_recovery_final_guard_min_scale), 1.0),
        )
        self._coverage_recovery_final_guard_min_confidence = max(
            0.0,
            min(float(coverage_recovery_final_guard_min_confidence), 1.0),
        )
        self._coverage_recovery_target_memory_option_credit = max(
            float(coverage_recovery_target_memory_option_credit),
            0.0,
        )
        self._coverage_recovery_target_memory_option_penalty = max(
            float(coverage_recovery_target_memory_option_penalty),
            0.0,
        )
        self._service_completion_gate_enabled = bool(service_completion_gate_enabled)
        self._service_completion_gate_bias = max(float(service_completion_gate_bias), 0.0)
        self._service_completion_gate_remaining_nodes_threshold = max(
            int(service_completion_gate_remaining_nodes_threshold),
            1,
        )
        self._service_completion_gate_event_suppression_scale = max(
            float(service_completion_gate_event_suppression_scale),
            0.0,
        )
        self._service_completion_gate_prefetch_suppression_scale = max(
            float(service_completion_gate_prefetch_suppression_scale),
            0.0,
        )
        self._service_completion_gate_fallback_suppression_scale = max(
            float(service_completion_gate_fallback_suppression_scale),
            0.0,
        )
        self._service_completion_gate_policy_coef = max(
            float(service_completion_gate_policy_coef),
            0.0,
        )
        self._service_completion_gate_event_coef = max(
            float(service_completion_gate_event_coef),
            0.0,
        )
        self._service_completion_gate_clip = max(float(service_completion_gate_clip), 0.0)
        self._backhaul_aware_policy_enabled = bool(backhaul_aware_policy_enabled)
        self._backhaul_aware_service_fill_bias = max(float(backhaul_aware_service_fill_bias), 0.0)
        self._backhaul_aware_redundant_fill_penalty = max(
            float(backhaul_aware_redundant_fill_penalty),
            0.0,
        )
        self._backhaul_aware_no_signal_prefetch_penalty = max(
            float(backhaul_aware_no_signal_prefetch_penalty),
            0.0,
        )
        self._backhaul_aware_no_signal_prepare_penalty = max(
            float(backhaul_aware_no_signal_prepare_penalty),
            0.0,
        )
        self._backhaul_aware_steady_bias = max(float(backhaul_aware_steady_bias), 0.0)
        self._backhaul_aware_service_pressure_floor = max(
            0.0,
            min(float(backhaul_aware_service_pressure_floor), 1.0),
        )
        self._advantage_weighted_behavior_regularization_enabled = bool(
            advantage_weighted_behavior_regularization_enabled
        )
        self._advantage_weighted_behavior_coef = max(float(advantage_weighted_behavior_coef), 0.0)
        self._advantage_weighted_behavior_positive_coef = max(
            float(advantage_weighted_behavior_positive_coef),
            0.0,
        )
        self._advantage_weighted_behavior_negative_coef = max(
            float(advantage_weighted_behavior_negative_coef),
            0.0,
        )
        self._advantage_weighted_behavior_temperature = max(
            float(advantage_weighted_behavior_temperature),
            1e-6,
        )
        self._advantage_weighted_behavior_max_weight = max(
            float(advantage_weighted_behavior_max_weight),
            0.0,
        )
        self._advantage_weighted_behavior_positive_gate = max(
            float(advantage_weighted_behavior_positive_gate),
            0.0,
        )
        self._advantage_weighted_behavior_negative_gate = max(
            float(advantage_weighted_behavior_negative_gate),
            0.0,
        )
        self._advantage_weighted_behavior_mechanism_scale = max(
            float(advantage_weighted_behavior_mechanism_scale),
            0.0,
        )
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
        self._steady_rsu_bias_enabled = bool(steady_rsu_bias_enabled)
        self._steady_rsu_bias_strength = max(float(steady_rsu_bias_strength), 0.0)
        self._steady_rsu_confidence_floor = max(
            0.0,
            min(float(steady_rsu_confidence_floor), 1.0),
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
        self._idle_popularity_fallback_enabled = bool(idle_popularity_fallback_enabled)
        self._idle_popularity_fallback_only_vehicle_fallback = bool(
            idle_popularity_fallback_only_vehicle_fallback
        )
        self._idle_popularity_prefetch_threshold = max(int(idle_popularity_prefetch_threshold), 1)
        self._idle_popularity_no_rsu_local_fallback_enabled = bool(
            idle_popularity_no_rsu_local_fallback_enabled
        )
        self._idle_popularity_no_rsu_local_requires_low_context = bool(
            idle_popularity_no_rsu_local_requires_low_context
        )
        self._idle_popularity_no_rsu_service_continuity_enabled = bool(
            idle_popularity_no_rsu_service_continuity_enabled
        )
        self._idle_popularity_no_rsu_any_action_override_enabled = bool(
            idle_popularity_no_rsu_any_action_override_enabled
        )
        self._idle_popularity_adapter_counts: dict[str, int] = {}
        self._option_gate_enabled = bool(option_gate_enabled)
        self._option_gate_count = max(int(option_gate_count), 1)
        self._option_gate_loss_coef = max(float(option_gate_loss_coef), 0.0)
        self._option_gate_entropy_coef = max(float(option_gate_entropy_coef), 0.0)
        self._option_gate_prior_coef = max(float(option_gate_prior_coef), 0.0)
        self._option_gate_prior_warmup_updates = max(int(option_gate_prior_warmup_updates), 0)
        self._option_gate_prior_decay = max(float(option_gate_prior_decay), 0.0)
        self._option_gate_prior_logit_bias = max(float(option_gate_prior_logit_bias), 0.0)
        self._option_gate_log_prob_weight = max(float(option_gate_log_prob_weight), 0.0)
        self._option_gate_context_prior_enabled = bool(option_gate_context_prior_enabled)
        self._option_gate_deterministic_prior_margin = max(float(option_gate_deterministic_prior_margin), 0.0)
        self._option_gate_idle_prior_enabled = bool(option_gate_idle_prior_enabled)
        self._option_gate_idle_recovery_mechanism_prior_enabled = bool(
            option_gate_idle_recovery_mechanism_prior_enabled
        )
        self._option_gate_idle_recovery_min_context = max(
            0.0,
            min(float(option_gate_idle_recovery_min_context), 1.0),
        )
        self._option_gate_mechanism_preserve_enabled = bool(option_gate_mechanism_preserve_enabled)
        self._option_gate_prd_enabled = bool(option_gate_prd_enabled)
        self._option_gate_prd_coef = max(float(option_gate_prd_coef), 0.0)
        self._option_gate_prd_clip = max(float(option_gate_prd_clip), 0.0)
        self._option_gate_counterfactual_prd_enabled = bool(option_gate_counterfactual_prd_enabled)
        self._option_gate_counterfactual_coef = max(float(option_gate_counterfactual_coef), 0.0)
        self._option_gate_counterfactual_clip = max(float(option_gate_counterfactual_clip), 0.0)
        self._option_counterfactual_critic_enabled = bool(
            option_counterfactual_critic_enabled
        )
        self._option_counterfactual_value_coef = max(
            float(option_counterfactual_value_coef),
            0.0,
        )
        self._option_counterfactual_advantage_coef = max(
            float(option_counterfactual_advantage_coef),
            0.0,
        )
        self._option_counterfactual_advantage_clip = max(
            float(option_counterfactual_advantage_clip),
            0.0,
        )
        self._option_counterfactual_warmup_updates = max(
            int(option_counterfactual_warmup_updates),
            0,
        )
        self._option_counterfactual_tail_weight = max(
            float(option_counterfactual_tail_weight),
            0.0,
        )
        self._option_counterfactual_policy_improvement_enabled = bool(
            self._option_counterfactual_critic_enabled
            if option_counterfactual_policy_improvement_enabled is None
            else option_counterfactual_policy_improvement_enabled
        )
        self._option_counterfactual_policy_improvement_coef = max(
            float(option_counterfactual_policy_improvement_coef),
            0.0,
        )
        self._option_counterfactual_policy_improvement_clip = max(
            float(option_counterfactual_policy_improvement_clip),
            0.0,
        )
        self._option_counterfactual_policy_improvement_deterministic_only = bool(
            option_counterfactual_policy_improvement_deterministic_only
        )
        self._option_counterfactual_model_rollout_enabled = bool(
            option_counterfactual_model_rollout_enabled
        )
        self._option_counterfactual_model_rollout_horizon = max(
            int(option_counterfactual_model_rollout_horizon),
            1,
        )
        self._env_action_model_critic_enabled = bool(
            env_action_model_critic_enabled
        )
        self._env_action_model_critic_value_coef = max(
            float(env_action_model_critic_value_coef),
            0.0,
        )
        self._env_action_model_critic_advantage_coef = max(
            float(env_action_model_critic_advantage_coef),
            0.0,
        )
        self._env_action_model_critic_policy_improvement_coef = max(
            float(env_action_model_critic_policy_improvement_coef),
            0.0,
        )
        self._env_action_model_critic_advantage_clip = max(
            float(env_action_model_critic_advantage_clip),
            0.0,
        )
        self._env_action_model_critic_warmup_updates = max(
            int(env_action_model_critic_warmup_updates),
            0,
        )
        self._env_action_model_rollout_enabled = bool(
            env_action_model_rollout_enabled
        )
        self._env_action_model_rollout_horizon = max(
            int(env_action_model_rollout_horizon),
            1,
        )
        raw_rollout_horizons = (
            list(env_action_model_rollout_horizons)
            if env_action_model_rollout_horizons is not None
            else [self._env_action_model_rollout_horizon]
        )
        self._env_action_model_rollout_horizons = tuple(
            sorted(
                {
                    max(int(horizon), 1)
                    for horizon in raw_rollout_horizons
                }
            )
        )
        self._env_action_model_rollout_horizon = max(
            self._env_action_model_rollout_horizons
        )
        self._env_action_model_imagination_replay_enabled = bool(
            env_action_model_imagination_replay_enabled
        )
        raw_imagination_depths = (
            list(env_action_model_imagination_replay_depths)
            if env_action_model_imagination_replay_depths is not None
            else [2, 4, 8]
        )
        self._env_action_model_imagination_replay_depths = tuple(
            sorted(
                {
                    max(int(depth), 1)
                    for depth in raw_imagination_depths
                    if int(depth) < self._env_action_model_rollout_horizon
                }
            )
        )
        raw_imagination_horizons = (
            list(env_action_model_imagination_replay_horizons)
            if env_action_model_imagination_replay_horizons is not None
            else [1]
        )
        self._env_action_model_imagination_replay_horizons = tuple(
            sorted(
                {
                    max(int(horizon), 1)
                    for horizon in raw_imagination_horizons
                }
            )
        )
        self._env_action_model_imagination_replay_recovery_only = bool(
            env_action_model_imagination_replay_recovery_only
        )
        self._env_action_model_imagination_beam_search_enabled = bool(
            env_action_model_imagination_beam_search_enabled
        )
        imagination_branch_mode = str(
            env_action_model_imagination_replay_branch_mode
        ).strip().lower()
        if imagination_branch_mode not in {"dominant", "top_k"}:
            raise ValueError(
                "env_action_model_imagination_replay_branch_mode must be "
                "one of {'dominant', 'top_k'}"
            )
        self._env_action_model_imagination_replay_branch_mode = (
            imagination_branch_mode
        )
        self._env_action_model_imagination_replay_branch_top_k = max(
            int(env_action_model_imagination_replay_branch_top_k),
            1,
        )
        self._env_action_model_policy_improvement_enabled = bool(
            env_action_model_policy_improvement_enabled
        )
        self._env_action_model_policy_improvement_coef = max(
            float(env_action_model_policy_improvement_coef),
            0.0,
        )
        self._env_action_model_policy_improvement_temperature = max(
            float(env_action_model_policy_improvement_temperature),
            0.05,
        )
        self._env_action_model_policy_improvement_robust_horizons_enabled = bool(
            env_action_model_policy_improvement_robust_horizons_enabled
        )
        self._env_action_model_policy_improvement_horizon_risk_coef = max(
            float(env_action_model_policy_improvement_horizon_risk_coef),
            0.0,
        )
        horizon_aggregation_mode = str(
            env_action_model_policy_improvement_horizon_aggregation_mode
        ).strip().lower()
        if horizon_aggregation_mode not in {"mean_std", "lambda_downside"}:
            raise ValueError(
                "env_action_model_policy_improvement_horizon_aggregation_mode "
                "must be one of {'mean_std', 'lambda_downside'}"
            )
        self._env_action_model_policy_improvement_horizon_aggregation_mode = (
            horizon_aggregation_mode
        )
        self._env_action_model_policy_improvement_horizon_lambda = min(
            max(
                float(env_action_model_policy_improvement_horizon_lambda),
                0.0,
            ),
            0.999,
        )
        self._env_action_model_policy_improvement_adaptive_kl_enabled = bool(
            env_action_model_policy_improvement_adaptive_kl_enabled
        )
        self._env_action_model_policy_improvement_target_kl = max(
            float(env_action_model_policy_improvement_target_kl),
            1e-5,
        )
        self._env_action_model_policy_improvement_regret_adaptive_kl_enabled = bool(
            env_action_model_policy_improvement_regret_adaptive_kl_enabled
        )
        self._env_action_model_policy_improvement_max_target_kl = max(
            float(env_action_model_policy_improvement_max_target_kl),
            self._env_action_model_policy_improvement_target_kl,
        )
        self._env_action_model_policy_improvement_regret_priority_coef = max(
            float(env_action_model_policy_improvement_regret_priority_coef),
            0.0,
        )
        self._env_action_model_policy_improvement_tail_distillation_enabled = bool(
            env_action_model_policy_improvement_tail_distillation_enabled
        )
        self._env_action_model_policy_improvement_tail_quantile = min(
            max(
                float(env_action_model_policy_improvement_tail_quantile),
                0.0,
            ),
            1.0,
        )
        self._env_action_model_policy_improvement_tail_min_regret = min(
            max(
                float(env_action_model_policy_improvement_tail_min_regret),
                0.0,
            ),
            1.0,
        )
        self._env_action_model_policy_improvement_tail_epochs = max(
            int(env_action_model_policy_improvement_tail_epochs),
            0,
        )
        self._env_action_model_policy_improvement_tail_coef = max(
            float(env_action_model_policy_improvement_tail_coef),
            0.0,
        )
        self._env_action_model_policy_improvement_tail_max_policy_kl = max(
            float(
                env_action_model_policy_improvement_tail_max_policy_kl
            ),
            0.0,
        )
        self._env_action_model_policy_improvement_tail_recovery_only = bool(
            env_action_model_policy_improvement_tail_recovery_only
        )
        self._env_action_model_policy_improvement_tail_adapter_only = bool(
            env_action_model_policy_improvement_tail_adapter_only
        )
        self._env_action_model_policy_improvement_tail_beam_only = bool(
            env_action_model_policy_improvement_tail_beam_only
        )
        self._env_action_model_policy_improvement_tail_planning_adapter_only = bool(
            env_action_model_policy_improvement_tail_planning_adapter_only
        )
        self._env_action_model_policy_improvement_tail_residual_optimizer_enabled = bool(
            env_action_model_policy_improvement_tail_residual_optimizer_enabled
        )
        self._env_action_model_policy_improvement_tail_residual_learning_rate = max(
            float(env_action_model_policy_improvement_tail_residual_learning_rate),
            1e-6,
        )
        self._env_action_model_policy_improvement_tail_residual_backtrack_factor = min(
            max(
                float(
                    env_action_model_policy_improvement_tail_residual_backtrack_factor
                ),
                0.05,
            ),
            0.95,
        )
        self._env_action_model_policy_improvement_tail_residual_min_learning_rate = min(
            max(
                float(
                    env_action_model_policy_improvement_tail_residual_min_learning_rate
                ),
                1e-7,
            ),
            self._env_action_model_policy_improvement_tail_residual_learning_rate,
        )
        self._env_action_model_policy_improvement_tail_residual_max_backtracks = max(
            int(env_action_model_policy_improvement_tail_residual_max_backtracks),
            0,
        )
        self._env_action_model_policy_improvement_tail_logit_projection_enabled = bool(
            env_action_model_policy_improvement_tail_logit_projection_enabled
        )
        self._env_action_model_policy_improvement_tail_target_balance_enabled = bool(
            env_action_model_policy_improvement_tail_target_balance_enabled
        )
        self._env_action_model_policy_improvement_tail_target_balance_power = min(
            max(
                float(
                    env_action_model_policy_improvement_tail_target_balance_power
                ),
                0.0,
            ),
            1.0,
        )
        self._env_action_model_policy_improvement_tail_target_balance_max_weight = max(
            float(
                env_action_model_policy_improvement_tail_target_balance_max_weight
            ),
            1.0,
        )
        self._learned_transition_model_enabled = bool(learned_transition_model_enabled)
        self._learned_transition_model_planner_enabled = bool(
            learned_transition_model_planner_enabled
        )
        self._learned_transition_model_ensemble_size = max(
            int(learned_transition_model_ensemble_size), 2
        )
        self._learned_transition_model_hidden_dim = max(
            int(learned_transition_model_hidden_dim), 8
        )
        self._learned_transition_model_learning_rate = max(
            float(learned_transition_model_learning_rate), 1e-8
        )
        self._learned_transition_model_fit_epochs = max(
            int(learned_transition_model_fit_epochs), 1
        )
        self._learned_transition_model_max_samples = max(
            int(learned_transition_model_max_samples), 1
        )
        self._learned_transition_model_min_samples = max(
            int(learned_transition_model_min_samples), 2
        )
        self._learned_transition_model_discount = float(
            np.clip(float(learned_transition_model_discount), 0.0, 1.0)
        )
        self._learned_transition_model_risk_coef = max(
            float(learned_transition_model_risk_coef), 0.0
        )
        self._learned_transition_model_exploration_coef = max(
            float(learned_transition_model_exploration_coef), 0.0
        )
        self._learned_transition_model_policy_coef = max(
            float(learned_transition_model_policy_coef), 0.0
        )
        self._learned_transition_model_policy_prior_coef = max(
            float(learned_transition_model_policy_prior_coef), 0.0
        )
        self._learned_transition_model_min_margin = max(
            float(learned_transition_model_min_margin), 0.0
        )
        self._learned_transition_model_warmup_updates = max(
            int(learned_transition_model_warmup_updates), 0
        )
        self._env_action_model_online_planner_enabled = bool(
            env_action_model_online_planner_enabled
        )
        self._env_action_model_online_planner_coef = max(
            float(env_action_model_online_planner_coef),
            0.0,
        )
        self._env_action_model_online_planner_mechanism_coef = max(
            float(env_action_model_online_planner_mechanism_coef),
            0.0,
        )
        self._env_action_model_online_planner_policy_prior_coef = max(
            float(env_action_model_online_planner_policy_prior_coef),
            0.0,
        )
        self._env_action_model_online_planner_min_margin = max(
            float(env_action_model_online_planner_min_margin),
            0.0,
        )
        self._env_action_model_online_planner_prefer_beam_targets = bool(
            env_action_model_online_planner_prefer_beam_targets
        )
        self._env_action_model_resource_constraint_enabled = bool(
            env_action_model_resource_constraint_enabled
        )
        self._env_action_model_resource_cost_coef = max(
            float(env_action_model_resource_cost_coef),
            0.0,
        )
        self._env_action_model_resource_cost_scale = max(
            float(env_action_model_resource_cost_scale),
            1e-6,
        )
        self._env_action_model_adaptive_horizon_enabled = bool(
            env_action_model_adaptive_horizon_enabled
        )
        self._env_action_model_adaptive_horizon_temperature = max(
            float(env_action_model_adaptive_horizon_temperature),
            0.05,
        )
        self._env_action_model_beam_search_enabled = bool(
            env_action_model_beam_search_enabled
        )
        self._env_action_model_beam_search_horizon = max(
            int(env_action_model_beam_search_horizon),
            1,
        )
        self._env_action_model_beam_search_width = max(
            int(env_action_model_beam_search_width),
            1,
        )
        self._env_action_model_beam_search_context_only = bool(
            env_action_model_beam_search_context_only
        )
        self._env_action_model_beam_search_min_eta = max(
            int(env_action_model_beam_search_min_eta),
            0,
        )
        self._env_action_model_beam_search_max_eta = max(
            int(env_action_model_beam_search_max_eta),
            self._env_action_model_beam_search_min_eta,
        )
        self._env_action_model_policy_improvement_prefer_beam_targets = bool(
            env_action_model_policy_improvement_prefer_beam_targets
        )
        self._counterfactual_teacher_prd_enabled = bool(counterfactual_teacher_prd_enabled)
        self._counterfactual_teacher_event_coef = max(float(counterfactual_teacher_event_coef), 0.0)
        self._counterfactual_teacher_option_coef = max(float(counterfactual_teacher_option_coef), 0.0)
        self._counterfactual_teacher_clip = max(float(counterfactual_teacher_clip), 0.0)
        self._counterfactual_teacher_mechanism_bonus = max(float(counterfactual_teacher_mechanism_bonus), 0.0)
        self._counterfactual_teacher_missed_prepare_penalty = max(
            float(counterfactual_teacher_missed_prepare_penalty),
            0.0,
        )
        self._counterfactual_teacher_local_bonus = max(float(counterfactual_teacher_local_bonus), 0.0)
        self._counterfactual_teacher_current_rsu_penalty = max(
            float(counterfactual_teacher_current_rsu_penalty),
            0.0,
        )
        self._counterfactual_teacher_invalid_mechanism_penalty = max(
            float(counterfactual_teacher_invalid_mechanism_penalty),
            0.0,
        )
        self._service_continuity_teacher_enabled = bool(service_continuity_teacher_enabled)
        self._service_continuity_current_bonus = max(float(service_continuity_current_bonus), 0.0)
        self._service_continuity_prepare_bonus = max(float(service_continuity_prepare_bonus), 0.0)
        self._service_continuity_local_penalty = max(float(service_continuity_local_penalty), 0.0)
        self._service_continuity_min_prepare_context = max(
            0.0,
            min(float(service_continuity_min_prepare_context), 1.0),
        )
        self._tail_risk_prd_enabled = bool(tail_risk_prd_enabled)
        self._tail_risk_policy_coef = max(float(tail_risk_policy_coef), 0.0)
        self._tail_risk_event_coef = max(float(tail_risk_event_coef), 0.0)
        self._tail_risk_option_coef = max(float(tail_risk_option_coef), 0.0)
        self._tail_risk_clip = max(float(tail_risk_clip), 0.0)
        self._tail_risk_quantile = max(0.0, min(float(tail_risk_quantile), 1.0))
        self._tail_risk_reward_shortfall_coef = max(float(tail_risk_reward_shortfall_coef), 0.0)
        self._tail_risk_service_coef = max(float(tail_risk_service_coef), 0.0)
        self._tail_risk_continuity_coef = max(float(tail_risk_continuity_coef), 0.0)
        self._tail_risk_handoff_failure_coef = max(float(tail_risk_handoff_failure_coef), 0.0)
        self._tail_risk_failed_mechanism_coef = max(float(tail_risk_failed_mechanism_coef), 0.0)
        self._tail_risk_redundant_mechanism_coef = max(float(tail_risk_redundant_mechanism_coef), 0.0)
        self._tail_risk_success_credit = max(float(tail_risk_success_credit), 0.0)
        self._opportunity_prd_enabled = bool(opportunity_prd_enabled)
        self._opportunity_policy_coef = max(float(opportunity_policy_coef), 0.0)
        self._opportunity_event_coef = max(float(opportunity_event_coef), 0.0)
        self._opportunity_option_coef = max(float(opportunity_option_coef), 0.0)
        self._opportunity_clip = max(float(opportunity_clip), 0.0)
        self._opportunity_reward_quantile = max(0.0, min(float(opportunity_reward_quantile), 1.0))
        self._opportunity_reward_surplus_coef = max(float(opportunity_reward_surplus_coef), 0.0)
        self._opportunity_service_success_coef = max(float(opportunity_service_success_coef), 0.0)
        self._opportunity_cache_hit_coef = max(float(opportunity_cache_hit_coef), 0.0)
        self._opportunity_continuity_coef = max(float(opportunity_continuity_coef), 0.0)
        self._opportunity_current_rsu_efficiency_coef = max(
            float(opportunity_current_rsu_efficiency_coef),
            0.0,
        )
        self._opportunity_local_fallback_coef = max(float(opportunity_local_fallback_coef), 0.0)
        self._opportunity_backhaul_penalty_coef = max(float(opportunity_backhaul_penalty_coef), 0.0)
        self._opportunity_delay_penalty_coef = max(float(opportunity_delay_penalty_coef), 0.0)
        self._opportunity_failed_service_penalty_coef = max(
            float(opportunity_failed_service_penalty_coef),
            0.0,
        )
        self._opportunity_mechanism_success_bonus = max(float(opportunity_mechanism_success_bonus), 0.0)
        self._handoff_risk_prd_enabled = bool(handoff_risk_prd_enabled)
        self._handoff_risk_event_coef = max(float(handoff_risk_event_coef), 0.0)
        self._handoff_risk_option_coef = max(float(handoff_risk_option_coef), 0.0)
        self._handoff_risk_clip = max(float(handoff_risk_clip), 0.0)
        self._handoff_risk_failure_penalty = max(float(handoff_risk_failure_penalty), 0.0)
        self._handoff_risk_ready_bonus = max(float(handoff_risk_ready_bonus), 0.0)
        self._handoff_risk_prepare_bonus = max(float(handoff_risk_prepare_bonus), 0.0)
        self._handoff_risk_unprepared_penalty = max(float(handoff_risk_unprepared_penalty), 0.0)
        self._handoff_risk_confidence_threshold = max(
            min(float(handoff_risk_confidence_threshold), 1.0),
            0.0,
        )
        self._handoff_risk_cost_dual_enabled = bool(handoff_risk_cost_dual_enabled)
        self._handoff_risk_cost_dual_lr = max(float(handoff_risk_cost_dual_lr), 0.0)
        self._handoff_risk_cost_target = max(float(handoff_risk_cost_target), 0.0)
        self._handoff_risk_cost_dual_max = max(float(handoff_risk_cost_dual_max), 0.0)
        self._handoff_risk_cost_dual = min(
            max(float(handoff_risk_cost_dual_initial), 0.0),
            self._handoff_risk_cost_dual_max,
        )
        self._idle_execution_prd_enabled = bool(idle_execution_prd_enabled)
        self._idle_execution_policy_coef = max(float(idle_execution_policy_coef), 0.0)
        self._idle_execution_option_coef = max(float(idle_execution_option_coef), 0.0)
        self._idle_execution_clip = max(float(idle_execution_clip), 0.0)
        self._idle_execution_current_rsu_delay_coef = max(
            float(idle_execution_current_rsu_delay_coef),
            0.0,
        )
        self._idle_execution_local_bonus = max(float(idle_execution_local_bonus), 0.0)
        self._idle_execution_mechanism_penalty = max(float(idle_execution_mechanism_penalty), 0.0)
        self._idle_execution_timing_threshold = max(
            0.0,
            min(float(idle_execution_timing_threshold), 1.0),
        )
        self._idle_execution_mechanism_preserve_bonus = max(
            float(idle_execution_mechanism_preserve_bonus),
            0.0,
        )
        self._net_utility_prd_enabled = bool(net_utility_prd_enabled)
        self._net_utility_backhaul_coef = max(float(net_utility_backhaul_coef), 0.0)
        self._net_utility_migration_coef = max(float(net_utility_migration_coef), 0.0)
        self._net_utility_expired_prefetch_coef = max(float(net_utility_expired_prefetch_coef), 0.0)
        self._net_utility_idle_prefetch_penalty = max(float(net_utility_idle_prefetch_penalty), 0.0)
        self._net_utility_failed_mechanism_penalty = max(float(net_utility_failed_mechanism_penalty), 0.0)
        self._net_utility_failed_mechanism_backhaul_coef = max(
            float(net_utility_failed_mechanism_backhaul_coef),
            0.0,
        )
        self._net_utility_mechanism_window_failed_penalty_scale = max(
            float(net_utility_mechanism_window_failed_penalty_scale),
            0.0,
        )
        self._net_utility_success_bonus = max(float(net_utility_success_bonus), 0.0)
        self._net_utility_backhaul_normalizer = max(float(net_utility_backhaul_normalizer), 1.0)
        self._net_utility_cost_dual_enabled = bool(net_utility_cost_dual_enabled)
        self._net_utility_cost_dual_lr = max(float(net_utility_cost_dual_lr), 0.0)
        self._net_utility_cost_target = max(float(net_utility_cost_target), 0.0)
        self._net_utility_cost_dual_max = max(float(net_utility_cost_dual_max), 0.0)
        self._net_utility_cost_dual = min(
            max(float(net_utility_cost_dual_initial), 0.0),
            self._net_utility_cost_dual_max,
        )
        self._net_utility_option_termination_enabled = bool(net_utility_option_termination_enabled)
        self._net_utility_option_termination_conservative_enabled = bool(
            net_utility_option_termination_conservative_enabled
        )
        self._net_utility_option_termination_max_timing_support = max(
            min(float(net_utility_option_termination_max_timing_support), 1.0),
            0.0,
        )
        self._dag_aware_option_termination_enabled = bool(dag_aware_option_termination_enabled)
        self._dag_aware_option_min_critical_path = max(int(dag_aware_option_min_critical_path), 1)
        self._dag_aware_option_short_workflow_max_nodes = max(int(dag_aware_option_short_workflow_max_nodes), 1)
        self._dag_aware_option_branching_successors = max(int(dag_aware_option_branching_successors), 0)
        self._dag_aware_idle_prefetch_confidence_floor = max(
            min(float(dag_aware_idle_prefetch_confidence_floor), 1.0),
            0.0,
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

        self._learned_transition_model = (
            UncertaintyTransitionEnsemble(
                observation_dim=9,
                action_count=5,
                ensemble_size=self._learned_transition_model_ensemble_size,
                hidden_dim=self._learned_transition_model_hidden_dim,
                learning_rate=self._learned_transition_model_learning_rate,
                fit_epochs=self._learned_transition_model_fit_epochs,
                max_samples=self._learned_transition_model_max_samples,
                min_samples=self._learned_transition_model_min_samples,
                discount=self._learned_transition_model_discount,
                random_seed=random_seed + 7919,
                device=device,
            )
            if self._learned_transition_model_enabled
            else None
        )

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
            option_gate_enabled=self._option_gate_enabled,
            option_gate_count=self._option_gate_count,
            option_counterfactual_critic_enabled=self._option_counterfactual_critic_enabled,
            env_action_model_critic_enabled=self._env_action_model_critic_enabled,
            digital_twin_handoff_fusion_enabled=self._digital_twin_handoff_fusion_enabled,
            digital_twin_handoff_slow_scale=self._digital_twin_handoff_slow_scale,
            digital_twin_handoff_fast_scale=self._digital_twin_handoff_fast_scale,
            digital_twin_handoff_event_scale=self._digital_twin_handoff_event_scale,
            digital_twin_handoff_critic_scale=self._digital_twin_handoff_critic_scale,
            digital_twin_planning_residual_enabled=(
                self._digital_twin_planning_residual_enabled
            ),
            digital_twin_planning_residual_scale=(
                self._digital_twin_planning_residual_scale
            ),
            outcome_memory_fusion_enabled=self._outcome_memory_fusion_enabled,
            outcome_memory_actor_scale=self._outcome_memory_actor_scale,
            outcome_memory_critic_scale=self._outcome_memory_critic_scale,
            outcome_recovery_residual_enabled=(
                self._outcome_recovery_residual_enabled
            ),
            outcome_recovery_residual_scale=(
                self._outcome_recovery_residual_scale
            ),
            outcome_context_residual_enabled=(
                self._outcome_context_residual_enabled
            ),
            hidden_dims=self._hidden_dims,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)

    def _apply_env_action_model_critic_improvement(
        self,
        *,
        policy_output: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not self._env_action_model_critic_enabled
            or self._env_action_model_critic_policy_improvement_coef <= 0.0
            or self._update_count < self._env_action_model_critic_warmup_updates
            or "env_action_q_values" not in policy_output
        ):
            return policy_output, {"enabled": False, "applied": False}

        q_values = policy_output["env_action_q_values"].detach()
        if self._use_hierarchy:
            actor_scores = self._hierarchical_env_action_scores(policy_output)
        else:
            actor_scores = policy_output["flat_logits"]
        valid_mask = torch.ones_like(q_values, dtype=torch.bool)
        if action_mask and len(action_mask) == int(q_values.shape[-1]):
            valid_mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self._device,
            )
        if not bool(valid_mask.any().item()):
            return policy_output, {"enabled": True, "applied": False}

        masked_scores = actor_scores.masked_fill(~valid_mask, -1.0e9)
        actor_probs = torch.softmax(masked_scores, dim=-1)
        q_baseline = torch.sum(actor_probs * q_values)
        q_advantage = (q_values - q_baseline).masked_fill(~valid_mask, 0.0)
        valid_advantage = q_advantage[valid_mask]
        q_scale = torch.sqrt(torch.mean(valid_advantage.square())).clamp_min(1e-6)
        normalized_advantage = (q_advantage / q_scale).masked_fill(~valid_mask, 0.0)
        if self._env_action_model_critic_advantage_clip > 0.0:
            normalized_advantage = torch.clamp(
                normalized_advantage,
                -self._env_action_model_critic_advantage_clip,
                self._env_action_model_critic_advantage_clip,
            )

        adjusted_output = dict(policy_output)
        critic_bias = (
            self._env_action_model_critic_policy_improvement_coef
            * normalized_advantage
        )
        existing_bias = policy_output.get("env_action_logits_bias")
        if isinstance(existing_bias, torch.Tensor) and existing_bias.shape == critic_bias.shape:
            critic_bias = critic_bias + existing_bias
        adjusted_output["env_action_logits_bias"] = critic_bias
        return adjusted_output, {
            "enabled": True,
            "applied": bool(torch.any(torch.abs(normalized_advantage) > 1e-8).item()),
            "normalized_advantage": [
                round(float(item), 6)
                for item in normalized_advantage.tolist()
            ],
            "policy_improvement_coef": round(
                self._env_action_model_critic_policy_improvement_coef,
                6,
            ),
        }

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = self._extract_semantic_state(info)
        action_mask = self._extract_action_mask(info)
        run_metadata = dict((info or {}).get("run_metadata", {}) or {})
        policy_evaluation_mode = str(
            run_metadata.get("policy_evaluation_mode", "safety_projected")
        ).strip().lower()
        if policy_evaluation_mode not in {"raw_policy", "safety_projected"}:
            policy_evaluation_mode = "safety_projected"
        run_metadata["policy_evaluation_mode"] = policy_evaluation_mode
        raw_policy_evaluation = policy_evaluation_mode == "raw_policy"
        deterministic = bool(self._deterministic_action or (info or {}).get("deterministic_policy", False))
        with torch.no_grad():
            policy_output = self._forward_policy(semantic_state, run_metadata=run_metadata)
            env_action_model_critic_info: dict[str, Any] = {
                "enabled": False,
                "applied": False,
            }
            if deterministic:
                (
                    policy_output,
                    env_action_model_critic_info,
                ) = self._apply_env_action_model_critic_improvement(
                    policy_output=policy_output,
                    action_mask=action_mask,
                )
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
            if (not raw_policy_evaluation) and self._should_hard_apply_continuity_guard(
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
            smoothing_info = (
                self._apply_deterministic_temporal_smoothing(
                    semantic_state=semantic_state,
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    deterministic=deterministic,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "reason": "raw_policy_evaluation"}
            )
            cache_warm_guard_info = (
                self._apply_cache_warm_start_guard_to_actions(
                    semantic_state=semantic_state,
                    selected_actions=selected_actions,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "guarded": False, "reason": "raw_policy_evaluation"}
            )
            if cache_warm_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            prefetch_admission_guard_info = (
                self._apply_predictive_prefetch_admission_guard_to_actions(
                    semantic_state=semantic_state,
                    selected_actions=selected_actions,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "guarded": False, "reason": "raw_policy_evaluation"}
            )
            if prefetch_admission_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            backhaul_guard_info = (
                self._apply_backhaul_guard_to_actions(
                    semantic_state=semantic_state,
                    selected_actions=selected_actions,
                    cache_warm_guard_info=cache_warm_guard_info,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "guarded": False, "reason": "raw_policy_evaluation"}
            )
            if backhaul_guard_info.get("guarded", False):
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            coverage_recovery_guard_info = (
                self._apply_coverage_recovery_guard_to_actions(
                    semantic_state=semantic_state,
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "guarded": False, "reason": "raw_policy_evaluation"}
            )
            if coverage_recovery_guard_info.get("guarded", False):
                selected_actions = self._head_targets_for_env_action(
                    int(coverage_recovery_guard_info["guarded_action"])
                )
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
            idle_popularity_fallback_info = (
                self._maybe_apply_idle_popularity_fallback(
                    semantic_state=semantic_state,
                    action_mask=action_mask,
                    original_env_action=env_action,
                    deterministic=deterministic,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "applied": False, "reason": "raw_policy_evaluation"}
            )
            if idle_popularity_fallback_info.get("applied", False):
                env_action = int(idle_popularity_fallback_info["fallback_action"])
                selected_actions = self._head_targets_for_env_action(env_action)
                aggregation_reason = "idle_popularity_fallback"
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            option_gate_info = (
                self._maybe_apply_option_gate(
                    semantic_state=semantic_state,
                    action_mask=action_mask,
                    policy_output=policy_output,
                    base_env_action=env_action,
                    deterministic=deterministic,
                    run_metadata=run_metadata,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "applied": False, "reason": "raw_policy_evaluation"}
            )
            option_gate_info.pop("_option_log_prob_tensor", None)
            option_entropy_tensor = option_gate_info.pop("_option_entropy_tensor", None)
            if option_gate_info.get("applied", False):
                env_action = int(option_gate_info["option_env_action"])
                selected_actions = self._head_targets_for_env_action(env_action)
                aggregation_reason = f"option_gate_{option_gate_info.get('option_label', 'unknown')}"
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            coverage_recovery_final_guard_info = (
                self._apply_coverage_recovery_final_guard_to_env_action(
                    semantic_state=semantic_state,
                    policy_output=policy_output,
                    env_action=env_action,
                    action_mask=action_mask,
                    option_gate_info=option_gate_info,
                )
                if not raw_policy_evaluation
                else {"enabled": False, "guarded": False, "reason": "raw_policy_evaluation"}
            )
            if coverage_recovery_final_guard_info.get("guarded", False):
                env_action = int(coverage_recovery_final_guard_info["guarded_action"])
                selected_actions = self._head_targets_for_env_action(env_action)
                aggregation_reason = "coverage_recovery_final_guard"
                option_gate_info["post_option_guarded"] = True
                option_gate_info["post_option_guarded_action"] = int(env_action)
                option_gate_info["post_option_guard_reason"] = str(
                    coverage_recovery_final_guard_info.get("reason", "unknown")
                )
                head_log_probs, head_entropies, action_prob_payload = self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            if guard_info:
                guard_info["guarded_action"] = int(env_action)
            guard_action_delta = bool(
                int(projected_env_action) != int(env_action)
                or dict(projected_head_actions) != dict(selected_actions)
            )
            env_action_log_prob, env_action_entropy, env_action_probs = self._env_action_distribution_statistics(
                policy_output=policy_output,
                env_action=env_action,
                action_mask=action_mask,
            )
            head_credit_weights = self._build_head_credit_weights(aggregation_reason=aggregation_reason)
            log_prob, entropy = self._combine_head_statistics(
                head_log_probs=head_log_probs,
                head_entropies=head_entropies,
                head_credit_weights=head_credit_weights,
            )
            if option_entropy_tensor is not None:
                entropy = 0.5 * (entropy + option_entropy_tensor)
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
            "policy_evaluation_mode": policy_evaluation_mode,
            "raw_policy_evaluation": raw_policy_evaluation,
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
            "env_action_log_prob": round(float(env_action_log_prob.item()), 6),
            "env_action_entropy": round(float(env_action_entropy.item()), 6),
            "env_action_probs": env_action_probs,
            "env_action_model_critic": env_action_model_critic_info,
            "env_action_model_rollout_enabled": bool(
                self._env_action_model_rollout_enabled
            ),
            "env_action_model_rollout_horizon": int(
                self._env_action_model_rollout_horizon
            ),
            "env_action_model_rollout_horizons": list(
                self._env_action_model_rollout_horizons
            ),
            "env_action_model_imagination_replay_enabled": bool(
                self._env_action_model_imagination_replay_enabled
            ),
            "env_action_model_imagination_replay_depths": list(
                self._env_action_model_imagination_replay_depths
            ),
            "env_action_model_imagination_replay_horizons": list(
                self._env_action_model_imagination_replay_horizons
            ),
            "env_action_model_imagination_replay_recovery_only": (
                self._env_action_model_imagination_replay_recovery_only
            ),
            "env_action_model_imagination_beam_search_enabled": (
                self._env_action_model_imagination_beam_search_enabled
            ),
            "env_action_model_imagination_replay_branch_mode": (
                self._env_action_model_imagination_replay_branch_mode
            ),
            "env_action_model_imagination_replay_branch_top_k": (
                self._env_action_model_imagination_replay_branch_top_k
            ),
            "env_action_model_online_planner_enabled": (
                self._env_action_model_online_planner_enabled
            ),
            "env_action_model_online_planner_coef": (
                self._env_action_model_online_planner_coef
            ),
            "env_action_model_online_planner_mechanism_coef": (
                self._env_action_model_online_planner_mechanism_coef
            ),
            "env_action_model_online_planner_policy_prior_coef": (
                self._env_action_model_online_planner_policy_prior_coef
            ),
            "env_action_model_online_planner_min_margin": (
                self._env_action_model_online_planner_min_margin
            ),
            "env_action_model_resource_constraint_enabled": (
                self._env_action_model_resource_constraint_enabled
            ),
            "env_action_model_resource_cost_coef": (
                self._env_action_model_resource_cost_coef
            ),
            "env_action_model_resource_cost_scale": (
                self._env_action_model_resource_cost_scale
            ),
            "env_action_model_adaptive_horizon_enabled": (
                self._env_action_model_adaptive_horizon_enabled
            ),
            "env_action_model_adaptive_horizon_temperature": (
                self._env_action_model_adaptive_horizon_temperature
            ),
            "env_action_model_beam_search_enabled": (
                self._env_action_model_beam_search_enabled
            ),
            "env_action_model_beam_search_horizon": (
                self._env_action_model_beam_search_horizon
            ),
            "env_action_model_beam_search_width": (
                self._env_action_model_beam_search_width
            ),
            "env_action_model_beam_search_context_only": (
                self._env_action_model_beam_search_context_only
            ),
            "env_action_model_beam_search_min_eta": (
                self._env_action_model_beam_search_min_eta
            ),
            "env_action_model_beam_search_max_eta": (
                self._env_action_model_beam_search_max_eta
            ),
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
            "outcome_recovery_residual_gate": round(
                float(
                    policy_output.get(
                        "outcome_recovery_residual_gate",
                        torch.tensor(0.0, device=self._device),
                    ).item()
                ),
                6,
            ),
            "digital_twin_planning_residual_gate": round(
                float(
                    policy_output.get(
                        "digital_twin_planning_residual_gate",
                        torch.tensor(0.0, device=self._device),
                    ).item()
                ),
                6,
            ),
            "outcome_recovery_residual_bias": [
                round(float(item), 6)
                for item in (
                    policy_output.get("outcome_recovery_residual_bias")
                    if policy_output.get("outcome_recovery_residual_bias")
                    is not None
                    else torch.zeros(5, device=self._device)
                ).detach().tolist()
            ],
            "outcome_context_residual_bias": [
                round(float(item), 6)
                for item in (
                    policy_output.get("outcome_context_residual_bias")
                    if policy_output.get("outcome_context_residual_bias")
                    is not None
                    else torch.zeros(5, device=self._device)
                ).detach().tolist()
            ],
            "digital_twin_planning_residual_bias": [
                round(float(item), 6)
                for item in (
                    policy_output.get("digital_twin_planning_residual_bias")
                    if policy_output.get("digital_twin_planning_residual_bias")
                    is not None
                    else torch.zeros(5, device=self._device)
                ).detach().tolist()
            ],
            "event_logit_temperature": round(active_event_logit_temperature, 6),
            "event_sharpening_info": dict(policy_output.get("event_sharpening_info", {})),
            "digital_twin_policy_prior": dict(policy_output.get("digital_twin_policy_prior_info", {})),
            "opportunity_constrained_policy": dict(policy_output.get("opportunity_constrained_policy_info", {})),
            "backhaul_aware_policy": dict(policy_output.get("backhaul_aware_policy_info", {})),
            "net_advantage_prepare_gate": dict(policy_output.get("net_advantage_prepare_gate_info", {})),
            "service_completion_gate": dict(policy_output.get("service_completion_gate_info", {})),
            "coverage_recovery_guard": dict(coverage_recovery_guard_info),
            "coverage_recovery_final_guard": dict(coverage_recovery_final_guard_info),
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
            "idle_popularity_fallback": idle_popularity_fallback_info,
            "option_gate": option_gate_info,
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
        run_metadata = dict((info or {}).get("run_metadata", {}) or {})
        with torch.no_grad():
            policy_output = self._forward_policy(semantic_state, run_metadata=run_metadata)
        return float(policy_output["value"].item())

    def _fit_learned_transition_model(
        self,
        rollout: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._learned_transition_model is None:
            return {"enabled": False, "ready": False, "skipped": True}
        return self._learned_transition_model.fit(rollout)

    def predict_learned_transition_targets(
        self,
        *,
        observation: Any,
        action_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Build action-conditioned UCB-train/LCB-eval targets from the model.

        Only a frozen model learned from prior environment rollout rows is used
        during action selection. Training adds calibrated disagreement to
        acquire coverage for uncertain actions; deterministic evaluation keeps
        the lower-confidence bound so extrapolation remains conservative.
        """
        model = self._learned_transition_model
        if (
            model is None
            or not self._learned_transition_model_planner_enabled
            or self._update_count < self._learned_transition_model_warmup_updates
            or not model.ready
        ):
            return {}
        action_mask = self._extract_action_mask(action_info)
        valid_actions = [
            action_id
            for action_id in range(5)
            if self._is_env_action_valid(action_id, action_mask)
        ]
        if len(valid_actions) < 2:
            return {}
        prediction = model.predict(observation, valid_actions)
        if not bool(prediction.get("ready", False)):
            return {}
        exploration_coef = (
            self._learned_transition_model_exploration_coef
            if not self._deterministic_action
            else 0.0
        )
        robust_targets = {
            str(action_id): float(
                mean
                - self._learned_transition_model_risk_coef * std
                + exploration_coef * std
            )
            for action_id, mean, std in zip(
                valid_actions,
                prediction.get("td_target_mean", []),
                prediction.get("td_target_std", []),
            )
        }
        return {
            "source": "learned_transition_ensemble",
            "protocol": "ucc_mappo_ucb_train_lcb_eval_v1",
            "action_td_targets": robust_targets,
            "action_td_targets_by_horizon": {"1": robust_targets},
            "model_query_count": len(valid_actions),
            "model_transition_count": len(valid_actions),
            "model_sample_count": int(model.sample_count),
            "model_update_count": int(model.update_count),
            "uncertainty_scale": float(model.uncertainty_scale),
            "risk_coef": float(self._learned_transition_model_risk_coef),
            "exploration_coef": float(exploration_coef),
            "exploration_mode": "lcb_eval" if self._deterministic_action else "ucb_train",
            "action_td_target_mean": {
                str(action_id): round(float(value), 6)
                for action_id, value in zip(valid_actions, prediction.get("td_target_mean", []))
            },
            "action_td_target_std": {
                str(action_id): round(float(value), 6)
                for action_id, value in zip(valid_actions, prediction.get("td_target_std", []))
            },
            "action_reward_mean": {
                str(action_id): round(float(value), 6)
                for action_id, value in zip(valid_actions, prediction.get("reward_mean", []))
            },
            "action_reward_std": {
                str(action_id): round(float(value), 6)
                for action_id, value in zip(valid_actions, prediction.get("reward_std", []))
            },
        }

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        if not rollout:
            return {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "policy_update_skipped": True,
                "reason": "empty_rollout",
                "update_count": self._update_count,
            }

        model_training_rollout = list(rollout)
        counterfactual_sample_count = 0
        for row in rollout:
            rollout_info = dict(
                row.get("action_info", {}).get("env_action_model_rollout", {})
                or {}
            )
            counterfactual_samples = rollout_info.get(
                "counterfactual_transition_samples", []
            )
            if isinstance(counterfactual_samples, list):
                model_training_rollout.extend(counterfactual_samples)
                counterfactual_sample_count += len(counterfactual_samples)
        learned_transition_model_stats = self._fit_learned_transition_model(
            model_training_rollout
        )
        learned_transition_model_stats["counterfactual_sample_count"] = (
            counterfactual_sample_count
        )
        imitation_rollout_stats = self._annotate_heuristic_imitation_targets(rollout)
        semantic_states = [self._extract_semantic_state(row.get("decision_info")) for row in rollout]
        run_metadata_by_row = [
            dict(row.get("decision_info", {}).get("run_metadata", {}) or {})
            for row in rollout
        ]
        action_masks = [self._extract_action_mask(row.get("decision_info")) for row in rollout]
        actions = [int(row["action"]) for row in rollout]
        returns = np.asarray([float(row.get("return", row.get("reward", 0.0))) for row in rollout], dtype=np.float32)
        advantages = np.asarray([float(row.get("advantage", 0.0)) for row in rollout], dtype=np.float32)
        values = np.asarray([float(row.get("value", 0.0)) for row in rollout], dtype=np.float32)
        old_log_probs = np.asarray([float(row.get("log_prob", 0.0)) for row in rollout], dtype=np.float32)
        old_env_action_log_prob_values: list[float] = []
        old_env_action_log_prob_missing_count = 0
        for row in rollout:
            action_info = dict(row.get("action_info", {}))
            action_projection = dict(action_info.get("action_projection", {}))
            if "env_action_log_prob" in action_info:
                old_env_action_log_prob_values.append(float(action_info.get("env_action_log_prob", 0.0) or 0.0))
            elif "masked_env_action_log_prob" in action_projection:
                old_env_action_log_prob_values.append(
                    float(action_projection.get("masked_env_action_log_prob", 0.0) or 0.0)
                )
                old_env_action_log_prob_missing_count += 1
            else:
                old_env_action_log_prob_values.append(float(row.get("log_prob", 0.0) or 0.0))
                old_env_action_log_prob_missing_count += 1
        old_env_action_log_probs = np.asarray(old_env_action_log_prob_values, dtype=np.float32)
        retention_active = self._mechanism_retention_active_for_update()
        effective_mechanism_aux_coef = self._effective_mechanism_aux_coef()
        effective_mechanism_entropy_coef = self._effective_mechanism_entropy_coef()
        mechanism_guidance_annotations = [
            self._build_mechanism_guidance_annotation(semantic_state, row)
            for semantic_state, row in zip(semantic_states, rollout)
        ]
        digital_twin_policy_prior_annotations = [
            self._build_digital_twin_policy_prior_annotation(
                semantic_state,
                run_metadata=run_metadata,
            )
            for semantic_state, run_metadata in zip(semantic_states, run_metadata_by_row)
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
        net_utility_cost_values = np.asarray(
            [self._net_utility_cost_signal(row) for row in rollout],
            dtype=np.float32,
        )
        net_utility_dual_before = float(self._net_utility_cost_dual)
        self._update_net_utility_cost_dual(net_utility_cost_values)
        net_utility_dual_after = float(self._net_utility_cost_dual)
        net_utility_adjustment_values = np.zeros(len(rollout), dtype=np.float32)
        if self._net_utility_prd_enabled:
            net_utility_adjustment_values = np.asarray(
                [self._net_utility_prd_adjustment(row) for row in rollout],
                dtype=np.float32,
            )
        event_prd_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if self._event_prd_advantage_enabled and self._event_prd_advantage_coef > 0.0:
            event_prd_credit_values = np.asarray(
                [self._event_partial_reward_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._event_prd_advantage_clip > 0.0:
                event_prd_credit_values = np.clip(
                    event_prd_credit_values,
                    -self._event_prd_advantage_clip,
                    self._event_prd_advantage_clip,
                )
            normalized_event_advantages = (
                normalized_event_advantages
                + self._event_prd_advantage_coef * event_prd_credit_values
            )
        handoff_risk_cost_values = np.asarray(
            [self._handoff_risk_cost_signal(row) for row in rollout],
            dtype=np.float32,
        )
        handoff_risk_dual_before = float(self._handoff_risk_cost_dual)
        self._update_handoff_risk_cost_dual(handoff_risk_cost_values)
        handoff_risk_dual_after = float(self._handoff_risk_cost_dual)
        handoff_risk_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if self._handoff_risk_prd_enabled and self._handoff_risk_event_coef > 0.0:
            handoff_risk_credit_values = np.asarray(
                [self._handoff_risk_prd_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._handoff_risk_clip > 0.0:
                handoff_risk_credit_values = np.clip(
                    handoff_risk_credit_values,
                    -self._handoff_risk_clip,
                    self._handoff_risk_clip,
                )
            normalized_event_advantages = (
                normalized_event_advantages
                + self._handoff_risk_event_coef * handoff_risk_credit_values
            )
        mechanism_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if (
            self._mechanism_credit_prd_enabled
            and (
                self._mechanism_credit_policy_coef > 0.0
                or self._mechanism_credit_event_coef > 0.0
            )
        ):
            mechanism_credit_values = np.asarray(
                [self._mechanism_credit_prd_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._mechanism_credit_clip > 0.0:
                mechanism_credit_values = np.clip(
                    mechanism_credit_values,
                    -self._mechanism_credit_clip,
                    self._mechanism_credit_clip,
                )
            if self._mechanism_credit_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._mechanism_credit_policy_coef * mechanism_credit_values
                )
            if self._mechanism_credit_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._mechanism_credit_event_coef * mechanism_credit_values
                )
        tail_risk_reward_floor = 0.0
        if self._tail_risk_prd_enabled and len(rollout) > 0:
            reward_values = np.asarray(
                [float(row.get("reward", 0.0) or 0.0) for row in rollout],
                dtype=np.float32,
            )
            tail_risk_reward_floor = float(np.quantile(reward_values, self._tail_risk_quantile))
        tail_risk_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if self._tail_risk_prd_enabled:
            tail_risk_credit_values = np.asarray(
                [
                    self._tail_risk_prd_credit(row, reward_floor=tail_risk_reward_floor)
                    for row in rollout
                ],
                dtype=np.float32,
            )
            if self._tail_risk_clip > 0.0:
                tail_risk_credit_values = np.clip(
                    tail_risk_credit_values,
                    -self._tail_risk_clip,
                    self._tail_risk_clip,
                )
            if self._tail_risk_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._tail_risk_policy_coef * tail_risk_credit_values
                )
            if self._tail_risk_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._tail_risk_event_coef * tail_risk_credit_values
                )
        opportunity_reward_floor = 0.0
        if self._opportunity_prd_enabled and len(rollout) > 0:
            reward_values = np.asarray(
                [float(row.get("reward", 0.0) or 0.0) for row in rollout],
                dtype=np.float32,
            )
            opportunity_reward_floor = float(np.quantile(reward_values, self._opportunity_reward_quantile))
        opportunity_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if self._opportunity_prd_enabled:
            opportunity_credit_values = np.asarray(
                [
                    self._opportunity_prd_credit(row, reward_floor=opportunity_reward_floor)
                    for row in rollout
                ],
                dtype=np.float32,
            )
            if self._opportunity_clip > 0.0:
                opportunity_credit_values = np.clip(
                    opportunity_credit_values,
                    -self._opportunity_clip,
                    self._opportunity_clip,
                )
            if self._opportunity_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._opportunity_policy_coef * opportunity_credit_values
                )
            if self._opportunity_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._opportunity_event_coef * opportunity_credit_values
                )
        idle_execution_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if self._idle_execution_prd_enabled and self._idle_execution_policy_coef > 0.0:
            idle_execution_credit_values = np.asarray(
                [self._idle_execution_prd_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._idle_execution_clip > 0.0:
                idle_execution_credit_values = np.clip(
                    idle_execution_credit_values,
                    -self._idle_execution_clip,
                    self._idle_execution_clip,
                )
            normalized_advantages = (
                normalized_advantages
                + self._idle_execution_policy_coef * idle_execution_credit_values
            )
        delayed_mechanism_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if (
            self._delayed_mechanism_credit_enabled
            and (
                self._delayed_mechanism_credit_policy_coef > 0.0
                or self._delayed_mechanism_credit_event_coef > 0.0
            )
        ):
            delayed_mechanism_credit_values = self._delayed_mechanism_credit_values(rollout)
            if self._delayed_mechanism_credit_clip > 0.0:
                delayed_mechanism_credit_values = np.clip(
                    delayed_mechanism_credit_values,
                    -self._delayed_mechanism_credit_clip,
                    self._delayed_mechanism_credit_clip,
                )
            if self._delayed_mechanism_credit_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._delayed_mechanism_credit_policy_coef * delayed_mechanism_credit_values
                )
            if self._delayed_mechanism_credit_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._delayed_mechanism_credit_event_coef * delayed_mechanism_credit_values
                )
        net_advantage_prepare_gate_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if (
            self._net_advantage_prepare_gate_enabled
            and (
                self._net_advantage_prepare_gate_policy_coef > 0.0
                or self._net_advantage_prepare_gate_event_coef > 0.0
            )
        ):
            net_advantage_prepare_gate_credit_values = np.asarray(
                [self._net_advantage_prepare_gate_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._net_advantage_prepare_gate_clip > 0.0:
                net_advantage_prepare_gate_credit_values = np.clip(
                    net_advantage_prepare_gate_credit_values,
                    -self._net_advantage_prepare_gate_clip,
                    self._net_advantage_prepare_gate_clip,
                )
            if self._net_advantage_prepare_gate_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._net_advantage_prepare_gate_policy_coef
                    * net_advantage_prepare_gate_credit_values
                )
            if self._net_advantage_prepare_gate_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._net_advantage_prepare_gate_event_coef
                    * net_advantage_prepare_gate_credit_values
                )
        service_completion_gate_credit_values = np.zeros(len(rollout), dtype=np.float32)
        if (
            self._service_completion_gate_enabled
            and (
                self._service_completion_gate_policy_coef > 0.0
                or self._service_completion_gate_event_coef > 0.0
            )
        ):
            service_completion_gate_credit_values = np.asarray(
                [self._service_completion_gate_credit(row) for row in rollout],
                dtype=np.float32,
            )
            if self._service_completion_gate_clip > 0.0:
                service_completion_gate_credit_values = np.clip(
                    service_completion_gate_credit_values,
                    -self._service_completion_gate_clip,
                    self._service_completion_gate_clip,
                )
            if self._service_completion_gate_policy_coef > 0.0:
                normalized_advantages = (
                    normalized_advantages
                    + self._service_completion_gate_policy_coef
                    * service_completion_gate_credit_values
                )
            if self._service_completion_gate_event_coef > 0.0:
                normalized_event_advantages = (
                    normalized_event_advantages
                    + self._service_completion_gate_event_coef
                    * service_completion_gate_credit_values
                )
        advantage_weighted_behavior_stats = self._annotate_advantage_weighted_behavior_targets(
            rollout,
            advantage_values=normalized_advantages,
        )
        normalized_event_advantages = normalized_event_advantages * mechanism_transition_weights
        option_return_mean = float(returns.mean()) if len(returns) > 0 else 0.0
        option_return_std = float(returns.std()) if len(returns) > 0 else 0.0
        normalized_option_returns = (
            (returns - option_return_mean) / (option_return_std + 1e-8)
            if len(returns) > 0
            else returns
        )
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self._device)
        option_return_tensor = torch.as_tensor(
            normalized_option_returns,
            dtype=torch.float32,
            device=self._device,
        )
        advantage_tensor = torch.as_tensor(normalized_advantages, dtype=torch.float32, device=self._device)
        event_advantage_tensor = torch.as_tensor(normalized_event_advantages, dtype=torch.float32, device=self._device)
        old_log_prob_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self._device)
        old_env_action_log_prob_tensor = torch.as_tensor(
            old_env_action_log_probs,
            dtype=torch.float32,
            device=self._device,
        )
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
        option_gate_loss_total = 0.0
        option_gate_entropy_total = 0.0
        option_gate_prior_loss_total = 0.0
        option_counterfactual_value_loss_total = 0.0
        option_counterfactual_advantage_abs_total = 0.0
        env_action_model_critic_loss_total = 0.0
        env_action_model_critic_advantage_abs_total = 0.0
        env_action_model_policy_improvement_loss_total = 0.0
        env_action_model_policy_improvement_target_kl_total = 0.0
        env_action_ppo_loss_total = 0.0
        env_action_counterfactual_margin_loss_total = 0.0
        argmax_margin_loss_total = 0.0
        advantage_weighted_behavior_loss_total = 0.0
        effective_imitation_coef = self._effective_heuristic_imitation_coef()
        effective_option_prior_coef = self._effective_option_gate_prior_coef()
        digital_twin_policy_prior_loss_total = 0.0
        effective_digital_twin_policy_prior_distill_coef = (
            self._effective_digital_twin_policy_prior_distill_coef()
        )
        mechanism_guidance_rollout_stats = self._summarize_mechanism_guidance_annotations(
            mechanism_guidance_annotations,
            rollout,
        )
        digital_twin_policy_prior_applied = [
            item for item in digital_twin_policy_prior_annotations if bool(item.get("apply", False))
        ]
        digital_twin_policy_prior_strength_mean = (
            float(fmean(float(item.get("strength", 0.0) or 0.0) for item in digital_twin_policy_prior_applied))
            if digital_twin_policy_prior_applied
            else 0.0
        )
        digital_twin_policy_prior_event_count = sum(
            1 for item in digital_twin_policy_prior_applied if int(item.get("event_target", 0)) == 1
        )
        digital_twin_policy_prior_prefetch_count = sum(
            1 for item in digital_twin_policy_prior_applied if int(item.get("slow_target", 0)) == 2
        )
        digital_twin_policy_prior_pacing_count = sum(
            1 for item in digital_twin_policy_prior_applied if bool(item.get("pacing_target", False))
        )
        digital_twin_policy_prior_continuation_count = sum(
            1 for item in digital_twin_policy_prior_applied if bool(item.get("continuation_target", False))
        )
        digital_twin_policy_prior_env_prepare_count = sum(
            1 for item in digital_twin_policy_prior_applied if int(item.get("env_target", -1)) == 4
        )
        digital_twin_policy_prior_env_wait_count = sum(
            1 for item in digital_twin_policy_prior_applied if int(item.get("env_target", -1)) == 2
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
                batch_run_metadata = [run_metadata_by_row[int(index)] for index in batch_index_list]
                batch_rows = [rollout[int(index)] for index in batch_index_list]
                batch_action_masks = [action_masks[int(index)] for index in batch_index_list]
                batch_outputs = [
                    self._forward_policy(state, run_metadata=run_metadata)
                    for state, run_metadata in zip(batch_states, batch_run_metadata)
                ]
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
                    env_action_ppo_loss = self._compute_env_action_ppo_loss(
                        batch_outputs=batch_outputs,
                        batch_action_masks=batch_action_masks,
                        batch_actions=action_tensor[batch_indices],
                        old_env_action_log_probs=old_env_action_log_prob_tensor[batch_indices],
                        base_advantage=advantage_tensor[batch_indices],
                        event_advantage=event_advantage_tensor[batch_indices],
                        batch_rows=batch_rows,
                    )
                    env_action_counterfactual_margin_loss = self._compute_env_action_counterfactual_margin_loss(
                        batch_outputs=batch_outputs,
                        batch_action_masks=batch_action_masks,
                        batch_rows=batch_rows,
                        base_advantage=advantage_tensor[batch_indices],
                        event_advantage=event_advantage_tensor[batch_indices],
                    )
                    argmax_margin_loss = self._compute_argmax_margin_regularization_loss(
                        batch_outputs=batch_outputs,
                        batch_action_masks=batch_action_masks,
                        batch_rows=batch_rows,
                    )
                    advantage_weighted_behavior_loss = self._compute_advantage_weighted_behavior_loss(
                        batch_outputs=batch_outputs,
                        batch_action_masks=batch_action_masks,
                        batch_rows=batch_rows,
                    )
                else:
                    logits = torch.stack(
                        [
                            self._masked_flat_logits(output["flat_logits"], mask)
                            for output, mask in zip(batch_outputs, batch_action_masks)
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
                    env_action_ppo_loss = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                    env_action_counterfactual_margin_loss = torch.tensor(
                        0.0,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    argmax_margin_loss = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                    advantage_weighted_behavior_loss = self._compute_advantage_weighted_behavior_loss(
                        batch_outputs=batch_outputs,
                        batch_action_masks=batch_action_masks,
                        batch_rows=batch_rows,
                    )

                value_prediction = torch.stack([output["value"] for output in batch_outputs], dim=0)
                ratio = torch.exp(new_log_prob - old_log_prob_tensor[batch_indices])
                value_loss = torch.mean((return_tensor[batch_indices] - value_prediction) ** 2)
                auxiliary_loss = self._compute_auxiliary_loss(batch_states=batch_states, batch_outputs=batch_outputs)
                heuristic_imitation_loss = self._compute_heuristic_imitation_loss(
                    batch_outputs=batch_outputs,
                    batch_rows=batch_rows,
                )
                batch_annotations = [mechanism_guidance_annotations[int(index)] for index in batch_index_list]
                batch_dt_policy_prior_annotations = [
                    digital_twin_policy_prior_annotations[int(index)]
                    for index in batch_index_list
                ]
                mechanism_aux_loss, mechanism_entropy_bonus = self._compute_mechanism_auxiliary_loss(
                    batch_outputs=batch_outputs,
                    batch_annotations=batch_annotations,
                )
                digital_twin_policy_prior_loss = self._compute_digital_twin_policy_prior_loss(
                    batch_outputs=batch_outputs,
                    batch_annotations=batch_dt_policy_prior_annotations,
                    batch_advantage=advantage_tensor[batch_indices],
                )
                (
                    option_gate_loss,
                    option_gate_entropy,
                    option_gate_prior_loss,
                    option_counterfactual_value_loss,
                    option_counterfactual_advantage_abs,
                ) = self._compute_option_gate_loss(
                    batch_outputs=batch_outputs,
                    batch_rows=batch_rows,
                    batch_advantage=advantage_tensor[batch_indices],
                    batch_option_returns=option_return_tensor[batch_indices],
                )
                (
                    env_action_model_critic_loss,
                    env_action_model_critic_advantage_abs,
                ) = self._compute_env_action_model_critic_loss(
                    batch_outputs=batch_outputs,
                    batch_rows=batch_rows,
                    batch_action_masks=batch_action_masks,
                )
                (
                    env_action_model_policy_improvement_loss,
                    env_action_model_policy_improvement_target_kl,
                ) = self._compute_env_action_model_policy_improvement_loss(
                    batch_outputs=batch_outputs,
                    batch_rows=batch_rows,
                    batch_action_masks=batch_action_masks,
                )
                total_loss = (
                    actor_loss
                    + self._value_coef * value_loss
                    - self._entropy_coef * entropy
                    + self._auxiliary_coef * auxiliary_loss
                    + effective_imitation_coef * heuristic_imitation_loss
                    + effective_mechanism_aux_coef * mechanism_aux_loss
                    - effective_mechanism_entropy_coef * mechanism_entropy_bonus
                    + effective_digital_twin_policy_prior_distill_coef * digital_twin_policy_prior_loss
                    + self._env_action_ppo_coef * env_action_ppo_loss
                    + self._env_action_counterfactual_margin_coef * env_action_counterfactual_margin_loss
                    + self._argmax_margin_coef * argmax_margin_loss
                    + self._advantage_weighted_behavior_coef * advantage_weighted_behavior_loss
                    + self._option_gate_loss_coef * self._option_gate_log_prob_weight * option_gate_loss
                    + effective_option_prior_coef * option_gate_prior_loss
                    - self._option_gate_entropy_coef * option_gate_entropy
                    + self._option_counterfactual_value_coef
                    * option_counterfactual_value_loss
                    + self._env_action_model_critic_value_coef
                    * env_action_model_critic_loss
                    + self._env_action_model_policy_improvement_coef
                    * env_action_model_policy_improvement_loss
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
                digital_twin_policy_prior_loss_total += float(digital_twin_policy_prior_loss.item())
                env_action_ppo_loss_total += float(env_action_ppo_loss.item())
                env_action_counterfactual_margin_loss_total += float(
                    env_action_counterfactual_margin_loss.item()
                )
                argmax_margin_loss_total += float(argmax_margin_loss.item())
                advantage_weighted_behavior_loss_total += float(advantage_weighted_behavior_loss.item())
                option_gate_loss_total += float(option_gate_loss.item())
                option_gate_entropy_total += float(option_gate_entropy.item())
                option_gate_prior_loss_total += float(option_gate_prior_loss.item())
                option_counterfactual_value_loss_total += float(
                    option_counterfactual_value_loss.item()
                )
                option_counterfactual_advantage_abs_total += float(
                    option_counterfactual_advantage_abs.item()
                )
                env_action_model_critic_loss_total += float(
                    env_action_model_critic_loss.item()
                )
                env_action_model_critic_advantage_abs_total += float(
                    env_action_model_critic_advantage_abs.item()
                )
                env_action_model_policy_improvement_loss_total += float(
                    env_action_model_policy_improvement_loss.item()
                )
                env_action_model_policy_improvement_target_kl_total += float(
                    env_action_model_policy_improvement_target_kl.item()
                )
                update_steps += 1
                epoch_kl_values.append(float(approx_kl.item()))
            executed_epochs += 1
            if self._kl_early_stop_enabled and self._target_kl > 0.0 and epoch_kl_values:
                epoch_kl_mean = float(sum(epoch_kl_values) / len(epoch_kl_values))
                if epoch_kl_mean >= self._target_kl:
                    early_stop_triggered = True
                    break

        tail_semantic_states = list(semantic_states)
        tail_run_metadata_by_row = list(run_metadata_by_row)
        tail_rollout = list(rollout)
        tail_action_masks = list(action_masks)
        imagined_recovery_state_count = 0
        if self._env_action_model_imagination_replay_enabled:
            for row in rollout:
                imagined_samples = (
                    row.get("action_info", {})
                    .get("env_action_model_rollout", {})
                    .get("imagined_recovery_samples", [])
                )
                if not isinstance(imagined_samples, list):
                    continue
                for sample in imagined_samples:
                    if not isinstance(sample, dict):
                        continue
                    decision_info = dict(sample.get("decision_info", {}))
                    tail_semantic_states.append(
                        self._extract_semantic_state(decision_info)
                    )
                    tail_run_metadata_by_row.append(
                        dict(decision_info.get("run_metadata", {}) or {})
                    )
                    tail_rollout.append(sample)
                    tail_action_masks.append(
                        self._extract_action_mask(decision_info)
                    )
                    imagined_recovery_state_count += 1

        env_action_model_tail_distillation_stats = (
            self._run_env_action_model_tail_distillation(
                semantic_states=tail_semantic_states,
                run_metadata_by_row=tail_run_metadata_by_row,
                rollout=tail_rollout,
                action_masks=tail_action_masks,
            )
        )
        env_action_model_tail_distillation_stats[
            "imagined_recovery_state_count"
        ] = imagined_recovery_state_count
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
            "conservative_imitation_enabled": self._conservative_imitation_enabled,
            "conservative_imitation_reward_floor": round(
                float(imitation_rollout_stats.get("reward_floor", 0.0)),
                6,
            ),
            "conservative_imitation_weight_mean": round(
                float(imitation_rollout_stats.get("weight_mean", 0.0)),
                6,
            ),
            "conservative_imitation_weight_max": round(
                float(imitation_rollout_stats.get("weight_max", 0.0)),
                6,
            ),
            "mechanism_aux_coef": round(self._mechanism_aux_coef, 6),
            "effective_mechanism_aux_coef": round(effective_mechanism_aux_coef, 6),
            "mechanism_aux_loss_mean": round(mechanism_aux_loss_total / denominator, 6),
            "mechanism_entropy_coef": round(self._mechanism_entropy_coef, 6),
            "effective_mechanism_entropy_coef": round(effective_mechanism_entropy_coef, 6),
            "mechanism_head_entropy": round(mechanism_entropy_bonus_total / denominator, 6),
            "option_gate_enabled": self._option_gate_enabled,
            "option_gate_loss_coef": round(self._option_gate_loss_coef, 6),
            "option_gate_loss": round(option_gate_loss_total / denominator, 6),
            "option_gate_entropy_coef": round(self._option_gate_entropy_coef, 6),
            "option_gate_entropy": round(option_gate_entropy_total / denominator, 6),
            "option_gate_prior_coef": round(self._option_gate_prior_coef, 6),
            "effective_option_gate_prior_coef": round(effective_option_prior_coef, 6),
            "option_gate_prior_loss": round(option_gate_prior_loss_total / denominator, 6),
            "option_counterfactual_critic_enabled": self._option_counterfactual_critic_enabled,
            "option_counterfactual_value_coef": round(
                self._option_counterfactual_value_coef,
                6,
            ),
            "option_counterfactual_value_loss": round(
                option_counterfactual_value_loss_total / denominator,
                6,
            ),
            "option_counterfactual_advantage_coef": round(
                self._option_counterfactual_advantage_coef,
                6,
            ),
            "option_counterfactual_advantage_abs_mean": round(
                option_counterfactual_advantage_abs_total / denominator,
                6,
            ),
            "option_counterfactual_warmup_active": bool(
                self._update_count <= self._option_counterfactual_warmup_updates
            ),
            "option_counterfactual_tail_weight": round(
                self._option_counterfactual_tail_weight,
                6,
            ),
            "option_counterfactual_policy_improvement_enabled": (
                self._option_counterfactual_policy_improvement_enabled
            ),
            "option_counterfactual_policy_improvement_coef": round(
                self._option_counterfactual_policy_improvement_coef,
                6,
            ),
            "option_counterfactual_policy_improvement_clip": round(
                self._option_counterfactual_policy_improvement_clip,
                6,
            ),
            "option_counterfactual_policy_improvement_deterministic_only": (
                self._option_counterfactual_policy_improvement_deterministic_only
            ),
            "option_counterfactual_model_rollout_enabled": (
                self._option_counterfactual_model_rollout_enabled
            ),
            "option_counterfactual_model_rollout_horizon": (
                self._option_counterfactual_model_rollout_horizon
            ),
            "env_action_model_critic_enabled": self._env_action_model_critic_enabled,
            "env_action_model_critic_value_coef": round(
                self._env_action_model_critic_value_coef,
                6,
            ),
            "env_action_model_critic_value_loss": round(
                env_action_model_critic_loss_total / denominator,
                6,
            ),
            "env_action_model_critic_advantage_coef": round(
                self._env_action_model_critic_advantage_coef,
                6,
            ),
            "env_action_model_critic_advantage_abs_mean": round(
                env_action_model_critic_advantage_abs_total / denominator,
                6,
            ),
            "env_action_model_critic_policy_improvement_coef": round(
                self._env_action_model_critic_policy_improvement_coef,
                6,
            ),
            "env_action_model_critic_advantage_clip": round(
                self._env_action_model_critic_advantage_clip,
                6,
            ),
            "env_action_model_critic_warmup_active": bool(
                self._update_count <= self._env_action_model_critic_warmup_updates
            ),
            "env_action_model_rollout_enabled": self._env_action_model_rollout_enabled,
            "env_action_model_rollout_horizon": self._env_action_model_rollout_horizon,
            "env_action_model_rollout_horizons": list(
                self._env_action_model_rollout_horizons
            ),
            "env_action_model_imagination_replay_enabled": (
                self._env_action_model_imagination_replay_enabled
            ),
            "env_action_model_imagination_replay_depths": list(
                self._env_action_model_imagination_replay_depths
            ),
            "env_action_model_imagination_replay_horizons": list(
                self._env_action_model_imagination_replay_horizons
            ),
            "env_action_model_imagination_replay_recovery_only": (
                self._env_action_model_imagination_replay_recovery_only
            ),
            "env_action_model_imagination_beam_search_enabled": (
                self._env_action_model_imagination_beam_search_enabled
            ),
            "env_action_model_imagination_replay_branch_mode": (
                self._env_action_model_imagination_replay_branch_mode
            ),
            "env_action_model_imagination_replay_branch_top_k": (
                self._env_action_model_imagination_replay_branch_top_k
            ),
            "env_action_model_policy_improvement_enabled": (
                self._env_action_model_policy_improvement_enabled
            ),
            "env_action_model_policy_improvement_coef": round(
                self._env_action_model_policy_improvement_coef,
                6,
            ),
            "env_action_model_policy_improvement_temperature": round(
                self._env_action_model_policy_improvement_temperature,
                6,
            ),
            "env_action_model_policy_improvement_robust_horizons_enabled": (
                self._env_action_model_policy_improvement_robust_horizons_enabled
            ),
            "env_action_model_policy_improvement_horizon_risk_coef": round(
                self._env_action_model_policy_improvement_horizon_risk_coef,
                6,
            ),
            "env_action_model_policy_improvement_horizon_aggregation_mode": (
                self._env_action_model_policy_improvement_horizon_aggregation_mode
            ),
            "env_action_model_policy_improvement_horizon_lambda": round(
                self._env_action_model_policy_improvement_horizon_lambda,
                6,
            ),
            "env_action_model_policy_improvement_adaptive_kl_enabled": (
                self._env_action_model_policy_improvement_adaptive_kl_enabled
            ),
            "env_action_model_policy_improvement_kl_constraint": round(
                self._env_action_model_policy_improvement_target_kl,
                6,
            ),
            "env_action_model_policy_improvement_regret_adaptive_kl_enabled": (
                self._env_action_model_policy_improvement_regret_adaptive_kl_enabled
            ),
            "env_action_model_policy_improvement_max_target_kl": round(
                self._env_action_model_policy_improvement_max_target_kl,
                6,
            ),
            "env_action_model_policy_improvement_regret_priority_coef": round(
                self._env_action_model_policy_improvement_regret_priority_coef,
                6,
            ),
            "env_action_model_policy_improvement_tail_distillation_enabled": (
                self._env_action_model_policy_improvement_tail_distillation_enabled
            ),
            "env_action_model_policy_improvement_tail_quantile": round(
                self._env_action_model_policy_improvement_tail_quantile,
                6,
            ),
            "env_action_model_policy_improvement_tail_min_regret": round(
                self._env_action_model_policy_improvement_tail_min_regret,
                6,
            ),
            "env_action_model_policy_improvement_tail_epochs": (
                self._env_action_model_policy_improvement_tail_epochs
            ),
            "env_action_model_policy_improvement_tail_coef": round(
                self._env_action_model_policy_improvement_tail_coef,
                6,
            ),
            "env_action_model_policy_improvement_tail_max_policy_kl": round(
                self._env_action_model_policy_improvement_tail_max_policy_kl,
                6,
            ),
            "env_action_model_policy_improvement_tail_target_balance_enabled": (
                self._env_action_model_policy_improvement_tail_target_balance_enabled
            ),
            "env_action_model_policy_improvement_tail_target_balance_power": round(
                self._env_action_model_policy_improvement_tail_target_balance_power,
                6,
            ),
            "env_action_model_policy_improvement_tail_target_balance_max_weight": round(
                self._env_action_model_policy_improvement_tail_target_balance_max_weight,
                6,
            ),
            "env_action_model_online_planner_enabled": (
                self._env_action_model_online_planner_enabled
            ),
            "env_action_model_online_planner_coef": round(
                self._env_action_model_online_planner_coef,
                6,
            ),
            "env_action_model_online_planner_mechanism_coef": round(
                self._env_action_model_online_planner_mechanism_coef,
                6,
            ),
            "env_action_model_online_planner_policy_prior_coef": round(
                self._env_action_model_online_planner_policy_prior_coef,
                6,
            ),
            "env_action_model_online_planner_min_margin": round(
                self._env_action_model_online_planner_min_margin,
                6,
            ),
            "env_action_model_resource_constraint_enabled": (
                self._env_action_model_resource_constraint_enabled
            ),
            "env_action_model_resource_cost_coef": round(
                self._env_action_model_resource_cost_coef,
                6,
            ),
            "env_action_model_resource_cost_scale": round(
                self._env_action_model_resource_cost_scale,
                6,
            ),
            "env_action_model_adaptive_horizon_enabled": (
                self._env_action_model_adaptive_horizon_enabled
            ),
            "env_action_model_adaptive_horizon_temperature": round(
                self._env_action_model_adaptive_horizon_temperature,
                6,
            ),
            "env_action_model_online_planner_prefer_beam_targets": (
                self._env_action_model_online_planner_prefer_beam_targets
            ),
            "env_action_model_policy_improvement_tail_residual_optimizer_enabled": (
                self._env_action_model_policy_improvement_tail_residual_optimizer_enabled
            ),
            "env_action_model_policy_improvement_tail_residual_learning_rate": round(
                self._env_action_model_policy_improvement_tail_residual_learning_rate,
                8,
            ),
            "env_action_model_policy_improvement_tail_residual_backtrack_factor": round(
                self._env_action_model_policy_improvement_tail_residual_backtrack_factor,
                6,
            ),
            "env_action_model_policy_improvement_tail_residual_min_learning_rate": round(
                self._env_action_model_policy_improvement_tail_residual_min_learning_rate,
                8,
            ),
            "env_action_model_policy_improvement_tail_residual_max_backtracks": (
                self._env_action_model_policy_improvement_tail_residual_max_backtracks
            ),
            "env_action_model_policy_improvement_loss": round(
                env_action_model_policy_improvement_loss_total / denominator,
                6,
            ),
            "env_action_model_policy_improvement_target_kl": round(
                env_action_model_policy_improvement_target_kl_total / denominator,
                6,
            ),
            "env_action_model_tail_distillation_candidate_count": int(
                env_action_model_tail_distillation_stats["candidate_count"]
            ),
            "env_action_model_tail_distillation_imagined_state_count": int(
                env_action_model_tail_distillation_stats[
                    "imagined_recovery_state_count"
                ]
            ),
            "env_action_model_tail_distillation_selected_count": int(
                env_action_model_tail_distillation_stats["selected_count"]
            ),
            "env_action_model_tail_distillation_selected_imagined_count": int(
                env_action_model_tail_distillation_stats[
                    "selected_imagined_count"
                ]
            ),
            "env_action_model_tail_distillation_selected_fraction": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "selected_fraction"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_regret_mean": round(
                float(env_action_model_tail_distillation_stats["regret_mean"]),
                6,
            ),
            "env_action_model_tail_distillation_regret_threshold": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "regret_threshold"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_selected_regret_mean": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "selected_regret_mean"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_loss": round(
                float(env_action_model_tail_distillation_stats["loss"]),
                6,
            ),
            "env_action_model_tail_distillation_update_steps": int(
                env_action_model_tail_distillation_stats["update_steps"]
            ),
            "env_action_model_tail_distillation_executed_epochs": int(
                env_action_model_tail_distillation_stats["executed_epochs"]
            ),
            "env_action_model_tail_distillation_early_stop_triggered": bool(
                env_action_model_tail_distillation_stats[
                    "early_stop_triggered"
                ]
            ),
            "env_action_model_tail_distillation_residual_optimizer_used": bool(
                env_action_model_tail_distillation_stats[
                    "residual_optimizer_used"
                ]
            ),
            "env_action_model_tail_distillation_residual_learning_rate_initial": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "residual_learning_rate_initial"
                    ]
                ),
                8,
            ),
            "env_action_model_tail_distillation_residual_learning_rate_final": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "residual_learning_rate_final"
                    ]
                ),
                8,
            ),
            "env_action_model_tail_distillation_backtrack_count": int(
                env_action_model_tail_distillation_stats[
                    "backtrack_count"
                ]
            ),
            "env_action_model_tail_distillation_rejected_epoch_count": int(
                env_action_model_tail_distillation_stats[
                    "rejected_epoch_count"
                ]
            ),
            "env_action_model_tail_distillation_target_action_counts": dict(
                env_action_model_tail_distillation_stats[
                    "target_action_counts"
                ]
            ),
            "env_action_model_tail_distillation_imagined_target_action_counts": dict(
                env_action_model_tail_distillation_stats[
                    "imagined_target_action_counts"
                ]
            ),
            "env_action_model_tail_distillation_selected_target_action_counts": dict(
                env_action_model_tail_distillation_stats[
                    "selected_target_action_counts"
                ]
            ),
            "env_action_model_tail_distillation_selected_imagined_target_action_counts": dict(
                env_action_model_tail_distillation_stats[
                    "selected_imagined_target_action_counts"
                ]
            ),
            "env_action_model_tail_distillation_target_balance_enabled": bool(
                env_action_model_tail_distillation_stats[
                    "target_balance_enabled"
                ]
            ),
            "env_action_model_tail_distillation_target_balance_weights": dict(
                env_action_model_tail_distillation_stats[
                    "target_balance_weights"
                ]
            ),
            "env_action_model_tail_distillation_target_balance_weight_min": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "target_balance_weight_min"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_target_balance_weight_mean": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "target_balance_weight_mean"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_target_balance_weight_max": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "target_balance_weight_max"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_policy_kl_before": round(
                float(
                    env_action_model_tail_distillation_stats[
                        "policy_kl_before"
                    ]
                ),
                6,
            ),
            "env_action_model_tail_distillation_policy_kl_after": round(
                float(
                    env_action_model_tail_distillation_stats["policy_kl_after"]
                ),
                6,
            ),
            "option_return_target_mean": round(option_return_mean, 6),
            "option_return_target_std": round(option_return_std, 6),
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
            "env_action_ppo_enabled": self._env_action_ppo_enabled,
            "env_action_ppo_coef": round(self._env_action_ppo_coef, 6),
            "env_action_ppo_advantage_blend": round(self._env_action_ppo_advantage_blend, 6),
            "env_action_ppo_teacher_coef": round(self._env_action_ppo_teacher_coef, 6),
            "env_action_ppo_mechanism_focus": round(self._env_action_ppo_mechanism_focus, 6),
            "env_action_sparse_recovery_focus": round(self._env_action_sparse_recovery_focus, 6),
            "env_action_risk_adjusted_recovery_coef": round(
                self._env_action_risk_adjusted_recovery_coef,
                6,
            ),
            "env_action_risk_adjusted_recovery_floor": round(
                self._env_action_risk_adjusted_recovery_floor,
                6,
            ),
            "env_action_adapter_miss_counterfactual_coef": round(
                self._env_action_adapter_miss_counterfactual_coef,
                6,
            ),
            "cache_feasibility_prior_enabled": self._cache_feasibility_prior_enabled,
            "cache_feasibility_cache_fill_bias": round(self._cache_feasibility_cache_fill_bias, 6),
            "cache_feasibility_steady_penalty": round(self._cache_feasibility_steady_penalty, 6),
            "cache_feasibility_prepare_penalty": round(self._cache_feasibility_prepare_penalty, 6),
            "cache_feasibility_prefetch_penalty": round(self._cache_feasibility_prefetch_penalty, 6),
            "cache_feasibility_current_miss_prepare_penalty": round(
                self._cache_feasibility_current_miss_prepare_penalty,
                6,
            ),
            "cache_feasibility_current_miss_prefetch_penalty": round(
                self._cache_feasibility_current_miss_prefetch_penalty,
                6,
            ),
            "cache_feasibility_min_context": round(self._cache_feasibility_min_context, 6),
            "handoff_alignment_barrier_enabled": self._handoff_alignment_barrier_enabled,
            "handoff_alignment_barrier_prepare_penalty": round(
                self._handoff_alignment_barrier_prepare_penalty,
                6,
            ),
            "handoff_alignment_barrier_prefetch_penalty": round(
                self._handoff_alignment_barrier_prefetch_penalty,
                6,
            ),
            "handoff_alignment_barrier_current_fill_bias": round(
                self._handoff_alignment_barrier_current_fill_bias,
                6,
            ),
            "handoff_alignment_barrier_target_mismatch_penalty": round(
                self._handoff_alignment_barrier_target_mismatch_penalty,
                6,
            ),
            "handoff_alignment_barrier_late_eta_penalty": round(
                self._handoff_alignment_barrier_late_eta_penalty,
                6,
            ),
            "handoff_alignment_barrier_min_context": round(
                self._handoff_alignment_barrier_min_context,
                6,
            ),
            "env_action_ppo_max_weight": round(self._env_action_ppo_max_weight, 6),
            "env_action_ppo_ratio_barrier_coef": round(self._env_action_ppo_ratio_barrier_coef, 6),
            "env_action_ppo_ratio_barrier_margin": round(self._env_action_ppo_ratio_barrier_margin, 6),
            "env_action_ppo_loss": round(env_action_ppo_loss_total / denominator, 6),
            "env_action_counterfactual_margin_enabled": self._env_action_counterfactual_margin_enabled,
            "env_action_counterfactual_margin_coef": round(
                self._env_action_counterfactual_margin_coef,
                6,
            ),
            "env_action_counterfactual_margin_loss": round(
                env_action_counterfactual_margin_loss_total / denominator,
                6,
            ),
            "env_action_counterfactual_margin_advantage_gate": round(
                self._env_action_counterfactual_margin_advantage_gate,
                6,
            ),
            "env_action_counterfactual_margin_advantage_blend": round(
                self._env_action_counterfactual_margin_advantage_blend,
                6,
            ),
            "argmax_margin_regularization_enabled": self._argmax_margin_regularization_enabled,
            "argmax_margin_coef": round(self._argmax_margin_coef, 6),
            "argmax_margin_min_gap": round(self._argmax_margin_min_gap, 6),
            "argmax_margin_tail_risk_threshold": round(self._argmax_margin_tail_risk_threshold, 6),
            "argmax_margin_loss": round(argmax_margin_loss_total / denominator, 6),
            "advantage_weighted_behavior_regularization_enabled": (
                self._advantage_weighted_behavior_regularization_enabled
            ),
            "advantage_weighted_behavior_coef": round(self._advantage_weighted_behavior_coef, 6),
            "advantage_weighted_behavior_loss": round(
                advantage_weighted_behavior_loss_total / denominator,
                6,
            ),
            "advantage_weighted_behavior_applied_count": int(
                advantage_weighted_behavior_stats["applied_count"]
            ),
            "advantage_weighted_behavior_positive_count": int(
                advantage_weighted_behavior_stats["positive_count"]
            ),
            "advantage_weighted_behavior_negative_count": int(
                advantage_weighted_behavior_stats["negative_count"]
            ),
            "advantage_weighted_behavior_teacher_match_rate": round(
                float(advantage_weighted_behavior_stats["teacher_match_rate"]),
                6,
            ),
            "advantage_weighted_behavior_weight_mean": round(
                float(advantage_weighted_behavior_stats["weight_mean"]),
                6,
            ),
            "advantage_weighted_behavior_weight_max": round(
                float(advantage_weighted_behavior_stats["weight_max"]),
                6,
            ),
            "env_action_log_prob_missing_count": int(old_env_action_log_prob_missing_count),
            "event_prd_advantage_enabled": self._event_prd_advantage_enabled,
            "event_prd_advantage_coef": round(self._event_prd_advantage_coef, 6),
            "event_prd_credit_mean": round(float(event_prd_credit_values.mean()), 6)
            if len(event_prd_credit_values) > 0
            else 0.0,
            "event_prd_credit_std": round(float(event_prd_credit_values.std()), 6)
            if len(event_prd_credit_values) > 0
            else 0.0,
            "delayed_mechanism_credit_enabled": self._delayed_mechanism_credit_enabled,
            "delayed_mechanism_credit_policy_coef": round(self._delayed_mechanism_credit_policy_coef, 6),
            "delayed_mechanism_credit_event_coef": round(self._delayed_mechanism_credit_event_coef, 6),
            "delayed_mechanism_credit_horizon": int(self._delayed_mechanism_credit_horizon),
            "delayed_mechanism_credit_decay": round(self._delayed_mechanism_credit_decay, 6),
            "delayed_mechanism_credit_clip": round(self._delayed_mechanism_credit_clip, 6),
            "delayed_mechanism_credit_strict_opportunity_enabled": (
                self._delayed_mechanism_credit_strict_opportunity_enabled
            ),
            "delayed_mechanism_credit_mean": round(float(delayed_mechanism_credit_values.mean()), 6)
            if len(delayed_mechanism_credit_values) > 0
            else 0.0,
            "delayed_mechanism_credit_std": round(float(delayed_mechanism_credit_values.std()), 6)
            if len(delayed_mechanism_credit_values) > 0
            else 0.0,
            "delayed_mechanism_credit_min": round(float(delayed_mechanism_credit_values.min()), 6)
            if len(delayed_mechanism_credit_values) > 0
            else 0.0,
            "delayed_mechanism_credit_max": round(float(delayed_mechanism_credit_values.max()), 6)
            if len(delayed_mechanism_credit_values) > 0
            else 0.0,
            "delayed_mechanism_credit_positive_count": int(np.sum(delayed_mechanism_credit_values > 1e-8))
            if len(delayed_mechanism_credit_values) > 0
            else 0,
            "delayed_mechanism_credit_negative_count": int(np.sum(delayed_mechanism_credit_values < -1e-8))
            if len(delayed_mechanism_credit_values) > 0
            else 0,
            "handoff_risk_prd_enabled": self._handoff_risk_prd_enabled,
            "handoff_risk_event_coef": round(self._handoff_risk_event_coef, 6),
            "handoff_risk_option_coef": round(self._handoff_risk_option_coef, 6),
            "handoff_risk_credit_mean": round(float(handoff_risk_credit_values.mean()), 6)
            if len(handoff_risk_credit_values) > 0
            else 0.0,
            "handoff_risk_credit_std": round(float(handoff_risk_credit_values.std()), 6)
            if len(handoff_risk_credit_values) > 0
            else 0.0,
            "handoff_risk_cost_dual_enabled": self._handoff_risk_cost_dual_enabled,
            "handoff_risk_cost_dual_before": round(handoff_risk_dual_before, 6),
            "handoff_risk_cost_dual_after": round(handoff_risk_dual_after, 6),
            "handoff_risk_cost_signal_mean": round(float(handoff_risk_cost_values.mean()), 6)
            if len(handoff_risk_cost_values) > 0
            else 0.0,
            "mechanism_credit_prd_enabled": self._mechanism_credit_prd_enabled,
            "mechanism_credit_policy_coef": round(self._mechanism_credit_policy_coef, 6),
            "mechanism_credit_event_coef": round(self._mechanism_credit_event_coef, 6),
            "mechanism_credit_option_coef": round(self._mechanism_credit_option_coef, 6),
            "mechanism_credit_mean": round(float(mechanism_credit_values.mean()), 6)
            if len(mechanism_credit_values) > 0
            else 0.0,
            "mechanism_credit_std": round(float(mechanism_credit_values.std()), 6)
            if len(mechanism_credit_values) > 0
            else 0.0,
            "mechanism_credit_min": round(float(mechanism_credit_values.min()), 6)
            if len(mechanism_credit_values) > 0
            else 0.0,
            "mechanism_credit_max": round(float(mechanism_credit_values.max()), 6)
            if len(mechanism_credit_values) > 0
            else 0.0,
            "mechanism_focal_aux_enabled": self._mechanism_focal_aux_enabled,
            "mechanism_focal_gamma": round(self._mechanism_focal_gamma, 6),
            "digital_twin_handoff_fusion_enabled": self._digital_twin_handoff_fusion_enabled,
            "digital_twin_handoff_slow_scale": round(self._digital_twin_handoff_slow_scale, 6),
            "digital_twin_handoff_fast_scale": round(self._digital_twin_handoff_fast_scale, 6),
            "digital_twin_handoff_event_scale": round(self._digital_twin_handoff_event_scale, 6),
            "digital_twin_handoff_critic_scale": round(self._digital_twin_handoff_critic_scale, 6),
            "outcome_memory_fusion_enabled": self._outcome_memory_fusion_enabled,
            "outcome_memory_actor_scale": round(
                self._outcome_memory_actor_scale,
                6,
            ),
            "outcome_memory_critic_scale": round(
                self._outcome_memory_critic_scale,
                6,
            ),
            "digital_twin_policy_prior_enabled": self._digital_twin_policy_prior_enabled,
            "digital_twin_policy_prior_logit_bias": round(
                self._digital_twin_policy_prior_logit_bias,
                6,
            ),
            "digital_twin_policy_prior_distill_coef": round(
                self._digital_twin_policy_prior_distill_coef,
                6,
            ),
            "effective_digital_twin_policy_prior_distill_coef": round(
                effective_digital_twin_policy_prior_distill_coef,
                6,
            ),
            "digital_twin_policy_prior_loss": round(
                digital_twin_policy_prior_loss_total / denominator,
                6,
            ),
            "digital_twin_policy_prior_applied_count": int(len(digital_twin_policy_prior_applied)),
            "digital_twin_policy_prior_event_count": int(digital_twin_policy_prior_event_count),
            "digital_twin_policy_prior_prefetch_count": int(digital_twin_policy_prior_prefetch_count),
            "digital_twin_policy_prior_pacing_count": int(digital_twin_policy_prior_pacing_count),
            "digital_twin_policy_prior_continuation_count": int(
                digital_twin_policy_prior_continuation_count
            ),
            "digital_twin_policy_prior_env_prepare_count": int(
                digital_twin_policy_prior_env_prepare_count
            ),
            "digital_twin_policy_prior_env_wait_count": int(
                digital_twin_policy_prior_env_wait_count
            ),
            "digital_twin_policy_prior_adaptive_wait_enabled": (
                self._digital_twin_policy_prior_adaptive_wait_enabled
            ),
            "digital_twin_policy_prior_wait_ready_threshold": round(
                self._digital_twin_policy_prior_wait_ready_threshold,
                6,
            ),
            "digital_twin_policy_prior_wait_timing_ceiling": round(
                self._digital_twin_policy_prior_wait_timing_ceiling,
                6,
            ),
            "digital_twin_policy_prior_strength_mean": round(
                digital_twin_policy_prior_strength_mean,
                6,
            ),
            "tail_risk_prd_enabled": self._tail_risk_prd_enabled,
            "tail_risk_policy_coef": round(self._tail_risk_policy_coef, 6),
            "tail_risk_event_coef": round(self._tail_risk_event_coef, 6),
            "tail_risk_option_coef": round(self._tail_risk_option_coef, 6),
            "tail_risk_reward_floor": round(float(tail_risk_reward_floor), 6),
            "tail_risk_credit_mean": round(float(tail_risk_credit_values.mean()), 6)
            if len(tail_risk_credit_values) > 0
            else 0.0,
            "tail_risk_credit_std": round(float(tail_risk_credit_values.std()), 6)
            if len(tail_risk_credit_values) > 0
            else 0.0,
            "opportunity_prd_enabled": self._opportunity_prd_enabled,
            "opportunity_policy_coef": round(self._opportunity_policy_coef, 6),
            "opportunity_event_coef": round(self._opportunity_event_coef, 6),
            "opportunity_option_coef": round(self._opportunity_option_coef, 6),
            "opportunity_reward_floor": round(float(opportunity_reward_floor), 6),
            "opportunity_credit_mean": round(float(opportunity_credit_values.mean()), 6)
            if len(opportunity_credit_values) > 0
            else 0.0,
            "opportunity_credit_std": round(float(opportunity_credit_values.std()), 6)
            if len(opportunity_credit_values) > 0
            else 0.0,
            "idle_execution_prd_enabled": self._idle_execution_prd_enabled,
            "idle_execution_policy_coef": round(self._idle_execution_policy_coef, 6),
            "idle_execution_option_coef": round(self._idle_execution_option_coef, 6),
            "idle_execution_credit_mean": round(float(idle_execution_credit_values.mean()), 6)
            if len(idle_execution_credit_values) > 0
            else 0.0,
            "idle_execution_credit_std": round(float(idle_execution_credit_values.std()), 6)
            if len(idle_execution_credit_values) > 0
            else 0.0,
            "net_utility_prd_enabled": self._net_utility_prd_enabled,
            "net_utility_backhaul_coef": round(self._net_utility_backhaul_coef, 6),
            "net_utility_migration_coef": round(self._net_utility_migration_coef, 6),
            "net_utility_expired_prefetch_coef": round(self._net_utility_expired_prefetch_coef, 6),
            "net_utility_idle_prefetch_penalty": round(self._net_utility_idle_prefetch_penalty, 6),
            "net_utility_failed_mechanism_penalty": round(self._net_utility_failed_mechanism_penalty, 6),
            "net_utility_failed_mechanism_backhaul_coef": round(
                self._net_utility_failed_mechanism_backhaul_coef,
                6,
            ),
            "net_utility_cost_dual_enabled": self._net_utility_cost_dual_enabled,
            "net_utility_cost_dual_before": round(net_utility_dual_before, 6),
            "net_utility_cost_dual_after": round(net_utility_dual_after, 6),
            "net_utility_cost_signal_mean": round(float(net_utility_cost_values.mean()), 6)
            if len(net_utility_cost_values) > 0
            else 0.0,
            "net_utility_option_termination_enabled": self._net_utility_option_termination_enabled,
            "net_utility_option_termination_conservative_enabled": (
                self._net_utility_option_termination_conservative_enabled
            ),
            "net_utility_option_termination_max_timing_support": round(
                self._net_utility_option_termination_max_timing_support,
                6,
            ),
            "dag_aware_option_termination_enabled": self._dag_aware_option_termination_enabled,
            "dag_aware_option_min_critical_path": self._dag_aware_option_min_critical_path,
            "dag_aware_option_short_workflow_max_nodes": self._dag_aware_option_short_workflow_max_nodes,
            "dag_aware_option_branching_successors": self._dag_aware_option_branching_successors,
            "dag_aware_idle_prefetch_confidence_floor": round(
                self._dag_aware_idle_prefetch_confidence_floor,
                6,
            ),
            "net_utility_prd_adjustment_mean": round(float(net_utility_adjustment_values.mean()), 6)
            if len(net_utility_adjustment_values) > 0
            else 0.0,
            "net_utility_prd_adjustment_std": round(float(net_utility_adjustment_values.std()), 6)
            if len(net_utility_adjustment_values) > 0
            else 0.0,
            "net_advantage_prepare_gate_enabled": self._net_advantage_prepare_gate_enabled,
            "net_advantage_prepare_gate_policy_coef": round(
                self._net_advantage_prepare_gate_policy_coef,
                6,
            ),
            "net_advantage_prepare_gate_event_coef": round(
                self._net_advantage_prepare_gate_event_coef,
                6,
            ),
            "net_advantage_prepare_gate_credit_mean": round(
                float(net_advantage_prepare_gate_credit_values.mean()),
                6,
            )
            if len(net_advantage_prepare_gate_credit_values) > 0
            else 0.0,
            "net_advantage_prepare_gate_credit_std": round(
                float(net_advantage_prepare_gate_credit_values.std()),
                6,
            )
            if len(net_advantage_prepare_gate_credit_values) > 0
            else 0.0,
            "net_advantage_prepare_gate_credit_positive_count": int(
                np.sum(net_advantage_prepare_gate_credit_values > 1e-8)
            )
            if len(net_advantage_prepare_gate_credit_values) > 0
            else 0,
            "net_advantage_prepare_gate_credit_negative_count": int(
                np.sum(net_advantage_prepare_gate_credit_values < -1e-8)
            )
            if len(net_advantage_prepare_gate_credit_values) > 0
            else 0,
            "service_completion_gate_enabled": self._service_completion_gate_enabled,
            "service_completion_gate_policy_coef": round(
                self._service_completion_gate_policy_coef,
                6,
            ),
            "service_completion_gate_event_coef": round(
                self._service_completion_gate_event_coef,
                6,
            ),
            "service_completion_gate_credit_mean": round(
                float(service_completion_gate_credit_values.mean()),
                6,
            )
            if len(service_completion_gate_credit_values) > 0
            else 0.0,
            "service_completion_gate_credit_std": round(
                float(service_completion_gate_credit_values.std()),
                6,
            )
            if len(service_completion_gate_credit_values) > 0
            else 0.0,
            "service_completion_gate_credit_positive_count": int(
                np.sum(service_completion_gate_credit_values > 1e-8)
            )
            if len(service_completion_gate_credit_values) > 0
            else 0,
            "service_completion_gate_credit_negative_count": int(
                np.sum(service_completion_gate_credit_values < -1e-8)
            )
            if len(service_completion_gate_credit_values) > 0
            else 0,
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
            "learned_transition_model_enabled": self._learned_transition_model_enabled,
            "learned_transition_model_planner_enabled": self._learned_transition_model_planner_enabled,
            "learned_transition_model_fit": dict(learned_transition_model_stats),
            "learned_transition_model_sample_count": int(
                self._learned_transition_model.sample_count
                if self._learned_transition_model is not None
                else 0
            ),
            "learned_transition_model_ready": bool(
                self._learned_transition_model is not None
                and self._learned_transition_model.ready
            ),
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
            "learned_transition_model_state": (
                self._learned_transition_model.state_dict()
                if self._learned_transition_model is not None
                else None
            ),
        }
        torch.save(checkpoint, output_path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(Path(path), map_location=self._device)
        network_state = checkpoint["network_state_dict"]
        current_state = self._network.state_dict()
        missing_keys = set(current_state) - set(network_state)
        unexpected_keys = set(network_state) - set(current_state)
        additive_head_warm_start = bool(
            missing_keys
            and not unexpected_keys
            and all(
                str(key).startswith(
                    (
                        "option_actor.",
                        "outcome_recovery_adapter.",
                        "outcome_context_residual_adapter.",
                        "digital_twin_planning_adapter.",
                    )
                )
                for key in missing_keys
            )
        )
        if additive_head_warm_start:
            merged_state = dict(current_state)
            for key, value in network_state.items():
                if key in merged_state and tuple(merged_state[key].shape) == tuple(value.shape):
                    merged_state[key] = value
            self._network.load_state_dict(merged_state, strict=True)
        else:
            self._network.load_state_dict(network_state)
        if checkpoint.get("optimizer_state_dict") is not None and not additive_head_warm_start:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learned_model_state = checkpoint.get("learned_transition_model_state")
        if self._learned_transition_model is not None and isinstance(learned_model_state, dict):
            self._learned_transition_model.load_state_dict(learned_model_state)
        self._update_count = int(checkpoint.get("update_count", 0))

    def _extract_semantic_state(self, info: dict[str, Any] | None) -> dict[str, Any]:
        semantic_state = (info or {}).get("semantic_state")
        if semantic_state is None:
            raise ValueError(f"{self.agent_name} 需要 info['semantic_state'] 才能做图结构编码。")
        algorithm_memory = (info or {}).get("algorithm_memory")
        if not isinstance(algorithm_memory, dict):
            return semantic_state
        augmented_state = dict(semantic_state)
        augmented_state["algorithm_memory"] = dict(algorithm_memory)
        return augmented_state

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

    def _select_allowed_env_action(self, preferred_actions: list[int], action_mask: list[bool] | None) -> int:
        for action_id in preferred_actions:
            if self._is_env_action_valid(int(action_id), action_mask):
                return int(action_id)
        if action_mask:
            for action_id, allowed in enumerate(action_mask):
                if allowed:
                    return int(action_id)
        return 3

    def _primary_vehicle_for_popularity(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        vehicles = list(semantic_state.get("vehicles", []))
        if not vehicles:
            return {}
        primary_vehicle_id = semantic_state.get("primary_vehicle_id")
        if primary_vehicle_id:
            for vehicle in vehicles:
                if str(vehicle.get("vehicle_id", "")) == str(primary_vehicle_id):
                    return dict(vehicle)
        return dict(vehicles[0])

    def _rsu_by_id_for_popularity(
        self,
        semantic_state: dict[str, Any],
        rsu_id: str | None,
    ) -> dict[str, Any]:
        if rsu_id is None:
            return {}
        for rsu in semantic_state.get("rsus", []):
            if str(rsu.get("rsu_id", "")) == str(rsu_id):
                return dict(rsu)
        return {}

    def _adapter_cached_for_popularity(
        self,
        semantic_state: dict[str, Any],
        rsu_id: str | None,
        adapter_id: str | None,
    ) -> bool:
        if not rsu_id or not adapter_id:
            return False
        rsu = self._rsu_by_id_for_popularity(semantic_state, rsu_id)
        return str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", [])}

    def _prediction_targets_for_popularity(
        self,
        semantic_state: dict[str, Any],
        vehicle_id: str | None,
    ) -> tuple[str | None, str | None]:
        predictions = semantic_state.get("predictions", {})
        if not vehicle_id or not isinstance(predictions, dict):
            return None, None
        next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        next_sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
        if next_rsu_id is None and next_sequence:
            next_rsu_id = next_sequence[0]
        handoff_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        return next_rsu_id, handoff_target

    def _remember_idle_popularity_adapter(self, adapter_id: str | None) -> int:
        if not adapter_id:
            return 0
        adapter_key = str(adapter_id)
        self._idle_popularity_adapter_counts[adapter_key] = (
            self._idle_popularity_adapter_counts.get(adapter_key, 0) + 1
        )
        return self._idle_popularity_adapter_counts[adapter_key]

    def _low_mechanism_no_rsu_context_for_popularity(
        self,
        *,
        semantic_state: dict[str, Any],
        vehicle_id: str | None,
        predicted_next_rsu_id: str | None,
        predicted_handoff_target: str | None,
    ) -> bool:
        vehicles = list(semantic_state.get("vehicles", []))
        if len(vehicles) > 1:
            return False
        predictions = semantic_state.get("predictions", {})
        if not isinstance(predictions, dict):
            return True
        predicted_handoff_vehicle_ids = predictions.get("predicted_handoff_vehicle_ids", [])
        if isinstance(predicted_handoff_vehicle_ids, list) and predicted_handoff_vehicle_ids:
            return False
        if predicted_next_rsu_id or predicted_handoff_target:
            return False
        if vehicle_id:
            next_sequence_by_vehicle = predictions.get("next_rsu_sequence", {})
            if isinstance(next_sequence_by_vehicle, dict):
                next_sequence = next_sequence_by_vehicle.get(vehicle_id, [])
                if any(item for item in next_sequence):
                    return False
        return True

    def _idle_popularity_candidate_action(
        self,
        semantic_state: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> tuple[int, str, dict[str, Any]]:
        current_node = semantic_state.get("current_workflow_node") or {}
        if not current_node:
            return self._select_allowed_env_action([3, 2, 0], action_mask), "no_current_workflow_node", {}

        vehicle = self._primary_vehicle_for_popularity(semantic_state)
        vehicle_id = str(vehicle.get("vehicle_id", "")) if vehicle else None
        current_rsu_id = vehicle.get("associated_rsu_id")
        required_adapter = current_node.get("required_adapter")
        adapter_seen_count = self._remember_idle_popularity_adapter(required_adapter)
        predicted_next_rsu_id, predicted_handoff_target = self._prediction_targets_for_popularity(
            semantic_state,
            vehicle_id,
        )
        extra = {
            "adapter_seen_count": adapter_seen_count,
            "low_mechanism_no_rsu_context": self._low_mechanism_no_rsu_context_for_popularity(
                semantic_state=semantic_state,
                vehicle_id=vehicle_id,
                predicted_next_rsu_id=predicted_next_rsu_id,
                predicted_handoff_target=predicted_handoff_target,
            ),
        }

        if current_rsu_id is None:
            if self._net_advantage_prepare_gate_enabled and (
                predicted_handoff_target or predicted_next_rsu_id
            ):
                return self._select_allowed_env_action(
                    [4, 1, 2],
                    action_mask,
                ), "no_associated_rsu_coverage_recovery_prepare", {
                    **extra,
                    "predicted_next_rsu_id": predicted_next_rsu_id,
                    "predicted_handoff_target": predicted_handoff_target,
                }
            if self._idle_popularity_no_rsu_service_continuity_enabled:
                return self._select_allowed_env_action(
                    [3, 2, 0],
                    action_mask,
                ), "no_associated_rsu_service_continuity", extra
            return self._select_allowed_env_action([2, 3, 0], action_mask), "no_associated_rsu_vehicle_fallback", extra

        if required_adapter and not self._adapter_cached_for_popularity(
            semantic_state,
            current_rsu_id,
            required_adapter,
        ):
            return self._select_allowed_env_action([0, 3, 2], action_mask), "popular_adapter_reactive_cache_fill", extra

        if (
            adapter_seen_count >= self._idle_popularity_prefetch_threshold
            and predicted_next_rsu_id
            and predicted_next_rsu_id != current_rsu_id
            and not self._adapter_cached_for_popularity(semantic_state, predicted_next_rsu_id, required_adapter)
        ):
            return self._select_allowed_env_action([1, 3, 4], action_mask), "popular_adapter_predictive_prefetch", {
                **extra,
                "predicted_next_rsu_id": predicted_next_rsu_id,
            }

        if predicted_handoff_target and predicted_handoff_target != current_rsu_id:
            return self._select_allowed_env_action([4, 3, 1], action_mask), "predicted_handoff_migration_prepare", {
                **extra,
                "predicted_handoff_target": predicted_handoff_target,
            }

        return self._select_allowed_env_action([3, 0, 2], action_mask), "popularity_steady_offload", extra

    def _maybe_apply_idle_popularity_fallback(
        self,
        *,
        semantic_state: dict[str, Any],
        action_mask: list[bool] | None,
        original_env_action: int,
        deterministic: bool,
    ) -> dict[str, Any]:
        if not self._idle_popularity_fallback_enabled:
            return {"enabled": False, "applied": False}
        fallback_action, fallback_reason, fallback_extra = self._idle_popularity_candidate_action(
            semantic_state,
            action_mask,
        )
        if not deterministic:
            return {
                "enabled": True,
                "applied": False,
                "reason": "non_deterministic_policy",
                "candidate_action": int(fallback_action),
                "candidate_reason": fallback_reason,
                **fallback_extra,
            }
        no_rsu_local_override = (
            self._idle_popularity_no_rsu_local_fallback_enabled
            and fallback_reason == "no_associated_rsu_vehicle_fallback"
            and (
                not self._idle_popularity_no_rsu_local_requires_low_context
                or bool(fallback_extra.get("low_mechanism_no_rsu_context", False))
            )
            and int(fallback_action) != int(original_env_action)
        )
        no_rsu_service_override = (
            self._idle_popularity_no_rsu_any_action_override_enabled
            and self._idle_popularity_no_rsu_service_continuity_enabled
            and fallback_reason
            in {
                "no_associated_rsu_coverage_recovery_prepare",
                "no_associated_rsu_service_continuity",
            }
            and int(fallback_action) in {0, 1, 3, 4}
            and int(fallback_action) != int(original_env_action)
        )
        if (
            self._idle_popularity_fallback_only_vehicle_fallback
            and int(original_env_action) != 2
            and not no_rsu_local_override
            and not no_rsu_service_override
        ):
            return {
                "enabled": True,
                "applied": False,
                "reason": "original_action_not_vehicle_fallback",
                "original_action": int(original_env_action),
                "candidate_action": int(fallback_action),
                "candidate_reason": fallback_reason,
                **fallback_extra,
            }
        if int(fallback_action) == int(original_env_action):
            return {
                "enabled": True,
                "applied": False,
                "reason": "candidate_same_as_original",
                "original_action": int(original_env_action),
                "candidate_action": int(fallback_action),
                "candidate_reason": fallback_reason,
                **fallback_extra,
            }
        return {
            "enabled": True,
            "applied": True,
            "reason": (
                "no_rsu_current_offload_replaced_by_local"
                if no_rsu_local_override
                else (
                    "no_rsu_service_continuity_replaced_by_popularity_option"
                    if no_rsu_service_override
                    else "vehicle_fallback_replaced_by_popularity_option"
                )
            ),
            "original_action": int(original_env_action),
            "fallback_action": int(fallback_action),
            "candidate_reason": fallback_reason,
            **fallback_extra,
        }

    def _mechanism_prepare_candidate_action(
        self,
        semantic_state: dict[str, Any],
        action_mask: list[bool] | None,
        base_env_action: int,
    ) -> tuple[int, str, bool]:
        vehicle = self._primary_vehicle_for_popularity(semantic_state)
        vehicle_id = str(vehicle.get("vehicle_id", "")) if vehicle else None
        current_rsu_id = vehicle.get("associated_rsu_id")
        predicted_next_rsu_id, predicted_handoff_target = self._prediction_targets_for_popularity(
            semantic_state,
            vehicle_id,
        )
        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        if predicted_handoff_target and predicted_handoff_target != current_rsu_id:
            return self._select_allowed_env_action([4, 1, 3], action_mask), "handoff_target_prepare", True
        if (
            predicted_next_rsu_id
            and predicted_next_rsu_id != current_rsu_id
            and required_adapter
            and not self._adapter_cached_for_popularity(semantic_state, predicted_next_rsu_id, required_adapter)
            ):
            return self._select_allowed_env_action([1, 4, 3], action_mask), "next_rsu_prefetch_prepare", True
        return int(base_env_action), "no_mechanism_candidate", False

    def _sparse_handoff_option_prior_context(
        self,
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_enabled = bool(
            self._sparse_handoff_option_prior_enabled
            or self._option_counterfactual_critic_enabled
        )
        if not (self._use_hierarchy and context_enabled):
            return {"enabled": context_enabled, "active": False}

        run_metadata = dict(run_metadata or {})
        window_class = str(
            semantic_state.get("window_class")
            or (semantic_state.get("run_info", {}) or {}).get("window_class")
            or run_metadata.get("window_class", "")
        )
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = _clamp01(float(timing_features.get("prepare_window_score", 0.0) or 0.0))
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        predicted_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        diagnostics = self._build_prediction_target_diagnostics(
            semantic_state=semantic_state,
            temporal_urgency=temporal_urgency,
            predicted_handoff_target_valid=predicted_target_valid,
        )
        confidence = _clamp01(float(diagnostics.get("prediction_confidence", 0.0) or 0.0))
        uncertainty = _clamp01(float(diagnostics.get("prediction_uncertainty", 1.0) or 1.0))
        reliability = _clamp01(confidence * (1.0 - uncertainty))
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        target_rsu_id = diagnostics.get("predicted_first_non_current_rsu")
        predictions = semantic_state.get("predictions", {})
        if isinstance(predictions, dict) and vehicle_id:
            if target_rsu_id is None:
                target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
            if target_rsu_id is None:
                target_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
            if target_rsu_id is None:
                next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
                target_rsu_id = next(
                    (item for item in next_sequence if item is not None and str(item) != str(current_rsu_id)),
                    None,
                )

        first_eta = int(diagnostics.get("predicted_first_non_current_eta", 0) or 0)
        target_differs = bool(target_rsu_id is not None and str(target_rsu_id) != str(current_rsu_id))
        lead_threshold = max(int(math.ceil(self._temporal_prepare_lead_steps)) + 1, 3)
        if first_eta <= 0 and target_differs:
            first_eta = lead_threshold
        max_eta = max(int(self._sparse_handoff_option_max_eta), 1)
        eta_support = 1.0
        if first_eta > 0:
            eta_support = _clamp01(1.0 - (float(first_eta) - 1.0) / float(max_eta))
        raw_candidate = bool(diagnostics.get("raw_handoff_candidate", False))
        sequence_contains_other = bool(diagnostics.get("predicted_sequence_contains_other_rsu", False))
        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        context = _clamp01(
            0.30 * reliability
            + 0.20 * prepare_score
            + 0.18 * temporal_urgency
            + 0.14 * eta_support
            + 0.10 * float(raw_candidate or sequence_contains_other or predicted_target_valid)
            + 0.08 * float(window_class == "idle_or_sparse")
        )
        active = bool(
            window_class == "idle_or_sparse"
            and required_adapter
            and target_differs
            and (raw_candidate or sequence_contains_other or predicted_target_valid)
            and 0 < first_eta <= max_eta
            and context + 1e-8 >= self._sparse_handoff_option_min_context
        )
        return {
            "enabled": True,
            "active": bool(active),
            "window_class": window_class,
            "context": round(float(context), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "prediction_reliability": round(float(reliability), 6),
            "predicted_first_non_current_eta": int(first_eta),
            "target_rsu_id": target_rsu_id,
            "current_rsu_id": current_rsu_id,
            "required_adapter": str(required_adapter) if required_adapter else None,
            "raw_handoff_candidate": bool(raw_candidate),
            "predicted_target_valid": bool(predicted_target_valid),
        }

    def _build_option_gate_candidates(
        self,
        *,
        semantic_state: dict[str, Any],
        action_mask: list[bool] | None,
        base_env_action: int,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_action = self._select_allowed_env_action([int(base_env_action), 3, 2, 0], action_mask)
        popularity_action, popularity_reason, popularity_extra = self._idle_popularity_candidate_action(
            semantic_state,
            action_mask,
        )
        vehicle = self._primary_vehicle_for_popularity(semantic_state)
        vehicle_id = str(vehicle.get("vehicle_id", "")) if vehicle else None
        current_rsu_id = vehicle.get("associated_rsu_id") if vehicle else None
        predicted_next_rsu_id, predicted_handoff_target = self._prediction_targets_for_popularity(
            semantic_state,
            vehicle_id,
        )
        no_rsu_available = bool(current_rsu_id is None and self._is_env_action_valid(2, action_mask))
        mechanism_action, mechanism_reason, mechanism_available = self._mechanism_prepare_candidate_action(
            semantic_state,
            action_mask,
            base_action,
        )
        coverage_recovery_no_rsu = bool(
            no_rsu_available
            and mechanism_available
            and self._net_advantage_prepare_gate_enabled
            and int(mechanism_action) in {1, 4}
        )
        if coverage_recovery_no_rsu:
            no_rsu_action = int(mechanism_action)
        elif no_rsu_available and self._idle_popularity_no_rsu_service_continuity_enabled:
            no_rsu_action = self._select_allowed_env_action([3, 2, 0], action_mask)
        elif no_rsu_available:
            no_rsu_action = 2
        else:
            no_rsu_action = base_action
        window_class = str((run_metadata or {}).get("window_class", "unknown"))
        idle_recovery_context = False
        if (
            self._option_gate_idle_recovery_mechanism_prior_enabled
            and window_class == "idle_or_sparse"
        ):
            timing_features = compute_temporal_prepare_window_score(
                semantic_state,
                preferred_lead_steps=self._temporal_prepare_lead_steps,
                sigma=self._temporal_prepare_sigma,
            )
            timing_support = max(
                float(timing_features.get("prepare_window_score", 0.0) or 0.0),
                float(timing_features.get("temporal_urgency", 0.0) or 0.0),
            )
            predicted_recovery = bool(
                predicted_handoff_target
                or (
                    predicted_next_rsu_id is not None
                    and str(predicted_next_rsu_id) != str(current_rsu_id)
                )
            )
            idle_recovery_context = bool(
                mechanism_available
                and int(mechanism_action) in {1, 4}
                and (
                    no_rsu_available
                    or predicted_recovery
                    or timing_support >= self._option_gate_idle_recovery_min_context
                )
            )
        sparse_option_context = self._sparse_handoff_option_prior_context(
            semantic_state,
            run_metadata=run_metadata,
        )
        if bool(sparse_option_context.get("active", False)):
            idle_recovery_context = True
            if not mechanism_available or int(mechanism_action) not in {1, 4}:
                mechanism_action = self._select_allowed_env_action([4, 1, 3], action_mask)
                mechanism_reason = "sparse_tail_risk_option_prepare"
                mechanism_available = bool(int(mechanism_action) in {1, 4})
        if self._option_gate_context_prior_enabled and window_class in {"idle_or_sparse", "active_non_mechanism"}:
            if not idle_recovery_context:
                mechanism_action = int(base_action)
                mechanism_reason = f"context_suppressed_for_{window_class}"
                mechanism_available = False
        option_actions = {
            0: int(base_action),
            1: int(popularity_action),
            2: int(no_rsu_action),
            3: int(mechanism_action),
        }
        option_mask = [True, True, bool(no_rsu_available), bool(mechanism_available)]
        if not self._is_env_action_valid(popularity_action, action_mask):
            option_mask[1] = False
        if not self._is_env_action_valid(mechanism_action, action_mask):
            option_mask[3] = False
        if len(option_mask) < self._option_gate_count:
            option_mask.extend(False for _ in range(self._option_gate_count - len(option_mask)))
        option_mask = option_mask[: self._option_gate_count]
        if not any(option_mask):
            option_mask[0] = True
        prior_target = self._option_gate_prior_target(
            option_actions=option_actions,
            option_mask=option_mask,
            popularity_reason=popularity_reason,
            popularity_extra=popularity_extra,
            no_rsu_available=no_rsu_available,
            mechanism_available=mechanism_available,
            idle_recovery_context=idle_recovery_context,
            run_metadata=run_metadata,
        )
        return {
            "option_actions": option_actions,
            "option_mask": option_mask,
            "prior_target": int(prior_target),
            "popularity_reason": popularity_reason,
            "mechanism_reason": mechanism_reason,
            "popularity_extra": popularity_extra,
            "no_rsu_available": bool(no_rsu_available),
            "coverage_recovery_no_rsu": bool(coverage_recovery_no_rsu),
            "mechanism_available": bool(mechanism_available),
            "idle_recovery_context": bool(idle_recovery_context),
            "sparse_tail_risk_option_context": dict(sparse_option_context),
            "window_class": window_class,
        }

    def _build_dag_opportunity_summary(self, semantic_state: dict[str, Any]) -> dict[str, float | int]:
        workflow = semantic_state.get("workflow", {})
        current_node = semantic_state.get("current_workflow_node") or {}
        nodes = list(workflow.get("nodes", []))
        node_map = {str(node.get("node_id")): node for node in nodes if node.get("node_id") is not None}
        completed_node_ids = {str(node_id) for node_id in workflow.get("completed_node_ids", [])}
        remaining_node_ids = [node_id for node_id in node_map if node_id not in completed_node_ids]
        frontier_node_ids = [
            node_id
            for node_id in remaining_node_ids
            if {str(item) for item in node_map[node_id].get("predecessors", [])}.issubset(completed_node_ids)
        ]

        longest_remaining_path_cache: dict[str, int] = {}

        def remaining_path_length(node_id: str | None) -> int:
            if not node_id or node_id not in node_map or node_id in completed_node_ids:
                return 0
            if node_id in longest_remaining_path_cache:
                return longest_remaining_path_cache[node_id]
            successors = [
                str(successor_id)
                for successor_id in node_map[node_id].get("successors", [])
                if str(successor_id) in node_map and str(successor_id) not in completed_node_ids
            ]
            best_successor_length = max((remaining_path_length(successor_id) for successor_id in successors), default=0)
            longest_remaining_path_cache[node_id] = 1 + best_successor_length
            return longest_remaining_path_cache[node_id]

        current_node_id = str(workflow.get("current_node_id") or current_node.get("node_id") or "")
        current_path_length = remaining_path_length(current_node_id)
        critical_path_length = max(
            (remaining_path_length(node_id) for node_id in frontier_node_ids),
            default=current_path_length,
        )
        current_node_record = node_map.get(current_node_id, current_node if isinstance(current_node, dict) else {})
        current_successor_count = len(list(current_node_record.get("successors", [])))
        node_count = len(node_map)
        continuity_features = build_graph_continuity_critic_features(
            semantic_state,
            prediction_gate_min_leak=self._prediction_gate_min_leak,
        )
        return {
            "node_count": int(node_count),
            "remaining_node_count": int(len(remaining_node_ids)),
            "frontier_width": int(len(frontier_node_ids)),
            "current_path_length": int(current_path_length),
            "critical_path_length": int(critical_path_length),
            "current_successor_count": int(current_successor_count),
            "remaining_nodes_ratio": float(continuity_features.get("remaining_nodes_ratio", 0.0)),
            "critical_path_length_norm": float(continuity_features.get("critical_path_length_norm", 0.0)),
            "predicted_path_switch_ratio": float(continuity_features.get("predicted_path_switch_ratio", 0.0)),
            "prediction_confidence": float(continuity_features.get("prediction_confidence", 0.0)),
            "prediction_reliability": float(continuity_features.get("prediction_reliability", 0.0)),
            "reliability_timing_alignment": float(
                continuity_features.get("reliability_timing_alignment", 0.0)
            ),
        }

    def _dag_aware_option_termination_reason(
        self,
        *,
        semantic_state: dict[str, Any],
        candidate_info: dict[str, Any],
        base_env_action: int,
        window_class: str,
    ) -> str | None:
        if not (
            self._dag_aware_option_termination_enabled
            and int(base_env_action) == 1
            and len(candidate_info["option_mask"]) > 1
            and bool(candidate_info["option_mask"][1])
            and int(candidate_info["option_actions"].get(1, int(base_env_action))) != int(base_env_action)
        ):
            return None

        dag_summary = self._build_dag_opportunity_summary(semantic_state)
        if (
            window_class == "idle_or_sparse"
            and str(candidate_info.get("popularity_reason", "")) == "no_associated_rsu_vehicle_fallback"
            and float(dag_summary.get("prediction_confidence", 0.0))
            <= self._dag_aware_idle_prefetch_confidence_floor
        ):
            return "dag_aware_idle_low_confidence_prefetch_termination"

        if window_class != "mechanism_activating":
            return None
        if int(dag_summary.get("node_count", 0)) > self._dag_aware_option_short_workflow_max_nodes:
            return None
        if int(dag_summary.get("critical_path_length", 0)) >= self._dag_aware_option_min_critical_path:
            return None
        if int(dag_summary.get("current_successor_count", 0)) < self._dag_aware_option_branching_successors:
            return None
        return "dag_aware_short_dag_prefetch_termination"

    def _should_apply_net_utility_option_termination(
        self,
        *,
        semantic_state: dict[str, Any],
        candidate_info: dict[str, Any],
        base_env_action: int,
        window_class: str,
    ) -> bool:
        if not (
            self._net_utility_option_termination_enabled
            and window_class == "idle_or_sparse"
            and int(base_env_action) == 1
            and str(candidate_info.get("popularity_reason", ""))
            == "no_associated_rsu_vehicle_fallback"
            and len(candidate_info["option_mask"]) > 1
            and bool(candidate_info["option_mask"][1])
            and int(candidate_info["option_actions"].get(1, int(base_env_action))) != int(base_env_action)
        ):
            return False
        if not self._net_utility_option_termination_conservative_enabled:
            return True
        popularity_extra = candidate_info.get("popularity_extra", {})
        if not bool(popularity_extra.get("low_mechanism_no_rsu_context", False)):
            return False
        if self._semantic_state_has_raw_handoff_candidate(semantic_state):
            return False
        if self._semantic_state_has_valid_predicted_handoff_target(semantic_state):
            return False
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        timing_support = max(
            float(timing_features.get("prepare_window_score", 0.0)),
            float(timing_features.get("temporal_urgency", 0.0)),
        )
        return bool(timing_support <= self._net_utility_option_termination_max_timing_support)

    def _option_gate_prior_target(
        self,
        *,
        option_actions: dict[int, int],
        option_mask: list[bool],
        popularity_reason: str,
        popularity_extra: dict[str, Any],
        no_rsu_available: bool,
        mechanism_available: bool,
        idle_recovery_context: bool = False,
        run_metadata: dict[str, Any] | None = None,
    ) -> int:
        window_class = str((run_metadata or {}).get("window_class", "unknown"))
        if (
            self._option_gate_idle_recovery_mechanism_prior_enabled
            and window_class == "idle_or_sparse"
            and idle_recovery_context
            and mechanism_available
            and len(option_mask) > 3
            and option_mask[3]
        ):
            return 3
        if (
            self._idle_popularity_no_rsu_service_continuity_enabled
            and no_rsu_available
            and len(option_mask) > 2
            and option_mask[2]
            and int(option_actions.get(2, 2)) in {0, 1, 3, 4}
        ):
            return 2
        if (
            self._option_gate_idle_prior_enabled
            and window_class == "idle_or_sparse"
            and len(option_mask) > 1
            and option_mask[1]
            and int(option_actions.get(1, option_actions.get(0, 3))) != int(option_actions.get(0, 3))
        ):
            return 1
        if mechanism_available and len(option_mask) > 3 and option_mask[3]:
            return 3
        if (
            no_rsu_available
            and len(option_mask) > 2
            and option_mask[2]
            and bool(popularity_extra.get("low_mechanism_no_rsu_context", False))
        ):
            return 2
        if (
            len(option_mask) > 1
            and option_mask[1]
            and int(option_actions.get(1, option_actions.get(0, 3))) != int(option_actions.get(0, 3))
            and popularity_reason
            in {
                "popular_adapter_reactive_cache_fill",
                "popular_adapter_predictive_prefetch",
                "predicted_handoff_migration_prepare",
                "popularity_steady_offload",
            }
        ):
            return 1
        return 0

    def _masked_option_logits(
        self,
        logits: torch.Tensor,
        option_mask: list[bool] | None,
        prior_target: int | None = None,
    ) -> torch.Tensor:
        adjusted = logits
        if self._option_gate_prior_logit_bias > 0.0 and prior_target is not None:
            adjusted = adjusted.clone()
            if 0 <= int(prior_target) < adjusted.shape[-1]:
                adjusted[int(prior_target)] = adjusted[int(prior_target)] + self._option_gate_prior_logit_bias
        if option_mask is None:
            return adjusted
        mask_tensor = torch.as_tensor(list(option_mask), dtype=torch.bool, device=adjusted.device)
        if mask_tensor.numel() != adjusted.shape[-1] or not bool(mask_tensor.any().item()):
            return adjusted
        return adjusted.masked_fill(~mask_tensor, -1.0e9)

    def _critic_improved_option_logits(
        self,
        *,
        option_logits: torch.Tensor,
        option_q_values: torch.Tensor | None,
        option_mask: list[bool] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_advantage = torch.zeros_like(option_logits)
        if (
            not self._option_counterfactual_policy_improvement_enabled
            or self._option_counterfactual_policy_improvement_coef <= 0.0
            or self._update_count < self._option_counterfactual_warmup_updates
            or option_q_values is None
            or option_q_values.shape != option_logits.shape
        ):
            return option_logits, zero_advantage

        valid_mask = torch.ones_like(option_logits, dtype=torch.bool)
        if option_mask and len(option_mask) == int(option_logits.shape[-1]):
            valid_mask = torch.as_tensor(
                option_mask,
                dtype=torch.bool,
                device=option_logits.device,
            )
        if not bool(valid_mask.any().item()):
            return option_logits, zero_advantage

        detached_q = option_q_values.detach()
        actor_probs = torch.softmax(option_logits, dim=-1).detach()
        actor_probs = actor_probs.masked_fill(~valid_mask, 0.0)
        actor_probs = actor_probs / actor_probs.sum().clamp_min(1e-8)
        q_baseline = torch.sum(actor_probs * detached_q)
        q_advantage = (detached_q - q_baseline).masked_fill(~valid_mask, 0.0)
        valid_advantages = q_advantage[valid_mask]
        q_scale = torch.sqrt(torch.mean(valid_advantages.square())).clamp_min(1e-6)
        normalized_advantage = (q_advantage / q_scale).masked_fill(~valid_mask, 0.0)
        if self._option_counterfactual_policy_improvement_clip > 0.0:
            normalized_advantage = torch.clamp(
                normalized_advantage,
                -self._option_counterfactual_policy_improvement_clip,
                self._option_counterfactual_policy_improvement_clip,
            )
        improved_logits = (
            option_logits
            + self._option_counterfactual_policy_improvement_coef
            * normalized_advantage
        )
        return improved_logits, normalized_advantage

    def _maybe_apply_option_gate(
        self,
        *,
        semantic_state: dict[str, Any],
        action_mask: list[bool] | None,
        policy_output: dict[str, Any],
        base_env_action: int,
        deterministic: bool,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._option_gate_enabled or "option_logits" not in policy_output:
            return {"enabled": False, "applied": False}
        window_class = str((run_metadata or {}).get("window_class", "unknown"))
        candidate_info: dict[str, Any] | None = None
        forced_selection_reason: str | None = None
        if (
            self._option_gate_context_prior_enabled
            and self._option_gate_mechanism_preserve_enabled
            and window_class == "mechanism_activating"
        ):
            candidate_info = self._build_option_gate_candidates(
                semantic_state=semantic_state,
                action_mask=action_mask,
                base_env_action=base_env_action,
                run_metadata=run_metadata,
            )
            forced_selection_reason = self._dag_aware_option_termination_reason(
                semantic_state=semantic_state,
                candidate_info=candidate_info,
                base_env_action=base_env_action,
                window_class=window_class,
            )
            if forced_selection_reason is None:
                return {
                    "enabled": False,
                    "applied": False,
                    "reason": "mechanism_window_preserve_mappo",
                    "base_env_action": int(base_env_action),
                    "window_class": window_class,
                }
        if candidate_info is None:
            candidate_info = self._build_option_gate_candidates(
                semantic_state=semantic_state,
                action_mask=action_mask,
                base_env_action=base_env_action,
                run_metadata=run_metadata,
            )
        prior_target = int(candidate_info["prior_target"])
        actor_option_logits = self._masked_option_logits(
            policy_output["option_logits"],
            list(candidate_info["option_mask"]),
            prior_target=prior_target,
        )
        if deterministic or not self._option_counterfactual_policy_improvement_deterministic_only:
            option_logits, critic_option_advantage = self._critic_improved_option_logits(
                option_logits=actor_option_logits,
                option_q_values=policy_output.get("option_q_values"),
                option_mask=list(candidate_info["option_mask"]),
            )
        else:
            option_logits = actor_option_logits
            critic_option_advantage = torch.zeros_like(actor_option_logits)
        distribution = Categorical(logits=option_logits)
        option_probs = torch.softmax(option_logits, dim=-1)
        top_tensor = torch.argmax(option_logits, dim=-1)
        top_action = int(top_tensor.item())
        selection_reason = "policy_argmax" if deterministic else "policy_sample"
        if deterministic:
            option_tensor = top_tensor
            if forced_selection_reason is not None:
                option_tensor = torch.tensor(1, dtype=torch.long, device=option_logits.device)
                selection_reason = forced_selection_reason
            elif self._should_apply_net_utility_option_termination(
                semantic_state=semantic_state,
                candidate_info=candidate_info,
                base_env_action=base_env_action,
                window_class=window_class,
            ):
                option_tensor = torch.tensor(1, dtype=torch.long, device=option_logits.device)
                selection_reason = (
                    "net_utility_conservative_idle_prefetch_termination"
                    if self._net_utility_option_termination_conservative_enabled
                    else "net_utility_idle_prefetch_termination"
                )
            else:
                dag_selection_reason = self._dag_aware_option_termination_reason(
                    semantic_state=semantic_state,
                    candidate_info=candidate_info,
                    base_env_action=base_env_action,
                    window_class=window_class,
                )
                if dag_selection_reason is not None:
                    option_tensor = torch.tensor(1, dtype=torch.long, device=option_logits.device)
                    selection_reason = dag_selection_reason
            if (
                self._option_gate_context_prior_enabled
                and 0 <= prior_target < int(option_logits.shape[-1])
                and prior_target < len(candidate_info["option_mask"])
                and bool(candidate_info["option_mask"][prior_target])
                and selection_reason == "policy_argmax"
            ):
                top_prob = float(option_probs[top_action].item())
                prior_prob = float(option_probs[prior_target].item())
                if top_action != prior_target and top_prob - prior_prob < self._option_gate_deterministic_prior_margin:
                    option_tensor = torch.tensor(prior_target, dtype=torch.long, device=option_logits.device)
                    selection_reason = "context_prior_margin"
        else:
            option_tensor = distribution.sample()
        option_action = int(option_tensor.item())
        option_env_action = int(candidate_info["option_actions"].get(option_action, int(base_env_action)))
        option_label = OPTION_GATE_LABELS.get(option_action, f"option_{option_action}")
        return {
            "enabled": True,
            "applied": bool(option_env_action != int(base_env_action)),
            "option_action": int(option_action),
            "option_label": option_label,
            "option_env_action": option_env_action,
            "base_env_action": int(base_env_action),
            "option_mask": list(candidate_info["option_mask"]),
            "option_actions": {str(key): int(value) for key, value in candidate_info["option_actions"].items()},
            "prior_target": prior_target,
            "prior_label": OPTION_GATE_LABELS.get(prior_target, f"option_{prior_target}"),
            "selection_reason": selection_reason,
            "top_option_action": top_action,
            "top_option_label": OPTION_GATE_LABELS.get(top_action, f"option_{top_action}"),
            "popularity_reason": str(candidate_info["popularity_reason"]),
            "mechanism_reason": str(candidate_info["mechanism_reason"]),
            "no_rsu_available": bool(candidate_info["no_rsu_available"]),
            "mechanism_available": bool(candidate_info["mechanism_available"]),
            "idle_recovery_context": bool(candidate_info.get("idle_recovery_context", False)),
            "sparse_tail_risk_option_context": dict(
                candidate_info.get("sparse_tail_risk_option_context", {})
            ),
            "window_class": str(candidate_info.get("window_class", "unknown")),
            "option_probs": [round(float(item), 6) for item in option_probs.tolist()],
            "top_option_prob": round(float(option_probs[top_action].item()), 6),
            "prior_option_prob": round(
                float(option_probs[prior_target].item()) if 0 <= prior_target < int(option_probs.shape[-1]) else 0.0,
                6,
            ),
            "base_option_prob": round(float(option_probs[0].item()) if int(option_probs.shape[-1]) > 0 else 0.0, 6),
            "option_log_prob": round(float(distribution.log_prob(option_tensor).item()), 6),
            "option_entropy": round(float(distribution.entropy().item()), 6),
            "critic_policy_improvement_enabled": bool(
                self._option_counterfactual_policy_improvement_enabled
            ),
            "critic_policy_improvement_deterministic_only": bool(
                self._option_counterfactual_policy_improvement_deterministic_only
            ),
            "counterfactual_model_rollout_enabled": bool(
                self._option_counterfactual_model_rollout_enabled
            ),
            "counterfactual_model_rollout_horizon": int(
                self._option_counterfactual_model_rollout_horizon
            ),
            "critic_option_advantages": [
                round(float(item), 6)
                for item in critic_option_advantage.tolist()
            ],
            "_option_log_prob_tensor": distribution.log_prob(option_tensor),
            "_option_entropy_tensor": distribution.entropy(),
        }

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
        scores = torch.stack(
            [
                event_log_probs[0] + slow_log_probs[1],
                event_log_probs[0] + slow_log_probs[2],
                event_log_probs[0] + slow_log_probs[0] + fast_log_probs[1],
                event_log_probs[0] + slow_log_probs[0] + fast_log_probs[0],
                event_log_probs[1],
            ],
            dim=0,
        )
        env_action_bias = policy_output.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == scores.numel():
            scores = scores + env_action_bias.to(device=scores.device, dtype=scores.dtype)
        return scores

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

    def _forward_policy(
        self,
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy_output = self._network.forward_single(
            semantic_state,
            event_logit_temperature=self._current_event_logit_temperature(),
        )
        if self._is_raw_policy_evaluation(run_metadata):
            return dict(policy_output)
        return self._apply_policy_adjustments(
            policy_output,
            semantic_state,
            run_metadata=run_metadata,
        )

    @staticmethod
    def _is_raw_policy_evaluation(run_metadata: dict[str, Any] | None) -> bool:
        return str(
            (run_metadata or {}).get("policy_evaluation_mode", "safety_projected")
        ).strip().lower() == "raw_policy"

    def _env_action_distribution_statistics(
        self,
        *,
        policy_output: dict[str, Any],
        env_action: int,
        action_mask: list[bool] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        action_tensor = torch.tensor(int(env_action), dtype=torch.long, device=self._device)
        if not self._use_hierarchy:
            logits = self._masked_flat_logits(policy_output["flat_logits"], action_mask)
        else:
            logits = self._masked_flat_logits(
                self._hierarchical_env_action_scores(policy_output),
                action_mask,
            )
        distribution = Categorical(logits=logits)
        return (
            distribution.log_prob(action_tensor),
            distribution.entropy(),
            [round(float(item), 6) for item in torch.softmax(logits, dim=-1).tolist()],
        )

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

    def _apply_coverage_recovery_guard_to_actions(
        self,
        *,
        semantic_state: dict[str, Any],
        policy_output: dict[str, Any],
        selected_actions: dict[str, int],
        action_mask: list[bool] | None,
    ) -> dict[str, Any]:
        if not (self._coverage_recovery_guard_enabled and self._use_hierarchy):
            return {"enabled": False, "guarded": False, "reason": "disabled"}
        gate_info = dict(policy_output.get("net_advantage_prepare_gate_info", {}))
        if not gate_info:
            return {"enabled": True, "guarded": False, "reason": "missing_net_advantage_gate"}
        recovery_context = bool(
            gate_info.get("current_rsu_id") is None
            and (
                gate_info.get("predicted_target_valid", False)
                or gate_info.get("target_differs", False)
            )
        )
        recovery_scale = _clamp01(float(gate_info.get("coverage_recovery_scale", 0.0) or 0.0))
        if not recovery_context:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "not_coverage_recovery_context",
                "coverage_recovery_scale": round(float(recovery_scale), 6),
            }
        min_scale = max(self._coverage_recovery_gate_min_scale, 0.20)
        if recovery_scale + 1e-8 < min_scale:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "coverage_recovery_scale_below_min",
                "coverage_recovery_scale": round(float(recovery_scale), 6),
                "min_scale": round(float(min_scale), 6),
            }
        target_action = 4
        if not self._is_env_action_valid(target_action, action_mask):
            return {
                "enabled": True,
                "guarded": False,
                "reason": "target_action_invalid",
                "target_action": int(target_action),
                "coverage_recovery_scale": round(float(recovery_scale), 6),
            }
        original_action, original_reason = 聚合层级动作(
            head_actions=selected_actions,
            use_hierarchy=self._use_hierarchy,
            event_head_enabled=self._event_head_enabled,
            adapter_prefetch_enabled=self._adapter_prefetch_enabled,
        )
        if int(original_action) == target_action:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "already_prepare",
                "original_action": int(original_action),
                "coverage_recovery_scale": round(float(recovery_scale), 6),
            }
        return {
            "enabled": True,
            "guarded": True,
            "reason": "coverage_recovery_prefers_prepare",
            "original_action": int(original_action),
            "original_aggregation_reason": original_reason,
            "guarded_action": int(target_action),
            "coverage_recovery_scale": round(float(recovery_scale), 6),
            "predicted_handoff_target_rsu_id": gate_info.get("predicted_handoff_target_rsu_id"),
            "predicted_next_rsu_id": gate_info.get("predicted_next_rsu_id"),
        }

    def _apply_coverage_recovery_final_guard_to_env_action(
        self,
        *,
        semantic_state: dict[str, Any],
        policy_output: dict[str, Any],
        env_action: int,
        action_mask: list[bool] | None,
        option_gate_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del semantic_state
        if not (self._coverage_recovery_final_guard_enabled and self._use_hierarchy):
            return {"enabled": False, "guarded": False, "reason": "disabled"}
        gate_info = dict(policy_output.get("net_advantage_prepare_gate_info", {}))
        if not gate_info:
            return {"enabled": True, "guarded": False, "reason": "missing_net_advantage_gate"}
        recovery_context = bool(
            gate_info.get("current_rsu_id") is None
            and (
                gate_info.get("predicted_target_valid", False)
                or gate_info.get("target_differs", False)
            )
        )
        recovery_scale = _clamp01(
            max(
                float(gate_info.get("coverage_recovery_scale", 0.0) or 0.0),
                float(gate_info.get("net_advantage_score", 0.0) or 0.0),
            )
        )
        confidence = _clamp01(float(gate_info.get("prediction_confidence", 0.0) or 0.0))
        if not recovery_context:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "not_coverage_recovery_context",
                "coverage_recovery_scale": round(float(recovery_scale), 6),
                "prediction_confidence": round(float(confidence), 6),
            }
        target_action = 4
        if int(env_action) == target_action:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "already_prepare",
                "original_action": int(env_action),
                "coverage_recovery_scale": round(float(recovery_scale), 6),
                "prediction_confidence": round(float(confidence), 6),
            }
        if not self._is_env_action_valid(target_action, action_mask):
            return {
                "enabled": True,
                "guarded": False,
                "reason": "target_action_invalid",
                "target_action": int(target_action),
                "original_action": int(env_action),
                "coverage_recovery_scale": round(float(recovery_scale), 6),
                "prediction_confidence": round(float(confidence), 6),
            }
        min_scale = max(self._coverage_recovery_final_guard_min_scale, 0.0)
        min_confidence = max(self._coverage_recovery_final_guard_min_confidence, 0.0)
        if recovery_scale + 1e-8 < min_scale and confidence + 1e-8 < min_confidence:
            return {
                "enabled": True,
                "guarded": False,
                "reason": "coverage_recovery_memory_below_min",
                "original_action": int(env_action),
                "coverage_recovery_scale": round(float(recovery_scale), 6),
                "min_scale": round(float(min_scale), 6),
                "prediction_confidence": round(float(confidence), 6),
                "min_confidence": round(float(min_confidence), 6),
            }
        option_info = dict(option_gate_info or {})
        return {
            "enabled": True,
            "guarded": True,
            "reason": "partial_observation_handoff_target_memory_prefers_prepare",
            "original_action": int(env_action),
            "guarded_action": int(target_action),
            "coverage_recovery_scale": round(float(recovery_scale), 6),
            "min_scale": round(float(min_scale), 6),
            "prediction_confidence": round(float(confidence), 6),
            "min_confidence": round(float(min_confidence), 6),
            "predicted_handoff_target_rsu_id": gate_info.get("predicted_handoff_target_rsu_id"),
            "predicted_next_rsu_id": gate_info.get("predicted_next_rsu_id"),
            "option_label": option_info.get("option_label"),
            "option_env_action": option_info.get("option_env_action"),
        }

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

    def _compute_env_action_ppo_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
        batch_actions: torch.Tensor,
        old_env_action_log_probs: torch.Tensor,
        base_advantage: torch.Tensor,
        event_advantage: torch.Tensor,
        batch_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        if (
            not self._env_action_ppo_enabled
            or self._env_action_ppo_coef <= 0.0
            or not self._use_hierarchy
            or not batch_outputs
        ):
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        env_scores = torch.stack(
            [
                self._masked_flat_logits(
                    self._hierarchical_env_action_scores(output),
                    action_mask,
                )
                for output, action_mask in zip(batch_outputs, batch_action_masks)
            ],
            dim=0,
        )
        distribution = Categorical(logits=env_scores)
        new_log_probs = distribution.log_prob(batch_actions)
        blend = self._env_action_ppo_advantage_blend
        action_advantage = blend * base_advantage + (1.0 - blend) * event_advantage
        if (
            self._env_action_model_critic_enabled
            and self._env_action_model_critic_advantage_coef > 0.0
            and self._update_count >= self._env_action_model_critic_warmup_updates
        ):
            model_advantages: list[torch.Tensor] = []
            for row_index, (output, action_mask) in enumerate(
                zip(batch_outputs, batch_action_masks)
            ):
                q_values = output.get("env_action_q_values")
                if not isinstance(q_values, torch.Tensor):
                    model_advantages.append(
                        torch.tensor(0.0, dtype=torch.float32, device=self._device)
                    )
                    continue
                valid_mask = torch.ones_like(q_values, dtype=torch.bool)
                if action_mask and len(action_mask) == int(q_values.shape[-1]):
                    valid_mask = torch.as_tensor(
                        action_mask,
                        dtype=torch.bool,
                        device=self._device,
                    )
                detached_q = q_values.detach().masked_fill(~valid_mask, 0.0)
                detached_probs = distribution.probs[row_index].detach().masked_fill(
                    ~valid_mask,
                    0.0,
                )
                detached_probs = (
                    detached_probs / detached_probs.sum().clamp_min(1e-8)
                )
                baseline_q = torch.sum(detached_probs * detached_q)
                selected_advantage = (
                    detached_q[int(batch_actions[row_index].item())] - baseline_q
                )
                if self._env_action_model_critic_advantage_clip > 0.0:
                    selected_advantage = torch.clamp(
                        selected_advantage,
                        -self._env_action_model_critic_advantage_clip,
                        self._env_action_model_critic_advantage_clip,
                    )
                model_advantages.append(selected_advantage)
            action_advantage = (
                action_advantage
                + self._env_action_model_critic_advantage_coef
                * torch.stack(model_advantages)
            )
        if self._counterfactual_teacher_prd_enabled and self._env_action_ppo_teacher_coef > 0.0:
            teacher_values: list[float] = []
            for row, action in zip(batch_rows, batch_actions.detach().cpu().tolist()):
                teacher_credit = self._counterfactual_teacher_action_credit(row, int(action))
                if self._counterfactual_teacher_clip > 0.0:
                    teacher_credit = max(
                        -self._counterfactual_teacher_clip,
                        min(self._counterfactual_teacher_clip, teacher_credit),
                    )
                teacher_values.append(float(teacher_credit))
            teacher_tensor = torch.as_tensor(teacher_values, dtype=torch.float32, device=self._device)
            action_advantage = action_advantage + self._env_action_ppo_teacher_coef * teacher_tensor
        if self._env_action_risk_adjusted_recovery_coef > 0.0:
            risk_values = [
                self._env_action_recovery_risk_score(row, int(action))
                for row, action in zip(batch_rows, batch_actions.detach().cpu().tolist())
            ]
            risk_tensor = torch.as_tensor(risk_values, dtype=torch.float32, device=self._device)
            action_advantage = (
                action_advantage
                - self._env_action_risk_adjusted_recovery_coef * risk_tensor
            )
        sample_weights = torch.ones_like(action_advantage)
        if self._env_action_ppo_mechanism_focus > 0.0:
            focus_values: list[float] = []
            for row in batch_rows:
                metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
                action_info = dict(row.get("action_info", {}))
                final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
                high_risk = self._handoff_risk_context(row, metrics)
                window_class = self._row_window_class(row)
                mechanism_window = window_class == "mechanism_activating"
                idle_local_collapse = bool(
                    self._service_continuity_teacher_enabled
                    and window_class == "idle_or_sparse"
                    and (
                        final_action == 2
                        or self._metric_float(metrics, "local_exec_count") > 0.0
                    )
                )
                focus_values.append(float(high_risk or mechanism_window or idle_local_collapse))
            focus_tensor = torch.as_tensor(focus_values, dtype=torch.float32, device=self._device)
            sample_weights = sample_weights + self._env_action_ppo_mechanism_focus * focus_tensor
        if self._env_action_sparse_recovery_focus > 0.0:
            sparse_values: list[float] = []
            for row in batch_rows:
                metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
                action_info = dict(row.get("action_info", {}))
                sparse_values.append(
                    float(
                        self._row_sparse_recovery_context(
                            row,
                            metrics=metrics,
                            action_info=action_info,
                        )
                    )
                )
            sparse_tensor = torch.as_tensor(sparse_values, dtype=torch.float32, device=self._device)
            sample_weights = sample_weights + self._env_action_sparse_recovery_focus * sparse_tensor
        if self._env_action_risk_adjusted_recovery_coef > 0.0:
            risk_focus = torch.relu(-action_advantage.detach())
            sample_weights = sample_weights + self._env_action_risk_adjusted_recovery_coef * risk_focus
        if self._env_action_ppo_max_weight > 0.0:
            sample_weights = torch.clamp(sample_weights, max=self._env_action_ppo_max_weight)
        ratio = torch.exp(new_log_probs - old_env_action_log_probs)
        surrogate_1 = ratio * action_advantage
        surrogate_2 = torch.clamp(
            ratio,
            1.0 - self._clip_ratio,
            1.0 + self._clip_ratio,
        ) * action_advantage
        weighted_surrogate = torch.min(surrogate_1, surrogate_2) * sample_weights
        loss = -(weighted_surrogate.sum() / torch.clamp(sample_weights.sum(), min=1e-6))
        if self._env_action_ppo_ratio_barrier_coef > 0.0:
            lower = max(1e-6, 1.0 - self._env_action_ppo_ratio_barrier_margin)
            upper = 1.0 + self._env_action_ppo_ratio_barrier_margin
            positive_advantage = torch.relu(action_advantage.detach())
            negative_advantage = torch.relu(-action_advantage.detach())
            under_update_gap = torch.relu(lower - ratio) * positive_advantage
            over_update_gap = torch.relu(ratio - upper) * negative_advantage
            barrier = (under_update_gap.square() + over_update_gap.square()) * sample_weights
            loss = loss + self._env_action_ppo_ratio_barrier_coef * (
                barrier.sum() / torch.clamp(sample_weights.sum(), min=1e-6)
            )
        return loss

    def _compute_env_action_counterfactual_margin_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
        batch_rows: list[dict[str, Any]],
        base_advantage: torch.Tensor | None = None,
        event_advantage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not self._env_action_counterfactual_margin_enabled
            or self._env_action_counterfactual_margin_coef <= 0.0
            or not self._counterfactual_teacher_prd_enabled
            or not self._use_hierarchy
            or not batch_outputs
        ):
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)

        advantage_gate_values: torch.Tensor | None = None
        if base_advantage is not None:
            advantage_gate_values = base_advantage.detach()
            if event_advantage is not None:
                blend = self._env_action_counterfactual_margin_advantage_blend
                advantage_gate_values = (
                    blend * base_advantage.detach()
                    + (1.0 - blend) * event_advantage.detach()
                )

        loss_terms: list[torch.Tensor] = []
        for row_index, (policy_output, action_mask, row) in enumerate(
            zip(batch_outputs, batch_action_masks, batch_rows)
        ):
            if (
                advantage_gate_values is not None
                and self._env_action_counterfactual_margin_advantage_gate > 0.0
            ):
                advantage_value = float(advantage_gate_values[row_index].item())
                if advantage_value <= self._env_action_counterfactual_margin_advantage_gate:
                    continue

            valid_actions = [
                action_id
                for action_id in range(5)
                if self._is_env_action_valid(action_id, action_mask)
            ]
            if not valid_actions:
                continue
            credits = [
                self._counterfactual_teacher_action_credit(row, action_id)
                for action_id in valid_actions
            ]
            best_index = max(range(len(valid_actions)), key=lambda index: credits[index])
            best_action = int(valid_actions[best_index])
            best_credit = float(credits[best_index])
            if self._counterfactual_teacher_clip > 0.0:
                best_credit = max(
                    -self._counterfactual_teacher_clip,
                    min(self._counterfactual_teacher_clip, best_credit),
                )
            if best_credit <= 0.0:
                continue

            action_info = dict(row.get("action_info", {}))
            current_action = int(action_info.get("final_env_action", row.get("action", best_action)) or best_action)
            current_credit = self._counterfactual_teacher_action_credit(row, current_action)
            credit_gap = max(best_credit - float(current_credit), 0.0)
            weight = max(best_credit, credit_gap + self._env_action_counterfactual_margin_min_gap)
            if advantage_gate_values is not None:
                advantage_value = max(float(advantage_gate_values[row_index].item()), 0.0)
                weight *= 1.0 + min(advantage_value, 1.0)
            if self._env_action_sparse_recovery_focus > 0.0 and self._row_sparse_recovery_context(row):
                weight *= 1.0 + self._env_action_sparse_recovery_focus
            if self._env_action_risk_adjusted_recovery_coef > 0.0:
                best_risk = self._env_action_recovery_risk_score(row, best_action)
                current_risk = self._env_action_recovery_risk_score(row, current_action)
                if best_action in {1, 4} and best_risk > 0.0:
                    weight *= max(
                        self._env_action_risk_adjusted_recovery_floor,
                        1.0 - self._env_action_risk_adjusted_recovery_coef * best_risk,
                    )
                elif current_risk > best_risk:
                    weight *= 1.0 + 0.5 * self._env_action_risk_adjusted_recovery_coef * (
                        current_risk - best_risk
                    )
            if self._env_action_counterfactual_margin_max_weight > 0.0:
                weight = min(weight, self._env_action_counterfactual_margin_max_weight)
            if weight <= 1e-8:
                continue

            env_scores = self._masked_flat_logits(
                self._hierarchical_env_action_scores(policy_output),
                action_mask,
            )
            target = torch.tensor([best_action], dtype=torch.long, device=self._device)
            loss_terms.append(
                nn.functional.cross_entropy(env_scores.unsqueeze(0), target) * float(weight)
            )

        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

    def _compute_argmax_margin_regularization_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
        batch_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        if (
            not self._argmax_margin_regularization_enabled
            or self._argmax_margin_coef <= 0.0
            or not self._use_hierarchy
            or not batch_outputs
        ):
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)

        loss_terms: list[torch.Tensor] = []
        for policy_output, action_mask, row in zip(batch_outputs, batch_action_masks, batch_rows):
            valid_actions = [
                action_id
                for action_id in range(5)
                if self._is_env_action_valid(action_id, action_mask)
            ]
            service_actions = [action_id for action_id in valid_actions if action_id in {0, 2, 3}]
            mechanism_actions = [action_id for action_id in valid_actions if action_id in {1, 4}]
            if not service_actions or not mechanism_actions:
                continue

            metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
            action_info = dict(row.get("action_info", {}))
            final_action = int(action_info.get("final_env_action", row.get("action", 3)) or 3)
            window_class = self._row_window_class(row)
            mechanism_context = bool(
                window_class == "mechanism_activating"
                or self._handoff_risk_context(row, metrics)
                or self._row_sparse_recovery_context(
                    row,
                    metrics=metrics,
                    action_info=action_info,
                )
            )

            continuity = _clamp01(self._metric_float(metrics, "workflow_continuity_rate", 1.0))
            adapter_miss_pressure = self._env_action_adapter_miss_pressure(row, metrics=metrics)
            cache_miss_pressure = min(
                max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0) / 4.8,
                1.0,
            )
            delay_pressure = min(
                max(self._metric_float(metrics, "delay_penalty_sum"), 0.0) / 16.0,
                1.0,
            )
            stall_pressure = float(self._metric_bool(metrics, "stall_occurred"))
            failed_mechanism = float(self._row_failed_mechanism_attempt(row, metrics))
            mechanism_success = self._row_mechanism_success(metrics)
            tail_risk_pressure = _clamp01(
                0.36 * adapter_miss_pressure
                + 0.20 * cache_miss_pressure
                + 0.16 * delay_pressure
                + 0.14 * stall_pressure
                + 0.18 * max(0.0, 0.86 - continuity) / 0.86
                + 0.18 * failed_mechanism
            )
            protected_success = bool(
                mechanism_success
                and continuity >= 0.86
                and adapter_miss_pressure <= 1e-8
                and cache_miss_pressure <= 0.10
            )
            if protected_success:
                continue
            if (
                not mechanism_context
                and tail_risk_pressure + 1e-8 < self._argmax_margin_tail_risk_threshold
            ):
                continue

            credits = {
                action_id: self._counterfactual_teacher_action_credit(row, action_id)
                for action_id in valid_actions
            }
            target_action = max(
                service_actions,
                key=lambda action_id: (
                    float(credits.get(action_id, 0.0)),
                    1.0 if action_id in {0, 3} else 0.0,
                    -float(action_id),
                ),
            )
            target_credit = float(credits.get(target_action, 0.0))
            best_mechanism_credit = max(float(credits.get(action_id, 0.0)) for action_id in mechanism_actions)
            if (
                target_credit + self._argmax_margin_min_gap <= best_mechanism_credit
                and tail_risk_pressure + 1e-8 < self._argmax_margin_tail_risk_threshold
            ):
                continue

            env_scores = self._masked_flat_logits(
                self._hierarchical_env_action_scores(policy_output),
                action_mask,
            )
            target_score = env_scores[int(target_action)]
            row_terms: list[torch.Tensor] = []
            for mechanism_action in mechanism_actions:
                credit_gap = target_credit - float(credits.get(mechanism_action, 0.0))
                adaptive_margin = self._argmax_margin_min_gap + max(credit_gap, 0.0)
                row_terms.append(torch.relu(env_scores[int(mechanism_action)] - target_score + adaptive_margin))
            if not row_terms:
                continue

            selected_mechanism_pressure = 1.0 if final_action in {1, 4} else 0.0
            weight = (
                1.0
                + self._argmax_margin_mechanism_penalty_scale
                * (
                    tail_risk_pressure
                    + 0.35 * selected_mechanism_pressure
                    + 0.25 * failed_mechanism
                    + 0.20 * max(target_credit - best_mechanism_credit, 0.0)
                )
            )
            if self._argmax_margin_max_weight > 0.0:
                weight = min(weight, self._argmax_margin_max_weight)
            loss_terms.append(torch.stack(row_terms).mean() * float(weight))

        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

    def _effective_option_gate_prior_coef(self) -> float:
        if self._option_gate_prior_coef <= 0.0:
            return 0.0
        if self._update_count < self._option_gate_prior_warmup_updates:
            return self._option_gate_prior_coef
        decay_steps = self._update_count - self._option_gate_prior_warmup_updates + 1
        return float(self._option_gate_prior_coef * (self._option_gate_prior_decay ** decay_steps))

    def _compute_option_gate_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
        batch_advantage: torch.Tensor,
        batch_option_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
        if not self._option_gate_enabled:
            return zero, zero, zero, zero, zero
        ppo_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        prior_terms: list[torch.Tensor] = []
        counterfactual_value_terms: list[torch.Tensor] = []
        counterfactual_advantage_terms: list[torch.Tensor] = []
        counterfactual_actor_active = bool(
            self._option_counterfactual_critic_enabled
            and self._update_count >= self._option_counterfactual_warmup_updates
            and self._option_counterfactual_advantage_coef > 0.0
        )
        for row_index, (policy_output, row) in enumerate(zip(batch_outputs, batch_rows)):
            if "option_logits" not in policy_output:
                continue
            option_info = dict(row.get("action_info", {}).get("option_gate", {}))
            if not bool(option_info.get("enabled", False)):
                continue
            option_action = int(option_info.get("option_action", 0) or 0)
            if option_action < 0 or option_action >= int(policy_output["option_logits"].shape[-1]):
                continue
            option_mask = list(option_info.get("option_mask", []))
            prior_target = int(option_info.get("prior_target", 0) or 0)
            actor_logits = self._masked_option_logits(
                policy_output["option_logits"],
                option_mask if option_mask else None,
                prior_target=prior_target,
            )
            if self._option_counterfactual_policy_improvement_deterministic_only:
                logits = actor_logits
            else:
                logits, _ = self._critic_improved_option_logits(
                    option_logits=actor_logits,
                    option_q_values=policy_output.get("option_q_values"),
                    option_mask=option_mask if option_mask else None,
                )
            distribution = Categorical(logits=logits)
            action_tensor = torch.tensor(option_action, dtype=torch.long, device=self._device)
            old_log_prob = torch.tensor(
                float(option_info.get("option_log_prob", 0.0) or 0.0),
                dtype=torch.float32,
                device=self._device,
            )
            new_log_prob = distribution.log_prob(action_tensor)
            ratio = torch.exp(new_log_prob - old_log_prob)
            option_advantage = self._option_gate_advantage(
                row=row,
                base_advantage=batch_advantage[row_index],
                option_probs=distribution.probs.detach(),
                option_mask=option_mask if option_mask else None,
            )
            if (
                self._option_counterfactual_critic_enabled
                and "option_q_values" in policy_output
            ):
                option_q_values = policy_output["option_q_values"]
                valid_mask = torch.ones_like(option_q_values, dtype=torch.bool)
                if option_mask and len(option_mask) == int(option_q_values.shape[-1]):
                    valid_mask = torch.as_tensor(
                        option_mask,
                        dtype=torch.bool,
                        device=self._device,
                    )
                selected_q = option_q_values[option_action]
                tail_context = dict(
                    option_info.get("sparse_tail_risk_option_context", {})
                )
                tail_scale = 1.0
                if bool(tail_context.get("active", False)):
                    tail_scale += self._option_counterfactual_tail_weight * _clamp01(
                        float(tail_context.get("context", 0.0) or 0.0)
                    )
                model_rollout_info = dict(
                    option_info.get("counterfactual_model_rollout", {})
                )
                model_targets_raw = model_rollout_info.get("option_td_targets", {})
                model_targets: dict[int, float] = {}
                if isinstance(model_targets_raw, dict):
                    for raw_key, raw_value in model_targets_raw.items():
                        try:
                            model_targets[int(raw_key)] = float(raw_value)
                        except (TypeError, ValueError):
                            continue
                valid_target_indices = [
                    option_index
                    for option_index in range(int(option_q_values.shape[-1]))
                    if bool(valid_mask[option_index].item())
                    and option_index in model_targets
                ]
                if valid_target_indices:
                    target_values = torch.as_tensor(
                        [model_targets[index] for index in valid_target_indices],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    centered_targets = target_values - target_values.mean()
                    target_scale = torch.sqrt(
                        torch.mean(centered_targets.square())
                    ).clamp_min(1e-6)
                    normalized_targets = centered_targets / target_scale
                    if self._option_counterfactual_advantage_clip > 0.0:
                        normalized_targets = torch.clamp(
                            normalized_targets,
                            -self._option_counterfactual_advantage_clip,
                            self._option_counterfactual_advantage_clip,
                        )
                    predicted_targets = option_q_values[valid_target_indices]
                    centered_predictions = (
                        predicted_targets - predicted_targets.mean()
                    )
                    counterfactual_value_terms.append(
                        tail_scale
                        * nn.functional.smooth_l1_loss(
                            centered_predictions,
                            normalized_targets.detach(),
                        )
                    )
                else:
                    counterfactual_value_terms.append(
                        tail_scale
                        * nn.functional.smooth_l1_loss(
                            selected_q,
                            batch_option_returns[row_index].detach(),
                        )
                    )
                detached_q = option_q_values.detach().masked_fill(~valid_mask, 0.0)
                detached_probs = distribution.probs.detach().masked_fill(~valid_mask, 0.0)
                probability_mass = detached_probs.sum().clamp_min(1e-8)
                detached_probs = detached_probs / probability_mass
                counterfactual_baseline = torch.sum(detached_probs * detached_q)
                counterfactual_advantage = (
                    detached_q[option_action] - counterfactual_baseline
                )
                if self._option_counterfactual_advantage_clip > 0.0:
                    counterfactual_advantage = torch.clamp(
                        counterfactual_advantage,
                        -self._option_counterfactual_advantage_clip,
                        self._option_counterfactual_advantage_clip,
                    )
                counterfactual_advantage_terms.append(
                    torch.abs(counterfactual_advantage)
                )
                if counterfactual_actor_active:
                    option_advantage = (
                        option_advantage
                        + self._option_counterfactual_advantage_coef
                        * tail_scale
                        * counterfactual_advantage
                    )
            surrogate_1 = ratio * option_advantage
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - self._clip_ratio,
                1.0 + self._clip_ratio,
            ) * option_advantage
            ppo_terms.append(-torch.min(surrogate_1, surrogate_2))
            entropy_terms.append(distribution.entropy())
            if 0 <= prior_target < int(logits.shape[-1]):
                if not option_mask or bool(option_mask[prior_target]):
                    target_tensor = torch.tensor([prior_target], dtype=torch.long, device=self._device)
                    prior_terms.append(nn.functional.cross_entropy(logits.unsqueeze(0), target_tensor))
        ppo_loss = torch.stack(ppo_terms).mean() if ppo_terms else zero
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else zero
        prior_loss = torch.stack(prior_terms).mean() if prior_terms else zero
        counterfactual_value_loss = (
            torch.stack(counterfactual_value_terms).mean()
            if counterfactual_value_terms
            else zero
        )
        counterfactual_advantage_abs = (
            torch.stack(counterfactual_advantage_terms).mean()
            if counterfactual_advantage_terms
            else zero
        )
        return (
            ppo_loss,
            entropy,
            prior_loss,
            counterfactual_value_loss,
            counterfactual_advantage_abs,
        )

    def _compute_env_action_model_critic_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
        if not self._env_action_model_critic_enabled:
            return zero, zero
        value_terms: list[torch.Tensor] = []
        advantage_terms: list[torch.Tensor] = []
        for output, row, action_mask in zip(
            batch_outputs,
            batch_rows,
            batch_action_masks,
        ):
            q_values = output.get("env_action_q_values")
            if not isinstance(q_values, torch.Tensor):
                continue
            rollout_info = dict(
                row.get("action_info", {}).get("env_action_model_rollout", {})
            )
            target_maps = self._extract_env_action_model_target_maps(
                rollout_info
            )
            valid_indices = [
                action_id
                for action_id in range(int(q_values.shape[-1]))
                if target_maps
                and all(action_id in targets for targets in target_maps)
                and (
                    not action_mask
                    or (
                        action_id < len(action_mask)
                        and bool(action_mask[action_id])
                    )
                )
            ]
            if len(valid_indices) < 2:
                continue
            normalized_targets = self._build_env_action_model_robust_advantage(
                target_maps=target_maps,
                valid_indices=valid_indices,
            )
            predicted_values = q_values[valid_indices]
            centered_predictions = predicted_values - predicted_values.mean()
            value_terms.append(
                nn.functional.smooth_l1_loss(
                    centered_predictions,
                    normalized_targets.detach(),
                )
            )
            advantage_terms.append(torch.mean(torch.abs(centered_predictions)))
        return (
            torch.stack(value_terms).mean() if value_terms else zero,
            torch.stack(advantage_terms).mean() if advantage_terms else zero,
        )

    @staticmethod
    def _parse_env_action_target_map(
        targets_raw: Any,
    ) -> dict[int, float]:
        targets: dict[int, float] = {}
        if not isinstance(targets_raw, dict):
            return targets
        for raw_key, raw_value in targets_raw.items():
            try:
                targets[int(raw_key)] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return targets

    def _extract_env_action_model_target_maps(
        self,
        rollout_info: dict[str, Any],
    ) -> list[dict[int, float]]:
        if self._env_action_model_policy_improvement_prefer_beam_targets:
            beam_targets = self._parse_env_action_target_map(
                rollout_info.get("beam_action_td_targets", {})
            )
            if beam_targets:
                return [beam_targets]
        if self._env_action_model_policy_improvement_robust_horizons_enabled:
            by_horizon_raw = rollout_info.get(
                "action_td_targets_by_horizon",
                {},
            )
            if isinstance(by_horizon_raw, dict):
                ordered_maps: list[tuple[int, dict[int, float]]] = []
                for raw_horizon, targets_raw in by_horizon_raw.items():
                    try:
                        horizon = int(raw_horizon)
                    except (TypeError, ValueError):
                        continue
                    targets = self._parse_env_action_target_map(targets_raw)
                    if targets:
                        ordered_maps.append((horizon, targets))
                if len(ordered_maps) >= 2:
                    return [
                        targets
                        for _, targets in sorted(ordered_maps)
                    ]
        targets = self._parse_env_action_target_map(
            rollout_info.get("action_td_targets", {})
        )
        return [targets] if targets else []

    def _extract_env_action_model_resource_cost_maps(
        self,
        rollout_info: dict[str, Any],
    ) -> list[dict[int, float]]:
        if not self._env_action_model_resource_constraint_enabled:
            return []
        by_horizon_raw = rollout_info.get(
            "action_resource_costs_by_horizon",
            {},
        )
        if isinstance(by_horizon_raw, dict):
            ordered_maps: list[tuple[int, dict[int, float]]] = []
            for raw_horizon, costs_raw in by_horizon_raw.items():
                try:
                    horizon = int(raw_horizon)
                except (TypeError, ValueError):
                    continue
                costs = self._parse_env_action_target_map(costs_raw)
                if costs:
                    ordered_maps.append((horizon, costs))
            if ordered_maps:
                return [costs for _, costs in sorted(ordered_maps)]
        costs = self._parse_env_action_target_map(
            rollout_info.get("action_resource_costs", {})
        )
        return [costs] if costs else []

    def _build_adaptive_horizon_weights(
        self,
        *,
        rollout_info: dict[str, Any],
        action_info: dict[str, Any],
        target_maps: list[dict[int, float]],
    ) -> torch.Tensor | None:
        if (
            not self._env_action_model_adaptive_horizon_enabled
            or len(target_maps) < 2
        ):
            return None
        raw_horizons = rollout_info.get("rollout_horizons", [])
        horizons: list[int] = []
        if isinstance(raw_horizons, (list, tuple)):
            for raw_horizon in raw_horizons:
                try:
                    horizons.append(max(int(raw_horizon), 1))
                except (TypeError, ValueError):
                    continue
        if len(horizons) != len(target_maps):
            horizons = list(range(1, len(target_maps) + 1))
        horizons = sorted(horizons)
        urgency = max(
            float(action_info.get("temporal_urgency", 0.0) or 0.0),
            float(action_info.get("prepare_window_score", 0.0) or 0.0),
        )
        confidence = float(action_info.get("prediction_confidence", 0.0) or 0.0)
        countdown = float(action_info.get("handoff_countdown_steps", 0.0) or 0.0)
        countdown_signal = 0.0
        if countdown > 0.0:
            countdown_signal = 1.0 / (1.0 + countdown)
        signal = float(
            np.clip(
                0.55 * urgency + 0.25 * countdown_signal + 0.20 * confidence,
                0.0,
                1.0,
            )
        )
        minimum_horizon = float(min(horizons))
        maximum_horizon = float(max(horizons))
        effective_horizon = minimum_horizon + signal * (
            maximum_horizon - minimum_horizon
        )
        temperature = self._env_action_model_adaptive_horizon_temperature
        distance = torch.as_tensor(
            [abs(float(horizon) - effective_horizon) for horizon in horizons],
            dtype=torch.float32,
            device=self._device,
        )
        weights = torch.softmax(-distance / temperature, dim=0)
        return weights / weights.sum().clamp_min(1e-8)

    def select_env_action_from_model_targets(
        self,
        *,
        action_info: dict[str, Any],
        rollout_info: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        """Perform a policy-prior-constrained counterfactual action improvement.

        The digital twin supplies action returns; the MAPPO policy remains a
        prior through the KL-like log-probability term. This keeps planning
        inside the agent contract instead of adding an evaluator-side rule.
        """
        current_action = int(action_info.get("final_env_action", 3))
        learned_model_source = str(rollout_info.get("source", "")) == (
            "learned_transition_ensemble"
        )
        planner_enabled = bool(
            self._env_action_model_online_planner_enabled or learned_model_source
        )
        model_coef = (
            self._learned_transition_model_policy_coef
            if learned_model_source
            else self._env_action_model_online_planner_coef
        )
        policy_prior_coef = (
            self._learned_transition_model_policy_prior_coef
            if learned_model_source
            else self._env_action_model_online_planner_policy_prior_coef
        )
        min_margin = (
            self._learned_transition_model_min_margin
            if learned_model_source
            else self._env_action_model_online_planner_min_margin
        )
        planner_stats: dict[str, Any] = {
            "enabled": planner_enabled,
            "applied": False,
            "protocol": (
                "ucc_mappo_lcb_policy_improvement_v1"
                if learned_model_source
                else "mappo_counterfactual_policy_improvement_v1"
            ),
            "candidate_count": 0,
            "target_maps": 0,
            "mechanism_target_success_count": 0,
            "current_action": current_action,
            "selected_action": current_action,
            "model_coef": model_coef,
            "mechanism_coef": self._env_action_model_online_planner_mechanism_coef,
            "policy_prior_coef": policy_prior_coef,
            "min_margin": min_margin,
            "resource_constraint_enabled": bool(
                self._env_action_model_resource_constraint_enabled
            ),
            "resource_cost_coef": self._env_action_model_resource_cost_coef,
        }
        if not planner_enabled:
            return current_action, planner_stats

        target_maps = self._extract_env_action_model_target_maps(rollout_info)
        if self._env_action_model_online_planner_prefer_beam_targets:
            beam_targets = self._parse_env_action_target_map(
                rollout_info.get("beam_action_td_targets", {})
            )
            if beam_targets:
                target_maps = [beam_targets]
        action_mask = action_info.get("action_mask")
        valid_indices = [
            action_id
            for action_id in range(5)
            if not action_mask
            or (
                action_id < len(action_mask)
                and bool(action_mask[action_id])
            )
        ]
        valid_indices = [
            action_id
            for action_id in valid_indices
            if target_maps and all(action_id in targets for targets in target_maps)
        ]
        planner_stats["target_maps"] = len(target_maps)
        planner_stats["candidate_count"] = len(valid_indices)
        if len(valid_indices) < 2:
            planner_stats["reason"] = "insufficient_counterfactual_support"
            return current_action, planner_stats

        action_projection = dict(action_info.get("action_projection", {}))
        prior_raw = action_projection.get(
            "masked_env_action_probs",
            action_info.get("env_action_probs", []),
        )
        if isinstance(prior_raw, list) and len(prior_raw) >= 5:
            prior = torch.as_tensor(
                [max(float(prior_raw[action_id]), 1e-8) for action_id in valid_indices],
                dtype=torch.float32,
                device=self._device,
            )
            prior = prior / prior.sum().clamp_min(1e-8)
        else:
            prior = torch.full(
                (len(valid_indices),),
                1.0 / len(valid_indices),
                dtype=torch.float32,
                device=self._device,
            )
        model_advantage = self._build_env_action_model_robust_advantage(
            target_maps=target_maps,
            valid_indices=valid_indices,
            horizon_weights=self._build_adaptive_horizon_weights(
                rollout_info=rollout_info,
                action_info=action_info,
                target_maps=target_maps,
            ),
            resource_cost_maps=self._extract_env_action_model_resource_cost_maps(
                rollout_info
            ),
            resource_cost_coef=(
                self._env_action_model_resource_cost_coef
                if self._env_action_model_resource_constraint_enabled
                else 0.0
            ),
        ).detach()
        mechanism_target_raw = rollout_info.get(
            "action_mechanism_targets",
            {},
        )
        mechanism_targets = self._parse_env_action_target_map(
            mechanism_target_raw
        )
        mechanism_target_success_count = sum(
            1
            for action_id in valid_indices
            if float(mechanism_targets.get(action_id, 0.0)) > 0.0
        )
        planner_stats["mechanism_target_success_count"] = (
            mechanism_target_success_count
        )
        mechanism_valid = all(
            action_id in mechanism_targets for action_id in valid_indices
        )
        mechanism_advantage = torch.zeros_like(model_advantage)
        if mechanism_valid and len(valid_indices) >= 2:
            mechanism_values = torch.as_tensor(
                [mechanism_targets[action_id] for action_id in valid_indices],
                dtype=torch.float32,
                device=self._device,
            )
            mechanism_centered = mechanism_values - mechanism_values.mean()
            mechanism_scale = torch.sqrt(
                torch.mean(mechanism_centered.square())
            ).clamp_min(1e-6)
            mechanism_advantage = mechanism_centered / mechanism_scale
        scores = (
            model_coef * model_advantage
            + self._env_action_model_online_planner_mechanism_coef
            * mechanism_advantage
            + policy_prior_coef
            * torch.log(prior.clamp_min(1e-8))
        )
        ranked_positions = torch.argsort(scores, descending=True).detach().cpu().tolist()
        best_position = int(ranked_positions[0])
        second_position = int(ranked_positions[1])
        score_margin = float(
            (scores[best_position] - scores[second_position]).item()
        )
        selected_action = int(valid_indices[best_position])
        if (
            score_margin < min_margin
            and current_action in valid_indices
        ):
            selected_action = current_action
            planner_stats["reason"] = "below_min_margin"
        else:
            planner_stats["reason"] = "counterfactual_policy_improvement"
        planner_stats.update(
            {
                "applied": selected_action != current_action,
                "selected_action": selected_action,
                "score_margin": round(score_margin, 6),
                "candidate_actions": valid_indices,
                "model_advantage": {
                    str(action_id): round(float(value), 6)
                    for action_id, value in zip(
                        valid_indices,
                        model_advantage.detach().cpu().tolist(),
                    )
                },
                "resource_cost_advantage": {
                    str(action_id): round(float(value), 6)
                    for action_id, value in zip(
                        valid_indices,
                        self._build_env_action_model_resource_cost_advantage(
                            resource_cost_maps=self._extract_env_action_model_resource_cost_maps(
                                rollout_info
                            ),
                            valid_indices=valid_indices,
                        ).detach().cpu().tolist(),
                    )
                },
                "adaptive_horizon_weights": (
                    self._build_adaptive_horizon_weights(
                        rollout_info=rollout_info,
                        action_info=action_info,
                        target_maps=target_maps,
                    ).detach().cpu().tolist()
                    if self._build_adaptive_horizon_weights(
                        rollout_info=rollout_info,
                        action_info=action_info,
                        target_maps=target_maps,
                    ) is not None
                    else []
                ),
                "mechanism_advantage": {
                    str(action_id): round(float(value), 6)
                    for action_id, value in zip(
                        valid_indices,
                        mechanism_advantage.detach().cpu().tolist(),
                    )
                },
                "mechanism_targets": {
                    str(action_id): round(
                        float(mechanism_targets.get(action_id, 0.0)),
                        6,
                    )
                    for action_id in valid_indices
                },
                "planner_scores": {
                    str(action_id): round(float(value), 6)
                    for action_id, value in zip(
                        valid_indices,
                        scores.detach().cpu().tolist(),
                    )
                },
            }
        )
        return selected_action, planner_stats

    def relabel_action_info_for_env_action(
        self,
        *,
        action_info: dict[str, Any],
        decision_info: dict[str, Any],
        env_action: int,
        planner_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep on-policy statistics aligned after model-based action improvement."""
        semantic_state = self._extract_semantic_state(decision_info)
        action_mask = self._extract_action_mask(decision_info)
        run_metadata = dict(decision_info.get("run_metadata", {}) or {})
        with torch.no_grad():
            policy_output = self._forward_policy(
                semantic_state,
                run_metadata=run_metadata,
            )
            selected_actions = self._head_targets_for_env_action(int(env_action))
            head_log_probs, head_entropies, action_prob_payload = (
                self._selected_action_statistics(
                    policy_output=policy_output,
                    selected_actions=selected_actions,
                    action_mask=action_mask,
                )
            )
            env_action_log_prob, env_action_entropy, env_action_probs = (
                self._env_action_distribution_statistics(
                    policy_output=policy_output,
                    env_action=int(env_action),
                    action_mask=action_mask,
                )
            )
            log_prob, entropy = self._combine_head_statistics(
                head_log_probs=head_log_probs,
                head_entropies=head_entropies,
                head_credit_weights=self._build_head_credit_weights(
                    aggregation_reason="online_counterfactual_planner"
                ),
            )
        updated = dict(action_info)
        updated["head_actions"] = selected_actions
        updated["head_action_labels"] = self._head_action_labels(selected_actions)
        updated["raw_env_action"] = int(env_action)
        updated["projected_env_action"] = int(env_action)
        updated["final_env_action"] = int(env_action)
        updated["aggregation_reason"] = "online_counterfactual_planner"
        updated["log_prob"] = round(float(log_prob.item()), 6)
        updated["env_action_log_prob"] = round(float(env_action_log_prob.item()), 6)
        updated["env_action_entropy"] = round(float(env_action_entropy.item()), 6)
        updated["env_action_probs"] = env_action_probs
        updated["action_probs"] = action_prob_payload
        updated["head_log_probs"] = {
            head_name: round(float(value.item()), 6)
            for head_name, value in head_log_probs.items()
        }
        updated["value"] = round(float(policy_output["value"].item()), 6)
        updated["entropy"] = round(float(entropy.item()), 6)
        projection = dict(updated.get("action_projection", {}))
        projection.update(
            {
                "raw_env_action": int(env_action),
                "projected_env_action": int(env_action),
                "masked_env_action_log_prob": round(
                    float(env_action_log_prob.item()),
                    6,
                ),
                "online_planner_relabelled": True,
            }
        )
        updated["action_projection"] = projection
        updated["online_counterfactual_planner"] = dict(planner_stats)
        updated["planner_original_action"] = int(
            planner_stats.get("current_action", action_info.get("final_env_action", 3))
        )
        return updated

    def _build_env_action_model_robust_advantage(
        self,
        *,
        target_maps: list[dict[int, float]],
        valid_indices: list[int],
        horizon_weights: torch.Tensor | None = None,
        resource_cost_maps: list[dict[int, float]] | None = None,
        resource_cost_coef: float = 0.0,
    ) -> torch.Tensor:
        normalized_by_horizon: list[torch.Tensor] = []
        for targets in target_maps:
            target_values = torch.as_tensor(
                [targets[action_id] for action_id in valid_indices],
                dtype=torch.float32,
                device=self._device,
            )
            centered_targets = target_values - target_values.mean()
            target_scale = torch.sqrt(
                torch.mean(centered_targets.square())
            ).clamp_min(1e-6)
            normalized_by_horizon.append(centered_targets / target_scale)
        stacked = torch.stack(normalized_by_horizon)
        if (
            len(normalized_by_horizon) > 1
            and self._env_action_model_policy_improvement_horizon_aggregation_mode
            == "lambda_downside"
        ):
            horizon_count = len(normalized_by_horizon)
            horizon_lambda = (
                self._env_action_model_policy_improvement_horizon_lambda
            )
            lambda_weights = [
                (1.0 - horizon_lambda) * (horizon_lambda**index)
                for index in range(horizon_count - 1)
            ]
            lambda_weights.append(horizon_lambda ** (horizon_count - 1))
            weight_tensor = torch.as_tensor(
                lambda_weights,
                dtype=torch.float32,
                device=self._device,
            )
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
            if horizon_weights is not None and len(horizon_weights) == len(
                normalized_by_horizon
            ):
                weight_tensor = horizon_weights.detach()
                weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
            robust_advantage = torch.sum(
                weight_tensor.unsqueeze(-1) * stacked,
                dim=0,
            )

            # Penalize only value deterioration at longer planning depths.
            # Delayed gains from prepare/cache actions remain opportunity-preserving.
            temporal_decline = torch.relu(stacked[:-1] - stacked[1:])
            decline_weights = weight_tensor[1:]
            decline_weights = decline_weights / decline_weights.sum().clamp_min(
                1e-6
            )
            temporal_downside = torch.sqrt(
                torch.sum(
                    decline_weights.unsqueeze(-1) * temporal_decline.square(),
                    dim=0,
                ).clamp_min(0.0)
            )
            robust_advantage = (
                robust_advantage
                - self._env_action_model_policy_improvement_horizon_risk_coef
                * temporal_downside
            )
        else:
            if horizon_weights is not None and len(horizon_weights) == len(
                normalized_by_horizon
            ):
                weight_tensor = horizon_weights.detach()
                weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
                robust_advantage = torch.sum(
                    weight_tensor.unsqueeze(-1) * stacked,
                    dim=0,
                )
            else:
                robust_advantage = stacked.mean(dim=0)
            if len(normalized_by_horizon) > 1:
                robust_advantage = (
                    robust_advantage
                    - self._env_action_model_policy_improvement_horizon_risk_coef
                    * stacked.std(dim=0, unbiased=False)
                )
        if resource_cost_maps and resource_cost_coef > 0.0:
            resource_cost_advantage = self._build_env_action_model_resource_cost_advantage(
                resource_cost_maps=resource_cost_maps,
                valid_indices=valid_indices,
                horizon_weights=horizon_weights,
            )
            robust_advantage = robust_advantage - float(
                resource_cost_coef
            ) * resource_cost_advantage
        robust_advantage = robust_advantage - robust_advantage.mean()
        if self._env_action_model_critic_advantage_clip > 0.0:
            robust_advantage = torch.clamp(
                robust_advantage,
                -self._env_action_model_critic_advantage_clip,
                self._env_action_model_critic_advantage_clip,
            )
        return robust_advantage

    def _build_env_action_model_resource_cost_advantage(
        self,
        *,
        resource_cost_maps: list[dict[int, float]],
        valid_indices: list[int],
        horizon_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not resource_cost_maps:
            return torch.zeros(
                len(valid_indices),
                dtype=torch.float32,
                device=self._device,
            )
        normalized_by_horizon: list[torch.Tensor] = []
        for costs in resource_cost_maps:
            values = torch.as_tensor(
                [
                    float(costs.get(action_id, 0.0))
                    / self._env_action_model_resource_cost_scale
                    for action_id in valid_indices
                ],
                dtype=torch.float32,
                device=self._device,
            )
            centered = values - values.mean()
            scale = torch.sqrt(torch.mean(centered.square())).clamp_min(1e-6)
            normalized_by_horizon.append(centered / scale)
        stacked = torch.stack(normalized_by_horizon)
        if horizon_weights is not None and len(horizon_weights) == len(
            normalized_by_horizon
        ):
            weights = horizon_weights.detach()
            weights = weights / weights.sum().clamp_min(1e-8)
            return torch.sum(weights.unsqueeze(-1) * stacked, dim=0)
        return stacked.mean(dim=0)

    def _build_env_action_model_improved_target(
        self,
        *,
        old_probs: torch.Tensor,
        normalized_advantage: torch.Tensor,
        target_kl: float | None = None,
    ) -> torch.Tensor:
        if not self._env_action_model_policy_improvement_adaptive_kl_enabled:
            target_logits = (
                torch.log(old_probs.clamp_min(1e-8))
                + normalized_advantage
                / self._env_action_model_policy_improvement_temperature
            )
            return torch.softmax(target_logits, dim=-1).detach()

        old_probs_detached = old_probs.detach()
        advantage_detached = normalized_advantage.detach()
        effective_target_kl = max(
            float(
                self._env_action_model_policy_improvement_target_kl
                if target_kl is None
                else target_kl
            ),
            1e-5,
        )

        def target_and_kl(temperature: float) -> tuple[torch.Tensor, float]:
            target = torch.softmax(
                torch.log(old_probs_detached.clamp_min(1e-8))
                + advantage_detached / max(float(temperature), 1e-4),
                dim=-1,
            )
            kl_value = torch.sum(
                target
                * (
                    torch.log(target.clamp_min(1e-8))
                    - torch.log(old_probs_detached.clamp_min(1e-8))
                )
            )
            return target, float(kl_value.item())

        lower = 1e-3
        upper = 100.0
        target, lower_kl = target_and_kl(lower)
        if lower_kl <= effective_target_kl:
            return target.detach()
        for _ in range(32):
            midpoint = 0.5 * (lower + upper)
            candidate, candidate_kl = target_and_kl(midpoint)
            if candidate_kl > effective_target_kl:
                lower = midpoint
            else:
                upper = midpoint
                target = candidate
        return target.detach()

    @staticmethod
    def _normalized_env_action_counterfactual_regret(
        *,
        old_probs: torch.Tensor,
        normalized_advantage: torch.Tensor,
    ) -> torch.Tensor:
        advantage_range = (
            normalized_advantage.max() - normalized_advantage.min()
        ).clamp_min(1e-6)
        policy_value = torch.sum(
            old_probs.detach() * normalized_advantage.detach()
        )
        regret = normalized_advantage.detach().max() - policy_value
        return torch.clamp(regret / advantage_range, 0.0, 1.0)

    def _build_env_action_model_policy_improvement_context(
        self,
        *,
        output: dict[str, Any],
        row: dict[str, Any],
        action_mask: list[bool] | None,
    ) -> dict[str, Any] | None:
        rollout_info = dict(
            row.get("action_info", {}).get("env_action_model_rollout", {})
        )
        target_maps = self._extract_env_action_model_target_maps(rollout_info)
        if self._use_hierarchy:
            actor_scores = self._hierarchical_env_action_scores(output)
        else:
            actor_scores = output["flat_logits"]
        valid_indices = [
            action_id
            for action_id in range(int(actor_scores.shape[-1]))
            if target_maps
            and all(action_id in targets for targets in target_maps)
            and (
                not action_mask
                or (
                    action_id < len(action_mask)
                    and bool(action_mask[action_id])
                )
            )
        ]
        if len(valid_indices) < 2:
            return None

        normalized_advantage = self._build_env_action_model_robust_advantage(
            target_maps=target_maps,
            valid_indices=valid_indices,
            horizon_weights=self._build_adaptive_horizon_weights(
                rollout_info=rollout_info,
                action_info=dict(row.get("action_info", {})),
                target_maps=target_maps,
            ),
            resource_cost_maps=self._extract_env_action_model_resource_cost_maps(
                rollout_info
            ),
            resource_cost_coef=(
                self._env_action_model_resource_cost_coef
                if self._env_action_model_resource_constraint_enabled
                else 0.0
            ),
        )
        projection_info = dict(
            row.get("action_info", {}).get("action_projection", {})
        )
        old_probs_raw = projection_info.get("masked_env_action_probs", [])
        if (
            isinstance(old_probs_raw, list)
            and len(old_probs_raw) == int(actor_scores.shape[-1])
        ):
            old_probs = torch.as_tensor(
                [
                    float(old_probs_raw[action_id])
                    for action_id in valid_indices
                ],
                dtype=torch.float32,
                device=self._device,
            )
            old_probs = old_probs / old_probs.sum().clamp_min(1e-8)
        else:
            old_probs = torch.softmax(
                actor_scores.detach()[valid_indices],
                dim=-1,
            )
        normalized_regret = self._normalized_env_action_counterfactual_regret(
            old_probs=old_probs,
            normalized_advantage=normalized_advantage,
        )
        effective_target_kl = (
            self._env_action_model_policy_improvement_target_kl
        )
        if (
            self._env_action_model_policy_improvement_regret_adaptive_kl_enabled
        ):
            effective_target_kl = (
                self._env_action_model_policy_improvement_target_kl
                + (
                    self._env_action_model_policy_improvement_max_target_kl
                    - self._env_action_model_policy_improvement_target_kl
                )
                * float(normalized_regret.item())
            )
        improved_target = self._build_env_action_model_improved_target(
            old_probs=old_probs,
            normalized_advantage=normalized_advantage,
            target_kl=effective_target_kl,
        )
        target_kl = torch.sum(
            improved_target
            * (
                torch.log(improved_target.clamp_min(1e-8))
                - torch.log(old_probs.clamp_min(1e-8))
            )
        )
        return {
            "actor_scores": actor_scores,
            "valid_indices": valid_indices,
            "old_probs": old_probs,
            "improved_target": improved_target,
            "target_action_id": valid_indices[
                int(torch.argmax(improved_target).item())
            ],
            "normalized_regret": normalized_regret,
            "target_kl": target_kl,
        }

    def _compute_env_action_model_policy_improvement_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
        logit_projection_enabled: bool = False,
        sample_weights: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
        if not self._env_action_model_policy_improvement_enabled:
            return zero, zero
        effective_sample_weights = (
            [1.0] * len(batch_outputs)
            if sample_weights is None
            else [max(float(weight), 0.0) for weight in sample_weights]
        )
        if len(effective_sample_weights) != len(batch_outputs):
            raise ValueError("sample_weights must align with batch_outputs")

        loss_terms: list[torch.Tensor] = []
        loss_weights: list[torch.Tensor] = []
        target_kl_terms: list[torch.Tensor] = []
        for output, row, action_mask, sample_weight in zip(
            batch_outputs,
            batch_rows,
            batch_action_masks,
            effective_sample_weights,
        ):
            context = self._build_env_action_model_policy_improvement_context(
                output=output,
                row=row,
                action_mask=action_mask,
            )
            if context is None:
                continue
            actor_scores = context["actor_scores"]
            valid_indices = context["valid_indices"]
            improved_target = context["improved_target"]
            normalized_regret = context["normalized_regret"]
            loss_weight = (
                1.0
                + self._env_action_model_policy_improvement_regret_priority_coef
                * normalized_regret
            )
            loss_weight = loss_weight * sample_weight
            if logit_projection_enabled:
                current_logits = actor_scores[valid_indices]
                current_logits = current_logits - current_logits.mean()
                target_logits = torch.log(improved_target.clamp_min(1e-8))
                target_logits = target_logits - target_logits.mean()
                projection_loss = nn.functional.smooth_l1_loss(
                    current_logits,
                    target_logits.detach(),
                )
                loss_terms.append(loss_weight * projection_loss)
            else:
                current_log_probs = torch.log_softmax(
                    actor_scores[valid_indices],
                    dim=-1,
                )
                loss_terms.append(
                    -loss_weight
                    * torch.sum(improved_target * current_log_probs)
                )
            loss_weights.append(loss_weight)
            target_kl_terms.append(context["target_kl"])

        return (
            (
                torch.stack(loss_terms).sum()
                / torch.stack(loss_weights).sum().clamp_min(1e-6)
                if loss_terms
                else zero
            ),
            torch.stack(target_kl_terms).mean() if target_kl_terms else zero,
        )

    @staticmethod
    def _build_tail_target_balance_weights(
        selected: list[tuple[int, float, int, bool]],
        *,
        power: float,
        max_weight: float,
    ) -> tuple[dict[int, float], dict[str, float]]:
        if not selected:
            return {}, {}
        target_counts: dict[int, int] = {}
        for _, _, target_action, _ in selected:
            target_counts[target_action] = target_counts.get(target_action, 0) + 1
        largest_count = max(target_counts.values())
        clipped_power = min(max(float(power), 0.0), 1.0)
        clipped_max_weight = max(float(max_weight), 1.0)
        raw_by_target = {
            target_action: min(
                (largest_count / max(count, 1)) ** clipped_power,
                clipped_max_weight,
            )
            for target_action, count in target_counts.items()
        }
        sample_mean = sum(
            target_counts[target_action] * raw_weight
            for target_action, raw_weight in raw_by_target.items()
        ) / len(selected)
        normalized_by_target = {
            target_action: raw_weight / max(sample_mean, 1e-8)
            for target_action, raw_weight in raw_by_target.items()
        }
        return (
            {
                index: normalized_by_target[target_action]
                for index, _, target_action, _ in selected
            },
            {
                str(target_action): normalized_by_target[target_action]
                for target_action in sorted(normalized_by_target)
            },
        )

    def _mean_env_action_model_policy_kl(
        self,
        *,
        semantic_states: list[dict[str, Any]],
        run_metadata_by_row: list[dict[str, Any]],
        rollout: list[dict[str, Any]],
        action_masks: list[list[bool] | None],
        selected_indices: list[int],
    ) -> float:
        kl_values: list[float] = []
        with torch.no_grad():
            for index in selected_indices:
                output = self._forward_policy(
                    semantic_states[index],
                    run_metadata=run_metadata_by_row[index],
                )
                context = (
                    self._build_env_action_model_policy_improvement_context(
                        output=output,
                        row=rollout[index],
                        action_mask=action_masks[index],
                    )
                )
                if context is None:
                    continue
                current_probs = torch.softmax(
                    context["actor_scores"][context["valid_indices"]],
                    dim=-1,
                )
                old_probs = context["old_probs"]
                policy_kl = torch.sum(
                    old_probs
                    * (
                        torch.log(old_probs.clamp_min(1e-8))
                        - torch.log(current_probs.clamp_min(1e-8))
                    )
                )
                kl_values.append(float(policy_kl.item()))
        return float(fmean(kl_values)) if kl_values else 0.0

    def _run_env_action_model_tail_distillation(
        self,
        *,
        semantic_states: list[dict[str, Any]],
        run_metadata_by_row: list[dict[str, Any]],
        rollout: list[dict[str, Any]],
        action_masks: list[list[bool] | None],
    ) -> dict[str, Any]:
        stats = {
            "enabled": (
                self._env_action_model_policy_improvement_tail_distillation_enabled
            ),
            "candidate_count": 0,
            "selected_count": 0,
            "selected_fraction": 0.0,
            "regret_mean": 0.0,
            "regret_threshold": 0.0,
            "selected_regret_mean": 0.0,
            "loss": 0.0,
            "update_steps": 0,
            "executed_epochs": 0,
            "policy_kl_before": 0.0,
            "policy_kl_after": 0.0,
            "early_stop_triggered": False,
            "residual_optimizer_used": False,
            "residual_learning_rate_initial": 0.0,
            "residual_learning_rate_final": 0.0,
            "backtrack_count": 0,
            "rejected_epoch_count": 0,
            "logit_projection_enabled": bool(
                self._env_action_model_policy_improvement_tail_logit_projection_enabled
            ),
            "target_balance_enabled": bool(
                self._env_action_model_policy_improvement_tail_target_balance_enabled
            ),
            "target_balance_power": float(
                self._env_action_model_policy_improvement_tail_target_balance_power
            ),
            "target_balance_max_weight": float(
                self._env_action_model_policy_improvement_tail_target_balance_max_weight
            ),
            "target_balance_weights": {},
            "target_balance_weight_min": 1.0,
            "target_balance_weight_mean": 1.0,
            "target_balance_weight_max": 1.0,
            "target_action_counts": {},
            "imagined_target_action_counts": {},
            "selected_target_action_counts": {},
            "selected_imagined_target_action_counts": {},
            "selected_imagined_count": 0,
            "recovery_candidate_count": 0,
        }
        if (
            not self._env_action_model_policy_improvement_enabled
            or not self._env_action_model_policy_improvement_tail_distillation_enabled
            or self._env_action_model_policy_improvement_tail_epochs <= 0
            or self._env_action_model_policy_improvement_tail_coef <= 0.0
        ):
            return stats

        regret_by_index: list[tuple[int, float, int, bool]] = []
        with torch.no_grad():
            for index, (state, run_metadata, row, action_mask) in enumerate(
                zip(
                    semantic_states,
                    run_metadata_by_row,
                    rollout,
                    action_masks,
                )
            ):
                output = self._forward_policy(
                    state,
                    run_metadata=run_metadata,
                )
                context = (
                    self._build_env_action_model_policy_improvement_context(
                        output=output,
                        row=row,
                        action_mask=action_mask,
                    )
                )
                if context is not None:
                    rollout_info = dict(
                        row.get("action_info", {}).get(
                            "env_action_model_rollout",
                            {},
                        )
                    )
                    if (
                        self._env_action_model_policy_improvement_tail_beam_only
                        and not rollout_info.get("beam_action_td_targets")
                    ):
                        continue
                    recovery_signal = self._semantic_state_recovery_signal(
                        state
                    )
                    if (
                        self._env_action_model_policy_improvement_tail_recovery_only
                        and recovery_signal <= 0.0
                    ):
                        continue
                    regret_by_index.append(
                        (
                            index,
                            float(context["normalized_regret"].item()),
                            int(context["target_action_id"]),
                            bool(row.get("imagination_depth")),
                        )
                    )
        if not regret_by_index:
            return stats

        regret_values = np.asarray(
            [regret for _, regret, _, _ in regret_by_index],
            dtype=np.float32,
        )
        regret_threshold = max(
            self._env_action_model_policy_improvement_tail_min_regret,
            float(
                np.quantile(
                    regret_values,
                    self._env_action_model_policy_improvement_tail_quantile,
                )
            ),
        )
        selected = [
            (index, regret, target_action, imagined)
            for index, regret, target_action, imagined in regret_by_index
            if regret + 1e-8 >= regret_threshold
        ]
        target_action_counts: dict[str, int] = {}
        for _, _, target_action, _ in regret_by_index:
            key = str(target_action)
            target_action_counts[key] = target_action_counts.get(key, 0) + 1
        selected_target_action_counts: dict[str, int] = {}
        for _, _, target_action, _ in selected:
            key = str(target_action)
            selected_target_action_counts[key] = (
                selected_target_action_counts.get(key, 0) + 1
            )
        imagined_target_action_counts: dict[str, int] = {}
        for _, _, target_action, imagined in regret_by_index:
            if imagined:
                key = str(target_action)
                imagined_target_action_counts[key] = (
                    imagined_target_action_counts.get(key, 0) + 1
                )
        selected_imagined_target_action_counts: dict[str, int] = {}
        for _, _, target_action, imagined in selected:
            if imagined:
                key = str(target_action)
                selected_imagined_target_action_counts[key] = (
                    selected_imagined_target_action_counts.get(key, 0) + 1
                )
        stats.update(
            {
                "candidate_count": len(regret_by_index),
                "selected_count": len(selected),
                "selected_fraction": (
                    len(selected) / max(len(regret_by_index), 1)
                ),
                "regret_mean": float(regret_values.mean()),
                "regret_threshold": regret_threshold,
                "selected_regret_mean": (
                    float(
                        fmean(
                            regret
                            for _, regret, _, _ in selected
                        )
                    )
                    if selected
                    else 0.0
                ),
                "target_action_counts": target_action_counts,
                "imagined_target_action_counts": (
                    imagined_target_action_counts
                ),
                "selected_target_action_counts": (
                    selected_target_action_counts
                ),
                "selected_imagined_target_action_counts": (
                    selected_imagined_target_action_counts
                ),
                "selected_imagined_count": sum(
                    1 for _, _, _, imagined in selected if imagined
                ),
                "recovery_candidate_count": len(regret_by_index),
            }
        )
        if not selected:
            return stats

        selected_indices = [index for index, _, _, _ in selected]
        target_balance_weight_by_index: dict[int, float] = {
            index: 1.0 for index in selected_indices
        }
        target_balance_weights: dict[str, float] = {}
        if self._env_action_model_policy_improvement_tail_target_balance_enabled:
            (
                target_balance_weight_by_index,
                target_balance_weights,
            ) = self._build_tail_target_balance_weights(
                selected,
                power=(
                    self._env_action_model_policy_improvement_tail_target_balance_power
                ),
                max_weight=(
                    self._env_action_model_policy_improvement_tail_target_balance_max_weight
                ),
            )
        selected_balance_weights = [
            target_balance_weight_by_index[index]
            for index in selected_indices
        ]
        stats.update(
            {
                "target_balance_weights": target_balance_weights,
                "target_balance_weight_min": min(selected_balance_weights),
                "target_balance_weight_mean": float(
                    fmean(selected_balance_weights)
                ),
                "target_balance_weight_max": max(selected_balance_weights),
            }
        )
        stats["policy_kl_before"] = self._mean_env_action_model_policy_kl(
            semantic_states=semantic_states,
            run_metadata_by_row=run_metadata_by_row,
            rollout=rollout,
            action_masks=action_masks,
            selected_indices=selected_indices,
        )
        batch_size = max(1, min(self._batch_size, len(selected_indices)))
        loss_total = 0.0
        update_steps = 0
        executed_epochs = 0
        early_stop_triggered = False
        backtrack_count = 0
        rejected_epoch_count = 0
        trainable_adapter_prefixes = (
            (
                "digital_twin_planning_adapter.",
                "outcome_recovery_adapter.",
                "outcome_context_residual_adapter.",
            )
            if self._env_action_model_policy_improvement_tail_planning_adapter_only
            else (
                "outcome_recovery_adapter.",
                "outcome_context_residual_adapter.",
            )
        )
        residual_parameters = [
            parameter
            for parameter_name, parameter in self._network.named_parameters()
            if parameter_name.startswith(trainable_adapter_prefixes)
        ]
        residual_optimizer: torch.optim.Optimizer | None = None
        if (
            self._env_action_model_policy_improvement_tail_residual_optimizer_enabled
            and residual_parameters
            and (
                self._env_action_model_policy_improvement_tail_adapter_only
                or self._env_action_model_policy_improvement_tail_planning_adapter_only
            )
        ):
            residual_optimizer = torch.optim.Adam(
                residual_parameters,
                lr=(
                    self._env_action_model_policy_improvement_tail_residual_learning_rate
                ),
            )
        active_optimizer = (
            residual_optimizer
            if residual_optimizer is not None
            else self._optimizer
        )
        epoch_index = 0
        while (
            epoch_index
            < self._env_action_model_policy_improvement_tail_epochs
        ):
            epoch_network_state = None
            epoch_optimizer_state = None
            epoch_loss_total_before = loss_total
            epoch_update_steps_before = update_steps
            if (
                self._env_action_model_policy_improvement_tail_max_policy_kl
                > 0.0
            ):
                epoch_network_state = deepcopy(
                    self._network.state_dict()
                )
                epoch_optimizer_state = deepcopy(
                    active_optimizer.state_dict()
                )
            permutation = torch.randperm(
                len(selected_indices),
                device=self._device,
            )
            for start_index in range(0, len(selected_indices), batch_size):
                batch_positions = permutation[
                    start_index : start_index + batch_size
                ].detach().cpu().tolist()
                batch_indices = [
                    selected_indices[int(position)]
                    for position in batch_positions
                ]
                batch_outputs = [
                    self._forward_policy(
                        semantic_states[index],
                        run_metadata=run_metadata_by_row[index],
                    )
                    for index in batch_indices
                ]
                tail_loss, _ = (
                    self._compute_env_action_model_policy_improvement_loss(
                        batch_outputs=batch_outputs,
                        batch_rows=[rollout[index] for index in batch_indices],
                    batch_action_masks=[
                        action_masks[index] for index in batch_indices
                    ],
                    logit_projection_enabled=(
                        self._env_action_model_policy_improvement_tail_logit_projection_enabled
                    ),
                    sample_weights=[
                        target_balance_weight_by_index[index]
                        for index in batch_indices
                    ],
                )
                )
                weighted_tail_loss = (
                    self._env_action_model_policy_improvement_tail_coef
                    * tail_loss
                )
                active_optimizer.zero_grad()
                weighted_tail_loss.backward()
                if (
                    self._env_action_model_policy_improvement_tail_adapter_only
                    or self._env_action_model_policy_improvement_tail_planning_adapter_only
                ):
                    for parameter_name, parameter in self._network.named_parameters():
                        if not parameter_name.startswith(
                            trainable_adapter_prefixes
                        ):
                            parameter.grad = None
                nn.utils.clip_grad_norm_(
                    (
                        residual_parameters
                        if residual_optimizer is not None
                        else self._network.parameters()
                    ),
                    max_norm=self._max_grad_norm,
                )
                active_optimizer.step()
                loss_total += float(tail_loss.item())
                update_steps += 1
            if (
                self._env_action_model_policy_improvement_tail_max_policy_kl
                > 0.0
            ):
                current_policy_kl = (
                    self._mean_env_action_model_policy_kl(
                        semantic_states=semantic_states,
                        run_metadata_by_row=run_metadata_by_row,
                        rollout=rollout,
                        action_masks=action_masks,
                        selected_indices=selected_indices,
                    )
                )
                if (
                    current_policy_kl
                    >= self._env_action_model_policy_improvement_tail_max_policy_kl
                ):
                    if epoch_network_state is not None:
                        self._network.load_state_dict(epoch_network_state)
                    if epoch_optimizer_state is not None:
                        active_optimizer.load_state_dict(epoch_optimizer_state)
                    loss_total = epoch_loss_total_before
                    update_steps = epoch_update_steps_before
                    rejected_epoch_count += 1
                    if (
                        residual_optimizer is not None
                        and backtrack_count
                        < self._env_action_model_policy_improvement_tail_residual_max_backtracks
                    ):
                        current_learning_rate = float(
                            active_optimizer.param_groups[0]["lr"]
                        )
                        next_learning_rate = max(
                            current_learning_rate
                            * self._env_action_model_policy_improvement_tail_residual_backtrack_factor,
                            self._env_action_model_policy_improvement_tail_residual_min_learning_rate,
                        )
                        if next_learning_rate < current_learning_rate - 1e-12:
                            for parameter_group in active_optimizer.param_groups:
                                parameter_group["lr"] = next_learning_rate
                            backtrack_count += 1
                            continue
                    early_stop_triggered = True
                    break
            executed_epochs += 1
            epoch_index += 1
        residual_learning_rate_final = (
            float(residual_optimizer.param_groups[0]["lr"])
            if residual_optimizer is not None
            else 0.0
        )
        stats.update(
            {
                "loss": loss_total / max(update_steps, 1),
                "update_steps": update_steps,
                "executed_epochs": executed_epochs,
                "early_stop_triggered": early_stop_triggered,
                "residual_optimizer_used": residual_optimizer is not None,
                "residual_learning_rate_initial": (
                    self._env_action_model_policy_improvement_tail_residual_learning_rate
                    if residual_optimizer is not None
                    else 0.0
                ),
                "residual_learning_rate_final": residual_learning_rate_final,
                "backtrack_count": backtrack_count,
                "rejected_epoch_count": rejected_epoch_count,
                "policy_kl_after": (
                    self._mean_env_action_model_policy_kl(
                        semantic_states=semantic_states,
                        run_metadata_by_row=run_metadata_by_row,
                        rollout=rollout,
                        action_masks=action_masks,
                        selected_indices=selected_indices,
                    )
                ),
            }
        )
        return stats

    @staticmethod
    def _semantic_state_recovery_signal(
        semantic_state: dict[str, Any],
    ) -> float:
        memory = semantic_state.get("algorithm_memory", {})
        if not isinstance(memory, dict):
            return 0.0
        return max(
            _clamp01(
                float(memory.get("failed_prepare_streak", 0) or 0) / 8.0
            ),
            _clamp01(
                float(memory.get("no_progress_streak", 0) or 0) / 8.0
            ),
            float(bool(memory.get("last_handoff_failed", False))),
            float(bool(memory.get("last_stall", False))),
        )

    def _net_utility_cost_signal(self, row: dict[str, Any]) -> float:
        if not self._net_utility_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        backhaul_units = max(
            float(metrics.get("backhaul_traffic_cost", 0.0) or 0.0),
            0.0,
        ) / self._net_utility_backhaul_normalizer
        migration_cost = max(
            float(metrics.get("adapter_state_migration_overhead", 0.0) or 0.0),
            0.0,
        )
        expired_prefetch = float(bool(metrics.get("prefetch_expired_miss", False)))
        idle_prefetch = float(
            self._row_window_class(row) == "idle_or_sparse"
            and bool(metrics.get("predictive_prefetch_requested", False))
            and not self._row_mechanism_success(metrics)
        )
        failed_mechanism = float(self._row_failed_mechanism_attempt(row, metrics))
        return float(
            backhaul_units
            + migration_cost
            + expired_prefetch
            + idle_prefetch
            + failed_mechanism
        )

    def _update_net_utility_cost_dual(self, cost_values: np.ndarray) -> None:
        if (
            not self._net_utility_prd_enabled
            or not self._net_utility_cost_dual_enabled
            or self._net_utility_cost_dual_lr <= 0.0
            or self._net_utility_cost_dual_max <= 0.0
            or len(cost_values) <= 0
        ):
            return
        observed_cost = float(cost_values.mean())
        self._net_utility_cost_dual = min(
            self._net_utility_cost_dual_max,
            max(
                0.0,
                self._net_utility_cost_dual
                + self._net_utility_cost_dual_lr
                * (observed_cost - self._net_utility_cost_target),
            ),
        )

    def _row_window_class(self, row: dict[str, Any]) -> str:
        action_info = dict(row.get("action_info", {}))
        option_info = dict(action_info.get("option_gate", {}))
        return str(
            option_info.get(
                "window_class",
                row.get("decision_info", {}).get("run_metadata", {}).get("window_class", "unknown"),
            )
        )

    def _row_mechanism_success(self, metrics: dict[str, Any]) -> bool:
        return bool(
            metrics.get("mechanism_success_strict", False)
            or metrics.get("handoff_ready", False)
            or metrics.get("prefetch_validated_hit", False)
            or float(metrics.get("mechanism_success_rate", 0.0) or 0.0) > 0.0
        )

    def _row_failed_mechanism_attempt(self, row: dict[str, Any], metrics: dict[str, Any]) -> bool:
        action_info = dict(row.get("action_info", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(action_info.get("head_actions", {}).get("event", 0) or 0)
        attempted = bool(
            metrics.get("mechanism_attempt_selected", False)
            or metrics.get("predictive_prefetch_requested", False)
            or metrics.get("migration_prepare_requested", False)
            or final_action in {1, 4}
            or event_action == 1
        )
        return bool(attempted and not self._row_mechanism_success(metrics))

    def _env_action_recovery_risk_score(self, row: dict[str, Any], env_action: int) -> float:
        if self._env_action_risk_adjusted_recovery_coef <= 0.0:
            return 0.0
        env_action = int(env_action)
        if env_action not in {1, 4}:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        window_class = self._row_window_class(row)
        mechanism_context = bool(
            window_class == "mechanism_activating"
            or self._handoff_risk_context(row, metrics)
            or self._row_sparse_recovery_context(row, metrics=metrics)
        )
        if not mechanism_context:
            return 0.0
        continuity = _clamp01(self._metric_float(metrics, "workflow_continuity_rate", 1.0))
        ready_rate = _clamp01(
            self._metric_float(
                metrics,
                "handoff_ready_rate",
                self._metric_float(metrics, "handoff_ready_ratio", 0.0),
            )
        )
        validated_hits = max(self._metric_float(metrics, "prefetch_validated_hit_count"), 0.0)
        migration_failed = max(self._metric_float(metrics, "migration_failed_count"), 0.0)
        cache_miss_penalty = max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0)
        workflow_unfinished = max(self._metric_float(metrics, "workflow_unfinished_count"), 0.0)
        handoff_failure = _clamp01(self._metric_float(metrics, "handoff_failure_rate", 0.0))
        failed_mechanism = float(self._row_failed_mechanism_attempt(row, metrics))
        mechanism_success = float(self._row_mechanism_success(metrics))
        risk_pressure = _clamp01(
            0.34 * failed_mechanism
            + 0.22 * min(migration_failed / 8.0, 1.0)
            + 0.18 * min(cache_miss_penalty / 12.0, 1.0)
            + 0.16 * min(workflow_unfinished, 1.0)
            + 0.16 * handoff_failure
        )
        support = _clamp01(
            0.30 * mechanism_success
            + 0.26 * ready_rate
            + 0.24 * continuity
            + 0.20 * min(validated_hits / 4.0, 1.0)
        )
        return _clamp01(risk_pressure - 0.45 * support)

    def _row_sparse_recovery_context(
        self,
        row: dict[str, Any],
        *,
        metrics: dict[str, Any] | None = None,
        action_info: dict[str, Any] | None = None,
    ) -> bool:
        if self._row_window_class(row) != "idle_or_sparse":
            return False
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {})) if metrics is None else metrics
        action_info = dict(row.get("action_info", {})) if action_info is None else action_info
        option_info = dict(action_info.get("option_gate", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        local_pressure = max(self._metric_float(metrics, "local_exec_count"), 0.0)
        current_service_pressure = max(self._metric_float(metrics, "current_rsu_exec_count"), 0.0)
        predicted_signal = bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        return bool(
            bool(option_info.get("idle_recovery_context", False))
            or bool(option_info.get("coverage_recovery_no_rsu", False))
            or predicted_signal
            or local_pressure > max(current_service_pressure, 0.0)
            or final_action == 2
        )

    def _sparse_handoff_realization_credit(
        self,
        row: dict[str, Any],
        *,
        env_action: int,
    ) -> float:
        if not self._sparse_handoff_realization_credit_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        if not self._row_sparse_recovery_context(
            row,
            metrics=metrics,
            action_info=action_info,
        ):
            return 0.0

        prepare_score = _clamp01(self._metric_float(action_info, "prepare_window_score", 0.0))
        urgency = _clamp01(self._metric_float(action_info, "temporal_urgency", 0.0))
        confidence = _clamp01(self._metric_float(action_info, "prediction_confidence", 0.0))
        option_info = dict(action_info.get("option_gate", {}))
        context_strength = _clamp01(
            0.38 * prepare_score
            + 0.30 * urgency
            + 0.22 * confidence
            + 0.10
            * float(
                bool(action_info.get("gate_pass", False))
                or bool(option_info.get("idle_recovery_context", False))
            )
        )
        if context_strength + 1e-8 < self._sparse_handoff_realization_min_context:
            return 0.0

        ready_rate = _clamp01(
            self._metric_float(
                metrics,
                "handoff_ready_rate",
                self._metric_float(metrics, "handoff_ready_ratio", 0.0),
            )
        )
        mechanism_success_rate = _clamp01(
            self._metric_float(metrics, "mechanism_success_rate", 0.0)
        )
        prefetch_hit_count = max(self._metric_float(metrics, "prefetch_validated_hit_count"), 0.0)
        prefetch_hit_rate = _clamp01(
            self._metric_float(metrics, "prefetch_validated_hit_rate", 0.0)
        )
        migration_success_count = max(self._metric_float(metrics, "migration_success_count"), 0.0)
        migration_failed_count = max(self._metric_float(metrics, "migration_failed_count"), 0.0)
        failed_mechanism = float(self._row_failed_mechanism_attempt(row, metrics))
        realized_score = _clamp01(
            0.34 * float(self._row_mechanism_success(metrics))
            + 0.24 * ready_rate
            + 0.18 * mechanism_success_rate
            + 0.14 * min(prefetch_hit_count / 4.0, 1.0)
            + 0.10 * min(migration_success_count / 2.0, 1.0)
            + 0.08 * prefetch_hit_rate
        )
        failed_prepare_pressure = _clamp01(
            0.52 * failed_mechanism
            + 0.22 * min(migration_failed_count / 4.0, 1.0)
            + 0.16 * float(ready_rate <= 1e-8)
            + 0.10 * float(prefetch_hit_count <= 1e-8)
        )

        env_action = int(env_action)
        context_scale = 0.58 + 0.42 * context_strength
        if env_action in {1, 4}:
            return (
                self._sparse_handoff_realization_success_bonus * realized_score
                + self._sparse_handoff_realization_ready_bonus * ready_rate
                + self._sparse_handoff_realization_prefetch_bonus * min(prefetch_hit_count / 4.0, 1.0)
                - self._sparse_handoff_realization_failed_prepare_penalty
                * failed_prepare_pressure
                * (1.0 - 0.45 * realized_score)
            ) * context_scale
        if env_action == 2:
            return -self._sparse_handoff_realization_local_penalty * context_scale
        if env_action in {0, 3}:
            idle_relief = max(0.0, failed_prepare_pressure - realized_score)
            return 0.22 * self._sparse_handoff_realization_local_penalty * idle_relief * context_scale
        return 0.0

    def _service_continuity_counterfactual_credit(
        self,
        row: dict[str, Any],
        *,
        env_action: int,
        window_class: str,
        context_strength: float,
        timing_support: float,
        ready_score: float = 0.0,
    ) -> float | None:
        if not self._service_continuity_teacher_enabled:
            return None
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        predicted_signal = bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        mechanism_success = self._row_mechanism_success(metrics)
        ready_rate = self._metric_float(
            metrics,
            "handoff_ready_rate",
            self._metric_float(metrics, "handoff_ready_ratio", 0.0),
        )
        continuity = _clamp01(self._metric_float(metrics, "workflow_continuity_rate", 1.0))
        adapter_miss = max(self._metric_float(metrics, "adapter_miss_count"), 0.0)
        migration_failed = max(self._metric_float(metrics, "migration_failed_count"), 0.0)
        local_pressure = max(self._metric_float(metrics, "local_exec_count"), 0.0)
        current_service_pressure = max(self._metric_float(metrics, "current_rsu_exec_count"), 0.0)
        failed_mechanism = self._row_failed_mechanism_attempt(row, metrics)
        adapter_miss_pressure = self._env_action_adapter_miss_pressure(row, metrics=metrics)
        prepare_context = bool(
            window_class == "mechanism_activating"
            or predicted_signal
            or mechanism_success
            or ready_rate > 0.0
            or local_pressure > max(current_service_pressure, 0.0)
            or float(timing_support) >= self._service_continuity_min_prepare_context
        )
        env_action = int(env_action)
        if adapter_miss_pressure > 0.0:
            pressure_scale = 0.74 + 0.26 * _clamp01(adapter_miss_pressure)
            if env_action == 0:
                return self._env_action_adapter_miss_counterfactual_coef * (
                    0.92 + 0.38 * pressure_scale
                )
            if env_action == 3:
                return -self._env_action_adapter_miss_counterfactual_coef * (
                    0.86 + 0.34 * pressure_scale
                )
            if env_action == 2:
                return -self._service_continuity_local_penalty * (
                    0.82 + 0.18 * pressure_scale
                )
            if env_action in {1, 4}:
                if self._handoff_alignment_barrier_enabled:
                    safe_prepare = bool(
                        mechanism_success
                        and ready_rate >= 0.55
                        and continuity >= 0.86
                        and adapter_miss <= 1.0
                    )
                    if not safe_prepare:
                        barrier_penalty = (
                            self._handoff_alignment_barrier_prepare_penalty
                            if env_action == 4
                            else self._handoff_alignment_barrier_prefetch_penalty
                        )
                        return -barrier_penalty * (
                            0.35
                            + 0.45 * _clamp01(adapter_miss_pressure)
                            + 0.20 * min(migration_failed / 6.0, 1.0)
                        )
                if predicted_signal or mechanism_success or ready_rate > 0.0:
                    return self._service_continuity_prepare_bonus * (
                        0.46 + 0.22 * _clamp01(context_strength) + 0.12 * _clamp01(ready_rate)
                    )
                return -0.38 * self._env_action_adapter_miss_counterfactual_coef * pressure_scale
        if env_action == 2:
            return -self._service_continuity_local_penalty * (
                0.75
                + 0.25 * (1.0 - _clamp01(timing_support))
                + 0.20 * float(prepare_context)
                + 0.18 * min(local_pressure / 8.0, 1.0)
            )
        if env_action == 3:
            return self._service_continuity_current_bonus * (
                0.80
                + 0.12 * float(not prepare_context)
                + 0.12 * _clamp01(ready_score)
                + 0.10 * min(local_pressure / 8.0, 1.0)
            )
        if env_action == 0:
            return 0.45 * self._service_continuity_current_bonus * (
                0.70 + 0.15 * float(not prepare_context)
            )
        if env_action in {1, 4}:
            if prepare_context:
                return self._service_continuity_prepare_bonus * (
                    0.62
                    + 0.45 * _clamp01(context_strength)
                    + 0.20 * float(mechanism_success)
                    + 0.12 * _clamp01(ready_rate)
                    + 0.14 * min(local_pressure / 8.0, 1.0)
                    - 0.10 * float(failed_mechanism and not mechanism_success)
                )
            return -0.60 * self._counterfactual_teacher_invalid_mechanism_penalty * (
                0.85 + 0.15 * (1.0 - _clamp01(timing_support))
            )
        return 0.0

    def _env_action_adapter_miss_pressure(
        self,
        row: dict[str, Any],
        *,
        metrics: dict[str, Any] | None = None,
    ) -> float:
        if self._env_action_adapter_miss_counterfactual_coef <= 0.0:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {})) if metrics is None else metrics
        cache_miss_penalty = max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0)
        adapter_miss = max(self._metric_float(metrics, "adapter_miss_count"), 0.0)
        stall = float(self._metric_bool(metrics, "stall_occurred"))
        cache_hit = bool(
            self._metric_bool(metrics, "cache_hit")
            or self._metric_bool(metrics, "warm_hit")
            or self._metric_bool(metrics, "prefetch_validated_hit")
        )
        if cache_hit and cache_miss_penalty <= 1e-8 and adapter_miss <= 1e-8:
            return 0.0
        action_info = dict(row.get("action_info", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        current_service_pressure = max(self._metric_float(metrics, "current_rsu_exec_count"), 0.0)
        no_prediction = not bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        current_cache_repair_context = bool(
            final_action in {0, 3}
            or current_service_pressure > 0.0
            or no_prediction
            or self._row_window_class(row) == "mechanism_activating"
        )
        if not current_cache_repair_context:
            return 0.0
        pressure = (
            0.42 * min(cache_miss_penalty / 1.2, 1.0)
            + 0.28 * min(adapter_miss, 1.0)
            + 0.20 * stall
            + 0.10 * float(not cache_hit)
        )
        return _clamp01(pressure)

    def _counterfactual_teacher_action_credit(self, row: dict[str, Any], env_action: int) -> float:
        if not self._counterfactual_teacher_prd_enabled:
            return 0.0
        action_info = dict(row.get("action_info", {}))
        window_class = self._row_window_class(row)
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        timing_support = max(prepare_score, urgency)
        context_strength = 0.45 * prepare_score + 0.35 * urgency + 0.20 * confidence
        if bool(action_info.get("gate_pass", False)):
            context_strength += 0.20
        context_strength = max(context_strength, 0.15 if window_class == "mechanism_activating" else 0.0)
        decision_info = dict(row.get("decision_info", {}))
        semantic_state = decision_info.get("semantic_state") if isinstance(decision_info, dict) else None
        wait_context: dict[str, Any] = {}
        if isinstance(semantic_state, dict):
            wait_context = self._digital_twin_wait_readiness_context(
                semantic_state,
                timing_support=timing_support,
                boundary_urgency=urgency,
                handoff_context=bool(
                    window_class == "mechanism_activating"
                    or action_info.get("predicted_handoff_target_valid", False)
                    or action_info.get("raw_handoff_candidate", False)
                ),
            )
        adaptive_wait_preferred = bool(
            self._digital_twin_policy_prior_adaptive_wait_enabled
            and wait_context.get("wait_preferred", False)
        )
        ready_score = float(wait_context.get("ready_score", 0.0) or 0.0)

        env_action = int(env_action)
        service_credit = self._service_continuity_counterfactual_credit(
            row,
            env_action=env_action,
            window_class=window_class,
            context_strength=context_strength,
            timing_support=timing_support,
            ready_score=ready_score,
        )
        if service_credit is not None:
            return float(service_credit)
        if window_class == "mechanism_activating":
            if adaptive_wait_preferred:
                if env_action == 2:
                    return self._counterfactual_teacher_local_bonus * (
                        0.90 + 0.40 * ready_score + 0.20 * (1.0 - timing_support)
                    )
                if env_action in {1, 4}:
                    return -0.35 * self._counterfactual_teacher_invalid_mechanism_penalty * (
                        0.50 + ready_score
                    )
                return -0.35 * self._counterfactual_teacher_current_rsu_penalty * (
                    0.50 + ready_score
                )
            if env_action in {1, 4}:
                return self._counterfactual_teacher_mechanism_bonus * (0.75 + 0.35 * context_strength)
            if env_action == 2:
                return -0.50 * self._counterfactual_teacher_missed_prepare_penalty * (0.5 + timing_support)
            return -self._counterfactual_teacher_missed_prepare_penalty * (0.45 + context_strength)

        if window_class in {"idle_or_sparse", "active_non_mechanism"}:
            if env_action == 2:
                return self._counterfactual_teacher_local_bonus * (1.0 + 0.25 * (1.0 - timing_support))
            if env_action in {1, 4}:
                return -self._counterfactual_teacher_invalid_mechanism_penalty * (1.0 + 0.25 * (1.0 - timing_support))
            if env_action in {0, 3}:
                return -self._counterfactual_teacher_current_rsu_penalty * max(0.35, 1.0 - timing_support)
        return 0.0

    def _counterfactual_teacher_option_advantage(
        self,
        row: dict[str, Any],
        *,
        option_probs: torch.Tensor | None,
        option_mask: list[bool] | None,
    ) -> float:
        if not self._counterfactual_teacher_prd_enabled:
            return 0.0
        action_info = dict(row.get("action_info", {}))
        option_info = dict(action_info.get("option_gate", {}))
        option_action = int(option_info.get("option_action", 0) or 0)
        base_env_action = int(
            option_info.get(
                "base_env_action",
                action_info.get("projected_env_action", action_info.get("final_env_action", row.get("action", 0))),
            )
            or 0
        )
        option_actions_raw = option_info.get("option_actions", {})
        option_actions: dict[int, int] = {}
        if isinstance(option_actions_raw, dict):
            for raw_key, raw_value in option_actions_raw.items():
                try:
                    option_actions[int(raw_key)] = int(raw_value)
                except (TypeError, ValueError):
                    continue
        valid_count = min(self._option_gate_count, len(option_mask or [])) if option_mask else self._option_gate_count
        utilities: list[float] = []
        probabilities: list[float] = []
        probs = option_probs.detach().cpu().tolist() if option_probs is not None else []
        for option_index in range(self._option_gate_count):
            is_valid = True
            if option_mask is not None and option_index < len(option_mask):
                is_valid = bool(option_mask[option_index])
            elif option_mask is not None:
                is_valid = False
            if option_index >= valid_count:
                is_valid = False
            option_env_action = int(option_actions.get(option_index, base_env_action))
            utilities.append(self._counterfactual_teacher_action_credit(row, option_env_action) if is_valid else 0.0)
            probabilities.append(float(probs[option_index]) if option_index < len(probs) and is_valid else 0.0)
        probability_sum = float(sum(probabilities))
        if probability_sum <= 1e-8:
            valid_indices = [
                index
                for index in range(self._option_gate_count)
                if option_mask is None or (index < len(option_mask) and bool(option_mask[index]))
            ]
            if not valid_indices:
                return 0.0
            uniform_prob = 1.0 / float(len(valid_indices))
            probabilities = [uniform_prob if index in valid_indices else 0.0 for index in range(self._option_gate_count)]
        else:
            probabilities = [probability / probability_sum for probability in probabilities]
        expected_credit = float(sum(probability * utility for probability, utility in zip(probabilities, utilities)))
        selected_credit = float(utilities[option_action]) if 0 <= option_action < len(utilities) else 0.0
        return float(selected_credit - expected_credit)

    def _metric_bool(self, metrics: dict[str, Any], field_name: str) -> bool:
        value = metrics.get(field_name, False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _metric_float(self, metrics: dict[str, Any], field_name: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(field_name, default) or default)
        except (TypeError, ValueError):
            return float(default)

    def _handoff_risk_context(self, row: dict[str, Any], metrics: dict[str, Any]) -> bool:
        action_info = dict(row.get("action_info", {}))
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        handoff_event_count = float(metrics.get("handoff_event_count", 0.0) or 0.0)
        return bool(
            handoff_event_count > 0.0
            or self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or (
                confidence >= self._handoff_risk_confidence_threshold
                and (prepare_score >= 0.35 or urgency >= 0.45)
            )
        )

    def _handoff_risk_cost_signal(self, row: dict[str, Any]) -> float:
        if not self._handoff_risk_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        high_risk = self._handoff_risk_context(row, metrics)
        handoff_failed = self._metric_bool(metrics, "handoff_failed") or float(
            metrics.get("handoff_failure_rate", 0.0) or 0.0
        ) > 0.0
        handoff_ready = self._metric_bool(metrics, "handoff_ready") or float(
            metrics.get("handoff_ready_ratio", metrics.get("handoff_ready_rate", 0.0)) or 0.0
        ) > 0.0
        action_info = dict(row.get("action_info", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        unprepared_high_risk = high_risk and not handoff_ready and final_action not in {1, 4}
        return float(handoff_failed) + 0.35 * float(unprepared_high_risk)

    def _update_handoff_risk_cost_dual(self, cost_values: np.ndarray) -> None:
        if (
            not self._handoff_risk_prd_enabled
            or not self._handoff_risk_cost_dual_enabled
            or self._handoff_risk_cost_dual_lr <= 0.0
            or self._handoff_risk_cost_dual_max <= 0.0
            or len(cost_values) <= 0
        ):
            return
        observed_cost = float(cost_values.mean())
        self._handoff_risk_cost_dual = min(
            self._handoff_risk_cost_dual_max,
            max(
                0.0,
                self._handoff_risk_cost_dual
                + self._handoff_risk_cost_dual_lr
                * (observed_cost - self._handoff_risk_cost_target),
            ),
        )

    def _handoff_risk_prd_credit(self, row: dict[str, Any]) -> float:
        if not self._handoff_risk_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        high_risk = self._handoff_risk_context(row, metrics)
        if not high_risk:
            return 0.0
        action_info = dict(row.get("action_info", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(action_info.get("head_actions", {}).get("event", 0) or 0)
        handoff_failed = self._metric_bool(metrics, "handoff_failed") or float(
            metrics.get("handoff_failure_rate", 0.0) or 0.0
        ) > 0.0
        handoff_ready = self._metric_bool(metrics, "handoff_ready") or float(
            metrics.get("handoff_ready_ratio", metrics.get("handoff_ready_rate", 0.0)) or 0.0
        ) > 0.0
        prepare_realized = self._metric_bool(metrics, "migration_prepare_realized") or self._metric_bool(
            metrics,
            "migration_prepare_requested",
        )
        mechanism_success = self._row_mechanism_success(metrics)
        dual_scale = 1.0 + float(self._handoff_risk_cost_dual)
        bonus = (
            self._handoff_risk_ready_bonus * float(handoff_ready)
            + self._handoff_risk_prepare_bonus * float(prepare_realized)
            + 0.25 * float(mechanism_success)
        )
        if final_action in {1, 4} or event_action == 1:
            bonus += 0.12
        unprepared_penalty = self._handoff_risk_unprepared_penalty * float(
            not handoff_ready and final_action not in {1, 4}
        )
        failure_penalty = self._handoff_risk_failure_penalty * float(handoff_failed)
        return float(bonus - dual_scale * (failure_penalty + unprepared_penalty))

    def _mechanism_credit_prd_credit(self, row: dict[str, Any]) -> float:
        if not self._mechanism_credit_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        head_actions = action_info.get("head_actions", {})
        if not isinstance(head_actions, dict):
            head_actions = {}
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(head_actions.get("event", 0) or 0)
        window_class = self._row_window_class(row)

        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        gate_pass = bool(action_info.get("gate_pass", False))
        context_strength = _clamp01(
            0.42 * prepare_score
            + 0.30 * urgency
            + 0.22 * confidence
            + 0.06 * float(gate_pass)
        )

        predicted_signal = bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        handoff_event_count = max(
            self._metric_float(metrics, "handoff_event_count"),
            self._metric_float(metrics, "handoff_count"),
            0.0,
        )
        opportunity = bool(
            window_class == "mechanism_activating"
            or handoff_event_count > 0.0
            or predicted_signal
        )
        if opportunity:
            context_strength = max(context_strength, self._mechanism_credit_min_context)

        selected_prepare = bool(
            self._metric_bool(metrics, "mechanism_attempt_selected")
            or self._metric_bool(metrics, "predictive_prefetch_requested")
            or self._metric_bool(metrics, "migration_prepare_requested")
            or final_action in {1, 4}
            or event_action == 1
        )
        prepare_realized = bool(
            self._metric_bool(metrics, "migration_prepare_realized")
            or self._metric_bool(metrics, "migration_prepare_requested")
        )
        handoff_ready = bool(
            self._metric_bool(metrics, "handoff_ready")
            or self._metric_float(
                metrics,
                "handoff_ready_ratio",
                self._metric_float(metrics, "handoff_ready_rate", 0.0),
            )
            > 0.0
        )
        prefetch_hit = bool(
            self._metric_bool(metrics, "prefetch_validated_hit")
            or self._metric_float(metrics, "prefetch_validated_hit_rate") > 0.0
        )
        mechanism_success = self._row_mechanism_success(metrics)
        handoff_failed = bool(
            self._metric_bool(metrics, "handoff_failed")
            or self._metric_float(metrics, "handoff_failure_rate") > 0.0
        )

        credit = 0.0
        if opportunity:
            if selected_prepare:
                credit += 0.08 + 0.20 * context_strength
                credit += self._mechanism_credit_success_bonus * float(mechanism_success)
                credit += self._mechanism_credit_prepare_bonus * float(prepare_realized)
                credit += self._mechanism_credit_ready_bonus * float(handoff_ready)
                credit += self._mechanism_credit_prefetch_hit_bonus * float(prefetch_hit)
                if handoff_failed and not mechanism_success:
                    credit -= self._mechanism_credit_miss_penalty * (0.80 + context_strength)
                if not prepare_realized and not mechanism_success:
                    credit -= 0.25 * self._mechanism_credit_miss_penalty * (
                        1.0 - min(context_strength, 1.0)
                    )
            else:
                credit -= self._mechanism_credit_miss_penalty * (0.65 + context_strength)
                if handoff_failed:
                    credit -= 0.50 * self._mechanism_credit_miss_penalty
        elif selected_prepare and not mechanism_success:
            timing_support = max(prepare_score, urgency)
            stale_scale = 1.0 + max(self._mechanism_credit_min_context - timing_support, 0.0) / max(
                self._mechanism_credit_min_context,
                1e-6,
            )
            credit -= self._mechanism_credit_false_positive_penalty * stale_scale

        if self._mechanism_credit_clip > 0.0:
            credit = max(
                -self._mechanism_credit_clip,
                min(self._mechanism_credit_clip, credit),
            )
        return float(credit)

    def _delayed_mechanism_credit_values(self, rollout: list[dict[str, Any]]) -> np.ndarray:
        values = np.zeros(len(rollout), dtype=np.float32)
        if not rollout or not self._delayed_mechanism_credit_enabled:
            return values

        segment_start = 0
        for row_index, row in enumerate(rollout):
            if bool(row.get("terminated", False)) or bool(row.get("truncated", False)):
                self._accumulate_delayed_mechanism_credit_segment(
                    rollout,
                    values,
                    segment_start=segment_start,
                    segment_end=row_index + 1,
                )
                segment_start = row_index + 1
        if segment_start < len(rollout):
            self._accumulate_delayed_mechanism_credit_segment(
                rollout,
                values,
                segment_start=segment_start,
                segment_end=len(rollout),
            )

        if self._delayed_mechanism_credit_clip > 0.0:
            values = np.clip(
                values,
                -self._delayed_mechanism_credit_clip,
                self._delayed_mechanism_credit_clip,
            ).astype(np.float32)
        return values

    def _accumulate_delayed_mechanism_credit_segment(
        self,
        rollout: list[dict[str, Any]],
        values: np.ndarray,
        *,
        segment_start: int,
        segment_end: int,
    ) -> None:
        if segment_end <= segment_start:
            return
        horizon = max(int(self._delayed_mechanism_credit_horizon), 1)
        decay = max(0.0, min(float(self._delayed_mechanism_credit_decay), 1.0))
        outcome_signals = [
            self._delayed_mechanism_outcome_signal(rollout[index])
            for index in range(segment_start, segment_end)
        ]
        positive_outcome_indices = {
            segment_start + local_index
            for local_index, signal in enumerate(outcome_signals)
            if signal > 1e-8
        }

        for local_outcome_index, outcome_signal in enumerate(outcome_signals):
            if abs(outcome_signal) <= 1e-8:
                continue
            outcome_index = segment_start + local_outcome_index
            first_source_index = max(segment_start, outcome_index - horizon)
            for source_index in range(first_source_index, outcome_index + 1):
                source_row = rollout[source_index]
                selected_mechanism = self._row_selected_mechanism_action(source_row)
                opportunity = self._row_mechanism_credit_opportunity(source_row)
                if not selected_mechanism and not opportunity:
                    continue
                context_strength = self._row_mechanism_credit_context(source_row)
                lag = outcome_index - source_index
                temporal_weight = decay ** lag
                context_weight = 0.35 + 0.65 * context_strength
                if selected_mechanism:
                    values[source_index] += float(outcome_signal * temporal_weight * context_weight)
                elif outcome_signal < 0.0:
                    values[source_index] += float(
                        outcome_signal
                        * temporal_weight
                        * context_weight
                        * self._delayed_mechanism_credit_missed_prepare_scale
                    )

        if self._delayed_mechanism_credit_stale_penalty <= 0.0:
            return
        for source_index in range(segment_start, segment_end):
            source_row = rollout[source_index]
            if not self._row_selected_mechanism_action(source_row):
                continue
            context_strength = self._row_mechanism_credit_context(source_row)
            has_positive_outcome = any(
                positive_index >= source_index
                and positive_index <= min(segment_end - 1, source_index + horizon)
                for positive_index in positive_outcome_indices
            )
            if has_positive_outcome:
                continue
            opportunity = self._row_mechanism_credit_opportunity(source_row)
            if opportunity and context_strength >= self._delayed_mechanism_credit_context_gate:
                continue
            stale_gap = max(self._delayed_mechanism_credit_context_gate - context_strength, 0.0)
            values[source_index] -= float(
                self._delayed_mechanism_credit_stale_penalty
                * (0.75 + stale_gap)
            )

    def _delayed_mechanism_outcome_signal(self, row: dict[str, Any]) -> float:
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        handoff_ready = bool(
            self._metric_bool(metrics, "handoff_ready")
            or self._metric_float(
                metrics,
                "handoff_ready_ratio",
                self._metric_float(metrics, "handoff_ready_rate", 0.0),
            )
            > 0.0
        )
        mechanism_success = self._row_mechanism_success(metrics)
        handoff_failed = bool(
            self._metric_bool(metrics, "handoff_failed")
            or self._metric_float(metrics, "handoff_failure_rate") > 0.0
        )
        handoff_event_count = max(
            self._metric_float(metrics, "handoff_event_count"),
            self._metric_float(metrics, "handoff_count"),
            0.0,
        )
        prefetch_hit = bool(
            self._metric_bool(metrics, "prefetch_validated_hit")
            or self._metric_float(metrics, "prefetch_validated_hit_rate") > 0.0
        )
        prepare_realized = bool(
            self._metric_bool(metrics, "migration_prepare_realized")
            or self._metric_bool(metrics, "migration_prepare_requested")
        )

        signal = 0.0
        signal += self._delayed_mechanism_credit_ready_bonus * float(handoff_ready)
        signal += self._delayed_mechanism_credit_success_bonus * float(mechanism_success)
        signal += 0.35 * self._delayed_mechanism_credit_success_bonus * float(prefetch_hit)
        signal += 0.20 * self._delayed_mechanism_credit_success_bonus * float(prepare_realized)
        signal -= self._delayed_mechanism_credit_failure_penalty * float(handoff_failed)
        if handoff_event_count > 0.0 and not handoff_ready:
            signal -= 0.35 * self._delayed_mechanism_credit_failure_penalty
        return float(signal)

    def _row_selected_mechanism_action(self, row: dict[str, Any]) -> bool:
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        head_actions = action_info.get("head_actions", {})
        if not isinstance(head_actions, dict):
            head_actions = {}
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        return bool(
            final_action in {1, 4}
            or int(head_actions.get("event", 0) or 0) == 1
            or int(head_actions.get("slow", 0) or 0) == 2
            or self._metric_bool(metrics, "predictive_prefetch_requested")
            or self._metric_bool(metrics, "migration_prepare_requested")
        )

    def _row_mechanism_credit_context(self, row: dict[str, Any]) -> float:
        action_info = dict(row.get("action_info", {}))
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        gate_pass = bool(action_info.get("gate_pass", False))
        predicted_signal = bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        context = (
            0.38 * prepare_score
            + 0.26 * urgency
            + 0.22 * confidence
            + 0.08 * float(gate_pass)
            + 0.06 * float(predicted_signal)
        )
        if (
            self._row_window_class(row) == "mechanism_activating"
            and not self._delayed_mechanism_credit_strict_opportunity_enabled
        ):
            context = max(context, self._delayed_mechanism_credit_context_gate)
        return float(_clamp01(context))

    def _row_mechanism_credit_opportunity(self, row: dict[str, Any]) -> bool:
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        handoff_event_count = max(
            self._metric_float(metrics, "handoff_event_count"),
            self._metric_float(metrics, "handoff_count"),
            0.0,
        )
        predicted_signal = bool(
            self._metric_bool(metrics, "predicted_handoff_signal")
            or self._metric_bool(metrics, "has_predicted_handoff_target")
            or bool(action_info.get("raw_handoff_candidate", False))
            or bool(action_info.get("predicted_handoff_target_valid", False))
        )
        window_class_opportunity = (
            self._row_window_class(row) == "mechanism_activating"
            and not self._delayed_mechanism_credit_strict_opportunity_enabled
        )
        return bool(
            window_class_opportunity
            or handoff_event_count > 0.0
            or predicted_signal
            or self._row_mechanism_credit_context(row) >= self._delayed_mechanism_credit_context_gate
        )

    def _tail_risk_prd_credit(self, row: dict[str, Any], *, reward_floor: float | None = None) -> float:
        if not self._tail_risk_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(action_info.get("head_actions", {}).get("event", 0) or 0)
        window_class = self._row_window_class(row)

        reward = float(row.get("reward", 0.0) or 0.0)
        floor = 0.0 if reward_floor is None else float(reward_floor)
        reward_scale = max(abs(floor), abs(reward), 1.0)
        reward_shortfall = max(floor - reward, 0.0) / reward_scale

        service_delay_sum = max(
            self._metric_float(metrics, "service_delay_sum"),
            self._metric_float(metrics, "end_to_end_workflow_delay"),
            0.0,
        )
        service_pressure = min(service_delay_sum / 4.0, 1.5)
        cache_miss_penalty = max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0)
        cache_pressure = min(cache_miss_penalty / 2.4, 1.5)
        continuity_rate = self._metric_float(metrics, "workflow_continuity_rate", 1.0)
        continuity_loss = min(max(1.0 - continuity_rate, 0.0), 1.5)
        handoff_failed = self._metric_bool(metrics, "handoff_failed") or self._metric_float(
            metrics,
            "handoff_failure_rate",
        ) > 0.0
        handoff_ready = self._metric_bool(metrics, "handoff_ready") or self._metric_float(
            metrics,
            "handoff_ready_ratio",
            self._metric_float(metrics, "handoff_ready_rate", 0.0),
        ) > 0.0
        mechanism_attempt = bool(
            self._metric_bool(metrics, "mechanism_attempt_selected")
            or self._metric_bool(metrics, "predictive_prefetch_requested")
            or self._metric_bool(metrics, "migration_prepare_requested")
            or final_action in {1, 4}
            or event_action == 1
        )
        strict_mechanism_success = bool(
            self._metric_bool(metrics, "mechanism_success_strict")
            or self._metric_bool(metrics, "prefetch_validated_hit")
            or handoff_ready
        )
        failed_mechanism = self._row_failed_mechanism_attempt(row, metrics)
        high_risk = self._handoff_risk_context(row, metrics)

        risk = (
            self._tail_risk_reward_shortfall_coef * reward_shortfall
            + self._tail_risk_service_coef * service_pressure
            + 0.45 * self._tail_risk_service_coef * cache_pressure
            + self._tail_risk_continuity_coef * continuity_loss
            + self._tail_risk_handoff_failure_coef * float(handoff_failed)
            + self._tail_risk_failed_mechanism_coef * float(failed_mechanism)
        )
        redundant_mechanism = bool(mechanism_attempt and not strict_mechanism_success)
        if redundant_mechanism:
            tail_context = service_pressure + cache_pressure + continuity_loss + float(handoff_failed)
            context_scale = 0.70 + min(tail_context, 2.0)
            if window_class != "mechanism_activating" and not high_risk:
                context_scale += 0.35
            risk += self._tail_risk_redundant_mechanism_coef * context_scale

        success_credit = 0.0
        if mechanism_attempt and strict_mechanism_success and not handoff_failed:
            low_tail_pressure = max(0.0, 1.0 - min(service_pressure + continuity_loss, 1.0))
            success_credit = self._tail_risk_success_credit * (0.35 + 0.65 * low_tail_pressure)
        return float(success_credit - risk)

    def _opportunity_prd_credit(self, row: dict[str, Any], *, reward_floor: float | None = None) -> float:
        if not self._opportunity_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        window_class = self._row_window_class(row)
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(action_info.get("head_actions", {}).get("event", 0) or 0)

        reward = float(row.get("reward", 0.0) or 0.0)
        floor = 0.0 if reward_floor is None else float(reward_floor)
        reward_scale = max(abs(floor), abs(reward), 1.0)
        reward_surplus = max(reward - floor, 0.0) / reward_scale

        service_success = max(
            self._metric_float(metrics, "service_success_count"),
            self._metric_float(metrics, "workflow_completed_count"),
            0.0,
        )
        service_wait = max(self._metric_float(metrics, "service_wait_sum"), 0.0)
        workflow_unfinished = max(self._metric_float(metrics, "workflow_unfinished_count"), 0.0)
        adapter_hit = max(self._metric_float(metrics, "adapter_hit_count"), 0.0)
        adapter_miss = max(self._metric_float(metrics, "adapter_miss_count"), 0.0)
        warm_hit = max(self._metric_float(metrics, "adapter_warm_hit_count"), 0.0)
        cache_admission = max(self._metric_float(metrics, "cache_admission_count"), 0.0)
        exec_counts = {
            "local": max(self._metric_float(metrics, "local_exec_count"), 0.0),
            "current": max(self._metric_float(metrics, "current_rsu_exec_count"), 0.0),
            "next": max(self._metric_float(metrics, "next_rsu_exec_count"), 0.0),
            "neighbor": max(self._metric_float(metrics, "neighbor_rsu_exec_count"), 0.0),
            "cloud": max(self._metric_float(metrics, "cloud_exec_count"), 0.0),
        }
        exec_total = max(sum(exec_counts.values()), 1.0)
        observed_units = max(
            service_success + workflow_unfinished,
            service_success + adapter_miss,
            service_success + service_wait,
            exec_total,
            1.0,
        )
        success_rate = min(service_success / observed_units, 1.0)
        warm_rate = min(warm_hit / max(adapter_hit + adapter_miss, 1.0), 1.0)
        continuity_rate = max(min(self._metric_float(metrics, "workflow_continuity_rate", 1.0), 1.0), 0.0)
        service_quality = 0.45 * success_rate + 0.30 * warm_rate + 0.25 * continuity_rate

        service_delay_sum = max(
            self._metric_float(metrics, "service_delay_sum"),
            self._metric_float(metrics, "end_to_end_workflow_delay"),
            0.0,
        )
        delay_pressure = min(service_delay_sum / max(4.0 * observed_units, 1.0), 1.5)
        failed_service_pressure = min((service_wait + adapter_miss + workflow_unfinished) / observed_units, 1.5)
        backhaul_units = max(self._metric_float(metrics, "backhaul_traffic_cost"), 0.0) / 64.0
        mechanism_attempt = bool(
            self._metric_bool(metrics, "mechanism_attempt_selected")
            or self._metric_bool(metrics, "predictive_prefetch_requested")
            or self._metric_bool(metrics, "migration_prepare_requested")
            or final_action in {1, 4}
            or event_action == 1
        )
        mechanism_success = self._row_mechanism_success(metrics)
        handoff_failed = self._metric_bool(metrics, "handoff_failed") or self._metric_float(
            metrics,
            "handoff_failure_rate",
        ) > 0.0
        high_risk = self._handoff_risk_context(row, metrics)

        credit = (
            self._opportunity_reward_surplus_coef * reward_surplus
            + self._opportunity_service_success_coef * service_quality
            + self._opportunity_cache_hit_coef * warm_rate
            + self._opportunity_continuity_coef * continuity_rate
        )
        credit -= self._opportunity_delay_penalty_coef * delay_pressure
        credit -= self._opportunity_failed_service_penalty_coef * failed_service_pressure

        current_share = (exec_counts["current"] + exec_counts["next"] + exec_counts["neighbor"]) / exec_total
        local_share = exec_counts["local"] / exec_total
        if final_action in {0, 3} or current_share >= 0.65:
            credit += self._opportunity_current_rsu_efficiency_coef * (
                0.45 * current_share + 0.35 * warm_rate + 0.20 * continuity_rate - 0.35 * delay_pressure
            )
        if final_action == 2 or local_share >= 0.25:
            local_context = failed_service_pressure + 0.35 * float(not high_risk)
            credit += self._opportunity_local_fallback_coef * (
                0.45 * local_context + 0.25 * success_rate - 0.35 * delay_pressure
            )
            if window_class == "mechanism_activating" and high_risk and not mechanism_success:
                credit -= 0.35 * self._opportunity_local_fallback_coef * max(local_share, 0.25)

        if mechanism_success and not handoff_failed:
            credit += self._opportunity_mechanism_success_bonus * (0.50 + 0.50 * service_quality)
        if mechanism_attempt and not mechanism_success:
            redundant_backhaul = max(backhaul_units + 0.25 * cache_admission - 1.0, 0.0)
            credit -= self._opportunity_backhaul_penalty_coef * (0.40 + redundant_backhaul)
        elif not mechanism_success:
            credit -= self._opportunity_backhaul_penalty_coef * max(backhaul_units - 1.0, 0.0)
        return float(credit)

    def _idle_execution_prd_credit(self, row: dict[str, Any]) -> float:
        if not self._idle_execution_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        window_class = self._row_window_class(row)
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        timing_support = max(prepare_score, urgency)
        high_risk = self._handoff_risk_context(row, metrics)
        mechanism_success = self._row_mechanism_success(metrics)
        service_delay_sum = max(
            self._metric_float(metrics, "service_delay_sum"),
            self._metric_float(metrics, "end_to_end_workflow_delay"),
            0.0,
        )
        cache_miss_penalty = max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0)
        delay_pressure = min(service_delay_sum / 4.0, 1.0)
        cache_pressure = min(cache_miss_penalty / 2.4, 1.0)
        low_timing_context = bool(timing_support <= self._idle_execution_timing_threshold and not high_risk)

        credit = 0.0
        if window_class in {"idle_or_sparse", "active_non_mechanism"}:
            if low_timing_context and final_action == 2:
                credit += self._idle_execution_local_bonus * (1.0 + 0.35 * (1.0 - delay_pressure))
            if low_timing_context and final_action in {0, 3} and not mechanism_success:
                credit -= self._idle_execution_current_rsu_delay_coef * (0.50 + delay_pressure)
                if final_action == 0 and cache_pressure <= 1e-6:
                    credit -= 0.12
            if final_action in {1, 4} and not mechanism_success:
                stale_mechanism_scale = 1.0 + max(
                    self._idle_execution_timing_threshold - timing_support,
                    0.0,
                ) / max(self._idle_execution_timing_threshold, 1e-6)
                credit -= self._idle_execution_mechanism_penalty * stale_mechanism_scale
            if mechanism_success:
                credit += self._idle_execution_mechanism_preserve_bonus * (0.35 + cache_pressure)
            return float(credit)

        if window_class == "mechanism_activating":
            if high_risk and final_action in {1, 4}:
                credit += self._idle_execution_mechanism_preserve_bonus
            if high_risk and final_action == 2:
                credit -= 0.5 * self._idle_execution_local_bonus
        return float(credit)

    def _net_advantage_prepare_gate_credit(self, row: dict[str, Any]) -> float:
        if not self._net_advantage_prepare_gate_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        action_info = dict(row.get("action_info", {}))
        gate_info = dict(action_info.get("net_advantage_prepare_gate", {}))
        if not gate_info:
            return 0.0

        head_actions = action_info.get("head_actions", {})
        if not isinstance(head_actions, dict):
            head_actions = {}
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(head_actions.get("event", 0) or 0)
        slow_action = int(head_actions.get("slow", 0) or 0)
        selected_prepare = bool(
            final_action in {1, 4}
            or event_action == 1
            or self._metric_bool(metrics, "migration_prepare_requested")
            or self._metric_bool(metrics, "mechanism_attempt_selected")
        )
        selected_prefetch = bool(
            final_action == 1
            or slow_action == 2
            or self._metric_bool(metrics, "predictive_prefetch_requested")
        )
        selected_service_fill = bool(final_action == 0 or slow_action == 1)
        selected_current_execution = bool(final_action in {0, 3})
        selected_local_fallback = bool(final_action == 2 or int(head_actions.get("fast", 0) or 0) == 1)

        score = _clamp01(float(gate_info.get("net_advantage_score", 0.0) or 0.0))
        score_surplus = score - self._net_advantage_prepare_gate_min_score
        service_pressure = _clamp01(float(gate_info.get("service_pressure", 0.0) or 0.0))
        missing_current_adapter = bool(gate_info.get("missing_current_adapter", False))
        current_rsu_available = gate_info.get("current_rsu_id") is not None
        continuity = _clamp01(self._metric_float(metrics, "workflow_continuity_rate", 1.0))
        handoff_ready = bool(
            self._metric_bool(metrics, "handoff_ready")
            or self._metric_float(
                metrics,
                "handoff_ready_ratio",
                self._metric_float(metrics, "handoff_ready_rate", 0.0),
            )
            > 0.0
        )
        prefetch_hit = bool(
            self._metric_bool(metrics, "prefetch_validated_hit")
            or self._metric_float(metrics, "prefetch_validated_hit_rate") > 0.0
            or self._metric_float(metrics, "prefetch_validated_hit_count") > 0.0
        )
        mechanism_success = self._row_mechanism_success(metrics)
        handoff_failed = bool(
            self._metric_bool(metrics, "handoff_failed")
            or self._metric_float(metrics, "handoff_failure_rate") > 0.0
        )
        service_success = max(
            self._metric_float(metrics, "service_success_count"),
            self._metric_float(metrics, "workflow_completed_count"),
            0.0,
        )
        service_wait = max(self._metric_float(metrics, "service_wait_sum"), 0.0)
        workflow_unfinished = max(self._metric_float(metrics, "workflow_unfinished_count"), 0.0)
        adapter_miss = max(self._metric_float(metrics, "adapter_miss_count"), 0.0)
        observed_units = max(service_success + workflow_unfinished + adapter_miss + service_wait, 1.0)
        service_quality = _clamp01(
            0.45 * min(service_success / observed_units, 1.0)
            + 0.35 * continuity
            + 0.20 * float(prefetch_hit or handoff_ready)
        )
        backhaul_units = max(self._metric_float(metrics, "backhaul_traffic_cost"), 0.0) / 64.0
        migration_cost = max(self._metric_float(metrics, "adapter_state_migration_overhead"), 0.0)
        expired_prefetch = self._metric_bool(metrics, "prefetch_expired_miss")
        failed_mechanism = self._row_failed_mechanism_attempt(row, metrics)
        cost_pressure = min(
            backhaul_units
            + 0.75 * migration_cost
            + 0.80 * float(expired_prefetch)
            + 0.90 * float(failed_mechanism),
            3.0,
        ) / 3.0
        high_net_window = bool(score >= self._net_advantage_prepare_gate_min_score)
        coverage_recovery_context = bool(
            not current_rsu_available
            and (
                gate_info.get("predicted_target_valid", False)
                or gate_info.get("target_differs", False)
            )
        )
        coverage_recovery_scale = _clamp01(
            max(
                float(gate_info.get("coverage_recovery_scale", 0.0) or 0.0),
                score,
            )
        )
        coverage_recovery_final_guard_info = dict(
            action_info.get("coverage_recovery_final_guard", {})
        )
        coverage_recovery_final_guarded = bool(
            coverage_recovery_final_guard_info.get("guarded", False)
        )
        coverage_recovery_confidence = _clamp01(
            float(gate_info.get("prediction_confidence", 0.0) or 0.0)
        )

        credit = 0.0
        if selected_prepare or selected_prefetch:
            credit += 0.70 * score_surplus
            credit += 0.42 * service_quality
            credit += 0.36 * float(mechanism_success)
            credit += 0.22 * float(handoff_ready)
            credit += 0.18 * float(prefetch_hit)
            credit -= self._net_advantage_prepare_gate_cost_scale * (0.85 * cost_pressure)
            if not high_net_window:
                credit -= 0.65 * abs(score_surplus) * (1.0 + cost_pressure)
            if handoff_failed and not mechanism_success:
                credit -= 0.55
        else:
            if high_net_window and (handoff_failed or not handoff_ready):
                credit -= 0.45 + 0.55 * score_surplus
            elif not high_net_window:
                credit += 0.12 * abs(score_surplus) * (1.0 - min(cost_pressure, 1.0))
        if current_rsu_available and service_pressure > 1e-8:
            if selected_service_fill:
                credit += (
                    0.28 * self._net_advantage_prepare_gate_service_fill_scale * service_pressure
                    + 0.18 * service_quality
                    - 0.08 * cost_pressure
                )
            elif selected_current_execution:
                credit += (
                    0.18 * self._net_advantage_prepare_gate_service_fill_scale * service_pressure
                    + 0.14 * continuity
                    - 0.06 * cost_pressure
                )
            elif selected_local_fallback:
                credit -= (
                    0.32
                    * self._net_advantage_prepare_gate_local_penalty_scale
                    * service_pressure
                    * (1.15 if not missing_current_adapter else 1.0)
                    + 0.14 * min(adapter_miss / observed_units, 1.0)
                )
        if coverage_recovery_context:
            if selected_prepare:
                credit += self._coverage_recovery_gate_prepare_credit * (
                    0.55 + 0.45 * coverage_recovery_scale
                )
                credit += 0.24 * score + 0.16 * float(handoff_ready or mechanism_success)
                if final_action == 4:
                    credit += 0.18 * coverage_recovery_scale
                if coverage_recovery_final_guarded:
                    credit += self._coverage_recovery_gate_prepare_credit * (
                        0.18
                        + 0.22 * coverage_recovery_scale
                        + 0.10 * coverage_recovery_confidence
                    )
                if failed_mechanism and not handoff_ready:
                    credit -= 0.10 * min(cost_pressure, 1.0)
            if selected_local_fallback:
                credit -= self._coverage_recovery_gate_fallback_penalty * (
                    0.70 + 0.30 * coverage_recovery_scale
                )
                credit -= 0.30 * min(adapter_miss / observed_units, 1.0) + 0.22 * cost_pressure
            elif selected_current_execution and not selected_prepare:
                credit -= 0.24 * self._coverage_recovery_gate_fallback_penalty * (
                    0.60 + 0.40 * coverage_recovery_scale
                )
        return float(credit)

    def _service_completion_gate_credit(self, row: dict[str, Any]) -> float:
        if not self._service_completion_gate_enabled:
            return 0.0
        action_info = dict(row.get("action_info", {}))
        gate_info = dict(action_info.get("service_completion_gate", {}))
        if not bool(gate_info.get("active", False)):
            return 0.0

        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        head_actions = action_info.get("head_actions", {})
        if not isinstance(head_actions, dict):
            head_actions = {}
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        event_action = int(head_actions.get("event", 0) or 0)
        slow_action = int(head_actions.get("slow", 0) or 0)
        fast_action = int(head_actions.get("fast", 0) or 0)
        target_action = int(gate_info.get("target_action", 3) or 3)
        score = _clamp01(float(gate_info.get("service_completion_score", 0.0) or 0.0))
        service_pressure = _clamp01(float(gate_info.get("service_pressure", 0.0) or 0.0))
        terminal_pressure = _clamp01(float(gate_info.get("terminal_pressure", 0.0) or 0.0))

        service_reward = max(self._metric_float(metrics, "service_reward_sum"), 0.0)
        if service_reward <= 0.0:
            service_reward = max(self._metric_float(metrics, "service_success_count"), 0.0)
        delay_pressure = min(
            max(
                self._metric_float(metrics, "service_delay_sum"),
                self._metric_float(metrics, "end_to_end_workflow_delay"),
                0.0,
            )
            / 64.0,
            1.5,
        )
        cache_miss_pressure = min(max(self._metric_float(metrics, "cache_miss_penalty_sum"), 0.0) / 4.8, 1.5)
        continuity = _clamp01(self._metric_float(metrics, "workflow_continuity_rate", 1.0))
        stall = float(self._metric_bool(metrics, "stall_occurred"))
        mechanism_success = self._row_mechanism_success(metrics)
        selected_target = bool(final_action == target_action)
        selected_current_service = bool(final_action in {0, 3} or slow_action == 1 or fast_action == 0)
        selected_local_fallback = bool(final_action == 2 or fast_action == 1)
        selected_mechanism = bool(final_action in {1, 4} or event_action == 1 or slow_action == 2)

        credit = 0.0
        if selected_target or selected_current_service:
            credit += 0.62 * score + 0.28 * service_pressure + 0.18 * continuity
            credit -= 0.12 * delay_pressure + 0.10 * cache_miss_pressure
        if selected_local_fallback:
            credit -= (
                0.74 * score
                + 0.34 * terminal_pressure
                + 0.24 * cache_miss_pressure
                + 0.18 * stall
            )
        if selected_mechanism and not mechanism_success:
            credit -= 0.52 * score + 0.24 * terminal_pressure + 0.16 * delay_pressure
        elif selected_mechanism and mechanism_success:
            credit += 0.20 * score
        return float(credit)

    def _net_utility_prd_adjustment(self, row: dict[str, Any]) -> float:
        if not self._net_utility_prd_enabled:
            return 0.0
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        window_class = self._row_window_class(row)
        backhaul_units = max(
            float(metrics.get("backhaul_traffic_cost", 0.0) or 0.0),
            0.0,
        ) / self._net_utility_backhaul_normalizer
        migration_cost = max(
            float(metrics.get("adapter_state_migration_overhead", 0.0) or 0.0),
            0.0,
        )
        expired_prefetch = bool(metrics.get("prefetch_expired_miss", False))
        pending_prefetch = bool(metrics.get("predictive_prefetch_requested", False))
        mechanism_success = self._row_mechanism_success(metrics)
        penalty_scale = 1.0 + float(self._net_utility_cost_dual)
        penalty = penalty_scale * (
            self._net_utility_backhaul_coef * backhaul_units
            + self._net_utility_migration_coef * migration_cost
            + self._net_utility_expired_prefetch_coef * float(expired_prefetch)
        )
        if window_class == "idle_or_sparse" and pending_prefetch and not mechanism_success:
            penalty += penalty_scale * self._net_utility_idle_prefetch_penalty
        if self._row_failed_mechanism_attempt(row, metrics):
            failed_penalty_scale = 1.0
            if window_class == "mechanism_activating":
                failed_penalty_scale = self._net_utility_mechanism_window_failed_penalty_scale
            penalty += penalty_scale * (
                failed_penalty_scale
                * (
                    self._net_utility_failed_mechanism_penalty
                    + self._net_utility_failed_mechanism_backhaul_coef * backhaul_units
                )
            )
        bonus = self._net_utility_success_bonus * float(mechanism_success)
        return float(bonus - penalty)

    def _event_partial_reward_credit(self, row: dict[str, Any]) -> float:
        action_info = dict(row.get("action_info", {}))
        window_class = self._row_window_class(row)
        event_action = int(action_info.get("head_actions", {}).get("event", 0) or 0)
        final_action = int(action_info.get("final_env_action", row.get("action", 0)) or 0)
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        gate_pass = bool(action_info.get("gate_pass", False))
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        mechanism_success = float(metrics.get("mechanism_success_rate", 0.0) or 0.0)
        ready_rate = float(metrics.get("handoff_ready_rate", metrics.get("handoff_ready_ratio", 0.0)) or 0.0)
        failure_rate = float(metrics.get("handoff_failure_rate", 0.0) or 0.0)
        context_strength = 0.45 * prepare_score + 0.30 * urgency + 0.20 * confidence
        if gate_pass:
            context_strength += 0.25
        outcome_strength = 0.55 * mechanism_success + 0.25 * ready_rate - 0.35 * failure_rate
        teacher_credit = self._counterfactual_teacher_action_credit(row, final_action)
        if self._counterfactual_teacher_clip > 0.0:
            teacher_credit = max(
                -self._counterfactual_teacher_clip,
                min(self._counterfactual_teacher_clip, teacher_credit),
            )
        teacher_adjustment = self._counterfactual_teacher_event_coef * teacher_credit
        sparse_realization_adjustment = self._sparse_handoff_realization_credit(
            row,
            env_action=final_action,
        )

        if window_class == "mechanism_activating":
            if event_action == 1:
                credit = context_strength + outcome_strength
                if final_action in {1, 4}:
                    credit += 0.18
            else:
                credit = outcome_strength - 0.35 * context_strength
            return float(
                credit
                + teacher_adjustment
                + sparse_realization_adjustment
                + self._net_utility_prd_adjustment(row)
            )
        if window_class == "idle_or_sparse":
            if event_action == 1:
                credit = 0.15 * context_strength - 0.35 * (1.0 - float(gate_pass))
            else:
                credit = 0.10
            return float(
                credit
                + teacher_adjustment
                + sparse_realization_adjustment
                + self._net_utility_prd_adjustment(row)
            )
        if window_class == "active_non_mechanism":
            credit = -0.20 if event_action == 1 else 0.08
            return float(
                credit
                + teacher_adjustment
                + sparse_realization_adjustment
                + self._net_utility_prd_adjustment(row)
            )
        return float(
            teacher_adjustment
            + sparse_realization_adjustment
            + self._net_utility_prd_adjustment(row)
        )

    def _option_gate_advantage(
        self,
        *,
        row: dict[str, Any],
        base_advantage: torch.Tensor,
        option_probs: torch.Tensor | None = None,
        option_mask: list[bool] | None = None,
    ) -> torch.Tensor:
        if not (
            (self._option_gate_prd_enabled and self._option_gate_prd_coef > 0.0)
            or (
                self._option_gate_counterfactual_prd_enabled
                and self._option_gate_counterfactual_coef > 0.0
            )
            or (self._handoff_risk_prd_enabled and self._handoff_risk_option_coef > 0.0)
            or (self._mechanism_credit_prd_enabled and self._mechanism_credit_option_coef > 0.0)
            or (self._idle_execution_prd_enabled and self._idle_execution_option_coef > 0.0)
            or (self._tail_risk_prd_enabled and self._tail_risk_option_coef > 0.0)
            or (self._opportunity_prd_enabled and self._opportunity_option_coef > 0.0)
            or (
                self._counterfactual_teacher_prd_enabled
                and self._counterfactual_teacher_option_coef > 0.0
            )
        ):
            return base_advantage
        partial_credit = 0.0
        if self._option_gate_prd_enabled and self._option_gate_prd_coef > 0.0:
            selected_credit = self._option_gate_partial_reward_credit(row)
            if self._option_gate_prd_clip > 0.0:
                selected_credit = max(-self._option_gate_prd_clip, min(self._option_gate_prd_clip, selected_credit))
            partial_credit += selected_credit * self._option_gate_prd_coef
        if (
            self._option_gate_counterfactual_prd_enabled
            and self._option_gate_counterfactual_coef > 0.0
        ):
            counterfactual_credit = self._option_gate_counterfactual_partial_credit(
                row,
                option_probs=option_probs,
                option_mask=option_mask,
            )
            if self._option_gate_counterfactual_clip > 0.0:
                counterfactual_credit = max(
                    -self._option_gate_counterfactual_clip,
                    min(self._option_gate_counterfactual_clip, counterfactual_credit),
                )
            partial_credit += counterfactual_credit * self._option_gate_counterfactual_coef
        if self._handoff_risk_prd_enabled and self._handoff_risk_option_coef > 0.0:
            risk_credit = self._handoff_risk_prd_credit(row)
            if self._handoff_risk_clip > 0.0:
                risk_credit = max(-self._handoff_risk_clip, min(self._handoff_risk_clip, risk_credit))
            partial_credit += risk_credit * self._handoff_risk_option_coef
        if self._mechanism_credit_prd_enabled and self._mechanism_credit_option_coef > 0.0:
            mechanism_credit = self._mechanism_credit_prd_credit(row)
            if self._mechanism_credit_clip > 0.0:
                mechanism_credit = max(
                    -self._mechanism_credit_clip,
                    min(self._mechanism_credit_clip, mechanism_credit),
                )
            partial_credit += mechanism_credit * self._mechanism_credit_option_coef
        if self._idle_execution_prd_enabled and self._idle_execution_option_coef > 0.0:
            idle_credit = self._idle_execution_prd_credit(row)
            if self._idle_execution_clip > 0.0:
                idle_credit = max(-self._idle_execution_clip, min(self._idle_execution_clip, idle_credit))
            partial_credit += idle_credit * self._idle_execution_option_coef
        if self._tail_risk_prd_enabled and self._tail_risk_option_coef > 0.0:
            tail_credit = self._tail_risk_prd_credit(row)
            if self._tail_risk_clip > 0.0:
                tail_credit = max(-self._tail_risk_clip, min(self._tail_risk_clip, tail_credit))
            partial_credit += tail_credit * self._tail_risk_option_coef
        if self._opportunity_prd_enabled and self._opportunity_option_coef > 0.0:
            opportunity_credit = self._opportunity_prd_credit(row)
            if self._opportunity_clip > 0.0:
                opportunity_credit = max(
                    -self._opportunity_clip,
                    min(self._opportunity_clip, opportunity_credit),
                )
            partial_credit += opportunity_credit * self._opportunity_option_coef
        if (
            self._counterfactual_teacher_prd_enabled
            and self._counterfactual_teacher_option_coef > 0.0
        ):
            teacher_credit = self._counterfactual_teacher_option_advantage(
                row,
                option_probs=option_probs,
                option_mask=option_mask,
            )
            if self._counterfactual_teacher_clip > 0.0:
                teacher_credit = max(
                    -self._counterfactual_teacher_clip,
                    min(self._counterfactual_teacher_clip, teacher_credit),
                )
            partial_credit += teacher_credit * self._counterfactual_teacher_option_coef
        credit_tensor = torch.tensor(
            partial_credit,
            dtype=torch.float32,
            device=self._device,
        )
        return base_advantage + credit_tensor

    def _option_gate_partial_reward_credit(self, row: dict[str, Any]) -> float:
        action_info = dict(row.get("action_info", {}))
        option_info = dict(action_info.get("option_gate", {}))
        option_label = str(option_info.get("option_label", "accept_mappo"))
        option_env_action = int(
            option_info.get(
                "option_env_action",
                action_info.get("final_env_action", row.get("action", 0)),
            )
            or 0
        )
        option_applied = bool(option_info.get("applied", False))
        return self._option_gate_partial_reward_credit_for_label(
            row,
            option_label=option_label,
            option_env_action=option_env_action,
            option_applied=option_applied,
        )

    def _option_gate_counterfactual_partial_credit(
        self,
        row: dict[str, Any],
        *,
        option_probs: torch.Tensor | None,
        option_mask: list[bool] | None,
    ) -> float:
        action_info = dict(row.get("action_info", {}))
        option_info = dict(action_info.get("option_gate", {}))
        option_action = int(option_info.get("option_action", 0) or 0)
        base_env_action = int(
            option_info.get(
                "base_env_action",
                action_info.get("projected_env_action", action_info.get("final_env_action", row.get("action", 0))),
            )
            or 0
        )
        option_actions_raw = option_info.get("option_actions", {})
        option_actions: dict[int, int] = {}
        if isinstance(option_actions_raw, dict):
            for raw_key, raw_value in option_actions_raw.items():
                try:
                    option_actions[int(raw_key)] = int(raw_value)
                except (TypeError, ValueError):
                    continue
        valid_count = min(self._option_gate_count, len(option_mask or [])) if option_mask else self._option_gate_count
        utilities: list[float] = []
        probabilities: list[float] = []
        probs = option_probs.detach().cpu().tolist() if option_probs is not None else []
        for option_index in range(self._option_gate_count):
            is_valid = True
            if option_mask is not None and option_index < len(option_mask):
                is_valid = bool(option_mask[option_index])
            elif option_mask is not None:
                is_valid = False
            if option_index >= valid_count:
                is_valid = False
            option_env_action = int(option_actions.get(option_index, base_env_action))
            option_label = OPTION_GATE_LABELS.get(option_index, f"option_{option_index}")
            option_applied = bool(option_env_action != base_env_action)
            utilities.append(
                self._option_gate_partial_reward_credit_for_label(
                    row,
                    option_label=option_label,
                    option_env_action=option_env_action,
                    option_applied=option_applied,
                )
                if is_valid
                else 0.0
            )
            probabilities.append(float(probs[option_index]) if option_index < len(probs) and is_valid else 0.0)
        probability_sum = float(sum(probabilities))
        if probability_sum <= 1e-8:
            valid_indices = [
                index
                for index in range(self._option_gate_count)
                if option_mask is None or (index < len(option_mask) and bool(option_mask[index]))
            ]
            if not valid_indices:
                return 0.0
            uniform_prob = 1.0 / float(len(valid_indices))
            probabilities = [uniform_prob if index in valid_indices else 0.0 for index in range(self._option_gate_count)]
        else:
            probabilities = [probability / probability_sum for probability in probabilities]
        expected_credit = float(sum(probability * utility for probability, utility in zip(probabilities, utilities)))
        if option_action < 0 or option_action >= len(utilities):
            selected_credit = self._option_gate_partial_reward_credit(row)
        else:
            selected_credit = float(utilities[option_action])
        return float(selected_credit - expected_credit)

    def _option_gate_partial_reward_credit_for_label(
        self,
        row: dict[str, Any],
        *,
        option_label: str,
        option_env_action: int,
        option_applied: bool,
    ) -> float:
        action_info = dict(row.get("action_info", {}))
        option_info = dict(action_info.get("option_gate", {}))
        window_class = str(
            option_info.get(
                "window_class",
                row.get("decision_info", {}).get("run_metadata", {}).get("window_class", "unknown"),
            )
        )
        prepare_score = max(float(action_info.get("prepare_window_score", 0.0) or 0.0), 0.0)
        urgency = max(float(action_info.get("temporal_urgency", 0.0) or 0.0), 0.0)
        confidence = max(float(action_info.get("prediction_confidence", 0.0) or 0.0), 0.0)
        gate_pass = bool(action_info.get("gate_pass", False))
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        mechanism_success = float(metrics.get("mechanism_success_rate", 0.0) or 0.0)
        ready_rate = float(metrics.get("handoff_ready_rate", metrics.get("handoff_ready_ratio", 0.0)) or 0.0)
        failure_rate = float(metrics.get("handoff_failure_rate", 0.0) or 0.0)
        reward_sign = 1.0 if float(row.get("reward", 0.0) or 0.0) >= 0.0 else -1.0

        credit = 0.0
        service_credit = self._service_continuity_counterfactual_credit(
            row,
            env_action=int(option_env_action),
            window_class=window_class,
            context_strength=0.45 * prepare_score + 0.30 * urgency + 0.20 * confidence + 0.25 * float(gate_pass),
            timing_support=max(prepare_score, urgency),
        )
        if service_credit is not None:
            credit += float(service_credit)
        credit += self._sparse_handoff_realization_credit(
            row,
            env_action=int(option_env_action),
        )
        gate_info = dict(action_info.get("net_advantage_prepare_gate", {}))
        final_guard_info = dict(action_info.get("coverage_recovery_final_guard", {}))
        target_memory_context = bool(
            gate_info.get("current_rsu_id") is None
            and (
                gate_info.get("predicted_target_valid", False)
                or gate_info.get("target_differs", False)
            )
        )
        if target_memory_context and (
            self._coverage_recovery_target_memory_option_credit > 0.0
            or self._coverage_recovery_target_memory_option_penalty > 0.0
        ):
            target_memory_scale = _clamp01(
                max(
                    float(gate_info.get("coverage_recovery_scale", 0.0) or 0.0),
                    float(gate_info.get("net_advantage_score", 0.0) or 0.0),
                    float(final_guard_info.get("coverage_recovery_scale", 0.0) or 0.0),
                )
            )
            confidence = _clamp01(float(gate_info.get("prediction_confidence", 0.0) or 0.0))
            memory_strength = 0.55 + 0.30 * target_memory_scale + 0.15 * confidence
            if option_label == "mechanism_prepare" or int(option_env_action) in {1, 4}:
                credit += self._coverage_recovery_target_memory_option_credit * memory_strength
            if option_label == "no_rsu_local" or int(option_env_action) == 2:
                guard_multiplier = 1.25 if bool(final_guard_info.get("guarded", False)) else 1.0
                credit -= (
                    self._coverage_recovery_target_memory_option_penalty
                    * memory_strength
                    * guard_multiplier
                )
        if window_class == "mechanism_activating":
            context_strength = 0.45 * prepare_score + 0.30 * urgency + 0.20 * confidence
            if gate_pass:
                context_strength += 0.25
            outcome_strength = 0.55 * mechanism_success + 0.25 * ready_rate - 0.35 * failure_rate
            if option_label == "mechanism_prepare":
                credit += context_strength + outcome_strength
                if int(option_env_action) in {1, 4}:
                    credit += 0.20
            elif option_label == "accept_mappo":
                credit += 0.35 * context_strength + outcome_strength
                if int(option_env_action) in {1, 4}:
                    credit += 0.10
            elif option_label in {"popularity_safe", "no_rsu_local"}:
                credit -= 0.35 * context_strength
        elif window_class == "idle_or_sparse":
            if self._counterfactual_teacher_prd_enabled and self._service_continuity_teacher_enabled:
                context_strength = 0.40 * prepare_score + 0.30 * urgency + 0.20 * confidence
                if gate_pass or bool(option_info.get("idle_recovery_context", False)):
                    context_strength += 0.25
                if int(option_env_action) == 2:
                    credit -= 0.62 + 0.28 * (1.0 - min(context_strength, 1.0))
                elif int(option_env_action) in {0, 3}:
                    credit += 0.24 + 0.10 * reward_sign + 0.12 * min(context_strength, 1.0)
                elif int(option_env_action) in {1, 4}:
                    credit += (
                        0.38
                        + 0.46 * min(context_strength, 1.0)
                        + 0.18 * mechanism_success
                        + 0.14 * ready_rate
                    )
            else:
                if option_label == "popularity_safe":
                    credit += 0.45 + 0.10 * reward_sign
                elif option_label == "no_rsu_local":
                    low_no_rsu_context = bool(option_info.get("no_rsu_available", False))
                    credit += (0.28 if low_no_rsu_context else 0.04) + 0.08 * reward_sign
                elif option_label == "mechanism_prepare":
                    credit -= 0.45
                elif option_label == "accept_mappo":
                    credit += 0.10 * reward_sign
        elif window_class == "active_non_mechanism":
            if option_label == "accept_mappo":
                credit += 0.15 * reward_sign
            elif bool(option_applied):
                credit -= 0.25
        return float(credit + self._net_utility_prd_adjustment(row))

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
        retrospective_label = {}
        decision_info = row.get("decision_info", {})
        if isinstance(decision_info, dict):
            retrospective_label = dict(decision_info.get("retrospective_handoff_label", {}) or {})
        if not retrospective_label and isinstance(row.get("retrospective_handoff_label", {}), dict):
            retrospective_label = dict(row.get("retrospective_handoff_label", {}) or {})
        retrospective_opportunity = bool(
            self._retrospective_handoff_aux_enabled
            and retrospective_label
            and float(retrospective_label.get("gt_handoff_opportunity", 0.0) or 0.0) > 0.5
        )
        retrospective_eta = max(
            float(retrospective_label.get("gt_first_handoff_steps", 0.0) or 0.0),
            0.0,
        )
        retrospective_target_rsu_id = retrospective_label.get("gt_first_next_rsu")
        retrospective_current_rsu_id = retrospective_label.get("current_rsu_id")
        retrospective_target_distinct = bool(
            retrospective_target_rsu_id is not None
            and (
                retrospective_current_rsu_id is None
                or str(retrospective_target_rsu_id) != str(retrospective_current_rsu_id)
            )
        )
        retrospective_window_score = 0.0
        if retrospective_eta > 0.0:
            normalized_eta_gap = (
                retrospective_eta - float(self._temporal_prepare_lead_steps)
            ) / max(float(self._temporal_prepare_sigma), 0.25)
            retrospective_window_score = float(math.exp(-0.5 * normalized_eta_gap * normalized_eta_gap))
        retrospective_guidance = bool(
            retrospective_opportunity
            and retrospective_target_distinct
            and 0.0 < retrospective_eta <= self._retrospective_handoff_aux_max_eta
            and retrospective_window_score >= self._retrospective_handoff_aux_min_score
        )
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
        prepare_aux_legal = bool(mechanism_action_legal and (valid_handoff_target or retrospective_guidance))
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
        predicted_event_guidance = bool(
            self._mechanism_aux_coef > 0.0
            and prepare_action_legal
            and timing_active
            and prediction_usable
            and (valid_handoff_target or not cache_ready or target_mismatch or not gate_pass)
        )
        retrospective_event_guidance = bool(
            self._mechanism_aux_coef > 0.0
            and prepare_aux_legal
            and retrospective_guidance
        )
        needs_event_guidance = bool(predicted_event_guidance or retrospective_event_guidance)
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
        if retrospective_event_guidance:
            retrospective_strength = _clamp01(
                self._retrospective_handoff_aux_prepare_weight
                * (0.55 + 0.45 * retrospective_window_score)
            )
            guidance_strength = max(guidance_strength, retrospective_strength)
        guidance_strength = max(guidance_strength, 0.25 if needs_guidance else 0.0)
        transition_weight = self._effective_mechanism_window_weight() if needs_guidance else 1.0
        if retrospective_event_guidance:
            transition_weight = max(transition_weight, self._retrospective_handoff_aux_transition_weight)
        return {
            "apply": needs_guidance,
            "event_guidance": needs_event_guidance,
            "prefetch_guidance": needs_prefetch_guidance,
            "predicted_event_guidance": predicted_event_guidance,
            "retrospective_event_guidance": retrospective_event_guidance,
            "retrospective_handoff_aux_enabled": self._retrospective_handoff_aux_enabled,
            "retrospective_handoff_opportunity": retrospective_opportunity,
            "retrospective_target_distinct": retrospective_target_distinct,
            "retrospective_handoff_eta": round(float(retrospective_eta), 6),
            "retrospective_handoff_score": round(float(retrospective_window_score), 6),
            "retrospective_target_rsu_id": retrospective_target_rsu_id,
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
            "prepare_action_legal": prepare_action_legal,
            "prepare_aux_legal": prepare_aux_legal,
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
            "mechanism_predicted_event_guidance_count": int(
                sum(1 for item in annotations if bool(item.get("predicted_event_guidance", False)))
            ),
            "mechanism_retrospective_event_guidance_count": int(
                sum(1 for item in annotations if bool(item.get("retrospective_event_guidance", False)))
            ),
            "mechanism_retrospective_handoff_aux_enabled": self._retrospective_handoff_aux_enabled,
            "mechanism_retrospective_handoff_opportunity_count": int(
                sum(1 for item in annotations if bool(item.get("retrospective_handoff_opportunity", False)))
            ),
            "mechanism_retrospective_target_distinct_count": int(
                sum(1 for item in annotations if bool(item.get("retrospective_target_distinct", False)))
            ),
            "mechanism_retrospective_handoff_score_mean": round(
                float(
                    fmean(
                        float(item.get("retrospective_handoff_score", 0.0) or 0.0)
                        for item in annotations
                        if bool(item.get("retrospective_event_guidance", False))
                    )
                )
                if any(bool(item.get("retrospective_event_guidance", False)) for item in annotations)
                else 0.0,
                6,
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
            for semantic_state, annotation in zip(semantic_states, annotations)
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
        for policy_output, annotation in zip(batch_outputs, batch_annotations):
            if not bool(annotation.get("apply", False)):
                continue
            event_logits = policy_output["event_logits"]
            event_target = torch.tensor([int(annotation.get("event_target", 1))], dtype=torch.long, device=self._device)
            event_loss = nn.functional.cross_entropy(event_logits.unsqueeze(0), event_target)
            if self._mechanism_focal_aux_enabled and self._mechanism_focal_gamma > 0.0:
                event_probs = torch.softmax(event_logits, dim=-1)
                event_target_index = int(event_target.item())
                if 0 <= event_target_index < int(event_probs.shape[-1]):
                    target_prob = torch.clamp(event_probs[event_target_index], min=1e-6, max=1.0)
                    event_loss = torch.pow(1.0 - target_prob, self._mechanism_focal_gamma) * event_loss
            event_distribution = Categorical(logits=event_logits)
            weighted_loss = float(annotation.get("event_weight", 1.0)) * event_loss
            slow_weight = float(annotation.get("slow_weight", 0.0) or 0.0)
            if slow_weight > 1e-8:
                slow_target = torch.tensor([int(annotation.get("slow_target", 0))], dtype=torch.long, device=self._device)
                slow_loss = nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), slow_target)
                if self._mechanism_focal_aux_enabled and self._mechanism_focal_gamma > 0.0:
                    slow_probs = torch.softmax(policy_output["slow_logits"], dim=-1)
                    slow_target_index = int(slow_target.item())
                    if 0 <= slow_target_index < int(slow_probs.shape[-1]):
                        slow_target_prob = torch.clamp(slow_probs[slow_target_index], min=1e-6, max=1.0)
                        slow_loss = torch.pow(1.0 - slow_target_prob, self._mechanism_focal_gamma) * slow_loss
                weighted_loss = weighted_loss + slow_weight * slow_loss
            loss_terms.append(weighted_loss)
            entropy_terms.append(event_distribution.entropy())
        if not loss_terms:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return zero, zero
        return torch.stack(loss_terms).mean(), torch.stack(entropy_terms).mean()

    def _apply_opportunity_constrained_policy(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._use_hierarchy or not self._opportunity_constrained_policy_enabled:
            return policy_output

        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = _clamp01(float(timing_features.get("prepare_window_score", 0.0) or 0.0))
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        predicted_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        diagnostics = self._build_prediction_target_diagnostics(
            semantic_state=semantic_state,
            temporal_urgency=temporal_urgency,
            predicted_handoff_target_valid=predicted_target_valid,
        )
        confidence = _clamp01(float(diagnostics.get("prediction_confidence", 0.0) or 0.0))
        uncertainty = _clamp01(float(diagnostics.get("prediction_uncertainty", 1.0) or 1.0))
        reliability = _clamp01(confidence * (1.0 - uncertainty))
        raw_candidate = bool(diagnostics.get("raw_handoff_candidate", False))
        gate_pass = bool(diagnostics.get("gate_pass", False))
        first_eta = int(diagnostics.get("predicted_first_non_current_eta", 0) or 0)
        reliable_candidate = bool(
            raw_candidate
            and confidence >= self._opportunity_constrained_confidence_floor
            and uncertainty <= self._opportunity_constrained_uncertainty_ceiling
        )
        trusted_candidate = bool(reliable_candidate and (gate_pass or reliability >= self._opportunity_constrained_reliability_floor))
        opportunity_context = _clamp01(
            0.34 * prepare_score
            + 0.22 * temporal_urgency
            + 0.26 * reliability
            + 0.08 * float(gate_pass)
            + 0.06 * float(raw_candidate)
            + 0.04 * float(predicted_target_valid)
        )
        strong_opportunity = bool(
            trusted_candidate
            and predicted_target_valid
            and opportunity_context >= self._opportunity_constrained_min_context
        )
        weak_opportunity = bool(
            trusted_candidate
            and opportunity_context >= self._opportunity_constrained_low_context
        )

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        required_adapter_key = str(required_adapter) if required_adapter else None
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predictions = semantic_state.get("predictions", {}) if isinstance(semantic_state.get("predictions", {}), dict) else {}
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        if predicted_next_rsu_id is None:
            next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
            predicted_next_rsu_id = next_sequence[0] if next_sequence else None
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        predicted_next_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_next_rsu_id)
        handoff_target_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_handoff_target_rsu_id)

        def has_adapter(rsu: dict[str, Any]) -> bool:
            return bool(
                required_adapter_key
                and required_adapter_key in {str(item) for item in rsu.get("cached_adapter_ids", [])}
            )

        current_cache_ready = has_adapter(current_rsu)
        predicted_next_cache_ready = has_adapter(predicted_next_rsu)
        handoff_target_cache_ready = has_adapter(handoff_target_rsu)
        predicted_first_non_current_rsu_id = diagnostics.get("predicted_first_non_current_rsu")
        predicted_first_non_current_rsu = _rsu_by_id_from_semantic_state(
            semantic_state,
            predicted_first_non_current_rsu_id,
        )
        first_non_current_cache_ready = has_adapter(predicted_first_non_current_rsu)
        next_differs = bool(predicted_next_rsu_id is not None and str(predicted_next_rsu_id) != str(current_rsu_id))
        target_differs = bool(
            predicted_handoff_target_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
        )
        target_mismatch = bool(
            predicted_handoff_target_rsu_id is not None
            and predicted_first_non_current_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(predicted_first_non_current_rsu_id)
        )

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)

        low_context_span = max(self._opportunity_constrained_min_context - self._opportunity_constrained_low_context, 1e-6)
        weak_scale = _clamp01((opportunity_context - self._opportunity_constrained_low_context) / low_context_span)
        suppress_scale = 1.0 - weak_scale if not strong_opportunity else 0.0
        if not weak_opportunity:
            suppress_scale = 1.0

        prepare_penalty = self._opportunity_constrained_prepare_penalty * suppress_scale
        prefetch_penalty = self._opportunity_constrained_prefetch_penalty * suppress_scale
        if prepare_penalty > 1e-8:
            event_logits[1] = event_logits[1] - prepare_penalty
            event_logits[0] = event_logits[0] + 0.20 * prepare_penalty
            env_action_logits_bias[4] = env_action_logits_bias[4] - prepare_penalty
        if prefetch_penalty > 1e-8:
            slow_logits[2] = slow_logits[2] - prefetch_penalty
            slow_logits[0] = slow_logits[0] + 0.12 * prefetch_penalty
            env_action_logits_bias[1] = env_action_logits_bias[1] - prefetch_penalty

        prepare_bias = 0.0
        prefetch_bias = 0.0
        no_rsu_prepare_bias_applied = 0.0
        if strong_opportunity:
            imminent = bool(first_eta <= 3 or prepare_score >= self._opportunity_constrained_min_context)
            if imminent and (handoff_target_cache_ready or current_cache_ready or prepare_score >= 0.72):
                prepare_bias = self._opportunity_constrained_prepare_bias * opportunity_context
                event_logits[1] = event_logits[1] + prepare_bias
                event_logits[0] = event_logits[0] - 0.16 * prepare_bias
                env_action_logits_bias[4] = env_action_logits_bias[4] + prepare_bias
            if next_differs and not predicted_next_cache_ready and (not imminent or not handoff_target_cache_ready):
                prefetch_bias = self._opportunity_constrained_prefetch_bias * opportunity_context
                slow_logits[2] = slow_logits[2] + prefetch_bias
                slow_logits[0] = slow_logits[0] - 0.10 * prefetch_bias
                env_action_logits_bias[1] = env_action_logits_bias[1] + prefetch_bias

        current_bias = 0.0
        local_bias = 0.0
        if suppress_scale > 1e-8:
            if current_rsu_id is not None:
                current_action = 3 if current_cache_ready else 0
                current_bias = self._opportunity_constrained_current_bias * suppress_scale
                env_action_logits_bias[current_action] = env_action_logits_bias[current_action] + current_bias
                if current_action == 0:
                    slow_logits[1] = slow_logits[1] + 0.40 * current_bias
                else:
                    fast_logits[0] = fast_logits[0] + 0.28 * current_bias
            else:
                no_rsu_service_bias = self._opportunity_constrained_no_rsu_service_bias
                no_rsu_local_penalty = self._opportunity_constrained_no_rsu_local_penalty
                if no_rsu_service_bias > 1e-8 or no_rsu_local_penalty > 1e-8:
                    no_rsu_scale = max(suppress_scale, 1.0)
                    current_bias = no_rsu_service_bias * no_rsu_scale
                    local_bias = -no_rsu_local_penalty * no_rsu_scale
                    env_action_logits_bias[3] = env_action_logits_bias[3] + current_bias
                    env_action_logits_bias[2] = env_action_logits_bias[2] + local_bias
                    fast_logits[0] = fast_logits[0] + 0.28 * current_bias
                    fast_logits[1] = fast_logits[1] + 0.35 * local_bias
                    no_rsu_prepare_context = max(
                        float(opportunity_context),
                        float(prepare_score),
                        float(temporal_urgency),
                        float(confidence) if (predicted_target_valid or target_differs or raw_candidate) else 0.0,
                    )
                    if (
                        self._opportunity_constrained_no_rsu_prepare_bias > 1e-8
                        and (predicted_target_valid or target_differs or raw_candidate)
                        and no_rsu_prepare_context + 1e-8
                        >= self._opportunity_constrained_no_rsu_prepare_min_context
                    ):
                        no_rsu_prepare_bias_applied = (
                            self._opportunity_constrained_no_rsu_prepare_bias
                            * (0.55 + 0.45 * _clamp01(no_rsu_prepare_context))
                        )
                        event_logits[1] = event_logits[1] + no_rsu_prepare_bias_applied
                        event_logits[0] = event_logits[0] - 0.10 * no_rsu_prepare_bias_applied
                        env_action_logits_bias[4] = (
                            env_action_logits_bias[4] + no_rsu_prepare_bias_applied
                        )
                        env_action_logits_bias[2] = (
                            env_action_logits_bias[2] - 0.18 * no_rsu_prepare_bias_applied
                        )
                else:
                    local_bias = self._opportunity_constrained_local_bias * suppress_scale
                    env_action_logits_bias[2] = env_action_logits_bias[2] + local_bias
                    fast_logits[1] = fast_logits[1] + 0.35 * local_bias
        if current_rsu_id is None and no_rsu_prepare_bias_applied <= 1e-8:
            no_rsu_prepare_context = max(
                float(opportunity_context),
                float(prepare_score),
                float(temporal_urgency),
                float(confidence) if (predicted_target_valid or target_differs or raw_candidate) else 0.0,
            )
            if (
                self._opportunity_constrained_no_rsu_prepare_bias > 1e-8
                and (predicted_target_valid or target_differs or raw_candidate)
                and no_rsu_prepare_context + 1e-8
                >= self._opportunity_constrained_no_rsu_prepare_min_context
            ):
                no_rsu_prepare_bias_applied = (
                    self._opportunity_constrained_no_rsu_prepare_bias
                    * (0.55 + 0.45 * _clamp01(no_rsu_prepare_context))
                )
                event_logits[1] = event_logits[1] + no_rsu_prepare_bias_applied
                event_logits[0] = event_logits[0] - 0.10 * no_rsu_prepare_bias_applied
                env_action_logits_bias[4] = env_action_logits_bias[4] + no_rsu_prepare_bias_applied
                env_action_logits_bias[2] = env_action_logits_bias[2] - 0.18 * no_rsu_prepare_bias_applied

        cache_feasibility_cache_fill_bias = 0.0
        cache_feasibility_steady_penalty = 0.0
        cache_feasibility_prepare_penalty = 0.0
        cache_feasibility_prefetch_penalty = 0.0
        cache_feasibility_current_miss_prepare_penalty = 0.0
        cache_feasibility_current_miss_prefetch_penalty = 0.0
        handoff_alignment_prepare_penalty = 0.0
        handoff_alignment_prefetch_penalty = 0.0
        handoff_alignment_current_fill_bias = 0.0
        handoff_alignment_immediate_prefetch_bias = 0.0
        handoff_alignment_steady_bias = 0.0
        handoff_alignment_barrier_active = False
        handoff_alignment_late_eta = False
        handoff_alignment_late_eta_threshold = max(int(math.ceil(self._temporal_prepare_lead_steps)) + 1, 3)
        handoff_alignment_first_hop_not_ready = False
        handoff_alignment_current_service_not_ready = False
        cache_feasibility_context = _clamp01(
            max(
                float(opportunity_context),
                0.60 * float(prepare_score),
                0.55 * float(temporal_urgency),
                0.50 * float(reliability),
            )
        )
        window_class = str(
            semantic_state.get("window_class")
            or (semantic_state.get("run_info", {}) or {}).get("window_class")
            or ""
        )
        cache_feasibility_prior_active = bool(
            self._cache_feasibility_prior_enabled
            and required_adapter_key
            and current_rsu_id is not None
            and not current_cache_ready
            and cache_feasibility_context + 1e-8 >= self._cache_feasibility_min_context
        )
        if cache_feasibility_prior_active:
            mechanism_window = window_class == "mechanism_activating"
            weak_signal = bool(not weak_opportunity or not predicted_target_valid)
            fill_scale = (
                0.72
                + 0.28 * float(suppress_scale)
                + 0.18 * float(weak_signal)
                + 0.14 * float(mechanism_window)
            )
            steady_scale = (
                0.76
                + 0.24 * float(suppress_scale)
                + 0.20 * float(weak_signal)
                + 0.16 * float(mechanism_window)
            )
            cache_feasibility_cache_fill_bias = (
                self._cache_feasibility_cache_fill_bias * fill_scale
            )
            cache_feasibility_steady_penalty = (
                self._cache_feasibility_steady_penalty * steady_scale
            )
            env_action_logits_bias[0] = (
                env_action_logits_bias[0] + cache_feasibility_cache_fill_bias
            )
            env_action_logits_bias[3] = (
                env_action_logits_bias[3] - cache_feasibility_steady_penalty
            )
            slow_logits[1] = slow_logits[1] + 0.42 * cache_feasibility_cache_fill_bias
            slow_logits[0] = slow_logits[0] - 0.10 * cache_feasibility_cache_fill_bias
            fast_logits[0] = fast_logits[0] - 0.30 * cache_feasibility_steady_penalty
            current_miss_signal = bool(predicted_target_valid or target_differs or raw_candidate or next_differs)
            if current_miss_signal:
                current_miss_scale = (
                    0.68
                    + 0.20 * float(strong_opportunity)
                    + 0.12 * float(mechanism_window)
                )
                cache_feasibility_current_miss_prepare_penalty = (
                    self._cache_feasibility_current_miss_prepare_penalty
                    * current_miss_scale
                )
                cache_feasibility_current_miss_prefetch_penalty = (
                    self._cache_feasibility_current_miss_prefetch_penalty
                    * current_miss_scale
                )
                if cache_feasibility_current_miss_prepare_penalty > 1e-8:
                    env_action_logits_bias[4] = (
                        env_action_logits_bias[4]
                        - cache_feasibility_current_miss_prepare_penalty
                    )
                    event_logits[1] = (
                        event_logits[1]
                        - 0.62 * cache_feasibility_current_miss_prepare_penalty
                    )
                    event_logits[0] = (
                        event_logits[0]
                        + 0.10 * cache_feasibility_current_miss_prepare_penalty
                    )
                if cache_feasibility_current_miss_prefetch_penalty > 1e-8:
                    env_action_logits_bias[1] = (
                        env_action_logits_bias[1]
                        - cache_feasibility_current_miss_prefetch_penalty
                    )
                    slow_logits[2] = (
                        slow_logits[2]
                        - 0.58 * cache_feasibility_current_miss_prefetch_penalty
                    )
                    slow_logits[0] = (
                        slow_logits[0]
                        + 0.08 * cache_feasibility_current_miss_prefetch_penalty
                    )
            if weak_signal:
                cache_feasibility_prepare_penalty = (
                    self._cache_feasibility_prepare_penalty
                    * (0.55 + 0.45 * cache_feasibility_context)
                )
                cache_feasibility_prefetch_penalty = (
                    self._cache_feasibility_prefetch_penalty
                    * (0.55 + 0.45 * cache_feasibility_context)
                )
                if cache_feasibility_prepare_penalty > 1e-8:
                    env_action_logits_bias[4] = (
                        env_action_logits_bias[4] - cache_feasibility_prepare_penalty
                    )
                    event_logits[1] = event_logits[1] - 0.55 * cache_feasibility_prepare_penalty
                    event_logits[0] = event_logits[0] + 0.12 * cache_feasibility_prepare_penalty
                if cache_feasibility_prefetch_penalty > 1e-8:
                    env_action_logits_bias[1] = (
                        env_action_logits_bias[1] - cache_feasibility_prefetch_penalty
                    )
                    slow_logits[2] = slow_logits[2] - 0.50 * cache_feasibility_prefetch_penalty
                    slow_logits[0] = slow_logits[0] + 0.10 * cache_feasibility_prefetch_penalty

        if self._handoff_alignment_barrier_enabled:
            mechanism_window = window_class == "mechanism_activating"
            alignment_context = _clamp01(
                max(
                    float(opportunity_context),
                    float(prepare_score),
                    float(temporal_urgency),
                    float(reliability),
                    0.24 if mechanism_window else 0.0,
                )
            )
            late_eta_threshold = handoff_alignment_late_eta_threshold
            late_eta = bool(first_eta > late_eta_threshold)
            current_service_not_ready = bool(
                required_adapter_key
                and current_rsu_id is not None
                and not current_cache_ready
            )
            first_hop_not_ready = bool(
                predicted_first_non_current_rsu_id is not None
                and not first_non_current_cache_ready
                and first_eta <= max(late_eta_threshold, 1)
            )
            handoff_alignment_late_eta = late_eta
            handoff_alignment_current_service_not_ready = current_service_not_ready
            handoff_alignment_first_hop_not_ready = first_hop_not_ready
            barrier_reason_active = bool(
                current_service_not_ready
                or target_mismatch
                or late_eta
                or (first_hop_not_ready and not handoff_target_cache_ready)
            )
            handoff_alignment_barrier_active = bool(
                barrier_reason_active
                and alignment_context + 1e-8 >= self._handoff_alignment_barrier_min_context
            )
            if handoff_alignment_barrier_active:
                base_scale = 0.72 + 0.18 * float(mechanism_window) + 0.10 * alignment_context
                handoff_alignment_prepare_penalty = (
                    self._handoff_alignment_barrier_prepare_penalty * base_scale
                    + self._handoff_alignment_barrier_target_mismatch_penalty * float(target_mismatch)
                    + self._handoff_alignment_barrier_late_eta_penalty * float(late_eta)
                )
                if handoff_alignment_prepare_penalty > 1e-8:
                    env_action_logits_bias[4] = env_action_logits_bias[4] - handoff_alignment_prepare_penalty
                    event_logits[1] = event_logits[1] - 0.74 * handoff_alignment_prepare_penalty
                    event_logits[0] = event_logits[0] + 0.18 * handoff_alignment_prepare_penalty

                suppress_prefetch = bool(current_service_not_ready or late_eta or first_hop_not_ready)
                if suppress_prefetch:
                    handoff_alignment_prefetch_penalty = (
                        self._handoff_alignment_barrier_prefetch_penalty
                        * (0.70 + 0.18 * float(current_service_not_ready) + 0.12 * alignment_context)
                    )
                    env_action_logits_bias[1] = env_action_logits_bias[1] - handoff_alignment_prefetch_penalty
                    slow_logits[2] = slow_logits[2] - 0.62 * handoff_alignment_prefetch_penalty
                    slow_logits[0] = slow_logits[0] + 0.10 * handoff_alignment_prefetch_penalty
                elif target_mismatch and next_differs and not predicted_next_cache_ready:
                    handoff_alignment_immediate_prefetch_bias = (
                        0.42 * self._handoff_alignment_barrier_prefetch_penalty
                        * (0.55 + 0.45 * alignment_context)
                    )
                    env_action_logits_bias[1] = env_action_logits_bias[1] + handoff_alignment_immediate_prefetch_bias
                    slow_logits[2] = slow_logits[2] + 0.36 * handoff_alignment_immediate_prefetch_bias

                if current_service_not_ready:
                    handoff_alignment_current_fill_bias = (
                        self._handoff_alignment_barrier_current_fill_bias
                        * (0.82 + 0.18 * float(mechanism_window))
                    )
                    env_action_logits_bias[0] = env_action_logits_bias[0] + handoff_alignment_current_fill_bias
                    slow_logits[1] = slow_logits[1] + 0.48 * handoff_alignment_current_fill_bias
                    slow_logits[0] = slow_logits[0] - 0.10 * handoff_alignment_current_fill_bias
                elif late_eta or first_hop_not_ready:
                    handoff_alignment_steady_bias = (
                        0.40 * self._handoff_alignment_barrier_current_fill_bias
                        * (0.60 + 0.40 * alignment_context)
                    )
                    env_action_logits_bias[3] = env_action_logits_bias[3] + handoff_alignment_steady_bias
                    fast_logits[0] = fast_logits[0] + 0.28 * handoff_alignment_steady_bias

        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        adjusted["opportunity_constrained_policy_info"] = {
            "enabled": True,
            "opportunity_context": round(float(opportunity_context), 6),
            "weak_opportunity": bool(weak_opportunity),
            "strong_opportunity": bool(strong_opportunity),
            "trusted_candidate": bool(trusted_candidate),
            "reliability": round(float(reliability), 6),
            "suppress_scale": round(float(suppress_scale), 6),
            "prepare_penalty": round(float(prepare_penalty), 6),
            "prefetch_penalty": round(float(prefetch_penalty), 6),
            "prepare_bias": round(float(prepare_bias), 6),
            "prefetch_bias": round(float(prefetch_bias), 6),
            "current_bias": round(float(current_bias), 6),
            "local_bias": round(float(local_bias), 6),
            "no_rsu_prepare_bias_applied": round(float(no_rsu_prepare_bias_applied), 6),
            "no_rsu_service_bias": round(
                float(self._opportunity_constrained_no_rsu_service_bias),
                6,
            ),
            "no_rsu_local_penalty": round(
                float(self._opportunity_constrained_no_rsu_local_penalty),
                6,
            ),
            "no_rsu_prepare_bias": round(
                float(self._opportunity_constrained_no_rsu_prepare_bias),
                6,
            ),
            "cache_feasibility_prior_enabled": bool(self._cache_feasibility_prior_enabled),
            "cache_feasibility_prior_active": bool(cache_feasibility_prior_active),
            "cache_feasibility_context": round(float(cache_feasibility_context), 6),
            "cache_feasibility_min_context": round(float(self._cache_feasibility_min_context), 6),
            "cache_feasibility_cache_fill_bias": round(
                float(cache_feasibility_cache_fill_bias),
                6,
            ),
            "cache_feasibility_steady_penalty": round(
                float(cache_feasibility_steady_penalty),
                6,
            ),
            "cache_feasibility_prepare_penalty": round(
                float(cache_feasibility_prepare_penalty),
                6,
            ),
            "cache_feasibility_prefetch_penalty": round(
                float(cache_feasibility_prefetch_penalty),
                6,
            ),
            "cache_feasibility_current_miss_prepare_penalty": round(
                float(cache_feasibility_current_miss_prepare_penalty),
                6,
            ),
            "cache_feasibility_current_miss_prefetch_penalty": round(
                float(cache_feasibility_current_miss_prefetch_penalty),
                6,
            ),
            "handoff_alignment_barrier_enabled": bool(self._handoff_alignment_barrier_enabled),
            "handoff_alignment_barrier_active": bool(handoff_alignment_barrier_active),
            "handoff_alignment_prepare_penalty": round(float(handoff_alignment_prepare_penalty), 6),
            "handoff_alignment_prefetch_penalty": round(float(handoff_alignment_prefetch_penalty), 6),
            "handoff_alignment_current_fill_bias": round(
                float(handoff_alignment_current_fill_bias),
                6,
            ),
            "handoff_alignment_immediate_prefetch_bias": round(
                float(handoff_alignment_immediate_prefetch_bias),
                6,
            ),
            "handoff_alignment_steady_bias": round(float(handoff_alignment_steady_bias), 6),
            "handoff_alignment_late_eta": bool(handoff_alignment_late_eta),
            "handoff_alignment_late_eta_threshold": int(handoff_alignment_late_eta_threshold),
            "handoff_alignment_current_service_not_ready": bool(
                handoff_alignment_current_service_not_ready
            ),
            "handoff_alignment_first_hop_not_ready": bool(
                handoff_alignment_first_hop_not_ready
            ),
            "handoff_alignment_first_hop_cache_ready": bool(first_non_current_cache_ready),
            "handoff_alignment_target_mismatch": bool(target_mismatch),
            "predicted_first_non_current_rsu_id": (
                str(predicted_first_non_current_rsu_id)
                if predicted_first_non_current_rsu_id is not None
                else None
            ),
            "raw_handoff_candidate": bool(raw_candidate),
            "predicted_handoff_target_valid": bool(predicted_target_valid),
            "gate_pass": bool(gate_pass),
            "prediction_confidence": round(float(confidence), 6),
            "prediction_uncertainty": round(float(uncertainty), 6),
            "reliability_floor": round(float(self._opportunity_constrained_reliability_floor), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "predicted_first_non_current_eta": int(first_eta),
            "current_cache_ready": bool(current_cache_ready),
            "predicted_next_cache_ready": bool(predicted_next_cache_ready),
            "handoff_target_cache_ready": bool(handoff_target_cache_ready),
            "next_differs": bool(next_differs),
            "target_differs": bool(target_differs),
            "env_action_logit_bias": [
                round(float(item), 6)
                for item in env_action_logits_bias.detach().cpu().tolist()
            ],
        }
        return adjusted

    def _apply_backhaul_aware_policy(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._use_hierarchy or not self._backhaul_aware_policy_enabled:
            return policy_output

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        required_adapter_key = str(required_adapter) if required_adapter else None
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        current_cache_ready = bool(
            required_adapter_key
            and required_adapter_key in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )

        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        predicted_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        diagnostics = self._build_prediction_target_diagnostics(
            semantic_state=semantic_state,
            temporal_urgency=temporal_urgency,
            predicted_handoff_target_valid=predicted_target_valid,
        )
        confidence = _clamp01(float(diagnostics.get("prediction_confidence", 0.0) or 0.0))
        uncertainty = _clamp01(float(diagnostics.get("prediction_uncertainty", 1.0) or 1.0))
        reliability = _clamp01(confidence * (1.0 - uncertainty))
        raw_candidate = bool(diagnostics.get("raw_handoff_candidate", False))
        gate_pass = bool(diagnostics.get("gate_pass", False))
        trusted_candidate = bool(
            raw_candidate
            and confidence >= self._opportunity_constrained_confidence_floor
            and uncertainty <= self._opportunity_constrained_uncertainty_ceiling
            and (gate_pass or reliability >= self._opportunity_constrained_reliability_floor)
        )
        no_trusted_signal = bool(not trusted_candidate and not predicted_target_valid and not raw_candidate)

        workflow = semantic_state.get("workflow") if isinstance(semantic_state.get("workflow"), dict) else {}
        nodes = workflow.get("nodes", []) if isinstance(workflow, dict) else []
        completed_nodes = set(str(item) for item in workflow.get("completed_node_ids", []) or [])
        remaining_nodes = 0
        if isinstance(nodes, list) and nodes:
            for node in nodes:
                node_id = str((node or {}).get("node_id", ""))
                if node_id and node_id not in completed_nodes:
                    remaining_nodes += 1
        else:
            remaining_nodes = 1 if current_node else 0
        successors = current_node.get("successors", []) if isinstance(current_node, dict) else []
        try:
            input_size = max(float(current_node.get("input_size", 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            input_size = 0.0
        service_pressure = 0.0
        if current_node:
            service_pressure = _clamp01(
                0.22
                + 0.22 * float(bool(required_adapter_key and not current_cache_ready))
                + 0.16 * float(current_rsu_id is not None)
                + min(float(remaining_nodes) / 8.0, 0.22)
                + 0.08 * float(bool(successors))
                + min(input_size / 256.0, 0.10)
            )

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)

        service_fill_bias = 0.0
        if (
            required_adapter_key
            and current_rsu_id is not None
            and not current_cache_ready
            and service_pressure >= self._backhaul_aware_service_pressure_floor
        ):
            candidate_relief = 0.55 if trusted_candidate else 1.0
            service_fill_bias = self._backhaul_aware_service_fill_bias * service_pressure * candidate_relief
            if service_fill_bias > 1e-8:
                slow_logits[1] = slow_logits[1] + service_fill_bias
                slow_logits[0] = slow_logits[0] - 0.08 * service_fill_bias
                env_action_logits_bias[0] = env_action_logits_bias[0] + service_fill_bias
                env_action_logits_bias[3] = env_action_logits_bias[3] - 0.18 * service_fill_bias

        redundant_fill_penalty = 0.0
        steady_bias = 0.0
        if current_cache_ready and self._backhaul_aware_redundant_fill_penalty > 0.0:
            redundant_fill_penalty = self._backhaul_aware_redundant_fill_penalty * (1.0 if no_trusted_signal else 0.55)
            slow_logits[1] = slow_logits[1] - redundant_fill_penalty
            env_action_logits_bias[0] = env_action_logits_bias[0] - redundant_fill_penalty
            steady_bias = self._backhaul_aware_steady_bias * (0.70 + 0.30 * service_pressure)
            if steady_bias > 1e-8:
                fast_logits[0] = fast_logits[0] + 0.30 * steady_bias
                env_action_logits_bias[3] = env_action_logits_bias[3] + steady_bias

        prefetch_penalty = 0.0
        prepare_penalty = 0.0
        if no_trusted_signal:
            low_service_relief = max(0.35, 1.0 - service_pressure)
            prefetch_penalty = self._backhaul_aware_no_signal_prefetch_penalty * low_service_relief
            prepare_penalty = self._backhaul_aware_no_signal_prepare_penalty * low_service_relief
            if prefetch_penalty > 1e-8:
                slow_logits[2] = slow_logits[2] - prefetch_penalty
                env_action_logits_bias[1] = env_action_logits_bias[1] - prefetch_penalty
            if prepare_penalty > 1e-8:
                event_logits[1] = event_logits[1] - prepare_penalty
                event_logits[0] = event_logits[0] + 0.12 * prepare_penalty
                env_action_logits_bias[4] = env_action_logits_bias[4] - prepare_penalty

        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        adjusted["backhaul_aware_policy_info"] = {
            "enabled": True,
            "service_pressure": round(float(service_pressure), 6),
            "service_fill_bias": round(float(service_fill_bias), 6),
            "redundant_fill_penalty": round(float(redundant_fill_penalty), 6),
            "prefetch_penalty": round(float(prefetch_penalty), 6),
            "prepare_penalty": round(float(prepare_penalty), 6),
            "steady_bias": round(float(steady_bias), 6),
            "current_cache_ready": bool(current_cache_ready),
            "missing_current_adapter": bool(required_adapter_key and not current_cache_ready),
            "trusted_candidate": bool(trusted_candidate),
            "no_trusted_signal": bool(no_trusted_signal),
            "remaining_nodes": int(remaining_nodes),
            "service_pressure_floor": round(float(self._backhaul_aware_service_pressure_floor), 6),
            "env_action_logit_bias": [
                round(float(item), 6)
                for item in env_action_logits_bias.detach().cpu().tolist()
            ],
        }
        return adjusted

    def _net_advantage_prepare_context(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        if not self._net_advantage_prepare_gate_enabled:
            return {"enabled": False, "reason": "disabled", "net_advantage_score": 0.0}

        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = _clamp01(float(timing_features.get("prepare_window_score", 0.0) or 0.0))
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        timing_support = max(prepare_score, temporal_urgency)

        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predictions = semantic_state.get("predictions", {})
        if not isinstance(predictions, dict):
            predictions = {}
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        if predicted_next_rsu_id is None and next_sequence:
            predicted_next_rsu_id = next_sequence[0]
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        if predicted_handoff_target_rsu_id is None:
            for rsu_id in next_sequence:
                if rsu_id is not None and str(rsu_id) != str(current_rsu_id):
                    predicted_handoff_target_rsu_id = rsu_id
                    break

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        required_adapter_key = str(required_adapter) if required_adapter else None
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        predicted_next_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_next_rsu_id)
        handoff_target_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_handoff_target_rsu_id)

        def has_adapter(rsu: dict[str, Any]) -> bool:
            return bool(
                required_adapter_key
                and required_adapter_key in {str(item) for item in rsu.get("cached_adapter_ids", [])}
            )

        current_cache_ready = has_adapter(current_rsu)
        predicted_next_cache_ready = has_adapter(predicted_next_rsu)
        handoff_target_cache_ready = has_adapter(handoff_target_rsu)
        next_differs = bool(
            predicted_next_rsu_id is not None and str(predicted_next_rsu_id) != str(current_rsu_id)
        )
        target_differs = bool(
            predicted_handoff_target_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
        )
        predicted_target_valid = bool(target_differs and bool(handoff_target_rsu))
        diagnostics = self._build_prediction_target_diagnostics(
            semantic_state=semantic_state,
            temporal_urgency=temporal_urgency,
            predicted_handoff_target_valid=predicted_target_valid,
        )
        confidence = _clamp01(float(diagnostics.get("prediction_confidence", 0.0) or 0.0))
        uncertainty = _clamp01(float(diagnostics.get("prediction_uncertainty", 1.0) or 1.0))
        reliability = _clamp01(confidence * (1.0 - uncertainty))
        raw_candidate = bool(diagnostics.get("raw_handoff_candidate", False))
        gate_pass = bool(diagnostics.get("gate_pass", False))

        workflow = semantic_state.get("workflow") if isinstance(semantic_state.get("workflow"), dict) else {}
        nodes = workflow.get("nodes", []) if isinstance(workflow, dict) else []
        completed_nodes = set(str(item) for item in workflow.get("completed_node_ids", []) or [])
        remaining_nodes = 0
        if isinstance(nodes, list) and nodes:
            for node in nodes:
                node_id = str((node or {}).get("node_id", ""))
                if node_id and node_id not in completed_nodes:
                    remaining_nodes += 1
        else:
            remaining_nodes = 1 if current_node else 0
        successors = current_node.get("successors", []) if isinstance(current_node, dict) else []
        try:
            input_size = max(float(current_node.get("input_size", 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            input_size = 0.0
        try:
            continuity_features = build_graph_continuity_critic_features(
                semantic_state,
                prediction_gate_min_leak=self._prediction_gate_min_leak,
            )
        except Exception:
            continuity_features = {}
        path_pressure = _clamp01(
            0.55 * float(continuity_features.get("critical_path_length_norm", 0.0) or 0.0)
            + 0.25 * float(continuity_features.get("frontier_width_ratio", 0.0) or 0.0)
            + min(float(remaining_nodes) / 10.0, 0.20)
        )
        service_pressure = _clamp01(
            0.16
            + 0.18 * float(bool(successors))
            + 0.18 * float(remaining_nodes >= 3)
            + 0.16 * float(required_adapter_key is not None)
            + min(input_size / 256.0, 0.12)
            + 0.20 * path_pressure
        )

        prediction_support = _clamp01(
            0.30 * prepare_score
            + 0.22 * temporal_urgency
            + 0.22 * reliability
            + 0.10 * float(gate_pass)
            + 0.08 * float(raw_candidate)
            + 0.08 * float(predicted_target_valid)
        )
        execution_support = _clamp01(
            0.24 * float(current_cache_ready)
            + 0.18 * float(handoff_target_cache_ready)
            + 0.18 * float(target_differs)
            + 0.16 * service_pressure
            + 0.14 * path_pressure
            + 0.10 * float(predicted_next_cache_ready)
        )
        target_requires_prefetch = bool(required_adapter_key and target_differs and not handoff_target_cache_ready)
        missing_current_adapter = bool(required_adapter_key and current_rsu_id is not None and not current_cache_ready)
        no_target_penalty = float(not target_differs)
        stale_prefetch_risk = float(target_requires_prefetch) * (1.0 - timing_support)
        uncertainty_risk = 0.55 * uncertainty + 0.45 * (1.0 - confidence)
        cost_pressure = _clamp01(
            0.32 * float(missing_current_adapter)
            + 0.24 * stale_prefetch_risk
            + 0.18 * no_target_penalty
            + 0.16 * uncertainty_risk
            + 0.10 * float(target_requires_prefetch and not gate_pass)
        )
        net_advantage_score = _clamp01(
            prediction_support
            + 0.42 * execution_support
            - self._net_advantage_prepare_gate_cost_scale * cost_pressure
        )
        return {
            "enabled": True,
            "net_advantage_score": round(float(net_advantage_score), 6),
            "prediction_support": round(float(prediction_support), 6),
            "execution_support": round(float(execution_support), 6),
            "cost_pressure": round(float(cost_pressure), 6),
            "service_pressure": round(float(service_pressure), 6),
            "path_pressure": round(float(path_pressure), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "timing_support": round(float(timing_support), 6),
            "prediction_confidence": round(float(confidence), 6),
            "prediction_uncertainty": round(float(uncertainty), 6),
            "prediction_reliability": round(float(reliability), 6),
            "raw_handoff_candidate": bool(raw_candidate),
            "gate_pass": bool(gate_pass),
            "predicted_target_valid": bool(predicted_target_valid),
            "current_cache_ready": bool(current_cache_ready),
            "predicted_next_cache_ready": bool(predicted_next_cache_ready),
            "handoff_target_cache_ready": bool(handoff_target_cache_ready),
            "target_requires_prefetch": bool(target_requires_prefetch),
            "missing_current_adapter": bool(missing_current_adapter),
            "next_differs": bool(next_differs),
            "target_differs": bool(target_differs),
            "current_rsu_id": current_rsu_id,
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
        }

    def _apply_net_advantage_prepare_gate(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._use_hierarchy or not self._net_advantage_prepare_gate_enabled:
            return policy_output

        gate_info = self._net_advantage_prepare_context(semantic_state)
        score = float(gate_info.get("net_advantage_score", 0.0) or 0.0)
        positive_span = max(1.0 - self._net_advantage_prepare_gate_min_score, 1e-6)
        positive_scale = _clamp01((score - self._net_advantage_prepare_gate_min_score) / positive_span)
        negative_threshold = min(
            self._net_advantage_prepare_gate_min_score + self._net_advantage_prepare_gate_margin,
            1.0,
        )
        negative_scale = _clamp01((negative_threshold - score) / max(negative_threshold, 1e-6))

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)

        prepare_bias = self._net_advantage_prepare_gate_bias * positive_scale
        prepare_penalty = self._net_advantage_prepare_gate_bias * negative_scale
        coverage_recovery_context = bool(
            gate_info.get("current_rsu_id") is None
            and (
                gate_info.get("predicted_target_valid", False)
                or gate_info.get("target_differs", False)
            )
        )
        coverage_recovery_scale = 0.0
        if coverage_recovery_context:
            coverage_context_score = _clamp01(
                0.24
                + 0.30 * float(gate_info.get("timing_support", 0.0) or 0.0)
                + 0.20 * float(gate_info.get("prediction_support", 0.0) or 0.0)
                + 0.16 * float(gate_info.get("execution_support", 0.0) or 0.0)
                + 0.10 * float(gate_info.get("net_advantage_score", 0.0) or 0.0)
            )
            coverage_recovery_scale = _clamp01(
                max(
                    positive_scale,
                    self._coverage_recovery_gate_min_scale,
                    coverage_context_score,
                )
            )
        coverage_recovery_bias = 0.0
        if prepare_bias > 1e-8:
            event_logits[1] = event_logits[1] + prepare_bias
            event_logits[0] = event_logits[0] - 0.12 * prepare_bias
            env_action_logits_bias[4] = env_action_logits_bias[4] + prepare_bias
        if prepare_penalty > 1e-8:
            effective_prepare_penalty = prepare_penalty
            if coverage_recovery_context:
                effective_prepare_penalty *= 1.0 - 0.85 * coverage_recovery_scale
            event_logits[1] = event_logits[1] - effective_prepare_penalty
            event_logits[0] = event_logits[0] + 0.14 * effective_prepare_penalty
            env_action_logits_bias[4] = env_action_logits_bias[4] - effective_prepare_penalty

        prefetch_bias = 0.0
        prefetch_penalty = 0.0
        if gate_info.get("target_requires_prefetch", False):
            prefetch_bias = (
                self._net_advantage_prepare_gate_bias
                * self._net_advantage_prepare_gate_prefetch_scale
                * positive_scale
                * max(float(gate_info.get("timing_support", 0.0) or 0.0), 0.35)
            )
            if prefetch_bias > 1e-8:
                slow_logits[2] = slow_logits[2] + prefetch_bias
                slow_logits[0] = slow_logits[0] - 0.08 * prefetch_bias
                env_action_logits_bias[1] = env_action_logits_bias[1] + prefetch_bias
        if negative_scale > 1e-8:
            prefetch_penalty = (
                self._net_advantage_prepare_gate_bias
                * self._net_advantage_prepare_gate_prefetch_scale
                * negative_scale
                * (1.15 if not gate_info.get("target_differs", False) else 1.0)
            )
            slow_logits[2] = slow_logits[2] - prefetch_penalty
            env_action_logits_bias[1] = env_action_logits_bias[1] - prefetch_penalty

        current_bias = 0.0
        local_bias = 0.0
        if negative_scale > 1e-8:
            relief = self._net_advantage_prepare_gate_current_scale * prepare_penalty
            if gate_info.get("current_rsu_id") is not None:
                current_action = 3 if gate_info.get("current_cache_ready", False) else 0
                current_bias = relief
                env_action_logits_bias[current_action] = env_action_logits_bias[current_action] + current_bias
                if current_action == 0:
                    slow_logits[1] = slow_logits[1] + 0.36 * current_bias
                else:
                    fast_logits[0] = fast_logits[0] + 0.30 * current_bias
            elif gate_info.get("predicted_target_valid", False) or gate_info.get("target_differs", False):
                current_bias = 0.85 * relief
                event_logits[1] = event_logits[1] + current_bias
                event_logits[0] = event_logits[0] - 0.10 * current_bias
                env_action_logits_bias[4] = env_action_logits_bias[4] + current_bias
                env_action_logits_bias[2] = env_action_logits_bias[2] - 1.00 * relief
                fast_logits[1] = fast_logits[1] - 0.62 * relief
            else:
                local_bias = relief
                env_action_logits_bias[2] = env_action_logits_bias[2] + local_bias
                fast_logits[1] = fast_logits[1] + 0.32 * local_bias

        coverage_recovery_fallback_suppression = 0.0
        coverage_recovery_fast_suppression = 0.0
        coverage_recovery_current_suppression = 0.0
        if coverage_recovery_context and coverage_recovery_scale > 1e-8:
            coverage_recovery_bias = (
                self._net_advantage_prepare_gate_bias
                * self._coverage_recovery_gate_bias_scale
                * coverage_recovery_scale
            )
            coverage_recovery_fallback_suppression = (
                self._net_advantage_prepare_gate_bias
                * self._coverage_recovery_gate_fallback_suppression_scale
                * coverage_recovery_scale
            )
            coverage_recovery_fast_suppression = (
                self._net_advantage_prepare_gate_bias
                * self._coverage_recovery_gate_fast_suppression_scale
                * coverage_recovery_scale
            )
            coverage_recovery_current_suppression = (
                self._net_advantage_prepare_gate_bias
                * self._coverage_recovery_gate_current_suppression_scale
                * coverage_recovery_scale
            )
            event_logits[1] = event_logits[1] + coverage_recovery_bias
            event_logits[0] = event_logits[0] - 0.12 * coverage_recovery_bias
            env_action_logits_bias[4] = env_action_logits_bias[4] + coverage_recovery_bias
            env_action_logits_bias[2] = (
                env_action_logits_bias[2] - coverage_recovery_fallback_suppression
            )
            fast_logits[1] = fast_logits[1] - coverage_recovery_fast_suppression
            if coverage_recovery_current_suppression > 1e-8:
                env_action_logits_bias[0] = (
                    env_action_logits_bias[0] - coverage_recovery_current_suppression
                )
                env_action_logits_bias[3] = (
                    env_action_logits_bias[3] - coverage_recovery_current_suppression
                )

        service_fill_bias = 0.0
        local_penalty = 0.0
        service_pressure = _clamp01(float(gate_info.get("service_pressure", 0.0) or 0.0))
        if (
            gate_info.get("current_rsu_id") is not None
            and self._net_advantage_prepare_gate_service_fill_scale > 0.0
            and service_pressure > 1e-8
        ):
            mechanism_relief = 1.0 - 0.35 * positive_scale
            current_cache_ready = bool(gate_info.get("current_cache_ready", False))
            service_current_action = 3 if current_cache_ready else 0
            current_cache_scale = 0.62 if current_cache_ready else 1.0
            service_fill_bias = (
                self._net_advantage_prepare_gate_bias
                * self._net_advantage_prepare_gate_service_fill_scale
                * service_pressure
                * current_cache_scale
                * max(mechanism_relief, 0.45)
            )
            env_action_logits_bias[service_current_action] = (
                env_action_logits_bias[service_current_action] + service_fill_bias
            )
            if service_current_action == 0:
                slow_logits[1] = slow_logits[1] + service_fill_bias
                slow_logits[2] = slow_logits[2] - 0.08 * service_fill_bias
            else:
                fast_logits[0] = fast_logits[0] + 0.42 * service_fill_bias
            fast_logits[1] = fast_logits[1] - 0.18 * service_fill_bias
        if (
            self._net_advantage_prepare_gate_local_penalty_scale > 0.0
            and gate_info.get("current_rsu_id") is not None
            and service_pressure > 1e-8
        ):
            local_penalty_context = bool(
                not gate_info.get("target_differs", False)
                or gate_info.get("current_cache_ready", False)
                or score < negative_threshold
            )
        else:
            local_penalty_context = False
        if local_penalty_context:
            local_penalty = (
                self._net_advantage_prepare_gate_bias
                * self._net_advantage_prepare_gate_local_penalty_scale
                * service_pressure
                * (0.60 + 0.40 * negative_scale)
            )
            fast_logits[1] = fast_logits[1] - local_penalty
            env_action_logits_bias[2] = env_action_logits_bias[2] - local_penalty

        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        gate_info.update(
            {
                "min_score": round(float(self._net_advantage_prepare_gate_min_score), 6),
                "margin": round(float(self._net_advantage_prepare_gate_margin), 6),
                "positive_scale": round(float(positive_scale), 6),
                "negative_scale": round(float(negative_scale), 6),
                "prepare_bias": round(float(prepare_bias), 6),
                "prepare_penalty": round(float(prepare_penalty), 6),
                "coverage_recovery_scale": round(float(coverage_recovery_scale), 6),
                "coverage_recovery_bias": round(float(coverage_recovery_bias), 6),
                "coverage_recovery_fallback_suppression": round(
                    float(coverage_recovery_fallback_suppression),
                    6,
                ),
                "coverage_recovery_fast_suppression": round(
                    float(coverage_recovery_fast_suppression),
                    6,
                ),
                "coverage_recovery_current_suppression": round(
                    float(coverage_recovery_current_suppression),
                    6,
                ),
                "prefetch_bias": round(float(prefetch_bias), 6),
                "prefetch_penalty": round(float(prefetch_penalty), 6),
                "current_bias": round(float(current_bias), 6),
                "local_bias": round(float(local_bias), 6),
                "service_fill_bias": round(float(service_fill_bias), 6),
                "local_penalty": round(float(local_penalty), 6),
                "env_action_logit_bias": [
                    round(float(item), 6)
                    for item in env_action_logits_bias.detach().cpu().tolist()
                ],
            }
        )
        adjusted["net_advantage_prepare_gate_info"] = gate_info
        return adjusted

    def _service_completion_context(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        if not self._service_completion_gate_enabled:
            return {"enabled": False, "reason": "disabled", "service_completion_score": 0.0}

        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        required_adapter_key = str(required_adapter) if required_adapter else None
        workflow = semantic_state.get("workflow") if isinstance(semantic_state.get("workflow"), dict) else {}
        nodes = workflow.get("nodes", []) if isinstance(workflow, dict) else []
        completed_nodes = set(str(item) for item in workflow.get("completed_node_ids", []) or [])
        remaining_nodes = 0
        if isinstance(nodes, list) and nodes:
            for node in nodes:
                node_id = str((node or {}).get("node_id", ""))
                if node_id and node_id not in completed_nodes:
                    remaining_nodes += 1
        else:
            remaining_nodes = 1 if current_node else 0
        if current_rsu_id is None or remaining_nodes <= 0:
            return {
                "enabled": True,
                "active": False,
                "reason": "no_current_rsu_or_no_remaining_node",
                "service_completion_score": 0.0,
                "current_rsu_id": current_rsu_id,
                "remaining_nodes": int(remaining_nodes),
            }

        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        current_cache_ready = bool(
            required_adapter_key
            and required_adapter_key in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predictions = semantic_state.get("predictions", {})
        if not isinstance(predictions, dict):
            predictions = {}
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        next_differs = bool(predicted_next_rsu_id is not None and str(predicted_next_rsu_id) != str(current_rsu_id))
        target_differs = bool(
            predicted_handoff_target_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
        )
        try:
            continuity_features = build_graph_continuity_critic_features(
                semantic_state,
                prediction_gate_min_leak=self._prediction_gate_min_leak,
            )
        except Exception:
            continuity_features = {}
        path_pressure = _clamp01(
            0.60 * float(continuity_features.get("critical_path_length_norm", 0.0) or 0.0)
            + 0.28 * float(continuity_features.get("frontier_width_ratio", 0.0) or 0.0)
            + min(float(remaining_nodes) / 10.0, 0.12)
        )
        threshold = max(float(self._service_completion_gate_remaining_nodes_threshold), 1.0)
        terminal_pressure = _clamp01((threshold - float(remaining_nodes) + 1.0) / threshold)
        service_pressure = _clamp01(
            0.24
            + 0.28 * terminal_pressure
            + 0.18 * path_pressure
            + 0.16 * float(required_adapter_key is not None)
            + 0.14 * float(current_cache_ready)
        )

        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = _clamp01(float(timing_features.get("prepare_window_score", 0.0) or 0.0))
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        handoff_pressure = _clamp01(0.55 * prepare_score + 0.45 * temporal_urgency)
        completion_score = _clamp01(
            0.40 * terminal_pressure
            + 0.30 * service_pressure
            + 0.20 * float(current_cache_ready)
            + 0.10 * (1.0 - 0.5 * handoff_pressure)
        )
        immediate_transfer_context = bool(next_differs and target_differs and handoff_pressure >= 0.20)
        active = bool(
            remaining_nodes <= self._service_completion_gate_remaining_nodes_threshold
            and not immediate_transfer_context
        )
        target_action = 3 if current_cache_ready else 0
        return {
            "enabled": True,
            "active": bool(active),
            "reason": (
                "service_completion_window"
                if active
                else "immediate_transfer_context"
                if immediate_transfer_context
                else "outside_completion_window"
            ),
            "service_completion_score": round(float(completion_score), 6),
            "service_pressure": round(float(service_pressure), 6),
            "terminal_pressure": round(float(terminal_pressure), 6),
            "path_pressure": round(float(path_pressure), 6),
            "handoff_pressure": round(float(handoff_pressure), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "target_action": int(target_action),
            "current_cache_ready": bool(current_cache_ready),
            "current_rsu_id": current_rsu_id,
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "next_differs": bool(next_differs),
            "target_differs": bool(target_differs),
            "remaining_nodes": int(remaining_nodes),
            "required_adapter": required_adapter_key,
        }

    def _apply_service_completion_gate(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._use_hierarchy or not self._service_completion_gate_enabled:
            return policy_output

        gate_info = self._service_completion_context(semantic_state)
        if not bool(gate_info.get("active", False)):
            adjusted = dict(policy_output)
            adjusted["service_completion_gate_info"] = gate_info
            return adjusted

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)

        score = _clamp01(float(gate_info.get("service_completion_score", 0.0) or 0.0))
        bias = self._service_completion_gate_bias * score
        target_action = int(gate_info.get("target_action", 3))
        event_suppression = self._service_completion_gate_event_suppression_scale * bias
        prefetch_suppression = self._service_completion_gate_prefetch_suppression_scale * bias
        fallback_suppression = self._service_completion_gate_fallback_suppression_scale * bias
        env_action_logits_bias[target_action] = env_action_logits_bias[target_action] + bias
        env_action_logits_bias[1] = env_action_logits_bias[1] - prefetch_suppression
        env_action_logits_bias[2] = env_action_logits_bias[2] - fallback_suppression
        env_action_logits_bias[4] = env_action_logits_bias[4] - event_suppression
        event_logits[1] = event_logits[1] - event_suppression
        event_logits[0] = event_logits[0] + 0.18 * event_suppression
        slow_logits[2] = slow_logits[2] - prefetch_suppression
        fast_logits[1] = fast_logits[1] - fallback_suppression
        if target_action == 0:
            slow_logits[1] = slow_logits[1] + bias
            slow_logits[0] = slow_logits[0] - 0.10 * bias
            fast_logits[0] = fast_logits[0] + 0.18 * bias
        else:
            slow_logits[0] = slow_logits[0] + 0.24 * bias
            fast_logits[0] = fast_logits[0] + bias

        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        gate_info.update(
            {
                "bias": round(float(bias), 6),
                "event_suppression": round(float(event_suppression), 6),
                "prefetch_suppression": round(float(prefetch_suppression), 6),
                "fallback_suppression": round(float(fallback_suppression), 6),
                "env_action_logit_bias": [
                    round(float(item), 6)
                    for item in env_action_logits_bias.detach().cpu().tolist()
                ],
            }
        )
        adjusted["service_completion_gate_info"] = gate_info
        return adjusted

    def _apply_sparse_handoff_recovery_prior(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._use_hierarchy or not self._sparse_handoff_recovery_prior_enabled:
            return policy_output

        run_metadata = dict(run_metadata or {})
        window_class = str(
            semantic_state.get("window_class")
            or (semantic_state.get("run_info", {}) or {}).get("window_class")
            or run_metadata.get("window_class", "")
        )
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        prepare_score = _clamp01(float(timing_features.get("prepare_window_score", 0.0) or 0.0))
        temporal_urgency = _clamp01(float(timing_features.get("temporal_urgency", 0.0) or 0.0))
        predicted_target_valid = self._semantic_state_has_valid_predicted_handoff_target(semantic_state)
        diagnostics = self._build_prediction_target_diagnostics(
            semantic_state=semantic_state,
            temporal_urgency=temporal_urgency,
            predicted_handoff_target_valid=predicted_target_valid,
        )
        confidence = _clamp01(float(diagnostics.get("prediction_confidence", 0.0) or 0.0))
        uncertainty = _clamp01(float(diagnostics.get("prediction_uncertainty", 1.0) or 1.0))
        reliability = _clamp01(confidence * (1.0 - uncertainty))

        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        first_eta = int(diagnostics.get("predicted_first_non_current_eta", 0) or 0)
        target_rsu_id = diagnostics.get("predicted_first_non_current_rsu")
        if target_rsu_id is None:
            predictions = semantic_state.get("predictions", {})
            vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
            if isinstance(predictions, dict) and vehicle_id:
                target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        target_differs = bool(target_rsu_id is not None and str(target_rsu_id) != str(current_rsu_id))
        lead_threshold = max(int(math.ceil(self._temporal_prepare_lead_steps)) + 1, 3)
        if first_eta <= 0 and target_differs:
            first_eta = lead_threshold

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        required_adapter_key = str(required_adapter) if required_adapter else None
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        target_rsu = _rsu_by_id_from_semantic_state(semantic_state, target_rsu_id)

        def has_adapter(rsu: dict[str, Any]) -> bool:
            return bool(
                required_adapter_key
                and required_adapter_key in {str(item) for item in rsu.get("cached_adapter_ids", [])}
            )

        current_cache_ready = has_adapter(current_rsu)
        target_cache_ready = has_adapter(target_rsu)
        current_cache_missing = bool(required_adapter_key and current_rsu_id is not None and not current_cache_ready)
        target_cache_missing = bool(required_adapter_key and target_differs and not target_cache_ready)
        raw_candidate = bool(diagnostics.get("raw_handoff_candidate", False))
        sequence_contains_other = bool(diagnostics.get("predicted_sequence_contains_other_rsu", False))
        sparse_window = window_class == "idle_or_sparse"
        eta_support = 1.0
        if first_eta > 0:
            eta_support = _clamp01(
                1.0 - (float(first_eta) - 1.0) / float(max(self._sparse_handoff_recovery_max_eta, 1))
            )
        recovery_context = _clamp01(
            0.30 * reliability
            + 0.20 * prepare_score
            + 0.18 * temporal_urgency
            + 0.14 * eta_support
            + 0.10 * float(raw_candidate or sequence_contains_other)
            + 0.08 * float(sparse_window)
        )
        active = bool(
            sparse_window
            and required_adapter_key
            and target_differs
            and (raw_candidate or sequence_contains_other or predicted_target_valid)
            and 0 < first_eta <= self._sparse_handoff_recovery_max_eta
            and recovery_context + 1e-8 >= self._sparse_handoff_recovery_min_context
        )
        prior_info = {
            "enabled": True,
            "active": bool(active),
            "window_class": window_class,
            "recovery_context": round(float(recovery_context), 6),
            "prepare_window_score": round(float(prepare_score), 6),
            "temporal_urgency": round(float(temporal_urgency), 6),
            "prediction_reliability": round(float(reliability), 6),
            "predicted_first_non_current_eta": int(first_eta),
            "current_rsu_id": current_rsu_id,
            "target_rsu_id": target_rsu_id,
            "required_adapter": required_adapter_key,
            "current_cache_ready": bool(current_cache_ready),
            "target_cache_ready": bool(target_cache_ready),
            "target_cache_missing": bool(target_cache_missing),
            "raw_handoff_candidate": bool(raw_candidate),
            "predicted_target_valid": bool(predicted_target_valid),
        }
        if not active:
            adjusted = dict(policy_output)
            adjusted["sparse_handoff_recovery_prior_info"] = prior_info
            return adjusted

        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)

        horizon = float(max(self._sparse_handoff_recovery_max_eta, 1))
        prefetch_freshness_scale = 0.72 + 0.28 * _clamp01(float(first_eta) / horizon)
        prepare_timing_scale = 0.62 + 0.38 * max(prepare_score, temporal_urgency, float(target_cache_ready))
        current_fill_scale = 0.56 + 0.44 * (1.0 - max(prepare_score, temporal_urgency))
        prefetch_bias = 0.0
        prepare_bias = 0.0
        current_fill_bias = 0.0
        steady_bias = 0.0
        local_penalty = self._sparse_handoff_recovery_local_penalty * recovery_context

        if target_cache_missing:
            prefetch_bias = (
                self._sparse_handoff_recovery_prefetch_bias
                * recovery_context
                * prefetch_freshness_scale
            )
            slow_logits[2] = slow_logits[2] + prefetch_bias
            slow_logits[0] = slow_logits[0] - 0.10 * prefetch_bias
            env_action_logits_bias[1] = env_action_logits_bias[1] + prefetch_bias
            env_action_logits_bias[3] = env_action_logits_bias[3] - 0.18 * prefetch_bias

        if target_cache_ready or first_eta <= lead_threshold:
            not_ready_scale = 0.48 if target_cache_missing else 1.0
            prepare_bias = (
                self._sparse_handoff_recovery_prepare_bias
                * recovery_context
                * prepare_timing_scale
                * not_ready_scale
            )
            event_logits[1] = event_logits[1] + prepare_bias
            event_logits[0] = event_logits[0] - 0.16 * prepare_bias
            env_action_logits_bias[4] = env_action_logits_bias[4] + prepare_bias
            if target_cache_ready:
                env_action_logits_bias[1] = env_action_logits_bias[1] - 0.16 * prepare_bias

        if current_cache_missing:
            current_fill_bias = (
                self._sparse_handoff_recovery_current_fill_bias
                * recovery_context
                * current_fill_scale
            )
            env_action_logits_bias[0] = env_action_logits_bias[0] + current_fill_bias
            slow_logits[1] = slow_logits[1] + 0.80 * current_fill_bias
            slow_logits[0] = slow_logits[0] - 0.08 * current_fill_bias

        if target_cache_ready and current_cache_ready and first_eta > lead_threshold:
            steady_bias = self._sparse_handoff_recovery_steady_bias * recovery_context
            env_action_logits_bias[3] = env_action_logits_bias[3] + steady_bias
            fast_logits[0] = fast_logits[0] + 0.62 * steady_bias
            event_logits[1] = event_logits[1] - 0.12 * steady_bias

        if local_penalty > 1e-8:
            env_action_logits_bias[2] = env_action_logits_bias[2] - local_penalty
            fast_logits[1] = fast_logits[1] - 0.45 * local_penalty

        option_prepare_bias = 0.0
        option_popularity_penalty = 0.0
        option_local_penalty = 0.0
        option_logits = adjusted.get("option_logits")
        if (
            self._sparse_handoff_option_prior_enabled
            and isinstance(option_logits, torch.Tensor)
            and option_logits.numel() >= 4
            and recovery_context + 1e-8 >= self._sparse_handoff_option_min_context
        ):
            option_logits = option_logits.clone()
            option_timing_scale = 0.58 + 0.42 * max(prepare_score, temporal_urgency, eta_support)
            option_prepare_bias = (
                self._sparse_handoff_option_prepare_bias
                * recovery_context
                * option_timing_scale
            )
            option_popularity_penalty = (
                self._sparse_handoff_option_popularity_penalty
                * recovery_context
                * (0.50 + 0.50 * eta_support)
            )
            option_local_penalty = (
                self._sparse_handoff_option_local_penalty
                * recovery_context
                * (0.45 + 0.55 * max(prepare_score, temporal_urgency))
            )
            option_logits[3] = option_logits[3] + option_prepare_bias
            option_logits[1] = option_logits[1] - option_popularity_penalty
            option_logits[2] = option_logits[2] - option_local_penalty
            adjusted["option_logits"] = option_logits

        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        prior_info.update(
            {
                "lead_threshold": int(lead_threshold),
                "prefetch_bias": round(float(prefetch_bias), 6),
                "prepare_bias": round(float(prepare_bias), 6),
                "current_fill_bias": round(float(current_fill_bias), 6),
                "steady_bias": round(float(steady_bias), 6),
                "local_penalty": round(float(local_penalty), 6),
                "option_prepare_bias": round(float(option_prepare_bias), 6),
                "option_popularity_penalty": round(float(option_popularity_penalty), 6),
                "option_local_penalty": round(float(option_local_penalty), 6),
                "env_action_logit_bias": [
                    round(float(item), 6)
                    for item in env_action_logits_bias.detach().cpu().tolist()
                ],
            }
        )
        adjusted["sparse_handoff_recovery_prior_info"] = prior_info
        return adjusted

    def _apply_policy_adjustments(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        adjusted = self._apply_digital_twin_policy_prior(
            policy_output,
            semantic_state,
            run_metadata=run_metadata,
        )
        adjusted = self._apply_opportunity_constrained_policy(adjusted, semantic_state)
        adjusted = self._apply_backhaul_aware_policy(adjusted, semantic_state)
        adjusted = self._apply_net_advantage_prepare_gate(adjusted, semantic_state)
        adjusted = self._apply_sparse_handoff_recovery_prior(
            adjusted,
            semantic_state,
            run_metadata=run_metadata,
        )
        adjusted = self._apply_continuity_guard(adjusted, semantic_state)
        adjusted = self._apply_service_completion_gate(adjusted, semantic_state)
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

    def _effective_digital_twin_policy_prior_distill_coef(self) -> float:
        if (
            not self._digital_twin_policy_prior_enabled
            or self._digital_twin_policy_prior_distill_coef <= 0.0
        ):
            return 0.0
        if self._update_count < self._digital_twin_policy_prior_distill_warmup_updates:
            return self._digital_twin_policy_prior_distill_coef
        decay_steps = self._update_count - self._digital_twin_policy_prior_distill_warmup_updates + 1
        return float(self._digital_twin_policy_prior_distill_coef * (self._digital_twin_policy_prior_distill_decay ** decay_steps))

    def _digital_twin_wait_readiness_context(
        self,
        semantic_state: dict[str, Any],
        *,
        timing_support: float,
        boundary_urgency: float = 0.0,
        handoff_context: bool = False,
    ) -> dict[str, Any]:
        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        predictions = semantic_state.get("predictions", {})
        if not isinstance(predictions, dict):
            predictions = {}
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
        if predicted_next_rsu_id is None and next_sequence:
            predicted_next_rsu_id = next_sequence[0]
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        predicted_next_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_next_rsu_id)
        handoff_target_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_handoff_target_rsu_id)
        adapter_id = str(required_adapter) if required_adapter else ""
        current_cache_ready = bool(
            adapter_id
            and adapter_id in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )
        predicted_next_cache_ready = bool(
            adapter_id
            and adapter_id in {str(item) for item in predicted_next_rsu.get("cached_adapter_ids", [])}
        )
        handoff_target_cache_ready = bool(
            adapter_id
            and adapter_id in {str(item) for item in handoff_target_rsu.get("cached_adapter_ids", [])}
        )
        target_differs = bool(
            predicted_handoff_target_rsu_id is not None
            and current_rsu_id is not None
            and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
        )
        next_differs = bool(
            predicted_next_rsu_id is not None
            and current_rsu_id is not None
            and str(predicted_next_rsu_id) != str(current_rsu_id)
        )
        ready_score = max(
            1.0 if handoff_target_cache_ready else 0.0,
            0.78 if predicted_next_cache_ready else 0.0,
            0.56 if current_cache_ready else 0.0,
        )
        wait_preferred = bool(
            self._digital_twin_policy_prior_adaptive_wait_enabled
            and ready_score >= self._digital_twin_policy_prior_wait_ready_threshold
            and float(timing_support) <= self._digital_twin_policy_prior_wait_timing_ceiling
            and (handoff_context or target_differs or next_differs)
            and float(boundary_urgency) < 0.82
        )
        return {
            "wait_preferred": wait_preferred,
            "ready_score": round(float(_clamp01(ready_score)), 6),
            "current_cache_ready": current_cache_ready,
            "predicted_next_cache_ready": predicted_next_cache_ready,
            "handoff_target_cache_ready": handoff_target_cache_ready,
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "current_rsu_id": current_rsu_id,
            "target_differs": target_differs,
            "next_differs": next_differs,
        }

    def _build_digital_twin_policy_prior_annotation(
        self,
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._digital_twin_policy_prior_enabled or not self._use_hierarchy:
            return {"apply": False, "reason": "disabled"}

        window_class = str((run_metadata or {}).get("window_class") or semantic_state.get("window_class", "unknown"))
        dt_features = _build_digital_twin_handoff_feature_tensor(semantic_state)
        feature_values = [float(item) for item in dt_features.detach().cpu().tolist()]
        (
            has_prediction,
            next_differs,
            target_differs,
            confidence,
            uncertainty,
            _dwell_norm,
            _horizon_norm,
            switch_ratio,
            _unique_future_ratio,
            non_current_ratio,
            eta_norm,
            boundary_urgency,
            service_pressure,
            load_pressure,
        ) = feature_values
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=self._temporal_prepare_lead_steps,
            sigma=self._temporal_prepare_sigma,
        )
        timing_support = max(
            float(timing_features.get("prepare_window_score", 0.0) or 0.0),
            float(timing_features.get("temporal_urgency", 0.0) or 0.0),
        )
        confidence_support = max(
            self._digital_twin_policy_prior_confidence_floor if has_prediction > 0.0 else 0.0,
            _clamp01(confidence * (1.0 - 0.45 * uncertainty)),
        )

        current_node = semantic_state.get("current_workflow_node") or {}
        required_adapter = current_node.get("required_adapter")
        predictions = semantic_state.get("predictions", {}) if isinstance(semantic_state.get("predictions", {}), dict) else {}
        primary_vehicle, _ = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []) or [])
        if predicted_next_rsu_id is None and next_sequence:
            predicted_next_rsu_id = next_sequence[0]
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = _rsu_by_id_from_semantic_state(semantic_state, current_rsu_id)
        predicted_next_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_next_rsu_id)
        handoff_target_rsu = _rsu_by_id_from_semantic_state(semantic_state, predicted_handoff_target_rsu_id)
        current_cache_ready = bool(
            required_adapter
            and required_adapter in {str(item) for item in current_rsu.get("cached_adapter_ids", [])}
        )
        predicted_next_cache_ready = bool(
            required_adapter
            and required_adapter in {str(item) for item in predicted_next_rsu.get("cached_adapter_ids", [])}
        )
        handoff_target_cache_ready = bool(
            required_adapter
            and required_adapter in {str(item) for item in handoff_target_rsu.get("cached_adapter_ids", [])}
        )
        workflow = semantic_state.get("workflow", {}) if isinstance(semantic_state.get("workflow", {}), dict) else {}
        nodes = list(workflow.get("nodes", []) or [])
        completed_node_ids = {str(item) for item in workflow.get("completed_node_ids", []) or []}
        remaining_node_count = len(
            [
                node
                for node in nodes
                if str(node.get("node_id", "")) not in completed_node_ids
            ]
        )
        handoff_pressure = _clamp01(
            0.26 * target_differs
            + 0.20 * next_differs
            + 0.18 * non_current_ratio
            + 0.13 * switch_ratio
            + 0.10 * boundary_urgency
            + 0.08 * service_pressure
            + 0.05 * (1.0 - eta_norm)
        )
        event_strength = _clamp01(
            (0.42 * handoff_pressure + 0.34 * timing_support + 0.16 * confidence_support + 0.08 * load_pressure)
            * (0.35 + 0.65 * has_prediction)
        )
        prefetch_useful = bool(
            self._adapter_prefetch_enabled
            and predicted_next_rsu_id
            and str(predicted_next_rsu_id) != str(current_rsu_id)
            and required_adapter
            and not predicted_next_cache_ready
        )
        prefetch_strength = _clamp01(
            (0.42 * next_differs + 0.28 * target_differs + 0.20 * non_current_ratio + 0.10 * confidence_support)
            * (1.0 if prefetch_useful else 0.35)
        )
        short_dag_pressure = 1.0 if (
            remaining_node_count > 0
            and remaining_node_count <= self._digital_twin_policy_prior_pacing_short_dag_threshold
        ) else 0.0
        current_cache_gap = 0.0 if current_cache_ready else 1.0
        pacing_strength = _clamp01(
            0.34 * handoff_pressure
            + 0.24 * boundary_urgency
            + 0.18 * short_dag_pressure
            + 0.14 * current_cache_gap
            + 0.10 * (1.0 - eta_norm)
        )
        continuation_strength = _clamp01(
            0.30 * handoff_pressure
            + 0.22 * timing_support
            + 0.18 * boundary_urgency
            + 0.14 * short_dag_pressure
            + 0.10 * confidence_support
            + 0.06 * (1.0 - eta_norm)
        )
        mechanism_context = window_class == "mechanism_activating"
        handoff_context = bool(target_differs > 0.0 or next_differs > 0.0 or non_current_ratio >= 0.25)
        wait_context = self._digital_twin_wait_readiness_context(
            semantic_state,
            timing_support=timing_support,
            boundary_urgency=boundary_urgency,
            handoff_context=handoff_context,
        )
        pacing_target = bool(
            self._digital_twin_policy_prior_pacing_enabled
            and pacing_strength >= self._digital_twin_policy_prior_pacing_threshold
            and handoff_context
            and (not current_cache_ready or short_dag_pressure > 0.0 or boundary_urgency >= 0.55)
        )
        continuation_target = bool(
            self._digital_twin_policy_prior_env_action_bias_enabled
            and continuation_strength >= self._digital_twin_policy_prior_continuation_threshold
            and remaining_node_count > 0
            and (mechanism_context or handoff_context)
        )
        event_target = int(
            self._event_head_enabled
            and event_strength >= self._digital_twin_policy_prior_prepare_threshold
            and handoff_context
        )
        slow_target = 2 if prefetch_strength >= self._digital_twin_policy_prior_prefetch_threshold and prefetch_useful else 0
        fast_target = 1 if pacing_target else 0
        env_target = -1
        env_target_reason = "none"
        if self._digital_twin_policy_prior_adaptive_wait_enabled:
            continuation_wait_target = bool(continuation_target and wait_context.get("wait_preferred", False))
            continuation_prepare_target = bool(
                continuation_target
                and not continuation_wait_target
                and (
                    not handoff_target_cache_ready
                    or timing_support >= 0.22
                    or boundary_urgency >= 0.38
                    or (
                        mechanism_context
                        and float(wait_context.get("ready_score", 0.0) or 0.0)
                        < self._digital_twin_policy_prior_wait_ready_threshold
                    )
                )
            )
        else:
            continuation_prepare_target = bool(
                continuation_target
                and (
                    mechanism_context
                    or timing_support >= 0.20
                    or boundary_urgency >= 0.35
                    or not handoff_target_cache_ready
                )
            )
            continuation_wait_target = bool(continuation_target and not continuation_prepare_target)
        if continuation_prepare_target:
            env_target = 4
            env_target_reason = "continuation_prepare"
            event_target = 1
            slow_target = 0
            fast_target = 0
            strength = continuation_strength
            if (
                self._digital_twin_policy_prior_adaptive_wait_enabled
                and not handoff_target_cache_ready
            ):
                strength *= self._digital_twin_policy_prior_prepare_not_ready_scale
        elif continuation_wait_target or pacing_target:
            env_target = 2
            env_target_reason = "continuation_wait" if continuation_wait_target else "handoff_pacing_wait"
            event_target = 0
            slow_target = 0
            fast_target = 1
            continuation_wait_strength = (
                continuation_strength * self._digital_twin_policy_prior_wait_cache_ready_scale
                if continuation_wait_target
                else 0.0
            )
            strength = max(
                continuation_wait_strength,
                pacing_strength if pacing_target else 0.0,
            )
        else:
            strength = max(
                event_strength if event_target == 1 else 0.0,
                prefetch_strength if slow_target == 2 else 0.0,
            )
        if strength <= 1e-8:
            return {
                "apply": False,
                "reason": "below_threshold",
                "event_strength": round(float(event_strength), 6),
                "prefetch_strength": round(float(prefetch_strength), 6),
                "pacing_strength": round(float(pacing_strength), 6),
                "continuation_strength": round(float(continuation_strength), 6),
            }
        return {
            "apply": True,
            "reason": (
                "dt_handoff_continuation_prepare_prior"
                if env_target == 4
                else "dt_handoff_continuation_wait_prior"
                if env_target == 2
                else "dt_handoff_pacing_prior"
                if pacing_target
                else "dt_handoff_policy_prior"
            ),
            "strength": round(float(strength), 6),
            "event_strength": round(float(event_strength), 6),
            "prefetch_strength": round(float(prefetch_strength), 6),
            "pacing_strength": round(float(pacing_strength), 6),
            "continuation_strength": round(float(continuation_strength), 6),
            "handoff_pressure": round(float(handoff_pressure), 6),
            "timing_support": round(float(timing_support), 6),
            "confidence_support": round(float(confidence_support), 6),
            "event_target": int(event_target),
            "slow_target": int(slow_target),
            "fast_target": int(fast_target),
            "pacing_target": bool(pacing_target),
            "continuation_target": bool(continuation_target),
            "continuation_prepare_target": bool(continuation_prepare_target),
            "continuation_wait_target": bool(continuation_wait_target),
            "env_target": int(env_target),
            "env_target_reason": env_target_reason,
            "adaptive_wait_enabled": bool(self._digital_twin_policy_prior_adaptive_wait_enabled),
            "adaptive_wait_preferred": bool(wait_context.get("wait_preferred", False)),
            "adaptive_wait_ready_score": float(wait_context.get("ready_score", 0.0) or 0.0),
            "window_class": window_class,
            "remaining_node_count": int(remaining_node_count),
            "short_dag_pressure": round(float(short_dag_pressure), 6),
            "current_cache_ready": bool(current_cache_ready),
            "predicted_next_cache_ready": bool(predicted_next_cache_ready),
            "handoff_target_cache_ready": bool(handoff_target_cache_ready),
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "current_rsu_id": current_rsu_id,
        }

    def _apply_digital_twin_policy_prior(
        self,
        policy_output: dict[str, Any],
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            not self._digital_twin_policy_prior_enabled
            or self._digital_twin_policy_prior_logit_bias <= 0.0
            or not self._use_hierarchy
        ):
            return policy_output
        annotation = self._build_digital_twin_policy_prior_annotation(
            semantic_state,
            run_metadata=run_metadata,
        )
        if not bool(annotation.get("apply", False)):
            return policy_output
        adjusted = dict(policy_output)
        slow_logits = adjusted["slow_logits"].clone()
        fast_logits = adjusted["fast_logits"].clone()
        event_logits = adjusted["event_logits"].clone()
        bias = self._digital_twin_policy_prior_logit_bias * float(annotation.get("strength", 0.0) or 0.0)
        env_action_bias = adjusted.get("env_action_logits_bias")
        if isinstance(env_action_bias, torch.Tensor) and env_action_bias.numel() == 5:
            env_action_logits_bias = env_action_bias.clone()
        else:
            env_action_logits_bias = torch.zeros(5, dtype=event_logits.dtype, device=event_logits.device)
        if int(annotation.get("event_target", 0)) == 1:
            event_bias = self._digital_twin_policy_prior_event_scale * bias
            event_logits[1] = event_logits[1] + event_bias
            event_logits[0] = event_logits[0] - 0.20 * event_bias
        if int(annotation.get("slow_target", 0)) == 2:
            slow_bias = self._digital_twin_policy_prior_slow_scale * bias
            slow_logits[2] = slow_logits[2] + slow_bias
            slow_logits[0] = slow_logits[0] - 0.10 * slow_bias
        if bool(annotation.get("pacing_target", False)):
            pacing_fast_bias = self._digital_twin_policy_prior_pacing_fast_scale * bias
            fast_logits[1] = fast_logits[1] + pacing_fast_bias
            if self._digital_twin_policy_prior_pacing_event_suppression > 0.0:
                event_suppression = self._digital_twin_policy_prior_pacing_event_suppression * bias
                event_logits[1] = event_logits[1] - event_suppression
                event_logits[0] = event_logits[0] + 0.20 * event_suppression
            if self._digital_twin_policy_prior_pacing_slow_suppression > 0.0:
                slow_suppression = self._digital_twin_policy_prior_pacing_slow_suppression * bias
                slow_logits[1] = slow_logits[1] - slow_suppression
                slow_logits[2] = slow_logits[2] - 0.50 * slow_suppression
        if self._digital_twin_policy_prior_fast_scale > 0.0:
            fast_bias = self._digital_twin_policy_prior_fast_scale * bias
            fast_logits[0] = fast_logits[0] + fast_bias
        env_target = int(annotation.get("env_target", -1))
        if (
            self._digital_twin_policy_prior_env_action_bias_enabled
            and self._digital_twin_policy_prior_env_action_logit_bias > 0.0
            and 0 <= env_target < 5
        ):
            env_bias = self._digital_twin_policy_prior_env_action_logit_bias * float(
                annotation.get("strength", 0.0) or 0.0
            )
            if env_target == 4:
                env_bias *= self._digital_twin_policy_prior_continuation_prepare_scale
                event_logits[1] = event_logits[1] + 0.35 * env_bias
                event_logits[0] = event_logits[0] - 0.12 * env_bias
            elif env_target == 2:
                env_bias *= self._digital_twin_policy_prior_continuation_wait_scale
                fast_logits[1] = fast_logits[1] + 0.45 * env_bias
                event_logits[1] = event_logits[1] - 0.25 * env_bias
                event_logits[0] = event_logits[0] + 0.08 * env_bias
            env_action_logits_bias[env_target] = env_action_logits_bias[env_target] + env_bias
            steady_suppression = self._digital_twin_policy_prior_continuation_steady_suppression * env_bias
            if steady_suppression > 0.0:
                env_action_logits_bias[3] = env_action_logits_bias[3] - steady_suppression
                if env_target == 4:
                    env_action_logits_bias[0] = env_action_logits_bias[0] - 0.35 * steady_suppression
                    env_action_logits_bias[1] = env_action_logits_bias[1] - 0.25 * steady_suppression
                elif env_target == 2:
                    env_action_logits_bias[4] = env_action_logits_bias[4] - 0.20 * steady_suppression
        adjusted["slow_logits"] = slow_logits
        adjusted["fast_logits"] = fast_logits
        adjusted["event_logits"] = event_logits
        adjusted["env_action_logits_bias"] = env_action_logits_bias
        adjusted["digital_twin_policy_prior_info"] = {
            **annotation,
            "logit_bias": round(float(bias), 6),
            "env_action_logit_bias": [
                round(float(item), 6)
                for item in env_action_logits_bias.detach().cpu().tolist()
            ],
        }
        return adjusted

    def _compute_digital_twin_policy_prior_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_annotations: list[dict[str, Any]],
        batch_advantage: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self._digital_twin_policy_prior_enabled
            or self._digital_twin_policy_prior_distill_coef <= 0.0
            or not self._use_hierarchy
        ):
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        loss_terms: list[torch.Tensor] = []
        for row_index, (policy_output, annotation) in enumerate(zip(batch_outputs, batch_annotations)):
            if not bool(annotation.get("apply", False)):
                continue
            weight = float(annotation.get("strength", 0.0) or 0.0)
            if batch_advantage.numel() > row_index:
                positive_advantage = max(float(batch_advantage[row_index].detach().item()), 0.0)
                weight *= 1.0 + self._digital_twin_policy_prior_advantage_weight * positive_advantage
            if self._digital_twin_policy_prior_max_weight > 0.0:
                weight = min(weight, self._digital_twin_policy_prior_max_weight)
            if weight <= 1e-8:
                continue
            term = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            if int(annotation.get("event_target", 0)) == 1:
                event_target = torch.tensor([1], dtype=torch.long, device=self._device)
                term = term + self._digital_twin_policy_prior_event_scale * nn.functional.cross_entropy(
                    policy_output["event_logits"].unsqueeze(0),
                    event_target,
                )
            elif bool(annotation.get("pacing_target", False)):
                event_target = torch.tensor([0], dtype=torch.long, device=self._device)
                term = term + self._digital_twin_policy_prior_pacing_event_suppression * nn.functional.cross_entropy(
                    policy_output["event_logits"].unsqueeze(0),
                    event_target,
                )
            slow_target_value = int(annotation.get("slow_target", 0))
            if slow_target_value == 2:
                slow_target = torch.tensor([slow_target_value], dtype=torch.long, device=self._device)
                term = term + self._digital_twin_policy_prior_slow_scale * nn.functional.cross_entropy(
                    policy_output["slow_logits"].unsqueeze(0),
                    slow_target,
                )
            fast_target_value = int(annotation.get("fast_target", 0))
            if bool(annotation.get("pacing_target", False)):
                fast_target = torch.tensor([fast_target_value], dtype=torch.long, device=self._device)
                term = term + self._digital_twin_policy_prior_pacing_fast_scale * nn.functional.cross_entropy(
                    policy_output["fast_logits"].unsqueeze(0),
                    fast_target,
                )
                slow_target = torch.tensor([0], dtype=torch.long, device=self._device)
                term = term + self._digital_twin_policy_prior_pacing_slow_suppression * nn.functional.cross_entropy(
                    policy_output["slow_logits"].unsqueeze(0),
                    slow_target,
                )
            elif self._digital_twin_policy_prior_fast_scale > 0.0:
                fast_target = torch.tensor([fast_target_value], dtype=torch.long, device=self._device)
                term = term + 0.25 * self._digital_twin_policy_prior_fast_scale * nn.functional.cross_entropy(
                    policy_output["fast_logits"].unsqueeze(0),
                    fast_target,
                )
            env_target_value = int(annotation.get("env_target", -1))
            if (
                self._digital_twin_policy_prior_env_action_bias_enabled
                and 0 <= env_target_value < 5
            ):
                env_target = torch.tensor([env_target_value], dtype=torch.long, device=self._device)
                env_scores = self._hierarchical_env_action_scores(policy_output)
                env_scale = (
                    self._digital_twin_policy_prior_continuation_prepare_scale
                    if env_target_value == 4
                    else self._digital_twin_policy_prior_continuation_wait_scale
                    if env_target_value == 2
                    else 1.0
                )
                term = term + env_scale * nn.functional.cross_entropy(
                    env_scores.unsqueeze(0),
                    env_target,
                )
            loss_terms.append(term * weight)
        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

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
            return {
                "applied_count": 0,
                "match_count": 0,
                "match_rate": 0.0,
                "reward_floor": 0.0,
                "weight_mean": 0.0,
                "weight_max": 0.0,
            }
        reward_floor = 0.0
        if self._conservative_imitation_enabled:
            reward_values = np.asarray(
                [float(row.get("reward", 0.0) or 0.0) for row in rollout],
                dtype=np.float32,
            )
            reward_floor = float(
                np.quantile(reward_values, self._conservative_imitation_reward_quantile)
            )
        teacher = PopularityCacheHeuristicAgent()
        weights: list[float] = []
        for row in rollout:
            row["imitation_applied"] = False
            row["imitation_weight"] = 0.0
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
            imitation_weight = self._conservative_imitation_weight(
                row,
                reward_floor=reward_floor,
                student_action=student_action,
                teacher_action=int(teacher_action),
            )
            row["imitation_weight"] = float(imitation_weight)
            weights.append(float(imitation_weight))
            applied_count += 1
            if student_action == int(teacher_action):
                match_count += 1
        match_rate = float(match_count) / float(applied_count) if applied_count else 0.0
        weight_mean = float(sum(weights) / len(weights)) if weights else 0.0
        weight_max = float(max(weights)) if weights else 0.0
        return {
            "applied_count": applied_count,
            "match_count": match_count,
            "match_rate": match_rate,
            "reward_floor": reward_floor,
            "weight_mean": weight_mean,
            "weight_max": weight_max,
        }

    def _conservative_imitation_weight(
        self,
        row: dict[str, Any],
        *,
        reward_floor: float,
        student_action: int,
        teacher_action: int,
    ) -> float:
        if not self._conservative_imitation_enabled:
            return 1.0
        reward = float(row.get("reward", 0.0) or 0.0)
        shortfall = max(float(reward_floor) - reward, 0.0)
        metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
        failed_mechanism = self._row_failed_mechanism_attempt(row, metrics)
        mechanism_success = self._row_mechanism_success(metrics)
        mismatch = int(student_action) != int(teacher_action)
        weight = (
            self._conservative_imitation_min_weight
            + self._conservative_imitation_shortfall_coef * min(shortfall, 1.0)
            + self._conservative_imitation_failure_coef * float(failed_mechanism)
            + self._conservative_imitation_mismatch_coef * float(mismatch)
        )
        if mechanism_success and shortfall <= 1e-8:
            weight *= self._conservative_imitation_success_decay
        return float(
            max(
                self._conservative_imitation_min_weight,
                min(self._conservative_imitation_max_weight, weight),
            )
        )

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
        for policy_output, row in zip(batch_outputs, batch_rows):
            if not bool(row.get("imitation_applied", False)):
                continue
            imitation_weight = torch.tensor(
                float(row.get("imitation_weight", 1.0) or 1.0),
                dtype=torch.float32,
                device=self._device,
            )
            teacher_action = int(row.get("teacher_action", 3))
            if not self._use_hierarchy:
                target = torch.tensor([teacher_action], dtype=torch.long, device=self._device)
                loss_terms.append(
                    imitation_weight
                    * nn.functional.cross_entropy(policy_output["flat_logits"].unsqueeze(0), target)
                )
                continue
            head_targets = dict(row.get("imitation_head_targets", self._head_targets_for_env_action(teacher_action)))
            if teacher_action == 4:
                target = torch.tensor([int(head_targets.get("event", 1))], dtype=torch.long, device=self._device)
                loss_terms.append(
                    imitation_weight
                    * nn.functional.cross_entropy(policy_output["event_logits"].unsqueeze(0), target)
                )
            elif teacher_action in {0, 1}:
                target = torch.tensor([int(head_targets.get("slow", 0))], dtype=torch.long, device=self._device)
                loss_terms.append(
                    imitation_weight
                    * nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), target)
                )
            elif teacher_action == 2:
                target = torch.tensor([int(head_targets.get("fast", 1))], dtype=torch.long, device=self._device)
                loss_terms.append(
                    imitation_weight
                    * nn.functional.cross_entropy(policy_output["fast_logits"].unsqueeze(0), target)
                )
            else:
                event_target = torch.tensor([0], dtype=torch.long, device=self._device)
                slow_target = torch.tensor([0], dtype=torch.long, device=self._device)
                fast_target = torch.tensor([0], dtype=torch.long, device=self._device)
                loss_terms.append(
                    imitation_weight
                    * (
                        0.5 * nn.functional.cross_entropy(policy_output["event_logits"].unsqueeze(0), event_target)
                        + 0.25 * nn.functional.cross_entropy(policy_output["slow_logits"].unsqueeze(0), slow_target)
                        + 0.25 * nn.functional.cross_entropy(policy_output["fast_logits"].unsqueeze(0), fast_target)
                    )
                )
        if not loss_terms:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        return torch.stack(loss_terms).mean()

    def _annotate_advantage_weighted_behavior_targets(
        self,
        rollout: list[dict[str, Any]],
        *,
        advantage_values: np.ndarray,
    ) -> dict[str, float | int]:
        for row in rollout:
            row["advantage_weighted_behavior_applied"] = False
            row["advantage_weighted_behavior_weight"] = 0.0
            row["advantage_weighted_behavior_mode"] = "none"
            row["advantage_weighted_behavior_target_action"] = int(row.get("action", 3) or 3)

        if (
            not self._advantage_weighted_behavior_regularization_enabled
            or self._advantage_weighted_behavior_coef <= 0.0
            or not rollout
        ):
            return {
                "applied_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "teacher_match_count": 0,
                "teacher_match_rate": 0.0,
                "weight_mean": 0.0,
                "weight_max": 0.0,
            }

        teacher = PopularityCacheHeuristicAgent()
        applied_count = 0
        positive_count = 0
        negative_count = 0
        teacher_match_count = 0
        teacher_seen_count = 0
        weights: list[float] = []
        for row_index, row in enumerate(rollout):
            decision_info = dict(row.get("decision_info", {}))
            try:
                semantic_state = self._extract_semantic_state(decision_info)
            except ValueError:
                continue
            action_mask = self._extract_action_mask(decision_info)
            teacher_action, teacher_info = teacher.act(
                None,
                {
                    "semantic_state": semantic_state,
                    "action_mask": action_mask,
                },
            )
            action_info = dict(row.get("action_info", {}))
            student_action = int(action_info.get("final_env_action", row.get("action", 3)) or 3)
            teacher_action = int(teacher_action)
            if not self._is_env_action_valid(teacher_action, action_mask):
                continue
            teacher_seen_count += 1
            if student_action == teacher_action:
                teacher_match_count += 1
                continue
            advantage_value = float(advantage_values[row_index]) if row_index < len(advantage_values) else 0.0
            target_action: int | None = None
            mode = "none"
            coefficient = 0.0
            magnitude = 0.0
            if (
                advantage_value > self._advantage_weighted_behavior_positive_gate
                and self._advantage_weighted_behavior_positive_coef > 0.0
            ):
                target_action = student_action
                mode = "positive_deviation"
                coefficient = self._advantage_weighted_behavior_positive_coef
                magnitude = advantage_value - self._advantage_weighted_behavior_positive_gate
            elif (
                advantage_value < -self._advantage_weighted_behavior_negative_gate
                and self._advantage_weighted_behavior_negative_coef > 0.0
            ):
                target_action = teacher_action
                mode = "negative_recovery"
                coefficient = self._advantage_weighted_behavior_negative_coef
                magnitude = -advantage_value - self._advantage_weighted_behavior_negative_gate
            if target_action is None or not self._is_env_action_valid(target_action, action_mask):
                continue

            weight = coefficient * math.exp(
                min(
                    magnitude / self._advantage_weighted_behavior_temperature,
                    math.log(max(self._advantage_weighted_behavior_max_weight, 1.0)),
                )
            )
            if self._advantage_weighted_behavior_max_weight > 0.0:
                weight = min(weight, self._advantage_weighted_behavior_max_weight)
            metrics = dict(row.get("env_info", {}).get("metrics_protocol", {}))
            mechanism_context = bool(
                self._row_window_class(row) == "mechanism_activating"
                or student_action in {1, 4}
                or teacher_action in {1, 4}
                or self._handoff_risk_context(row, metrics)
            )
            if mechanism_context:
                weight *= self._advantage_weighted_behavior_mechanism_scale
            if weight <= 1e-8:
                continue

            row["advantage_weighted_behavior_applied"] = True
            row["advantage_weighted_behavior_weight"] = float(weight)
            row["advantage_weighted_behavior_mode"] = mode
            row["advantage_weighted_behavior_target_action"] = int(target_action)
            row["advantage_weighted_behavior_teacher_action"] = teacher_action
            row["advantage_weighted_behavior_student_action"] = int(student_action)
            row["advantage_weighted_behavior_advantage"] = float(advantage_value)
            row["advantage_weighted_behavior_teacher_reason"] = str(
                teacher_info.get("heuristic_reason", "unknown")
            )
            weights.append(float(weight))
            applied_count += 1
            if mode == "positive_deviation":
                positive_count += 1
            elif mode == "negative_recovery":
                negative_count += 1

        return {
            "applied_count": applied_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "teacher_match_count": teacher_match_count,
            "teacher_match_rate": (
                float(teacher_match_count) / float(teacher_seen_count)
                if teacher_seen_count
                else 0.0
            ),
            "weight_mean": float(sum(weights) / len(weights)) if weights else 0.0,
            "weight_max": float(max(weights)) if weights else 0.0,
        }

    def _compute_advantage_weighted_behavior_loss(
        self,
        *,
        batch_outputs: list[dict[str, Any]],
        batch_action_masks: list[list[bool] | None],
        batch_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        if (
            not self._advantage_weighted_behavior_regularization_enabled
            or self._advantage_weighted_behavior_coef <= 0.0
            or not batch_outputs
        ):
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        loss_terms: list[torch.Tensor] = []
        for policy_output, action_mask, row in zip(batch_outputs, batch_action_masks, batch_rows):
            if not bool(row.get("advantage_weighted_behavior_applied", False)):
                continue
            target_action = int(row.get("advantage_weighted_behavior_target_action", row.get("action", 3)) or 3)
            if not self._is_env_action_valid(target_action, action_mask):
                continue
            weight = float(row.get("advantage_weighted_behavior_weight", 0.0) or 0.0)
            if weight <= 1e-8:
                continue
            if not self._use_hierarchy:
                logits = self._masked_flat_logits(policy_output["flat_logits"], action_mask)
            else:
                logits = self._masked_flat_logits(
                    self._hierarchical_env_action_scores(policy_output),
                    action_mask,
                )
            target = torch.tensor([target_action], dtype=torch.long, device=self._device)
            loss_terms.append(
                nn.functional.cross_entropy(logits.unsqueeze(0), target) * float(weight)
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
            "conservative_imitation_enabled": self._conservative_imitation_enabled,
            "conservative_imitation_reward_quantile": self._conservative_imitation_reward_quantile,
            "conservative_imitation_min_weight": self._conservative_imitation_min_weight,
            "conservative_imitation_max_weight": self._conservative_imitation_max_weight,
            "conservative_imitation_shortfall_coef": self._conservative_imitation_shortfall_coef,
            "conservative_imitation_failure_coef": self._conservative_imitation_failure_coef,
            "conservative_imitation_mismatch_coef": self._conservative_imitation_mismatch_coef,
            "conservative_imitation_success_decay": self._conservative_imitation_success_decay,
            "mechanism_aux_coef": self._mechanism_aux_coef,
            "mechanism_window_weight": self._mechanism_window_weight,
            "prepare_action_prior_weight": self._prepare_action_prior_weight,
            "mechanism_entropy_coef": self._mechanism_entropy_coef,
            "mechanism_retention_start_update": self._mechanism_retention_start_update,
            "mechanism_aux_coef_floor_after_update": self._mechanism_aux_coef_floor_after_update,
            "mechanism_window_weight_floor_after_update": self._mechanism_window_weight_floor_after_update,
            "mechanism_entropy_floor_after_update": self._mechanism_entropy_floor_after_update,
            "mechanism_aux_current_cache_fill_enabled": self._mechanism_aux_current_cache_fill_enabled,
            "mechanism_credit_prd_enabled": self._mechanism_credit_prd_enabled,
            "mechanism_credit_policy_coef": self._mechanism_credit_policy_coef,
            "mechanism_credit_event_coef": self._mechanism_credit_event_coef,
            "mechanism_credit_option_coef": self._mechanism_credit_option_coef,
            "mechanism_credit_clip": self._mechanism_credit_clip,
            "mechanism_credit_success_bonus": self._mechanism_credit_success_bonus,
            "mechanism_credit_prepare_bonus": self._mechanism_credit_prepare_bonus,
            "mechanism_credit_ready_bonus": self._mechanism_credit_ready_bonus,
            "mechanism_credit_prefetch_hit_bonus": self._mechanism_credit_prefetch_hit_bonus,
            "mechanism_credit_miss_penalty": self._mechanism_credit_miss_penalty,
            "mechanism_credit_false_positive_penalty": self._mechanism_credit_false_positive_penalty,
            "mechanism_credit_min_context": self._mechanism_credit_min_context,
            "mechanism_focal_aux_enabled": self._mechanism_focal_aux_enabled,
            "mechanism_focal_gamma": self._mechanism_focal_gamma,
            "digital_twin_handoff_fusion_enabled": self._digital_twin_handoff_fusion_enabled,
            "digital_twin_handoff_slow_scale": self._digital_twin_handoff_slow_scale,
            "digital_twin_handoff_fast_scale": self._digital_twin_handoff_fast_scale,
            "digital_twin_handoff_event_scale": self._digital_twin_handoff_event_scale,
            "digital_twin_handoff_critic_scale": self._digital_twin_handoff_critic_scale,
            "digital_twin_planning_residual_enabled": (
                self._digital_twin_planning_residual_enabled
            ),
            "digital_twin_planning_residual_scale": (
                self._digital_twin_planning_residual_scale
            ),
            "outcome_memory_fusion_enabled": self._outcome_memory_fusion_enabled,
            "outcome_memory_actor_scale": self._outcome_memory_actor_scale,
            "outcome_memory_critic_scale": self._outcome_memory_critic_scale,
            "outcome_recovery_residual_enabled": (
                self._outcome_recovery_residual_enabled
            ),
            "outcome_recovery_residual_scale": (
                self._outcome_recovery_residual_scale
            ),
            "outcome_context_residual_enabled": (
                self._outcome_context_residual_enabled
            ),
            "digital_twin_policy_prior_enabled": self._digital_twin_policy_prior_enabled,
            "digital_twin_policy_prior_logit_bias": self._digital_twin_policy_prior_logit_bias,
            "digital_twin_policy_prior_event_scale": self._digital_twin_policy_prior_event_scale,
            "digital_twin_policy_prior_slow_scale": self._digital_twin_policy_prior_slow_scale,
            "digital_twin_policy_prior_fast_scale": self._digital_twin_policy_prior_fast_scale,
            "digital_twin_policy_prior_prepare_threshold": self._digital_twin_policy_prior_prepare_threshold,
            "digital_twin_policy_prior_prefetch_threshold": self._digital_twin_policy_prior_prefetch_threshold,
            "digital_twin_policy_prior_confidence_floor": self._digital_twin_policy_prior_confidence_floor,
            "digital_twin_policy_prior_distill_coef": self._digital_twin_policy_prior_distill_coef,
            "digital_twin_policy_prior_distill_warmup_updates": (
                self._digital_twin_policy_prior_distill_warmup_updates
            ),
            "digital_twin_policy_prior_distill_decay": self._digital_twin_policy_prior_distill_decay,
            "digital_twin_policy_prior_advantage_weight": self._digital_twin_policy_prior_advantage_weight,
            "digital_twin_policy_prior_max_weight": self._digital_twin_policy_prior_max_weight,
            "digital_twin_policy_prior_pacing_enabled": self._digital_twin_policy_prior_pacing_enabled,
            "digital_twin_policy_prior_pacing_threshold": self._digital_twin_policy_prior_pacing_threshold,
            "digital_twin_policy_prior_pacing_fast_scale": self._digital_twin_policy_prior_pacing_fast_scale,
            "digital_twin_policy_prior_pacing_event_suppression": (
                self._digital_twin_policy_prior_pacing_event_suppression
            ),
            "digital_twin_policy_prior_pacing_slow_suppression": (
                self._digital_twin_policy_prior_pacing_slow_suppression
            ),
            "digital_twin_policy_prior_pacing_short_dag_threshold": (
                self._digital_twin_policy_prior_pacing_short_dag_threshold
            ),
            "digital_twin_policy_prior_env_action_bias_enabled": (
                self._digital_twin_policy_prior_env_action_bias_enabled
            ),
            "digital_twin_policy_prior_env_action_logit_bias": (
                self._digital_twin_policy_prior_env_action_logit_bias
            ),
            "digital_twin_policy_prior_continuation_threshold": (
                self._digital_twin_policy_prior_continuation_threshold
            ),
            "digital_twin_policy_prior_continuation_prepare_scale": (
                self._digital_twin_policy_prior_continuation_prepare_scale
            ),
            "digital_twin_policy_prior_continuation_wait_scale": (
                self._digital_twin_policy_prior_continuation_wait_scale
            ),
            "digital_twin_policy_prior_continuation_steady_suppression": (
                self._digital_twin_policy_prior_continuation_steady_suppression
            ),
            "digital_twin_policy_prior_adaptive_wait_enabled": (
                self._digital_twin_policy_prior_adaptive_wait_enabled
            ),
            "digital_twin_policy_prior_wait_ready_threshold": (
                self._digital_twin_policy_prior_wait_ready_threshold
            ),
            "digital_twin_policy_prior_wait_timing_ceiling": (
                self._digital_twin_policy_prior_wait_timing_ceiling
            ),
            "digital_twin_policy_prior_wait_cache_ready_scale": (
                self._digital_twin_policy_prior_wait_cache_ready_scale
            ),
            "digital_twin_policy_prior_prepare_not_ready_scale": (
                self._digital_twin_policy_prior_prepare_not_ready_scale
            ),
            "env_action_ppo_enabled": self._env_action_ppo_enabled,
            "env_action_ppo_coef": self._env_action_ppo_coef,
            "env_action_ppo_advantage_blend": self._env_action_ppo_advantage_blend,
            "env_action_ppo_teacher_coef": self._env_action_ppo_teacher_coef,
            "env_action_ppo_mechanism_focus": self._env_action_ppo_mechanism_focus,
            "env_action_sparse_recovery_focus": self._env_action_sparse_recovery_focus,
            "env_action_risk_adjusted_recovery_coef": self._env_action_risk_adjusted_recovery_coef,
            "env_action_risk_adjusted_recovery_floor": self._env_action_risk_adjusted_recovery_floor,
            "env_action_adapter_miss_counterfactual_coef": (
                self._env_action_adapter_miss_counterfactual_coef
            ),
            "cache_feasibility_prior_enabled": self._cache_feasibility_prior_enabled,
            "cache_feasibility_cache_fill_bias": self._cache_feasibility_cache_fill_bias,
            "cache_feasibility_steady_penalty": self._cache_feasibility_steady_penalty,
            "cache_feasibility_prepare_penalty": self._cache_feasibility_prepare_penalty,
            "cache_feasibility_prefetch_penalty": self._cache_feasibility_prefetch_penalty,
            "cache_feasibility_current_miss_prepare_penalty": (
                self._cache_feasibility_current_miss_prepare_penalty
            ),
            "cache_feasibility_current_miss_prefetch_penalty": (
                self._cache_feasibility_current_miss_prefetch_penalty
            ),
            "cache_feasibility_min_context": self._cache_feasibility_min_context,
            "handoff_alignment_barrier_enabled": self._handoff_alignment_barrier_enabled,
            "handoff_alignment_barrier_prepare_penalty": (
                self._handoff_alignment_barrier_prepare_penalty
            ),
            "handoff_alignment_barrier_prefetch_penalty": (
                self._handoff_alignment_barrier_prefetch_penalty
            ),
            "handoff_alignment_barrier_current_fill_bias": (
                self._handoff_alignment_barrier_current_fill_bias
            ),
            "handoff_alignment_barrier_target_mismatch_penalty": (
                self._handoff_alignment_barrier_target_mismatch_penalty
            ),
            "handoff_alignment_barrier_late_eta_penalty": (
                self._handoff_alignment_barrier_late_eta_penalty
            ),
            "handoff_alignment_barrier_min_context": self._handoff_alignment_barrier_min_context,
            "sparse_handoff_recovery_prior_enabled": (
                self._sparse_handoff_recovery_prior_enabled
            ),
            "sparse_handoff_recovery_prefetch_bias": self._sparse_handoff_recovery_prefetch_bias,
            "sparse_handoff_recovery_prepare_bias": self._sparse_handoff_recovery_prepare_bias,
            "sparse_handoff_recovery_current_fill_bias": (
                self._sparse_handoff_recovery_current_fill_bias
            ),
            "sparse_handoff_recovery_steady_bias": self._sparse_handoff_recovery_steady_bias,
            "sparse_handoff_recovery_local_penalty": self._sparse_handoff_recovery_local_penalty,
            "sparse_handoff_recovery_min_context": self._sparse_handoff_recovery_min_context,
            "sparse_handoff_recovery_max_eta": self._sparse_handoff_recovery_max_eta,
            "sparse_handoff_realization_credit_enabled": (
                self._sparse_handoff_realization_credit_enabled
            ),
            "sparse_handoff_realization_success_bonus": (
                self._sparse_handoff_realization_success_bonus
            ),
            "sparse_handoff_realization_ready_bonus": (
                self._sparse_handoff_realization_ready_bonus
            ),
            "sparse_handoff_realization_prefetch_bonus": (
                self._sparse_handoff_realization_prefetch_bonus
            ),
            "sparse_handoff_realization_failed_prepare_penalty": (
                self._sparse_handoff_realization_failed_prepare_penalty
            ),
            "sparse_handoff_realization_local_penalty": (
                self._sparse_handoff_realization_local_penalty
            ),
            "sparse_handoff_realization_min_context": (
                self._sparse_handoff_realization_min_context
            ),
            "sparse_handoff_option_prior_enabled": (
                self._sparse_handoff_option_prior_enabled
            ),
            "sparse_handoff_option_prepare_bias": self._sparse_handoff_option_prepare_bias,
            "sparse_handoff_option_popularity_penalty": (
                self._sparse_handoff_option_popularity_penalty
            ),
            "sparse_handoff_option_local_penalty": self._sparse_handoff_option_local_penalty,
            "sparse_handoff_option_min_context": self._sparse_handoff_option_min_context,
            "sparse_handoff_option_max_eta": self._sparse_handoff_option_max_eta,
            "env_action_ppo_max_weight": self._env_action_ppo_max_weight,
            "env_action_ppo_ratio_barrier_coef": self._env_action_ppo_ratio_barrier_coef,
            "env_action_ppo_ratio_barrier_margin": self._env_action_ppo_ratio_barrier_margin,
            "env_action_counterfactual_margin_enabled": self._env_action_counterfactual_margin_enabled,
            "env_action_counterfactual_margin_coef": self._env_action_counterfactual_margin_coef,
            "env_action_counterfactual_margin_min_gap": self._env_action_counterfactual_margin_min_gap,
            "env_action_counterfactual_margin_max_weight": self._env_action_counterfactual_margin_max_weight,
            "env_action_counterfactual_margin_advantage_gate": (
                self._env_action_counterfactual_margin_advantage_gate
            ),
            "env_action_counterfactual_margin_advantage_blend": (
                self._env_action_counterfactual_margin_advantage_blend
            ),
            "argmax_margin_regularization_enabled": self._argmax_margin_regularization_enabled,
            "argmax_margin_coef": self._argmax_margin_coef,
            "argmax_margin_min_gap": self._argmax_margin_min_gap,
            "argmax_margin_max_weight": self._argmax_margin_max_weight,
            "argmax_margin_tail_risk_threshold": self._argmax_margin_tail_risk_threshold,
            "argmax_margin_mechanism_penalty_scale": self._argmax_margin_mechanism_penalty_scale,
            "event_prd_advantage_enabled": self._event_prd_advantage_enabled,
            "event_prd_advantage_coef": self._event_prd_advantage_coef,
            "event_prd_advantage_clip": self._event_prd_advantage_clip,
            "delayed_mechanism_credit_enabled": self._delayed_mechanism_credit_enabled,
            "delayed_mechanism_credit_policy_coef": self._delayed_mechanism_credit_policy_coef,
            "delayed_mechanism_credit_event_coef": self._delayed_mechanism_credit_event_coef,
            "delayed_mechanism_credit_horizon": self._delayed_mechanism_credit_horizon,
            "delayed_mechanism_credit_decay": self._delayed_mechanism_credit_decay,
            "delayed_mechanism_credit_clip": self._delayed_mechanism_credit_clip,
            "delayed_mechanism_credit_ready_bonus": self._delayed_mechanism_credit_ready_bonus,
            "delayed_mechanism_credit_success_bonus": self._delayed_mechanism_credit_success_bonus,
            "delayed_mechanism_credit_failure_penalty": self._delayed_mechanism_credit_failure_penalty,
            "delayed_mechanism_credit_missed_prepare_scale": (
                self._delayed_mechanism_credit_missed_prepare_scale
            ),
            "delayed_mechanism_credit_stale_penalty": self._delayed_mechanism_credit_stale_penalty,
            "delayed_mechanism_credit_context_gate": self._delayed_mechanism_credit_context_gate,
            "delayed_mechanism_credit_strict_opportunity_enabled": (
                self._delayed_mechanism_credit_strict_opportunity_enabled
            ),
            "opportunity_constrained_policy_enabled": self._opportunity_constrained_policy_enabled,
            "opportunity_constrained_min_context": self._opportunity_constrained_min_context,
            "opportunity_constrained_low_context": self._opportunity_constrained_low_context,
            "opportunity_constrained_prepare_penalty": self._opportunity_constrained_prepare_penalty,
            "opportunity_constrained_prefetch_penalty": self._opportunity_constrained_prefetch_penalty,
            "opportunity_constrained_prepare_bias": self._opportunity_constrained_prepare_bias,
            "opportunity_constrained_prefetch_bias": self._opportunity_constrained_prefetch_bias,
            "opportunity_constrained_current_bias": self._opportunity_constrained_current_bias,
            "opportunity_constrained_local_bias": self._opportunity_constrained_local_bias,
            "opportunity_constrained_no_rsu_service_bias": (
                self._opportunity_constrained_no_rsu_service_bias
            ),
            "opportunity_constrained_no_rsu_local_penalty": (
                self._opportunity_constrained_no_rsu_local_penalty
            ),
            "opportunity_constrained_no_rsu_prepare_bias": (
                self._opportunity_constrained_no_rsu_prepare_bias
            ),
            "opportunity_constrained_no_rsu_prepare_min_context": (
                self._opportunity_constrained_no_rsu_prepare_min_context
            ),
            "opportunity_constrained_confidence_floor": self._opportunity_constrained_confidence_floor,
            "opportunity_constrained_uncertainty_ceiling": self._opportunity_constrained_uncertainty_ceiling,
            "opportunity_constrained_reliability_floor": self._opportunity_constrained_reliability_floor,
            "net_advantage_prepare_gate_enabled": self._net_advantage_prepare_gate_enabled,
            "net_advantage_prepare_gate_bias": self._net_advantage_prepare_gate_bias,
            "net_advantage_prepare_gate_min_score": self._net_advantage_prepare_gate_min_score,
            "net_advantage_prepare_gate_margin": self._net_advantage_prepare_gate_margin,
            "net_advantage_prepare_gate_prefetch_scale": self._net_advantage_prepare_gate_prefetch_scale,
            "net_advantage_prepare_gate_current_scale": self._net_advantage_prepare_gate_current_scale,
            "net_advantage_prepare_gate_service_fill_scale": (
                self._net_advantage_prepare_gate_service_fill_scale
            ),
            "net_advantage_prepare_gate_local_penalty_scale": (
                self._net_advantage_prepare_gate_local_penalty_scale
            ),
            "net_advantage_prepare_gate_cost_scale": self._net_advantage_prepare_gate_cost_scale,
            "net_advantage_prepare_gate_policy_coef": self._net_advantage_prepare_gate_policy_coef,
            "net_advantage_prepare_gate_event_coef": self._net_advantage_prepare_gate_event_coef,
            "net_advantage_prepare_gate_clip": self._net_advantage_prepare_gate_clip,
            "coverage_recovery_gate_bias_scale": self._coverage_recovery_gate_bias_scale,
            "coverage_recovery_gate_min_scale": self._coverage_recovery_gate_min_scale,
            "coverage_recovery_gate_fallback_suppression_scale": (
                self._coverage_recovery_gate_fallback_suppression_scale
            ),
            "coverage_recovery_gate_fast_suppression_scale": (
                self._coverage_recovery_gate_fast_suppression_scale
            ),
            "coverage_recovery_gate_current_suppression_scale": (
                self._coverage_recovery_gate_current_suppression_scale
            ),
            "coverage_recovery_gate_prepare_credit": self._coverage_recovery_gate_prepare_credit,
            "coverage_recovery_gate_fallback_penalty": self._coverage_recovery_gate_fallback_penalty,
            "coverage_recovery_guard_enabled": self._coverage_recovery_guard_enabled,
            "coverage_recovery_final_guard_enabled": self._coverage_recovery_final_guard_enabled,
            "coverage_recovery_final_guard_min_scale": self._coverage_recovery_final_guard_min_scale,
            "coverage_recovery_final_guard_min_confidence": (
                self._coverage_recovery_final_guard_min_confidence
            ),
            "coverage_recovery_target_memory_option_credit": (
                self._coverage_recovery_target_memory_option_credit
            ),
            "coverage_recovery_target_memory_option_penalty": (
                self._coverage_recovery_target_memory_option_penalty
            ),
            "service_completion_gate_enabled": self._service_completion_gate_enabled,
            "service_completion_gate_bias": self._service_completion_gate_bias,
            "service_completion_gate_remaining_nodes_threshold": (
                self._service_completion_gate_remaining_nodes_threshold
            ),
            "service_completion_gate_event_suppression_scale": (
                self._service_completion_gate_event_suppression_scale
            ),
            "service_completion_gate_prefetch_suppression_scale": (
                self._service_completion_gate_prefetch_suppression_scale
            ),
            "service_completion_gate_fallback_suppression_scale": (
                self._service_completion_gate_fallback_suppression_scale
            ),
            "service_completion_gate_policy_coef": self._service_completion_gate_policy_coef,
            "service_completion_gate_event_coef": self._service_completion_gate_event_coef,
            "service_completion_gate_clip": self._service_completion_gate_clip,
            "retrospective_handoff_aux_enabled": self._retrospective_handoff_aux_enabled,
            "retrospective_handoff_aux_max_eta": self._retrospective_handoff_aux_max_eta,
            "retrospective_handoff_aux_min_score": self._retrospective_handoff_aux_min_score,
            "retrospective_handoff_aux_prepare_weight": self._retrospective_handoff_aux_prepare_weight,
            "retrospective_handoff_aux_transition_weight": self._retrospective_handoff_aux_transition_weight,
            "backhaul_aware_policy_enabled": self._backhaul_aware_policy_enabled,
            "backhaul_aware_service_fill_bias": self._backhaul_aware_service_fill_bias,
            "backhaul_aware_redundant_fill_penalty": self._backhaul_aware_redundant_fill_penalty,
            "backhaul_aware_no_signal_prefetch_penalty": self._backhaul_aware_no_signal_prefetch_penalty,
            "backhaul_aware_no_signal_prepare_penalty": self._backhaul_aware_no_signal_prepare_penalty,
            "backhaul_aware_steady_bias": self._backhaul_aware_steady_bias,
            "backhaul_aware_service_pressure_floor": self._backhaul_aware_service_pressure_floor,
            "advantage_weighted_behavior_regularization_enabled": (
                self._advantage_weighted_behavior_regularization_enabled
            ),
            "advantage_weighted_behavior_coef": self._advantage_weighted_behavior_coef,
            "advantage_weighted_behavior_positive_coef": self._advantage_weighted_behavior_positive_coef,
            "advantage_weighted_behavior_negative_coef": self._advantage_weighted_behavior_negative_coef,
            "advantage_weighted_behavior_temperature": self._advantage_weighted_behavior_temperature,
            "advantage_weighted_behavior_max_weight": self._advantage_weighted_behavior_max_weight,
            "advantage_weighted_behavior_positive_gate": self._advantage_weighted_behavior_positive_gate,
            "advantage_weighted_behavior_negative_gate": self._advantage_weighted_behavior_negative_gate,
            "advantage_weighted_behavior_mechanism_scale": self._advantage_weighted_behavior_mechanism_scale,
            "latency_fallback_bias_enabled": self._latency_fallback_bias_enabled,
            "latency_fallback_bias_strength": self._latency_fallback_bias_strength,
            "latency_fallback_confidence_floor": self._latency_fallback_confidence_floor,
            "latency_fallback_slow_suppression_strength": self._latency_fallback_slow_suppression_strength,
            "steady_rsu_bias_enabled": self._steady_rsu_bias_enabled,
            "steady_rsu_bias_strength": self._steady_rsu_bias_strength,
            "steady_rsu_confidence_floor": self._steady_rsu_confidence_floor,
            "backhaul_guard_enabled": self._backhaul_guard_enabled,
            "backhaul_guard_max_reactive_fills_per_adapter": self._backhaul_guard_max_reactive_fills_per_adapter,
            "cache_warm_start_guard_enabled": self._cache_warm_start_guard_enabled,
            "cache_warm_start_guard_min_countdown": self._cache_warm_start_guard_min_countdown,
            "cache_warm_start_guard_max_prefetch_countdown": self._cache_warm_start_guard_max_prefetch_countdown,
            "predictive_prefetch_admission_guard_enabled": self._predictive_prefetch_admission_guard_enabled,
            "predictive_prefetch_admission_min_confidence": self._predictive_prefetch_admission_min_confidence,
            "predictive_prefetch_admission_require_distinct_next": self._predictive_prefetch_admission_require_distinct_next,
            "idle_popularity_fallback_enabled": self._idle_popularity_fallback_enabled,
            "idle_popularity_fallback_only_vehicle_fallback": self._idle_popularity_fallback_only_vehicle_fallback,
            "idle_popularity_prefetch_threshold": self._idle_popularity_prefetch_threshold,
            "idle_popularity_no_rsu_local_fallback_enabled": self._idle_popularity_no_rsu_local_fallback_enabled,
            "idle_popularity_no_rsu_local_requires_low_context": self._idle_popularity_no_rsu_local_requires_low_context,
            "idle_popularity_no_rsu_service_continuity_enabled": (
                self._idle_popularity_no_rsu_service_continuity_enabled
            ),
            "idle_popularity_no_rsu_any_action_override_enabled": (
                self._idle_popularity_no_rsu_any_action_override_enabled
            ),
            "option_gate_enabled": self._option_gate_enabled,
            "option_gate_count": self._option_gate_count,
            "option_gate_loss_coef": self._option_gate_loss_coef,
            "option_gate_entropy_coef": self._option_gate_entropy_coef,
            "option_gate_prior_coef": self._option_gate_prior_coef,
            "option_gate_prior_warmup_updates": self._option_gate_prior_warmup_updates,
            "option_gate_prior_decay": self._option_gate_prior_decay,
            "option_gate_prior_logit_bias": self._option_gate_prior_logit_bias,
            "option_gate_log_prob_weight": self._option_gate_log_prob_weight,
            "option_gate_context_prior_enabled": self._option_gate_context_prior_enabled,
            "option_gate_deterministic_prior_margin": self._option_gate_deterministic_prior_margin,
            "option_gate_idle_prior_enabled": self._option_gate_idle_prior_enabled,
            "option_gate_idle_recovery_mechanism_prior_enabled": (
                self._option_gate_idle_recovery_mechanism_prior_enabled
            ),
            "option_gate_idle_recovery_min_context": self._option_gate_idle_recovery_min_context,
            "option_gate_mechanism_preserve_enabled": self._option_gate_mechanism_preserve_enabled,
            "option_gate_prd_enabled": self._option_gate_prd_enabled,
            "option_gate_prd_coef": self._option_gate_prd_coef,
            "option_gate_prd_clip": self._option_gate_prd_clip,
            "option_gate_counterfactual_prd_enabled": self._option_gate_counterfactual_prd_enabled,
            "option_gate_counterfactual_coef": self._option_gate_counterfactual_coef,
            "option_gate_counterfactual_clip": self._option_gate_counterfactual_clip,
            "option_counterfactual_critic_enabled": self._option_counterfactual_critic_enabled,
            "option_counterfactual_value_coef": self._option_counterfactual_value_coef,
            "option_counterfactual_advantage_coef": self._option_counterfactual_advantage_coef,
            "option_counterfactual_advantage_clip": self._option_counterfactual_advantage_clip,
            "option_counterfactual_warmup_updates": self._option_counterfactual_warmup_updates,
            "option_counterfactual_tail_weight": self._option_counterfactual_tail_weight,
            "option_counterfactual_policy_improvement_enabled": (
                self._option_counterfactual_policy_improvement_enabled
            ),
            "option_counterfactual_policy_improvement_coef": (
                self._option_counterfactual_policy_improvement_coef
            ),
            "option_counterfactual_policy_improvement_clip": (
                self._option_counterfactual_policy_improvement_clip
            ),
            "option_counterfactual_policy_improvement_deterministic_only": (
                self._option_counterfactual_policy_improvement_deterministic_only
            ),
            "option_counterfactual_model_rollout_enabled": (
                self._option_counterfactual_model_rollout_enabled
            ),
            "option_counterfactual_model_rollout_horizon": (
                self._option_counterfactual_model_rollout_horizon
            ),
            "env_action_model_critic_enabled": self._env_action_model_critic_enabled,
            "env_action_model_critic_value_coef": self._env_action_model_critic_value_coef,
            "env_action_model_critic_advantage_coef": (
                self._env_action_model_critic_advantage_coef
            ),
            "env_action_model_critic_policy_improvement_coef": (
                self._env_action_model_critic_policy_improvement_coef
            ),
            "env_action_model_critic_advantage_clip": (
                self._env_action_model_critic_advantage_clip
            ),
            "env_action_model_critic_warmup_updates": (
                self._env_action_model_critic_warmup_updates
            ),
            "env_action_model_rollout_enabled": self._env_action_model_rollout_enabled,
            "env_action_model_rollout_horizon": self._env_action_model_rollout_horizon,
            "env_action_model_rollout_horizons": list(
                self._env_action_model_rollout_horizons
            ),
            "env_action_model_imagination_replay_enabled": (
                self._env_action_model_imagination_replay_enabled
            ),
            "env_action_model_imagination_replay_depths": list(
                self._env_action_model_imagination_replay_depths
            ),
            "env_action_model_imagination_replay_horizons": list(
                self._env_action_model_imagination_replay_horizons
            ),
            "env_action_model_imagination_replay_recovery_only": (
                self._env_action_model_imagination_replay_recovery_only
            ),
            "env_action_model_imagination_beam_search_enabled": (
                self._env_action_model_imagination_beam_search_enabled
            ),
            "env_action_model_policy_improvement_enabled": (
                self._env_action_model_policy_improvement_enabled
            ),
            "env_action_model_policy_improvement_coef": (
                self._env_action_model_policy_improvement_coef
            ),
            "env_action_model_policy_improvement_temperature": (
                self._env_action_model_policy_improvement_temperature
            ),
            "env_action_model_policy_improvement_robust_horizons_enabled": (
                self._env_action_model_policy_improvement_robust_horizons_enabled
            ),
            "env_action_model_policy_improvement_horizon_risk_coef": (
                self._env_action_model_policy_improvement_horizon_risk_coef
            ),
            "env_action_model_policy_improvement_horizon_aggregation_mode": (
                self._env_action_model_policy_improvement_horizon_aggregation_mode
            ),
            "env_action_model_policy_improvement_horizon_lambda": (
                self._env_action_model_policy_improvement_horizon_lambda
            ),
            "env_action_model_policy_improvement_adaptive_kl_enabled": (
                self._env_action_model_policy_improvement_adaptive_kl_enabled
            ),
            "env_action_model_policy_improvement_target_kl": (
                self._env_action_model_policy_improvement_target_kl
            ),
            "env_action_model_policy_improvement_regret_adaptive_kl_enabled": (
                self._env_action_model_policy_improvement_regret_adaptive_kl_enabled
            ),
            "env_action_model_policy_improvement_max_target_kl": (
                self._env_action_model_policy_improvement_max_target_kl
            ),
            "env_action_model_policy_improvement_regret_priority_coef": (
                self._env_action_model_policy_improvement_regret_priority_coef
            ),
            "env_action_model_policy_improvement_tail_distillation_enabled": (
                self._env_action_model_policy_improvement_tail_distillation_enabled
            ),
            "env_action_model_policy_improvement_tail_quantile": (
                self._env_action_model_policy_improvement_tail_quantile
            ),
            "env_action_model_policy_improvement_tail_min_regret": (
                self._env_action_model_policy_improvement_tail_min_regret
            ),
            "env_action_model_policy_improvement_tail_epochs": (
                self._env_action_model_policy_improvement_tail_epochs
            ),
            "env_action_model_policy_improvement_tail_coef": (
                self._env_action_model_policy_improvement_tail_coef
            ),
            "env_action_model_policy_improvement_tail_max_policy_kl": (
                self._env_action_model_policy_improvement_tail_max_policy_kl
            ),
            "env_action_model_policy_improvement_tail_recovery_only": (
                self._env_action_model_policy_improvement_tail_recovery_only
            ),
            "env_action_model_policy_improvement_tail_adapter_only": (
                self._env_action_model_policy_improvement_tail_adapter_only
            ),
            "env_action_model_policy_improvement_tail_beam_only": (
                self._env_action_model_policy_improvement_tail_beam_only
            ),
            "env_action_model_policy_improvement_tail_planning_adapter_only": (
                self._env_action_model_policy_improvement_tail_planning_adapter_only
            ),
            "env_action_model_policy_improvement_tail_residual_optimizer_enabled": (
                self._env_action_model_policy_improvement_tail_residual_optimizer_enabled
            ),
            "env_action_model_policy_improvement_tail_residual_learning_rate": (
                self._env_action_model_policy_improvement_tail_residual_learning_rate
            ),
            "env_action_model_policy_improvement_tail_residual_backtrack_factor": (
                self._env_action_model_policy_improvement_tail_residual_backtrack_factor
            ),
            "env_action_model_policy_improvement_tail_residual_min_learning_rate": (
                self._env_action_model_policy_improvement_tail_residual_min_learning_rate
            ),
            "env_action_model_policy_improvement_tail_residual_max_backtracks": (
                self._env_action_model_policy_improvement_tail_residual_max_backtracks
            ),
            "env_action_model_policy_improvement_tail_logit_projection_enabled": (
                self._env_action_model_policy_improvement_tail_logit_projection_enabled
            ),
            "env_action_model_policy_improvement_tail_target_balance_enabled": (
                self._env_action_model_policy_improvement_tail_target_balance_enabled
            ),
            "env_action_model_policy_improvement_tail_target_balance_power": (
                self._env_action_model_policy_improvement_tail_target_balance_power
            ),
            "env_action_model_policy_improvement_tail_target_balance_max_weight": (
                self._env_action_model_policy_improvement_tail_target_balance_max_weight
            ),
            "learned_transition_model_enabled": self._learned_transition_model_enabled,
            "learned_transition_model_planner_enabled": self._learned_transition_model_planner_enabled,
            "learned_transition_model_ensemble_size": self._learned_transition_model_ensemble_size,
            "learned_transition_model_hidden_dim": self._learned_transition_model_hidden_dim,
            "learned_transition_model_learning_rate": self._learned_transition_model_learning_rate,
            "learned_transition_model_fit_epochs": self._learned_transition_model_fit_epochs,
            "learned_transition_model_max_samples": self._learned_transition_model_max_samples,
            "learned_transition_model_min_samples": self._learned_transition_model_min_samples,
            "learned_transition_model_risk_coef": self._learned_transition_model_risk_coef,
            "learned_transition_model_exploration_coef": self._learned_transition_model_exploration_coef,
            "learned_transition_model_policy_coef": self._learned_transition_model_policy_coef,
            "learned_transition_model_policy_prior_coef": self._learned_transition_model_policy_prior_coef,
            "learned_transition_model_min_margin": self._learned_transition_model_min_margin,
            "learned_transition_model_discount": self._learned_transition_model_discount,
            "learned_transition_model_warmup_updates": self._learned_transition_model_warmup_updates,
            "env_action_model_online_planner_enabled": (
                self._env_action_model_online_planner_enabled
            ),
            "env_action_model_online_planner_coef": (
                self._env_action_model_online_planner_coef
            ),
            "env_action_model_online_planner_mechanism_coef": (
                self._env_action_model_online_planner_mechanism_coef
            ),
            "env_action_model_online_planner_policy_prior_coef": (
                self._env_action_model_online_planner_policy_prior_coef
            ),
            "env_action_model_online_planner_min_margin": (
                self._env_action_model_online_planner_min_margin
            ),
            "env_action_model_resource_constraint_enabled": (
                self._env_action_model_resource_constraint_enabled
            ),
            "env_action_model_resource_cost_coef": (
                self._env_action_model_resource_cost_coef
            ),
            "env_action_model_resource_cost_scale": (
                self._env_action_model_resource_cost_scale
            ),
            "env_action_model_adaptive_horizon_enabled": (
                self._env_action_model_adaptive_horizon_enabled
            ),
            "env_action_model_adaptive_horizon_temperature": (
                self._env_action_model_adaptive_horizon_temperature
            ),
            "env_action_model_online_planner_prefer_beam_targets": (
                self._env_action_model_online_planner_prefer_beam_targets
            ),
            "env_action_model_beam_search_enabled": (
                self._env_action_model_beam_search_enabled
            ),
            "env_action_model_beam_search_horizon": (
                self._env_action_model_beam_search_horizon
            ),
            "env_action_model_beam_search_width": (
                self._env_action_model_beam_search_width
            ),
            "env_action_model_beam_search_context_only": (
                self._env_action_model_beam_search_context_only
            ),
            "env_action_model_beam_search_min_eta": (
                self._env_action_model_beam_search_min_eta
            ),
            "env_action_model_beam_search_max_eta": (
                self._env_action_model_beam_search_max_eta
            ),
            "env_action_model_policy_improvement_prefer_beam_targets": (
                self._env_action_model_policy_improvement_prefer_beam_targets
            ),
            "counterfactual_teacher_prd_enabled": self._counterfactual_teacher_prd_enabled,
            "counterfactual_teacher_event_coef": self._counterfactual_teacher_event_coef,
            "counterfactual_teacher_option_coef": self._counterfactual_teacher_option_coef,
            "counterfactual_teacher_clip": self._counterfactual_teacher_clip,
            "counterfactual_teacher_mechanism_bonus": self._counterfactual_teacher_mechanism_bonus,
            "counterfactual_teacher_missed_prepare_penalty": self._counterfactual_teacher_missed_prepare_penalty,
            "counterfactual_teacher_local_bonus": self._counterfactual_teacher_local_bonus,
            "counterfactual_teacher_current_rsu_penalty": self._counterfactual_teacher_current_rsu_penalty,
            "counterfactual_teacher_invalid_mechanism_penalty": (
                self._counterfactual_teacher_invalid_mechanism_penalty
            ),
            "service_continuity_teacher_enabled": self._service_continuity_teacher_enabled,
            "service_continuity_current_bonus": self._service_continuity_current_bonus,
            "service_continuity_prepare_bonus": self._service_continuity_prepare_bonus,
            "service_continuity_local_penalty": self._service_continuity_local_penalty,
            "service_continuity_min_prepare_context": self._service_continuity_min_prepare_context,
            "tail_risk_prd_enabled": self._tail_risk_prd_enabled,
            "tail_risk_policy_coef": self._tail_risk_policy_coef,
            "tail_risk_event_coef": self._tail_risk_event_coef,
            "tail_risk_option_coef": self._tail_risk_option_coef,
            "tail_risk_clip": self._tail_risk_clip,
            "tail_risk_quantile": self._tail_risk_quantile,
            "tail_risk_reward_shortfall_coef": self._tail_risk_reward_shortfall_coef,
            "tail_risk_service_coef": self._tail_risk_service_coef,
            "tail_risk_continuity_coef": self._tail_risk_continuity_coef,
            "tail_risk_handoff_failure_coef": self._tail_risk_handoff_failure_coef,
            "tail_risk_failed_mechanism_coef": self._tail_risk_failed_mechanism_coef,
            "tail_risk_redundant_mechanism_coef": self._tail_risk_redundant_mechanism_coef,
            "tail_risk_success_credit": self._tail_risk_success_credit,
            "opportunity_prd_enabled": self._opportunity_prd_enabled,
            "opportunity_policy_coef": self._opportunity_policy_coef,
            "opportunity_event_coef": self._opportunity_event_coef,
            "opportunity_option_coef": self._opportunity_option_coef,
            "opportunity_clip": self._opportunity_clip,
            "opportunity_reward_quantile": self._opportunity_reward_quantile,
            "opportunity_reward_surplus_coef": self._opportunity_reward_surplus_coef,
            "opportunity_service_success_coef": self._opportunity_service_success_coef,
            "opportunity_cache_hit_coef": self._opportunity_cache_hit_coef,
            "opportunity_continuity_coef": self._opportunity_continuity_coef,
            "opportunity_current_rsu_efficiency_coef": self._opportunity_current_rsu_efficiency_coef,
            "opportunity_local_fallback_coef": self._opportunity_local_fallback_coef,
            "opportunity_backhaul_penalty_coef": self._opportunity_backhaul_penalty_coef,
            "opportunity_delay_penalty_coef": self._opportunity_delay_penalty_coef,
            "opportunity_failed_service_penalty_coef": self._opportunity_failed_service_penalty_coef,
            "opportunity_mechanism_success_bonus": self._opportunity_mechanism_success_bonus,
            "handoff_risk_prd_enabled": self._handoff_risk_prd_enabled,
            "handoff_risk_event_coef": self._handoff_risk_event_coef,
            "handoff_risk_option_coef": self._handoff_risk_option_coef,
            "handoff_risk_clip": self._handoff_risk_clip,
            "handoff_risk_failure_penalty": self._handoff_risk_failure_penalty,
            "handoff_risk_ready_bonus": self._handoff_risk_ready_bonus,
            "handoff_risk_prepare_bonus": self._handoff_risk_prepare_bonus,
            "handoff_risk_unprepared_penalty": self._handoff_risk_unprepared_penalty,
            "handoff_risk_confidence_threshold": self._handoff_risk_confidence_threshold,
            "handoff_risk_cost_dual_enabled": self._handoff_risk_cost_dual_enabled,
            "handoff_risk_cost_dual_lr": self._handoff_risk_cost_dual_lr,
            "handoff_risk_cost_target": self._handoff_risk_cost_target,
            "handoff_risk_cost_dual_max": self._handoff_risk_cost_dual_max,
            "handoff_risk_cost_dual_initial": self._handoff_risk_cost_dual,
            "idle_execution_prd_enabled": self._idle_execution_prd_enabled,
            "idle_execution_policy_coef": self._idle_execution_policy_coef,
            "idle_execution_option_coef": self._idle_execution_option_coef,
            "idle_execution_clip": self._idle_execution_clip,
            "idle_execution_current_rsu_delay_coef": self._idle_execution_current_rsu_delay_coef,
            "idle_execution_local_bonus": self._idle_execution_local_bonus,
            "idle_execution_mechanism_penalty": self._idle_execution_mechanism_penalty,
            "idle_execution_timing_threshold": self._idle_execution_timing_threshold,
            "idle_execution_mechanism_preserve_bonus": self._idle_execution_mechanism_preserve_bonus,
            "net_utility_prd_enabled": self._net_utility_prd_enabled,
            "net_utility_backhaul_coef": self._net_utility_backhaul_coef,
            "net_utility_migration_coef": self._net_utility_migration_coef,
            "net_utility_expired_prefetch_coef": self._net_utility_expired_prefetch_coef,
            "net_utility_idle_prefetch_penalty": self._net_utility_idle_prefetch_penalty,
            "net_utility_failed_mechanism_penalty": self._net_utility_failed_mechanism_penalty,
            "net_utility_failed_mechanism_backhaul_coef": self._net_utility_failed_mechanism_backhaul_coef,
            "net_utility_mechanism_window_failed_penalty_scale": (
                self._net_utility_mechanism_window_failed_penalty_scale
            ),
            "net_utility_success_bonus": self._net_utility_success_bonus,
            "net_utility_backhaul_normalizer": self._net_utility_backhaul_normalizer,
            "net_utility_cost_dual_enabled": self._net_utility_cost_dual_enabled,
            "net_utility_cost_dual_lr": self._net_utility_cost_dual_lr,
            "net_utility_cost_target": self._net_utility_cost_target,
            "net_utility_cost_dual_max": self._net_utility_cost_dual_max,
            "net_utility_cost_dual_initial": self._net_utility_cost_dual,
            "net_utility_option_termination_enabled": self._net_utility_option_termination_enabled,
            "net_utility_option_termination_conservative_enabled": (
                self._net_utility_option_termination_conservative_enabled
            ),
            "net_utility_option_termination_max_timing_support": (
                self._net_utility_option_termination_max_timing_support
            ),
            "dag_aware_option_termination_enabled": self._dag_aware_option_termination_enabled,
            "dag_aware_option_min_critical_path": self._dag_aware_option_min_critical_path,
            "dag_aware_option_short_workflow_max_nodes": self._dag_aware_option_short_workflow_max_nodes,
            "dag_aware_option_branching_successors": self._dag_aware_option_branching_successors,
            "dag_aware_idle_prefetch_confidence_floor": self._dag_aware_idle_prefetch_confidence_floor,
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
        for semantic_state, policy_output in zip(batch_states, batch_outputs):
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
        run_metadata: dict[str, Any] | None = None,
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
                steady_rsu_candidate = bool(
                    float(pseudo_targets.get("steady_rsu_candidate", 0.0) or 0.0) > 0.0
                )
                if (
                    self._steady_rsu_bias_enabled
                    and steady_rsu_candidate
                    and self._steady_rsu_bias_strength > 0.0
                ):
                    fast_logits[0] = fast_logits[0] + self._steady_rsu_bias_strength * confidence
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
                    "steady_rsu_candidate": steady_rsu_candidate,
                }
        adjusted = self._apply_digital_twin_policy_prior(
            adjusted,
            semantic_state,
            run_metadata=run_metadata,
        )
        adjusted = self._apply_opportunity_constrained_policy(adjusted, semantic_state)
        adjusted = self._apply_backhaul_aware_policy(adjusted, semantic_state)
        adjusted = self._apply_net_advantage_prepare_gate(adjusted, semantic_state)
        adjusted = self._apply_sparse_handoff_recovery_prior(
            adjusted,
            semantic_state,
            run_metadata=run_metadata,
        )
        adjusted = self._apply_continuity_guard(adjusted, semantic_state)
        adjusted = self._apply_service_completion_gate(adjusted, semantic_state)
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
        steady_rsu_candidate = bool(
            self._steady_rsu_bias_enabled
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
        if steady_rsu_candidate:
            confidence_weight = max(confidence_weight, self._steady_rsu_confidence_floor)

        return {
            "slow_target": slow_target,
            "fast_target": fast_target,
            "event_target": event_target,
            "confidence_weight": confidence_weight,
            "event_soft_target": event_soft_target,
            "latency_fallback_candidate": float(latency_fallback_candidate),
            "steady_rsu_candidate": float(steady_rsu_candidate),
            "temporal_urgency": float(timing_features["temporal_urgency"]),
            "prepare_window_score": float(timing_features["prepare_window_score"]),
            "handoff_countdown_steps": float(timing_features["countdown_steps"]),
        }


class JSONCheckpointMixin:
    """用于启发式或非 torch 智能体的最小 JSON checkpoint。"""

    def _save_json_checkpoint(self, path: str, payload: dict[str, Any]) -> None:
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
