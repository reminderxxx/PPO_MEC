"""Digital-twin handoff and service-migration PPO baseline."""

from __future__ import annotations

import math
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


def _build_dt_handoff_feature_tensor(semantic_state: dict[str, Any]) -> torch.Tensor:
    primary_vehicle = _resolve_primary_vehicle(semantic_state)
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

    current_rsu = _rsu_by_id(semantic_state, current_rsu_id)
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


class _DTHandoffPolicyNetwork(nn.Module):
    """Flat semantic actor-critic augmented with raw digital-twin prediction scalars."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.encoder = FlatSemanticEncoder(hidden_dim=hidden_dim)
        self.dt_projection = nn.Sequential(
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
        dt_features = _build_dt_handoff_feature_tensor(semantic_state)
        dt_embedding = self.dt_projection(dt_features.unsqueeze(0)).squeeze(0)
        slow_context = self.slow_norm(encoded["slow_context"] + 0.45 * dt_embedding)
        fast_context = self.fast_norm(encoded["fast_context"] + 0.50 * dt_embedding)
        event_context = self.event_norm(encoded["event_context"] + dt_embedding)
        critic_context = self.critic_norm(encoded["centralized_critic_context"] + dt_embedding)
        temperature = max(float(event_logit_temperature if event_logit_temperature is not None else 1.0), 0.25)
        value = self.central_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
        encoded["encoder_mode"] = "dt_handoff_domain_baseline"
        encoded["dt_prediction_available"] = dt_features[0:1]
        encoded["dt_target_differs"] = dt_features[2:3]
        encoded["dt_boundary_urgency"] = dt_features[11:12]
        return {
            "encoded": encoded,
            "slow_logits": self.slow_actor(slow_context.unsqueeze(0)).squeeze(0),
            "fast_logits": self.fast_actor(fast_context.unsqueeze(0)).squeeze(0),
            "event_logits": self.event_actor(event_context.unsqueeze(0)).squeeze(0) / temperature,
            "value": value,
            "critic_mode": "dt_handoff_centralized_critic",
            "critic_context_key": "flat_semantic_plus_digital_twin_handoff_scalars",
            "head_values": {"slow": value, "fast": value, "event": value},
        }


class DTHandoffDRLAgent(PPOBaseAgent):
    """Digital-twin-assisted handoff/service-migration learned baseline.

    The baseline consumes raw digital-twin prediction snapshot fields such as
    predicted RSU sequence, dwell time, confidence, future load, and boundary
    pressure. It deliberately excludes SA-GHMAPPO's calibrated surrogate gate,
    uncertainty-aware event scaling, temporal smoothing, and handoff guards.
    """

    observation_contract = "flat_semantic_plus_dt_handoff_v1"
    action_contract = "semantic_discrete_5_dt_handoff_3head"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="dt_handoff_drl",
            policy_type="dt_handoff_drl_policy",
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
        self._network = _DTHandoffPolicyNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self.baseline_config = {
            "family": "dt_handoff_drl",
            "domain_focus": "digital_twin_handoff_and_service_migration",
            "flat_policy": False,
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "digital_twin_snapshot_features": True,
            "graph_encoder": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": True,
            "ctde_scope": "controller_level_cache_execution_handoff",
            "paper_grade_independent_baseline": True,
            "reference_basis": "Digital-twin-assisted VEC offloading and service-migration DRL literature.",
            "excluded_sa_mechanisms": [
                "graph_encoder",
                "calibrated_surrogate_gate",
                "uncertainty_aware_event_scaling",
                "dag_dependency_aware_features",
                "mechanism_auxiliary_loss",
                "heuristic_imitation",
                "deterministic_temporal_smoothing",
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
                "domain_feature_block": "digital_twin_handoff_scalars",
            }
        )
        return config
