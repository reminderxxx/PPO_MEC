"""图编码、RSU 编码与预测融合编码器。"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.encoders.dag_graph_encoder import DAGGraphEncoder
from src.encoders.rsu_state_encoder import RSUStateEncoder


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _resolve_primary_vehicle_from_semantic_state(semantic_state: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(semantic_state.get("vehicles", []))
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    if primary_vehicle_id:
        primary_vehicle_id = str(primary_vehicle_id)
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == primary_vehicle_id:
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


def compute_temporal_handoff_features(semantic_state: dict[str, Any]) -> dict[str, float]:
    rsus = semantic_state.get("rsus", [])
    predictions = semantic_state.get("predictions", {})
    primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
    predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    horizon = max(len(next_sequence), 1)

    countdown_steps: float | None = None
    for step_index, next_rsu_id in enumerate(next_sequence, start=1):
        if next_rsu_id is None:
            continue
        if current_rsu_id is None or next_rsu_id != current_rsu_id:
            countdown_steps = float(step_index)
            break
    has_predicted_handoff = bool(predicted_handoff_target_rsu_id is not None or countdown_steps is not None)
    effective_countdown = float(countdown_steps if countdown_steps is not None else horizon + 1)
    countdown_norm = 1.0 if not has_predicted_handoff else _clamp01((effective_countdown - 1.0) / max(float(horizon), 1.0))
    urgency_from_countdown = 0.0 if not has_predicted_handoff else 1.0 - countdown_norm

    current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
    coverage_radius = max(float(current_rsu.get("coverage_radius", 0.0)), 1.0)
    dx = float(primary_vehicle.get("position_x", 0.0)) - float(current_rsu.get("position_x", primary_vehicle.get("position_x", 0.0)))
    dy = float(primary_vehicle.get("position_y", 0.0)) - float(current_rsu.get("position_y", primary_vehicle.get("position_y", 0.0)))
    distance_to_center = math.sqrt(dx * dx + dy * dy) if current_rsu else 0.0
    distance_to_boundary = max(coverage_radius - distance_to_center, 0.0) if current_rsu else 0.0
    boundary_margin_norm = _clamp01(distance_to_boundary / coverage_radius) if current_rsu else 0.0
    boundary_urgency = 1.0 - boundary_margin_norm if current_rsu else 1.0

    service_steps_remaining = float(semantic_state.get("current_node_service_steps_remaining", 0.0) or 0.0)
    service_pressure = 0.0
    if has_predicted_handoff:
        service_pressure = _clamp01(service_steps_remaining / max(effective_countdown, 1.0))

    temporal_urgency = _clamp01(
        0.55 * urgency_from_countdown
        + 0.25 * boundary_urgency
        + 0.20 * service_pressure
    )
    return {
        "has_predicted_handoff": float(has_predicted_handoff),
        "countdown_steps": effective_countdown,
        "countdown_norm": countdown_norm,
        "boundary_margin_norm": boundary_margin_norm,
        "boundary_urgency": boundary_urgency,
        "service_pressure": service_pressure,
        "temporal_urgency": temporal_urgency,
    }


def compute_temporal_prepare_window_score(
    semantic_state: dict[str, Any],
    preferred_lead_steps: float = 2.5,
    sigma: float = 1.25,
) -> dict[str, float]:
    temporal_features = compute_temporal_handoff_features(semantic_state)
    effective_sigma = max(float(sigma), 0.25)
    if not bool(temporal_features["has_predicted_handoff"]):
        return {
            **temporal_features,
            "prepare_window_score": 0.0,
            "preferred_lead_steps": float(preferred_lead_steps),
            "sigma": effective_sigma,
        }
    normalized_distance = (float(temporal_features["countdown_steps"]) - float(preferred_lead_steps)) / effective_sigma
    gaussian_score = math.exp(-0.5 * normalized_distance * normalized_distance)
    prepare_window_score = _clamp01(
        gaussian_score * (0.45 + 0.55 * float(temporal_features["temporal_urgency"]))
    )
    return {
        **temporal_features,
        "prepare_window_score": prepare_window_score,
        "preferred_lead_steps": float(preferred_lead_steps),
        "sigma": effective_sigma,
    }


def build_prediction_reliability_summary(
    semantic_state: dict[str, Any],
    *,
    prediction_gate_min_leak: float = 0.0,
) -> dict[str, float]:
    predictions = semantic_state.get("predictions", {})
    primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    temporal_features = compute_temporal_prepare_window_score(semantic_state)
    confidence = _clamp01(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
    uncertainty = _clamp01(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0))
    raw_reliability = _clamp01(confidence * (1.0 - uncertainty))
    temporal_urgency = _clamp01(temporal_features.get("temporal_urgency", 0.0))
    prepare_window_score = _clamp01(temporal_features.get("prepare_window_score", 0.0))
    timing_support = max(temporal_urgency, prepare_window_score)
    urgency_support = 0.7 + 0.3 * temporal_urgency
    gate_value = _clamp01(raw_reliability * urgency_support)
    reliability_reference = max(float(prediction_gate_min_leak), 0.2, 1e-6)
    calibrated_reliability = _clamp01(gate_value / reliability_reference)
    reliability = _clamp01(0.4 * raw_reliability + 0.6 * calibrated_reliability)
    reliability_timing_alignment = _clamp01(
        reliability * (0.35 + 0.65 * timing_support)
    )
    conservative_prepare_pressure = _clamp01(
        (1.0 - reliability) * (0.4 + 0.6 * timing_support)
    )
    return {
        "prediction_confidence": confidence,
        "prediction_uncertainty": uncertainty,
        "prediction_reliability_raw": raw_reliability,
        "prediction_reliability": reliability,
        "prediction_gate_value": gate_value,
        "urgency_support": urgency_support,
        "timing_support": timing_support,
        "reliability_timing_alignment": reliability_timing_alignment,
        "conservative_prepare_pressure": conservative_prepare_pressure,
        "temporal_urgency": temporal_urgency,
        "prepare_window_score": prepare_window_score,
    }


def build_graph_continuity_critic_features(
    semantic_state: dict[str, Any],
    *,
    prediction_gate_min_leak: float = 0.0,
) -> dict[str, float]:
    workflow = semantic_state.get("workflow", {})
    current_node = semantic_state.get("current_workflow_node") or {}
    predictions = semantic_state.get("predictions", {})
    rsus = semantic_state.get("rsus", [])
    primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
    predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    next_sequence = [
        rsu_id for rsu_id in predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
        if rsu_id is not None
    ]
    unique_future_rsus = []
    for rsu_id in next_sequence:
        if rsu_id not in unique_future_rsus:
            unique_future_rsus.append(rsu_id)
    horizon = max(len(predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])), 1)
    future_switch_count = 0
    previous_rsu_id = current_rsu_id
    for rsu_id in next_sequence:
        if previous_rsu_id is not None and rsu_id != previous_rsu_id:
            future_switch_count += 1
        previous_rsu_id = rsu_id

    nodes = list(workflow.get("nodes", []))
    node_map = {str(node.get("node_id")): node for node in nodes}
    completed_node_ids = {str(node_id) for node_id in workflow.get("completed_node_ids", [])}
    remaining_node_ids = [
        node_id for node_id in node_map
        if node_id not in completed_node_ids
    ]
    remaining_nodes_ratio = float(len(remaining_node_ids)) / max(len(node_map), 1)
    frontier_node_ids = [
        node_id for node_id in remaining_node_ids
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
    critical_path_length = max((remaining_path_length(node_id) for node_id in frontier_node_ids), default=current_path_length)
    normalized_current_path = float(current_path_length) / max(len(node_map), 1)
    normalized_critical_path = float(critical_path_length) / max(len(node_map), 1)
    frontier_width_ratio = float(len(frontier_node_ids)) / max(len(node_map), 1)

    reliability_summary = build_prediction_reliability_summary(
        semantic_state,
        prediction_gate_min_leak=prediction_gate_min_leak,
    )
    current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
    target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_handoff_target_rsu_id), {})
    required_adapter = current_node.get("required_adapter")
    current_future_load = float(predictions.get("future_load", {}).get(current_rsu_id, 0.0))
    target_future_load = float(predictions.get("future_load", {}).get(predicted_handoff_target_rsu_id, 0.0))
    current_has_adapter = 1.0 if required_adapter and required_adapter in current_rsu.get("cached_adapter_ids", []) else 0.0
    target_has_adapter = 1.0 if required_adapter and required_adapter in target_rsu.get("cached_adapter_ids", []) else 0.0
    positive_target_future_load_gap = max((target_future_load - current_future_load) / 10.0, 0.0)
    conservative_prepare_pressure = float(reliability_summary["conservative_prepare_pressure"])
    low_reliability_switch_pressure = conservative_prepare_pressure * (
        float(future_switch_count) / max(float(horizon), 1.0)
    )
    low_reliability_load_pressure = conservative_prepare_pressure * positive_target_future_load_gap
    low_reliability_migration_pressure = conservative_prepare_pressure * (
        1.0 if predicted_handoff_target_rsu_id and predicted_handoff_target_rsu_id != current_rsu_id else 0.0
    )

    return {
        "remaining_nodes_ratio": remaining_nodes_ratio,
        "frontier_width_ratio": frontier_width_ratio,
        "current_path_length_norm": normalized_current_path,
        "critical_path_length_norm": normalized_critical_path,
        "predicted_path_switch_ratio": float(future_switch_count) / max(float(horizon), 1.0),
        "future_unique_rsu_ratio": float(len(unique_future_rsus)) / max(float(horizon), 1.0),
        "predicted_next_differs": 1.0 if predicted_next_rsu_id and predicted_next_rsu_id != current_rsu_id else 0.0,
        "predicted_target_differs": 1.0 if predicted_handoff_target_rsu_id and predicted_handoff_target_rsu_id != current_rsu_id else 0.0,
        "current_has_adapter": current_has_adapter,
        "target_has_adapter": target_has_adapter,
        "target_future_load_gap": (target_future_load - current_future_load) / 10.0,
        "prediction_confidence": float(reliability_summary["prediction_confidence"]),
        "prediction_uncertainty": float(reliability_summary["prediction_uncertainty"]),
        "prediction_reliability_raw": float(reliability_summary["prediction_reliability_raw"]),
        "prediction_reliability": float(reliability_summary["prediction_reliability"]),
        "reliability_timing_alignment": float(reliability_summary["reliability_timing_alignment"]),
        "conservative_prepare_pressure": conservative_prepare_pressure,
        "low_reliability_switch_pressure": low_reliability_switch_pressure,
        "low_reliability_load_pressure": low_reliability_load_pressure,
        "low_reliability_migration_pressure": low_reliability_migration_pressure,
        "temporal_urgency": float(reliability_summary["temporal_urgency"]),
        "prepare_window_score": float(reliability_summary["prepare_window_score"]),
    }


class FlatSemanticEncoder(nn.Module):
    """flat baseline encoder。"""

    def __init__(self, input_dim: int = 18, hidden_dim: int = 64) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._hidden_dim = int(hidden_dim)
        self._projection = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.Tanh(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.Tanh(),
        )

    def forward(self, semantic_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        feature_tensor = self._build_feature_tensor(semantic_state)
        centralized_feature_tensor = self._build_centralized_feature_tensor(semantic_state)
        embedding = self._projection(feature_tensor.unsqueeze(0)).squeeze(0)
        centralized_embedding = self._projection(centralized_feature_tensor.unsqueeze(0)).squeeze(0)
        return {
            "shared_embedding": embedding,
            "slow_context": embedding,
            "fast_context": embedding,
            "event_context": embedding,
            "critic_context": embedding,
            "centralized_critic_context": centralized_embedding,
            "prediction_gate": torch.tensor([1.0], dtype=torch.float32, device=embedding.device),
            "encoder_mode": "flat_baseline",
        }

    def _build_feature_tensor(self, semantic_state: dict[str, Any]) -> torch.Tensor:
        predictions = semantic_state.get("predictions", {})
        vehicles = semantic_state.get("vehicles", [])
        rsus = semantic_state.get("rsus", [])
        workflow = semantic_state.get("workflow", {})
        current_node = semantic_state.get("current_workflow_node") or {}
        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        completed_count = len(workflow.get("completed_node_ids", []))
        planned_count = max(len(workflow.get("execution_order", [])), 1)
        progress = float(completed_count) / float(planned_count)
        future_load = predictions.get("future_load", {})
        mean_future_load = sum(float(value) for value in future_load.values()) / max(len(future_load), 1) if future_load else 0.0
        confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
        uncertainty = float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0))
        dwell_time = float(predictions.get("dwell_time", {}).get(vehicle_id, 0.0))
        feature_list = [
            float(semantic_state.get("time_index", 0.0)) / 10000.0,
            float(len(vehicles)) / 20.0,
            float(len(rsus)) / 10.0,
            progress,
            float(len(semantic_state.get("handoff_events", []))) / 4.0,
            1.0 if current_node else 0.0,
            float(current_node.get("input_size", 0.0)) / 1000.0,
            float(current_node.get("output_size", 0.0)) / 1000.0,
            float(len(current_node.get("predecessors", []))) / 4.0,
            float(len(current_node.get("successors", []))) / 4.0,
            1.0 if primary_vehicle.get("associated_rsu_id") else 0.0,
            float(primary_vehicle.get("speed", 0.0)) / 50.0,
            dwell_time / 20.0,
            mean_future_load / 10.0,
            float(len(predictions.get("predicted_handoff_vehicle_ids", []))) / 10.0,
            confidence,
            uncertainty,
            1.0 if predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id) else 0.0,
        ]
        return torch.tensor(feature_list, dtype=torch.float32)

    def _build_centralized_feature_tensor(self, semantic_state: dict[str, Any]) -> torch.Tensor:
        predictions = semantic_state.get("predictions", {})
        vehicles = list(semantic_state.get("vehicles", []))
        rsus = list(semantic_state.get("rsus", []))
        workflow = semantic_state.get("workflow", {})
        current_node = semantic_state.get("current_workflow_node") or {}
        completed_node_ids = set(workflow.get("completed_node_ids", []))
        execution_order = list(workflow.get("execution_order", []))
        nodes = list(workflow.get("nodes", []))
        planned_count = max(len(execution_order), len(nodes), 1)
        progress = float(len(completed_node_ids)) / float(planned_count)
        remaining_ratio = 1.0 - progress
        frontier_count = 0
        for node in nodes:
            node_id = node.get("node_id")
            if node_id in completed_node_ids:
                continue
            predecessors = set(node.get("predecessors", []))
            if predecessors.issubset(completed_node_ids):
                frontier_count += 1
        cache_occupancies = []
        for rsu in rsus:
            capacity = max(float(rsu.get("cache_capacity", rsu.get("adapter_cache_capacity", 1.0)) or 1.0), 1.0)
            cache_occupancies.append(float(len(rsu.get("cached_adapter_ids", []))) / capacity)
        mean_cache_occupancy = sum(cache_occupancies) / max(len(cache_occupancies), 1) if cache_occupancies else 0.0
        max_cache_occupancy = max(cache_occupancies) if cache_occupancies else 0.0
        future_load = predictions.get("future_load", {})
        future_load_values = [float(value) for value in future_load.values()] if isinstance(future_load, dict) else []
        mean_future_load = sum(future_load_values) / max(len(future_load_values), 1) if future_load_values else 0.0
        max_future_load = max(future_load_values) if future_load_values else 0.0
        confidence_values = [
            float(value)
            for value in predictions.get("prediction_confidence_by_vehicle", {}).values()
        ] if isinstance(predictions, dict) else []
        uncertainty_values = [
            float(value)
            for value in predictions.get("prediction_uncertainty_by_vehicle", {}).values()
        ] if isinstance(predictions, dict) else []
        mean_confidence = sum(confidence_values) / max(len(confidence_values), 1) if confidence_values else 0.0
        mean_uncertainty = sum(uncertainty_values) / max(len(uncertainty_values), 1) if uncertainty_values else 1.0
        next_rsu_sequences = predictions.get("next_rsu_sequence", {}) if isinstance(predictions, dict) else {}
        non_null_future_hops = 0
        total_future_hops = 0
        unique_future_rsus: set[Any] = set()
        if isinstance(next_rsu_sequences, dict):
            for sequence in next_rsu_sequences.values():
                for rsu_id in sequence or []:
                    total_future_hops += 1
                    if rsu_id is not None:
                        non_null_future_hops += 1
                        unique_future_rsus.add(rsu_id)
        feature_list = [
            float(semantic_state.get("time_index", 0.0)) / 10000.0,
            float(len(vehicles)) / 20.0,
            float(len(rsus)) / 10.0,
            progress,
            remaining_ratio,
            float(frontier_count) / float(planned_count),
            float(len(semantic_state.get("handoff_events", []))) / 4.0,
            float(len(predictions.get("predicted_handoff_vehicle_ids", []))) / 10.0 if isinstance(predictions, dict) else 0.0,
            mean_future_load / 10.0,
            max_future_load / 10.0,
            mean_cache_occupancy,
            max_cache_occupancy,
            mean_confidence,
            mean_uncertainty,
            float(non_null_future_hops) / max(float(total_future_hops), 1.0),
            float(len(unique_future_rsus)) / max(float(len(rsus)), 1.0),
            1.0 if current_node else 0.0,
            float(current_node.get("input_size", 0.0)) / 1000.0,
        ]
        return torch.tensor(feature_list, dtype=torch.float32)


class SurrogateFusionEncoder(nn.Module):
    """主方法使用的 DAG + RSU + surrogate 融合编码器。"""

    def __init__(
        self,
        hidden_dim: int = 64,
        use_prediction_features: bool = True,
        use_uncertainty_signal: bool = True,
        use_dependency_aware: bool = True,
        prediction_feature_dim: int = 13,
        prediction_gate_min_leak: float = 0.0,
        graph_continuity_critic_enabled: bool = False,
        uncertainty_aware_critic_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self._use_prediction_features = bool(use_prediction_features)
        self._use_uncertainty_signal = bool(use_uncertainty_signal)
        self._prediction_feature_dim = int(prediction_feature_dim)
        self._prediction_gate_min_leak = max(0.0, min(float(prediction_gate_min_leak), 1.0))
        self._graph_continuity_critic_enabled = bool(graph_continuity_critic_enabled)
        self._uncertainty_aware_critic_enabled = bool(uncertainty_aware_critic_enabled)
        self._dag_encoder = DAGGraphEncoder(
            hidden_dim=self._hidden_dim,
            use_dependency_aware=use_dependency_aware,
        )
        self._rsu_encoder = RSUStateEncoder(hidden_dim=self._hidden_dim)
        self._vehicle_projection = nn.Sequential(
            nn.Linear(10, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self._prediction_projection = nn.Sequential(
            nn.Linear(self._prediction_feature_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self._shared_fusion = nn.Sequential(
            nn.Linear(self._hidden_dim * 7, self._hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim * 2, self._hidden_dim),
            nn.ReLU(),
        )
        self._slow_head_context = nn.Linear(self._hidden_dim * 2, self._hidden_dim)
        self._fast_head_context = nn.Linear(self._hidden_dim * 2, self._hidden_dim)
        self._event_head_context = nn.Linear(self._hidden_dim * 2, self._hidden_dim)
        critic_context_input_dim = self._hidden_dim * 2
        if self._graph_continuity_critic_enabled:
            self._critic_continuity_projection = nn.Sequential(
                nn.Linear(22, self._hidden_dim),
                nn.ReLU(),
                nn.Linear(self._hidden_dim, self._hidden_dim),
            )
            critic_context_input_dim += self._hidden_dim
            if self._uncertainty_aware_critic_enabled:
                self._critic_reliability_projection = nn.Sequential(
                    nn.Linear(8, self._hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self._hidden_dim, self._hidden_dim),
                )
                self._critic_reliability_gate = nn.Sequential(
                    nn.Linear(8, self._hidden_dim),
                    nn.Sigmoid(),
                )
        self._critic_context = nn.Linear(critic_context_input_dim, self._hidden_dim)

    def forward(self, semantic_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        dag_output = self._dag_encoder(semantic_state)
        rsu_output = self._rsu_encoder(semantic_state)
        vehicle_embedding = self._vehicle_projection(self._build_vehicle_feature_tensor(semantic_state).unsqueeze(0)).squeeze(0)
        prediction_embedding, prediction_gate, temporal_features = self._build_prediction_embedding(semantic_state)
        fused = self._shared_fusion(
            torch.cat(
                [
                    dag_output["graph_embedding"],
                    dag_output["current_node_embedding"],
                    dag_output["frontier_embedding"],
                    rsu_output["set_embedding"],
                    rsu_output["current_rsu_embedding"],
                    rsu_output["target_rsu_embedding"],
                    vehicle_embedding + prediction_embedding,
                ],
                dim=-1,
            ).unsqueeze(0)
        ).squeeze(0)

        slow_context = torch.tanh(
            self._slow_head_context(torch.cat([fused, rsu_output["target_rsu_embedding"] + prediction_embedding], dim=-1))
        )
        fast_context = torch.tanh(
            self._fast_head_context(torch.cat([fused, dag_output["current_node_embedding"] + rsu_output["current_rsu_embedding"]], dim=-1))
        )
        event_context = torch.tanh(
            self._event_head_context(torch.cat([fused, rsu_output["target_rsu_embedding"] + prediction_embedding], dim=-1))
        )
        reliability_summary = build_prediction_reliability_summary(
            semantic_state,
            prediction_gate_min_leak=self._prediction_gate_min_leak,
        )
        critic_context_components = [
            fused,
            dag_output["graph_embedding"] + rsu_output["set_embedding"],
        ]
        if self._graph_continuity_critic_enabled:
            continuity_features = build_graph_continuity_critic_features(
                semantic_state,
                prediction_gate_min_leak=self._prediction_gate_min_leak,
            )
            continuity_tensor = torch.tensor(
                [
                    float(continuity_features["remaining_nodes_ratio"]),
                    float(continuity_features["frontier_width_ratio"]),
                    float(continuity_features["current_path_length_norm"]),
                    float(continuity_features["critical_path_length_norm"]),
                    float(continuity_features["predicted_path_switch_ratio"]),
                    float(continuity_features["future_unique_rsu_ratio"]),
                    float(continuity_features["predicted_next_differs"]),
                    float(continuity_features["predicted_target_differs"]),
                    float(continuity_features["current_has_adapter"]),
                    float(continuity_features["target_has_adapter"]),
                    float(continuity_features["target_future_load_gap"]),
                    float(continuity_features["prediction_confidence"]),
                    float(continuity_features["prediction_uncertainty"]),
                    float(continuity_features["prediction_reliability_raw"]),
                    float(continuity_features["prediction_reliability"]),
                    float(continuity_features["reliability_timing_alignment"]),
                    float(continuity_features["conservative_prepare_pressure"]),
                    float(continuity_features["low_reliability_switch_pressure"]),
                    float(continuity_features["low_reliability_load_pressure"]),
                    float(continuity_features["low_reliability_migration_pressure"]),
                    float(continuity_features["temporal_urgency"]),
                    float(continuity_features["prepare_window_score"]),
                ],
                dtype=torch.float32,
                device=fused.device,
            )
            continuity_embedding = self._critic_continuity_projection(
                continuity_tensor.unsqueeze(0)
            ).squeeze(0)
            critic_context_components.append(continuity_embedding)
        critic_pre = self._critic_context(torch.cat(critic_context_components, dim=-1))
        if self._graph_continuity_critic_enabled and self._uncertainty_aware_critic_enabled:
            reliability_tensor = torch.tensor(
                [
                    float(reliability_summary["prediction_confidence"]),
                    float(reliability_summary["prediction_uncertainty"]),
                    float(reliability_summary["prediction_reliability_raw"]),
                    float(reliability_summary["prediction_reliability"]),
                    float(reliability_summary["prediction_gate_value"]),
                    float(reliability_summary["timing_support"]),
                    float(reliability_summary["reliability_timing_alignment"]),
                    float(reliability_summary["conservative_prepare_pressure"]),
                ],
                dtype=torch.float32,
                device=fused.device,
            )
            reliability_gate = 0.35 + 0.65 * self._critic_reliability_gate(
                reliability_tensor.unsqueeze(0)
            ).squeeze(0)
            reliability_bias = self._critic_reliability_projection(
                reliability_tensor.unsqueeze(0)
            ).squeeze(0)
            critic_context = torch.tanh(critic_pre * reliability_gate + reliability_bias)
        else:
            critic_context = torch.tanh(critic_pre)
        return {
            "shared_embedding": fused,
            "slow_context": slow_context,
            "fast_context": fast_context,
            "event_context": event_context,
            "critic_context": critic_context,
            "prediction_gate": prediction_gate,
            "handoff_countdown_steps": torch.tensor(
                [float(temporal_features["countdown_steps"])],
                dtype=torch.float32,
                device=fused.device,
            ),
            "temporal_urgency": torch.tensor(
                [float(temporal_features["temporal_urgency"])],
                dtype=torch.float32,
                device=fused.device,
            ),
            "prepare_window_score": torch.tensor(
                [float(temporal_features["prepare_window_score"])],
                dtype=torch.float32,
                device=fused.device,
            ),
            "prediction_reliability": torch.tensor(
                [float(reliability_summary["prediction_reliability"])],
                dtype=torch.float32,
                device=fused.device,
            ),
            "conservative_prepare_pressure": torch.tensor(
                [float(reliability_summary["conservative_prepare_pressure"])],
                dtype=torch.float32,
                device=fused.device,
            ),
            "encoder_mode": "graph_surrogate_main",
        }

    def _build_vehicle_feature_tensor(self, semantic_state: dict[str, Any]) -> torch.Tensor:
        current_node = semantic_state.get("current_workflow_node") or {}
        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predictions = semantic_state.get("predictions", {})
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        dwell_time = float(predictions.get("dwell_time", {}).get(vehicle_id, 0.0))
        confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
        uncertainty = float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0))
        return torch.tensor(
            [
                float(primary_vehicle.get("position_x", 0.0)) / 1000.0,
                float(primary_vehicle.get("position_y", 0.0)) / 1000.0,
                float(primary_vehicle.get("speed", 0.0)) / 50.0,
                1.0 if current_rsu_id else 0.0,
                float(current_node.get("input_size", 0.0)) / 1000.0,
                float(current_node.get("output_size", 0.0)) / 1000.0,
                float(len(current_node.get("predecessors", []))) / 4.0,
                dwell_time / 20.0,
                confidence,
                uncertainty,
            ],
            dtype=torch.float32,
        )

    def _build_prediction_embedding(
        self,
        semantic_state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        predictions = semantic_state.get("predictions", {})
        workflow = semantic_state.get("workflow", {})
        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        future_load = predictions.get("future_load", {})
        mean_future_load = sum(float(value) for value in future_load.values()) / max(len(future_load), 1) if future_load else 0.0
        confidence = float(predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0))
        uncertainty = float(predictions.get("prediction_uncertainty_by_vehicle", {}).get(vehicle_id, 1.0))
        planned_nodes = max(len(workflow.get("execution_order", [])), 1)
        completed_nodes = len(workflow.get("completed_node_ids", []))
        progress = float(completed_nodes) / float(planned_nodes)
        temporal_features = compute_temporal_prepare_window_score(semantic_state)
        prediction_feature_values = [
            1.0 if predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id) else 0.0,
            1.0 if predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id) else 0.0,
            float(len(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))) / 4.0,
            float(predictions.get("dwell_time", {}).get(vehicle_id, 0.0)) / 20.0,
            mean_future_load / 10.0,
            progress,
            confidence,
            uncertainty,
            float(temporal_features["countdown_norm"]),
            float(temporal_features["boundary_margin_norm"]),
            float(temporal_features["boundary_urgency"]),
            float(temporal_features["service_pressure"]),
            float(temporal_features["temporal_urgency"]),
        ]
        if self._prediction_feature_dim <= len(prediction_feature_values):
            prediction_feature_values = prediction_feature_values[: self._prediction_feature_dim]
        else:
            prediction_feature_values.extend([0.0] * (self._prediction_feature_dim - len(prediction_feature_values)))
        prediction_features = torch.tensor(prediction_feature_values, dtype=torch.float32)
        base_embedding = self._prediction_projection(prediction_features.unsqueeze(0)).squeeze(0)
        if not self._use_prediction_features:
            return (
                torch.zeros_like(base_embedding),
                torch.tensor([0.0], dtype=torch.float32, device=base_embedding.device),
                temporal_features,
            )
        if not self._use_uncertainty_signal:
            return (
                base_embedding,
                torch.tensor([1.0], dtype=torch.float32, device=base_embedding.device),
                temporal_features,
            )
        urgency_support = 0.7 + 0.3 * float(temporal_features["temporal_urgency"])
        gate_value = max(0.0, min(1.0, confidence * (1.0 - uncertainty) * urgency_support))
        # Keep a minimum leak so the surrogate branch cannot be fully muted early in training.
        gate_value = max(self._prediction_gate_min_leak, gate_value)
        gate_tensor = torch.tensor([gate_value], dtype=torch.float32, device=base_embedding.device)
        return base_embedding * gate_value, gate_tensor, temporal_features
