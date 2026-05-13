"""RSU 状态编码器。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _resolve_primary_vehicle_from_semantic_state(semantic_state: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(semantic_state.get("vehicles", []))
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    if primary_vehicle_id:
        primary_vehicle_id = str(primary_vehicle_id)
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == primary_vehicle_id:
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


class RSUStateEncoder(nn.Module):
    """编码 RSU 集合、cache 状态与预测负载。"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._hidden_dim = int(hidden_dim)
        self._rsu_projection = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )

    def forward(self, semantic_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        rsus = list(semantic_state.get("rsus", []))
        if not rsus:
            zero = torch.zeros(self._hidden_dim, dtype=torch.float32)
            return {
                "set_embedding": zero,
                "current_rsu_embedding": zero,
                "predicted_rsu_embedding": zero,
                "target_rsu_embedding": zero,
                "rsu_embeddings": zero.unsqueeze(0),
                "rsu_ids": [],
            }

        feature_tensor, rsu_ids = self._build_rsu_feature_tensor(semantic_state=semantic_state, rsus=rsus)
        rsu_embeddings = self._rsu_projection(feature_tensor)
        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        predictions = semantic_state.get("predictions", {})
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)

        return {
            "set_embedding": rsu_embeddings.mean(dim=0),
            "current_rsu_embedding": self._select_embedding(rsu_embeddings, rsu_ids, current_rsu_id),
            "predicted_rsu_embedding": self._select_embedding(rsu_embeddings, rsu_ids, predicted_next_rsu_id),
            "target_rsu_embedding": self._select_embedding(rsu_embeddings, rsu_ids, predicted_handoff_target_rsu_id),
            "rsu_embeddings": rsu_embeddings,
            "rsu_ids": rsu_ids,
        }

    def _build_rsu_feature_tensor(
        self,
        semantic_state: dict[str, Any],
        rsus: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[str]]:
        predictions = semantic_state.get("predictions", {})
        current_node = semantic_state.get("current_workflow_node") or {}
        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        future_load = predictions.get("future_load", {})
        demand_scores = predictions.get("cache_demand", {}).get("demand_score_by_rsu", {})
        required_adapter = current_node.get("required_adapter")
        mean_future_load = 0.0
        if future_load:
            mean_future_load = sum(float(value) for value in future_load.values()) / max(len(future_load), 1)

        feature_rows: list[list[float]] = []
        rsu_ids: list[str] = []
        for rsu in rsus:
            rsu_id = str(rsu.get("rsu_id"))
            rsu_ids.append(rsu_id)
            rsu_future_load = float(future_load.get(rsu_id, 0.0))
            demand_score = 0.0
            if required_adapter is not None:
                demand_score = float(demand_scores.get(rsu_id, {}).get(required_adapter, 0.0))
            feature_rows.append(
                [
                    float(rsu.get("coverage_radius", 0.0)) / 100.0,
                    float(len(rsu.get("active_vehicle_ids", []))) / 10.0,
                    float(len(rsu.get("cached_adapter_ids", []))) / 10.0,
                    1.0 if rsu_id == current_rsu_id else 0.0,
                    1.0 if rsu_id == predicted_next_rsu_id else 0.0,
                    1.0 if rsu_id == predicted_handoff_target_rsu_id else 0.0,
                    rsu_future_load / 10.0,
                    mean_future_load / 10.0,
                    demand_score / 5.0,
                    1.0 if required_adapter in rsu.get("cached_adapter_ids", []) else 0.0,
                ]
            )
        return torch.tensor(feature_rows, dtype=torch.float32), rsu_ids

    def _select_embedding(
        self,
        rsu_embeddings: torch.Tensor,
        rsu_ids: list[str],
        target_rsu_id: str | None,
    ) -> torch.Tensor:
        if target_rsu_id is None:
            return torch.zeros(self._hidden_dim, dtype=rsu_embeddings.dtype, device=rsu_embeddings.device)
        for index, rsu_id in enumerate(rsu_ids):
            if rsu_id == target_rsu_id:
                return rsu_embeddings[index]
        return torch.zeros(self._hidden_dim, dtype=rsu_embeddings.dtype, device=rsu_embeddings.device)
