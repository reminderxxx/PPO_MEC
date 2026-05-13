"""DAG 感知编码器。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _计算_frontier节点(
    nodes: list[dict[str, Any]],
    completed_node_ids: set[str],
) -> set[str]:
    frontier_node_ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_id"))
        if node_id in completed_node_ids:
            continue
        predecessors = {str(item) for item in node.get("predecessors", [])}
        if predecessors.issubset(completed_node_ids):
            frontier_node_ids.add(node_id)
    return frontier_node_ids


def _resolve_primary_vehicle_from_semantic_state(semantic_state: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(semantic_state.get("vehicles", []))
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    if primary_vehicle_id:
        primary_vehicle_id = str(primary_vehicle_id)
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == primary_vehicle_id:
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


class DAGGraphEncoder(nn.Module):
    """对 workflow DAG 做轻量 message passing 编码。"""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
        use_dependency_aware: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._hidden_dim = int(hidden_dim)
        self._message_passing_steps = int(message_passing_steps)
        self._use_dependency_aware = bool(use_dependency_aware)

        self._node_projection = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self._predecessor_update = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._successor_update = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._self_update = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._layer_norm = nn.LayerNorm(self._hidden_dim)

    def forward(self, semantic_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        workflow = semantic_state.get("workflow", {})
        nodes = list(workflow.get("nodes", []))
        edges = [tuple(edge) for edge in workflow.get("edges", [])]
        if not nodes:
            zero = torch.zeros(self._hidden_dim, dtype=torch.float32)
            return {
                "graph_embedding": zero,
                "frontier_embedding": zero,
                "current_node_embedding": zero,
                "node_embeddings": zero.unsqueeze(0),
                "node_ids": [],
            }

        node_features, node_ids = self._build_node_feature_tensor(semantic_state=semantic_state, nodes=nodes)
        node_embeddings = self._node_projection(node_features)

        if self._use_dependency_aware and edges:
            node_embeddings = self._run_message_passing(
                node_embeddings=node_embeddings,
                node_ids=node_ids,
                edges=edges,
            )

        current_node_id = workflow.get("current_node_id")
        completed_node_ids = {str(item) for item in workflow.get("completed_node_ids", [])}
        frontier_node_ids = _计算_frontier节点(nodes=nodes, completed_node_ids=completed_node_ids)

        graph_embedding = node_embeddings.mean(dim=0)
        current_node_embedding = self._pool_selected_nodes(node_embeddings, node_ids, {str(current_node_id)})[0]
        frontier_embedding = self._pool_selected_nodes(node_embeddings, node_ids, frontier_node_ids)[0]

        return {
            "graph_embedding": graph_embedding,
            "frontier_embedding": frontier_embedding,
            "current_node_embedding": current_node_embedding,
            "node_embeddings": node_embeddings,
            "node_ids": node_ids,
        }

    def _build_node_feature_tensor(
        self,
        semantic_state: dict[str, Any],
        nodes: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[str]]:
        workflow = semantic_state.get("workflow", {})
        predictions = semantic_state.get("predictions", {})
        rsus = semantic_state.get("rsus", [])
        current_node_id = str(workflow.get("current_node_id"))
        completed_node_ids = {str(item) for item in workflow.get("completed_node_ids", [])}
        frontier_node_ids = _计算_frontier节点(nodes=nodes, completed_node_ids=completed_node_ids)

        primary_vehicle = _resolve_primary_vehicle_from_semantic_state(semantic_state)
        vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
        predicted_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_next_rsu_id), {})
        handoff_target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_handoff_target_rsu_id), {})

        max_input_size = max(float(node.get("input_size", 1.0)) for node in nodes)
        max_output_size = max(float(node.get("output_size", 1.0)) for node in nodes)

        feature_rows: list[list[float]] = []
        node_ids: list[str] = []
        for node in nodes:
            node_id = str(node.get("node_id"))
            node_ids.append(node_id)
            required_adapter = node.get("required_adapter")
            feature_rows.append(
                [
                    float(node.get("input_size", 0.0)) / max(max_input_size, 1.0),
                    float(node.get("output_size", 0.0)) / max(max_output_size, 1.0),
                    float(len(node.get("predecessors", []))) / 4.0,
                    float(len(node.get("successors", []))) / 4.0,
                    1.0 if node_id == current_node_id else 0.0,
                    1.0 if node_id in completed_node_ids else 0.0,
                    1.0 if node_id in frontier_node_ids else 0.0,
                    1.0 if required_adapter in current_rsu.get("cached_adapter_ids", []) else 0.0,
                    1.0 if required_adapter in predicted_rsu.get("cached_adapter_ids", []) else 0.0,
                    1.0 if required_adapter in handoff_target_rsu.get("cached_adapter_ids", []) else 0.0,
                ]
            )
        return torch.tensor(feature_rows, dtype=torch.float32), node_ids

    def _run_message_passing(
        self,
        node_embeddings: torch.Tensor,
        node_ids: list[str],
        edges: list[tuple[str, str]],
    ) -> torch.Tensor:
        node_index_map = {node_id: index for index, node_id in enumerate(node_ids)}
        hidden = node_embeddings
        for _ in range(self._message_passing_steps):
            predecessor_aggregate = torch.zeros_like(hidden)
            predecessor_degree = torch.zeros((hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device)
            successor_aggregate = torch.zeros_like(hidden)
            successor_degree = torch.zeros((hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device)
            for source_id, target_id in edges:
                if source_id not in node_index_map or target_id not in node_index_map:
                    continue
                source_index = node_index_map[source_id]
                target_index = node_index_map[target_id]
                predecessor_aggregate[target_index] = predecessor_aggregate[target_index] + hidden[source_index]
                predecessor_degree[target_index] = predecessor_degree[target_index] + 1.0
                successor_aggregate[source_index] = successor_aggregate[source_index] + hidden[target_index]
                successor_degree[source_index] = successor_degree[source_index] + 1.0

            predecessor_mean = predecessor_aggregate / predecessor_degree.clamp_min(1.0)
            successor_mean = successor_aggregate / successor_degree.clamp_min(1.0)
            updated = torch.relu(
                self._self_update(hidden)
                + self._predecessor_update(predecessor_mean)
                + self._successor_update(successor_mean)
            )
            hidden = self._layer_norm(hidden + updated)
        return hidden

    def _pool_selected_nodes(
        self,
        node_embeddings: torch.Tensor,
        node_ids: list[str],
        selected_node_ids: set[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not selected_node_ids:
            zero = torch.zeros(self._hidden_dim, dtype=node_embeddings.dtype, device=node_embeddings.device)
            return zero, zero.unsqueeze(0)
        indices = [
            index
            for index, node_id in enumerate(node_ids)
            if node_id in selected_node_ids
        ]
        if not indices:
            zero = torch.zeros(self._hidden_dim, dtype=node_embeddings.dtype, device=node_embeddings.device)
            return zero, zero.unsqueeze(0)
        selected = node_embeddings[indices]
        return selected.mean(dim=0), selected
