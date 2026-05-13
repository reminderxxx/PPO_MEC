"""Dependency-aware DAG offloading PPO baseline."""

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


def _adapter_cached(rsu: dict[str, Any], adapter_id: str | None) -> float:
    if not adapter_id:
        return 0.0
    return 1.0 if str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", [])} else 0.0


def _build_dag_feature_tensor(semantic_state: dict[str, Any]) -> torch.Tensor:
    workflow = semantic_state.get("workflow", {}) or {}
    nodes = list(workflow.get("nodes", []) or [])
    current_node = semantic_state.get("current_workflow_node") or {}
    completed_node_ids = {str(item) for item in workflow.get("completed_node_ids", []) or []}
    execution_order = list(workflow.get("execution_order", []) or [])
    planned_count = max(len(execution_order), len(nodes), 1)
    progress = float(len(completed_node_ids)) / float(planned_count)
    remaining_ratio = 1.0 - progress

    frontier_count = 0
    node_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("node_id"))
        node_map[node_id] = node
        if node_id in completed_node_ids:
            continue
        predecessors = {str(item) for item in node.get("predecessors", []) or []}
        if predecessors.issubset(completed_node_ids):
            frontier_count += 1
    frontier_ratio = float(frontier_count) / float(planned_count)

    longest_cache: dict[str, int] = {}

    def remaining_path_length(node_id: str | None) -> int:
        if not node_id or node_id not in node_map or node_id in completed_node_ids:
            return 0
        if node_id in longest_cache:
            return longest_cache[node_id]
        successors = [
            str(item)
            for item in node_map[node_id].get("successors", []) or []
            if str(item) in node_map and str(item) not in completed_node_ids
        ]
        best_successor = max((remaining_path_length(successor_id) for successor_id in successors), default=0)
        longest_cache[node_id] = 1 + best_successor
        return longest_cache[node_id]

    current_node_id = str(workflow.get("current_node_id") or current_node.get("node_id") or "")
    current_path_norm = float(remaining_path_length(current_node_id)) / float(planned_count)
    critical_path_norm = 0.0
    if node_map:
        critical_path_norm = max(
            (float(remaining_path_length(node_id)) / float(planned_count) for node_id in node_map),
            default=current_path_norm,
        )

    predecessors = list(current_node.get("predecessors", []) or [])
    successors = list(current_node.get("successors", []) or [])
    max_input_size = max([float(node.get("input_size", 1.0) or 1.0) for node in nodes] or [1.0])
    max_output_size = max([float(node.get("output_size", 1.0) or 1.0) for node in nodes] or [1.0])

    primary_vehicle = _resolve_primary_vehicle(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    predictions = semantic_state.get("predictions", {}) or {}
    predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
    predicted_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    rsus = list(semantic_state.get("rsus", []) or [])
    current_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == current_rsu_id), {})
    predicted_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_next_rsu_id), {})
    target_rsu = next((rsu for rsu in rsus if rsu.get("rsu_id") == predicted_target_rsu_id), {})
    required_adapter = current_node.get("required_adapter")

    features = [
        progress,
        remaining_ratio,
        frontier_ratio,
        float(len(predecessors)) / 4.0,
        float(len(successors)) / 4.0,
        current_path_norm,
        critical_path_norm,
        float(current_node.get("input_size", 0.0) or 0.0) / max(max_input_size, 1.0),
        float(current_node.get("output_size", 0.0) or 0.0) / max(max_output_size, 1.0),
        _adapter_cached(current_rsu, required_adapter),
        _adapter_cached(predicted_rsu, required_adapter),
        _adapter_cached(target_rsu, required_adapter),
    ]
    return torch.tensor([_clamp01(item) for item in features], dtype=torch.float32)


class _DAGOffloadPolicyNetwork(nn.Module):
    """Flat semantic actor-critic augmented with DAG scalar state."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.encoder = FlatSemanticEncoder(hidden_dim=hidden_dim)
        self.dag_projection = nn.Sequential(
            nn.Linear(12, hidden_dim),
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
        dag_features = _build_dag_feature_tensor(semantic_state)
        dag_embedding = self.dag_projection(dag_features.unsqueeze(0)).squeeze(0)
        slow_context = self.slow_norm(encoded["slow_context"] + 0.50 * dag_embedding)
        fast_context = self.fast_norm(encoded["fast_context"] + dag_embedding)
        event_context = self.event_norm(encoded["event_context"] + 0.35 * dag_embedding)
        critic_context = self.critic_norm(encoded["centralized_critic_context"] + dag_embedding)
        temperature = max(float(event_logit_temperature if event_logit_temperature is not None else 1.0), 0.25)
        value = self.central_critic(critic_context.unsqueeze(0)).squeeze(0).squeeze(-1)
        encoded["encoder_mode"] = "dag_offload_domain_baseline"
        encoded["dag_progress"] = dag_features[0:1]
        encoded["dag_critical_path_norm"] = dag_features[6:7]
        return {
            "encoded": encoded,
            "slow_logits": self.slow_actor(slow_context.unsqueeze(0)).squeeze(0),
            "fast_logits": self.fast_actor(fast_context.unsqueeze(0)).squeeze(0),
            "event_logits": self.event_actor(event_context.unsqueeze(0)).squeeze(0) / temperature,
            "value": value,
            "critic_mode": "dag_offload_centralized_critic",
            "critic_context_key": "flat_semantic_plus_dag_scalars",
            "head_values": {"slow": value, "fast": value, "event": value},
        }


class DAGOffloadDRLAgent(PPOBaseAgent):
    """Dependency-aware DAG offloading learned baseline.

    The baseline uses scalar DAG workload/dependency features only. It does not
    use SA-GHMAPPO graph message passing, surrogate fusion, uncertainty-aware
    scaling, auxiliary mechanism losses, imitation, or policy guards.
    """

    observation_contract = "flat_semantic_plus_dag_scalar_offload_v1"
    action_contract = "semantic_discrete_5_dag_offload_3head"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="dag_offload_drl",
            policy_type="dag_offload_drl_policy",
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
        self._network = _DAGOffloadPolicyNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self.baseline_config = {
            "family": "dag_offload_drl",
            "domain_focus": "dependency_aware_dag_task_offloading",
            "flat_policy": False,
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "dag_scalar_features": True,
            "graph_encoder": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": True,
            "ctde_scope": "controller_level_cache_execution_handoff",
            "paper_grade_independent_baseline": True,
            "reference_basis": "Dependency-aware DRL offloading literature adapted to the current VEC contract.",
            "excluded_sa_mechanisms": [
                "graph_message_passing_encoder",
                "surrogate_prediction_features",
                "uncertainty_signal",
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
                "domain_feature_block": "dag_scalar_offload",
            }
        )
        return config
