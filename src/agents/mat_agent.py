"""Controller-level Multi-Agent Transformer PPO baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent
from src.encoders import FlatSemanticEncoder


class _ControllerMATPolicyNetwork(nn.Module):
    """Flat semantic encoder plus transformer-coupled controller actors."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, int] = (64, 64),
        num_attention_heads: int = 4,
        transformer_layers: int = 1,
        transformer_ff_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attention_heads = max(int(num_attention_heads), 1)
        while hidden_dim % attention_heads != 0 and attention_heads > 1:
            attention_heads -= 1
        self.encoder = FlatSemanticEncoder(hidden_dim=hidden_dim)
        self.role_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=max(int(transformer_ff_dim), hidden_dim),
            dropout=max(float(dropout), 0.0),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.controller_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=max(int(transformer_layers), 1),
        )
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
        encoded = self.encoder(semantic_state)
        tokens = torch.stack(
            [
                encoded["slow_context"],
                encoded["fast_context"],
                encoded["event_context"],
            ],
            dim=0,
        )
        tokens = tokens + self.role_embedding
        transformed = self.controller_transformer(tokens.unsqueeze(0)).squeeze(0)
        pooled_context = transformed.mean(dim=0)
        temperature = max(float(event_logit_temperature if event_logit_temperature is not None else 1.0), 0.25)
        value = self.central_critic(pooled_context.unsqueeze(0)).squeeze(0).squeeze(-1)
        return {
            "encoded": encoded,
            "slow_logits": self.slow_actor(transformed[0].unsqueeze(0)).squeeze(0),
            "fast_logits": self.fast_actor(transformed[1].unsqueeze(0)).squeeze(0),
            "event_logits": self.event_actor(transformed[2].unsqueeze(0)).squeeze(0) / temperature,
            "value": value,
            "critic_mode": "controller_transformer_ctde",
            "critic_context_key": "controller_transformer_pooled_context",
            "head_values": {
                "slow": value,
                "fast": value,
                "event": value,
            },
        }


class ControllerMATAgent(PPOBaseAgent):
    """Controller-level MAT-style baseline.

    This baseline keeps the current project contract: cache, execution/offload,
    and handoff-event controllers are the agent tokens. It does not use the
    graph encoder, surrogate prediction features, uncertainty features,
    dependency-aware features, mechanism auxiliary loss, imitation, or guards.
    """

    observation_contract = "flat_semantic_controller_transformer_ctde_v1"
    action_contract = "semantic_discrete_5_controller_transformer_3head"
    support_level = "trainable"

    def __init__(
        self,
        *,
        mat_num_attention_heads: int = 4,
        mat_transformer_layers: int = 1,
        mat_transformer_ff_dim: int = 128,
        mat_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_name="controller_mat",
            policy_type="controller_mat_policy",
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
        self._mat_num_attention_heads = max(int(mat_num_attention_heads), 1)
        self._mat_transformer_layers = max(int(mat_transformer_layers), 1)
        self._mat_transformer_ff_dim = max(int(mat_transformer_ff_dim), self._hidden_dim)
        self._mat_dropout = max(float(mat_dropout), 0.0)
        self._network = _ControllerMATPolicyNetwork(
            hidden_dim=self._hidden_dim,
            hidden_dims=self._hidden_dims,
            num_attention_heads=self._mat_num_attention_heads,
            transformer_layers=self._mat_transformer_layers,
            transformer_ff_dim=self._mat_transformer_ff_dim,
            dropout=self._mat_dropout,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        self.baseline_config = {
            "family": "controller_mat",
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "controller_attention": True,
            "graph_encoder": False,
            "hierarchy": True,
            "event_head_enabled": True,
            "surrogate_enhanced_head": False,
            "centralized_critic": True,
            "ctde_scope": "controller_level_cache_execution_handoff",
            "vehicle_or_rsu_agent_ctde": False,
            "paper_grade_independent_baseline": True,
            "reference_basis": "MAT-style transformer policy adapted to the current controller-agent contract.",
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
                "mat_num_attention_heads": self._mat_num_attention_heads,
                "mat_transformer_layers": self._mat_transformer_layers,
                "mat_transformer_ff_dim": self._mat_transformer_ff_dim,
                "mat_dropout": self._mat_dropout,
            }
        )
        return config
