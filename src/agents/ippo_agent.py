"""Diagnostic IPPO placeholder over the shared semantic action contract."""

from __future__ import annotations

from typing import Any

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent


class IPPOAgent(PPOBaseAgent):
    """Diagnostic IPPO-style PPO variant.

    The current wrapper exposes one shared decision stream. That is not enough
    to instantiate a paper-grade independent multi-agent PPO baseline, so this
    class is kept only for artifact compatibility and diagnostic reruns.
    """

    observation_contract = "flat_semantic_encoder_v1"
    action_contract = "semantic_discrete_5"
    support_level = "diagnostic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="ippo",
            policy_type="ippo_policy",
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
            head_credit_enabled=False,
            mechanism_logit_bias_strength=0.0,
            mechanism_confidence_floor=0.0,
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
            continuity_guard_enabled=False,
            handoff_target_alignment_guard_enabled=False,
            heuristic_imitation_coef=0.0,
            mechanism_aux_coef=0.0,
            mechanism_window_weight=1.0,
            prepare_action_prior_weight=0.0,
            mechanism_entropy_coef=0.0,
            **kwargs,
        )
        self.baseline_config = {
            "family": "ippo",
            "minimal_viable_ippo": False,
            "paper_grade_independent_baseline": False,
            "contract_blocked_reason": "single_wrapper_decision_stream_has_no_independent_per_agent_action_surface",
            "independent_policy_execution": True,
            "single_wrapper_decision_stream": True,
            "graph_encoder": False,
            "hierarchy": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": False,
            "uses_sa_mechanism_bias": False,
        }
