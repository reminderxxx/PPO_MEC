"""MAPPO baseline over the controller-level CTDE action contract."""

from __future__ import annotations

from typing import Any

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent


class MAPPOAgent(PPOBaseAgent):
    """Controller-level MAPPO baseline.

    This is not a vehicle-agent or RSU-agent MAPPO wrapper. It is the
    paper-grade baseline that matches the current project contract: three
    decentralized controller actors for cache, execution/offload, and handoff
    event decisions, trained with a centralized flat semantic critic.
    """

    observation_contract = "flat_semantic_multi_controller_ctde_v1"
    action_contract = "semantic_discrete_5_multi_controller_3head"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        head_credit_enabled = bool(kwargs.pop("head_credit_enabled", True))
        head_credit_protocol = str(
            kwargs.pop("head_credit_protocol", "aggregation_reason_weighted_controller_ppo_v3")
        )
        slow_policy_credit_floor = float(kwargs.pop("slow_policy_credit_floor", 0.25))
        fast_policy_credit_floor = float(kwargs.pop("fast_policy_credit_floor", 0.10))
        event_policy_credit_floor = float(kwargs.pop("event_policy_credit_floor", 0.12))
        slow_entropy_coef_scale = float(kwargs.pop("slow_entropy_coef_scale", 1.25))
        fast_entropy_coef_scale = float(kwargs.pop("fast_entropy_coef_scale", 1.0))
        event_entropy_coef_scale = float(kwargs.pop("event_entropy_coef_scale", 1.35))
        slow_entropy_credit_floor = float(kwargs.pop("slow_entropy_credit_floor", 0.20))
        fast_entropy_credit_floor = float(kwargs.pop("fast_entropy_credit_floor", 0.08))
        event_entropy_credit_floor = float(kwargs.pop("event_entropy_credit_floor", 0.12))
        event_advantage_blend = float(kwargs.pop("event_advantage_blend", 0.85))
        event_logit_temperature = float(kwargs.pop("event_logit_temperature", 1.0))
        event_logit_temperature_final = float(kwargs.pop("event_logit_temperature_final", 1.0))
        event_temperature_decay_updates = int(kwargs.pop("event_temperature_decay_updates", 0))
        super().__init__(
            agent_name="mappo",
            policy_type="mappo_policy",
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
            head_credit_enabled=head_credit_enabled,
            head_credit_protocol=head_credit_protocol,
            slow_policy_credit_floor=slow_policy_credit_floor,
            fast_policy_credit_floor=fast_policy_credit_floor,
            slow_entropy_coef_scale=slow_entropy_coef_scale,
            fast_entropy_coef_scale=fast_entropy_coef_scale,
            event_entropy_coef_scale=event_entropy_coef_scale,
            slow_entropy_credit_floor=slow_entropy_credit_floor,
            fast_entropy_credit_floor=fast_entropy_credit_floor,
            event_entropy_credit_floor=event_entropy_credit_floor,
            event_policy_credit_floor=event_policy_credit_floor,
            event_advantage_blend=event_advantage_blend,
            event_logit_temperature=event_logit_temperature,
            event_logit_temperature_final=event_logit_temperature_final,
            event_temperature_decay_updates=event_temperature_decay_updates,
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
        self.baseline_config = {
            "family": "mappo",
            "flat_policy": False,
            "multi_controller_ctde": True,
            "controller_agents": ["cache_agent", "execution_agent", "handoff_event_agent"],
            "graph_encoder": False,
            "hierarchy": True,
            "event_head_enabled": True,
            "controller_head_credit": head_credit_enabled,
            "head_credit_protocol": head_credit_protocol,
            "head_credit_basis": "controller_head_anti_collapse_credit_assignment_not_sa_specific",
            "controller_head_credit_floors": {
                "slow": slow_policy_credit_floor,
                "fast": fast_policy_credit_floor,
                "event": event_policy_credit_floor,
            },
            "controller_entropy_credit_floors": {
                "slow": slow_entropy_credit_floor,
                "fast": fast_entropy_credit_floor,
                "event": event_entropy_credit_floor,
            },
            "controller_entropy_scales": {
                "slow": slow_entropy_coef_scale,
                "fast": fast_entropy_coef_scale,
                "event": event_entropy_coef_scale,
            },
            "event_advantage_blend": event_advantage_blend,
            "action_mix_audit_target": "avoid_controller_head_collapse_without_sa_graph_surrogate_guard_mechanisms",
            "surrogate_enhanced_head": False,
            "centralized_critic": True,
            "centralized_critic_context": "global_semantic_flat_v1",
            "ctde_scope": "controller_level_cache_execution_handoff",
            "vehicle_or_rsu_agent_ctde": False,
            "paper_grade_independent_baseline": True,
            "contract_blocked_reason": "",
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
