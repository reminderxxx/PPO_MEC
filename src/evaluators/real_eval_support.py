"""Agent and checkpoint helpers for real-sample evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.agents.base_agent import BaseAgent
from src.agents.registry import build_agent, checkpoint_required_agents, get_algo_spec


CHECKPOINT_REQUIRED_AGENTS = checkpoint_required_agents()
SA_GHMAPPO_DETERMINISTIC_EVAL_DEFAULTS = {
    "deterministic_temporal_smoothing_enabled": True,
    "deterministic_temporal_smoothing_steps": 1,
    "deterministic_event_borderline_prob": 0.43,
    "deterministic_event_borderline_margin": -0.10,
    "deterministic_temporal_urgency_floor": 0.35,
    "deterministic_high_prepare_override_enabled": True,
    "deterministic_high_prepare_threshold": 0.55,
    "deterministic_high_urgency_threshold": 0.50,
    "deterministic_high_prepare_relaxed_margin": -0.12,
}
SA_GHMAPPO_V11_REWARD_PROFILE = "top_journal_mechanism_v11_mappo_reward"
SA_GHMAPPO_V11_REWARD_EVAL_DEFAULTS = {
    "idle_popularity_fallback_enabled": True,
    "idle_popularity_fallback_only_vehicle_fallback": True,
    "idle_popularity_prefetch_threshold": 2,
    "idle_popularity_no_rsu_local_fallback_enabled": False,
    "idle_popularity_no_rsu_local_requires_low_context": True,
}


def ensure_agent_checkpoint_path(agent_name: str, checkpoint_path: str) -> None:
    spec = get_algo_spec(agent_name)
    if spec.get("support_level") not in {"trainable", "heuristic", "diagnostic"}:
        raise NotImplementedError(f"agent={agent_name} is registered as {spec.get('support_level')}: {spec.get('notes')}")
    if agent_name not in CHECKPOINT_REQUIRED_AGENTS:
        return
    if not checkpoint_path:
        raise ValueError(f"agent={agent_name} requires checkpoint_path, but it is empty.")
    resolved = Path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"agent={agent_name} checkpoint does not exist: {resolved}")


def _load_checkpoint_config(checkpoint_path: str) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    inferred_prediction_feature_dim = _infer_prediction_feature_dim_from_payload(payload)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        config = dict(payload["config"])
    else:
        config = {}
    if inferred_prediction_feature_dim is not None:
        config.setdefault("prediction_feature_dim", inferred_prediction_feature_dim)
    return config


def _infer_checkpoint_profile(checkpoint_path: str, checkpoint_config: dict[str, Any]) -> str:
    profile = checkpoint_config.get("config_profile")
    if profile:
        return str(profile)
    train_summary_path = Path(checkpoint_path).parent.parent / "train_summary.json"
    if not train_summary_path.exists():
        return ""
    try:
        payload = json.loads(train_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("config_profile", ""))


def _infer_prediction_feature_dim_from_payload(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    state_dict = payload.get("network_state_dict")
    if not isinstance(state_dict, dict):
        return None
    weight = state_dict.get("encoder._prediction_projection.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[1])
    return None


def _filter_checkpoint_config(agent_name: str, checkpoint_config: dict[str, Any]) -> dict[str, Any]:
    common_fields = {
        "learning_rate",
        "clip_ratio",
        "entropy_coef",
        "value_coef",
        "train_epochs",
        "batch_size",
        "max_grad_norm",
        "hidden_dim",
        "hidden_dims",
        "deterministic_action",
        "target_kl",
        "kl_early_stop_enabled",
    }
    if agent_name in {
        "ippo",
        "ppo",
        "dag_offload_drl",
        "cache_offload_drl",
        "dt_handoff_drl",
    }:
        return {key: value for key, value in checkpoint_config.items() if key in common_fields}
    if agent_name == "mappo":
        mappo_fields = {
            "head_credit_enabled",
            "head_credit_protocol",
            "slow_policy_credit_floor",
            "fast_policy_credit_floor",
            "event_policy_credit_floor",
            "event_advantage_blend",
            "slow_entropy_coef_scale",
            "fast_entropy_coef_scale",
            "event_entropy_coef_scale",
            "slow_entropy_credit_floor",
            "fast_entropy_credit_floor",
            "event_entropy_credit_floor",
            "event_logit_temperature",
            "event_logit_temperature_final",
            "event_temperature_decay_updates",
        }
        return {
            key: value
            for key, value in checkpoint_config.items()
            if key in common_fields | mappo_fields
        }
    if agent_name == "controller_mat":
        mat_fields = {
            "mat_num_attention_heads",
            "mat_transformer_layers",
            "mat_transformer_ff_dim",
            "mat_dropout",
        }
        return {
            key: value
            for key, value in checkpoint_config.items()
            if key in common_fields | mat_fields
        }
    if agent_name in {"dqn", "ddqn", "dueling_dqn", "dueling_ddqn"}:
        dqn_fields = {
            "double_dqn",
            "dueling",
            "gamma",
            "replay_capacity",
            "min_replay_size",
            "target_update_interval",
            "epsilon_start",
            "epsilon_final",
            "epsilon_decay_updates",
        }
        return {
            key: value
            for key, value in checkpoint_config.items()
            if key in common_fields | dqn_fields
        }
    if agent_name == "qmix":
        qmix_fields = {
            "mixing",
            "double_qmix",
            "gamma",
            "replay_capacity",
            "min_replay_size",
            "target_update_interval",
            "epsilon_start",
            "epsilon_final",
            "epsilon_decay_updates",
            "mixer_hidden_dim",
        }
        return {
            key: value
            for key, value in checkpoint_config.items()
            if key in common_fields | qmix_fields
        }

    if agent_name != "sa_ghmappo":
        raise NotImplementedError(f"checkpoint loading is not enabled for agent={agent_name}")

    extra_fields = {
        "auxiliary_coef",
        "head_credit_enabled",
        "head_credit_protocol",
        "mechanism_logit_bias_strength",
        "mechanism_confidence_floor",
        "prediction_feature_dim",
        "prediction_gate_min_leak",
        "slow_policy_credit_floor",
        "fast_policy_credit_floor",
        "event_policy_credit_floor",
        "event_advantage_blend",
        "slow_entropy_coef_scale",
        "fast_entropy_coef_scale",
        "event_entropy_coef_scale",
        "slow_entropy_credit_floor",
        "fast_entropy_credit_floor",
        "event_entropy_credit_floor",
        "event_logit_temperature",
        "event_logit_temperature_final",
        "event_temperature_decay_updates",
        "event_logit_sharpening_final_scale",
        "event_logit_sharpening_timing_gain",
        "event_actor_loss_extra_gain",
        "event_prepare_margin_boost",
        "graph_continuity_critic_enabled",
        "uncertainty_aware_event_scaling_enabled",
        "uncertainty_aware_critic_enabled",
        "temporal_consistency_coef",
        "temporal_prepare_lead_steps",
        "temporal_prepare_sigma",
        "temporal_prepare_activation_threshold",
        "deterministic_temporal_smoothing_enabled",
        "deterministic_temporal_smoothing_steps",
        "deterministic_event_borderline_prob",
        "deterministic_event_borderline_margin",
        "deterministic_temporal_urgency_floor",
        "deterministic_high_prepare_override_enabled",
        "deterministic_high_prepare_threshold",
        "deterministic_high_urgency_threshold",
        "deterministic_high_prepare_relaxed_margin",
        "predictive_prepare_hard_override_enabled",
        "predictive_prepare_hard_override_score_threshold",
        "predictive_prepare_hard_override_confidence_threshold",
        "continuity_guard_enabled",
        "handoff_target_alignment_guard_enabled",
        "continuity_guard_logit_penalty",
        "continuity_guard_prepare_boost",
        "continuity_guard_confidence_threshold",
        "continuity_guard_prepare_score_threshold",
        "continuity_guard_hard_override_enabled",
        "heuristic_imitation_coef",
        "heuristic_imitation_warmup_updates",
        "heuristic_imitation_decay",
        "mechanism_aux_current_cache_fill_enabled",
        "event_prd_advantage_enabled",
        "event_prd_advantage_coef",
        "event_prd_advantage_clip",
        "latency_fallback_bias_enabled",
        "latency_fallback_bias_strength",
        "latency_fallback_confidence_floor",
        "latency_fallback_slow_suppression_strength",
        "steady_rsu_bias_enabled",
        "steady_rsu_bias_strength",
        "steady_rsu_confidence_floor",
        "backhaul_guard_enabled",
        "backhaul_guard_max_reactive_fills_per_adapter",
        "cache_warm_start_guard_enabled",
        "cache_warm_start_guard_min_countdown",
        "cache_warm_start_guard_max_prefetch_countdown",
        "predictive_prefetch_admission_guard_enabled",
        "predictive_prefetch_admission_min_confidence",
        "predictive_prefetch_admission_require_distinct_next",
        "idle_popularity_fallback_enabled",
        "idle_popularity_fallback_only_vehicle_fallback",
        "idle_popularity_prefetch_threshold",
        "idle_popularity_no_rsu_local_fallback_enabled",
        "idle_popularity_no_rsu_local_requires_low_context",
        "option_gate_enabled",
        "option_gate_count",
        "option_gate_loss_coef",
        "option_gate_entropy_coef",
        "option_gate_prior_coef",
        "option_gate_prior_warmup_updates",
        "option_gate_prior_decay",
        "option_gate_prior_logit_bias",
        "option_gate_log_prob_weight",
        "option_gate_context_prior_enabled",
        "option_gate_deterministic_prior_margin",
        "option_gate_idle_prior_enabled",
        "option_gate_mechanism_preserve_enabled",
        "option_gate_prd_enabled",
        "option_gate_prd_coef",
        "option_gate_prd_clip",
        "option_gate_counterfactual_prd_enabled",
        "option_gate_counterfactual_coef",
        "option_gate_counterfactual_clip",
        "net_utility_prd_enabled",
        "net_utility_backhaul_coef",
        "net_utility_migration_coef",
        "net_utility_expired_prefetch_coef",
        "net_utility_idle_prefetch_penalty",
        "net_utility_success_bonus",
        "net_utility_backhaul_normalizer",
        "net_utility_cost_dual_enabled",
        "net_utility_cost_dual_lr",
        "net_utility_cost_target",
        "net_utility_cost_dual_max",
        "net_utility_cost_dual_initial",
        "net_utility_option_termination_enabled",
        "net_utility_option_termination_conservative_enabled",
        "net_utility_option_termination_max_timing_support",
        "dag_aware_option_termination_enabled",
        "dag_aware_option_min_critical_path",
        "dag_aware_option_short_workflow_max_nodes",
        "dag_aware_option_branching_successors",
        "dag_aware_idle_prefetch_confidence_floor",
        "auxiliary_slow_weight",
        "auxiliary_fast_weight",
        "auxiliary_event_weight",
    }
    filtered = {key: value for key, value in checkpoint_config.items() if key in common_fields | extra_fields}
    filtered["use_prediction_features"] = bool(checkpoint_config.get("use_prediction_features", True))
    filtered["use_graph_encoder"] = str(checkpoint_config.get("encoder_kind", "graph")) == "graph"
    filtered["use_hierarchy"] = bool(checkpoint_config.get("use_hierarchy", True))
    filtered["use_event_agent"] = bool(checkpoint_config.get("event_head_enabled", True))
    filtered["use_adapter_prefetch"] = bool(checkpoint_config.get("adapter_prefetch_enabled", True))
    filtered["use_dependency_aware"] = bool(checkpoint_config.get("use_dependency_aware", True))
    filtered["use_uncertainty_signal"] = bool(checkpoint_config.get("use_uncertainty_signal", True))
    return filtered


def build_inference_agent(
    agent_name: str,
    random_seed: int,
    checkpoint_path: str = "",
    deterministic_action: bool = True,
    agent_config_overrides: dict[str, Any] | None = None,
) -> BaseAgent:
    ensure_agent_checkpoint_path(agent_name, checkpoint_path)
    if agent_name in CHECKPOINT_REQUIRED_AGENTS:
        raw_checkpoint_config = _load_checkpoint_config(checkpoint_path)
        checkpoint_profile = _infer_checkpoint_profile(checkpoint_path, raw_checkpoint_config)
        checkpoint_config = _filter_checkpoint_config(agent_name, raw_checkpoint_config)
    else:
        checkpoint_profile = ""
        checkpoint_config = {}
    checkpoint_config.update(
        {
            "random_seed": random_seed,
            "deterministic_action": deterministic_action,
        }
    )
    if agent_name == "sa_ghmappo" and deterministic_action:
        for key, value in SA_GHMAPPO_DETERMINISTIC_EVAL_DEFAULTS.items():
            checkpoint_config.setdefault(key, value)
    if agent_name == "sa_ghmappo" and checkpoint_profile == SA_GHMAPPO_V11_REWARD_PROFILE:
        for key, value in SA_GHMAPPO_V11_REWARD_EVAL_DEFAULTS.items():
            checkpoint_config.setdefault(key, value)
    checkpoint_config.update(dict(agent_config_overrides or {}))
    agent = build_agent(agent_name, **checkpoint_config)
    if checkpoint_path:
        agent.load(str(Path(checkpoint_path)))
    return agent
