"""训练 SA-GHMAPPO 主方法。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent
from src.encoders.fusion_encoder import compute_temporal_prepare_window_score
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    apply_frozen_window_plan,
    classify_experiment_scale,
    build_selected_workflow_states,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    load_checkpoint_metadata,
    load_window_bundle,
    resolve_agent_checkpoint,
    resolve_window_candidates,
)
from src.evaluators.real_eval_support import build_inference_agent
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


EVAL_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
    "validated_predictive_prefetch_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "handoff_total_count",
    "prefetch_request_count",
    "prefetch_request_rate",
    "prefetch_validated_hit_count",
    "prefetch_validated_hit_rate",
    "migration_prepare_count",
    "migration_prepare_rate",
    "handoff_ready_rate",
    "mechanism_realization_rate",
    "mechanism_success_rate",
]
MECHANISM_DIAGNOSTIC_FIELDS = [
    "prefetch_request_count",
    "prefetch_request_rate",
    "prefetch_validated_hit_count",
    "prefetch_validated_hit_rate",
    "mechanism_attempt_count",
    "mechanism_validated_success_count",
    "mechanism_success_rate",
    "mechanism_pending_success_count",
    "migration_prepare_count",
    "migration_prepare_rate",
    "handoff_total_count",
    "handoff_ready_rate",
    "mechanism_realization_rate",
]
REWARD_BREAKDOWN_FIELDS = [
    "service_reward",
    "delay_penalty",
    "cache_miss_penalty",
    "migration_cost",
    "continuity_bonus",
    "mechanism_exploration_bonus",
]
POLICY_DIAGNOSTIC_FIELDS = [
    "first_vehicle_matches_primary_rate",
    "policy_current_rsu_non_null_rate",
    "gt_current_rsu_non_null_rate",
    "primary_vehicle_lookup_fallback_rate",
    "stochastic_event_prepare_rate",
    "deterministic_event_prepare_rate",
    "gap_event_prepare_rate",
    "gt_handoff_opportunity_step_count",
    "gt_handoff_opportunity_rate",
    "gt_first_handoff_eta_mean",
    "gt_first_handoff_eta_p25",
    "gt_first_handoff_eta_p75",
    "gt_first_next_rsu_non_null_step_count",
    "predictor_invoked_step_count",
    "predictor_invoked_rate",
    "raw_handoff_candidate_step_count",
    "raw_handoff_candidate_rate",
    "pred_first_non_current_rsu_step_count",
    "pred_first_non_current_rsu_rate",
    "pred_first_non_current_rsu_eta_mean",
    "pred_first_non_current_rsu_eta_p25",
    "pred_first_non_current_rsu_eta_p75",
    "predicted_sequence_all_null_count",
    "predicted_sequence_all_current_rsu_count",
    "predicted_sequence_contains_other_rsu_count",
    "gt_pred_next_rsu_match_count",
    "gt_pred_next_rsu_mismatch_count",
    "prediction_confidence_mean",
    "prediction_confidence_p75",
    "prediction_uncertainty_mean",
    "prediction_uncertainty_p25",
    "prediction_uncertainty_p75",
    "predictor_brier_score_proxy_mean",
    "predictor_calibration_error_proxy_mean",
    "predictor_handoff_target_precision_proxy_mean",
    "predictor_handoff_target_recall_proxy_mean",
    "urgency_support_mean",
    "urgency_support_p75",
    "prediction_gate_value_mean",
    "prediction_gate_value_p75",
    "gate_pass_step_count",
    "gate_pass_rate",
    "candidate_block_reason_no_next_rsu_count",
    "candidate_block_reason_same_rsu_count",
    "candidate_block_reason_no_eta_count",
    "candidate_block_reason_eta_outside_window_count",
    "candidate_block_reason_low_handoff_risk_count",
    "candidate_block_reason_short_target_dwell_count",
    "candidate_block_reason_missing_prediction_state_count",
    "invalid_reason_no_candidate_count",
    "invalid_reason_low_confidence_count",
    "invalid_reason_high_uncertainty_count",
    "invalid_reason_gate_below_threshold_count",
    "valid_handoff_target_step_count",
    "valid_handoff_target_rate",
    "timing_active_step_count",
    "temporal_urgency_mean",
    "temporal_urgency_p75",
    "countdown_steps_mean",
    "countdown_steps_p25",
    "countdown_steps_p75",
    "prepare_window_score_mean",
    "prepare_window_score_p75",
    "high_prepare_step_count",
    "prepare_window_score_mean_on_valid_target",
    "prepare_window_score_p75_on_valid_target",
    "event_prepare_prob_mean",
    "event_prepare_prob_p75",
    "event_margin_mean",
    "event_margin_p75",
    "stochastic_event_prepare_rate_on_valid_target",
    "deterministic_event_prepare_rate_on_valid_target",
    "stochastic_event_prepare_rate_on_timing_active",
    "deterministic_event_prepare_rate_on_timing_active",
    "borderline_trigger_count",
    "override_trigger_count",
    "deterministic_temporal_smoothing_rate",
    "continuity_guard_trigger_count",
    "continuity_guard_trigger_rate",
    "target_mismatch_guard_count",
    "guard_prefetch_to_prepare_count",
    "guard_hard_override_count",
    "action_projection_count",
    "action_projection_rate",
    "invalid_action_attempt_count",
    "invalid_action_attempt_rate",
    "guard_action_delta_count",
    "guard_action_delta_rate",
    "backhaul_guard_count",
    "backhaul_guard_rate",
    "backhaul_aware_policy_adjust_count",
    "backhaul_aware_policy_adjust_rate",
    "backhaul_aware_service_fill_bias_mean",
    "backhaul_aware_redundant_fill_penalty_mean",
    "backhaul_aware_no_signal_penalty_mean",
    "cache_warm_start_guard_count",
    "cache_warm_start_guard_rate",
    "predictive_prefetch_admission_guard_count",
    "predictive_prefetch_admission_guard_rate",
    "coverage_recovery_final_guard_count",
    "coverage_recovery_final_guard_rate",
    "dag_frontier_size_mean",
    "dag_critical_path_pressure_mean",
    "dag_current_node_dependency_pressure_mean",
    "dag_remaining_nodes_mean",
]
PROFILE_DEFAULTS = {
    "smoke": {
        "episodes": 4,
        "update_every": 2,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "clip_ratio": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "auxiliary_coef": 0.05,
        "max_steps": 6,
        "train_window_count": 1,
    },
    "baseline_safe": {
        "episodes": 24,
        "update_every": 4,
        "batch_size": 24,
        "learning_rate": 1e-4,
        "clip_ratio": 0.1,
        "entropy_coef": 0.003,
        "value_coef": 0.7,
        "auxiliary_coef": 0.1,
        "max_steps": 12,
        "train_window_count": 2,
    },
    "formal_main": {
        "episodes": 48,
        "update_every": 6,
        "batch_size": 32,
        "learning_rate": 8e-5,
        "clip_ratio": 0.1,
        "entropy_coef": 0.0015,
        "value_coef": 0.8,
        "auxiliary_coef": 0.2,
        "max_steps": 14,
        "train_window_count": 3,
    },
    "formal_main_stable": {
        "episodes": 48,
        "update_every": 6,
        "batch_size": 32,
        "learning_rate": 6e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.001,
        "value_coef": 0.85,
        "auxiliary_coef": 0.24,
        "max_steps": 14,
        "train_window_count": 3,
    },
    "sa_advantage_round1": {
        "episodes": 64,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 6e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.001,
        "value_coef": 0.85,
        "auxiliary_coef": 0.28,
        "max_steps": 14,
        "train_window_count": 4,
    },
    "sa_mechanism_policy_round2": {
        "episodes": 64,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 6e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.001,
        "value_coef": 0.85,
        "auxiliary_coef": 0.28,
        "max_steps": 14,
        "train_window_count": 4,
    },
    "sa_mechanism_retention_round3": {
        "episodes": 64,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 6e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.001,
        "value_coef": 0.85,
        "auxiliary_coef": 0.28,
        "max_steps": 14,
        "train_window_count": 4,
    },
    "top_journal_mechanism_v1": {
        "episodes": 96,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.30,
        "max_steps": 16,
        "train_window_count": 5,
    },
    "top_journal_mechanism_v2": {
        "episodes": 96,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 5,
    },
    "top_journal_mechanism_v3": {
        "episodes": 96,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.30,
        "max_steps": 16,
        "train_window_count": 5,
    },
    "top_journal_mechanism_v5_perf_robust": {
        "episodes": 96,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "clip_ratio": 0.08,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.30,
        "max_steps": 16,
        "train_window_count": 5,
    },
    "top_journal_mechanism_v6_strong_competition": {
        "episodes": 128,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 4.5e-5,
        "clip_ratio": 0.075,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 6,
    },
    "top_journal_mechanism_v7_latency_fallback": {
        "episodes": 128,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 4.5e-5,
        "clip_ratio": 0.075,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 6,
    },
    "top_journal_mechanism_v8_strict_full": {
        "episodes": 96,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 4.5e-5,
        "clip_ratio": 0.075,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v9_pareto_safe": {
        "episodes": 96,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 4.0e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0010,
        "value_coef": 0.85,
        "auxiliary_coef": 0.30,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v10_mappo_rl": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 4.0e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.28,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v11_mappo_reward": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 4.2e-5,
        "clip_ratio": 0.075,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v12_learned_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 4.0e-5,
        "clip_ratio": 0.075,
        "entropy_coef": 0.0012,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v13_prd_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.5e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v14_net_utility_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.2e-5,
        "clip_ratio": 0.065,
        "entropy_coef": 0.0010,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v15_terminal_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.5e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v16_conservative_terminal_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.5e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v17_dag_aware_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.5e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v18_counterfactual_option": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.4e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v19_handoff_risk_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.45e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v20_idle_execution_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.45e-5,
        "clip_ratio": 0.07,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v21_efficiency_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.35e-5,
        "clip_ratio": 0.065,
        "entropy_coef": 0.0011,
        "value_coef": 0.85,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v22_validated_utility_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.25e-5,
        "clip_ratio": 0.06,
        "entropy_coef": 0.001,
        "value_coef": 0.88,
        "auxiliary_coef": 0.34,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v23_counterfactual_constrained_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.30e-5,
        "clip_ratio": 0.06,
        "entropy_coef": 0.001,
        "value_coef": 0.88,
        "auxiliary_coef": 0.34,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v24_tail_risk_constrained_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.20e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.0009,
        "value_coef": 0.90,
        "auxiliary_coef": 0.34,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v25_opportunity_risk_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.25e-5,
        "clip_ratio": 0.060,
        "entropy_coef": 0.0010,
        "value_coef": 0.90,
        "auxiliary_coef": 0.36,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.25e-5,
        "clip_ratio": 0.060,
        "entropy_coef": 0.0010,
        "value_coef": 0.88,
        "auxiliary_coef": 0.34,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v27_conservative_advantage_imitation": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.40e-5,
        "clip_ratio": 0.065,
        "entropy_coef": 0.0010,
        "value_coef": 0.86,
        "auxiliary_coef": 0.32,
        "max_steps": 16,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v28_credit_focal_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.35e-5,
        "clip_ratio": 0.065,
        "entropy_coef": 0.0012,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v29_dt_fused_credit_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.30e-5,
        "clip_ratio": 0.065,
        "entropy_coef": 0.0012,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v30_dt_prior_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.25e-5,
        "clip_ratio": 0.060,
        "entropy_coef": 0.0011,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v31_handoff_pacing_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.20e-5,
        "clip_ratio": 0.058,
        "entropy_coef": 0.0010,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v32_dt_continuation_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.15e-5,
        "clip_ratio": 0.056,
        "entropy_coef": 0.0010,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v33_env_action_ppo_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.10e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.0011,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v34_adaptive_wait_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.08e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00115,
        "value_coef": 0.86,
        "auxiliary_coef": 0.34,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v35_guard_relaxed_action_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.05e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00125,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v36_counterfactual_margin_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.00e-5,
        "clip_ratio": 0.052,
        "entropy_coef": 0.00135,
        "value_coef": 0.86,
        "auxiliary_coef": 0.28,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v37_advantage_gated_counterfactual_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.04e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00125,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v38_undiscounted_dt_prior_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.08e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00135,
        "value_coef": 0.88,
        "auxiliary_coef": 0.30,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v39_delayed_credit_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.04e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00130,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v40_advantage_weighted_behavior_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.02e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00135,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v41_conservative_recovery_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.03e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00130,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "max_steps": 22,
        "train_window_count": 20,
    },
    "top_journal_mechanism_v42_completion_aligned_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 3.0e-5,
        "clip_ratio": 0.055,
        "entropy_coef": 0.00130,
        "value_coef": 0.86,
        "auxiliary_coef": 0.30,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v43_strict_opportunity_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.8e-5,
        "clip_ratio": 0.05,
        "entropy_coef": 0.00115,
        "value_coef": 0.88,
        "auxiliary_coef": 0.26,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v44_opportunity_constrained_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.6e-5,
        "clip_ratio": 0.05,
        "entropy_coef": 0.00105,
        "value_coef": 0.88,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v45_balanced_refresh_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.6e-5,
        "clip_ratio": 0.05,
        "entropy_coef": 0.00108,
        "value_coef": 0.88,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v46_net_utility_constrained_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.55e-5,
        "clip_ratio": 0.048,
        "entropy_coef": 0.00105,
        "value_coef": 0.90,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v47_service_backhaul_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00102,
        "value_coef": 0.90,
        "auxiliary_coef": 0.23,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v48_service_fill_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.50e-5,
        "clip_ratio": 0.048,
        "entropy_coef": 0.00104,
        "value_coef": 0.90,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
    },
    "top_journal_mechanism_v49_retrospective_handoff_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.55e-5,
        "clip_ratio": 0.048,
        "entropy_coef": 0.00106,
        "value_coef": 0.90,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 8,
    },
    "top_journal_mechanism_v50_long_horizon_handoff_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.55e-5,
        "clip_ratio": 0.048,
        "entropy_coef": 0.00106,
        "value_coef": 0.90,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
    },
    "top_journal_mechanism_v51_physical_transfer_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.55e-5,
        "clip_ratio": 0.048,
        "entropy_coef": 0.00106,
        "value_coef": 0.90,
        "auxiliary_coef": 0.24,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
    },
    "top_journal_mechanism_v52_net_advantage_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.50e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00108,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
    },
    "top_journal_mechanism_v53_service_net_advantage_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
    },
    "top_journal_mechanism_v54_service_completion_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
    },
    "top_journal_mechanism_v55_coverage_recovery_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v56_partial_observation_handoff_memory_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v57_service_continuity_counterfactual_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v58_ready_aware_counterfactual_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v59_sparse_recovery_curriculum_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.45e-5,
        "clip_ratio": 0.046,
        "entropy_coef": 0.00110,
        "value_coef": 0.90,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v60_risk_adjusted_recovery_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.42e-5,
        "clip_ratio": 0.044,
        "entropy_coef": 0.00105,
        "value_coef": 0.92,
        "auxiliary_coef": 0.21,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.40e-5,
        "clip_ratio": 0.042,
        "entropy_coef": 0.00100,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v62_cache_feasibility_prior_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.22e-5,
        "clip_ratio": 0.040,
        "entropy_coef": 0.00095,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v63_current_ready_two_stage_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.18e-5,
        "clip_ratio": 0.040,
        "entropy_coef": 0.00090,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v64_handoff_alignment_barrier_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.12e-5,
        "clip_ratio": 0.038,
        "entropy_coef": 0.00086,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v65_argmax_margin_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 2.08e-5,
        "clip_ratio": 0.036,
        "entropy_coef": 0.00105,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v66_sparse_safe_cpi_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 1.88e-5,
        "clip_ratio": 0.030,
        "entropy_coef": 0.00092,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v67_sparse_handoff_recovery_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 1.86e-5,
        "clip_ratio": 0.030,
        "entropy_coef": 0.00094,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v68_actor_native_sparse_recovery_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 1.84e-5,
        "clip_ratio": 0.030,
        "entropy_coef": 0.00096,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v69_realization_credit_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 1.84e-5,
        "clip_ratio": 0.030,
        "entropy_coef": 0.00098,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v70_sparse_tail_option_mappo": {
        "episodes": 128,
        "update_every": 8,
        "batch_size": 32,
        "learning_rate": 1.84e-5,
        "clip_ratio": 0.030,
        "entropy_coef": 0.00102,
        "value_coef": 0.94,
        "auxiliary_coef": 0.22,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
    },
    "top_journal_mechanism_v71_tail_counterfactual_option_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v72_digital_twin_counterfactual_option_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v73_multistep_counterfactual_option_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v74_model_based_env_action_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v75_conservative_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v76_recurrent_robust_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v77_temporal_downside_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v78_regret_adaptive_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v79_tail_distilled_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v80_imagination_replay_model_policy_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v81_trust_region_imagination_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v82_recovery_imagination_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v83_recovery_residual_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v84_search_distilled_residual_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v85_trust_region_residual_search_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v86_recovery_beam_replay_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v87_mirror_residual_projection_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v88_contextual_recovery_expert_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v89_diverse_recovery_branch_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v90_causally_scoped_recovery_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v91_balanced_recovery_distillation_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v92_online_counterfactual_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v93_mechanism_aware_online_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v94_ucc_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v95_ucc_risk_calibrated_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v96_ucc_ucb_exploration_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v97_ucc_counterfactual_calibration_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v98_ucc_counterfactual_policy_improvement_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "top_journal_mechanism_v99_adaptive_resource_mappo": {
        "episodes": 256,
        "update_every": 8,
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "clip_ratio": 0.12,
        "entropy_coef": 0.004,
        "value_coef": 0.70,
        "auxiliary_coef": 0.06,
        "max_steps": 22,
        "train_window_count": 20,
        "window_mode": "full_stratified",
        "train_window_mode": "rotate",
        "primary_vehicle_selection": "handoff_pressure",
        "reward_positive_offset": 0.0,
        "prediction_horizon": 16,
        "post_training_audit_mode": "compact",
        "temporal_reward_shaping_enabled": False,
        "update_eval_max_windows": 4,
        "update_eval_max_workflows": 1,
    },
    "sa_reward_tiebreak_round4": {
        "episodes": 16,
        "update_every": 4,
        "batch_size": 32,
        "learning_rate": 3e-5,
        "clip_ratio": 0.06,
        "entropy_coef": 0.0008,
        "value_coef": 0.85,
        "auxiliary_coef": 0.18,
        "max_steps": 14,
        "train_window_count": 4,
    },
}

MECHANISM_COVERAGE_PROFILES = {
    "top_journal_mechanism_v8_strict_full",
    "top_journal_mechanism_v9_pareto_safe",
    "top_journal_mechanism_v10_mappo_rl",
    "top_journal_mechanism_v11_mappo_reward",
    "top_journal_mechanism_v12_learned_option",
    "top_journal_mechanism_v13_stratified_full_pool",
    "top_journal_mechanism_v14_reward_gate",
    "top_journal_mechanism_v15_mappo_credit",
    "top_journal_mechanism_v16_timing_credit",
    "top_journal_mechanism_v17_predictor_aligned",
    "top_journal_mechanism_v18_oracle_boundary",
    "top_journal_mechanism_v19_reward_credit",
    "top_journal_mechanism_v20_mechanism_retention",
    "top_journal_mechanism_v21_event_credit",
    "top_journal_mechanism_v22_pretrain_event",
    "top_journal_mechanism_v23_predictor_calibrated",
    "top_journal_mechanism_v24_target_alignment",
    "top_journal_mechanism_v25_mappo_advantage",
    "top_journal_mechanism_v26_stability_repair",
    "top_journal_mechanism_v27_replay_stabilized",
    "top_journal_mechanism_v28_credit_shaping",
    "top_journal_mechanism_v29_predictor_policy_fusion",
    "top_journal_mechanism_v30_counterfactual_credit",
    "top_journal_mechanism_v31_mechanism_distillation",
    "top_journal_mechanism_v32_actor_regularized",
    "top_journal_mechanism_v33_prediction_trust_region",
    "top_journal_mechanism_v34_handoff_timing",
    "top_journal_mechanism_v35_transition_credit",
    "top_journal_mechanism_v36_advantage_safe",
    "top_journal_mechanism_v37_mechanism_locked",
    "top_journal_mechanism_v38_mechanism_oracle_gap",
    "top_journal_mechanism_v39_mappo_timing_bridge",
    "top_journal_mechanism_v40_timing_reward",
    "top_journal_mechanism_v41_credit_stable",
    "top_journal_mechanism_v42_mechanism_ready",
    "top_journal_mechanism_v43_adaptive_option",
    "top_journal_mechanism_v44_backhaul_margin",
    "top_journal_mechanism_v45_balanced_margin",
    "top_journal_mechanism_v46_net_utility_constrained_mappo",
    "top_journal_mechanism_v47_service_backhaul_mappo",
    "top_journal_mechanism_v48_service_fill_mappo",
    "top_journal_mechanism_v49_retrospective_handoff_mappo",
    "top_journal_mechanism_v50_long_horizon_handoff_mappo",
    "top_journal_mechanism_v51_physical_transfer_mappo",
    "top_journal_mechanism_v52_net_advantage_mappo",
    "top_journal_mechanism_v53_service_net_advantage_mappo",
    "top_journal_mechanism_v54_service_completion_mappo",
    "top_journal_mechanism_v55_coverage_recovery_mappo",
    "top_journal_mechanism_v56_partial_observation_handoff_memory_mappo",
    "top_journal_mechanism_v57_service_continuity_counterfactual_mappo",
    "top_journal_mechanism_v58_ready_aware_counterfactual_mappo",
    "top_journal_mechanism_v59_sparse_recovery_curriculum_mappo",
    "top_journal_mechanism_v60_risk_adjusted_recovery_mappo",
    "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo",
    "top_journal_mechanism_v62_cache_feasibility_prior_mappo",
    "top_journal_mechanism_v63_current_ready_two_stage_mappo",
    "top_journal_mechanism_v64_handoff_alignment_barrier_mappo",
    "top_journal_mechanism_v65_argmax_margin_mappo",
    "top_journal_mechanism_v66_sparse_safe_cpi_mappo",
    "top_journal_mechanism_v67_sparse_handoff_recovery_mappo",
    "top_journal_mechanism_v68_actor_native_sparse_recovery_mappo",
    "top_journal_mechanism_v69_realization_credit_mappo",
    "top_journal_mechanism_v70_sparse_tail_option_mappo",
    "top_journal_mechanism_v71_tail_counterfactual_option_mappo",
    "top_journal_mechanism_v72_digital_twin_counterfactual_option_mappo",
    "top_journal_mechanism_v73_multistep_counterfactual_option_mappo",
    "top_journal_mechanism_v74_model_based_env_action_mappo",
    "top_journal_mechanism_v75_conservative_model_policy_mappo",
    "top_journal_mechanism_v76_recurrent_robust_model_policy_mappo",
    "top_journal_mechanism_v77_temporal_downside_model_policy_mappo",
    "top_journal_mechanism_v78_regret_adaptive_model_policy_mappo",
    "top_journal_mechanism_v79_tail_distilled_model_policy_mappo",
    "top_journal_mechanism_v80_imagination_replay_model_policy_mappo",
    "top_journal_mechanism_v81_trust_region_imagination_mappo",
    "top_journal_mechanism_v82_recovery_imagination_mappo",
    "top_journal_mechanism_v83_recovery_residual_mappo",
    "top_journal_mechanism_v84_search_distilled_residual_mappo",
    "top_journal_mechanism_v85_trust_region_residual_search_mappo",
    "top_journal_mechanism_v86_recovery_beam_replay_mappo",
    "top_journal_mechanism_v87_mirror_residual_projection_mappo",
    "top_journal_mechanism_v88_contextual_recovery_expert_mappo",
    "top_journal_mechanism_v89_diverse_recovery_branch_mappo",
    "top_journal_mechanism_v90_causally_scoped_recovery_mappo",
    "top_journal_mechanism_v91_balanced_recovery_distillation_mappo",
    "top_journal_mechanism_v92_online_counterfactual_mappo",
    "top_journal_mechanism_v93_mechanism_aware_online_mappo",
    "top_journal_mechanism_v94_ucc_mappo",
    "top_journal_mechanism_v95_ucc_risk_calibrated_mappo",
    "top_journal_mechanism_v96_ucc_ucb_exploration_mappo",
    "top_journal_mechanism_v97_ucc_counterfactual_calibration_mappo",
    "top_journal_mechanism_v98_ucc_counterfactual_policy_improvement_mappo",
    "top_journal_mechanism_v99_adaptive_resource_mappo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 SA-GHMAPPO 主方法")
    parser.add_argument("--agent_name", type=str, default="sa_ghmappo", choices=["sa_ghmappo"])
    parser.add_argument("--profile", type=str, default="baseline_safe", choices=sorted(PROFILE_DEFAULTS))
    parser.add_argument("--train_window_mode", type=str, default="rotate", choices=["fixed", "rotate", "sampled"])
    parser.add_argument("--train_window_count", type=int, default=None)
    parser.add_argument("--window_mode", type=str, default="activating_only", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--window_plan_path", type=str, default="")
    parser.add_argument("--eval_window_plan_path", type=str, default="")
    parser.add_argument("--mobility_source", type=str, default="ngsim", choices=["ngsim", "lust"])
    parser.add_argument(
        "--primary_vehicle_selection",
        type=str,
        default=None,
        choices=["stable_first", "handoff_pressure"],
    )
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reward_positive_offset", type=float, default=None)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate", choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--update_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--value_coef", type=float, default=None)
    parser.add_argument("--auxiliary_coef", type=float, default=None)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--smoke_run", action="store_true")
    parser.add_argument("--disable_prediction", action="store_true")
    parser.add_argument("--disable_graph_encoder", action="store_true")
    parser.add_argument("--disable_hierarchy", action="store_true")
    parser.add_argument("--disable_event_agent", action="store_true")
    parser.add_argument("--disable_adapter_prefetch", action="store_true")
    parser.add_argument("--disable_dag_dependency_aware", action="store_true")
    parser.add_argument("--disable_uncertainty_signal", action="store_true")
    parser.add_argument("--latency_fallback_bias_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--steady_rsu_bias_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--steady_rsu_bias_strength", type=float, default=None)
    parser.add_argument("--steady_rsu_confidence_floor", type=float, default=None)
    parser.add_argument("--prediction_horizon", type=int, default=None)
    parser.add_argument("--prediction_noise_std", type=float, default=0.0)
    parser.add_argument("--prediction_confidence_scale", type=float, default=1.0)
    parser.add_argument("--prediction_delay_steps", type=int, default=0)
    parser.add_argument("--drop_handoff_prediction_prob", type=float, default=0.0)
    parser.add_argument(
        "--temporal_reward_shaping_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--predictor_kind", type=str, default="baseline", choices=["baseline", "oracle", "learned_or_calibrated", "supervised"])
    parser.add_argument("--predictor_checkpoint_path", type=str, default="")
    parser.add_argument("--continuity_guard_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--handoff_target_alignment_guard_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continuity_guard_logit_penalty", type=float, default=None)
    parser.add_argument("--continuity_guard_prepare_boost", type=float, default=None)
    parser.add_argument("--continuity_guard_confidence_threshold", type=float, default=None)
    parser.add_argument("--continuity_guard_prepare_score_threshold", type=float, default=None)
    parser.add_argument("--continuity_guard_hard_override_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--heuristic_imitation_coef", type=float, default=None)
    parser.add_argument("--heuristic_imitation_warmup_updates", type=int, default=None)
    parser.add_argument("--heuristic_imitation_decay", type=float, default=None)
    parser.add_argument("--mechanism_aux_coef", type=float, default=None)
    parser.add_argument("--mechanism_window_weight", type=float, default=None)
    parser.add_argument("--prepare_action_prior_weight", type=float, default=None)
    parser.add_argument("--mechanism_entropy_coef", type=float, default=None)
    parser.add_argument("--mechanism_retention_start_update", type=int, default=None)
    parser.add_argument("--mechanism_aux_coef_floor_after_update", type=float, default=None)
    parser.add_argument("--mechanism_window_weight_floor_after_update", type=float, default=None)
    parser.add_argument("--mechanism_entropy_floor_after_update", type=float, default=None)
    parser.add_argument("--cache_warm_start_guard_max_prefetch_countdown", type=float, default=None)
    parser.add_argument("--predictive_prefetch_admission_guard_enabled", action="store_true", default=None)
    parser.add_argument("--predictive_prefetch_admission_min_confidence", type=float, default=None)
    parser.add_argument("--predictive_prefetch_admission_require_distinct_next", type=int, choices=[0, 1], default=None)
    parser.add_argument("--mechanism_window_oversample_ratio", type=float, default=1.0)
    parser.add_argument("--handoff_imminent_oversample_ratio", type=float, default=1.0)
    parser.add_argument("--target_mismatch_sample_weight", type=float, default=1.0)
    parser.add_argument("--min_mechanism_activating_windows", type=int, default=0)
    parser.add_argument("--warm_start_checkpoint_path", type=str, default="")
    parser.add_argument("--audit_update_checkpoints", action="store_true")
    parser.add_argument("--post_training_audit_mode", choices=["full", "compact"], default=None)
    parser.add_argument("--post_training_audit_max_windows", type=int, default=None)
    parser.add_argument("--post_training_audit_max_workflows", type=int, default=None)
    parser.add_argument("--update_eval_max_windows", type=int, default=None)
    parser.add_argument("--update_eval_max_workflows", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "training" / "main_agents"))
    args = parser.parse_args()
    if args.smoke_run:
        args.profile = "smoke"
    profile_defaults = PROFILE_DEFAULTS[args.profile]
    for field_name in ["episodes", "update_every", "batch_size", "learning_rate", "clip_ratio", "entropy_coef", "value_coef", "auxiliary_coef", "max_steps", "train_window_count"]:
        if getattr(args, field_name) is None:
            setattr(args, field_name, profile_defaults[field_name])
    for field_name in ["window_mode", "train_window_mode"]:
        default_value = profile_defaults.get(field_name)
        if default_value is not None and getattr(args, field_name) == parser.get_default(field_name):
            setattr(args, field_name, default_value)
    if args.primary_vehicle_selection is None:
        args.primary_vehicle_selection = str(
            profile_defaults.get("primary_vehicle_selection", "stable_first")
        )
    if args.reward_positive_offset is None:
        args.reward_positive_offset = float(profile_defaults.get("reward_positive_offset", 5.0))
    if args.prediction_horizon is None:
        args.prediction_horizon = int(profile_defaults.get("prediction_horizon", 3))
    if args.post_training_audit_mode is None:
        args.post_training_audit_mode = str(profile_defaults.get("post_training_audit_mode", "full"))
    if args.temporal_reward_shaping_enabled is None:
        args.temporal_reward_shaping_enabled = bool(
            profile_defaults.get("temporal_reward_shaping_enabled", True)
        )
    for field_name in ["update_eval_max_windows", "update_eval_max_workflows"]:
        if getattr(args, field_name) is None and profile_defaults.get(field_name) is not None:
            setattr(args, field_name, int(profile_defaults[field_name]))
    if "gamma" in profile_defaults and float(args.gamma) == 0.99:
        args.gamma = float(profile_defaults["gamma"])
    if "gae_lambda" in profile_defaults and float(args.gae_lambda) == 0.95:
        args.gae_lambda = float(profile_defaults["gae_lambda"])
    if args.profile in {
        "top_journal_mechanism_v1",
        "top_journal_mechanism_v2",
        "top_journal_mechanism_v3",
        "top_journal_mechanism_v5_perf_robust",
        "top_journal_mechanism_v6_strong_competition",
        "top_journal_mechanism_v7_latency_fallback",
    }:
        strong_competition_profiles = {
            "top_journal_mechanism_v6_strong_competition",
            "top_journal_mechanism_v7_latency_fallback",
        }
        if float(args.mechanism_window_oversample_ratio) == 1.0:
            args.mechanism_window_oversample_ratio = (
                2.75
                if args.profile in strong_competition_profiles
                else 2.5
                if args.profile == "top_journal_mechanism_v5_perf_robust"
                else 2.0
            )
        if float(args.handoff_imminent_oversample_ratio) == 1.0:
            args.handoff_imminent_oversample_ratio = (
                1.90
                if args.profile in strong_competition_profiles
                else 1.75
                if args.profile == "top_journal_mechanism_v5_perf_robust"
                else 1.5
            )
        if float(args.target_mismatch_sample_weight) == 1.0:
            args.target_mismatch_sample_weight = (
                1.90
                if args.profile in strong_competition_profiles
                else 1.75
                if args.profile == "top_journal_mechanism_v5_perf_robust"
                else 1.5
            )
        if int(args.min_mechanism_activating_windows) == 0:
            args.min_mechanism_activating_windows = (
                3 if args.profile in strong_competition_profiles else 2
            )
    if args.profile in MECHANISM_COVERAGE_PROFILES:
        if float(args.mechanism_window_oversample_ratio) == 1.0:
            args.mechanism_window_oversample_ratio = 2.25
        if float(args.handoff_imminent_oversample_ratio) == 1.0:
            args.handoff_imminent_oversample_ratio = 1.50
        if float(args.target_mismatch_sample_weight) == 1.0:
            args.target_mismatch_sample_weight = 1.50
        if int(args.min_mechanism_activating_windows) == 0:
            args.min_mechanism_activating_windows = 4
    return args


def extract_reward_breakdown_means(summary: dict[str, Any]) -> dict[str, float]:
    reward_breakdown = summary.get("reward_breakdown", {})
    return {
        field_name: round(float(reward_breakdown.get(field_name, {}).get("mean", 0.0)), 6)
        for field_name in REWARD_BREAKDOWN_FIELDS
    }


def safe_rate(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 1e-8:
        return 0.0
    return float(numerator) / denominator


def extract_mechanism_diagnostics(summary: dict[str, Any]) -> dict[str, float]:
    run_info = summary.get("run_info", {})
    total_steps = max(int(run_info.get("total_steps", 0) or 0), 1)
    prefetch_summary = summary.get("prefetch_summary", {})
    validation_summary = summary.get("prefetch_validation_summary", {})
    handoff_summary = summary.get("handoff_summary", {})
    step_trace = [step for step in summary.get("step_trace", []) if isinstance(step, dict)]
    prefetch_request_count = int(prefetch_summary.get("true_predictive_prefetch_count", 0) or 0)
    prefetch_validated_hit_count = int(validation_summary.get("prefetch_validated_hit_count", 0) or 0)
    prefetch_expired_miss_count = int(validation_summary.get("prefetch_expired_miss_count", 0) or 0)
    migration_prepare_count = int(handoff_summary.get("migration_prepare_count", 0) or 0)
    migration_during_handoff_count = int(handoff_summary.get("migration_during_handoff_count", 0) or 0)
    handoff_total_count = int(handoff_summary.get("handoff_total_count", 0) or 0)
    handoff_ready_count = int(handoff_summary.get("handoff_ready_count", 0) or 0)
    mechanism_attempt_count = int(
        sum(
            1
            for step in step_trace
            if bool(step.get("mechanism_attempt_selected", False))
            or bool(step.get("predictive_prefetch_requested", False))
            or bool(step.get("migration_prepare_requested", False))
        )
    )
    mechanism_validated_success_count = int(
        prefetch_validated_hit_count
        + sum(1 for step in step_trace if bool(step.get("mechanism_success_strict", False)))
    )
    mechanism_pending_success_count = int(
        sum(1 for step in step_trace if bool(step.get("mechanism_success_gate_pending", False)))
    )
    mechanism_realized = float(
        prefetch_validated_hit_count > 0
        or migration_during_handoff_count > 0
        or handoff_ready_count > 0
    )
    return {
        "prefetch_request_count": prefetch_request_count,
        "prefetch_request_rate": round(safe_rate(prefetch_request_count, total_steps), 6),
        "prefetch_validated_hit_count": prefetch_validated_hit_count,
        "prefetch_validated_hit_rate": round(safe_rate(prefetch_validated_hit_count, max(prefetch_request_count, 1)), 6),
        "prefetch_expired_miss_count": prefetch_expired_miss_count,
        "mechanism_attempt_count": mechanism_attempt_count,
        "mechanism_validated_success_count": mechanism_validated_success_count,
        "mechanism_success_rate": round(safe_rate(mechanism_validated_success_count, max(mechanism_attempt_count, 1)), 6),
        "mechanism_pending_success_count": mechanism_pending_success_count,
        "migration_prepare_count": migration_prepare_count,
        "migration_prepare_rate": round(safe_rate(migration_prepare_count, total_steps), 6),
        "migration_during_handoff_count": migration_during_handoff_count,
        "handoff_total_count": handoff_total_count,
        "handoff_ready_count": handoff_ready_count,
        "handoff_ready_rate": round(safe_rate(handoff_ready_count, max(handoff_total_count, 1)), 6),
        "mechanism_realization_rate": round(mechanism_realized, 6),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in EVAL_METRICS}
    return {metric: round(fmean(float(row.get(metric, 0.0)) for row in rows), 6) for metric in EVAL_METRICS}


def aggregate_reward_breakdown(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {field_name: 0.0 for field_name in REWARD_BREAKDOWN_FIELDS}
    return {
        field_name: round(fmean(float(row.get(field_name, 0.0)) for row in rows), 6)
        for field_name in REWARD_BREAKDOWN_FIELDS
    }


def default_policy_diagnostics() -> dict[str, float]:
    return {field_name: 0.0 for field_name in POLICY_DIAGNOSTIC_FIELDS}


def safe_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=np.float32), percentile)), 6)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(fmean(values)), 6)


def resolve_primary_vehicle_from_semantic_state(
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
    return resolved_vehicle, {
        "primary_vehicle_id": primary_vehicle_id,
        "primary_vehicle_present": bool(
            semantic_state.get("primary_vehicle_present", False)
            or (primary_vehicle_id and resolved_vehicle_id == primary_vehicle_id)
        ),
        "primary_vehicle_reordered_to_front": bool(
            semantic_state.get("primary_vehicle_reordered_to_front", False)
        ),
        "first_vehicle_id": first_vehicle_id,
        "first_vehicle_matches_primary": bool(
            primary_vehicle_id and first_vehicle_id and str(first_vehicle_id) == str(primary_vehicle_id)
        ),
        "primary_vehicle_lookup_fallback": lookup_fallback,
    }


def semantic_state_has_valid_predicted_handoff_target(semantic_state: dict[str, Any]) -> bool:
    predictions = semantic_state.get("predictions", {})
    primary_vehicle, _ = resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    predicted_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    return bool(predicted_target)


def semantic_state_has_raw_handoff_candidate(semantic_state: dict[str, Any]) -> bool:
    predictions = semantic_state.get("predictions", {})
    primary_vehicle, _ = resolve_primary_vehicle_from_semantic_state(semantic_state)
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


def semantic_state_predictor_invoked(semantic_state: dict[str, Any]) -> bool:
    predictions = semantic_state.get("predictions", {})
    return bool(isinstance(predictions, dict) and str(predictions.get("predictor_name", "")))


def semantic_state_candidate_block_reason(semantic_state: dict[str, Any]) -> str:
    predictions = semantic_state.get("predictions", {})
    if not isinstance(predictions, dict) or not predictions:
        return "missing_prediction_state"
    if not str(predictions.get("predictor_name", "")):
        return "missing_prediction_state"
    primary_vehicle, _ = resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    next_sequence_map = predictions.get("next_rsu_sequence", {})
    if not isinstance(next_sequence_map, dict):
        return "missing_prediction_state"
    sequence = list(next_sequence_map.get(vehicle_id, []))
    non_null_sequence = [rsu_id for rsu_id in sequence if rsu_id is not None]
    if len(sequence) <= 0 or len(non_null_sequence) <= 0:
        return "no_next_rsu"
    if not semantic_state_has_raw_handoff_candidate(semantic_state):
        if current_rsu_id is not None and all(rsu_id == current_rsu_id for rsu_id in non_null_sequence):
            return "same_rsu"
        return "same_rsu"
    return "none"


def build_gt_handoff_context(env: Any) -> dict[str, Any]:
    core_env = getattr(env, "core_env", None)
    trajectory_frames = list(getattr(getattr(core_env, "_mobility_provider", None), "_trajectory_frames", []))
    time_to_index = {
        int(frame.get("time_index", 0)): index
        for index, frame in enumerate(trajectory_frames)
    }
    rsu_mapper = None
    if core_env is not None:
        rsu_mapper = RSUMapper([clone_rsu_state(rsu_state) for rsu_state in getattr(core_env, "rsu_states", [])])
    return {
        "core_env": core_env,
        "trajectory_frames": trajectory_frames,
        "time_to_index": time_to_index,
        "rsu_mapper": rsu_mapper,
    }


def build_gt_handoff_probe(gt_context: dict[str, Any], semantic_state: dict[str, Any]) -> dict[str, Any]:
    core_env = gt_context.get("core_env")
    trajectory_frames = list(gt_context.get("trajectory_frames", []))
    time_to_index = dict(gt_context.get("time_to_index", {}))
    rsu_mapper = gt_context.get("rsu_mapper")
    if core_env is None or rsu_mapper is None or not trajectory_frames:
        return {
            "gt_handoff_opportunity": 0.0,
            "gt_first_handoff_steps": 0.0,
            "gt_first_next_rsu": None,
            "current_rsu_id": None,
        }
    if hasattr(core_env, "_extract_primary_vehicle_from_state") and callable(getattr(core_env, "_extract_primary_vehicle_from_state")):
        primary_vehicle = dict(core_env._extract_primary_vehicle_from_state(semantic_state))
    else:
        primary_vehicle, _ = resolve_primary_vehicle_from_semantic_state(semantic_state)
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    if not vehicle_id:
        return {
            "gt_handoff_opportunity": 0.0,
            "gt_first_handoff_steps": 0.0,
            "gt_first_next_rsu": None,
            "current_rsu_id": current_rsu_id,
        }
    current_time_index = int(semantic_state.get("time_index", 0) or 0)
    current_frame_index = time_to_index.get(current_time_index)
    if current_frame_index is None:
        return {
            "gt_handoff_opportunity": 0.0,
            "gt_first_handoff_steps": 0.0,
            "gt_first_next_rsu": None,
            "current_rsu_id": current_rsu_id,
        }
    for future_frame_index in range(current_frame_index + 1, len(trajectory_frames)):
        future_frame = trajectory_frames[future_frame_index]
        if hasattr(core_env, "_frame_to_vehicle_states") and callable(getattr(core_env, "_frame_to_vehicle_states")):
            future_vehicle_states = core_env._frame_to_vehicle_states(future_frame)
        else:
            future_vehicle_states = future_frame.get("vehicles", [])
        future_associations = rsu_mapper.associate(future_vehicle_states)
        future_associated_rsu_id = future_associations.get(vehicle_id)
        if future_associated_rsu_id is not None and future_associated_rsu_id != current_rsu_id:
            return {
                "gt_handoff_opportunity": 1.0,
                "gt_first_handoff_steps": float(future_frame_index - current_frame_index),
                "gt_first_next_rsu": str(future_associated_rsu_id),
                "current_rsu_id": current_rsu_id,
            }
    return {
        "gt_handoff_opportunity": 0.0,
        "gt_first_handoff_steps": 0.0,
        "gt_first_next_rsu": None,
        "current_rsu_id": current_rsu_id,
    }


def annotate_retrospective_handoff_auxiliary_targets(
    *,
    rollout: list[dict[str, Any]],
    env: Any,
    agent: Any,
) -> dict[str, Any]:
    if not bool(getattr(agent, "_retrospective_handoff_aux_enabled", False)) or not rollout:
        return {
            "enabled": False,
            "annotated_count": 0,
            "opportunity_count": 0,
            "target_distinct_count": 0,
        }
    gt_context = build_gt_handoff_context(env)
    max_eta = float(getattr(agent, "_retrospective_handoff_aux_max_eta", 6.0))
    opportunity_count = 0
    target_distinct_count = 0
    eta_values: list[float] = []
    for row in rollout:
        decision_info = dict(row.get("decision_info", {}) or {})
        semantic_state = dict(decision_info.get("semantic_state", {}) or {})
        gt_probe = build_gt_handoff_probe(gt_context, semantic_state)
        gt_opportunity = float(gt_probe.get("gt_handoff_opportunity", 0.0) or 0.0)
        gt_eta = float(gt_probe.get("gt_first_handoff_steps", 0.0) or 0.0)
        gt_target = gt_probe.get("gt_first_next_rsu")
        current_rsu_id = gt_probe.get("current_rsu_id")
        target_distinct = bool(
            gt_target is not None
            and (current_rsu_id is None or str(gt_target) != str(current_rsu_id))
        )
        effective = bool(gt_opportunity > 0.5 and target_distinct and 0.0 < gt_eta <= max_eta)
        if gt_opportunity > 0.5:
            opportunity_count += 1
        if target_distinct:
            target_distinct_count += 1
        if effective:
            eta_values.append(gt_eta)
        decision_info["retrospective_handoff_label"] = {
            "enabled": True,
            "gt_handoff_opportunity": gt_opportunity,
            "gt_first_handoff_steps": gt_eta,
            "gt_first_next_rsu": gt_target,
            "current_rsu_id": current_rsu_id,
            "target_distinct": target_distinct,
            "effective_aux_candidate": effective,
            "max_eta": max_eta,
            "source": "training_retrospective_gt_handoff_probe",
        }
        row["decision_info"] = decision_info
    return {
        "enabled": True,
        "annotated_count": len(rollout),
        "opportunity_count": int(opportunity_count),
        "target_distinct_count": int(target_distinct_count),
        "effective_aux_candidate_count": len(eta_values),
        "effective_aux_candidate_rate": round(float(len(eta_values)) / float(max(len(rollout), 1)), 6),
        "eta_mean": safe_mean(eta_values),
        "eta_p25": safe_percentile(eta_values, 25.0),
        "eta_p75": safe_percentile(eta_values, 75.0),
    }


def build_policy_alignment_sample(step_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_index": int(step_row.get("debug_step_index", 0) or 0),
        "primary_vehicle_id": step_row.get("debug_primary_vehicle_id"),
        "first_vehicle_id": step_row.get("debug_first_vehicle_id"),
        "gt_current_rsu": step_row.get("debug_gt_current_rsu"),
        "policy_current_rsu": step_row.get("debug_policy_current_rsu"),
        "current_rsu": step_row.get("debug_policy_current_rsu"),
        "gt_first_next_rsu": step_row.get("debug_gt_first_next_rsu"),
        "gt_eta": int(step_row.get("debug_gt_first_handoff_eta", 0) or 0),
        "predicted_sequence": list(step_row.get("debug_predicted_sequence_preview", [])),
        "predicted_first_non_current_rsu": step_row.get("debug_predicted_first_non_current_rsu"),
        "pred_eta": int(step_row.get("debug_predicted_first_non_current_eta", 0) or 0),
    }


def build_stochastic_event_probe(
    agent: Any,
    semantic_state: dict[str, Any],
    *,
    policy_evaluation_mode: str = "safety_projected",
) -> dict[str, float]:
    default_probe = {
        "stochastic_event_prepare": 0.0,
        "event_prepare_prob": 0.0,
        "event_margin": 0.0,
    }
    forward_policy = getattr(agent, "_forward_policy", None)
    if not callable(forward_policy):
        return default_probe
    with torch.no_grad():
        policy_output = forward_policy(
            semantic_state,
            run_metadata={"policy_evaluation_mode": policy_evaluation_mode},
        )
    if bool(getattr(agent, "_use_hierarchy", False)) and "event_logits" in policy_output:
        event_logits = policy_output["event_logits"]
        event_probs = torch.softmax(event_logits, dim=-1)
        sample_actions = getattr(agent, "_sample_actions", None)
        if callable(sample_actions):
            sampled_actions, _, _, _, _ = sample_actions(policy_output, deterministic=False)
            stochastic_prepare = 1.0 if int(sampled_actions.get("event", 0)) == 1 else 0.0
        else:
            stochastic_prepare = 1.0 if int(torch.distributions.Categorical(logits=event_logits).sample().item()) == 1 else 0.0
        return {
            "stochastic_event_prepare": stochastic_prepare,
            "event_prepare_prob": float(event_probs[1].item()) if event_probs.numel() > 1 else 0.0,
            "event_margin": float((event_logits[1] - event_logits[0]).item()) if event_logits.numel() > 1 else 0.0,
        }
    flat_logits = policy_output.get("flat_logits")
    if flat_logits is None or int(flat_logits.numel()) <= 4:
        return default_probe
    action_probs = torch.softmax(flat_logits, dim=-1)
    stochastic_action = int(torch.distributions.Categorical(logits=flat_logits).sample().item())
    logit_values = [float(item) for item in flat_logits.detach().cpu().tolist()]
    best_other = max((value for index, value in enumerate(logit_values) if index != 4), default=logit_values[4])
    return {
        "stochastic_event_prepare": 1.0 if stochastic_action == 4 else 0.0,
        "event_prepare_prob": float(action_probs[4].item()),
        "event_margin": float(logit_values[4] - best_other),
    }


def build_policy_step_diagnostic(
    *,
    agent: Any,
    action: int,
    action_info: dict[str, Any],
    semantic_state: dict[str, Any],
    stochastic_probe: dict[str, float],
    gt_probe: dict[str, Any],
    env_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing_features = compute_temporal_prepare_window_score(
        semantic_state,
        preferred_lead_steps=float(getattr(agent, "_temporal_prepare_lead_steps", 2.5)),
        sigma=float(getattr(agent, "_temporal_prepare_sigma", 1.25)),
    )
    primary_vehicle, primary_resolution = resolve_primary_vehicle_from_semantic_state(semantic_state)
    metrics_protocol = dict((env_info or {}).get("metrics_protocol", {}))
    vehicle_id = str(primary_vehicle.get("vehicle_id", ""))
    predictions = semantic_state.get("predictions", {})
    prediction_audit = dict(predictions.get("prediction_quality_audit", {})) if isinstance(predictions, dict) else {}
    next_rsu_sequence_map = predictions.get("next_rsu_sequence", {}) if isinstance(predictions, dict) else {}
    semantic_prediction_sequence = list(next_rsu_sequence_map.get(vehicle_id, [])) if isinstance(next_rsu_sequence_map, dict) else []
    current_rsu_id = action_info.get("current_rsu_id", primary_vehicle.get("associated_rsu_id"))
    predicted_sequence_preview = list(action_info.get("predicted_sequence_preview", semantic_prediction_sequence[:6]))
    predicted_first_non_current_rsu = action_info.get("predicted_first_non_current_rsu")
    predicted_first_non_current_eta = int(action_info.get("predicted_first_non_current_eta", 0) or 0)
    if not predicted_first_non_current_rsu:
        for step_index, rsu_id in enumerate(semantic_prediction_sequence, start=1):
            if rsu_id is None:
                continue
            if current_rsu_id is None or rsu_id != current_rsu_id:
                predicted_first_non_current_rsu = str(rsu_id)
                predicted_first_non_current_eta = int(step_index)
                break
    valid_handoff_target = bool(
        action_info.get(
            "predicted_handoff_target_valid",
            semantic_state_has_valid_predicted_handoff_target(semantic_state),
        )
    )
    prepare_window_score = float(action_info.get("prepare_window_score", timing_features.get("prepare_window_score", 0.0)) or 0.0)
    temporal_urgency = float(action_info.get("temporal_urgency", timing_features.get("temporal_urgency", 0.0)) or 0.0)
    countdown_steps = float(action_info.get("handoff_countdown_steps", timing_features.get("countdown_steps", 0.0)) or 0.0)
    raw_handoff_candidate = bool(
        action_info.get(
            "raw_handoff_candidate",
            semantic_state_has_raw_handoff_candidate(semantic_state),
        )
    )
    predictor_invoked = bool(action_info.get("predictor_invoked", semantic_state_predictor_invoked(semantic_state)))
    candidate_block_reason = str(action_info.get("candidate_block_reason", semantic_state_candidate_block_reason(semantic_state)) or "none")
    prediction_sequence_horizon = int(
        action_info.get(
            "prediction_sequence_horizon",
            len(semantic_prediction_sequence),
        )
        or 0
    )
    gt_handoff_opportunity = float(gt_probe.get("gt_handoff_opportunity", 0.0) or 0.0)
    gt_first_handoff_steps = float(gt_probe.get("gt_first_handoff_steps", 0.0) or 0.0)
    gt_first_next_rsu = gt_probe.get("gt_first_next_rsu")
    gt_current_rsu = gt_probe.get("current_rsu_id")
    first_vehicle_id = action_info.get("first_vehicle_id", primary_resolution.get("first_vehicle_id"))
    primary_vehicle_id = action_info.get("primary_vehicle_id", primary_resolution.get("primary_vehicle_id"))
    first_vehicle_matches_primary = bool(
        action_info.get(
            "first_vehicle_matches_primary",
            primary_resolution.get("first_vehicle_matches_primary", False),
        )
    )
    primary_vehicle_lookup_fallback = bool(
        action_info.get(
            "primary_vehicle_lookup_fallback",
            primary_resolution.get("primary_vehicle_lookup_fallback", False),
        )
    )
    predicted_sequence_all_null = bool(action_info.get("predicted_sequence_all_null", False))
    predicted_sequence_all_current_rsu = bool(action_info.get("predicted_sequence_all_current_rsu", False))
    predicted_sequence_contains_other_rsu = bool(action_info.get("predicted_sequence_contains_other_rsu", bool(predicted_first_non_current_rsu)))
    eta_outside_window = bool(
        not raw_handoff_candidate
        and predictor_invoked
        and gt_handoff_opportunity > 0.5
        and prediction_sequence_horizon > 0
        and gt_first_handoff_steps > float(prediction_sequence_horizon)
    )
    smoothing_info = dict(action_info.get("deterministic_temporal_smoothing", {}))
    head_actions = dict(action_info.get("head_actions", {}))
    if "event" in head_actions:
        deterministic_event_prepare = 1.0 if int(head_actions.get("event", 0)) == 1 else 0.0
    else:
        deterministic_event_prepare = 1.0 if int(action) == 4 else 0.0
    activation_threshold = float(getattr(agent, "_temporal_prepare_activation_threshold", 0.35))
    high_prepare_threshold = float(getattr(agent, "_deterministic_high_prepare_threshold", 0.55))
    borderline_triggered = bool(smoothing_info.get("borderline_triggered", False))
    override_triggered = bool(smoothing_info.get("override_triggered", False))
    guard_info = dict(action_info.get("continuity_guard", {}))
    guard_triggered = bool(action_info.get("guard_triggered", False))
    cache_warm_guard_info = dict(action_info.get("cache_warm_start_guard", {}))
    prefetch_admission_guard_info = dict(action_info.get("predictive_prefetch_admission_guard", {}))
    coverage_recovery_final_guard_info = dict(action_info.get("coverage_recovery_final_guard", {}))
    backhaul_guard_info = dict(action_info.get("backhaul_guard", {}))
    backhaul_aware_info = dict(action_info.get("backhaul_aware_policy", {}))
    action_projection_applied = bool(action_info.get("action_projection_applied", False))
    invalid_action_attempt_count = int(action_info.get("invalid_action_attempt_count", 0) or 0)
    guard_action_delta = bool(action_info.get("guard_action_delta", False))
    gt_first_next_rsu_non_null = bool(gt_first_next_rsu)
    pred_first_non_current_rsu_present = bool(predicted_first_non_current_rsu)
    gt_pred_next_rsu_match = bool(
        gt_first_next_rsu_non_null
        and pred_first_non_current_rsu_present
        and str(gt_first_next_rsu) == str(predicted_first_non_current_rsu)
    )
    gt_pred_next_rsu_mismatch = bool(
        gt_first_next_rsu_non_null
        and pred_first_non_current_rsu_present
        and str(gt_first_next_rsu) != str(predicted_first_non_current_rsu)
    )
    return {
        "first_vehicle_matches_primary": 1.0 if first_vehicle_matches_primary else 0.0,
        "policy_current_rsu_non_null": 1.0 if current_rsu_id else 0.0,
        "gt_current_rsu_non_null": 1.0 if gt_current_rsu else 0.0,
        "primary_vehicle_lookup_fallback": 1.0 if primary_vehicle_lookup_fallback else 0.0,
        "stochastic_event_prepare": float(stochastic_probe.get("stochastic_event_prepare", 0.0) or 0.0),
        "deterministic_event_prepare": deterministic_event_prepare,
        "gt_handoff_opportunity": gt_handoff_opportunity,
        "gt_first_handoff_eta": gt_first_handoff_steps,
        "gt_first_next_rsu_non_null": 1.0 if gt_first_next_rsu_non_null else 0.0,
        "predictor_invoked": 1.0 if predictor_invoked else 0.0,
        "raw_handoff_candidate": 1.0 if raw_handoff_candidate else 0.0,
        "pred_first_non_current_rsu": 1.0 if pred_first_non_current_rsu_present else 0.0,
        "pred_first_non_current_rsu_eta": float(predicted_first_non_current_eta),
        "predicted_sequence_all_null": 1.0 if predicted_sequence_all_null else 0.0,
        "predicted_sequence_all_current_rsu": 1.0 if predicted_sequence_all_current_rsu else 0.0,
        "predicted_sequence_contains_other_rsu": 1.0 if predicted_sequence_contains_other_rsu else 0.0,
        "gt_pred_next_rsu_match": 1.0 if gt_pred_next_rsu_match else 0.0,
        "gt_pred_next_rsu_mismatch": 1.0 if gt_pred_next_rsu_mismatch else 0.0,
        "prediction_confidence": float(action_info.get("prediction_confidence", 0.0) or 0.0),
        "prediction_uncertainty": float(action_info.get("prediction_uncertainty", 1.0) or 1.0),
        "predictor_brier_score_proxy": float(
            metrics_protocol.get(
                "predictor_brier_score_proxy",
                prediction_audit.get("brier_score_proxy", 0.0),
            )
            or 0.0
        ),
        "predictor_calibration_error_proxy": float(
            metrics_protocol.get(
                "predictor_confidence_calibration_error_proxy",
                prediction_audit.get("confidence_calibration_error_proxy", 0.0),
            )
            or 0.0
        ),
        "predictor_handoff_target_precision_proxy": float(
            metrics_protocol.get(
                "predictor_handoff_target_precision_proxy",
                prediction_audit.get("handoff_target_precision_proxy", 0.0),
            )
            or 0.0
        ),
        "predictor_handoff_target_recall_proxy": float(
            metrics_protocol.get(
                "predictor_handoff_target_recall_proxy",
                prediction_audit.get("handoff_target_recall_proxy", 0.0),
            )
            or 0.0
        ),
        "urgency_support": float(action_info.get("urgency_support", 0.7 + 0.3 * temporal_urgency) or 0.0),
        "prediction_gate_value": float(action_info.get("prediction_gate_value", 0.0) or 0.0),
        "gate_pass": 1.0 if bool(action_info.get("gate_pass", False)) else 0.0,
        "candidate_block_reason_no_next_rsu": 1.0 if candidate_block_reason == "no_next_rsu" else 0.0,
        "candidate_block_reason_same_rsu": 1.0 if candidate_block_reason == "same_rsu" else 0.0,
        "candidate_block_reason_no_eta": 0.0,
        "candidate_block_reason_eta_outside_window": 1.0 if eta_outside_window else 0.0,
        "candidate_block_reason_low_handoff_risk": 0.0,
        "candidate_block_reason_short_target_dwell": 0.0,
        "candidate_block_reason_missing_prediction_state": 1.0 if candidate_block_reason == "missing_prediction_state" else 0.0,
        "invalid_reason_no_candidate": 1.0 if str(action_info.get("prediction_invalid_reason", "none")) == "no_candidate" else 0.0,
        "invalid_reason_low_confidence": 1.0 if str(action_info.get("prediction_invalid_reason", "none")) == "low_confidence" else 0.0,
        "invalid_reason_high_uncertainty": 1.0 if str(action_info.get("prediction_invalid_reason", "none")) == "high_uncertainty" else 0.0,
        "invalid_reason_gate_below_threshold": 1.0 if str(action_info.get("prediction_invalid_reason", "none")) == "gate_below_threshold" else 0.0,
        "prepare_window_score": prepare_window_score,
        "timing_active": 1.0 if prepare_window_score >= activation_threshold else 0.0,
        "high_prepare": 1.0 if prepare_window_score >= high_prepare_threshold else 0.0,
        "event_prepare_prob": float(action_info.get("event_prepare_prob", stochastic_probe.get("event_prepare_prob", 0.0)) or 0.0),
        "event_margin": float(action_info.get("event_margin", stochastic_probe.get("event_margin", 0.0)) or 0.0),
        "temporal_urgency": temporal_urgency,
        "countdown_steps": countdown_steps,
        "prediction_sequence_horizon": float(prediction_sequence_horizon),
        "gt_first_handoff_steps": gt_first_handoff_steps,
        "predicted_handoff_target_valid": 1.0 if valid_handoff_target else 0.0,
        "borderline_triggered": 1.0 if borderline_triggered else 0.0,
        "override_triggered": 1.0 if override_triggered else 0.0,
        "forced_temporal_intervention": 1.0 if (borderline_triggered or override_triggered) else 0.0,
        "continuity_guard_triggered": 1.0 if guard_triggered else 0.0,
        "target_mismatch_guard": 1.0 if guard_triggered and bool(guard_info.get("target_mismatch", False)) else 0.0,
        "guard_prefetch_to_prepare": 1.0 if guard_triggered and int(action_info.get("original_action", -1)) == 1 and int(action_info.get("guarded_action", -1)) == 4 else 0.0,
        "guard_hard_override": 1.0 if bool(guard_info.get("hard_override_applied", False)) else 0.0,
        "action_projection_applied": 1.0 if action_projection_applied else 0.0,
        "invalid_action_attempt_count": float(invalid_action_attempt_count),
        "guard_action_delta": 1.0 if guard_action_delta else 0.0,
        "cache_warm_start_guarded": 1.0 if bool(cache_warm_guard_info.get("guarded", False)) else 0.0,
        "predictive_prefetch_admission_guarded": 1.0 if bool(prefetch_admission_guard_info.get("guarded", False)) else 0.0,
        "coverage_recovery_final_guarded": (
            1.0 if bool(coverage_recovery_final_guard_info.get("guarded", False)) else 0.0
        ),
        "backhaul_guarded": 1.0 if bool(backhaul_guard_info.get("guarded", False)) else 0.0,
        "backhaul_aware_policy_adjusted": 1.0
        if bool(backhaul_aware_info.get("enabled", False))
        and (
            abs(float(backhaul_aware_info.get("service_fill_bias", 0.0) or 0.0)) > 1e-8
            or abs(float(backhaul_aware_info.get("redundant_fill_penalty", 0.0) or 0.0)) > 1e-8
            or abs(float(backhaul_aware_info.get("prefetch_penalty", 0.0) or 0.0)) > 1e-8
            or abs(float(backhaul_aware_info.get("prepare_penalty", 0.0) or 0.0)) > 1e-8
        )
        else 0.0,
        "backhaul_aware_service_fill_bias": float(backhaul_aware_info.get("service_fill_bias", 0.0) or 0.0),
        "backhaul_aware_redundant_fill_penalty": float(
            backhaul_aware_info.get("redundant_fill_penalty", 0.0) or 0.0
        ),
        "backhaul_aware_no_signal_penalty": float(backhaul_aware_info.get("prefetch_penalty", 0.0) or 0.0)
        + float(backhaul_aware_info.get("prepare_penalty", 0.0) or 0.0),
        "dag_frontier_size": float(metrics_protocol.get("dag_frontier_size", 0.0) or 0.0),
        "dag_critical_path_pressure": float(metrics_protocol.get("dag_critical_path_pressure", 0.0) or 0.0),
        "dag_current_node_dependency_pressure": float(
            metrics_protocol.get("dag_current_node_dependency_pressure", 0.0) or 0.0
        ),
        "dag_remaining_nodes": float(metrics_protocol.get("dag_remaining_nodes", 0.0) or 0.0),
        "mechanism_attempt_selected": 1.0 if bool(metrics_protocol.get("mechanism_attempt_selected", False)) else 0.0,
        "mechanism_success_strict": 1.0 if bool(metrics_protocol.get("mechanism_success_strict", False)) else 0.0,
        "mechanism_success_gate_pending": 1.0 if bool(metrics_protocol.get("mechanism_success_gate_pending", False)) else 0.0,
        "debug_raw_env_action": int(action_info.get("raw_env_action", action) or action),
        "debug_projected_env_action": int(action_info.get("projected_env_action", action) or action),
        "debug_final_env_action": int(action_info.get("final_env_action", action) or action),
        "debug_primary_vehicle_id": primary_vehicle_id,
        "debug_first_vehicle_id": first_vehicle_id,
        "debug_gt_current_rsu": gt_current_rsu,
        "debug_policy_current_rsu": current_rsu_id,
        "debug_current_rsu_id": current_rsu_id,
        "debug_gt_first_next_rsu": gt_first_next_rsu,
        "debug_gt_first_handoff_eta": gt_first_handoff_steps,
        "debug_predicted_sequence_preview": predicted_sequence_preview,
        "debug_predicted_first_non_current_rsu": predicted_first_non_current_rsu,
        "debug_predicted_first_non_current_eta": predicted_first_non_current_eta,
    }


def build_policy_action_signature(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint deterministic policy actions on the fixed development protocol.

    The signature deliberately records policy-derived choices only.  It is used to
    reject a checkpoint-selection run when the raw, masked policy never changes
    across checkpoints, even if safety rules make the projected evaluation look
    stable or favourable.
    """
    action_counts: dict[str, int] = {}
    payload_rows: list[dict[str, Any]] = []
    for row in step_rows:
        final_action = int(row.get("debug_final_env_action", 0) or 0)
        action_counts[str(final_action)] = action_counts.get(str(final_action), 0) + 1
        payload_rows.append(
            {
                "step": int(row.get("debug_step_index", 0) or 0),
                "action": final_action,
                "raw_action": int(row.get("debug_raw_env_action", final_action) or final_action),
                "projected_action": int(row.get("debug_projected_env_action", final_action) or final_action),
                "event_prepare": round(float(row.get("deterministic_event_prepare", 0.0) or 0.0), 8),
                "event_margin": round(float(row.get("event_margin", 0.0) or 0.0), 8),
            }
        )
    payload = json.dumps(payload_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "step_count": len(payload_rows),
        "action_counts": action_counts,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def aggregate_policy_action_signatures(signature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_rows = [
        {
            "workflow_id": str(row.get("workflow_id", "")),
            "window_id": str(row.get("window_id", "")),
            "step_count": int(row.get("step_count", 0) or 0),
            "digest": str(row.get("digest", "")),
            "action_counts": {
                str(key): int(value)
                for key, value in sorted(dict(row.get("action_counts", {})).items(), key=lambda item: str(item[0]))
            },
        }
        for row in signature_rows
    ]
    payload = json.dumps(canonical_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "episode_count": len(canonical_rows),
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "episodes": canonical_rows,
    }


def aggregate_policy_diagnostics(step_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not step_rows:
        return default_policy_diagnostics()
    total_steps = max(len(step_rows), 1)
    first_vehicle_matches_primary_rate = float(
        fmean(float(row.get("first_vehicle_matches_primary", 0.0)) for row in step_rows)
    )
    policy_current_rsu_non_null_rate = float(
        fmean(float(row.get("policy_current_rsu_non_null", 0.0)) for row in step_rows)
    )
    gt_current_rsu_non_null_rate = float(
        fmean(float(row.get("gt_current_rsu_non_null", 0.0)) for row in step_rows)
    )
    primary_vehicle_lookup_fallback_rate = float(
        fmean(float(row.get("primary_vehicle_lookup_fallback", 0.0)) for row in step_rows)
    )
    stochastic_event_prepare_rate = float(fmean(float(row.get("stochastic_event_prepare", 0.0)) for row in step_rows))
    deterministic_event_prepare_rate = float(fmean(float(row.get("deterministic_event_prepare", 0.0)) for row in step_rows))
    gt_handoff_opportunity_step_count = int(round(sum(float(row.get("gt_handoff_opportunity", 0.0)) for row in step_rows)))
    gt_first_next_rsu_non_null_step_count = int(round(sum(float(row.get("gt_first_next_rsu_non_null", 0.0)) for row in step_rows)))
    predictor_invoked_step_count = int(round(sum(float(row.get("predictor_invoked", 0.0)) for row in step_rows)))
    raw_handoff_candidate_step_count = int(round(sum(float(row.get("raw_handoff_candidate", 0.0)) for row in step_rows)))
    pred_first_non_current_rsu_step_count = int(round(sum(float(row.get("pred_first_non_current_rsu", 0.0)) for row in step_rows)))
    predicted_sequence_all_null_count = int(round(sum(float(row.get("predicted_sequence_all_null", 0.0)) for row in step_rows)))
    predicted_sequence_all_current_rsu_count = int(round(sum(float(row.get("predicted_sequence_all_current_rsu", 0.0)) for row in step_rows)))
    predicted_sequence_contains_other_rsu_count = int(round(sum(float(row.get("predicted_sequence_contains_other_rsu", 0.0)) for row in step_rows)))
    gt_pred_next_rsu_match_count = int(round(sum(float(row.get("gt_pred_next_rsu_match", 0.0)) for row in step_rows)))
    gt_pred_next_rsu_mismatch_count = int(round(sum(float(row.get("gt_pred_next_rsu_mismatch", 0.0)) for row in step_rows)))
    prediction_confidences = [float(row.get("prediction_confidence", 0.0)) for row in step_rows]
    prediction_uncertainties = [float(row.get("prediction_uncertainty", 1.0)) for row in step_rows]
    predictor_brier_values = [float(row.get("predictor_brier_score_proxy", 0.0)) for row in step_rows]
    predictor_calibration_values = [float(row.get("predictor_calibration_error_proxy", 0.0)) for row in step_rows]
    predictor_precision_values = [float(row.get("predictor_handoff_target_precision_proxy", 0.0)) for row in step_rows]
    predictor_recall_values = [float(row.get("predictor_handoff_target_recall_proxy", 0.0)) for row in step_rows]
    urgency_supports = [float(row.get("urgency_support", 0.0)) for row in step_rows]
    prediction_gate_values = [float(row.get("prediction_gate_value", 0.0)) for row in step_rows]
    gate_pass_step_count = int(round(sum(float(row.get("gate_pass", 0.0)) for row in step_rows)))
    candidate_block_reason_no_next_rsu_count = int(round(sum(float(row.get("candidate_block_reason_no_next_rsu", 0.0)) for row in step_rows)))
    candidate_block_reason_same_rsu_count = int(round(sum(float(row.get("candidate_block_reason_same_rsu", 0.0)) for row in step_rows)))
    candidate_block_reason_no_eta_count = int(round(sum(float(row.get("candidate_block_reason_no_eta", 0.0)) for row in step_rows)))
    candidate_block_reason_eta_outside_window_count = int(round(sum(float(row.get("candidate_block_reason_eta_outside_window", 0.0)) for row in step_rows)))
    candidate_block_reason_low_handoff_risk_count = int(round(sum(float(row.get("candidate_block_reason_low_handoff_risk", 0.0)) for row in step_rows)))
    candidate_block_reason_short_target_dwell_count = int(round(sum(float(row.get("candidate_block_reason_short_target_dwell", 0.0)) for row in step_rows)))
    candidate_block_reason_missing_prediction_state_count = int(round(sum(float(row.get("candidate_block_reason_missing_prediction_state", 0.0)) for row in step_rows)))
    prepare_scores = [float(row.get("prepare_window_score", 0.0)) for row in step_rows]
    event_prepare_probs = [float(row.get("event_prepare_prob", 0.0)) for row in step_rows]
    event_margins = [float(row.get("event_margin", 0.0)) for row in step_rows]
    temporal_urgencies = [float(row.get("temporal_urgency", 0.0)) for row in step_rows]
    valid_target_rows = [row for row in step_rows if float(row.get("predicted_handoff_target_valid", 0.0)) > 0.5]
    timing_active_rows = [row for row in step_rows if float(row.get("timing_active", 0.0)) > 0.5]
    gt_eta_values = [float(row.get("gt_first_handoff_eta", 0.0)) for row in step_rows if float(row.get("gt_first_next_rsu_non_null", 0.0)) > 0.5]
    pred_eta_values = [float(row.get("pred_first_non_current_rsu_eta", 0.0)) for row in step_rows if float(row.get("pred_first_non_current_rsu", 0.0)) > 0.5]
    countdown_values_on_valid_target = [float(row.get("countdown_steps", 0.0)) for row in valid_target_rows]
    prepare_scores_on_valid_target = [float(row.get("prepare_window_score", 0.0)) for row in valid_target_rows]
    borderline_trigger_count = int(round(sum(float(row.get("borderline_triggered", 0.0)) for row in step_rows)))
    override_trigger_count = int(round(sum(float(row.get("override_triggered", 0.0)) for row in step_rows)))
    forced_temporal_intervention_count = int(round(sum(float(row.get("forced_temporal_intervention", 0.0)) for row in step_rows)))
    continuity_guard_trigger_count = int(round(sum(float(row.get("continuity_guard_triggered", 0.0)) for row in step_rows)))
    target_mismatch_guard_count = int(round(sum(float(row.get("target_mismatch_guard", 0.0)) for row in step_rows)))
    guard_prefetch_to_prepare_count = int(round(sum(float(row.get("guard_prefetch_to_prepare", 0.0)) for row in step_rows)))
    guard_hard_override_count = int(round(sum(float(row.get("guard_hard_override", 0.0)) for row in step_rows)))
    action_projection_count = int(round(sum(float(row.get("action_projection_applied", 0.0)) for row in step_rows)))
    invalid_action_attempt_count = int(round(sum(float(row.get("invalid_action_attempt_count", 0.0)) for row in step_rows)))
    guard_action_delta_count = int(round(sum(float(row.get("guard_action_delta", 0.0)) for row in step_rows)))
    cache_warm_start_guard_count = int(round(sum(float(row.get("cache_warm_start_guarded", 0.0)) for row in step_rows)))
    predictive_prefetch_admission_guard_count = int(round(sum(float(row.get("predictive_prefetch_admission_guarded", 0.0)) for row in step_rows)))
    coverage_recovery_final_guard_count = int(
        round(sum(float(row.get("coverage_recovery_final_guarded", 0.0)) for row in step_rows))
    )
    backhaul_guard_count = int(round(sum(float(row.get("backhaul_guarded", 0.0)) for row in step_rows)))
    backhaul_aware_policy_adjust_count = int(
        round(sum(float(row.get("backhaul_aware_policy_adjusted", 0.0)) for row in step_rows))
    )
    backhaul_aware_service_fill_biases = [
        float(row.get("backhaul_aware_service_fill_bias", 0.0) or 0.0)
        for row in step_rows
    ]
    backhaul_aware_redundant_fill_penalties = [
        float(row.get("backhaul_aware_redundant_fill_penalty", 0.0) or 0.0)
        for row in step_rows
    ]
    backhaul_aware_no_signal_penalties = [
        float(row.get("backhaul_aware_no_signal_penalty", 0.0) or 0.0)
        for row in step_rows
    ]
    dag_frontier_sizes = [float(row.get("dag_frontier_size", 0.0)) for row in step_rows]
    dag_critical_path_pressures = [float(row.get("dag_critical_path_pressure", 0.0)) for row in step_rows]
    dag_dependency_pressures = [float(row.get("dag_current_node_dependency_pressure", 0.0)) for row in step_rows]
    dag_remaining_nodes = [float(row.get("dag_remaining_nodes", 0.0)) for row in step_rows]
    valid_handoff_target_step_count = int(round(sum(float(row.get("predicted_handoff_target_valid", 0.0)) for row in step_rows)))
    invalid_reason_no_candidate_count = int(round(sum(float(row.get("invalid_reason_no_candidate", 0.0)) for row in step_rows)))
    invalid_reason_low_confidence_count = int(round(sum(float(row.get("invalid_reason_low_confidence", 0.0)) for row in step_rows)))
    invalid_reason_high_uncertainty_count = int(round(sum(float(row.get("invalid_reason_high_uncertainty", 0.0)) for row in step_rows)))
    invalid_reason_gate_below_threshold_count = int(round(sum(float(row.get("invalid_reason_gate_below_threshold", 0.0)) for row in step_rows)))
    return {
        "first_vehicle_matches_primary_rate": round(first_vehicle_matches_primary_rate, 6),
        "policy_current_rsu_non_null_rate": round(policy_current_rsu_non_null_rate, 6),
        "gt_current_rsu_non_null_rate": round(gt_current_rsu_non_null_rate, 6),
        "primary_vehicle_lookup_fallback_rate": round(primary_vehicle_lookup_fallback_rate, 6),
        "stochastic_event_prepare_rate": round(stochastic_event_prepare_rate, 6),
        "deterministic_event_prepare_rate": round(deterministic_event_prepare_rate, 6),
        "gap_event_prepare_rate": round(stochastic_event_prepare_rate - deterministic_event_prepare_rate, 6),
        "gt_handoff_opportunity_step_count": gt_handoff_opportunity_step_count,
        "gt_handoff_opportunity_rate": round(float(gt_handoff_opportunity_step_count) / float(total_steps), 6),
        "gt_first_handoff_eta_mean": safe_mean(gt_eta_values),
        "gt_first_handoff_eta_p25": safe_percentile(gt_eta_values, 25.0),
        "gt_first_handoff_eta_p75": safe_percentile(gt_eta_values, 75.0),
        "gt_first_next_rsu_non_null_step_count": gt_first_next_rsu_non_null_step_count,
        "predictor_invoked_step_count": predictor_invoked_step_count,
        "predictor_invoked_rate": round(float(predictor_invoked_step_count) / float(total_steps), 6),
        "raw_handoff_candidate_step_count": raw_handoff_candidate_step_count,
        "raw_handoff_candidate_rate": round(float(raw_handoff_candidate_step_count) / float(total_steps), 6),
        "pred_first_non_current_rsu_step_count": pred_first_non_current_rsu_step_count,
        "pred_first_non_current_rsu_rate": round(float(pred_first_non_current_rsu_step_count) / float(total_steps), 6),
        "pred_first_non_current_rsu_eta_mean": safe_mean(pred_eta_values),
        "pred_first_non_current_rsu_eta_p25": safe_percentile(pred_eta_values, 25.0),
        "pred_first_non_current_rsu_eta_p75": safe_percentile(pred_eta_values, 75.0),
        "predicted_sequence_all_null_count": predicted_sequence_all_null_count,
        "predicted_sequence_all_current_rsu_count": predicted_sequence_all_current_rsu_count,
        "predicted_sequence_contains_other_rsu_count": predicted_sequence_contains_other_rsu_count,
        "gt_pred_next_rsu_match_count": gt_pred_next_rsu_match_count,
        "gt_pred_next_rsu_mismatch_count": gt_pred_next_rsu_mismatch_count,
        "prediction_confidence_mean": safe_mean(prediction_confidences),
        "prediction_confidence_p75": safe_percentile(prediction_confidences, 75.0),
        "prediction_uncertainty_mean": safe_mean(prediction_uncertainties),
        "prediction_uncertainty_p25": safe_percentile(prediction_uncertainties, 25.0),
        "prediction_uncertainty_p75": safe_percentile(prediction_uncertainties, 75.0),
        "predictor_brier_score_proxy_mean": safe_mean(predictor_brier_values),
        "predictor_calibration_error_proxy_mean": safe_mean(predictor_calibration_values),
        "predictor_handoff_target_precision_proxy_mean": safe_mean(predictor_precision_values),
        "predictor_handoff_target_recall_proxy_mean": safe_mean(predictor_recall_values),
        "urgency_support_mean": safe_mean(urgency_supports),
        "urgency_support_p75": safe_percentile(urgency_supports, 75.0),
        "prediction_gate_value_mean": safe_mean(prediction_gate_values),
        "prediction_gate_value_p75": safe_percentile(prediction_gate_values, 75.0),
        "gate_pass_step_count": gate_pass_step_count,
        "gate_pass_rate": round(float(gate_pass_step_count) / float(total_steps), 6),
        "candidate_block_reason_no_next_rsu_count": candidate_block_reason_no_next_rsu_count,
        "candidate_block_reason_same_rsu_count": candidate_block_reason_same_rsu_count,
        "candidate_block_reason_no_eta_count": candidate_block_reason_no_eta_count,
        "candidate_block_reason_eta_outside_window_count": candidate_block_reason_eta_outside_window_count,
        "candidate_block_reason_low_handoff_risk_count": candidate_block_reason_low_handoff_risk_count,
        "candidate_block_reason_short_target_dwell_count": candidate_block_reason_short_target_dwell_count,
        "candidate_block_reason_missing_prediction_state_count": candidate_block_reason_missing_prediction_state_count,
        "invalid_reason_no_candidate_count": invalid_reason_no_candidate_count,
        "invalid_reason_low_confidence_count": invalid_reason_low_confidence_count,
        "invalid_reason_high_uncertainty_count": invalid_reason_high_uncertainty_count,
        "invalid_reason_gate_below_threshold_count": invalid_reason_gate_below_threshold_count,
        "valid_handoff_target_step_count": valid_handoff_target_step_count,
        "valid_handoff_target_rate": round(float(valid_handoff_target_step_count) / float(total_steps), 6),
        "timing_active_step_count": int(round(sum(float(row.get("timing_active", 0.0)) for row in step_rows))),
        "temporal_urgency_mean": safe_mean(temporal_urgencies),
        "temporal_urgency_p75": safe_percentile(temporal_urgencies, 75.0),
        "countdown_steps_mean": safe_mean(countdown_values_on_valid_target),
        "countdown_steps_p25": safe_percentile(countdown_values_on_valid_target, 25.0),
        "countdown_steps_p75": safe_percentile(countdown_values_on_valid_target, 75.0),
        "prepare_window_score_mean": round(float(fmean(prepare_scores)), 6),
        "prepare_window_score_p75": safe_percentile(prepare_scores, 75.0),
        "high_prepare_step_count": int(round(sum(float(row.get("high_prepare", 0.0)) for row in step_rows))),
        "prepare_window_score_mean_on_valid_target": safe_mean(prepare_scores_on_valid_target),
        "prepare_window_score_p75_on_valid_target": safe_percentile(prepare_scores_on_valid_target, 75.0),
        "event_prepare_prob_mean": round(float(fmean(event_prepare_probs)), 6),
        "event_prepare_prob_p75": safe_percentile(event_prepare_probs, 75.0),
        "event_margin_mean": round(float(fmean(event_margins)), 6),
        "event_margin_p75": safe_percentile(event_margins, 75.0),
        "stochastic_event_prepare_rate_on_valid_target": safe_mean(
            [float(row.get("stochastic_event_prepare", 0.0)) for row in valid_target_rows]
        ),
        "deterministic_event_prepare_rate_on_valid_target": safe_mean(
            [float(row.get("deterministic_event_prepare", 0.0)) for row in valid_target_rows]
        ),
        "stochastic_event_prepare_rate_on_timing_active": safe_mean(
            [float(row.get("stochastic_event_prepare", 0.0)) for row in timing_active_rows]
        ),
        "deterministic_event_prepare_rate_on_timing_active": safe_mean(
            [float(row.get("deterministic_event_prepare", 0.0)) for row in timing_active_rows]
        ),
        "borderline_trigger_count": borderline_trigger_count,
        "override_trigger_count": override_trigger_count,
        "deterministic_temporal_smoothing_rate": round(float(forced_temporal_intervention_count) / float(total_steps), 6),
        "continuity_guard_trigger_count": continuity_guard_trigger_count,
        "continuity_guard_trigger_rate": round(float(continuity_guard_trigger_count) / float(total_steps), 6),
        "target_mismatch_guard_count": target_mismatch_guard_count,
        "guard_prefetch_to_prepare_count": guard_prefetch_to_prepare_count,
        "guard_hard_override_count": guard_hard_override_count,
        "action_projection_count": action_projection_count,
        "action_projection_rate": round(float(action_projection_count) / float(total_steps), 6),
        "invalid_action_attempt_count": invalid_action_attempt_count,
        "invalid_action_attempt_rate": round(float(invalid_action_attempt_count) / float(total_steps), 6),
        "guard_action_delta_count": guard_action_delta_count,
        "guard_action_delta_rate": round(float(guard_action_delta_count) / float(total_steps), 6),
        "cache_warm_start_guard_count": cache_warm_start_guard_count,
        "cache_warm_start_guard_rate": round(float(cache_warm_start_guard_count) / float(total_steps), 6),
        "predictive_prefetch_admission_guard_count": predictive_prefetch_admission_guard_count,
        "predictive_prefetch_admission_guard_rate": round(
            float(predictive_prefetch_admission_guard_count) / float(total_steps),
            6,
        ),
        "coverage_recovery_final_guard_count": coverage_recovery_final_guard_count,
        "coverage_recovery_final_guard_rate": round(
            float(coverage_recovery_final_guard_count) / float(total_steps),
            6,
        ),
        "backhaul_guard_count": backhaul_guard_count,
        "backhaul_guard_rate": round(float(backhaul_guard_count) / float(total_steps), 6),
        "backhaul_aware_policy_adjust_count": backhaul_aware_policy_adjust_count,
        "backhaul_aware_policy_adjust_rate": round(
            float(backhaul_aware_policy_adjust_count) / float(total_steps),
            6,
        ),
        "backhaul_aware_service_fill_bias_mean": safe_mean(backhaul_aware_service_fill_biases),
        "backhaul_aware_redundant_fill_penalty_mean": safe_mean(backhaul_aware_redundant_fill_penalties),
        "backhaul_aware_no_signal_penalty_mean": safe_mean(backhaul_aware_no_signal_penalties),
        "dag_frontier_size_mean": safe_mean(dag_frontier_sizes),
        "dag_critical_path_pressure_mean": safe_mean(dag_critical_path_pressures),
        "dag_current_node_dependency_pressure_mean": safe_mean(dag_dependency_pressures),
        "dag_remaining_nodes_mean": safe_mean(dag_remaining_nodes),
    }


def summarize_policy_alignment_samples(
    step_rows: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for step_row in step_rows:
        if len(samples) >= limit:
            break
        samples.append(build_policy_alignment_sample(step_row))
    return samples


def summarize_policy_diagnostic_history(update_eval_history: list[dict[str, Any]], agent_name: str) -> dict[str, Any]:
    history = [
        dict(item.get("aggregate_policy_diagnostics_by_agent", {}).get(agent_name, default_policy_diagnostics()))
        for item in update_eval_history
    ]
    if not history:
        return {
            "mean_update_eval_policy_diagnostics": default_policy_diagnostics(),
            "latest_update_eval_policy_diagnostics": default_policy_diagnostics(),
            "update_eval_policy_diagnostics_trend": {field_name: [] for field_name in POLICY_DIAGNOSTIC_FIELDS},
        }
    mean_payload = {
        field_name: round(float(fmean(float(item.get(field_name, 0.0)) for item in history)), 6)
        for field_name in POLICY_DIAGNOSTIC_FIELDS
    }
    return {
        "mean_update_eval_policy_diagnostics": mean_payload,
        "latest_update_eval_policy_diagnostics": dict(history[-1]),
        "update_eval_policy_diagnostics_trend": {
            field_name: [round(float(item.get(field_name, 0.0)), 6) for item in history]
            for field_name in POLICY_DIAGNOSTIC_FIELDS
        },
    }


def build_episode_metric(summary: dict[str, Any], episode_index: int, updated: bool) -> dict[str, Any]:
    metrics = summary["system_metrics"]
    validation_summary = summary["prefetch_validation_summary"]
    handoff_summary = summary["handoff_summary"]
    reward_diag = extract_reward_breakdown_means(summary)
    mechanism_diag = extract_mechanism_diagnostics(summary)
    learn_info = summary.get("agent_info", {}).get("learn_info", {})
    return {
        "episode_index": episode_index,
        "workflow_id": summary["run_info"].get("workflow_id"),
        "window_id": summary["run_info"].get("window_id"),
        "window_class": summary["run_info"].get("window_class"),
        "primary_vehicle_selection": summary["run_info"].get("primary_vehicle_selection", "stable_first"),
        "updated": updated,
        "total_reward": summary["reward_breakdown"]["total"]["sum"],
        "end_to_end_workflow_delay": metrics["end_to_end_workflow_delay"],
        "workflow_continuity_rate": metrics["workflow_continuity_rate"],
        "handoff_failure_rate": metrics["handoff_failure_rate"],
        "handoff_ready_ratio": metrics["handoff_ready_ratio"],
        "adapter_warm_hit_ratio": metrics["adapter_warm_hit_ratio"],
        "cross_rsu_cold_start_frequency": metrics["cross_rsu_cold_start_frequency"],
        "backhaul_traffic_cost": metrics["backhaul_traffic_cost"],
        "adapter_state_migration_overhead": metrics["adapter_state_migration_overhead"],
        "predictive_prefetch_precision": metrics["predictive_prefetch_precision"],
        "validated_predictive_prefetch_count": validation_summary["validated_predictive_prefetch_count"],
        "migration_during_handoff_count": handoff_summary["migration_during_handoff_count"],
        "handoff_ready_count": handoff_summary["handoff_ready_count"],
        "policy_entropy": learn_info.get("policy_entropy"),
        "approx_kl": learn_info.get("approx_kl"),
        "clip_fraction": learn_info.get("clip_fraction"),
        "value_mean": learn_info.get("value_mean"),
        "return_mean": learn_info.get("return_mean"),
        "explained_variance": learn_info.get("explained_variance"),
        **reward_diag,
        **mechanism_diag,
    }


def build_sa_ghmappo_profile_kwargs(profile: str) -> dict[str, Any]:
    if profile == "top_journal_mechanism_v6_strong_competition":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v1")
        kwargs.update(
            {
                "event_prepare_margin_boost": 0.55,
                "temporal_prepare_activation_threshold": 0.30,
                "event_logit_temperature_final": 0.74,
                "event_logit_sharpening_final_scale": 2.65,
                "mechanism_window_weight": 1.65,
                "prepare_action_prior_weight": 0.62,
                "heuristic_imitation_warmup_updates": 8,
                "heuristic_imitation_decay": 0.90,
                "mechanism_aux_coef_floor_after_update": 0.09,
                "mechanism_window_weight_floor_after_update": 1.60,
                "mechanism_entropy_floor_after_update": 0.0007,
                "predictive_prepare_hard_override_enabled": False,
                "latency_fallback_bias_enabled": False,
                "cache_warm_start_guard_max_prefetch_countdown": 6.0,
                "predictive_prefetch_admission_guard_enabled": True,
                "predictive_prefetch_admission_min_confidence": 0.55,
                "predictive_prefetch_admission_require_distinct_next": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v7_latency_fallback":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v6_strong_competition")
        kwargs.update(
            {
                "latency_fallback_bias_enabled": True,
                "latency_fallback_bias_strength": 1.20,
                "latency_fallback_confidence_floor": 0.62,
                "latency_fallback_slow_suppression_strength": 1.20,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v8_strict_full":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v6_strong_competition")
        kwargs.update(
            {
                "latency_fallback_bias_enabled": False,
                "latency_fallback_bias_strength": 0.0,
                "latency_fallback_confidence_floor": 0.0,
                "latency_fallback_slow_suppression_strength": 0.0,
                "steady_rsu_bias_enabled": True,
                "steady_rsu_bias_strength": 1.20,
                "steady_rsu_confidence_floor": 0.62,
                "continuity_guard_logit_penalty": 0.80,
                "continuity_guard_prepare_boost": 1.00,
                "continuity_guard_confidence_threshold": 0.65,
                "continuity_guard_prepare_score_threshold": 0.35,
                "predictive_prefetch_admission_min_confidence": 0.62,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v9_pareto_safe":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v8_strict_full")
        kwargs.update(
            {
                "steady_rsu_bias_strength": 1.35,
                "steady_rsu_confidence_floor": 0.66,
                "continuity_guard_logit_penalty": 0.70,
                "continuity_guard_prepare_boost": 0.90,
                "continuity_guard_confidence_threshold": 0.68,
                "continuity_guard_prepare_score_threshold": 0.38,
                "predictive_prefetch_admission_min_confidence": 0.68,
                "predictive_prefetch_admission_require_distinct_next": True,
                "backhaul_guard_enabled": True,
                "backhaul_guard_max_reactive_fills_per_adapter": 1,
                "cache_warm_start_guard_max_prefetch_countdown": 5.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v10_mappo_rl":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v9_pareto_safe")
        kwargs.update(
            {
                "head_credit_enabled": True,
                "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
                "slow_policy_credit_floor": 0.25,
                "fast_policy_credit_floor": 0.10,
                "event_policy_credit_floor": 0.12,
                "slow_entropy_coef_scale": 1.25,
                "fast_entropy_coef_scale": 1.0,
                "event_entropy_coef_scale": 1.35,
                "slow_entropy_credit_floor": 0.20,
                "fast_entropy_credit_floor": 0.08,
                "event_entropy_credit_floor": 0.12,
                "event_advantage_blend": 0.85,
                "event_actor_loss_extra_gain": 1.05,
                "event_logit_temperature": 1.20,
                "event_logit_temperature_final": 0.90,
                "event_logit_sharpening_final_scale": 1.55,
                "event_logit_sharpening_timing_gain": 0.55,
                "heuristic_imitation_coef": 0.04,
                "heuristic_imitation_warmup_updates": 4,
                "heuristic_imitation_decay": 0.70,
                "mechanism_aux_coef": 0.035,
                "mechanism_window_weight": 1.20,
                "prepare_action_prior_weight": 0.20,
                "mechanism_entropy_coef": 0.0004,
                "mechanism_aux_coef_floor_after_update": 0.035,
                "mechanism_window_weight_floor_after_update": 1.20,
                "mechanism_entropy_floor_after_update": 0.0004,
                "mechanism_aux_current_cache_fill_enabled": False,
                "train_epochs": 8,
                "target_kl": 0.012,
                "kl_early_stop_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v11_mappo_reward":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v8_strict_full")
        kwargs.update(
            {
                "head_credit_enabled": True,
                "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
                "slow_policy_credit_floor": 0.25,
                "fast_policy_credit_floor": 0.12,
                "event_policy_credit_floor": 0.12,
                "slow_entropy_coef_scale": 1.25,
                "fast_entropy_coef_scale": 1.00,
                "event_entropy_coef_scale": 1.35,
                "slow_entropy_credit_floor": 0.20,
                "fast_entropy_credit_floor": 0.09,
                "event_entropy_credit_floor": 0.12,
                "event_advantage_blend": 0.88,
                "heuristic_imitation_coef": 0.08,
                "heuristic_imitation_warmup_updates": 6,
                "heuristic_imitation_decay": 0.84,
                "mechanism_aux_coef": 0.08,
                "mechanism_window_weight": 1.45,
                "prepare_action_prior_weight": 0.55,
                "mechanism_aux_coef_floor_after_update": 0.08,
                "mechanism_window_weight_floor_after_update": 1.45,
                "train_epochs": 7,
                "target_kl": 0.012,
                "idle_popularity_fallback_enabled": True,
                "idle_popularity_fallback_only_vehicle_fallback": True,
                "idle_popularity_prefetch_threshold": 2,
                "idle_popularity_no_rsu_local_fallback_enabled": False,
                "idle_popularity_no_rsu_local_requires_low_context": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v12_learned_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v11_mappo_reward")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.06,
                "heuristic_imitation_warmup_updates": 4,
                "heuristic_imitation_decay": 0.76,
                "option_gate_enabled": True,
                "option_gate_count": 4,
                "option_gate_loss_coef": 0.28,
                "option_gate_entropy_coef": 0.0012,
                "option_gate_prior_coef": 0.045,
                "option_gate_prior_warmup_updates": 5,
                "option_gate_prior_decay": 0.78,
                "option_gate_prior_logit_bias": 0.6,
                "option_gate_log_prob_weight": 0.35,
                "option_gate_context_prior_enabled": True,
                "option_gate_deterministic_prior_margin": 0.18,
                "option_gate_idle_prior_enabled": True,
                "idle_popularity_fallback_enabled": True,
                "idle_popularity_fallback_only_vehicle_fallback": True,
                "idle_popularity_no_rsu_local_fallback_enabled": False,
                "idle_popularity_no_rsu_local_requires_low_context": True,
                "train_epochs": 7,
                "target_kl": 0.012,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v13_prd_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v12_learned_option")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.045,
                "heuristic_imitation_warmup_updates": 3,
                "heuristic_imitation_decay": 0.70,
                "mechanism_aux_coef": 0.07,
                "mechanism_window_weight": 1.55,
                "prepare_action_prior_weight": 0.62,
                "mechanism_aux_coef_floor_after_update": 0.07,
                "mechanism_window_weight_floor_after_update": 1.55,
                "option_gate_loss_coef": 0.32,
                "option_gate_prior_coef": 0.035,
                "option_gate_prior_warmup_updates": 4,
                "option_gate_prior_decay": 0.72,
                "option_gate_prior_logit_bias": 0.45,
                "option_gate_log_prob_weight": 0.42,
                "option_gate_deterministic_prior_margin": 0.08,
                "option_gate_mechanism_preserve_enabled": True,
                "option_gate_prd_enabled": True,
                "option_gate_prd_coef": 0.24,
                "option_gate_prd_clip": 1.6,
                "event_prd_advantage_enabled": True,
                "event_prd_advantage_coef": 0.38,
                "event_prd_advantage_clip": 1.8,
                "train_epochs": 7,
                "target_kl": 0.012,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v14_net_utility_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v13_prd_option")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.035,
                "heuristic_imitation_warmup_updates": 2,
                "heuristic_imitation_decay": 0.62,
                "mechanism_aux_coef": 0.065,
                "mechanism_window_weight": 1.50,
                "prepare_action_prior_weight": 0.58,
                "mechanism_aux_coef_floor_after_update": 0.065,
                "mechanism_window_weight_floor_after_update": 1.50,
                "option_gate_loss_coef": 0.34,
                "option_gate_prior_coef": 0.026,
                "option_gate_prior_warmup_updates": 3,
                "option_gate_prior_decay": 0.68,
                "option_gate_prior_logit_bias": 0.35,
                "option_gate_log_prob_weight": 0.48,
                "option_gate_deterministic_prior_margin": 0.05,
                "event_prd_advantage_coef": 0.42,
                "event_prd_advantage_clip": 1.6,
                "option_gate_prd_coef": 0.28,
                "option_gate_prd_clip": 1.5,
                "net_utility_prd_enabled": True,
                "net_utility_backhaul_coef": 0.16,
                "net_utility_migration_coef": 0.22,
                "net_utility_expired_prefetch_coef": 0.55,
                "net_utility_idle_prefetch_penalty": 0.65,
                "net_utility_success_bonus": 0.16,
                "net_utility_backhaul_normalizer": 64.0,
                "net_utility_cost_dual_enabled": True,
                "net_utility_cost_dual_lr": 0.08,
                "net_utility_cost_target": 0.20,
                "net_utility_cost_dual_max": 1.25,
                "net_utility_cost_dual_initial": 0.0,
                "net_utility_option_termination_enabled": True,
                "train_epochs": 7,
                "target_kl": 0.010,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v15_terminal_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v13_prd_option")
        kwargs.update(
            {
                "net_utility_prd_enabled": False,
                "net_utility_cost_dual_enabled": False,
                "net_utility_cost_dual_initial": 0.0,
                "net_utility_option_termination_enabled": True,
                "train_epochs": 7,
                "target_kl": 0.012,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v16_conservative_terminal_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v15_terminal_option")
        kwargs.update(
            {
                "net_utility_option_termination_conservative_enabled": True,
                "net_utility_option_termination_max_timing_support": 1.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v17_dag_aware_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v16_conservative_terminal_option")
        kwargs.update(
            {
                "dag_aware_option_termination_enabled": True,
                "dag_aware_option_min_critical_path": 6,
                "dag_aware_option_short_workflow_max_nodes": 12,
                "dag_aware_option_branching_successors": 3,
                "dag_aware_idle_prefetch_confidence_floor": 0.65,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v18_counterfactual_option":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v17_dag_aware_option")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.035,
                "heuristic_imitation_warmup_updates": 2,
                "heuristic_imitation_decay": 0.64,
                "mechanism_aux_coef": 0.062,
                "mechanism_window_weight": 1.48,
                "prepare_action_prior_weight": 0.56,
                "mechanism_aux_coef_floor_after_update": 0.062,
                "mechanism_window_weight_floor_after_update": 1.48,
                "option_gate_mechanism_preserve_enabled": False,
                "option_gate_loss_coef": 0.36,
                "option_gate_prior_coef": 0.020,
                "option_gate_prior_warmup_updates": 3,
                "option_gate_prior_decay": 0.64,
                "option_gate_prior_logit_bias": 0.32,
                "option_gate_log_prob_weight": 0.54,
                "option_gate_deterministic_prior_margin": 0.06,
                "option_gate_prd_coef": 0.30,
                "option_gate_prd_clip": 1.45,
                "option_gate_counterfactual_prd_enabled": True,
                "option_gate_counterfactual_coef": 0.38,
                "option_gate_counterfactual_clip": 1.25,
                "event_prd_advantage_coef": 0.44,
                "event_prd_advantage_clip": 1.55,
                "train_epochs": 7,
                "target_kl": 0.011,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v19_handoff_risk_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v17_dag_aware_option")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.038,
                "heuristic_imitation_warmup_updates": 2,
                "heuristic_imitation_decay": 0.66,
                "mechanism_aux_coef": 0.068,
                "mechanism_window_weight": 1.52,
                "prepare_action_prior_weight": 0.60,
                "mechanism_aux_coef_floor_after_update": 0.068,
                "mechanism_window_weight_floor_after_update": 1.52,
                "option_gate_loss_coef": 0.34,
                "option_gate_prior_coef": 0.024,
                "option_gate_prior_warmup_updates": 3,
                "option_gate_prior_decay": 0.66,
                "option_gate_prior_logit_bias": 0.34,
                "option_gate_log_prob_weight": 0.50,
                "option_gate_deterministic_prior_margin": 0.05,
                "event_prd_advantage_coef": 0.42,
                "event_prd_advantage_clip": 1.55,
                "handoff_risk_prd_enabled": True,
                "handoff_risk_event_coef": 0.38,
                "handoff_risk_option_coef": 0.44,
                "handoff_risk_clip": 1.35,
                "handoff_risk_failure_penalty": 1.15,
                "handoff_risk_ready_bonus": 0.85,
                "handoff_risk_prepare_bonus": 0.34,
                "handoff_risk_unprepared_penalty": 0.44,
                "handoff_risk_confidence_threshold": 0.56,
                "handoff_risk_cost_dual_enabled": True,
                "handoff_risk_cost_dual_lr": 0.10,
                "handoff_risk_cost_target": 0.05,
                "handoff_risk_cost_dual_max": 1.40,
                "handoff_risk_cost_dual_initial": 0.0,
                "train_epochs": 7,
                "target_kl": 0.011,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v20_idle_execution_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v19_handoff_risk_prd")
        kwargs.update(
            {
                "idle_execution_prd_enabled": True,
                "idle_execution_policy_coef": 0.30,
                "idle_execution_option_coef": 0.42,
                "idle_execution_clip": 1.25,
                "idle_execution_current_rsu_delay_coef": 0.42,
                "idle_execution_local_bonus": 0.30,
                "idle_execution_mechanism_penalty": 0.34,
                "idle_execution_timing_threshold": 0.30,
                "idle_execution_mechanism_preserve_bonus": 0.20,
                "option_gate_loss_coef": 0.36,
                "option_gate_prior_coef": 0.028,
                "option_gate_prior_decay": 0.68,
                "option_gate_log_prob_weight": 0.54,
                "option_gate_deterministic_prior_margin": 0.06,
                "event_prd_advantage_coef": 0.40,
                "handoff_risk_event_coef": 0.34,
                "handoff_risk_option_coef": 0.40,
                "train_epochs": 7,
                "target_kl": 0.011,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v21_efficiency_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v20_idle_execution_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.034,
                "heuristic_imitation_decay": 0.62,
                "mechanism_aux_coef": 0.058,
                "mechanism_window_weight": 1.42,
                "prepare_action_prior_weight": 0.50,
                "mechanism_aux_coef_floor_after_update": 0.052,
                "mechanism_window_weight_floor_after_update": 1.36,
                "option_gate_loss_coef": 0.38,
                "option_gate_prior_coef": 0.022,
                "option_gate_prior_decay": 0.62,
                "option_gate_log_prob_weight": 0.58,
                "event_prd_advantage_coef": 0.36,
                "handoff_risk_event_coef": 0.30,
                "handoff_risk_option_coef": 0.36,
                "idle_execution_policy_coef": 0.46,
                "idle_execution_option_coef": 0.64,
                "idle_execution_clip": 1.35,
                "idle_execution_current_rsu_delay_coef": 0.58,
                "idle_execution_local_bonus": 0.46,
                "idle_execution_mechanism_penalty": 0.48,
                "idle_execution_timing_threshold": 0.46,
                "idle_execution_mechanism_preserve_bonus": 0.16,
                "net_utility_prd_enabled": True,
                "net_utility_backhaul_coef": 0.12,
                "net_utility_migration_coef": 0.18,
                "net_utility_expired_prefetch_coef": 0.48,
                "net_utility_idle_prefetch_penalty": 0.58,
                "net_utility_success_bonus": 0.18,
                "net_utility_backhaul_normalizer": 64.0,
                "net_utility_cost_dual_enabled": True,
                "net_utility_cost_dual_lr": 0.06,
                "net_utility_cost_target": 0.10,
                "net_utility_cost_dual_max": 1.20,
                "net_utility_cost_dual_initial": 0.0,
                "net_utility_option_termination_enabled": True,
                "net_utility_option_termination_conservative_enabled": True,
                "net_utility_option_termination_max_timing_support": 0.85,
                "train_epochs": 7,
                "target_kl": 0.010,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v22_validated_utility_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v21_efficiency_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.026,
                "heuristic_imitation_decay": 0.55,
                "mechanism_aux_coef": 0.042,
                "mechanism_window_weight": 1.18,
                "prepare_action_prior_weight": 0.34,
                "mechanism_aux_coef_floor_after_update": 0.030,
                "mechanism_window_weight_floor_after_update": 1.10,
                "option_gate_loss_coef": 0.34,
                "option_gate_prior_coef": 0.014,
                "option_gate_prior_decay": 0.55,
                "option_gate_log_prob_weight": 0.54,
                "option_gate_deterministic_prior_margin": 0.04,
                "event_prd_advantage_coef": 0.28,
                "handoff_risk_event_coef": 0.24,
                "handoff_risk_option_coef": 0.30,
                "handoff_risk_prepare_bonus": 0.14,
                "idle_execution_policy_coef": 0.62,
                "idle_execution_option_coef": 0.82,
                "idle_execution_clip": 1.55,
                "idle_execution_current_rsu_delay_coef": 0.78,
                "idle_execution_local_bonus": 0.64,
                "idle_execution_mechanism_penalty": 0.72,
                "idle_execution_timing_threshold": 0.52,
                "idle_execution_mechanism_preserve_bonus": 0.06,
                "net_utility_backhaul_coef": 0.22,
                "net_utility_migration_coef": 0.24,
                "net_utility_expired_prefetch_coef": 0.62,
                "net_utility_idle_prefetch_penalty": 0.78,
                "net_utility_failed_mechanism_penalty": 0.84,
                "net_utility_failed_mechanism_backhaul_coef": 0.36,
                "net_utility_success_bonus": 0.28,
                "net_utility_cost_dual_lr": 0.08,
                "net_utility_cost_target": 0.08,
                "net_utility_cost_dual_max": 1.60,
                "net_utility_option_termination_max_timing_support": 0.72,
                "train_epochs": 7,
                "target_kl": 0.009,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v23_counterfactual_constrained_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v22_validated_utility_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.032,
                "heuristic_imitation_decay": 0.60,
                "mechanism_aux_coef": 0.070,
                "mechanism_window_weight": 1.58,
                "prepare_action_prior_weight": 0.62,
                "mechanism_aux_coef_floor_after_update": 0.058,
                "mechanism_window_weight_floor_after_update": 1.42,
                "option_gate_mechanism_preserve_enabled": False,
                "option_gate_loss_coef": 0.42,
                "option_gate_prior_coef": 0.030,
                "option_gate_prior_warmup_updates": 3,
                "option_gate_prior_decay": 0.64,
                "option_gate_prior_logit_bias": 0.40,
                "option_gate_log_prob_weight": 0.66,
                "option_gate_deterministic_prior_margin": 0.10,
                "option_gate_counterfactual_prd_enabled": True,
                "option_gate_counterfactual_coef": 0.46,
                "option_gate_counterfactual_clip": 1.45,
                "event_prd_advantage_coef": 0.46,
                "handoff_risk_event_coef": 0.32,
                "handoff_risk_option_coef": 0.42,
                "idle_execution_policy_coef": 0.72,
                "idle_execution_option_coef": 0.90,
                "idle_execution_clip": 1.60,
                "idle_execution_current_rsu_delay_coef": 0.84,
                "idle_execution_local_bonus": 0.70,
                "idle_execution_mechanism_penalty": 0.78,
                "idle_execution_timing_threshold": 0.54,
                "idle_execution_mechanism_preserve_bonus": 0.04,
                "net_utility_backhaul_coef": 0.18,
                "net_utility_migration_coef": 0.20,
                "net_utility_expired_prefetch_coef": 0.56,
                "net_utility_idle_prefetch_penalty": 0.82,
                "net_utility_failed_mechanism_penalty": 0.42,
                "net_utility_failed_mechanism_backhaul_coef": 0.16,
                "net_utility_mechanism_window_failed_penalty_scale": 0.10,
                "net_utility_success_bonus": 0.30,
                "net_utility_cost_dual_lr": 0.07,
                "net_utility_cost_target": 0.09,
                "net_utility_cost_dual_max": 1.55,
                "counterfactual_teacher_prd_enabled": True,
                "counterfactual_teacher_event_coef": 0.58,
                "counterfactual_teacher_option_coef": 0.72,
                "counterfactual_teacher_clip": 2.0,
                "counterfactual_teacher_mechanism_bonus": 0.92,
                "counterfactual_teacher_missed_prepare_penalty": 0.78,
                "counterfactual_teacher_local_bonus": 0.42,
                "counterfactual_teacher_current_rsu_penalty": 0.16,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.82,
                "train_epochs": 7,
                "target_kl": 0.0085,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v24_tail_risk_constrained_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v23_counterfactual_constrained_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.030,
                "heuristic_imitation_decay": 0.58,
                "mechanism_aux_coef": 0.052,
                "mechanism_window_weight": 1.36,
                "prepare_action_prior_weight": 0.46,
                "mechanism_aux_coef_floor_after_update": 0.044,
                "mechanism_window_weight_floor_after_update": 1.24,
                "option_gate_mechanism_preserve_enabled": True,
                "option_gate_loss_coef": 0.36,
                "option_gate_prior_coef": 0.020,
                "option_gate_prior_warmup_updates": 3,
                "option_gate_prior_decay": 0.58,
                "option_gate_prior_logit_bias": 0.26,
                "option_gate_log_prob_weight": 0.58,
                "option_gate_deterministic_prior_margin": 0.06,
                "option_gate_counterfactual_prd_enabled": True,
                "option_gate_counterfactual_coef": 0.34,
                "option_gate_counterfactual_clip": 1.30,
                "event_prd_advantage_coef": 0.34,
                "handoff_risk_event_coef": 0.36,
                "handoff_risk_option_coef": 0.42,
                "handoff_risk_failure_penalty": 1.35,
                "handoff_risk_ready_bonus": 0.86,
                "handoff_risk_prepare_bonus": 0.22,
                "handoff_risk_unprepared_penalty": 0.44,
                "idle_execution_policy_coef": 0.64,
                "idle_execution_option_coef": 0.82,
                "idle_execution_clip": 1.45,
                "idle_execution_current_rsu_delay_coef": 0.80,
                "idle_execution_local_bonus": 0.64,
                "idle_execution_mechanism_penalty": 0.78,
                "idle_execution_timing_threshold": 0.52,
                "idle_execution_mechanism_preserve_bonus": 0.05,
                "net_utility_backhaul_coef": 0.20,
                "net_utility_migration_coef": 0.24,
                "net_utility_expired_prefetch_coef": 0.62,
                "net_utility_idle_prefetch_penalty": 0.82,
                "net_utility_failed_mechanism_penalty": 0.78,
                "net_utility_failed_mechanism_backhaul_coef": 0.30,
                "net_utility_mechanism_window_failed_penalty_scale": 0.72,
                "net_utility_success_bonus": 0.28,
                "net_utility_cost_dual_lr": 0.08,
                "net_utility_cost_target": 0.08,
                "net_utility_cost_dual_max": 1.70,
                "counterfactual_teacher_event_coef": 0.38,
                "counterfactual_teacher_option_coef": 0.46,
                "counterfactual_teacher_clip": 1.60,
                "counterfactual_teacher_mechanism_bonus": 0.64,
                "counterfactual_teacher_missed_prepare_penalty": 0.72,
                "counterfactual_teacher_local_bonus": 0.42,
                "counterfactual_teacher_current_rsu_penalty": 0.18,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.88,
                "tail_risk_prd_enabled": True,
                "tail_risk_policy_coef": 0.52,
                "tail_risk_event_coef": 0.36,
                "tail_risk_option_coef": 0.46,
                "tail_risk_clip": 1.80,
                "tail_risk_quantile": 0.25,
                "tail_risk_reward_shortfall_coef": 0.72,
                "tail_risk_service_coef": 0.80,
                "tail_risk_continuity_coef": 1.10,
                "tail_risk_handoff_failure_coef": 1.35,
                "tail_risk_failed_mechanism_coef": 0.82,
                "tail_risk_redundant_mechanism_coef": 0.70,
                "tail_risk_success_credit": 0.18,
                "train_epochs": 7,
                "target_kl": 0.008,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v25_opportunity_risk_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v24_tail_risk_constrained_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.026,
                "heuristic_imitation_decay": 0.54,
                "mechanism_aux_coef": 0.056,
                "mechanism_window_weight": 1.40,
                "prepare_action_prior_weight": 0.50,
                "mechanism_aux_coef_floor_after_update": 0.046,
                "mechanism_window_weight_floor_after_update": 1.26,
                "option_gate_loss_coef": 0.38,
                "option_gate_prior_coef": 0.024,
                "option_gate_prior_decay": 0.56,
                "option_gate_prior_logit_bias": 0.30,
                "option_gate_log_prob_weight": 0.62,
                "option_gate_deterministic_prior_margin": 0.07,
                "option_gate_counterfactual_coef": 0.38,
                "option_gate_counterfactual_clip": 1.35,
                "event_prd_advantage_coef": 0.36,
                "handoff_risk_event_coef": 0.34,
                "handoff_risk_option_coef": 0.40,
                "idle_execution_policy_coef": 0.70,
                "idle_execution_option_coef": 0.88,
                "idle_execution_clip": 1.55,
                "idle_execution_current_rsu_delay_coef": 0.74,
                "idle_execution_local_bonus": 0.70,
                "idle_execution_mechanism_penalty": 0.72,
                "idle_execution_timing_threshold": 0.54,
                "idle_execution_mechanism_preserve_bonus": 0.07,
                "net_utility_backhaul_coef": 0.28,
                "net_utility_migration_coef": 0.24,
                "net_utility_expired_prefetch_coef": 0.60,
                "net_utility_idle_prefetch_penalty": 0.74,
                "net_utility_failed_mechanism_penalty": 0.72,
                "net_utility_failed_mechanism_backhaul_coef": 0.42,
                "net_utility_mechanism_window_failed_penalty_scale": 0.68,
                "net_utility_success_bonus": 0.34,
                "net_utility_cost_dual_lr": 0.08,
                "net_utility_cost_target": 0.07,
                "net_utility_cost_dual_max": 1.75,
                "counterfactual_teacher_event_coef": 0.42,
                "counterfactual_teacher_option_coef": 0.52,
                "counterfactual_teacher_clip": 1.70,
                "counterfactual_teacher_mechanism_bonus": 0.72,
                "counterfactual_teacher_missed_prepare_penalty": 0.68,
                "counterfactual_teacher_local_bonus": 0.48,
                "counterfactual_teacher_current_rsu_penalty": 0.14,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.84,
                "tail_risk_policy_coef": 0.40,
                "tail_risk_event_coef": 0.28,
                "tail_risk_option_coef": 0.34,
                "tail_risk_clip": 1.65,
                "tail_risk_quantile": 0.22,
                "tail_risk_reward_shortfall_coef": 0.56,
                "tail_risk_service_coef": 0.66,
                "tail_risk_continuity_coef": 0.95,
                "tail_risk_handoff_failure_coef": 1.25,
                "tail_risk_failed_mechanism_coef": 0.72,
                "tail_risk_redundant_mechanism_coef": 0.56,
                "tail_risk_success_credit": 0.22,
                "opportunity_prd_enabled": True,
                "opportunity_policy_coef": 0.42,
                "opportunity_event_coef": 0.24,
                "opportunity_option_coef": 0.36,
                "opportunity_clip": 1.65,
                "opportunity_reward_quantile": 0.60,
                "opportunity_reward_surplus_coef": 0.72,
                "opportunity_service_success_coef": 0.54,
                "opportunity_cache_hit_coef": 0.36,
                "opportunity_continuity_coef": 0.42,
                "opportunity_current_rsu_efficiency_coef": 0.36,
                "opportunity_local_fallback_coef": 0.30,
                "opportunity_backhaul_penalty_coef": 0.34,
                "opportunity_delay_penalty_coef": 0.24,
                "opportunity_failed_service_penalty_coef": 0.44,
                "opportunity_mechanism_success_bonus": 0.32,
                "train_epochs": 8,
                "target_kl": 0.0085,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v22_validated_utility_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.026,
                "heuristic_imitation_decay": 0.56,
                "mechanism_aux_coef": 0.048,
                "mechanism_window_weight": 1.26,
                "prepare_action_prior_weight": 0.40,
                "mechanism_aux_coef_floor_after_update": 0.036,
                "mechanism_window_weight_floor_after_update": 1.16,
                "option_gate_mechanism_preserve_enabled": True,
                "option_gate_loss_coef": 0.35,
                "option_gate_prior_coef": 0.017,
                "option_gate_prior_warmup_updates": 2,
                "option_gate_prior_decay": 0.56,
                "option_gate_prior_logit_bias": 0.18,
                "option_gate_log_prob_weight": 0.56,
                "option_gate_deterministic_prior_margin": 0.05,
                "option_gate_counterfactual_prd_enabled": True,
                "option_gate_counterfactual_coef": 0.28,
                "option_gate_counterfactual_clip": 1.20,
                "event_prd_advantage_coef": 0.32,
                "handoff_risk_event_coef": 0.26,
                "handoff_risk_option_coef": 0.34,
                "handoff_risk_prepare_bonus": 0.18,
                "idle_execution_policy_coef": 0.66,
                "idle_execution_option_coef": 0.86,
                "idle_execution_clip": 1.55,
                "idle_execution_current_rsu_delay_coef": 0.80,
                "idle_execution_local_bonus": 0.66,
                "idle_execution_mechanism_penalty": 0.80,
                "idle_execution_timing_threshold": 0.55,
                "idle_execution_mechanism_preserve_bonus": 0.05,
                "net_utility_backhaul_coef": 0.24,
                "net_utility_migration_coef": 0.24,
                "net_utility_expired_prefetch_coef": 0.62,
                "net_utility_idle_prefetch_penalty": 0.82,
                "net_utility_failed_mechanism_penalty": 0.90,
                "net_utility_failed_mechanism_backhaul_coef": 0.42,
                "net_utility_mechanism_window_failed_penalty_scale": 0.55,
                "net_utility_success_bonus": 0.32,
                "net_utility_cost_dual_lr": 0.08,
                "net_utility_cost_target": 0.08,
                "net_utility_cost_dual_max": 1.70,
                "counterfactual_teacher_prd_enabled": True,
                "counterfactual_teacher_event_coef": 0.32,
                "counterfactual_teacher_option_coef": 0.40,
                "counterfactual_teacher_clip": 1.30,
                "counterfactual_teacher_mechanism_bonus": 0.52,
                "counterfactual_teacher_missed_prepare_penalty": 0.40,
                "counterfactual_teacher_local_bonus": 0.22,
                "counterfactual_teacher_current_rsu_penalty": 0.06,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.92,
                "tail_risk_prd_enabled": False,
                "opportunity_prd_enabled": False,
                "train_epochs": 7,
                "target_kl": 0.0085,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v27_conservative_advantage_imitation":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v20_idle_execution_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.045,
                "heuristic_imitation_warmup_updates": 6,
                "heuristic_imitation_decay": 0.82,
                "conservative_imitation_enabled": True,
                "conservative_imitation_reward_quantile": 0.38,
                "conservative_imitation_min_weight": 0.08,
                "conservative_imitation_max_weight": 1.45,
                "conservative_imitation_shortfall_coef": 0.70,
                "conservative_imitation_failure_coef": 0.85,
                "conservative_imitation_mismatch_coef": 0.22,
                "conservative_imitation_success_decay": 0.20,
                "mechanism_aux_coef": 0.052,
                "mechanism_window_weight": 1.32,
                "prepare_action_prior_weight": 0.46,
                "mechanism_aux_coef_floor_after_update": 0.040,
                "mechanism_window_weight_floor_after_update": 1.18,
                "option_gate_loss_coef": 0.34,
                "option_gate_prior_coef": 0.022,
                "option_gate_prior_decay": 0.64,
                "option_gate_log_prob_weight": 0.52,
                "option_gate_deterministic_prior_margin": 0.05,
                "event_prd_advantage_coef": 0.36,
                "handoff_risk_event_coef": 0.32,
                "handoff_risk_option_coef": 0.38,
                "idle_execution_policy_coef": 0.28,
                "idle_execution_option_coef": 0.40,
                "idle_execution_clip": 1.20,
                "idle_execution_current_rsu_delay_coef": 0.40,
                "idle_execution_local_bonus": 0.30,
                "idle_execution_mechanism_penalty": 0.36,
                "idle_execution_timing_threshold": 0.32,
                "idle_execution_mechanism_preserve_bonus": 0.18,
                "tail_risk_prd_enabled": False,
                "opportunity_prd_enabled": False,
                "counterfactual_teacher_prd_enabled": False,
                "train_epochs": 7,
                "target_kl": 0.010,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v28_credit_focal_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v20_idle_execution_prd")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.035,
                "heuristic_imitation_warmup_updates": 4,
                "heuristic_imitation_decay": 0.78,
                "conservative_imitation_enabled": True,
                "conservative_imitation_reward_quantile": 0.36,
                "conservative_imitation_min_weight": 0.06,
                "conservative_imitation_max_weight": 1.35,
                "conservative_imitation_shortfall_coef": 0.55,
                "conservative_imitation_failure_coef": 0.68,
                "conservative_imitation_mismatch_coef": 0.18,
                "conservative_imitation_success_decay": 0.18,
                "mechanism_aux_coef": 0.075,
                "mechanism_window_weight": 1.65,
                "prepare_action_prior_weight": 0.62,
                "mechanism_entropy_coef": 0.0009,
                "mechanism_retention_start_update": 4,
                "mechanism_aux_coef_floor_after_update": 0.070,
                "mechanism_window_weight_floor_after_update": 1.45,
                "mechanism_entropy_floor_after_update": 0.0010,
                "mechanism_aux_current_cache_fill_enabled": False,
                "mechanism_credit_prd_enabled": True,
                "mechanism_credit_policy_coef": 0.30,
                "mechanism_credit_event_coef": 0.85,
                "mechanism_credit_option_coef": 0.42,
                "mechanism_credit_clip": 1.80,
                "mechanism_credit_success_bonus": 1.00,
                "mechanism_credit_prepare_bonus": 0.55,
                "mechanism_credit_ready_bonus": 0.85,
                "mechanism_credit_prefetch_hit_bonus": 0.65,
                "mechanism_credit_miss_penalty": 0.55,
                "mechanism_credit_false_positive_penalty": 0.38,
                "mechanism_credit_min_context": 0.15,
                "mechanism_focal_aux_enabled": True,
                "mechanism_focal_gamma": 1.40,
                "event_prd_advantage_coef": 0.48,
                "handoff_risk_event_coef": 0.52,
                "handoff_risk_option_coef": 0.46,
                "handoff_risk_ready_bonus": 0.92,
                "handoff_risk_prepare_bonus": 0.32,
                "handoff_risk_unprepared_penalty": 0.46,
                "idle_execution_policy_coef": 0.22,
                "idle_execution_option_coef": 0.28,
                "idle_execution_clip": 1.20,
                "idle_execution_current_rsu_delay_coef": 0.38,
                "idle_execution_local_bonus": 0.28,
                "idle_execution_mechanism_penalty": 0.34,
                "idle_execution_timing_threshold": 0.32,
                "idle_execution_mechanism_preserve_bonus": 0.22,
                "option_gate_loss_coef": 0.36,
                "option_gate_prior_coef": 0.025,
                "option_gate_prior_decay": 0.66,
                "option_gate_log_prob_weight": 0.58,
                "option_gate_deterministic_prior_margin": 0.06,
                "tail_risk_prd_enabled": False,
                "opportunity_prd_enabled": False,
                "counterfactual_teacher_prd_enabled": False,
                "train_epochs": 7,
                "target_kl": 0.011,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v29_dt_fused_credit_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v28_credit_focal_mappo")
        kwargs.update(
            {
                "digital_twin_handoff_fusion_enabled": True,
                "digital_twin_handoff_slow_scale": 0.38,
                "digital_twin_handoff_fast_scale": 0.48,
                "digital_twin_handoff_event_scale": 0.95,
                "digital_twin_handoff_critic_scale": 0.78,
                "mechanism_credit_policy_coef": 0.36,
                "mechanism_credit_event_coef": 0.98,
                "mechanism_credit_option_coef": 0.50,
                "mechanism_credit_success_bonus": 1.10,
                "mechanism_credit_prepare_bonus": 0.62,
                "mechanism_credit_ready_bonus": 0.95,
                "mechanism_credit_prefetch_hit_bonus": 0.70,
                "mechanism_credit_miss_penalty": 0.50,
                "mechanism_credit_false_positive_penalty": 0.34,
                "mechanism_aux_coef": 0.082,
                "mechanism_window_weight": 1.72,
                "prepare_action_prior_weight": 0.66,
                "mechanism_aux_coef_floor_after_update": 0.076,
                "mechanism_window_weight_floor_after_update": 1.52,
                "handoff_risk_event_coef": 0.56,
                "handoff_risk_option_coef": 0.50,
                "handoff_risk_ready_bonus": 1.00,
                "handoff_risk_prepare_bonus": 0.36,
                "idle_execution_mechanism_preserve_bonus": 0.26,
                "option_gate_loss_coef": 0.38,
                "option_gate_log_prob_weight": 0.62,
                "target_kl": 0.010,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v30_dt_prior_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v29_dt_fused_credit_mappo")
        kwargs.update(
            {
                "digital_twin_policy_prior_enabled": True,
                "digital_twin_policy_prior_logit_bias": 3.20,
                "digital_twin_policy_prior_event_scale": 1.22,
                "digital_twin_policy_prior_slow_scale": 0.82,
                "digital_twin_policy_prior_fast_scale": 0.20,
                "digital_twin_policy_prior_prepare_threshold": 0.16,
                "digital_twin_policy_prior_prefetch_threshold": 0.20,
                "digital_twin_policy_prior_confidence_floor": 0.18,
                "digital_twin_policy_prior_distill_coef": 0.095,
                "digital_twin_policy_prior_distill_warmup_updates": 3,
                "digital_twin_policy_prior_distill_decay": 0.82,
                "digital_twin_policy_prior_advantage_weight": 0.45,
                "digital_twin_policy_prior_max_weight": 2.20,
                "mechanism_credit_policy_coef": 0.40,
                "mechanism_credit_event_coef": 1.05,
                "mechanism_credit_option_coef": 0.52,
                "mechanism_aux_coef": 0.086,
                "mechanism_window_weight": 1.78,
                "prepare_action_prior_weight": 0.70,
                "mechanism_aux_coef_floor_after_update": 0.080,
                "mechanism_window_weight_floor_after_update": 1.58,
                "event_logit_sharpening_timing_gain": 0.38,
                "event_prepare_margin_boost": 0.42,
                "handoff_risk_event_coef": 0.60,
                "handoff_risk_option_coef": 0.52,
                "handoff_risk_ready_bonus": 1.08,
                "handoff_risk_prepare_bonus": 0.40,
                "target_kl": 0.0095,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v31_handoff_pacing_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v30_dt_prior_mappo")
        kwargs.update(
            {
                "digital_twin_policy_prior_pacing_enabled": True,
                "digital_twin_policy_prior_pacing_threshold": 0.30,
                "digital_twin_policy_prior_pacing_fast_scale": 1.35,
                "digital_twin_policy_prior_pacing_event_suppression": 0.72,
                "digital_twin_policy_prior_pacing_slow_suppression": 0.68,
                "digital_twin_policy_prior_pacing_short_dag_threshold": 10,
                "digital_twin_policy_prior_logit_bias": 3.60,
                "digital_twin_policy_prior_distill_coef": 0.105,
                "digital_twin_policy_prior_advantage_weight": 0.52,
                "digital_twin_policy_prior_max_weight": 2.40,
                "mechanism_credit_policy_coef": 0.42,
                "mechanism_credit_event_coef": 1.08,
                "handoff_risk_event_coef": 0.62,
                "handoff_risk_option_coef": 0.54,
                "target_kl": 0.009,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v32_dt_continuation_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v31_handoff_pacing_mappo")
        kwargs.update(
            {
                "digital_twin_policy_prior_env_action_bias_enabled": True,
                "digital_twin_policy_prior_env_action_logit_bias": 4.80,
                "digital_twin_policy_prior_continuation_threshold": 0.32,
                "digital_twin_policy_prior_continuation_prepare_scale": 1.35,
                "digital_twin_policy_prior_continuation_wait_scale": 0.95,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.42,
                "digital_twin_policy_prior_logit_bias": 4.15,
                "digital_twin_policy_prior_pacing_threshold": 0.26,
                "digital_twin_policy_prior_pacing_event_suppression": 0.48,
                "digital_twin_policy_prior_pacing_slow_suppression": 0.56,
                "digital_twin_policy_prior_distill_coef": 0.120,
                "digital_twin_policy_prior_advantage_weight": 0.58,
                "digital_twin_policy_prior_max_weight": 2.60,
                "mechanism_credit_policy_coef": 0.46,
                "mechanism_credit_event_coef": 1.18,
                "mechanism_credit_option_coef": 0.58,
                "handoff_risk_event_coef": 0.68,
                "handoff_risk_option_coef": 0.60,
                "target_kl": 0.0085,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v33_env_action_ppo_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v32_dt_continuation_mappo")
        kwargs.update(
            {
                "env_action_ppo_enabled": True,
                "env_action_ppo_coef": 0.72,
                "env_action_ppo_advantage_blend": 0.54,
                "env_action_ppo_teacher_coef": 0.34,
                "env_action_ppo_mechanism_focus": 0.65,
                "env_action_ppo_max_weight": 2.25,
                "counterfactual_teacher_prd_enabled": True,
                "counterfactual_teacher_event_coef": 0.34,
                "counterfactual_teacher_option_coef": 0.42,
                "counterfactual_teacher_clip": 1.60,
                "counterfactual_teacher_mechanism_bonus": 0.64,
                "counterfactual_teacher_missed_prepare_penalty": 0.50,
                "counterfactual_teacher_local_bonus": 0.32,
                "counterfactual_teacher_current_rsu_penalty": 0.10,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.86,
                "digital_twin_policy_prior_env_action_logit_bias": 3.20,
                "digital_twin_policy_prior_continuation_threshold": 0.38,
                "digital_twin_policy_prior_continuation_prepare_scale": 1.05,
                "digital_twin_policy_prior_continuation_wait_scale": 1.10,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.30,
                "digital_twin_policy_prior_logit_bias": 3.55,
                "digital_twin_policy_prior_distill_coef": 0.090,
                "digital_twin_policy_prior_advantage_weight": 0.46,
                "digital_twin_policy_prior_max_weight": 2.20,
                "mechanism_credit_policy_coef": 0.48,
                "mechanism_credit_event_coef": 1.10,
                "mechanism_credit_option_coef": 0.54,
                "handoff_risk_event_coef": 0.66,
                "handoff_risk_option_coef": 0.58,
                "target_kl": 0.0085,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v34_adaptive_wait_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v33_env_action_ppo_mappo")
        kwargs.update(
            {
                "digital_twin_policy_prior_adaptive_wait_enabled": True,
                "digital_twin_policy_prior_wait_ready_threshold": 0.50,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.58,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.42,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.92,
                "digital_twin_policy_prior_env_action_logit_bias": 3.70,
                "digital_twin_policy_prior_continuation_threshold": 0.34,
                "digital_twin_policy_prior_continuation_prepare_scale": 0.88,
                "digital_twin_policy_prior_continuation_wait_scale": 1.48,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.36,
                "digital_twin_policy_prior_logit_bias": 3.70,
                "digital_twin_policy_prior_distill_coef": 0.082,
                "digital_twin_policy_prior_advantage_weight": 0.42,
                "digital_twin_policy_prior_max_weight": 2.10,
                "env_action_ppo_coef": 0.82,
                "env_action_ppo_advantage_blend": 0.50,
                "env_action_ppo_teacher_coef": 0.48,
                "env_action_ppo_mechanism_focus": 0.82,
                "env_action_ppo_max_weight": 2.30,
                "env_action_ppo_ratio_barrier_coef": 0.045,
                "env_action_ppo_ratio_barrier_margin": 0.32,
                "counterfactual_teacher_mechanism_bonus": 0.58,
                "counterfactual_teacher_missed_prepare_penalty": 0.34,
                "counterfactual_teacher_local_bonus": 0.54,
                "counterfactual_teacher_current_rsu_penalty": 0.08,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.72,
                "mechanism_credit_policy_coef": 0.46,
                "mechanism_credit_event_coef": 1.02,
                "mechanism_credit_option_coef": 0.52,
                "handoff_risk_event_coef": 0.62,
                "handoff_risk_option_coef": 0.56,
                "target_kl": 0.008,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v35_guard_relaxed_action_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v34_adaptive_wait_mappo")
        kwargs.update(
            {
                "continuity_guard_enabled": False,
                "handoff_target_alignment_guard_enabled": False,
                "backhaul_guard_enabled": False,
                "cache_warm_start_guard_enabled": False,
                "predictive_prefetch_admission_guard_enabled": False,
                "heuristic_imitation_coef": 0.018,
                "heuristic_imitation_warmup_updates": 2,
                "heuristic_imitation_decay": 0.55,
                "mechanism_aux_coef": 0.040,
                "mechanism_window_weight": 1.18,
                "prepare_action_prior_weight": 0.30,
                "mechanism_entropy_coef": 0.0012,
                "mechanism_aux_coef_floor_after_update": 0.028,
                "mechanism_window_weight_floor_after_update": 1.08,
                "mechanism_entropy_floor_after_update": 0.0010,
                "digital_twin_policy_prior_wait_ready_threshold": 0.30,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.92,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.72,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.74,
                "digital_twin_policy_prior_env_action_logit_bias": 4.10,
                "digital_twin_policy_prior_continuation_threshold": 0.26,
                "digital_twin_policy_prior_continuation_prepare_scale": 0.72,
                "digital_twin_policy_prior_continuation_wait_scale": 1.80,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.18,
                "digital_twin_policy_prior_logit_bias": 3.90,
                "digital_twin_policy_prior_distill_coef": 0.074,
                "digital_twin_policy_prior_advantage_weight": 0.48,
                "digital_twin_policy_prior_max_weight": 2.30,
                "env_action_ppo_coef": 0.98,
                "env_action_ppo_advantage_blend": 0.46,
                "env_action_ppo_teacher_coef": 0.66,
                "env_action_ppo_mechanism_focus": 1.05,
                "env_action_ppo_max_weight": 2.65,
                "env_action_ppo_ratio_barrier_coef": 0.060,
                "env_action_ppo_ratio_barrier_margin": 0.30,
                "counterfactual_teacher_mechanism_bonus": 0.74,
                "counterfactual_teacher_missed_prepare_penalty": 0.24,
                "counterfactual_teacher_local_bonus": 0.68,
                "counterfactual_teacher_current_rsu_penalty": 0.04,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.58,
                "mechanism_credit_policy_coef": 0.54,
                "mechanism_credit_event_coef": 1.16,
                "mechanism_credit_option_coef": 0.62,
                "handoff_risk_event_coef": 0.72,
                "handoff_risk_option_coef": 0.64,
                "idle_execution_policy_coef": 0.54,
                "idle_execution_option_coef": 0.72,
                "idle_execution_local_bonus": 0.58,
                "idle_execution_mechanism_penalty": 0.22,
                "target_kl": 0.0075,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v36_counterfactual_margin_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v35_guard_relaxed_action_mappo")
        kwargs.update(
            {
                "env_action_counterfactual_margin_enabled": True,
                "env_action_counterfactual_margin_coef": 0.22,
                "env_action_counterfactual_margin_min_gap": 0.04,
                "env_action_counterfactual_margin_max_weight": 2.40,
                "env_action_ppo_coef": 0.88,
                "env_action_ppo_teacher_coef": 0.56,
                "env_action_ppo_mechanism_focus": 0.92,
                "env_action_ppo_max_weight": 2.45,
                "env_action_ppo_ratio_barrier_coef": 0.070,
                "env_action_ppo_ratio_barrier_margin": 0.28,
                "digital_twin_policy_prior_wait_ready_threshold": 0.18,
                "digital_twin_policy_prior_wait_timing_ceiling": 1.00,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.90,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.68,
                "digital_twin_policy_prior_continuation_threshold": 0.20,
                "digital_twin_policy_prior_continuation_wait_scale": 2.05,
                "digital_twin_policy_prior_continuation_prepare_scale": 0.64,
                "counterfactual_teacher_local_bonus": 0.92,
                "counterfactual_teacher_mechanism_bonus": 0.78,
                "counterfactual_teacher_current_rsu_penalty": 0.03,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.66,
                "idle_execution_policy_coef": 0.62,
                "idle_execution_option_coef": 0.80,
                "idle_execution_local_bonus": 0.72,
                "mechanism_aux_coef": 0.034,
                "mechanism_window_weight": 1.12,
                "prepare_action_prior_weight": 0.22,
                "mechanism_entropy_coef": 0.0014,
                "target_kl": 0.007,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v37_advantage_gated_counterfactual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v35_guard_relaxed_action_mappo")
        kwargs.update(
            {
                "env_action_counterfactual_margin_enabled": True,
                "env_action_counterfactual_margin_coef": 0.055,
                "env_action_counterfactual_margin_min_gap": 0.10,
                "env_action_counterfactual_margin_max_weight": 1.35,
                "env_action_counterfactual_margin_advantage_gate": 0.12,
                "env_action_counterfactual_margin_advantage_blend": 0.70,
                "env_action_ppo_coef": 1.00,
                "env_action_ppo_teacher_coef": 0.60,
                "env_action_ppo_mechanism_focus": 0.95,
                "env_action_ppo_max_weight": 2.55,
                "env_action_ppo_ratio_barrier_coef": 0.075,
                "env_action_ppo_ratio_barrier_margin": 0.30,
                "digital_twin_policy_prior_wait_ready_threshold": 0.30,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.92,
                "digital_twin_policy_prior_continuation_wait_scale": 1.78,
                "digital_twin_policy_prior_continuation_prepare_scale": 0.72,
                "counterfactual_teacher_local_bonus": 0.70,
                "counterfactual_teacher_mechanism_bonus": 0.72,
                "counterfactual_teacher_missed_prepare_penalty": 0.22,
                "counterfactual_teacher_current_rsu_penalty": 0.04,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.56,
                "mechanism_aux_coef": 0.038,
                "mechanism_window_weight": 1.16,
                "prepare_action_prior_weight": 0.28,
                "mechanism_entropy_coef": 0.0011,
                "target_kl": 0.0075,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v38_undiscounted_dt_prior_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v35_guard_relaxed_action_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.010,
                "mechanism_aux_coef": 0.036,
                "mechanism_window_weight": 1.14,
                "prepare_action_prior_weight": 0.24,
                "mechanism_entropy_coef": 0.0016,
                "digital_twin_policy_prior_logit_bias": 4.60,
                "digital_twin_policy_prior_env_action_logit_bias": 5.20,
                "digital_twin_policy_prior_distill_coef": 0.145,
                "digital_twin_policy_prior_distill_warmup_updates": 1,
                "digital_twin_policy_prior_distill_decay": 0.94,
                "digital_twin_policy_prior_advantage_weight": 0.78,
                "digital_twin_policy_prior_max_weight": 3.20,
                "digital_twin_policy_prior_continuation_threshold": 0.18,
                "digital_twin_policy_prior_continuation_prepare_scale": 1.10,
                "digital_twin_policy_prior_continuation_wait_scale": 1.35,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.12,
                "digital_twin_policy_prior_wait_ready_threshold": 0.24,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.96,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.84,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.82,
                "env_action_ppo_coef": 1.08,
                "env_action_ppo_advantage_blend": 0.58,
                "env_action_ppo_teacher_coef": 0.74,
                "env_action_ppo_mechanism_focus": 1.18,
                "env_action_ppo_max_weight": 3.10,
                "env_action_ppo_ratio_barrier_coef": 0.055,
                "env_action_ppo_ratio_barrier_margin": 0.34,
                "env_action_counterfactual_margin_enabled": False,
                "counterfactual_teacher_mechanism_bonus": 0.82,
                "counterfactual_teacher_missed_prepare_penalty": 0.20,
                "counterfactual_teacher_local_bonus": 0.70,
                "counterfactual_teacher_current_rsu_penalty": 0.035,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.52,
                "mechanism_credit_policy_coef": 0.60,
                "mechanism_credit_event_coef": 1.24,
                "mechanism_credit_option_coef": 0.66,
                "target_kl": 0.008,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v39_delayed_credit_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v35_guard_relaxed_action_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.012,
                "heuristic_imitation_warmup_updates": 1,
                "heuristic_imitation_decay": 0.50,
                "mechanism_aux_coef": 0.036,
                "mechanism_window_weight": 1.16,
                "prepare_action_prior_weight": 0.26,
                "mechanism_entropy_coef": 0.0014,
                "digital_twin_policy_prior_logit_bias": 3.70,
                "digital_twin_policy_prior_env_action_logit_bias": 3.95,
                "digital_twin_policy_prior_distill_coef": 0.052,
                "digital_twin_policy_prior_advantage_weight": 0.40,
                "digital_twin_policy_prior_max_weight": 2.20,
                "digital_twin_policy_prior_wait_ready_threshold": 0.30,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.92,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.72,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.74,
                "digital_twin_policy_prior_continuation_threshold": 0.26,
                "digital_twin_policy_prior_continuation_prepare_scale": 0.72,
                "digital_twin_policy_prior_continuation_wait_scale": 1.80,
                "env_action_ppo_coef": 1.06,
                "env_action_ppo_advantage_blend": 0.44,
                "env_action_ppo_teacher_coef": 0.46,
                "env_action_ppo_mechanism_focus": 1.24,
                "env_action_ppo_max_weight": 2.80,
                "env_action_ppo_ratio_barrier_coef": 0.065,
                "env_action_ppo_ratio_barrier_margin": 0.30,
                "env_action_counterfactual_margin_enabled": False,
                "delayed_mechanism_credit_enabled": True,
                "delayed_mechanism_credit_policy_coef": 0.46,
                "delayed_mechanism_credit_event_coef": 1.05,
                "delayed_mechanism_credit_horizon": 5,
                "delayed_mechanism_credit_decay": 0.72,
                "delayed_mechanism_credit_clip": 1.80,
                "delayed_mechanism_credit_ready_bonus": 1.35,
                "delayed_mechanism_credit_success_bonus": 0.88,
                "delayed_mechanism_credit_failure_penalty": 1.02,
                "delayed_mechanism_credit_missed_prepare_scale": 0.62,
                "delayed_mechanism_credit_stale_penalty": 0.42,
                "delayed_mechanism_credit_context_gate": 0.24,
                "counterfactual_teacher_mechanism_bonus": 0.66,
                "counterfactual_teacher_missed_prepare_penalty": 0.22,
                "counterfactual_teacher_local_bonus": 0.66,
                "counterfactual_teacher_current_rsu_penalty": 0.04,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.54,
                "mechanism_credit_policy_coef": 0.50,
                "mechanism_credit_event_coef": 1.10,
                "mechanism_credit_option_coef": 0.60,
                "handoff_risk_event_coef": 0.74,
                "handoff_risk_option_coef": 0.64,
                "target_kl": 0.0075,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v40_advantage_weighted_behavior_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v39_delayed_credit_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.008,
                "heuristic_imitation_warmup_updates": 1,
                "heuristic_imitation_decay": 0.42,
                "env_action_ppo_coef": 1.00,
                "env_action_ppo_advantage_blend": 0.48,
                "env_action_ppo_teacher_coef": 0.34,
                "env_action_ppo_mechanism_focus": 1.10,
                "env_action_ppo_max_weight": 2.55,
                "env_action_ppo_ratio_barrier_coef": 0.075,
                "env_action_ppo_ratio_barrier_margin": 0.30,
                "advantage_weighted_behavior_regularization_enabled": True,
                "advantage_weighted_behavior_coef": 0.24,
                "advantage_weighted_behavior_positive_coef": 1.08,
                "advantage_weighted_behavior_negative_coef": 0.92,
                "advantage_weighted_behavior_temperature": 0.62,
                "advantage_weighted_behavior_max_weight": 2.20,
                "advantage_weighted_behavior_positive_gate": 0.08,
                "advantage_weighted_behavior_negative_gate": 0.04,
                "advantage_weighted_behavior_mechanism_scale": 1.35,
                "delayed_mechanism_credit_policy_coef": 0.38,
                "delayed_mechanism_credit_event_coef": 0.92,
                "delayed_mechanism_credit_clip": 1.65,
                "counterfactual_teacher_mechanism_bonus": 0.60,
                "counterfactual_teacher_missed_prepare_penalty": 0.26,
                "counterfactual_teacher_local_bonus": 0.62,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.58,
                "mechanism_aux_coef": 0.034,
                "mechanism_window_weight": 1.12,
                "prepare_action_prior_weight": 0.24,
                "mechanism_entropy_coef": 0.0015,
                "target_kl": 0.0075,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v41_conservative_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v39_delayed_credit_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.010,
                "heuristic_imitation_warmup_updates": 1,
                "heuristic_imitation_decay": 0.48,
                "env_action_ppo_coef": 1.04,
                "env_action_ppo_advantage_blend": 0.44,
                "env_action_ppo_teacher_coef": 0.42,
                "env_action_ppo_mechanism_focus": 1.18,
                "env_action_ppo_max_weight": 2.70,
                "env_action_ppo_ratio_barrier_coef": 0.070,
                "env_action_ppo_ratio_barrier_margin": 0.28,
                "advantage_weighted_behavior_regularization_enabled": True,
                "advantage_weighted_behavior_coef": 0.115,
                "advantage_weighted_behavior_positive_coef": 0.0,
                "advantage_weighted_behavior_negative_coef": 1.18,
                "advantage_weighted_behavior_temperature": 0.70,
                "advantage_weighted_behavior_max_weight": 1.80,
                "advantage_weighted_behavior_positive_gate": 0.20,
                "advantage_weighted_behavior_negative_gate": 0.025,
                "advantage_weighted_behavior_mechanism_scale": 0.95,
                "delayed_mechanism_credit_policy_coef": 0.44,
                "delayed_mechanism_credit_event_coef": 1.02,
                "delayed_mechanism_credit_clip": 1.75,
                "counterfactual_teacher_mechanism_bonus": 0.64,
                "counterfactual_teacher_missed_prepare_penalty": 0.24,
                "counterfactual_teacher_local_bonus": 0.64,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.56,
                "mechanism_aux_coef": 0.035,
                "mechanism_window_weight": 1.14,
                "prepare_action_prior_weight": 0.25,
                "mechanism_entropy_coef": 0.0014,
                "target_kl": 0.0075,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v42_completion_aligned_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v41_conservative_recovery_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.008,
                "heuristic_imitation_decay": 0.52,
                "env_action_ppo_coef": 1.02,
                "env_action_ppo_advantage_blend": 0.48,
                "env_action_ppo_teacher_coef": 0.38,
                "env_action_ppo_mechanism_focus": 1.22,
                "env_action_ppo_max_weight": 2.65,
                "advantage_weighted_behavior_coef": 0.10,
                "advantage_weighted_behavior_negative_coef": 1.12,
                "advantage_weighted_behavior_temperature": 0.78,
                "advantage_weighted_behavior_max_weight": 1.70,
                "delayed_mechanism_credit_policy_coef": 0.46,
                "delayed_mechanism_credit_event_coef": 1.08,
                "delayed_mechanism_credit_clip": 1.72,
                "delayed_mechanism_credit_ready_bonus": 1.42,
                "delayed_mechanism_credit_success_bonus": 0.92,
                "delayed_mechanism_credit_failure_penalty": 1.12,
                "counterfactual_teacher_mechanism_bonus": 0.66,
                "counterfactual_teacher_missed_prepare_penalty": 0.26,
                "counterfactual_teacher_local_bonus": 0.66,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.58,
                "mechanism_aux_coef": 0.034,
                "mechanism_window_weight": 1.16,
                "prepare_action_prior_weight": 0.24,
                "mechanism_entropy_coef": 0.0014,
                "target_kl": 0.0070,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v43_strict_opportunity_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v41_conservative_recovery_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.026,
                "heuristic_imitation_warmup_updates": 10,
                "heuristic_imitation_decay": 0.86,
                "env_action_ppo_coef": 0.82,
                "env_action_ppo_advantage_blend": 0.56,
                "env_action_ppo_teacher_coef": 0.58,
                "env_action_ppo_mechanism_focus": 0.72,
                "env_action_ppo_max_weight": 2.25,
                "env_action_ppo_ratio_barrier_coef": 0.11,
                "env_action_ppo_ratio_barrier_margin": 0.24,
                "advantage_weighted_behavior_regularization_enabled": True,
                "advantage_weighted_behavior_coef": 0.18,
                "advantage_weighted_behavior_positive_coef": 0.0,
                "advantage_weighted_behavior_negative_coef": 1.45,
                "advantage_weighted_behavior_temperature": 0.58,
                "advantage_weighted_behavior_max_weight": 2.05,
                "advantage_weighted_behavior_negative_gate": 0.015,
                "advantage_weighted_behavior_mechanism_scale": 1.35,
                "delayed_mechanism_credit_policy_coef": 0.18,
                "delayed_mechanism_credit_event_coef": 0.42,
                "delayed_mechanism_credit_horizon": 4,
                "delayed_mechanism_credit_decay": 0.64,
                "delayed_mechanism_credit_clip": 1.40,
                "delayed_mechanism_credit_ready_bonus": 0.95,
                "delayed_mechanism_credit_success_bonus": 0.55,
                "delayed_mechanism_credit_failure_penalty": 1.75,
                "delayed_mechanism_credit_missed_prepare_scale": 0.16,
                "delayed_mechanism_credit_stale_penalty": 0.92,
                "delayed_mechanism_credit_context_gate": 0.46,
                "delayed_mechanism_credit_strict_opportunity_enabled": True,
                "counterfactual_teacher_mechanism_bonus": 0.28,
                "counterfactual_teacher_missed_prepare_penalty": 0.42,
                "counterfactual_teacher_local_bonus": 0.78,
                "counterfactual_teacher_current_rsu_penalty": 0.02,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.86,
                "mechanism_aux_coef": 0.014,
                "mechanism_window_weight": 0.82,
                "prepare_action_prior_weight": 0.075,
                "mechanism_entropy_coef": 0.0008,
                "target_kl": 0.0065,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v44_opportunity_constrained_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v43_strict_opportunity_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.018,
                "heuristic_imitation_warmup_updates": 6,
                "heuristic_imitation_decay": 0.74,
                "env_action_ppo_coef": 0.94,
                "env_action_ppo_advantage_blend": 0.62,
                "env_action_ppo_teacher_coef": 0.42,
                "env_action_ppo_mechanism_focus": 0.54,
                "env_action_ppo_ratio_barrier_coef": 0.09,
                "advantage_weighted_behavior_coef": 0.16,
                "advantage_weighted_behavior_negative_coef": 1.28,
                "advantage_weighted_behavior_temperature": 0.62,
                "advantage_weighted_behavior_mechanism_scale": 1.10,
                "delayed_mechanism_credit_policy_coef": 0.12,
                "delayed_mechanism_credit_event_coef": 0.26,
                "delayed_mechanism_credit_failure_penalty": 1.95,
                "delayed_mechanism_credit_stale_penalty": 1.12,
                "delayed_mechanism_credit_context_gate": 0.50,
                "counterfactual_teacher_mechanism_bonus": 0.18,
                "counterfactual_teacher_missed_prepare_penalty": 0.34,
                "counterfactual_teacher_invalid_mechanism_penalty": 1.02,
                "counterfactual_teacher_local_bonus": 0.86,
                "mechanism_aux_coef": 0.008,
                "mechanism_window_weight": 0.62,
                "prepare_action_prior_weight": 0.035,
                "mechanism_entropy_coef": 0.0006,
                "opportunity_constrained_policy_enabled": True,
                "opportunity_constrained_min_context": 0.54,
                "opportunity_constrained_low_context": 0.32,
                "opportunity_constrained_prepare_penalty": 7.50,
                "opportunity_constrained_prefetch_penalty": 4.25,
                "opportunity_constrained_prepare_bias": 1.05,
                "opportunity_constrained_prefetch_bias": 0.65,
                "opportunity_constrained_current_bias": 1.85,
                "opportunity_constrained_local_bias": 0.80,
                "opportunity_constrained_confidence_floor": 0.24,
                "opportunity_constrained_uncertainty_ceiling": 0.72,
                "opportunity_constrained_reliability_floor": 0.28,
                "target_kl": 0.0060,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v45_balanced_refresh_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v44_opportunity_constrained_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.016,
                "env_action_ppo_coef": 1.02,
                "env_action_ppo_teacher_coef": 0.48,
                "env_action_ppo_mechanism_focus": 0.62,
                "advantage_weighted_behavior_coef": 0.18,
                "advantage_weighted_behavior_negative_coef": 1.18,
                "delayed_mechanism_credit_policy_coef": 0.14,
                "delayed_mechanism_credit_event_coef": 0.30,
                "delayed_mechanism_credit_failure_penalty": 1.82,
                "delayed_mechanism_credit_stale_penalty": 0.96,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.90,
                "counterfactual_teacher_local_bonus": 0.76,
                "mechanism_aux_coef": 0.006,
                "mechanism_window_weight": 0.58,
                "prepare_action_prior_weight": 0.028,
                "opportunity_constrained_min_context": 0.50,
                "opportunity_constrained_low_context": 0.28,
                "opportunity_constrained_prepare_penalty": 5.75,
                "opportunity_constrained_prefetch_penalty": 3.10,
                "opportunity_constrained_prepare_bias": 1.28,
                "opportunity_constrained_prefetch_bias": 0.90,
                "opportunity_constrained_current_bias": 2.75,
                "opportunity_constrained_local_bias": 0.10,
                "opportunity_constrained_confidence_floor": 0.20,
                "opportunity_constrained_uncertainty_ceiling": 0.76,
                "opportunity_constrained_reliability_floor": 0.22,
                "target_kl": 0.0060,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v46_net_utility_constrained_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v44_opportunity_constrained_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.015,
                "env_action_ppo_coef": 0.98,
                "env_action_ppo_advantage_blend": 0.64,
                "env_action_ppo_teacher_coef": 0.40,
                "env_action_ppo_mechanism_focus": 0.50,
                "advantage_weighted_behavior_coef": 0.15,
                "advantage_weighted_behavior_negative_coef": 1.34,
                "advantage_weighted_behavior_temperature": 0.60,
                "advantage_weighted_behavior_mechanism_scale": 1.18,
                "delayed_mechanism_credit_policy_coef": 0.10,
                "delayed_mechanism_credit_event_coef": 0.22,
                "delayed_mechanism_credit_failure_penalty": 2.05,
                "delayed_mechanism_credit_stale_penalty": 1.18,
                "counterfactual_teacher_invalid_mechanism_penalty": 1.08,
                "counterfactual_teacher_local_bonus": 0.88,
                "mechanism_aux_coef": 0.007,
                "mechanism_window_weight": 0.60,
                "prepare_action_prior_weight": 0.030,
                "opportunity_constrained_min_context": 0.55,
                "opportunity_constrained_low_context": 0.34,
                "opportunity_constrained_prepare_penalty": 7.25,
                "opportunity_constrained_prefetch_penalty": 4.45,
                "opportunity_constrained_prepare_bias": 1.02,
                "opportunity_constrained_prefetch_bias": 0.62,
                "opportunity_constrained_current_bias": 2.15,
                "opportunity_constrained_local_bias": 0.78,
                "opportunity_constrained_confidence_floor": 0.24,
                "opportunity_constrained_uncertainty_ceiling": 0.72,
                "opportunity_constrained_reliability_floor": 0.30,
                "net_utility_prd_enabled": True,
                "net_utility_backhaul_coef": 0.24,
                "net_utility_migration_coef": 0.22,
                "net_utility_expired_prefetch_coef": 0.58,
                "net_utility_idle_prefetch_penalty": 0.76,
                "net_utility_failed_mechanism_penalty": 0.82,
                "net_utility_failed_mechanism_backhaul_coef": 0.36,
                "net_utility_mechanism_window_failed_penalty_scale": 0.58,
                "net_utility_success_bonus": 0.30,
                "net_utility_backhaul_normalizer": 64.0,
                "net_utility_cost_dual_enabled": True,
                "net_utility_cost_dual_lr": 0.07,
                "net_utility_cost_target": 0.08,
                "net_utility_cost_dual_max": 1.65,
                "net_utility_cost_dual_initial": 0.0,
                "net_utility_option_termination_enabled": False,
                "target_kl": 0.0055,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v47_service_backhaul_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v46_net_utility_constrained_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.012,
                "env_action_ppo_coef": 1.04,
                "env_action_ppo_advantage_blend": 0.66,
                "env_action_ppo_teacher_coef": 0.34,
                "advantage_weighted_behavior_coef": 0.13,
                "advantage_weighted_behavior_negative_coef": 1.22,
                "opportunity_constrained_current_bias": 2.35,
                "opportunity_constrained_local_bias": 0.56,
                "net_utility_backhaul_coef": 0.20,
                "net_utility_migration_coef": 0.24,
                "net_utility_failed_mechanism_penalty": 0.90,
                "net_utility_failed_mechanism_backhaul_coef": 0.42,
                "net_utility_cost_target": 0.10,
                "net_utility_cost_dual_max": 1.50,
                "backhaul_guard_enabled": True,
                "backhaul_guard_max_reactive_fills_per_adapter": 1,
                "backhaul_aware_policy_enabled": True,
                "backhaul_aware_service_fill_bias": 2.40,
                "backhaul_aware_redundant_fill_penalty": 2.20,
                "backhaul_aware_no_signal_prefetch_penalty": 1.20,
                "backhaul_aware_no_signal_prepare_penalty": 1.65,
                "backhaul_aware_steady_bias": 1.15,
                "backhaul_aware_service_pressure_floor": 0.34,
                "target_kl": 0.0052,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v48_service_fill_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v46_net_utility_constrained_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.014,
                "env_action_ppo_coef": 1.02,
                "env_action_ppo_advantage_blend": 0.66,
                "env_action_ppo_teacher_coef": 0.36,
                "advantage_weighted_behavior_coef": 0.14,
                "advantage_weighted_behavior_negative_coef": 1.28,
                "opportunity_constrained_current_bias": 2.30,
                "opportunity_constrained_local_bias": 0.72,
                "net_utility_backhaul_coef": 0.22,
                "net_utility_migration_coef": 0.23,
                "net_utility_failed_mechanism_penalty": 0.86,
                "net_utility_failed_mechanism_backhaul_coef": 0.38,
                "net_utility_cost_target": 0.09,
                "net_utility_cost_dual_max": 1.55,
                "backhaul_guard_enabled": False,
                "backhaul_aware_policy_enabled": True,
                "backhaul_aware_service_fill_bias": 1.35,
                "backhaul_aware_redundant_fill_penalty": 0.85,
                "backhaul_aware_no_signal_prefetch_penalty": 0.0,
                "backhaul_aware_no_signal_prepare_penalty": 0.0,
                "backhaul_aware_steady_bias": 0.45,
                "backhaul_aware_service_pressure_floor": 0.42,
                "target_kl": 0.0055,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v49_retrospective_handoff_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v46_net_utility_constrained_mappo")
        kwargs.update(
            {
                "heuristic_imitation_coef": 0.014,
                "env_action_ppo_coef": 1.02,
                "env_action_ppo_advantage_blend": 0.67,
                "env_action_ppo_teacher_coef": 0.36,
                "advantage_weighted_behavior_coef": 0.14,
                "advantage_weighted_behavior_negative_coef": 1.28,
                "mechanism_aux_coef": 0.018,
                "mechanism_aux_coef_floor_after_update": 0.018,
                "mechanism_window_weight": 1.20,
                "mechanism_window_weight_floor_after_update": 1.20,
                "prepare_action_prior_weight": 0.060,
                "opportunity_constrained_prepare_penalty": 6.40,
                "opportunity_constrained_prefetch_penalty": 4.10,
                "opportunity_constrained_prepare_bias": 1.18,
                "opportunity_constrained_prefetch_bias": 0.68,
                "opportunity_constrained_current_bias": 2.25,
                "opportunity_constrained_local_bias": 0.70,
                "net_utility_backhaul_coef": 0.22,
                "net_utility_migration_coef": 0.23,
                "net_utility_failed_mechanism_penalty": 0.86,
                "net_utility_failed_mechanism_backhaul_coef": 0.38,
                "net_utility_cost_target": 0.09,
                "net_utility_cost_dual_max": 1.55,
                "retrospective_handoff_aux_enabled": True,
                "retrospective_handoff_aux_max_eta": 6.0,
                "retrospective_handoff_aux_min_score": 0.08,
                "retrospective_handoff_aux_prepare_weight": 0.72,
                "retrospective_handoff_aux_transition_weight": 1.60,
                "backhaul_guard_enabled": False,
                "backhaul_aware_policy_enabled": True,
                "backhaul_aware_service_fill_bias": 1.20,
                "backhaul_aware_redundant_fill_penalty": 0.75,
                "backhaul_aware_no_signal_prefetch_penalty": 0.0,
                "backhaul_aware_no_signal_prepare_penalty": 0.0,
                "backhaul_aware_steady_bias": 0.40,
                "backhaul_aware_service_pressure_floor": 0.42,
                "target_kl": 0.0055,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v50_long_horizon_handoff_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v49_retrospective_handoff_mappo")
        kwargs.update(
            {
                "mechanism_aux_coef": 0.020,
                "mechanism_aux_coef_floor_after_update": 0.020,
                "mechanism_window_weight": 1.28,
                "mechanism_window_weight_floor_after_update": 1.24,
                "prepare_action_prior_weight": 0.072,
                "opportunity_constrained_prepare_bias": 1.28,
                "opportunity_constrained_prefetch_bias": 0.74,
                "opportunity_constrained_prepare_penalty": 5.80,
                "opportunity_constrained_prefetch_penalty": 3.80,
                "retrospective_handoff_aux_max_eta": 16.0,
                "retrospective_handoff_aux_min_score": 0.04,
                "retrospective_handoff_aux_prepare_weight": 0.82,
                "retrospective_handoff_aux_transition_weight": 1.85,
                "backhaul_aware_service_fill_bias": 1.15,
                "backhaul_aware_redundant_fill_penalty": 0.70,
                "target_kl": 0.0055,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v51_physical_transfer_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v50_long_horizon_handoff_mappo")
        kwargs.update(
            {
                "mechanism_aux_coef": 0.024,
                "mechanism_aux_coef_floor_after_update": 0.022,
                "mechanism_window_weight": 1.36,
                "mechanism_window_weight_floor_after_update": 1.30,
                "prepare_action_prior_weight": 0.084,
                "opportunity_constrained_prepare_bias": 1.38,
                "opportunity_constrained_prefetch_bias": 0.80,
                "opportunity_constrained_prepare_penalty": 5.20,
                "opportunity_constrained_prefetch_penalty": 3.40,
                "retrospective_handoff_aux_min_score": 0.035,
                "retrospective_handoff_aux_prepare_weight": 0.90,
                "retrospective_handoff_aux_transition_weight": 2.05,
                "backhaul_aware_service_fill_bias": 1.10,
                "backhaul_aware_redundant_fill_penalty": 0.66,
                "target_kl": 0.0052,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v52_net_advantage_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v51_physical_transfer_mappo")
        kwargs.update(
            {
                "mechanism_aux_coef": 0.014,
                "mechanism_aux_coef_floor_after_update": 0.010,
                "mechanism_window_weight": 1.18,
                "mechanism_window_weight_floor_after_update": 1.12,
                "prepare_action_prior_weight": 0.052,
                "opportunity_constrained_prepare_bias": 0.92,
                "opportunity_constrained_prefetch_bias": 0.52,
                "opportunity_constrained_prepare_penalty": 6.10,
                "opportunity_constrained_prefetch_penalty": 4.15,
                "retrospective_handoff_aux_prepare_weight": 0.58,
                "retrospective_handoff_aux_transition_weight": 1.35,
                "backhaul_aware_service_fill_bias": 1.22,
                "backhaul_aware_redundant_fill_penalty": 0.78,
                "net_utility_backhaul_coef": 0.28,
                "net_utility_migration_coef": 0.28,
                "net_utility_failed_mechanism_penalty": 1.04,
                "net_utility_failed_mechanism_backhaul_coef": 0.48,
                "net_utility_cost_target": 0.075,
                "net_utility_cost_dual_max": 1.85,
                "net_advantage_prepare_gate_enabled": True,
                "net_advantage_prepare_gate_bias": 1.85,
                "net_advantage_prepare_gate_min_score": 0.50,
                "net_advantage_prepare_gate_margin": 0.16,
                "net_advantage_prepare_gate_prefetch_scale": 0.42,
                "net_advantage_prepare_gate_current_scale": 0.70,
                "net_advantage_prepare_gate_cost_scale": 0.58,
                "net_advantage_prepare_gate_policy_coef": 0.18,
                "net_advantage_prepare_gate_event_coef": 0.42,
                "net_advantage_prepare_gate_clip": 1.35,
                "target_kl": 0.0048,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v53_service_net_advantage_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v52_net_advantage_mappo")
        kwargs.update(
            {
                "backhaul_aware_service_fill_bias": 1.55,
                "backhaul_aware_redundant_fill_penalty": 0.70,
                "idle_execution_local_bonus": 0.42,
                "idle_execution_current_rsu_delay_coef": 0.50,
                "opportunity_current_rsu_efficiency_coef": 0.42,
                "opportunity_local_fallback_coef": 0.18,
                "opportunity_backhaul_penalty_coef": 0.34,
                "net_advantage_prepare_gate_bias": 2.10,
                "net_advantage_prepare_gate_min_score": 0.52,
                "net_advantage_prepare_gate_margin": 0.14,
                "net_advantage_prepare_gate_prefetch_scale": 0.48,
                "net_advantage_prepare_gate_current_scale": 0.86,
                "net_advantage_prepare_gate_service_fill_scale": 0.92,
                "net_advantage_prepare_gate_local_penalty_scale": 0.72,
                "net_advantage_prepare_gate_cost_scale": 0.62,
                "net_advantage_prepare_gate_policy_coef": 0.24,
                "net_advantage_prepare_gate_event_coef": 0.44,
                "net_advantage_prepare_gate_clip": 1.45,
                "target_kl": 0.0046,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v54_service_completion_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v53_service_net_advantage_mappo")
        kwargs.update(
            {
                "service_completion_gate_enabled": True,
                "service_completion_gate_bias": 3.20,
                "service_completion_gate_remaining_nodes_threshold": 2,
                "service_completion_gate_event_suppression_scale": 0.82,
                "service_completion_gate_prefetch_suppression_scale": 0.52,
                "service_completion_gate_fallback_suppression_scale": 1.10,
                "service_completion_gate_policy_coef": 0.36,
                "service_completion_gate_event_coef": 0.24,
                "service_completion_gate_clip": 1.60,
                "net_advantage_prepare_gate_bias": 2.35,
                "net_advantage_prepare_gate_current_scale": 0.94,
                "net_advantage_prepare_gate_service_fill_scale": 1.02,
                "net_advantage_prepare_gate_local_penalty_scale": 0.88,
                "net_advantage_prepare_gate_policy_coef": 0.30,
                "net_advantage_prepare_gate_event_coef": 0.48,
                "target_kl": 0.0046,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v55_coverage_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v54_service_completion_mappo")
        kwargs.update(
            {
                "coverage_recovery_gate_bias_scale": 1.65,
                "coverage_recovery_gate_min_scale": 0.62,
                "coverage_recovery_gate_fallback_suppression_scale": 2.35,
                "coverage_recovery_gate_fast_suppression_scale": 1.50,
                "coverage_recovery_gate_current_suppression_scale": 0.36,
                "coverage_recovery_gate_prepare_credit": 1.15,
                "coverage_recovery_gate_fallback_penalty": 1.55,
                "coverage_recovery_guard_enabled": True,
                "net_advantage_prepare_gate_bias": 2.65,
                "net_advantage_prepare_gate_policy_coef": 0.42,
                "net_advantage_prepare_gate_event_coef": 0.60,
                "net_advantage_prepare_gate_clip": 2.10,
                "service_completion_gate_bias": 3.35,
                "service_completion_gate_policy_coef": 0.38,
                "service_completion_gate_event_coef": 0.24,
                "target_kl": 0.0044,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v56_partial_observation_handoff_memory_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v55_coverage_recovery_mappo")
        kwargs.update(
            {
                "coverage_recovery_gate_bias_scale": 1.82,
                "coverage_recovery_gate_min_scale": 0.35,
                "coverage_recovery_gate_fallback_suppression_scale": 2.80,
                "coverage_recovery_gate_fast_suppression_scale": 1.72,
                "coverage_recovery_gate_current_suppression_scale": 0.46,
                "coverage_recovery_gate_prepare_credit": 1.35,
                "coverage_recovery_gate_fallback_penalty": 1.85,
                "coverage_recovery_final_guard_enabled": True,
                "coverage_recovery_final_guard_min_scale": 0.18,
                "coverage_recovery_final_guard_min_confidence": 0.35,
                "coverage_recovery_target_memory_option_credit": 0.72,
                "coverage_recovery_target_memory_option_penalty": 1.25,
                "net_advantage_prepare_gate_bias": 2.95,
                "net_advantage_prepare_gate_policy_coef": 0.48,
                "net_advantage_prepare_gate_event_coef": 0.66,
                "net_advantage_prepare_gate_clip": 2.25,
                "option_gate_prd_coef": 0.36,
                "option_gate_counterfactual_coef": 0.44,
                "handoff_risk_option_coef": 0.46,
                "service_completion_gate_bias": 3.20,
                "service_completion_gate_policy_coef": 0.34,
                "service_completion_gate_event_coef": 0.22,
                "target_kl": 0.0042,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v57_service_continuity_counterfactual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v56_partial_observation_handoff_memory_mappo"
        )
        kwargs.update(
            {
                "idle_popularity_no_rsu_service_continuity_enabled": True,
                "opportunity_constrained_no_rsu_service_bias": 4.20,
                "opportunity_constrained_no_rsu_local_penalty": 4.80,
                "service_continuity_teacher_enabled": True,
                "service_continuity_current_bonus": 1.05,
                "service_continuity_prepare_bonus": 0.78,
                "service_continuity_local_penalty": 1.45,
                "service_continuity_min_prepare_context": 0.16,
                "option_gate_counterfactual_prd_enabled": True,
                "option_gate_prd_coef": 0.46,
                "option_gate_counterfactual_coef": 0.62,
                "option_gate_counterfactual_clip": 1.80,
                "option_gate_log_prob_weight": 0.74,
                "env_action_ppo_coef": 1.08,
                "env_action_ppo_teacher_coef": 0.76,
                "env_action_ppo_mechanism_focus": 1.18,
                "env_action_ppo_max_weight": 2.85,
                "env_action_ppo_ratio_barrier_coef": 0.072,
                "env_action_ppo_ratio_barrier_margin": 0.30,
                "counterfactual_teacher_local_bonus": 0.0,
                "counterfactual_teacher_current_rsu_penalty": 0.0,
                "counterfactual_teacher_invalid_mechanism_penalty": 0.64,
                "net_utility_idle_prefetch_penalty": 0.62,
                "net_utility_failed_mechanism_penalty": 0.72,
                "net_utility_mechanism_window_failed_penalty_scale": 0.36,
                "coverage_recovery_gate_min_scale": 0.28,
                "coverage_recovery_final_guard_min_scale": 0.12,
                "target_kl": 0.0042,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v58_ready_aware_counterfactual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v57_service_continuity_counterfactual_mappo"
        )
        kwargs.update(
            {
                "counterfactual_teacher_prd_enabled": True,
                "counterfactual_teacher_event_coef": 0.24,
                "counterfactual_teacher_option_coef": 0.66,
                "counterfactual_teacher_clip": 2.10,
                "env_action_counterfactual_margin_enabled": True,
                "env_action_counterfactual_margin_coef": 0.18,
                "env_action_counterfactual_margin_min_gap": 0.06,
                "env_action_counterfactual_margin_max_weight": 2.65,
                "env_action_counterfactual_margin_advantage_gate": 0.0,
                "env_action_counterfactual_margin_advantage_blend": 0.58,
                "idle_popularity_no_rsu_any_action_override_enabled": True,
                "option_gate_idle_recovery_mechanism_prior_enabled": True,
                "option_gate_idle_recovery_min_context": 0.0,
                "option_gate_deterministic_prior_margin": 0.96,
                "option_gate_prior_logit_bias": 1.95,
                "option_gate_prior_coef": 0.052,
                "option_gate_prior_warmup_updates": 6,
                "option_gate_prior_decay": 0.82,
                "option_gate_loss_coef": 0.40,
                "option_gate_log_prob_weight": 0.86,
                "opportunity_constrained_no_rsu_prepare_bias": 3.10,
                "opportunity_constrained_no_rsu_prepare_min_context": 0.0,
                "opportunity_constrained_no_rsu_service_bias": 5.00,
                "opportunity_constrained_no_rsu_local_penalty": 5.70,
                "service_continuity_current_bonus": 1.28,
                "service_continuity_prepare_bonus": 1.16,
                "service_continuity_local_penalty": 1.95,
                "service_continuity_min_prepare_context": 0.06,
                "net_utility_idle_prefetch_penalty": 0.38,
                "net_utility_failed_mechanism_penalty": 0.54,
                "net_utility_mechanism_window_failed_penalty_scale": 0.24,
                "env_action_ppo_teacher_coef": 0.96,
                "env_action_ppo_mechanism_focus": 1.45,
                "env_action_ppo_max_weight": 3.10,
                "coverage_recovery_gate_min_scale": 0.18,
                "coverage_recovery_final_guard_min_scale": 0.06,
                "target_kl": 0.0042,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v59_sparse_recovery_curriculum_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v58_ready_aware_counterfactual_mappo"
        )
        kwargs.update(
            {
                "counterfactual_teacher_event_coef": 0.30,
                "counterfactual_teacher_option_coef": 0.82,
                "counterfactual_teacher_clip": 2.25,
                "env_action_counterfactual_margin_coef": 0.26,
                "env_action_counterfactual_margin_max_weight": 3.10,
                "env_action_counterfactual_margin_advantage_blend": 0.62,
                "env_action_sparse_recovery_focus": 1.18,
                "option_gate_deterministic_prior_margin": 1.00,
                "option_gate_prior_logit_bias": 2.35,
                "option_gate_prior_coef": 0.060,
                "option_gate_prior_decay": 0.86,
                "option_gate_loss_coef": 0.46,
                "option_gate_log_prob_weight": 0.92,
                "opportunity_constrained_no_rsu_prepare_bias": 3.60,
                "opportunity_constrained_no_rsu_service_bias": 5.45,
                "opportunity_constrained_no_rsu_local_penalty": 6.30,
                "service_continuity_current_bonus": 1.55,
                "service_continuity_prepare_bonus": 1.42,
                "service_continuity_local_penalty": 2.65,
                "service_continuity_min_prepare_context": 0.02,
                "net_utility_idle_prefetch_penalty": 0.22,
                "net_utility_failed_mechanism_penalty": 0.42,
                "net_utility_mechanism_window_failed_penalty_scale": 0.18,
                "env_action_ppo_teacher_coef": 1.20,
                "env_action_ppo_mechanism_focus": 1.75,
                "env_action_ppo_max_weight": 3.80,
                "coverage_recovery_gate_min_scale": 0.12,
                "coverage_recovery_final_guard_min_scale": 0.04,
                "target_kl": 0.0040,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v60_risk_adjusted_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v59_sparse_recovery_curriculum_mappo"
        )
        kwargs.update(
            {
                "counterfactual_teacher_event_coef": 0.22,
                "counterfactual_teacher_option_coef": 0.58,
                "counterfactual_teacher_clip": 1.85,
                "env_action_counterfactual_margin_coef": 0.16,
                "env_action_counterfactual_margin_max_weight": 2.35,
                "env_action_counterfactual_margin_advantage_blend": 0.70,
                "env_action_sparse_recovery_focus": 0.54,
                "env_action_risk_adjusted_recovery_coef": 2.20,
                "env_action_risk_adjusted_recovery_floor": 0.16,
                "option_gate_deterministic_prior_margin": 0.72,
                "option_gate_prior_logit_bias": 1.60,
                "option_gate_prior_coef": 0.044,
                "option_gate_prior_decay": 0.78,
                "option_gate_loss_coef": 0.38,
                "option_gate_log_prob_weight": 0.76,
                "opportunity_constrained_no_rsu_prepare_bias": 2.75,
                "opportunity_constrained_no_rsu_service_bias": 5.10,
                "opportunity_constrained_no_rsu_local_penalty": 6.10,
                "service_continuity_current_bonus": 1.75,
                "service_continuity_prepare_bonus": 0.88,
                "service_continuity_local_penalty": 2.95,
                "service_continuity_min_prepare_context": 0.10,
                "net_utility_idle_prefetch_penalty": 0.38,
                "net_utility_failed_mechanism_penalty": 1.05,
                "net_utility_mechanism_window_failed_penalty_scale": 0.70,
                "env_action_ppo_teacher_coef": 0.72,
                "env_action_ppo_mechanism_focus": 1.10,
                "env_action_ppo_max_weight": 2.75,
                "env_action_ppo_ratio_barrier_coef": 0.11,
                "env_action_ppo_ratio_barrier_margin": 0.24,
                "advantage_weighted_behavior_coef": 0.20,
                "advantage_weighted_behavior_negative_coef": 1.62,
                "advantage_weighted_behavior_positive_coef": 0.72,
                "advantage_weighted_behavior_mechanism_scale": 1.32,
                "coverage_recovery_gate_min_scale": 0.08,
                "coverage_recovery_final_guard_min_scale": 0.04,
                "target_kl": 0.0036,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v60_risk_adjusted_recovery_mappo"
        )
        kwargs.update(
            {
                "counterfactual_teacher_event_coef": 0.24,
                "counterfactual_teacher_option_coef": 0.72,
                "counterfactual_teacher_clip": 3.10,
                "env_action_counterfactual_margin_coef": 0.34,
                "env_action_counterfactual_margin_max_weight": 3.40,
                "env_action_counterfactual_margin_advantage_blend": 0.76,
                "env_action_sparse_recovery_focus": 0.34,
                "env_action_risk_adjusted_recovery_coef": 2.40,
                "env_action_risk_adjusted_recovery_floor": 0.12,
                "env_action_adapter_miss_counterfactual_coef": 2.85,
                "option_gate_deterministic_prior_margin": 0.58,
                "option_gate_prior_logit_bias": 1.25,
                "option_gate_prior_coef": 0.036,
                "option_gate_prior_decay": 0.74,
                "option_gate_loss_coef": 0.34,
                "option_gate_log_prob_weight": 0.68,
                "opportunity_constrained_no_rsu_prepare_bias": 2.35,
                "opportunity_constrained_no_rsu_service_bias": 4.55,
                "opportunity_constrained_no_rsu_local_penalty": 6.30,
                "service_continuity_current_bonus": 1.05,
                "service_continuity_prepare_bonus": 0.78,
                "service_continuity_local_penalty": 3.15,
                "service_continuity_min_prepare_context": 0.14,
                "net_utility_idle_prefetch_penalty": 0.42,
                "net_utility_failed_mechanism_penalty": 1.18,
                "net_utility_mechanism_window_failed_penalty_scale": 0.78,
                "env_action_ppo_teacher_coef": 1.02,
                "env_action_ppo_mechanism_focus": 1.25,
                "env_action_ppo_max_weight": 3.20,
                "env_action_ppo_ratio_barrier_coef": 0.14,
                "env_action_ppo_ratio_barrier_margin": 0.22,
                "advantage_weighted_behavior_coef": 0.24,
                "advantage_weighted_behavior_negative_coef": 1.82,
                "advantage_weighted_behavior_positive_coef": 0.70,
                "advantage_weighted_behavior_mechanism_scale": 1.40,
                "coverage_recovery_gate_min_scale": 0.08,
                "coverage_recovery_final_guard_min_scale": 0.04,
                "target_kl": 0.0034,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v62_cache_feasibility_prior_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo"
        )
        kwargs.update(
            {
                "cache_feasibility_prior_enabled": True,
                "cache_feasibility_cache_fill_bias": 5.70,
                "cache_feasibility_steady_penalty": 7.40,
                "cache_feasibility_prepare_penalty": 1.10,
                "cache_feasibility_prefetch_penalty": 1.35,
                "cache_feasibility_min_context": 0.0,
                "env_action_adapter_miss_counterfactual_coef": 3.15,
                "env_action_risk_adjusted_recovery_coef": 2.55,
                "env_action_risk_adjusted_recovery_floor": 0.10,
                "env_action_counterfactual_margin_coef": 0.38,
                "env_action_counterfactual_margin_max_weight": 3.70,
                "env_action_ppo_teacher_coef": 1.10,
                "env_action_ppo_mechanism_focus": 1.35,
                "env_action_ppo_max_weight": 3.35,
                "option_gate_deterministic_prior_margin": 0.52,
                "option_gate_prior_logit_bias": 1.18,
                "option_gate_prior_coef": 0.034,
                "opportunity_constrained_current_bias": 3.35,
                "opportunity_constrained_no_rsu_prepare_bias": 2.15,
                "opportunity_constrained_no_rsu_service_bias": 4.80,
                "opportunity_constrained_no_rsu_local_penalty": 6.45,
                "service_continuity_current_bonus": 1.18,
                "service_continuity_prepare_bonus": 0.70,
                "service_continuity_local_penalty": 3.25,
                "service_continuity_min_prepare_context": 0.16,
                "net_utility_failed_mechanism_penalty": 1.24,
                "net_utility_mechanism_window_failed_penalty_scale": 0.82,
                "advantage_weighted_behavior_negative_coef": 1.92,
                "target_kl": 0.0030,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v63_current_ready_two_stage_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v62_cache_feasibility_prior_mappo"
        )
        kwargs.update(
            {
                "cache_feasibility_cache_fill_bias": 7.20,
                "cache_feasibility_steady_penalty": 6.20,
                "cache_feasibility_current_miss_prepare_penalty": 8.80,
                "cache_feasibility_current_miss_prefetch_penalty": 5.60,
                "cache_feasibility_prepare_penalty": 1.45,
                "cache_feasibility_prefetch_penalty": 1.65,
                "env_action_adapter_miss_counterfactual_coef": 3.45,
                "env_action_risk_adjusted_recovery_coef": 2.70,
                "env_action_counterfactual_margin_coef": 0.42,
                "env_action_counterfactual_margin_max_weight": 3.95,
                "env_action_ppo_teacher_coef": 1.18,
                "env_action_ppo_mechanism_focus": 1.45,
                "env_action_ppo_max_weight": 3.55,
                "service_continuity_current_bonus": 1.45,
                "service_continuity_prepare_bonus": 0.58,
                "service_continuity_local_penalty": 3.35,
                "net_utility_failed_mechanism_penalty": 1.32,
                "net_utility_mechanism_window_failed_penalty_scale": 0.90,
                "advantage_weighted_behavior_negative_coef": 2.05,
                "target_kl": 0.0028,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v64_handoff_alignment_barrier_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v63_current_ready_two_stage_mappo"
        )
        kwargs.update(
            {
                "handoff_alignment_barrier_enabled": True,
                "handoff_alignment_barrier_prepare_penalty": 12.50,
                "handoff_alignment_barrier_prefetch_penalty": 7.20,
                "handoff_alignment_barrier_current_fill_bias": 8.80,
                "handoff_alignment_barrier_target_mismatch_penalty": 5.60,
                "handoff_alignment_barrier_late_eta_penalty": 3.20,
                "handoff_alignment_barrier_min_context": 0.0,
                "cache_feasibility_cache_fill_bias": 8.40,
                "cache_feasibility_current_miss_prepare_penalty": 12.80,
                "cache_feasibility_current_miss_prefetch_penalty": 7.80,
                "service_continuity_current_bonus": 1.78,
                "service_continuity_prepare_bonus": 0.34,
                "env_action_adapter_miss_counterfactual_coef": 4.20,
                "env_action_risk_adjusted_recovery_coef": 3.10,
                "env_action_ppo_teacher_coef": 0.92,
                "env_action_ppo_mechanism_focus": 1.10,
                "advantage_weighted_behavior_negative_coef": 2.45,
                "target_kl": 0.0026,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v65_argmax_margin_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v64_handoff_alignment_barrier_mappo"
        )
        kwargs.update(
            {
                "argmax_margin_regularization_enabled": True,
                "argmax_margin_coef": 0.76,
                "argmax_margin_min_gap": 0.42,
                "argmax_margin_max_weight": 4.80,
                "argmax_margin_tail_risk_threshold": 0.04,
                "argmax_margin_mechanism_penalty_scale": 1.65,
                "event_logit_sharpening_final_scale": 1.55,
                "event_prepare_margin_boost": 0.12,
                "digital_twin_policy_prior_logit_bias": 2.05,
                "digital_twin_policy_prior_distill_coef": 0.026,
                "net_advantage_prepare_gate_bias": 2.18,
                "opportunity_constrained_prepare_bias": 0.58,
                "opportunity_constrained_prefetch_bias": 0.34,
                "coverage_recovery_gate_prepare_credit": 0.92,
                "cache_feasibility_cache_fill_bias": 9.20,
                "cache_feasibility_current_miss_prepare_penalty": 14.60,
                "cache_feasibility_current_miss_prefetch_penalty": 9.40,
                "service_continuity_current_bonus": 2.18,
                "service_continuity_prepare_bonus": 0.18,
                "env_action_adapter_miss_counterfactual_coef": 4.80,
                "env_action_risk_adjusted_recovery_coef": 3.45,
                "env_action_counterfactual_margin_coef": 0.58,
                "env_action_counterfactual_margin_min_gap": 0.10,
                "env_action_counterfactual_margin_max_weight": 4.60,
                "env_action_ppo_teacher_coef": 0.98,
                "env_action_ppo_mechanism_focus": 1.20,
                "advantage_weighted_behavior_negative_coef": 2.75,
                "target_kl": 0.0024,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v66_sparse_safe_cpi_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v65_argmax_margin_mappo"
        )
        kwargs.update(
            {
                "argmax_margin_coef": 0.62,
                "argmax_margin_min_gap": 0.34,
                "argmax_margin_max_weight": 4.20,
                "argmax_margin_tail_risk_threshold": 0.06,
                "argmax_margin_mechanism_penalty_scale": 1.38,
                "digital_twin_policy_prior_logit_bias": 2.85,
                "digital_twin_policy_prior_env_action_logit_bias": 4.35,
                "digital_twin_policy_prior_distill_coef": 0.058,
                "digital_twin_policy_prior_advantage_weight": 0.60,
                "digital_twin_policy_prior_max_weight": 2.85,
                "env_action_sparse_recovery_focus": 1.08,
                "env_action_risk_adjusted_recovery_coef": 3.70,
                "env_action_adapter_miss_counterfactual_coef": 4.55,
                "env_action_counterfactual_margin_coef": 0.50,
                "env_action_counterfactual_margin_max_weight": 4.20,
                "env_action_ppo_teacher_coef": 1.20,
                "env_action_ppo_mechanism_focus": 1.55,
                "env_action_ppo_max_weight": 3.90,
                "env_action_ppo_ratio_barrier_coef": 0.24,
                "env_action_ppo_ratio_barrier_margin": 0.18,
                "opportunity_constrained_prepare_bias": 0.82,
                "opportunity_constrained_prefetch_bias": 0.52,
                "net_advantage_prepare_gate_bias": 2.36,
                "net_advantage_prepare_gate_service_fill_scale": 0.18,
                "service_continuity_current_bonus": 2.05,
                "service_continuity_prepare_bonus": 0.48,
                "service_continuity_min_prepare_context": 0.08,
                "advantage_weighted_behavior_negative_coef": 3.10,
                "advantage_weighted_behavior_positive_coef": 0.78,
                "target_kl": 0.0016,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v67_sparse_handoff_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v66_sparse_safe_cpi_mappo"
        )
        kwargs.update(
            {
                "sparse_handoff_recovery_prior_enabled": True,
                "sparse_handoff_recovery_prefetch_bias": 6.80,
                "sparse_handoff_recovery_prepare_bias": 5.40,
                "sparse_handoff_recovery_current_fill_bias": 2.40,
                "sparse_handoff_recovery_steady_bias": 1.45,
                "sparse_handoff_recovery_local_penalty": 6.20,
                "sparse_handoff_recovery_min_context": 0.16,
                "sparse_handoff_recovery_max_eta": 16,
                "digital_twin_policy_prior_distill_coef": 0.064,
                "digital_twin_policy_prior_env_action_logit_bias": 4.55,
                "env_action_sparse_recovery_focus": 1.22,
                "env_action_risk_adjusted_recovery_coef": 3.85,
                "env_action_adapter_miss_counterfactual_coef": 4.80,
                "env_action_ppo_teacher_coef": 1.28,
                "env_action_ppo_ratio_barrier_coef": 0.26,
                "service_continuity_prepare_bonus": 0.56,
                "advantage_weighted_behavior_negative_coef": 3.25,
                "target_kl": 0.0015,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v68_actor_native_sparse_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v67_sparse_handoff_recovery_mappo"
        )
        kwargs.update(
            {
                "idle_popularity_fallback_enabled": False,
                "idle_popularity_no_rsu_service_continuity_enabled": False,
                "idle_popularity_no_rsu_any_action_override_enabled": False,
                "sparse_handoff_recovery_prefetch_bias": 8.80,
                "sparse_handoff_recovery_prepare_bias": 8.20,
                "sparse_handoff_recovery_current_fill_bias": 3.60,
                "sparse_handoff_recovery_steady_bias": 0.60,
                "sparse_handoff_recovery_local_penalty": 8.50,
                "sparse_handoff_recovery_min_context": 0.0,
                "sparse_handoff_recovery_max_eta": 24,
                "digital_twin_policy_prior_env_action_logit_bias": 4.85,
                "digital_twin_policy_prior_distill_coef": 0.070,
                "env_action_sparse_recovery_focus": 1.36,
                "env_action_risk_adjusted_recovery_coef": 4.05,
                "env_action_adapter_miss_counterfactual_coef": 5.10,
                "env_action_ppo_teacher_coef": 1.36,
                "env_action_ppo_ratio_barrier_coef": 0.28,
                "service_continuity_prepare_bonus": 0.64,
                "advantage_weighted_behavior_negative_coef": 3.40,
                "target_kl": 0.0014,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v69_realization_credit_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v67_sparse_handoff_recovery_mappo"
        )
        kwargs.update(
            {
                "sparse_handoff_realization_credit_enabled": True,
                "sparse_handoff_realization_success_bonus": 1.55,
                "sparse_handoff_realization_ready_bonus": 1.25,
                "sparse_handoff_realization_prefetch_bonus": 1.05,
                "sparse_handoff_realization_failed_prepare_penalty": 2.55,
                "sparse_handoff_realization_local_penalty": 1.65,
                "sparse_handoff_realization_min_context": 0.08,
                "sparse_handoff_recovery_prefetch_bias": 5.95,
                "sparse_handoff_recovery_prepare_bias": 5.10,
                "sparse_handoff_recovery_current_fill_bias": 2.65,
                "sparse_handoff_recovery_steady_bias": 1.20,
                "sparse_handoff_recovery_local_penalty": 6.05,
                "sparse_handoff_recovery_min_context": 0.12,
                "net_utility_failed_mechanism_penalty": 1.50,
                "net_utility_idle_prefetch_penalty": 0.50,
                "coverage_recovery_target_memory_option_credit": 0.62,
                "coverage_recovery_target_memory_option_penalty": 1.42,
                "env_action_sparse_recovery_focus": 1.30,
                "env_action_risk_adjusted_recovery_coef": 4.05,
                "env_action_adapter_miss_counterfactual_coef": 4.95,
                "env_action_ppo_teacher_coef": 1.32,
                "env_action_ppo_ratio_barrier_coef": 0.27,
                "target_kl": 0.0014,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v70_sparse_tail_option_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v69_realization_credit_mappo"
        )
        kwargs.update(
            {
                "sparse_handoff_option_prior_enabled": True,
                "sparse_handoff_option_prepare_bias": 7.40,
                "sparse_handoff_option_popularity_penalty": 4.35,
                "sparse_handoff_option_local_penalty": 5.85,
                "sparse_handoff_option_min_context": 0.08,
                "sparse_handoff_option_max_eta": 22,
                "sparse_handoff_recovery_prefetch_bias": 6.35,
                "sparse_handoff_recovery_prepare_bias": 5.85,
                "sparse_handoff_recovery_local_penalty": 6.45,
                "sparse_handoff_recovery_min_context": 0.10,
                "option_gate_idle_recovery_mechanism_prior_enabled": True,
                "option_gate_idle_recovery_min_context": 0.06,
                "option_gate_deterministic_prior_margin": 0.22,
                "handoff_risk_option_coef": 0.58,
                "handoff_risk_event_coef": 0.82,
                "handoff_risk_unprepared_penalty": 0.58,
                "handoff_risk_ready_bonus": 1.18,
                "handoff_risk_prepare_bonus": 0.48,
                "idle_execution_mechanism_penalty": 0.72,
                "target_kl": 0.0014,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v73_multistep_counterfactual_option_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v72_digital_twin_counterfactual_option_mappo"
        )
        kwargs.update(
            {
                "option_counterfactual_model_rollout_horizon": 4,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v74_model_based_env_action_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v71_tail_counterfactual_option_mappo"
        )
        kwargs.update(
            {
                "option_gate_enabled": False,
                "option_counterfactual_critic_enabled": False,
                "option_counterfactual_policy_improvement_enabled": False,
                "option_counterfactual_model_rollout_enabled": False,
                "env_action_model_critic_enabled": True,
                "env_action_model_critic_value_coef": 0.80,
                "env_action_model_critic_advantage_coef": 1.25,
                "env_action_model_critic_policy_improvement_coef": 2.0,
                "env_action_model_critic_advantage_clip": 2.0,
                "env_action_model_critic_warmup_updates": 2,
                "env_action_model_rollout_enabled": True,
                "env_action_model_rollout_horizon": 4,
                "env_action_ppo_coef": 1.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v75_conservative_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v74_model_based_env_action_mappo"
        )
        kwargs.update(
            {
                "env_action_model_critic_value_coef": 0.50,
                "env_action_model_critic_advantage_coef": 0.0,
                "env_action_model_critic_policy_improvement_coef": 0.0,
                "env_action_model_policy_improvement_enabled": True,
                "env_action_model_policy_improvement_coef": 0.35,
                "env_action_model_policy_improvement_temperature": 2.5,
                "env_action_ppo_coef": 0.35,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v76_recurrent_robust_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v75_conservative_model_policy_mappo"
        )
        kwargs.update(
            {
                "outcome_memory_fusion_enabled": True,
                "outcome_memory_actor_scale": 0.70,
                "outcome_memory_critic_scale": 0.85,
                "env_action_model_rollout_horizon": 8,
                "env_action_model_rollout_horizons": (1, 2, 4, 8),
                "env_action_model_policy_improvement_robust_horizons_enabled": True,
                "env_action_model_policy_improvement_horizon_risk_coef": 0.75,
                "env_action_model_policy_improvement_adaptive_kl_enabled": True,
                "env_action_model_policy_improvement_target_kl": 0.03,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v77_temporal_downside_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v76_recurrent_robust_model_policy_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_horizon_aggregation_mode": (
                    "lambda_downside"
                ),
                "env_action_model_policy_improvement_horizon_lambda": 0.90,
                "env_action_model_policy_improvement_horizon_risk_coef": 0.75,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v78_regret_adaptive_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v77_temporal_downside_model_policy_mappo"
        )
        kwargs.update(
            {
                "env_action_model_rollout_horizon": 16,
                "env_action_model_rollout_horizons": (1, 2, 4, 8, 16),
                "env_action_model_policy_improvement_regret_adaptive_kl_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_max_target_kl": 0.35,
                "env_action_model_policy_improvement_regret_priority_coef": 2.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v79_tail_distilled_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v78_regret_adaptive_model_policy_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_tail_distillation_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_tail_quantile": 0.75,
                "env_action_model_policy_improvement_tail_min_regret": 0.50,
                "env_action_model_policy_improvement_tail_epochs": 8,
                "env_action_model_policy_improvement_tail_coef": 1.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v80_imagination_replay_model_policy_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v79_tail_distilled_model_policy_mappo"
        )
        kwargs.update(
            {
                "env_action_model_imagination_replay_enabled": True,
                "env_action_model_imagination_replay_depths": (2, 4, 8),
                "env_action_model_policy_improvement_tail_epochs": 4,
                "env_action_model_policy_improvement_tail_coef": 0.75,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v81_trust_region_imagination_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v80_imagination_replay_model_policy_mappo"
        )
        kwargs.update(
            {
                "env_action_model_imagination_replay_horizons": (1, 2, 4),
                "env_action_model_policy_improvement_tail_max_policy_kl": 0.05,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v82_recovery_imagination_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v81_trust_region_imagination_mappo"
        )
        kwargs.update(
            {
                "env_action_model_imagination_replay_recovery_only": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v83_recovery_residual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v82_recovery_imagination_mappo"
        )
        kwargs.update(
            {
                "outcome_recovery_residual_enabled": True,
                "outcome_recovery_residual_scale": 1.0,
                "env_action_model_policy_improvement_tail_recovery_only": True,
                "env_action_model_policy_improvement_tail_adapter_only": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v84_search_distilled_residual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v83_recovery_residual_mappo"
        )
        kwargs.update(
            {
                "digital_twin_planning_residual_enabled": True,
                "digital_twin_planning_residual_scale": 1.0,
                "env_action_model_beam_search_enabled": True,
                "env_action_model_beam_search_horizon": 6,
                "env_action_model_beam_search_width": 2,
                "env_action_model_beam_search_context_only": True,
                "env_action_model_beam_search_min_eta": 2,
                "env_action_model_beam_search_max_eta": 8,
                "env_action_model_policy_improvement_prefer_beam_targets": True,
                "env_action_model_policy_improvement_tail_recovery_only": False,
                "env_action_model_policy_improvement_tail_adapter_only": False,
                "env_action_model_policy_improvement_tail_beam_only": True,
                "env_action_model_policy_improvement_tail_planning_adapter_only": (
                    True
                ),
                "env_action_model_policy_improvement_tail_epochs": 8,
                "env_action_model_policy_improvement_tail_coef": 1.0,
                "env_action_model_policy_improvement_tail_max_policy_kl": 0.20,
                "env_action_model_policy_improvement_tail_min_regret": 0.35,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v85_trust_region_residual_search_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v84_search_distilled_residual_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_tail_epochs": 24,
                "env_action_model_policy_improvement_tail_max_policy_kl": 0.35,
                "env_action_model_policy_improvement_tail_residual_optimizer_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_tail_residual_learning_rate": (
                    0.02
                ),
                "env_action_model_policy_improvement_tail_residual_backtrack_factor": (
                    0.5
                ),
                "env_action_model_policy_improvement_tail_residual_min_learning_rate": (
                    0.00015625
                ),
                "env_action_model_policy_improvement_tail_residual_max_backtracks": (
                    7
                ),
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v86_recovery_beam_replay_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v85_trust_region_residual_search_mappo"
        )
        kwargs.update(
            {
                "env_action_model_imagination_beam_search_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v87_mirror_residual_projection_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v86_recovery_beam_replay_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_tail_logit_projection_enabled": (
                    True
                ),
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v88_contextual_recovery_expert_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v87_mirror_residual_projection_mappo"
        )
        kwargs.update(
            {
                "outcome_context_residual_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v89_diverse_recovery_branch_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v88_contextual_recovery_expert_mappo"
        )
        kwargs.update(
            {
                "env_action_model_imagination_replay_branch_mode": "top_k",
                "env_action_model_imagination_replay_branch_top_k": 2,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v90_causally_scoped_recovery_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v89_diverse_recovery_branch_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_tail_adapter_only": True,
                "env_action_model_policy_improvement_tail_planning_adapter_only": (
                    False
                ),
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v91_balanced_recovery_distillation_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v90_causally_scoped_recovery_mappo"
        )
        kwargs.update(
            {
                "env_action_model_policy_improvement_tail_target_balance_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_tail_target_balance_power": (
                    0.5
                ),
                "env_action_model_policy_improvement_tail_target_balance_max_weight": (
                    4.0
                ),
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v92_online_counterfactual_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v91_balanced_recovery_distillation_mappo"
        )
        kwargs.update(
            {
                "env_action_model_online_planner_enabled": True,
                "env_action_model_online_planner_coef": 1.0,
                "env_action_model_online_planner_policy_prior_coef": 0.15,
                "env_action_model_online_planner_min_margin": 0.0,
                "env_action_model_online_planner_prefer_beam_targets": False,
                "env_action_model_imagination_beam_search_enabled": False,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v93_mechanism_aware_online_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v92_online_counterfactual_mappo"
        )
        kwargs.update(
            {
                "env_action_model_online_planner_mechanism_coef": 2.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v94_ucc_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v71_tail_counterfactual_option_mappo"
        )
        kwargs.update(
            {
                "option_gate_enabled": False,
                "option_counterfactual_critic_enabled": False,
                "option_counterfactual_policy_improvement_enabled": False,
                "option_counterfactual_model_rollout_enabled": False,
                "env_action_model_critic_enabled": False,
                "env_action_model_rollout_enabled": False,
                "env_action_model_policy_improvement_enabled": True,
                "env_action_model_policy_improvement_coef": 0.22,
                "env_action_model_policy_improvement_temperature": 2.0,
                "env_action_model_policy_improvement_robust_horizons_enabled": False,
                "env_action_model_online_planner_enabled": False,
                "env_action_ppo_enabled": True,
                "env_action_ppo_coef": 0.35,
                "learned_transition_model_enabled": True,
                "learned_transition_model_planner_enabled": True,
                "learned_transition_model_ensemble_size": 5,
                "learned_transition_model_hidden_dim": 64,
                "learned_transition_model_learning_rate": 3e-3,
                "learned_transition_model_fit_epochs": 4,
                "learned_transition_model_max_samples": 4096,
                "learned_transition_model_min_samples": 64,
                "learned_transition_model_discount": 0.99,
                "learned_transition_model_risk_coef": 0.65,
                "learned_transition_model_policy_coef": 1.0,
                "learned_transition_model_policy_prior_coef": 0.15,
                "learned_transition_model_min_margin": 0.02,
                "learned_transition_model_warmup_updates": 1,
                "train_epochs": 8,
                "target_kl": 0.015,
                "kl_early_stop_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v95_ucc_risk_calibrated_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v94_ucc_mappo"
        )
        kwargs.update(
            {
                # Reduce pessimism for under-covered but semantically valid steady actions.
                "learned_transition_model_risk_coef": 0.25,
                "learned_transition_model_policy_prior_coef": 0.05,
                "learned_transition_model_min_margin": 0.01,
                "learned_transition_model_policy_coef": 1.10,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v96_ucc_ucb_exploration_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v94_ucc_mappo"
        )
        kwargs.update(
            {
                # Train with calibrated UCB exploration, then switch to LCB at evaluation.
                "learned_transition_model_exploration_coef": 0.80,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v97_ucc_counterfactual_calibration_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v94_ucc_mappo"
        )
        kwargs.update(
            {
                # Exact one-step branches calibrate UCC replay; action selection remains UCC-LCB.
                "env_action_model_rollout_enabled": True,
                "env_action_model_rollout_horizon": 1,
                "env_action_model_rollout_horizons": [1],
                "env_action_model_policy_improvement_enabled": False,
                "env_action_model_online_planner_enabled": False,
                "learned_transition_model_exploration_coef": 0.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v98_ucc_counterfactual_policy_improvement_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v97_ucc_counterfactual_calibration_mappo"
        )
        kwargs.update(
            {
                # UCC supplies calibrated model support; exact one-step targets provide a second policy-improvement view.
                "env_action_model_online_planner_enabled": True,
                "env_action_model_online_planner_coef": 1.0,
                "env_action_model_online_planner_policy_prior_coef": 0.15,
                "env_action_model_online_planner_min_margin": 0.0,
                "env_action_model_online_planner_mechanism_coef": 0.0,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v99_adaptive_resource_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v98_ucc_counterfactual_policy_improvement_mappo"
        )
        kwargs.update(
            {
                # Native-action MAPPO is improved with exact multi-horizon branch
                # targets; resource cost is an auxiliary constrained objective.
                "env_action_model_rollout_enabled": True,
                "env_action_model_rollout_horizon": 6,
                "env_action_model_rollout_horizons": [1, 3, 6],
                "env_action_model_policy_improvement_enabled": True,
                "env_action_model_policy_improvement_coef": 0.30,
                "env_action_model_policy_improvement_robust_horizons_enabled": True,
                "env_action_model_policy_improvement_horizon_aggregation_mode": (
                    "lambda_downside"
                ),
                "env_action_model_policy_improvement_horizon_lambda": 0.85,
                "env_action_model_policy_improvement_horizon_risk_coef": 0.45,
                "env_action_model_policy_improvement_adaptive_kl_enabled": True,
                "env_action_model_policy_improvement_target_kl": 0.025,
                "env_action_model_adaptive_horizon_enabled": True,
                "env_action_model_adaptive_horizon_temperature": 1.0,
                "env_action_model_resource_constraint_enabled": True,
                "env_action_model_resource_cost_coef": 0.22,
                "env_action_model_resource_cost_scale": 64.0,
                "env_action_model_online_planner_coef": 1.0,
                "env_action_model_online_planner_policy_prior_coef": 0.10,
                "env_action_model_online_planner_min_margin": 0.01,
                "env_action_ppo_coef": 0.30,
                "train_epochs": 6,
                "target_kl": 0.012,
                "kl_early_stop_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v72_digital_twin_counterfactual_option_mappo":
        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v71_tail_counterfactual_option_mappo"
        )
        kwargs.update(
            {
                "option_counterfactual_policy_improvement_deterministic_only": True,
                "option_counterfactual_model_rollout_enabled": True,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v71_tail_counterfactual_option_mappo":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "digital_twin_handoff_fusion_enabled": True,
            "digital_twin_handoff_slow_scale": 0.30,
            "digital_twin_handoff_fast_scale": 0.35,
            "digital_twin_handoff_event_scale": 0.70,
            "digital_twin_handoff_critic_scale": 0.85,
            "head_credit_enabled": True,
            "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
            "slow_policy_credit_floor": 0.18,
            "fast_policy_credit_floor": 0.12,
            "event_policy_credit_floor": 0.16,
            "slow_entropy_coef_scale": 1.05,
            "fast_entropy_coef_scale": 1.00,
            "event_entropy_coef_scale": 1.15,
            "slow_entropy_credit_floor": 0.10,
            "fast_entropy_credit_floor": 0.08,
            "event_entropy_credit_floor": 0.10,
            "event_advantage_blend": 0.82,
            "event_logit_temperature": 1.0,
            "event_logit_temperature_final": 1.0,
            "event_logit_sharpening_final_scale": 1.0,
            "event_logit_sharpening_timing_gain": 0.0,
            "event_prepare_margin_boost": 0.0,
            "temporal_consistency_coef": 0.0,
            "mechanism_logit_bias_strength": 0.0,
            "continuity_guard_enabled": False,
            "handoff_target_alignment_guard_enabled": False,
            "heuristic_imitation_coef": 0.0,
            "mechanism_aux_coef": 0.0,
            "mechanism_entropy_coef": 0.0,
            "backhaul_guard_enabled": False,
            "cache_warm_start_guard_enabled": False,
            "predictive_prefetch_admission_guard_enabled": False,
            "idle_popularity_fallback_enabled": False,
            "idle_popularity_no_rsu_local_fallback_enabled": False,
            "idle_popularity_no_rsu_service_continuity_enabled": False,
            "idle_popularity_no_rsu_any_action_override_enabled": False,
            "option_gate_enabled": True,
            "option_gate_count": 4,
            "option_gate_loss_coef": 1.0,
            "option_gate_entropy_coef": 0.004,
            "option_gate_prior_coef": 0.0,
            "option_gate_prior_logit_bias": 0.0,
            "option_gate_log_prob_weight": 1.0,
            "option_gate_context_prior_enabled": False,
            "option_gate_deterministic_prior_margin": 0.0,
            "option_gate_idle_prior_enabled": False,
            "option_gate_idle_recovery_mechanism_prior_enabled": False,
            "option_gate_mechanism_preserve_enabled": False,
            "option_gate_prd_enabled": False,
            "option_gate_counterfactual_prd_enabled": False,
            "option_counterfactual_critic_enabled": True,
            "option_counterfactual_value_coef": 0.65,
            "option_counterfactual_advantage_coef": 1.20,
            "option_counterfactual_advantage_clip": 2.0,
            "option_counterfactual_warmup_updates": 2,
            "option_counterfactual_tail_weight": 2.25,
            "option_counterfactual_policy_improvement_enabled": True,
            "option_counterfactual_policy_improvement_coef": 1.0,
            "option_counterfactual_policy_improvement_clip": 2.0,
            "sparse_handoff_option_prior_enabled": False,
            "sparse_handoff_option_prepare_bias": 0.0,
            "sparse_handoff_option_popularity_penalty": 0.0,
            "sparse_handoff_option_local_penalty": 0.0,
            "sparse_handoff_option_min_context": 0.08,
            "sparse_handoff_option_max_eta": 22,
            "sparse_handoff_recovery_prior_enabled": False,
            "sparse_handoff_realization_credit_enabled": False,
            "digital_twin_policy_prior_enabled": False,
            "digital_twin_policy_prior_logit_bias": 0.0,
            "digital_twin_policy_prior_env_action_bias_enabled": False,
            "digital_twin_policy_prior_env_action_logit_bias": 0.0,
            "digital_twin_policy_prior_distill_coef": 0.0,
            "cache_feasibility_prior_enabled": False,
            "handoff_alignment_barrier_enabled": False,
            "opportunity_constrained_policy_enabled": False,
            "net_advantage_prepare_gate_enabled": False,
            "coverage_recovery_guard_enabled": False,
            "coverage_recovery_final_guard_enabled": False,
            "service_completion_gate_enabled": False,
            "backhaul_aware_policy_enabled": False,
            "advantage_weighted_behavior_regularization_enabled": False,
            "counterfactual_teacher_prd_enabled": False,
            "service_continuity_teacher_enabled": False,
            "tail_risk_prd_enabled": False,
            "opportunity_prd_enabled": False,
            "handoff_risk_prd_enabled": False,
            "idle_execution_prd_enabled": False,
            "net_utility_prd_enabled": False,
            "delayed_mechanism_credit_enabled": False,
            "env_action_ppo_enabled": True,
            "env_action_ppo_coef": 0.35,
            "env_action_ppo_advantage_blend": 0.65,
            "env_action_ppo_teacher_coef": 0.0,
            "env_action_ppo_mechanism_focus": 0.0,
            "env_action_sparse_recovery_focus": 0.0,
            "env_action_risk_adjusted_recovery_coef": 0.0,
            "env_action_adapter_miss_counterfactual_coef": 0.0,
            "env_action_counterfactual_margin_enabled": False,
            "argmax_margin_regularization_enabled": False,
            "train_epochs": 8,
            "target_kl": 0.015,
            "kl_early_stop_enabled": True,
        }
    if profile == "top_journal_mechanism_v5_perf_robust":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v1")
        kwargs.update(
            {
                "event_prepare_margin_boost": 0.60,
                "temporal_prepare_activation_threshold": 0.28,
                "event_logit_temperature_final": 0.72,
                "event_logit_sharpening_final_scale": 2.75,
                "mechanism_window_weight": 1.75,
                "prepare_action_prior_weight": 0.65,
                "heuristic_imitation_warmup_updates": 8,
                "heuristic_imitation_decay": 0.92,
                "predictive_prepare_hard_override_enabled": False,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v2":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v1")
        kwargs.update(
            {
                "auxiliary_fast_weight": 0.90,
                "heuristic_imitation_coef": 0.06,
                "heuristic_imitation_warmup_updates": 4,
                "heuristic_imitation_decay": 0.70,
                "latency_fallback_bias_enabled": True,
                "latency_fallback_bias_strength": 1.20,
                "latency_fallback_confidence_floor": 0.62,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v3":
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v1")
        kwargs.update(
            {
                "latency_fallback_bias_enabled": True,
                "latency_fallback_bias_strength": 1.20,
                "latency_fallback_confidence_floor": 0.62,
                "latency_fallback_slow_suppression_strength": 1.20,
            }
        )
        return kwargs
    if profile == "top_journal_mechanism_v1":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.10,
            "mechanism_confidence_floor": 0.42,
            "event_policy_credit_floor": 0.25,
            "event_advantage_blend": 0.92,
            "event_entropy_coef_scale": 1.15,
            "event_entropy_credit_floor": 0.08,
            "event_logit_temperature": 1.30,
            "event_logit_temperature_final": 0.75,
            "event_logit_sharpening_final_scale": 2.50,
            "event_logit_sharpening_timing_gain": 0.95,
            "event_actor_loss_extra_gain": 1.25,
            "event_prepare_margin_boost": 0.50,
            "temporal_consistency_coef": 0.50,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.0,
            "temporal_prepare_activation_threshold": 0.30,
            "auxiliary_slow_weight": 1.15,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.75,
            "train_epochs": 7,
            "target_kl": 0.012,
            "kl_early_stop_enabled": True,
            "continuity_guard_enabled": True,
            "handoff_target_alignment_guard_enabled": True,
            "continuity_guard_logit_penalty": 1.30,
            "continuity_guard_prepare_boost": 1.60,
            "continuity_guard_confidence_threshold": 0.36,
            "continuity_guard_prepare_score_threshold": 0.22,
            "continuity_guard_hard_override_enabled": False,
            "heuristic_imitation_coef": 0.10,
            "heuristic_imitation_warmup_updates": 6,
            "heuristic_imitation_decay": 0.88,
            "mechanism_aux_coef": 0.09,
            "mechanism_window_weight": 1.50,
            "prepare_action_prior_weight": 0.60,
            "mechanism_entropy_coef": 0.0006,
            "mechanism_retention_start_update": 6,
            "mechanism_aux_coef_floor_after_update": 0.08,
            "mechanism_window_weight_floor_after_update": 1.50,
            "mechanism_entropy_floor_after_update": 0.0006,
            "mechanism_aux_current_cache_fill_enabled": True,
            "backhaul_guard_enabled": True,
            "backhaul_guard_max_reactive_fills_per_adapter": 1,
            "cache_warm_start_guard_enabled": True,
            "cache_warm_start_guard_min_countdown": 0.0,
        }
    if profile == "sa_reward_tiebreak_round4":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.05,
            "mechanism_confidence_floor": 0.42,
            "event_policy_credit_floor": 0.22,
            "event_advantage_blend": 0.9,
            "event_logit_temperature": 1.35,
            "event_logit_temperature_final": 0.78,
            "event_logit_sharpening_final_scale": 2.4,
            "event_logit_sharpening_timing_gain": 0.9,
            "event_actor_loss_extra_gain": 1.2,
            "event_prepare_margin_boost": 0.45,
            "temporal_consistency_coef": 0.48,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.05,
            "temporal_prepare_activation_threshold": 0.32,
            "auxiliary_slow_weight": 1.1,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.45,
            "train_epochs": 4,
            "target_kl": 0.010,
            "kl_early_stop_enabled": True,
            "continuity_guard_enabled": True,
            "handoff_target_alignment_guard_enabled": True,
            "continuity_guard_logit_penalty": 1.25,
            "continuity_guard_prepare_boost": 1.55,
            "continuity_guard_confidence_threshold": 0.38,
            "continuity_guard_prepare_score_threshold": 0.24,
            "continuity_guard_hard_override_enabled": False,
            "heuristic_imitation_coef": 0.05,
            "heuristic_imitation_warmup_updates": 2,
            "heuristic_imitation_decay": 0.55,
            "mechanism_aux_coef": 0.04,
            "mechanism_window_weight": 1.15,
            "prepare_action_prior_weight": 0.25,
            "mechanism_entropy_coef": 0.0003,
        }
    if profile == "sa_mechanism_retention_round3":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.05,
            "mechanism_confidence_floor": 0.42,
            "event_policy_credit_floor": 0.22,
            "event_advantage_blend": 0.9,
            "event_logit_temperature": 1.35,
            "event_logit_temperature_final": 0.78,
            "event_logit_sharpening_final_scale": 2.4,
            "event_logit_sharpening_timing_gain": 0.9,
            "event_actor_loss_extra_gain": 1.2,
            "event_prepare_margin_boost": 0.45,
            "temporal_consistency_coef": 0.48,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.05,
            "temporal_prepare_activation_threshold": 0.32,
            "auxiliary_slow_weight": 1.1,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.7,
            "train_epochs": 7,
            "target_kl": 0.012,
            "kl_early_stop_enabled": True,
            "continuity_guard_enabled": True,
            "handoff_target_alignment_guard_enabled": True,
            "continuity_guard_logit_penalty": 1.25,
            "continuity_guard_prepare_boost": 1.55,
            "continuity_guard_confidence_threshold": 0.38,
            "continuity_guard_prepare_score_threshold": 0.24,
            "continuity_guard_hard_override_enabled": False,
            "heuristic_imitation_coef": 0.12,
            "heuristic_imitation_warmup_updates": 4,
            "heuristic_imitation_decay": 0.75,
            "mechanism_aux_coef": 0.08,
            "mechanism_window_weight": 1.35,
            "prepare_action_prior_weight": 0.55,
            "mechanism_entropy_coef": 0.0005,
            "mechanism_retention_start_update": 8,
            "mechanism_aux_coef_floor_after_update": 0.10,
            "mechanism_window_weight_floor_after_update": 1.40,
            "mechanism_entropy_floor_after_update": 0.0008,
        }
    if profile == "sa_mechanism_policy_round2":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.05,
            "mechanism_confidence_floor": 0.42,
            "event_policy_credit_floor": 0.22,
            "event_advantage_blend": 0.9,
            "event_logit_temperature": 1.35,
            "event_logit_temperature_final": 0.78,
            "event_logit_sharpening_final_scale": 2.4,
            "event_logit_sharpening_timing_gain": 0.9,
            "event_actor_loss_extra_gain": 1.2,
            "event_prepare_margin_boost": 0.45,
            "temporal_consistency_coef": 0.48,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.05,
            "temporal_prepare_activation_threshold": 0.32,
            "auxiliary_slow_weight": 1.1,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.7,
            "train_epochs": 7,
            "target_kl": 0.012,
            "kl_early_stop_enabled": True,
            "continuity_guard_enabled": True,
            "handoff_target_alignment_guard_enabled": True,
            "continuity_guard_logit_penalty": 1.25,
            "continuity_guard_prepare_boost": 1.55,
            "continuity_guard_confidence_threshold": 0.38,
            "continuity_guard_prepare_score_threshold": 0.24,
            "continuity_guard_hard_override_enabled": False,
            "heuristic_imitation_coef": 0.12,
            "heuristic_imitation_warmup_updates": 4,
            "heuristic_imitation_decay": 0.75,
            "mechanism_aux_coef": 0.08,
            "mechanism_window_weight": 1.35,
            "prepare_action_prior_weight": 0.55,
            "mechanism_entropy_coef": 0.0005,
        }
    if profile == "sa_advantage_round1":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.05,
            "mechanism_confidence_floor": 0.42,
            "event_policy_credit_floor": 0.22,
            "event_advantage_blend": 0.9,
            "event_logit_temperature": 1.35,
            "event_logit_temperature_final": 0.78,
            "event_logit_sharpening_final_scale": 2.4,
            "event_logit_sharpening_timing_gain": 0.9,
            "event_actor_loss_extra_gain": 1.2,
            "event_prepare_margin_boost": 0.45,
            "temporal_consistency_coef": 0.48,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.05,
            "temporal_prepare_activation_threshold": 0.32,
            "auxiliary_slow_weight": 1.1,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.7,
            "train_epochs": 7,
            "target_kl": 0.012,
            "kl_early_stop_enabled": True,
            "continuity_guard_enabled": True,
            "handoff_target_alignment_guard_enabled": True,
            "continuity_guard_logit_penalty": 1.25,
            "continuity_guard_prepare_boost": 1.55,
            "continuity_guard_confidence_threshold": 0.38,
            "continuity_guard_prepare_score_threshold": 0.24,
            "continuity_guard_hard_override_enabled": False,
            "heuristic_imitation_coef": 0.12,
            "heuristic_imitation_warmup_updates": 3,
            "heuristic_imitation_decay": 0.65,
        }
    if profile == "formal_main_stable":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 1.0,
            "mechanism_confidence_floor": 0.4,
            "event_policy_credit_floor": 0.2,
            "event_advantage_blend": 0.9,
            "event_logit_temperature": 1.45,
            "event_logit_temperature_final": 0.8,
            "event_logit_sharpening_final_scale": 2.3,
            "event_logit_sharpening_timing_gain": 0.85,
            "event_actor_loss_extra_gain": 1.15,
            "temporal_consistency_coef": 0.45,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.1,
            "temporal_prepare_activation_threshold": 0.35,
            "auxiliary_slow_weight": 1.25,
            "auxiliary_fast_weight": 0.2,
            "auxiliary_event_weight": 1.6,
            "train_epochs": 6,
            "target_kl": 0.012,
            "kl_early_stop_enabled": True,
        }
    if profile == "formal_main":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 0.9,
            "mechanism_confidence_floor": 0.35,
            "event_policy_credit_floor": 0.18,
            "event_advantage_blend": 0.88,
            "event_logit_temperature": 1.4,
            "event_logit_temperature_final": 0.82,
            "event_logit_sharpening_final_scale": 2.3,
            "event_logit_sharpening_timing_gain": 0.85,
            "event_actor_loss_extra_gain": 1.15,
            "temporal_consistency_coef": 0.4,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.15,
            "temporal_prepare_activation_threshold": 0.35,
            "auxiliary_slow_weight": 1.15,
            "auxiliary_fast_weight": 0.25,
            "auxiliary_event_weight": 1.4,
            "train_epochs": 8,
            "target_kl": 0.0,
            "kl_early_stop_enabled": False,
        }
    if profile == "baseline_safe":
        return {
            "graph_continuity_critic_enabled": True,
            "uncertainty_aware_event_scaling_enabled": True,
            "uncertainty_aware_critic_enabled": True,
            "head_credit_enabled": True,
            "mechanism_logit_bias_strength": 0.6,
            "mechanism_confidence_floor": 0.2,
            "event_policy_credit_floor": 0.15,
            "event_advantage_blend": 0.85,
            "event_logit_temperature": 1.4,
            "event_logit_temperature_final": 0.85,
            "event_logit_sharpening_final_scale": 2.3,
            "event_logit_sharpening_timing_gain": 0.85,
            "event_actor_loss_extra_gain": 1.15,
            "temporal_consistency_coef": 0.35,
            "temporal_prepare_lead_steps": 2.5,
            "temporal_prepare_sigma": 1.25,
            "temporal_prepare_activation_threshold": 0.35,
            "auxiliary_slow_weight": 1.0,
            "auxiliary_fast_weight": 0.35,
            "auxiliary_event_weight": 1.15,
            "train_epochs": 6,
            "target_kl": 0.0,
            "kl_early_stop_enabled": False,
        }
    return {
        "graph_continuity_critic_enabled": True,
        "uncertainty_aware_event_scaling_enabled": True,
        "uncertainty_aware_critic_enabled": True,
        "head_credit_enabled": False,
        "mechanism_logit_bias_strength": 0.0,
        "mechanism_confidence_floor": 0.0,
        "auxiliary_slow_weight": 1.0,
        "auxiliary_fast_weight": 0.5,
        "auxiliary_event_weight": 1.0,
        "train_epochs": 4,
        "target_kl": 0.0,
        "kl_early_stop_enabled": False,
    }


def build_agent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "random_seed": args.random_seed,
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "batch_size": args.batch_size,
        "deterministic_action": False,
    }
    planned_updates = max(1, math.ceil(float(args.episodes) / float(max(args.update_every, 1))))
    kwargs.update(
        {
            "auxiliary_coef": args.auxiliary_coef,
            "use_prediction_features": not args.disable_prediction,
            "use_graph_encoder": not args.disable_graph_encoder,
            "use_hierarchy": not args.disable_hierarchy,
            "use_event_agent": not args.disable_event_agent,
            "use_adapter_prefetch": not args.disable_adapter_prefetch,
            "use_dependency_aware": not args.disable_dag_dependency_aware,
            "use_uncertainty_signal": not args.disable_uncertainty_signal,
            "event_temperature_decay_updates": max(2, int(math.ceil(0.7 * planned_updates))),
            **build_sa_ghmappo_profile_kwargs(args.profile),
        }
    )
    if args.profile == "sa_reward_tiebreak_round4":
        kwargs["event_temperature_decay_updates"] = 12
    optional_agent_fields = [
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
        "mechanism_aux_coef",
        "mechanism_window_weight",
        "prepare_action_prior_weight",
        "mechanism_entropy_coef",
        "mechanism_retention_start_update",
        "mechanism_aux_coef_floor_after_update",
        "mechanism_window_weight_floor_after_update",
        "mechanism_entropy_floor_after_update",
        "latency_fallback_bias_enabled",
        "latency_fallback_bias_strength",
        "latency_fallback_confidence_floor",
        "latency_fallback_slow_suppression_strength",
        "steady_rsu_bias_enabled",
        "steady_rsu_bias_strength",
        "steady_rsu_confidence_floor",
        "cache_warm_start_guard_max_prefetch_countdown",
        "predictive_prefetch_admission_guard_enabled",
        "predictive_prefetch_admission_min_confidence",
        "predictive_prefetch_admission_require_distinct_next",
    ]
    for field_name in optional_agent_fields:
        value = getattr(args, field_name, None)
        if value is not None:
            if field_name == "predictive_prefetch_admission_require_distinct_next":
                value = bool(int(value))
            kwargs[field_name] = value
    return kwargs


def build_predictor_runtime_kwargs(args: argparse.Namespace, *, random_seed: int) -> dict[str, Any]:
    return {
        "horizon": int(args.prediction_horizon),
        "predictor_kind": str(args.predictor_kind),
        "predictor_checkpoint_path": str(args.predictor_checkpoint_path),
        "prediction_noise_std": float(args.prediction_noise_std),
        "prediction_confidence_scale": float(args.prediction_confidence_scale),
        "prediction_delay_steps": int(args.prediction_delay_steps),
        "drop_handoff_prediction_prob": float(args.drop_handoff_prediction_prob),
        "random_seed": int(random_seed),
    }


def gaussian_window_score(value: float, center: float, sigma: float) -> float:
    effective_sigma = max(float(sigma), 0.25)
    normalized_distance = (float(value) - float(center)) / effective_sigma
    return float(math.exp(-0.5 * normalized_distance * normalized_distance))


def build_temporal_reward_shaping_config(agent: Any, args: argparse.Namespace) -> dict[str, float | bool]:
    if (
        args.agent_name != "sa_ghmappo"
        or not bool(getattr(args, "temporal_reward_shaping_enabled", True))
    ):
        return {"enabled": False}
    preferred_lead_steps = float(getattr(agent, "_temporal_prepare_lead_steps", 2.5))
    sigma = float(getattr(agent, "_temporal_prepare_sigma", 1.25))
    return {
        "enabled": True,
        "preferred_lead_steps": preferred_lead_steps,
        "sigma": sigma,
        "prepare_bonus_scale": 1.2,
        "miss_penalty_scale": 0.45,
        "false_positive_penalty_scale": 0.6,
        "realized_bonus_scale": 0.7,
        "handoff_ready_bonus_scale": 0.9,
    }


def recompute_rollout_targets(
    rollout: list[dict[str, Any]],
    agent: Any,
    gamma: float,
    gae_lambda: float,
) -> None:
    if not rollout:
        return
    final_row = rollout[-1]
    if bool(final_row.get("terminated", False)):
        last_value = 0.0
    else:
        last_value = float(agent.evaluate_value(final_row.get("next_observation"), final_row.get("env_info")))
    gae = 0.0
    for index in reversed(range(len(rollout))):
        row = rollout[index]
        if index == len(rollout) - 1:
            next_value = last_value
        else:
            next_value = float(rollout[index + 1].get("value", 0.0))
        next_non_terminal = 0.0 if bool(row.get("terminated", False)) else 1.0
        delta = float(row.get("reward", 0.0)) + float(gamma) * next_value * next_non_terminal - float(row.get("value", 0.0))
        gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * gae
        row["advantage"] = float(gae)
        row["return"] = float(gae + float(row.get("value", 0.0)))


def attach_event_head_advantages(
    rollout: list[dict[str, Any]],
    gamma: float,
    gae_lambda: float,
) -> dict[str, float]:
    if not rollout:
        return {
            "event_advantage_mean": 0.0,
            "event_advantage_std": 0.0,
            "event_active_count": 0,
        }
    event_gae = 0.0
    raw_event_advantages = [0.0 for _ in rollout]
    active_mask = [False for _ in rollout]
    for index in reversed(range(len(rollout))):
        row = rollout[index]
        shaping = row.get("reward_shaping", {})
        metrics_protocol = row.get("env_info", {}).get("metrics_protocol", {})
        event_reward = float(shaping.get("reward_delta", 0.0))
        event_active = bool(
            shaping.get("predicted_handoff_active", False)
            or shaping.get("event_prepare_selected", False)
            or metrics_protocol.get("migration_prepare_requested", False)
            or metrics_protocol.get("migration_prepare_realized", False)
            or metrics_protocol.get("handoff_ready", False)
        )
        next_non_terminal = 0.0 if bool(row.get("terminated", False)) else 1.0
        event_gae = event_reward + float(gamma) * float(gae_lambda) * next_non_terminal * event_gae
        raw_event_advantages[index] = float(event_gae)
        active_mask[index] = event_active
    active_values = [raw_event_advantages[index] for index, is_active in enumerate(active_mask) if is_active]
    event_advantage_mean = float(fmean(active_values)) if active_values else 0.0
    event_advantage_std = float(np.std(active_values)) if active_values else 0.0
    for index, row in enumerate(rollout):
        event_advantage = raw_event_advantages[index]
        if active_mask[index]:
            normalized_event_advantage = (event_advantage - event_advantage_mean) / (event_advantage_std + 1e-8)
        else:
            normalized_event_advantage = 0.0
        row["event_advantage"] = float(event_advantage)
        row["event_advantage_normalized"] = float(normalized_event_advantage)
    return {
        "event_advantage_mean": round(event_advantage_mean, 6),
        "event_advantage_std": round(event_advantage_std, 6),
        "event_active_count": int(sum(1 for is_active in active_mask if is_active)),
    }


def apply_temporal_reward_shaping_to_rollout(
    rollout: list[dict[str, Any]],
    agent: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = build_temporal_reward_shaping_config(agent=agent, args=args)
    if not bool(config.get("enabled", False)) or not rollout:
        return {
            "enabled": False,
            "applied_row_count": 0,
            "reward_delta_sum": 0.0,
            "reward_delta_mean": 0.0,
            "positive_delta_count": 0,
            "negative_delta_count": 0,
        }

    shaping_rows: list[dict[str, Any]] = []
    reward_deltas: list[float] = []
    positive_delta_count = 0
    negative_delta_count = 0
    for row in rollout:
        semantic_state = row.get("decision_info", {}).get("semantic_state", {})
        timing_features = compute_temporal_prepare_window_score(
            semantic_state,
            preferred_lead_steps=float(config["preferred_lead_steps"]),
            sigma=float(config["sigma"]),
        )
        metrics_protocol = row.get("env_info", {}).get("metrics_protocol", {})
        event_prepare_selected = bool(
            int(row.get("action_info", {}).get("head_actions", {}).get("event", 0) or 0) == 1
            or int(row.get("action", -1)) == 4
        )
        predicted_handoff_active = bool(
            timing_features.get("has_predicted_handoff", 0.0)
            or metrics_protocol.get("predicted_handoff_signal", False)
            or metrics_protocol.get("has_predicted_handoff_target", False)
        )
        window_score = float(timing_features.get("prepare_window_score", 0.0))
        temporal_urgency = float(timing_features.get("temporal_urgency", 0.0))
        reward_delta = 0.0

        prepare_bonus = 0.0
        miss_penalty = 0.0
        false_positive_penalty = 0.0
        realized_bonus = 0.0
        handoff_ready_bonus = 0.0
        realized_score = 0.0
        prepare_age = metrics_protocol.get("migration_prepare_age")
        if prepare_age is not None:
            realized_score = gaussian_window_score(
                float(prepare_age),
                center=float(config["preferred_lead_steps"]),
                sigma=float(config["sigma"]),
            )
            window_score = max(window_score, realized_score)

        if event_prepare_selected:
            prepare_bonus = float(config["prepare_bonus_scale"]) * window_score
            reward_delta += prepare_bonus
            if not predicted_handoff_active and not bool(metrics_protocol.get("migration_prepare_realized", False)) and not bool(metrics_protocol.get("handoff_ready", False)):
                false_positive_penalty = float(config["false_positive_penalty_scale"])
                reward_delta -= false_positive_penalty
            else:
                false_positive_penalty = float(config["false_positive_penalty_scale"]) * max(0.0, 0.3 - window_score) / 0.3
                reward_delta -= false_positive_penalty
        elif predicted_handoff_active and window_score > 1e-6:
            miss_penalty = float(config["miss_penalty_scale"]) * window_score * (0.5 + 0.5 * temporal_urgency)
            reward_delta -= miss_penalty

        if bool(metrics_protocol.get("migration_prepare_realized", False)):
            if prepare_age is None:
                realized_score = max(realized_score, window_score)
            realized_bonus = float(config["realized_bonus_scale"]) * realized_score
            reward_delta += realized_bonus

        if bool(metrics_protocol.get("handoff_ready", False)):
            handoff_ready_bonus = float(config["handoff_ready_bonus_scale"]) * max(window_score, 0.5)
            reward_delta += handoff_ready_bonus

        base_reward = float(row.get("reward", 0.0))
        row["base_reward"] = base_reward
        row["reward"] = base_reward + reward_delta
        row["reward_shaping"] = {
            "timing_window_score": round(window_score, 6),
            "temporal_urgency": round(temporal_urgency, 6),
            "handoff_countdown_steps": round(float(timing_features.get("countdown_steps", 0.0)), 6),
            "event_prepare_selected": event_prepare_selected,
            "predicted_handoff_active": predicted_handoff_active,
            "prepare_bonus": round(prepare_bonus, 6),
            "miss_penalty": round(miss_penalty, 6),
            "false_positive_penalty": round(false_positive_penalty, 6),
            "realized_bonus": round(realized_bonus, 6),
            "handoff_ready_bonus": round(handoff_ready_bonus, 6),
            "reward_delta": round(reward_delta, 6),
        }
        shaping_rows.append(dict(row["reward_shaping"]))
        reward_deltas.append(reward_delta)
        if reward_delta > 1e-8:
            positive_delta_count += 1
        elif reward_delta < -1e-8:
            negative_delta_count += 1

    recompute_rollout_targets(rollout=rollout, agent=agent, gamma=args.gamma, gae_lambda=args.gae_lambda)
    event_advantage_stats = attach_event_head_advantages(
        rollout=rollout,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    reward_delta_sum = sum(reward_deltas)
    return {
        "enabled": True,
        "preferred_lead_steps": round(float(config["preferred_lead_steps"]), 6),
        "sigma": round(float(config["sigma"]), 6),
        "applied_row_count": len(shaping_rows),
        "reward_delta_sum": round(reward_delta_sum, 6),
        "reward_delta_mean": round(reward_delta_sum / max(len(shaping_rows), 1), 6),
        "positive_delta_count": positive_delta_count,
        "negative_delta_count": negative_delta_count,
        "mean_timing_window_score": round(
            fmean(float(item.get("timing_window_score", 0.0)) for item in shaping_rows),
            6,
        ) if shaping_rows else 0.0,
        **event_advantage_stats,
    }



def annotate_checkpoint_metadata(checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # Defensive audit path for partially written checkpoints.
        return {
            "exists": True,
            "load_error": f"{type(exc).__name__}: {exc}",
            "parameter_count": 0,
            "parameter_sum": 0.0,
            "parameter_abs_sum": 0.0,
            "update_count": 0,
            "config": {},
        }
    if isinstance(payload, dict):
        payload["training_metadata"] = dict(metadata)
        torch.save(payload, checkpoint_path)


def training_window_weight(window_candidate: dict[str, Any], args: argparse.Namespace) -> float:
    window_class = str(window_candidate.get("window_class", "unknown"))
    weight = 1.0
    if window_class == "mechanism_activating":
        weight *= max(float(args.mechanism_window_oversample_ratio), 1.0)
        weight *= max(float(args.handoff_imminent_oversample_ratio), 1.0)
        weight *= max(float(args.target_mismatch_sample_weight), 1.0)
    return max(weight, 1.0)


def build_training_window_plan(
    window_payload: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected_windows = list(window_payload["selected_windows"])
    min_mechanism_windows = max(int(args.min_mechanism_activating_windows), 0)
    if min_mechanism_windows > 0:
        selected_ids = {str(item.get("window_id")) for item in selected_windows}
        current_mechanism_count = len(
            [item for item in selected_windows if str(item.get("window_class", "")) == "mechanism_activating"]
        )
        for candidate in list(window_payload.get("mechanism_activating_windows", [])):
            if current_mechanism_count >= min_mechanism_windows:
                break
            if str(candidate.get("window_id")) in selected_ids:
                continue
            selected_windows.append(dict(candidate))
            selected_ids.add(str(candidate.get("window_id")))
            current_mechanism_count += 1
    mechanism_count = len(
        [item for item in selected_windows if str(item.get("window_class", "")) == "mechanism_activating"]
    )
    if str(getattr(args, "window_mode", "")) in {"activating_only", "mixed_informative", "full_stratified"} and mechanism_count <= 0:
        raise RuntimeError(
            "mechanism-aware training window plan contains no mechanism_activating windows; "
            "increase max_mobility_rows/window_count or relax the handoff-window selection protocol"
        )
    expanded_plan: list[dict[str, Any]] = []
    for window_candidate in selected_windows:
        duplicate_count = max(1, int(round(training_window_weight(dict(window_candidate), args))))
        expanded_plan.extend(dict(window_candidate) for _ in range(duplicate_count))
    return expanded_plan or selected_windows


def choose_training_window(window_plan: list[dict[str, Any]], episode_index: int, mode: str, rng: random.Random) -> dict[str, Any]:
    if not window_plan:
        raise RuntimeError("?????????")
    if mode == "fixed":
        return dict(window_plan[0])
    if mode == "rotate":
        return dict(window_plan[(episode_index - 1) % len(window_plan)])
    if mode == "sampled":
        return dict(rng.choice(window_plan))
    raise ValueError(f"?? train_window_mode: {mode}")


def build_run_scale_info(config_profile: str, episodes: int, update_count: int) -> dict[str, Any]:
    return classify_experiment_scale(config_profile=config_profile, episodes=episodes, update_count=update_count)



def build_checkpoint_fingerprint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "exists": False,
            "parameter_count": 0,
            "parameter_sum": 0.0,
            "parameter_abs_sum": 0.0,
            "update_count": 0,
            "config": {},
        }
    payload = torch.load(checkpoint_path, map_location="cpu")
    network_state = payload.get("network_state_dict", {}) if isinstance(payload, dict) else {}
    parameter_sum = 0.0
    parameter_abs_sum = 0.0
    parameter_count = 0
    for tensor in network_state.values():
        parameter_sum += float(tensor.float().sum().item())
        parameter_abs_sum += float(tensor.float().abs().sum().item())
        parameter_count += int(tensor.numel())
    return {
        "exists": True,
        "parameter_count": parameter_count,
        "parameter_sum": round(parameter_sum, 6),
        "parameter_abs_sum": round(parameter_abs_sum, 6),
        "update_count": int(payload.get("update_count", 0)) if isinstance(payload, dict) else 0,
        "config": dict(payload.get("config", {})) if isinstance(payload, dict) and isinstance(payload.get("config"), dict) else {},
    }



def _compare_agents_for_checkpoint(current_agent_name: str, checkpoint_path: Path, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    del args
    compare_agents = [current_agent_name]
    checkpoint_map = {current_agent_name: str(checkpoint_path)}
    return compare_agents, checkpoint_map


def run_real_episode_with_policy_diagnostics(
    *,
    agent_name: str,
    checkpoint_map: dict[str, str],
    workflow_state: Any,
    workflow_source_path: str,
    mobility_bundle: Any,
    seed: int,
    max_steps: int,
    mobility_source: str,
    primary_vehicle_selection: str,
    reward_positive_offset: float,
    predictor_runtime_kwargs: dict[str, Any],
    run_metadata: dict[str, Any],
    policy_evaluation_mode: str = "safety_projected",
) -> dict[str, Any]:
    policy_evaluation_mode = str(policy_evaluation_mode).strip().lower()
    if policy_evaluation_mode not in {"raw_policy", "safety_projected"}:
        raise ValueError(f"unsupported policy_evaluation_mode: {policy_evaluation_mode}")
    recorder = EpisodeRecorder(prefetch_validation_window=6)
    adapter_catalog = AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json")
    rsu_states = [clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states]
    workflow_state_runtime = clone_workflow_state(workflow_state)
    trajectory_frames = clone_frames(mobility_bundle.frames)
    checkpoint_path = resolve_agent_checkpoint(agent_name, checkpoint_map)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint_path) if checkpoint_path else {
        "checkpoint_path": "",
        "config_profile": "non_checkpoint_agent",
        "run_id": agent_name,
        "episodes": 0,
        "update_count": 0,
        "is_smoke_checkpoint": False,
        "smoke_warning": False,
    }
    core_env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=trajectory_frames),
        workflow_state=workflow_state_runtime,
        adapter_catalog=adapter_catalog,
        rsu_states=rsu_states,
        predictor_manager=PredictorManager(**predictor_runtime_kwargs),
        max_steps=max(max_steps + 2, 8),
        mobility_source=mobility_source,
        primary_vehicle_selection=primary_vehicle_selection,
        reward_positive_offset=reward_positive_offset,
    )
    env = GymVecEnv(core_env=core_env, recorder=recorder)
    agent = build_inference_agent(
        agent_name=agent_name,
        random_seed=seed,
        checkpoint_path=checkpoint_path,
        deterministic_action=True,
    )
    recorder.start_episode(
        run_metadata={
            **run_metadata,
            "workflow_id": workflow_state_runtime.workflow_id,
            "workflow_source_path": workflow_source_path,
            "agent_name": agent_name,
            "seed": seed,
            "window_id": mobility_bundle.rsu_metadata.get("window_id"),
            "rsu_layout": mobility_bundle.rsu_metadata.get("effective_rsu_layout"),
            "predictor_runtime_kwargs": dict(predictor_runtime_kwargs),
            "reward_positive_offset": float(reward_positive_offset),
            "checkpoint_run_id": checkpoint_metadata.get("run_id"),
            "checkpoint_profile": checkpoint_metadata.get("config_profile"),
            "checkpoint_is_smoke": checkpoint_metadata.get("is_smoke_checkpoint", False),
        }
    )
    observation, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    policy_step_trace: list[dict[str, Any]] = []
    gt_context = build_gt_handoff_context(env)
    while not terminated and not truncated and step_count < max_steps:
        semantic_state = dict(info.get("semantic_state", {}))
        policy_info = dict(info)
        policy_info["run_metadata"] = {
            **dict(info.get("run_metadata", {}) or {}),
            **dict(run_metadata),
            "policy_evaluation_mode": policy_evaluation_mode,
        }
        action, action_info = agent.act(observation, policy_info)
        stochastic_probe = build_stochastic_event_probe(
            agent,
            semantic_state,
            policy_evaluation_mode=policy_evaluation_mode,
        )
        gt_probe = build_gt_handoff_probe(gt_context, semantic_state)
        next_observation, _, terminated, truncated, next_info = env.step(int(action))
        step_diagnostic = build_policy_step_diagnostic(
            agent=agent,
            action=int(action),
            action_info=dict(action_info),
            semantic_state=semantic_state,
            stochastic_probe=stochastic_probe,
            gt_probe=gt_probe,
            env_info=next_info,
        )
        step_diagnostic["debug_step_index"] = int(step_count)
        step_diagnostic["policy_evaluation_mode"] = str(
            action_info.get("policy_evaluation_mode", policy_evaluation_mode)
        )
        policy_step_trace.append(step_diagnostic)
        observation = next_observation
        info = next_info
        step_count += 1
    summary = recorder.build_summary()
    summary["trainer_info"] = {
        "trainer_name": "deterministic_eval_runner",
        "max_steps": max_steps,
        "collected_steps": step_count,
    }
    summary["agent_info"] = {
        "agent_name": agent_name,
        "learn_info": {
            "agent_name": agent_name,
            "policy_update_skipped": True,
            "reason": "evaluation_only",
            "collected_steps": step_count,
        },
    }
    summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
    summary["run_info"]["rsu_metadata"] = mobility_bundle.rsu_metadata
    summary["run_info"]["checkpoint_metadata"] = checkpoint_metadata
    summary["policy_diagnostics_trace"] = list(policy_step_trace)
    summary["policy_diagnostics_summary"] = aggregate_policy_diagnostics(policy_step_trace)
    summary["policy_alignment_samples"] = summarize_policy_alignment_samples(policy_step_trace, limit=5)
    summary["policy_evaluation_mode"] = policy_evaluation_mode
    summary["policy_action_signature"] = build_policy_action_signature(policy_step_trace)
    return summary



def evaluate_checkpoint_protocol(
    *,
    current_agent_name: str,
    checkpoint_path: Path,
    workflow_states: list[Any],
    eval_windows: list[dict[str, Any]],
    args: argparse.Namespace,
    include_reference_agents: bool = True,
    max_workflows: int | None = None,
    max_windows: int | None = None,
    protocol_name: str = "checkpoint_eval",
    policy_evaluation_mode: str = "safety_projected",
) -> dict[str, Any]:
    policy_evaluation_mode = str(policy_evaluation_mode).strip().lower()
    if policy_evaluation_mode not in {"raw_policy", "safety_projected"}:
        raise ValueError(f"unsupported policy_evaluation_mode: {policy_evaluation_mode}")
    compare_agents, checkpoint_map = _compare_agents_for_checkpoint(current_agent_name=current_agent_name, checkpoint_path=checkpoint_path, args=args)
    if not include_reference_agents:
        compare_agents = [current_agent_name]
        checkpoint_map = {current_agent_name: str(checkpoint_path)}
    selected_workflows = list(workflow_states[: max_workflows or len(workflow_states)])
    selected_windows = list(eval_windows[: max_windows or len(eval_windows)])
    rows: list[dict[str, Any]] = []
    policy_step_diagnostics_by_agent: dict[str, list[dict[str, Any]]] = {agent_name: [] for agent_name in compare_agents}
    policy_alignment_samples_by_agent: dict[str, list[dict[str, Any]]] = {agent_name: [] for agent_name in compare_agents}
    policy_action_signature_rows_by_agent: dict[str, list[dict[str, Any]]] = {agent_name: [] for agent_name in compare_agents}
    for eval_window in selected_windows:
        mobility_bundle = load_window_bundle(
            root_dir=ROOT_DIR,
            mobility_source=args.mobility_source,
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root=args.lust_scenario_root,
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout=str(eval_window.get("recommended_rsu_layout", args.rsu_layout)),
            frame_offset=int(eval_window["frame_offset"]),
            window_length=int(eval_window["window_length"]),
            random_seed=args.random_seed,
        )
        mobility_bundle.rsu_metadata["window_class"] = eval_window.get("window_class", "unknown")
        for workflow_state in selected_workflows:
            for agent_name in compare_agents:
                summary = run_real_episode_with_policy_diagnostics(
                    agent_name=agent_name,
                    checkpoint_map=checkpoint_map,
                    workflow_state=workflow_state,
                    workflow_source_path=args.workflow_csv_path,
                    mobility_bundle=mobility_bundle,
                    seed=args.random_seed,
                    max_steps=args.max_steps,
                    mobility_source=args.mobility_source,
                    primary_vehicle_selection=args.primary_vehicle_selection,
                    reward_positive_offset=args.reward_positive_offset,
                    predictor_runtime_kwargs=build_predictor_runtime_kwargs(args, random_seed=args.random_seed),
                    run_metadata={
                        "script": "scripts/train_sa_ghmappo_real_sample.py",
                        "mode": protocol_name,
                        "mainline": "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba",
                        "window_class": eval_window.get("window_class", "unknown"),
                        "config_profile": args.profile,
                        "primary_vehicle_selection": args.primary_vehicle_selection,
                        "reward_positive_offset": args.reward_positive_offset,
                        "policy_evaluation_mode": policy_evaluation_mode,
                        "experiment_run_type": build_run_scale_info(args.profile, args.episodes, max(1, len(selected_windows))).get("experiment_run_type"),
                    },
                    policy_evaluation_mode=policy_evaluation_mode,
                )
                metrics = summary["system_metrics"]
                validation_summary = summary["prefetch_validation_summary"]
                handoff_summary = summary["handoff_summary"]
                policy_summary = dict(summary.get("policy_diagnostics_summary", default_policy_diagnostics()))
                policy_step_diagnostics_by_agent.setdefault(agent_name, []).extend(list(summary.get("policy_diagnostics_trace", [])))
                policy_action_signature_rows_by_agent.setdefault(agent_name, []).append(
                    {
                        "workflow_id": workflow_state.workflow_id,
                        "window_id": eval_window.get("window_id"),
                        **dict(summary.get("policy_action_signature", {})),
                    }
                )
                for sample in list(summary.get("policy_alignment_samples", [])):
                    agent_samples = policy_alignment_samples_by_agent.setdefault(agent_name, [])
                    if len(agent_samples) >= 5:
                        break
                    agent_samples.append(
                        {
                            "workflow_id": workflow_state.workflow_id,
                            "window_id": eval_window.get("window_id"),
                            "window_class": eval_window.get("window_class", "unknown"),
                            **dict(sample),
                        }
                    )
                rows.append(
                    {
                        "agent_name": agent_name,
                        "policy_evaluation_mode": policy_evaluation_mode,
                        "workflow_id": workflow_state.workflow_id,
                        "window_id": eval_window.get("window_id"),
                        "window_class": eval_window.get("window_class", "unknown"),
                        "total_reward": summary["reward_breakdown"]["total"]["sum"],
                        "workflow_continuity_rate": metrics["workflow_continuity_rate"],
                        "handoff_failure_rate": metrics["handoff_failure_rate"],
                        "handoff_ready_ratio": metrics["handoff_ready_ratio"],
                        "adapter_warm_hit_ratio": metrics["adapter_warm_hit_ratio"],
                        "backhaul_traffic_cost": metrics["backhaul_traffic_cost"],
                        "predictive_prefetch_precision": metrics["predictive_prefetch_precision"],
                        "validated_predictive_prefetch_count": validation_summary["validated_predictive_prefetch_count"],
                        "migration_during_handoff_count": handoff_summary["migration_during_handoff_count"],
                        "handoff_ready_count": handoff_summary["handoff_ready_count"],
                        **extract_reward_breakdown_means(summary),
                        **extract_mechanism_diagnostics(summary),
                        **policy_summary,
                    }
                )
    aggregate_by_agent: dict[str, dict[str, float]] = {}
    reward_breakdown_by_agent: dict[str, dict[str, float]] = {}
    mechanism_diagnostics_by_agent: dict[str, dict[str, float]] = {}
    policy_diagnostics_by_agent: dict[str, dict[str, float]] = {}
    policy_action_signatures_by_agent: dict[str, dict[str, Any]] = {}
    for agent_name in compare_agents:
        agent_rows = [row for row in rows if row["agent_name"] == agent_name]
        aggregate_by_agent[agent_name] = aggregate_metrics(agent_rows)
        reward_breakdown_by_agent[agent_name] = aggregate_reward_breakdown(agent_rows)
        mechanism_diagnostics_by_agent[agent_name] = {
            field_name: round(fmean(float(row.get(field_name, 0.0)) for row in agent_rows), 6) if agent_rows else 0.0
            for field_name in MECHANISM_DIAGNOSTIC_FIELDS
        }
        policy_diagnostics_by_agent[agent_name] = aggregate_policy_diagnostics(policy_step_diagnostics_by_agent.get(agent_name, []))
        policy_action_signatures_by_agent[agent_name] = aggregate_policy_action_signatures(
            policy_action_signature_rows_by_agent.get(agent_name, [])
        )
    return {
        "protocol_name": protocol_name,
        "deterministic_eval": True,
        "policy_evaluation_mode": policy_evaluation_mode,
        "rows": rows,
        "aggregate_by_agent": aggregate_by_agent,
        "aggregate_reward_breakdown_by_agent": reward_breakdown_by_agent,
        "aggregate_mechanism_diagnostics_by_agent": mechanism_diagnostics_by_agent,
        "aggregate_policy_diagnostics_by_agent": policy_diagnostics_by_agent,
        "policy_action_signatures_by_agent": policy_action_signatures_by_agent,
        "policy_alignment_samples_by_agent": policy_alignment_samples_by_agent,
        "eval_windows": selected_windows,
        "eval_window_ids": [item.get("window_id") for item in selected_windows],
        "workflow_ids": [workflow_state.workflow_id for workflow_state in selected_workflows],
        "compare_agents": compare_agents,
    }



def run_update_eval(
    *,
    current_agent_name: str,
    checkpoint_path: Path,
    workflow_states: list[Any],
    eval_windows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    safety_projected_eval = evaluate_checkpoint_protocol(
        current_agent_name=current_agent_name,
        checkpoint_path=checkpoint_path,
        workflow_states=workflow_states,
        eval_windows=eval_windows,
        args=args,
        include_reference_agents=True,
        max_workflows=args.update_eval_max_workflows,
        max_windows=args.update_eval_max_windows,
        protocol_name="update_eval",
        policy_evaluation_mode="safety_projected",
    )
    safety_projected_eval["raw_policy_eval"] = evaluate_checkpoint_protocol(
        current_agent_name=current_agent_name,
        checkpoint_path=checkpoint_path,
        workflow_states=workflow_states,
        eval_windows=eval_windows,
        args=args,
        include_reference_agents=False,
        max_workflows=args.update_eval_max_workflows,
        max_windows=args.update_eval_max_windows,
        protocol_name="update_eval_raw_policy",
        policy_evaluation_mode="raw_policy",
    )
    return safety_projected_eval



def metrics_match(expected: dict[str, float], actual: dict[str, float], tolerance: float = 1e-6) -> bool:
    keys = set(expected.keys()) | set(actual.keys())
    for key in keys:
        if abs(float(expected.get(key, 0.0)) - float(actual.get(key, 0.0))) > tolerance:
            return False
    return True


def build_policy_learning_gate(
    update_eval_history: list[dict[str, Any]],
    current_agent_name: str,
) -> dict[str, Any]:
    """Decide whether a checkpoint can be selected as a learned-policy result.

    A safety-projected score is not sufficient evidence that MAPPO learned a
    decision rule.  We require two or more checkpoints and inspect the fixed
    development protocol under the raw (masked but otherwise unprojected)
    policy.  If both actions and raw metrics are invariant, no checkpoint from
    the run can be promoted by the automatic selector.
    """
    raw_records: list[dict[str, Any]] = []
    for item in update_eval_history:
        raw_eval = dict(item.get("raw_policy_eval", {}) or {})
        signature = dict(raw_eval.get("policy_action_signatures_by_agent", {}).get(current_agent_name, {}) or {})
        digest = str(signature.get("digest", "") or "")
        if not digest:
            continue
        raw_records.append(
            {
                "update_index": int(item.get("update_index", 0) or 0),
                "digest": digest,
                "metrics": dict(raw_eval.get("aggregate_by_agent", {}).get(current_agent_name, {}) or {}),
            }
        )
    observed_count = len(raw_records)
    base_payload: dict[str, Any] = {
        "gate_name": "raw_policy_invariance_gate",
        "policy_evaluation_mode": "raw_policy",
        "required_checkpoint_count": 2,
        "observed_checkpoint_count": observed_count,
        "evaluated_update_indices": [item["update_index"] for item in raw_records],
        "selection_eligible": False,
        "policy_invariant": False,
    }
    if observed_count < 2:
        return {
            **base_payload,
            "status": "pending_insufficient_checkpoints",
            "reason": "requires_two_raw_policy_evaluations_before_checkpoint_selection",
            "first_signature_digest": raw_records[0]["digest"] if raw_records else "",
            "current_signature_digest": raw_records[-1]["digest"] if raw_records else "",
        }
    first_record = raw_records[0]
    current_record = raw_records[-1]
    signature_invariant = all(item["digest"] == first_record["digest"] for item in raw_records[1:])
    metrics_invariant = all(
        metrics_match(first_record["metrics"], item["metrics"])
        for item in raw_records[1:]
    )
    policy_invariant = bool(signature_invariant and metrics_invariant)
    if policy_invariant:
        return {
            **base_payload,
            "status": "blocked_policy_invariant",
            "reason": "raw_policy_actions_and_metrics_are_invariant_across_evaluated_checkpoints",
            "selection_eligible": False,
            "policy_invariant": True,
            "signature_invariant": True,
            "metrics_invariant": True,
            "first_signature_digest": first_record["digest"],
            "current_signature_digest": current_record["digest"],
        }
    return {
        **base_payload,
        "status": "passed_policy_changed",
        "reason": "raw_policy_signature_or_metrics_changed_across_evaluated_checkpoints",
        "selection_eligible": True,
        "policy_invariant": False,
        "signature_invariant": signature_invariant,
        "metrics_invariant": metrics_invariant,
        "first_signature_digest": first_record["digest"],
        "current_signature_digest": current_record["digest"],
    }


def resolve_post_training_audit_scope(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(getattr(args, "post_training_audit_mode", "full") or "full").strip().lower()
    if mode not in {"full", "compact"}:
        mode = "full"
    explicit_windows = getattr(args, "post_training_audit_max_windows", None)
    explicit_workflows = getattr(args, "post_training_audit_max_workflows", None)
    max_windows = int(explicit_windows) if explicit_windows is not None and int(explicit_windows) > 0 else None
    max_workflows = int(explicit_workflows) if explicit_workflows is not None and int(explicit_workflows) > 0 else None
    if mode == "compact":
        max_windows = max_windows or 1
        max_workflows = max_workflows or 1
    return {
        "mode": mode,
        "max_windows": max_windows,
        "max_workflows": max_workflows,
        "full_protocol": bool(mode == "full" and max_windows is None and max_workflows is None),
    }



def run_checkpoint_consistency_audit(
    *,
    current_agent_name: str,
    checkpoint_root: Path,
    workflow_states: list[Any],
    eval_windows: list[dict[str, Any]],
    args: argparse.Namespace,
    best_record: dict[str, Any],
) -> dict[str, Any]:
    audit_scope = resolve_post_training_audit_scope(args)
    selection_evaluation_mode = str(
        best_record.get("selection_protocol", {}).get("policy_evaluation_mode", "safety_projected")
    ).strip().lower()
    if selection_evaluation_mode not in {"raw_policy", "safety_projected"}:
        selection_evaluation_mode = "safety_projected"
    checkpoint_entries = [("latest", checkpoint_root / "latest.pt")]
    checkpoint_entries.extend([("warm_start", checkpoint_root / "warm_start.pt")])
    if bool(getattr(args, "audit_update_checkpoints", False)):
        checkpoint_entries.extend((path.stem, path) for path in sorted(checkpoint_root.glob("update_*.pt")))
    best_labels = [
        "best_by_reward",
        "best_by_continuity",
        "best_by_mechanism_balanced",
        "best_by_advantage_score",
        "best_by_mechanism_advantage_score",
        "best_by_round2_mechanism_score",
        "best_by_retained_mechanism_score",
        "best_by_reward_tiebreak_score",
    ]
    for best_label in best_labels:
        source_path_text = str(
            best_record.get(best_label, {}).get("source_checkpoint_path", "")
            or best_record.get(best_label, {}).get("path", "")
            or ""
        )
        if source_path_text:
            checkpoint_entries.append((f"source_{best_label}", Path(source_path_text)))
    checkpoint_entries.extend(
        [
            ("best_by_reward", checkpoint_root / "best_by_reward.pt"),
            ("best_by_continuity", checkpoint_root / "best_by_continuity.pt"),
            ("best_by_mechanism_balanced", checkpoint_root / "best_by_mechanism_balanced.pt"),
            ("best_by_advantage_score", checkpoint_root / "best_by_advantage_score.pt"),
            ("best_by_mechanism_advantage_score", checkpoint_root / "best_by_mechanism_advantage_score.pt"),
            ("best_by_round2_mechanism_score", checkpoint_root / "best_by_round2_mechanism_score.pt"),
            ("best_by_retained_mechanism_score", checkpoint_root / "best_by_retained_mechanism_score.pt"),
            ("best_by_reward_tiebreak_score", checkpoint_root / "best_by_reward_tiebreak_score.pt"),
        ]
    )
    deduped_entries: list[tuple[str, Path]] = []
    seen_checkpoint_paths: set[str] = set()
    for checkpoint_label, checkpoint_path in checkpoint_entries:
        key = str(checkpoint_path)
        if key in seen_checkpoint_paths:
            continue
        seen_checkpoint_paths.add(key)
        deduped_entries.append((checkpoint_label, checkpoint_path))
    checkpoint_entries = deduped_entries
    audited: list[dict[str, Any]] = []
    for checkpoint_label, checkpoint_path in checkpoint_entries:
        if not checkpoint_path.exists():
            audited.append(
                {
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_path": str(checkpoint_path),
                    "exists": False,
                    "loaded_metadata": {},
                    "checkpoint_fingerprint": build_checkpoint_fingerprint(checkpoint_path),
                    "protocol_eval_metrics": {},
                    "probe_eval_metrics": {},
                    "matches_best_by_reward_record": False,
                    "matches_best_by_continuity_record": False,
                    "matches_best_by_mechanism_balanced_record": False,
                    "matches_best_by_advantage_score_record": False,
                    "matches_best_by_mechanism_advantage_score_record": False,
                    "matches_best_by_round2_mechanism_score_record": False,
                    "matches_best_by_retained_mechanism_score_record": False,
                    "matches_best_by_reward_tiebreak_score_record": False,
                }
            )
            continue
        try:
            protocol_eval = evaluate_checkpoint_protocol(
                current_agent_name=current_agent_name,
                checkpoint_path=checkpoint_path,
                workflow_states=workflow_states,
                eval_windows=eval_windows,
                args=args,
                include_reference_agents=False,
                max_workflows=audit_scope["max_workflows"],
                max_windows=audit_scope["max_windows"],
                protocol_name="checkpoint_consistency_protocol_eval",
                policy_evaluation_mode=selection_evaluation_mode,
            )
            probe_eval = evaluate_checkpoint_protocol(
                current_agent_name=current_agent_name,
                checkpoint_path=checkpoint_path,
                workflow_states=workflow_states,
                eval_windows=eval_windows,
                args=args,
                include_reference_agents=False,
                max_workflows=1,
                max_windows=1,
                protocol_name="checkpoint_consistency_probe_eval",
                policy_evaluation_mode=selection_evaluation_mode,
            )
            loaded_metadata = load_checkpoint_metadata(str(checkpoint_path))
        except Exception as exc:  # Defensive audit path for partially written checkpoints.
            audited.append(
                {
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_path": str(checkpoint_path),
                    "exists": True,
                    "load_error": f"{type(exc).__name__}: {exc}",
                    "loaded_metadata": {},
                    "checkpoint_fingerprint": build_checkpoint_fingerprint(checkpoint_path),
                    "protocol_eval_metrics": {},
                    "probe_eval_metrics": {},
                    "matches_best_by_reward_record": False,
                    "matches_best_by_continuity_record": False,
                    "matches_best_by_mechanism_balanced_record": False,
                    "matches_best_by_advantage_score_record": False,
                    "matches_best_by_mechanism_advantage_score_record": False,
                    "matches_best_by_round2_mechanism_score_record": False,
                    "matches_best_by_retained_mechanism_score_record": False,
                    "matches_best_by_reward_tiebreak_score_record": False,
                }
            )
            continue
        agent_metrics = dict(protocol_eval["aggregate_by_agent"].get(current_agent_name, {}))
        agent_policy_diag = dict(protocol_eval.get("aggregate_policy_diagnostics_by_agent", {}).get(current_agent_name, {}))
        reward_record = dict(best_record.get("best_by_reward", {}).get("metrics", {}))
        continuity_record = dict(best_record.get("best_by_continuity", {}).get("metrics", {}))
        mechanism_balanced_record = dict(best_record.get("best_by_mechanism_balanced", {}).get("metrics", {}))
        advantage_record = dict(best_record.get("best_by_advantage_score", {}).get("metrics", {}))
        mechanism_advantage_record = dict(best_record.get("best_by_mechanism_advantage_score", {}).get("metrics", {}))
        round2_record = dict(best_record.get("best_by_round2_mechanism_score", {}).get("metrics", {}))
        retained_record = dict(best_record.get("best_by_retained_mechanism_score", {}).get("metrics", {}))
        reward_tiebreak_record = dict(best_record.get("best_by_reward_tiebreak_score", {}).get("metrics", {}))
        protocol_eval_rows = list(protocol_eval.get("rows", []))
        mechanism_advantage_priority = mechanism_advantage_score_priority_tuple(
            agent_metrics,
            rows=protocol_eval_rows,
            current_agent_name=current_agent_name,
        )
        update_count = int(loaded_metadata.get("update_count", 0) or 0)
        retained_priority = retained_mechanism_score_priority_tuple(
            agent_metrics,
            agent_policy_diag,
            update_index=update_count,
        )
        reward_tiebreak_priority = reward_tiebreak_score_priority_tuple(
            agent_metrics,
            rows=protocol_eval_rows,
            current_agent_name=current_agent_name,
        )
        audited.append(
            {
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "exists": True,
                "loaded_metadata": loaded_metadata,
                "checkpoint_fingerprint": build_checkpoint_fingerprint(checkpoint_path),
                "protocol_eval_metrics": agent_metrics,
                "protocol_eval_rows": protocol_eval_rows,
                "protocol_eval_policy_diagnostics": agent_policy_diag,
                "probe_eval_window_id": probe_eval.get("eval_window_ids", [""])[0] if probe_eval.get("eval_window_ids") else "",
                "probe_eval_workflow_id": probe_eval.get("workflow_ids", [""])[0] if probe_eval.get("workflow_ids") else "",
                "probe_eval_metrics": dict(probe_eval["aggregate_by_agent"].get(current_agent_name, {})),
                "mechanism_advantage_score_breakdown": compute_mechanism_advantage_checkpoint_score(
                    agent_metrics,
                    rows=protocol_eval_rows,
                    current_agent_name=current_agent_name,
                ),
                "mechanism_advantage_priority_tuple": list(mechanism_advantage_priority),
                "retained_mechanism_score_breakdown": compute_retained_mechanism_checkpoint_score(
                    agent_metrics,
                    agent_policy_diag,
                    update_index=update_count,
                ),
                "retained_mechanism_priority_tuple": list(retained_priority),
                "reward_tiebreak_score_breakdown": compute_reward_tiebreak_checkpoint_score(
                    agent_metrics,
                    rows=protocol_eval_rows,
                    current_agent_name=current_agent_name,
                ),
                "reward_tiebreak_priority_tuple": list(reward_tiebreak_priority),
                "matches_best_by_reward_record": metrics_match(reward_record, agent_metrics),
                "matches_best_by_continuity_record": metrics_match(continuity_record, agent_metrics),
                "matches_best_by_mechanism_balanced_record": metrics_match(mechanism_balanced_record, agent_metrics),
                "matches_best_by_advantage_score_record": metrics_match(advantage_record, agent_metrics),
                "matches_best_by_mechanism_advantage_score_record": metrics_match(mechanism_advantage_record, agent_metrics),
                "matches_best_by_round2_mechanism_score_record": metrics_match(round2_record, agent_metrics),
                "matches_best_by_retained_mechanism_score_record": metrics_match(retained_record, agent_metrics),
                "matches_best_by_reward_tiebreak_score_record": metrics_match(reward_tiebreak_record, agent_metrics),
            }
        )
    source_candidates = [
        item
        for item in audited
        if item.get("exists")
        and (
            item.get("checkpoint_label", "").startswith("update_")
            or item.get("checkpoint_label", "").startswith("source_")
            or item.get("checkpoint_label") == "warm_start"
        )
    ]
    if not source_candidates:
        source_candidates = [item for item in audited if item.get("exists") and item.get("checkpoint_label") == "latest"]
    expected_reward_entry = max(source_candidates, key=lambda item: float(item.get("protocol_eval_metrics", {}).get("total_reward", float("-inf"))), default=None)
    expected_continuity_entry = max(
        source_candidates,
        key=lambda item: continuity_priority_tuple(item.get("protocol_eval_metrics", {})),
        default=None,
    )
    expected_mechanism_balanced_entry = max(
        source_candidates,
        key=lambda item: (
            mechanism_balanced_priority_tuple(item.get("protocol_eval_metrics", {})),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    expected_advantage_entry = max(
        source_candidates,
        key=lambda item: (
            advantage_score_priority_tuple(item.get("protocol_eval_metrics", {})),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    expected_mechanism_advantage_entry = max(
        source_candidates,
        key=lambda item: (
            tuple(item.get("mechanism_advantage_priority_tuple", [])),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    expected_round2_entry = max(
        source_candidates,
        key=lambda item: (
            round2_mechanism_score_priority_tuple(
                item.get("protocol_eval_metrics", {}),
                item.get("protocol_eval_policy_diagnostics", {}),
            ),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    expected_retained_entry = max(
        source_candidates,
        key=lambda item: (
            tuple(item.get("retained_mechanism_priority_tuple", [])),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    expected_reward_tiebreak_entry = max(
        source_candidates,
        key=lambda item: (
            tuple(item.get("reward_tiebreak_priority_tuple", [])),
            int(item.get("loaded_metadata", {}).get("update_count", 0) or 0),
        ),
        default=None,
    )
    recorded_reward_path = str(best_record.get("best_by_reward", {}).get("path", ""))
    recorded_continuity_path = str(best_record.get("best_by_continuity", {}).get("path", ""))
    recorded_mechanism_balanced_path = str(best_record.get("best_by_mechanism_balanced", {}).get("path", ""))
    recorded_advantage_path = str(best_record.get("best_by_advantage_score", {}).get("path", ""))
    recorded_mechanism_advantage_path = str(best_record.get("best_by_mechanism_advantage_score", {}).get("path", ""))
    recorded_round2_path = str(best_record.get("best_by_round2_mechanism_score", {}).get("path", ""))
    recorded_retained_path = str(best_record.get("best_by_retained_mechanism_score", {}).get("path", ""))
    recorded_reward_tiebreak_path = str(best_record.get("best_by_reward_tiebreak_score", {}).get("path", ""))
    recorded_reward_source_path = str(best_record.get("best_by_reward", {}).get("source_checkpoint_path", recorded_reward_path))
    recorded_continuity_source_path = str(best_record.get("best_by_continuity", {}).get("source_checkpoint_path", recorded_continuity_path))
    recorded_mechanism_balanced_source_path = str(best_record.get("best_by_mechanism_balanced", {}).get("source_checkpoint_path", recorded_mechanism_balanced_path))
    recorded_advantage_source_path = str(best_record.get("best_by_advantage_score", {}).get("source_checkpoint_path", recorded_advantage_path))
    recorded_mechanism_advantage_source_path = str(best_record.get("best_by_mechanism_advantage_score", {}).get("source_checkpoint_path", recorded_mechanism_advantage_path))
    recorded_round2_source_path = str(best_record.get("best_by_round2_mechanism_score", {}).get("source_checkpoint_path", recorded_round2_path))
    recorded_retained_source_path = str(best_record.get("best_by_retained_mechanism_score", {}).get("source_checkpoint_path", recorded_retained_path))
    recorded_reward_tiebreak_source_path = str(best_record.get("best_by_reward_tiebreak_score", {}).get("source_checkpoint_path", recorded_reward_tiebreak_path))
    return {
        "agent_name": current_agent_name,
        "config_profile": args.profile,
        "audit_scope": {
            "mode": audit_scope["mode"],
            "full_protocol": audit_scope["full_protocol"],
            "repair_best_record_eligible": bool(audit_scope["full_protocol"]),
            "max_windows": audit_scope["max_windows"],
            "max_workflows": audit_scope["max_workflows"],
            "scope_note": (
                "compact audit is a post-training liveness/provenance probe only; "
                "it must not be used as paper-grade checkpoint consistency evidence"
                if audit_scope["mode"] == "compact"
                else "full selected-window checkpoint consistency audit"
            ),
        },
        "run_scale": build_run_scale_info(args.profile, args.episodes, max(1, len(source_candidates))),
        "selection_protocol": {
            "protocol_name": "all_selected_windows_x_all_selected_workflows_deterministic",
            "policy_evaluation_mode": selection_evaluation_mode,
            "eval_window_ids": [
                item.get("window_id")
                for item in list(eval_windows[: audit_scope["max_windows"] or len(eval_windows)])
            ],
            "workflow_ids": [
                workflow_state.workflow_id
                for workflow_state in list(workflow_states[: audit_scope["max_workflows"] or len(workflow_states)])
            ],
        },
        "audited_checkpoints": audited,
        "recorded_best_paths": {
            "best_by_reward": recorded_reward_path,
            "best_by_continuity": recorded_continuity_path,
            "best_by_mechanism_balanced": recorded_mechanism_balanced_path,
            "best_by_advantage_score": recorded_advantage_path,
            "best_by_mechanism_advantage_score": recorded_mechanism_advantage_path,
            "best_by_round2_mechanism_score": recorded_round2_path,
            "best_by_retained_mechanism_score": recorded_retained_path,
            "best_by_reward_tiebreak_score": recorded_reward_tiebreak_path,
        },
        "recorded_best_sources": {
            "best_by_reward": recorded_reward_source_path,
            "best_by_continuity": recorded_continuity_source_path,
            "best_by_mechanism_balanced": recorded_mechanism_balanced_source_path,
            "best_by_advantage_score": recorded_advantage_source_path,
            "best_by_mechanism_advantage_score": recorded_mechanism_advantage_source_path,
            "best_by_round2_mechanism_score": recorded_round2_source_path,
            "best_by_retained_mechanism_score": recorded_retained_source_path,
            "best_by_reward_tiebreak_score": recorded_reward_tiebreak_source_path,
        },
        "expected_best_sources": {
            "best_by_reward": expected_reward_entry.get("checkpoint_path", "") if expected_reward_entry else "",
            "best_by_continuity": expected_continuity_entry.get("checkpoint_path", "") if expected_continuity_entry else "",
            "best_by_mechanism_balanced": expected_mechanism_balanced_entry.get("checkpoint_path", "") if expected_mechanism_balanced_entry else "",
            "best_by_advantage_score": expected_advantage_entry.get("checkpoint_path", "") if expected_advantage_entry else "",
            "best_by_mechanism_advantage_score": expected_mechanism_advantage_entry.get("checkpoint_path", "") if expected_mechanism_advantage_entry else "",
            "best_by_round2_mechanism_score": expected_round2_entry.get("checkpoint_path", "") if expected_round2_entry else "",
            "best_by_retained_mechanism_score": expected_retained_entry.get("checkpoint_path", "") if expected_retained_entry else "",
            "best_by_reward_tiebreak_score": expected_reward_tiebreak_entry.get("checkpoint_path", "") if expected_reward_tiebreak_entry else "",
        },
        "best_selection_mismatch": {
            "best_by_reward": bool(expected_reward_entry and not metrics_match(dict(best_record.get("best_by_reward", {}).get("metrics", {})), dict(expected_reward_entry.get("protocol_eval_metrics", {})))),
            "best_by_continuity": bool(expected_continuity_entry and not metrics_match(dict(best_record.get("best_by_continuity", {}).get("metrics", {})), dict(expected_continuity_entry.get("protocol_eval_metrics", {})))),
            "best_by_mechanism_balanced": bool(expected_mechanism_balanced_entry and not metrics_match(dict(best_record.get("best_by_mechanism_balanced", {}).get("metrics", {})), dict(expected_mechanism_balanced_entry.get("protocol_eval_metrics", {})))),
            "best_by_advantage_score": bool(expected_advantage_entry and not metrics_match(dict(best_record.get("best_by_advantage_score", {}).get("metrics", {})), dict(expected_advantage_entry.get("protocol_eval_metrics", {})))),
            "best_by_mechanism_advantage_score": bool(expected_mechanism_advantage_entry and not metrics_match(dict(best_record.get("best_by_mechanism_advantage_score", {}).get("metrics", {})), dict(expected_mechanism_advantage_entry.get("protocol_eval_metrics", {})))),
            "best_by_round2_mechanism_score": bool(expected_round2_entry and not metrics_match(dict(best_record.get("best_by_round2_mechanism_score", {}).get("metrics", {})), dict(expected_round2_entry.get("protocol_eval_metrics", {})))),
            "best_by_retained_mechanism_score": bool(expected_retained_entry and not metrics_match(dict(best_record.get("best_by_retained_mechanism_score", {}).get("metrics", {})), dict(expected_retained_entry.get("protocol_eval_metrics", {})))),
            "best_by_reward_tiebreak_score": bool(expected_reward_tiebreak_entry and not metrics_match(dict(best_record.get("best_by_reward_tiebreak_score", {}).get("metrics", {})), dict(expected_reward_tiebreak_entry.get("protocol_eval_metrics", {})))),
        },
    }


def repair_best_checkpoint_record_from_audit(
    *,
    checkpoint_root: Path,
    best_record: dict[str, Any],
    audit_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    changed = False
    audited_by_path = {item.get("checkpoint_path", ""): item for item in audit_payload.get("audited_checkpoints", []) if item.get("exists")}
    for best_label in [
        "best_by_reward",
        "best_by_continuity",
        "best_by_mechanism_balanced",
        "best_by_advantage_score",
        "best_by_mechanism_advantage_score",
        "best_by_round2_mechanism_score",
        "best_by_retained_mechanism_score",
        "best_by_reward_tiebreak_score",
    ]:
        if not audit_payload.get("best_selection_mismatch", {}).get(best_label, False):
            continue
        expected_source = str(audit_payload.get("expected_best_sources", {}).get(best_label, ""))
        if not expected_source:
            continue
        target_name = f"{best_label}.pt"
        target_path = checkpoint_root / target_name
        shutil.copy2(expected_source, target_path)
        source_entry = audited_by_path.get(expected_source, {})
        current_entry = dict(best_record.get(best_label, {}))
        current_entry["path"] = str(target_path)
        current_entry["source_checkpoint_path"] = expected_source
        current_entry["metrics"] = dict(source_entry.get("protocol_eval_metrics", current_entry.get("metrics", {})))
        loaded_metadata = dict(source_entry.get("loaded_metadata", {}))
        if loaded_metadata:
            current_entry["update_index"] = int(loaded_metadata.get("update_count", current_entry.get("update_index", 0)) or 0)
        if best_label == "best_by_reward":
            current_entry["score"] = float(current_entry.get("metrics", {}).get("total_reward", current_entry.get("score", 0.0)))
        elif best_label == "best_by_continuity":
            current_entry["priority_tuple"] = list(continuity_priority_tuple(current_entry.get("metrics", {})))
        elif best_label == "best_by_mechanism_balanced":
            current_entry["priority_tuple"] = list(mechanism_balanced_priority_tuple(current_entry.get("metrics", {})))
        elif best_label == "best_by_advantage_score":
            current_entry["priority_tuple"] = list(advantage_score_priority_tuple(current_entry.get("metrics", {})))
            current_entry["score_breakdown"] = compute_advantage_checkpoint_score(current_entry.get("metrics", {}))
        elif best_label == "best_by_mechanism_advantage_score":
            current_entry["priority_tuple"] = list(
                mechanism_advantage_score_priority_tuple(current_entry.get("metrics", {}))
            )
            current_entry["score_breakdown"] = compute_mechanism_advantage_checkpoint_score(
                current_entry.get("metrics", {})
            )
        elif best_label == "best_by_round2_mechanism_score":
            policy_diagnostics = dict(audited_by_path.get(expected_source, {}).get("protocol_eval_policy_diagnostics", {}))
            current_entry["priority_tuple"] = list(
                round2_mechanism_score_priority_tuple(current_entry.get("metrics", {}), policy_diagnostics)
            )
            current_entry["score_breakdown"] = compute_round2_mechanism_checkpoint_score(
                current_entry.get("metrics", {}),
                policy_diagnostics,
            )
        elif best_label == "best_by_retained_mechanism_score":
            policy_diagnostics = dict(audited_by_path.get(expected_source, {}).get("protocol_eval_policy_diagnostics", {}))
            update_count = int(loaded_metadata.get("update_count", current_entry.get("update_index", 0)) or 0)
            current_entry["priority_tuple"] = list(
                retained_mechanism_score_priority_tuple(
                    current_entry.get("metrics", {}),
                    policy_diagnostics,
                    update_index=update_count,
                )
            )
            current_entry["score_breakdown"] = compute_retained_mechanism_checkpoint_score(
                current_entry.get("metrics", {}),
                policy_diagnostics,
                update_index=update_count,
            )
        else:
            current_entry["priority_tuple"] = list(
                reward_tiebreak_score_priority_tuple(
                    current_entry.get("metrics", {}),
                    rows=list(audited_by_path.get(expected_source, {}).get("protocol_eval_rows", [])),
                )
            )
            current_entry["score_breakdown"] = compute_reward_tiebreak_checkpoint_score(
                current_entry.get("metrics", {}),
                rows=list(audited_by_path.get(expected_source, {}).get("protocol_eval_rows", [])),
            )
        best_record[best_label] = current_entry
        changed = True
    return best_record, changed


def build_trend_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"first": 0.0, "last": 0.0, "delta": 0.0, "min": 0.0, "max": 0.0}
    return {
        "first": round(values[0], 6),
        "last": round(values[-1], 6),
        "delta": round(values[-1] - values[0], 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def build_training_audit(update_logs: list[dict[str, Any]], update_eval_history: list[dict[str, Any]], current_agent_name: str) -> dict[str, Any]:
    policy_entropy = [float(item.get("policy_entropy", 0.0) or 0.0) for item in update_logs]
    approx_kl = [float(item.get("approx_kl", 0.0) or 0.0) for item in update_logs]
    clip_fraction = [float(item.get("clip_fraction", 0.0) or 0.0) for item in update_logs]
    explained_variance = [float(item.get("explained_variance", 0.0) or 0.0) for item in update_logs]
    current_eval_reward = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("total_reward", 0.0)) for item in update_eval_history]
    current_eval_continuity = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("workflow_continuity_rate", 0.0)) for item in update_eval_history]
    current_eval_prefetch_request_rate = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("prefetch_request_rate", 0.0)) for item in update_eval_history]
    current_eval_prefetch_validated_hit_rate = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("prefetch_validated_hit_rate", 0.0)) for item in update_eval_history]
    current_eval_migration_prepare_rate = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("migration_prepare_rate", 0.0)) for item in update_eval_history]
    current_eval_handoff_ready_rate = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("handoff_ready_rate", 0.0)) for item in update_eval_history]
    current_eval_mechanism_realization_rate = [float(item.get("aggregate_by_agent", {}).get(current_agent_name, {}).get("mechanism_realization_rate", 0.0)) for item in update_eval_history]
    current_eval_policy_diagnostics = [
        dict(item.get("aggregate_policy_diagnostics_by_agent", {}).get(current_agent_name, default_policy_diagnostics()))
        for item in update_eval_history
    ]
    return {
        "update_count": len(update_logs),
        "policy_learning_gate": build_policy_learning_gate(
            update_eval_history=update_eval_history,
            current_agent_name=current_agent_name,
        ),
        "policy_entropy_trend": [round(item, 6) for item in policy_entropy],
        "approx_kl_trend": [round(item, 6) for item in approx_kl],
        "clip_fraction_trend": [round(item, 6) for item in clip_fraction],
        "explained_variance_trend": [round(item, 6) for item in explained_variance],
        "current_eval_reward_trend": [round(item, 6) for item in current_eval_reward],
        "current_eval_continuity_trend": [round(item, 6) for item in current_eval_continuity],
        "current_eval_prefetch_request_rate_trend": [round(item, 6) for item in current_eval_prefetch_request_rate],
        "current_eval_prefetch_validated_hit_rate_trend": [round(item, 6) for item in current_eval_prefetch_validated_hit_rate],
        "current_eval_migration_prepare_rate_trend": [round(item, 6) for item in current_eval_migration_prepare_rate],
        "current_eval_handoff_ready_rate_trend": [round(item, 6) for item in current_eval_handoff_ready_rate],
        "current_eval_mechanism_realization_rate_trend": [round(item, 6) for item in current_eval_mechanism_realization_rate],
        "current_eval_policy_diagnostics_trend": {
            field_name: [round(float(item.get(field_name, 0.0)), 6) for item in current_eval_policy_diagnostics]
            for field_name in POLICY_DIAGNOSTIC_FIELDS
        },
        "trend_summary": {
            "policy_entropy": build_trend_summary(policy_entropy),
            "approx_kl": build_trend_summary(approx_kl),
            "clip_fraction": build_trend_summary(clip_fraction),
            "explained_variance": build_trend_summary(explained_variance),
            "current_eval_reward": build_trend_summary(current_eval_reward),
            "current_eval_continuity": build_trend_summary(current_eval_continuity),
            "current_eval_prefetch_request_rate": build_trend_summary(current_eval_prefetch_request_rate),
            "current_eval_prefetch_validated_hit_rate": build_trend_summary(current_eval_prefetch_validated_hit_rate),
            "current_eval_migration_prepare_rate": build_trend_summary(current_eval_migration_prepare_rate),
            "current_eval_handoff_ready_rate": build_trend_summary(current_eval_handoff_ready_rate),
            "current_eval_mechanism_realization_rate": build_trend_summary(current_eval_mechanism_realization_rate),
            "current_eval_policy_diagnostics": {
                field_name: build_trend_summary([float(item.get(field_name, 0.0)) for item in current_eval_policy_diagnostics])
                for field_name in POLICY_DIAGNOSTIC_FIELDS
            },
        },
    }


def continuity_priority_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("workflow_continuity_rate", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("total_reward", 0.0)),
    )


def mechanism_balanced_priority_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(metrics.get("workflow_continuity_rate", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        float(metrics.get("validated_predictive_prefetch_count", 0.0)),
        float(metrics.get("handoff_ready_count", 0.0)),
        float(metrics.get("migration_during_handoff_count", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        float(metrics.get("total_reward", 0.0)),
    )


def compute_advantage_checkpoint_score(metrics: dict[str, float]) -> dict[str, float]:
    reward = float(metrics.get("total_reward", 0.0))
    continuity = float(metrics.get("workflow_continuity_rate", 0.0))
    handoff_ready = float(metrics.get("handoff_ready_ratio", 0.0))
    mechanism = float(metrics.get("mechanism_realization_rate", 0.0))
    handoff_failure = float(metrics.get("handoff_failure_rate", 0.0))
    backhaul_cost = float(metrics.get("backhaul_traffic_cost", 0.0))
    migration_overhead = float(metrics.get("adapter_state_migration_overhead", 0.0))
    score = (
        reward
        + 40.0 * continuity
        + 15.0 * handoff_ready
        + 20.0 * mechanism
        - 30.0 * handoff_failure
        - 0.1 * backhaul_cost
        - 0.2 * migration_overhead
    )
    return {
        "score": round(score, 6),
        "total_reward": round(reward, 6),
        "workflow_continuity_rate": round(continuity, 6),
        "handoff_ready_ratio": round(handoff_ready, 6),
        "mechanism_realization_rate": round(mechanism, 6),
        "handoff_failure_rate": round(handoff_failure, 6),
        "backhaul_traffic_cost": round(backhaul_cost, 6),
        "adapter_state_migration_overhead": round(migration_overhead, 6),
        "formula": (
            "total_reward + 40*workflow_continuity_rate + 15*handoff_ready_ratio "
            "+ 20*mechanism_realization_rate - 30*handoff_failure_rate "
            "- 0.1*backhaul_traffic_cost - 0.2*adapter_state_migration_overhead"
        ),
    }


MECHANISM_ADVANTAGE_SCORE_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "handoff_ready_ratio",
    "mechanism_realization_rate",
    "adapter_state_migration_overhead",
]


def _float_metric(metrics: dict[str, Any], metric_name: str, missing_metrics: list[str]) -> float:
    if metric_name not in metrics:
        missing_metrics.append(metric_name)
        return 0.0
    value = metrics.get(metric_name, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        missing_metrics.append(metric_name)
        return 0.0


def _population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = fmean(values)
    return float(math.sqrt(fmean([(value - mean_value) ** 2 for value in values])))


def build_mechanism_advantage_stability_summary(
    *,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    agent_rows = [
        row
        for row in (rows or [])
        if str(row.get("agent_name", current_agent_name)) == current_agent_name
    ]
    if agent_rows:
        continuity_values = [float(row.get("workflow_continuity_rate", 0.0) or 0.0) for row in agent_rows]
        failure_values = [float(row.get("handoff_failure_rate", 0.0) or 0.0) for row in agent_rows]
        ready_values = [float(row.get("handoff_ready_ratio", 0.0) or 0.0) for row in agent_rows]
        mechanism_values = [float(row.get("mechanism_realization_rate", 0.0) or 0.0) for row in agent_rows]
        reward_values = [float(row.get("total_reward", 0.0) or 0.0) for row in agent_rows]
        source = "window_rows"
    else:
        continuity_values = [float(metrics.get("workflow_continuity_rate", 0.0) or 0.0)]
        failure_values = [float(metrics.get("handoff_failure_rate", 0.0) or 0.0)]
        ready_values = [float(metrics.get("handoff_ready_ratio", 0.0) or 0.0)]
        mechanism_values = [float(metrics.get("mechanism_realization_rate", 0.0) or 0.0)]
        reward_values = [float(metrics.get("total_reward", 0.0) or 0.0)]
        source = "aggregate_fallback"
    continuity_floor = min(continuity_values) if continuity_values else 0.0
    failure_ceiling = max(failure_values) if failure_values else 0.0
    ready_floor = min(ready_values) if ready_values else 0.0
    mechanism_floor = min(mechanism_values) if mechanism_values else 0.0
    reward_std = _population_std(reward_values)
    penalty = (
        35.0 * max(0.0, 0.98 - continuity_floor)
        + 25.0 * failure_ceiling
        + 0.03 * reward_std
        + 10.0 * max(0.0, 0.15 - ready_floor)
        + 10.0 * max(0.0, 0.25 - mechanism_floor)
    )
    return {
        "source": source,
        "row_count": len(agent_rows),
        "continuity_floor": round(continuity_floor, 6),
        "handoff_failure_ceiling": round(failure_ceiling, 6),
        "handoff_ready_floor": round(ready_floor, 6),
        "mechanism_realization_floor": round(mechanism_floor, 6),
        "reward_std": round(reward_std, 6),
        "stability_penalty": round(penalty, 6),
        "penalty_formula": (
            "35*max(0,0.98-continuity_floor) + 25*failure_ceiling + "
            "0.03*reward_std + 10*max(0,0.15-ready_floor) + "
            "10*max(0,0.25-mechanism_floor)"
        ),
    }


def compute_mechanism_advantage_checkpoint_score(
    metrics: dict[str, Any],
    *,
    reference_metrics: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    missing_metrics: list[str] = []
    reference_missing_metrics: list[str] = []
    reference_metrics = reference_metrics or {}
    reference_available = bool(reference_metrics)
    reward = _float_metric(metrics, "total_reward", missing_metrics)
    continuity = _float_metric(metrics, "workflow_continuity_rate", missing_metrics)
    handoff_failure = _float_metric(metrics, "handoff_failure_rate", missing_metrics)
    backhaul_cost = _float_metric(metrics, "backhaul_traffic_cost", missing_metrics)
    handoff_ready = _float_metric(metrics, "handoff_ready_ratio", missing_metrics)
    mechanism = _float_metric(metrics, "mechanism_realization_rate", missing_metrics)
    migration_overhead = _float_metric(metrics, "adapter_state_migration_overhead", missing_metrics)
    prefetch_hit_rate = float(metrics.get("prefetch_validated_hit_rate", 0.0) or 0.0)
    migration_prepare_rate = float(metrics.get("migration_prepare_rate", 0.0) or 0.0)
    if reference_available:
        reward_term = reward - _float_metric(reference_metrics, "total_reward", reference_missing_metrics)
        continuity_term = continuity - _float_metric(reference_metrics, "workflow_continuity_rate", reference_missing_metrics)
        failure_term = _float_metric(reference_metrics, "handoff_failure_rate", reference_missing_metrics) - handoff_failure
        backhaul_term = _float_metric(reference_metrics, "backhaul_traffic_cost", reference_missing_metrics) - backhaul_cost
        ready_term = handoff_ready - _float_metric(reference_metrics, "handoff_ready_ratio", reference_missing_metrics)
        mechanism_term = mechanism - _float_metric(reference_metrics, "mechanism_realization_rate", reference_missing_metrics)
        score_mode = "advantage_against_reference"
    else:
        reward_term = reward
        continuity_term = continuity
        failure_term = -handoff_failure
        backhaul_term = -backhaul_cost
        ready_term = handoff_ready
        mechanism_term = mechanism
        score_mode = "self_metrics_fallback"
    stability_summary = build_mechanism_advantage_stability_summary(
        metrics=metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    stability_penalty = float(stability_summary["stability_penalty"])
    score = (
        reward_term
        + 45.0 * continuity_term
        + 35.0 * failure_term
        + 0.08 * backhaul_term
        + 30.0 * ready_term
        + 28.0 * mechanism_term
        + 8.0 * prefetch_hit_rate
        + 5.0 * migration_prepare_rate
        - 0.2 * migration_overhead
        - stability_penalty
    )
    return {
        "score": round(score, 6),
        "score_mode": score_mode,
        "reference_available": reference_available,
        "reward_term": round(reward_term, 6),
        "continuity_advantage_term": round(continuity_term, 6),
        "handoff_failure_reduction_term": round(failure_term, 6),
        "backhaul_cost_reduction_term": round(backhaul_term, 6),
        "handoff_ready_advantage_term": round(ready_term, 6),
        "mechanism_realization_advantage_term": round(mechanism_term, 6),
        "prefetch_validated_hit_rate": round(prefetch_hit_rate, 6),
        "migration_prepare_rate": round(migration_prepare_rate, 6),
        "adapter_state_migration_overhead": round(migration_overhead, 6),
        "stability_summary": stability_summary,
        "missing_metrics": sorted(set(missing_metrics)),
        "reference_missing_metrics": sorted(set(reference_missing_metrics)),
        "formula": (
            "reward_term + 45*continuity_advantage + 35*handoff_failure_reduction "
            "+ 0.08*backhaul_cost_reduction + 30*handoff_ready_advantage "
            "+ 28*mechanism_realization_advantage + 8*prefetch_validated_hit_rate "
            "+ 5*migration_prepare_rate - 0.2*adapter_state_migration_overhead "
            "- stability_penalty"
        ),
        "guardrail_note": (
            "Selection is external to env reward and keeps reward, continuity, failure, "
            "traffic cost, readiness, mechanism realization, and window stability in the score."
        ),
    }


def mechanism_advantage_score_priority_tuple(
    metrics: dict[str, Any],
    *,
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> tuple[float, float, float, float, float, float, float, float]:
    score_payload = compute_mechanism_advantage_checkpoint_score(
        metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    stability_summary = score_payload.get("stability_summary", {})
    return (
        float(score_payload["score"]),
        float(metrics.get("workflow_continuity_rate", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        -float(metrics.get("backhaul_traffic_cost", 0.0)),
        -float(stability_summary.get("stability_penalty", 0.0)),
        float(metrics.get("total_reward", 0.0)),
    )


def compute_round2_mechanism_checkpoint_score(
    metrics: dict[str, Any],
    policy_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_diagnostics = policy_diagnostics or {}
    reward = float(metrics.get("total_reward", 0.0) or 0.0)
    continuity = float(metrics.get("workflow_continuity_rate", 0.0) or 0.0)
    handoff_ready = float(metrics.get("handoff_ready_ratio", 0.0) or 0.0)
    mechanism = float(metrics.get("mechanism_realization_rate", 0.0) or 0.0)
    handoff_failure = float(metrics.get("handoff_failure_rate", 0.0) or 0.0)
    backhaul_cost = float(metrics.get("backhaul_traffic_cost", 0.0) or 0.0)
    migration_overhead = float(metrics.get("adapter_state_migration_overhead", 0.0) or 0.0)
    prefetch_hit_rate = float(metrics.get("prefetch_validated_hit_rate", 0.0) or 0.0)
    migration_prepare_rate = float(metrics.get("migration_prepare_rate", 0.0) or 0.0)
    event_prepare_prob = float(policy_diagnostics.get("event_prepare_prob_mean", 0.0) or 0.0)
    guard_prefetch_to_prepare_count = float(policy_diagnostics.get("guard_prefetch_to_prepare_count", 0.0) or 0.0)
    score = (
        reward
        + 42.0 * continuity
        + 24.0 * handoff_ready
        + 32.0 * mechanism
        - 35.0 * handoff_failure
        - 0.08 * backhaul_cost
        - 0.2 * migration_overhead
        + 6.0 * prefetch_hit_rate
        + 5.0 * migration_prepare_rate
        + 4.0 * event_prepare_prob
        + 0.08 * guard_prefetch_to_prepare_count
    )
    return {
        "score": round(score, 6),
        "total_reward": round(reward, 6),
        "workflow_continuity_rate": round(continuity, 6),
        "handoff_ready_ratio": round(handoff_ready, 6),
        "mechanism_realization_rate": round(mechanism, 6),
        "handoff_failure_rate": round(handoff_failure, 6),
        "backhaul_traffic_cost": round(backhaul_cost, 6),
        "adapter_state_migration_overhead": round(migration_overhead, 6),
        "prefetch_validated_hit_rate": round(prefetch_hit_rate, 6),
        "migration_prepare_rate": round(migration_prepare_rate, 6),
        "event_prepare_prob_mean": round(event_prepare_prob, 6),
        "guard_prefetch_to_prepare_count": round(guard_prefetch_to_prepare_count, 6),
        "formula": (
            "total_reward + 42*continuity + 24*handoff_ready + 32*mechanism "
            "- 35*handoff_failure - 0.08*backhaul - 0.2*migration_overhead "
            "+ 6*prefetch_hit_rate + 5*migration_prepare_rate "
            "+ 4*event_prepare_prob_mean + 0.08*guard_prefetch_to_prepare_count"
        ),
    }


def round2_mechanism_score_priority_tuple(
    metrics: dict[str, Any],
    policy_diagnostics: dict[str, Any] | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    score_payload = compute_round2_mechanism_checkpoint_score(metrics, policy_diagnostics)
    return (
        float(score_payload["score"]),
        float(metrics.get("workflow_continuity_rate", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        -float(metrics.get("backhaul_traffic_cost", 0.0)),
        float(metrics.get("total_reward", 0.0)),
    )


def compute_retained_mechanism_checkpoint_score(
    metrics: dict[str, Any],
    policy_diagnostics: dict[str, Any] | None = None,
    *,
    update_index: int = 0,
    retention_start_update: int = 8,
) -> dict[str, Any]:
    policy_diagnostics = policy_diagnostics or {}
    reward = float(metrics.get("total_reward", 0.0) or 0.0)
    continuity = float(metrics.get("workflow_continuity_rate", 0.0) or 0.0)
    handoff_ready = float(metrics.get("handoff_ready_ratio", 0.0) or 0.0)
    mechanism = float(metrics.get("mechanism_realization_rate", 0.0) or 0.0)
    handoff_failure = float(metrics.get("handoff_failure_rate", 0.0) or 0.0)
    backhaul_cost = float(metrics.get("backhaul_traffic_cost", 0.0) or 0.0)
    migration_overhead = float(metrics.get("adapter_state_migration_overhead", 0.0) or 0.0)
    prefetch_hit_rate = float(metrics.get("prefetch_validated_hit_rate", 0.0) or 0.0)
    migration_prepare_rate = float(metrics.get("migration_prepare_rate", 0.0) or 0.0)
    event_prepare_prob = float(policy_diagnostics.get("event_prepare_prob_mean", 0.0) or 0.0)
    guard_prefetch_to_prepare_count = float(policy_diagnostics.get("guard_prefetch_to_prepare_count", 0.0) or 0.0)
    late_stage = bool(int(update_index) >= int(retention_start_update) > 0)
    late_prepare_stability_bonus = 0.0
    if late_stage:
        late_prepare_stability_bonus = (
            6.0 * event_prepare_prob
            + 0.10 * guard_prefetch_to_prepare_count
            + 2.0 * migration_prepare_rate
        )
    failure_guardrail_penalty = 45.0 * max(0.0, handoff_failure - 0.12)
    backhaul_guardrail_penalty = 0.12 * max(0.0, backhaul_cost - 170.0)
    score = (
        reward
        + 44.0 * continuity
        + 30.0 * handoff_ready
        + 34.0 * mechanism
        - 45.0 * handoff_failure
        - 0.09 * backhaul_cost
        - 0.2 * migration_overhead
        + 6.0 * prefetch_hit_rate
        + 6.0 * migration_prepare_rate
        + 5.0 * event_prepare_prob
        + 0.08 * guard_prefetch_to_prepare_count
        + late_prepare_stability_bonus
        - failure_guardrail_penalty
        - backhaul_guardrail_penalty
    )
    return {
        "score": round(score, 6),
        "total_reward": round(reward, 6),
        "workflow_continuity_rate": round(continuity, 6),
        "handoff_ready_ratio": round(handoff_ready, 6),
        "mechanism_realization_rate": round(mechanism, 6),
        "handoff_failure_rate": round(handoff_failure, 6),
        "backhaul_traffic_cost": round(backhaul_cost, 6),
        "adapter_state_migration_overhead": round(migration_overhead, 6),
        "prefetch_validated_hit_rate": round(prefetch_hit_rate, 6),
        "migration_prepare_rate": round(migration_prepare_rate, 6),
        "event_prepare_prob_mean": round(event_prepare_prob, 6),
        "guard_prefetch_to_prepare_count": round(guard_prefetch_to_prepare_count, 6),
        "update_index": int(update_index),
        "retention_start_update": int(retention_start_update),
        "late_stage": late_stage,
        "late_prepare_stability_bonus": round(late_prepare_stability_bonus, 6),
        "failure_guardrail_penalty": round(failure_guardrail_penalty, 6),
        "backhaul_guardrail_penalty": round(backhaul_guardrail_penalty, 6),
        "formula": (
            "total_reward + 44*continuity + 30*handoff_ready + 34*mechanism "
            "- 45*handoff_failure - 0.09*backhaul - 0.2*migration_overhead "
            "+ 6*prefetch_hit_rate + 6*migration_prepare_rate + 5*event_prepare_prob_mean "
            "+ 0.08*guard_prefetch_to_prepare_count + late_prepare_stability_bonus "
            "- failure_guardrail_penalty - backhaul_guardrail_penalty"
        ),
    }


def retained_mechanism_score_priority_tuple(
    metrics: dict[str, Any],
    policy_diagnostics: dict[str, Any] | None = None,
    *,
    update_index: int = 0,
    retention_start_update: int = 8,
) -> tuple[float, float, float, float, float, float, float, float]:
    score_payload = compute_retained_mechanism_checkpoint_score(
        metrics,
        policy_diagnostics,
        update_index=update_index,
        retention_start_update=retention_start_update,
    )
    return (
        float(score_payload["score"]),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        float(metrics.get("workflow_continuity_rate", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        -float(metrics.get("backhaul_traffic_cost", 0.0)),
        float(policy_diagnostics.get("event_prepare_prob_mean", 0.0) or 0.0),
        float(update_index),
    )


def compute_reward_tiebreak_checkpoint_score(
    metrics: dict[str, Any],
    *,
    reference_metrics: dict[str, Any] | None = None,
    full_metrics: dict[str, Any] | None = None,
    full_reference_metrics: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    reference_metrics = reference_metrics or {}
    full_metrics = full_metrics or {}
    full_reference_metrics = full_reference_metrics or {}
    missing_metrics: list[str] = []
    reference_missing_metrics: list[str] = []
    full_missing_metrics: list[str] = []
    full_reference_missing_metrics: list[str] = []

    mixed_reward = _float_metric(metrics, "total_reward", missing_metrics)
    mixed_continuity = _float_metric(metrics, "workflow_continuity_rate", missing_metrics)
    mixed_failure = _float_metric(metrics, "handoff_failure_rate", missing_metrics)
    mixed_backhaul = _float_metric(metrics, "backhaul_traffic_cost", missing_metrics)
    mixed_ready = _float_metric(metrics, "handoff_ready_ratio", missing_metrics)
    mixed_mechanism = _float_metric(metrics, "mechanism_realization_rate", missing_metrics)
    mixed_migration = _float_metric(metrics, "adapter_state_migration_overhead", missing_metrics)

    reference_available = bool(reference_metrics)
    if reference_available:
        mixed_reward_gap = mixed_reward - _float_metric(reference_metrics, "total_reward", reference_missing_metrics)
        mixed_continuity_gap = mixed_continuity - _float_metric(reference_metrics, "workflow_continuity_rate", reference_missing_metrics)
        mixed_failure_reduction = _float_metric(reference_metrics, "handoff_failure_rate", reference_missing_metrics) - mixed_failure
        mixed_backhaul_advantage = _float_metric(reference_metrics, "backhaul_traffic_cost", reference_missing_metrics) - mixed_backhaul
        mixed_ready_gap = mixed_ready - _float_metric(reference_metrics, "handoff_ready_ratio", reference_missing_metrics)
        mixed_mechanism_gap = mixed_mechanism - _float_metric(reference_metrics, "mechanism_realization_rate", reference_missing_metrics)
    else:
        mixed_reward_gap = mixed_reward
        mixed_continuity_gap = mixed_continuity
        mixed_failure_reduction = -mixed_failure
        mixed_backhaul_advantage = 0.0
        mixed_ready_gap = mixed_ready
        mixed_mechanism_gap = mixed_mechanism

    full_available = bool(full_metrics)
    full_reference_available = bool(full_reference_metrics)
    if full_available:
        full_reward = _float_metric(full_metrics, "total_reward", full_missing_metrics)
        full_continuity = _float_metric(full_metrics, "workflow_continuity_rate", full_missing_metrics)
        full_failure = _float_metric(full_metrics, "handoff_failure_rate", full_missing_metrics)
        full_backhaul = _float_metric(full_metrics, "backhaul_traffic_cost", full_missing_metrics)
        full_ready = _float_metric(full_metrics, "handoff_ready_ratio", full_missing_metrics)
        full_mechanism = _float_metric(full_metrics, "mechanism_realization_rate", full_missing_metrics)
    else:
        full_reward = mixed_reward
        full_continuity = mixed_continuity
        full_failure = mixed_failure
        full_backhaul = mixed_backhaul
        full_ready = mixed_ready
        full_mechanism = mixed_mechanism

    if full_reference_available:
        full_reward_gap = full_reward - _float_metric(full_reference_metrics, "total_reward", full_reference_missing_metrics)
        full_continuity_gap = full_continuity - _float_metric(full_reference_metrics, "workflow_continuity_rate", full_reference_missing_metrics)
        full_failure_reduction = _float_metric(full_reference_metrics, "handoff_failure_rate", full_reference_missing_metrics) - full_failure
        full_backhaul_advantage = _float_metric(full_reference_metrics, "backhaul_traffic_cost", full_reference_missing_metrics) - full_backhaul
        full_ready_gap = full_ready - _float_metric(full_reference_metrics, "handoff_ready_ratio", full_reference_missing_metrics)
        full_mechanism_gap = full_mechanism - _float_metric(full_reference_metrics, "mechanism_realization_rate", full_reference_missing_metrics)
    else:
        full_reward_gap = full_reward if full_available else 0.0
        full_continuity_gap = full_continuity if full_available else 0.0
        full_failure_reduction = -full_failure if full_available else 0.0
        full_backhaul_advantage = 0.0
        full_ready_gap = full_ready if full_available else 0.0
        full_mechanism_gap = full_mechanism if full_available else 0.0

    stability_summary = build_mechanism_advantage_stability_summary(
        metrics=metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    stability_penalty = float(stability_summary.get("stability_penalty", 0.0) or 0.0)
    safety_penalty = 0.0
    if reference_available:
        safety_penalty += 120.0 * max(0.0, -mixed_continuity_gap)
        safety_penalty += 100.0 * max(0.0, -mixed_failure_reduction)
        safety_penalty += 0.5 * max(0.0, 8.0 - mixed_backhaul_advantage)
    else:
        safety_penalty += 120.0 * max(0.0, 0.98 - mixed_continuity)
        safety_penalty += 100.0 * max(0.0, mixed_failure - 0.02)
        safety_penalty += 0.15 * max(0.0, mixed_backhaul - 170.0)
    if full_available:
        safety_penalty += 120.0 * max(0.0, 0.92 - full_continuity)
        safety_penalty += 100.0 * max(0.0, full_failure - 0.18)
        safety_penalty += 0.35 * max(0.0, full_backhaul - 158.222222)
    if full_reference_available:
        safety_penalty += 120.0 * max(0.0, -full_continuity_gap)
        safety_penalty += 100.0 * max(0.0, -full_failure_reduction)
        safety_penalty += 0.5 * max(0.0, 4.0 - full_backhaul_advantage)
        safety_penalty += 80.0 * max(0.0, -full_reward_gap)

    score = (
        5.0 * mixed_reward_gap
        + 1.5 * full_reward_gap
        + 65.0 * mixed_continuity_gap
        + 75.0 * mixed_failure_reduction
        + 0.06 * mixed_backhaul_advantage
        + 18.0 * mixed_ready_gap
        + 18.0 * mixed_mechanism_gap
        + 35.0 * full_continuity_gap
        + 45.0 * full_failure_reduction
        + 0.03 * full_backhaul_advantage
        + 8.0 * full_ready_gap
        + 8.0 * full_mechanism_gap
        - 0.15 * mixed_migration
        - stability_penalty
        - safety_penalty
    )
    return {
        "score": round(score, 6),
        "score_mode": "reward_tiebreak_against_reference" if reference_available else "reward_tiebreak_self_metrics_fallback",
        "reference_available": reference_available,
        "full_metrics_available": full_available,
        "full_reference_available": full_reference_available,
        "mixed_reward_gap": round(mixed_reward_gap, 6),
        "full_reward_gap": round(full_reward_gap, 6),
        "continuity_gap": round(mixed_continuity_gap, 6),
        "failure_reduction": round(mixed_failure_reduction, 6),
        "backhaul_advantage": round(mixed_backhaul_advantage, 6),
        "handoff_ready_ratio_gap": round(mixed_ready_gap, 6),
        "mechanism_realization_gap": round(mixed_mechanism_gap, 6),
        "full_continuity_gap": round(full_continuity_gap, 6),
        "full_failure_reduction": round(full_failure_reduction, 6),
        "full_backhaul_advantage": round(full_backhaul_advantage, 6),
        "full_handoff_ready_ratio_gap": round(full_ready_gap, 6),
        "full_mechanism_realization_gap": round(full_mechanism_gap, 6),
        "adapter_state_migration_overhead": round(mixed_migration, 6),
        "stability_summary": stability_summary,
        "safety_penalty": round(safety_penalty, 6),
        "missing_metrics": sorted(set(missing_metrics)),
        "reference_missing_metrics": sorted(set(reference_missing_metrics)),
        "full_missing_metrics": sorted(set(full_missing_metrics)),
        "full_reference_missing_metrics": sorted(set(full_reference_missing_metrics)),
        "formula": (
            "5*mixed_reward_gap + 1.5*full_reward_gap + 65*continuity_gap "
            "+ 75*failure_reduction + 0.06*backhaul_advantage "
            "+ 18*handoff_ready_gap + 18*mechanism_gap + full safety terms "
            "- 0.15*migration_overhead - stability_penalty - safety_penalty"
        ),
        "guardrail_note": (
            "External checkpoint ranking only. It does not alter env reward; if reference/full metrics "
            "are unavailable, missing fields fall back to self-metrics and are recorded."
        ),
    }


def reward_tiebreak_score_priority_tuple(
    metrics: dict[str, Any],
    *,
    reference_metrics: dict[str, Any] | None = None,
    full_metrics: dict[str, Any] | None = None,
    full_reference_metrics: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> tuple[float, float, float, float, float, float, float, float]:
    score_payload = compute_reward_tiebreak_checkpoint_score(
        metrics,
        reference_metrics=reference_metrics,
        full_metrics=full_metrics,
        full_reference_metrics=full_reference_metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    stability_summary = score_payload.get("stability_summary", {})
    return (
        float(score_payload["score"]),
        float(metrics.get("workflow_continuity_rate", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        -float(metrics.get("backhaul_traffic_cost", 0.0)),
        float(metrics.get("total_reward", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        -float(stability_summary.get("stability_penalty", 0.0)),
    )


def compute_pareto_safe_checkpoint_score(
    metrics: dict[str, Any],
    *,
    reference_metrics: dict[str, Any] | None = None,
    reference_agent: str = "ppo",
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    reference_metrics = reference_metrics or {}
    missing_metrics: list[str] = []
    reference_missing_metrics: list[str] = []
    reference_available = bool(reference_metrics)
    reward = _float_metric(metrics, "total_reward", missing_metrics)
    continuity = _float_metric(metrics, "workflow_continuity_rate", missing_metrics)
    handoff_failure = _float_metric(metrics, "handoff_failure_rate", missing_metrics)
    backhaul_cost = _float_metric(metrics, "backhaul_traffic_cost", missing_metrics)
    handoff_ready = _float_metric(metrics, "handoff_ready_ratio", missing_metrics)
    mechanism = _float_metric(metrics, "mechanism_realization_rate", missing_metrics)
    migration_overhead = _float_metric(metrics, "adapter_state_migration_overhead", missing_metrics)
    if reference_available:
        reward_gap = reward - _float_metric(reference_metrics, "total_reward", reference_missing_metrics)
        continuity_gap = continuity - _float_metric(reference_metrics, "workflow_continuity_rate", reference_missing_metrics)
        failure_reduction = _float_metric(reference_metrics, "handoff_failure_rate", reference_missing_metrics) - handoff_failure
        backhaul_advantage = _float_metric(reference_metrics, "backhaul_traffic_cost", reference_missing_metrics) - backhaul_cost
    else:
        reward_gap = reward
        continuity_gap = continuity
        failure_reduction = -handoff_failure
        backhaul_advantage = -backhaul_cost
    stability_summary = build_mechanism_advantage_stability_summary(
        metrics=metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    stability_penalty = float(stability_summary.get("stability_penalty", 0.0) or 0.0)
    noninferiority_penalty = (
        180.0 * max(0.0, -failure_reduction)
        + 1.20 * max(0.0, -backhaul_advantage)
        + 120.0 * max(0.0, -continuity_gap)
    )
    score = (
        4.0 * reward_gap
        + 70.0 * continuity_gap
        + 140.0 * failure_reduction
        + 0.35 * backhaul_advantage
        + 15.0 * handoff_ready
        + 12.0 * mechanism
        - 0.15 * migration_overhead
        - stability_penalty
        - noninferiority_penalty
    )
    return {
        "score": round(score, 6),
        "score_mode": "pareto_safe_against_reference" if reference_available else "pareto_safe_self_metrics_fallback",
        "reference_available": reference_available,
        "reference_agent": reference_agent if reference_available else "",
        "reward_gap": round(reward_gap, 6),
        "continuity_gap": round(continuity_gap, 6),
        "handoff_failure_reduction": round(failure_reduction, 6),
        "backhaul_advantage": round(backhaul_advantage, 6),
        "handoff_ready_ratio": round(handoff_ready, 6),
        "mechanism_realization_rate": round(mechanism, 6),
        "adapter_state_migration_overhead": round(migration_overhead, 6),
        "stability_summary": stability_summary,
        "noninferiority_penalty": round(noninferiority_penalty, 6),
        "missing_metrics": sorted(set(missing_metrics)),
        "reference_missing_metrics": sorted(set(reference_missing_metrics)),
        "formula": (
            "4*reward_gap + 70*continuity_gap + 140*handoff_failure_reduction "
            "+ 0.35*backhaul_advantage + 15*handoff_ready + 12*mechanism "
            "- 0.15*migration_overhead - stability_penalty - noninferiority_penalty"
        ),
        "noninferiority_rule": (
            "Prefer checkpoints that do not regress handoff failure or backhaul against the "
            "available reference. This is checkpoint ranking only and does not alter env reward."
        ),
    }


def pareto_safe_score_priority_tuple(
    metrics: dict[str, Any],
    *,
    reference_metrics: dict[str, Any] | None = None,
    reference_agent: str = "ppo",
    rows: list[dict[str, Any]] | None = None,
    current_agent_name: str = "sa_ghmappo",
) -> tuple[float, float, float, float, float, float, float]:
    score_payload = compute_pareto_safe_checkpoint_score(
        metrics,
        reference_metrics=reference_metrics,
        reference_agent=reference_agent,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    return (
        float(score_payload["score"]),
        float(score_payload["handoff_failure_reduction"]),
        float(score_payload["backhaul_advantage"]),
        float(score_payload["continuity_gap"]),
        float(score_payload["reward_gap"]),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
    )


def build_reward_tiebreak_selection_reason(
    *,
    candidate_metrics: dict[str, Any],
    popularity_metrics: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    selected: bool,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    score_breakdown = compute_reward_tiebreak_checkpoint_score(
        candidate_metrics,
        reference_metrics=popularity_metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    reason = build_advantage_selection_reason(
        candidate_metrics=candidate_metrics,
        popularity_metrics=popularity_metrics,
        selected=selected,
    )
    reason["selection_rule"] = "best_by_reward_tiebreak_score"
    reason["selection_intent"] = (
        "round4 reward tie-break checkpoint ranking; keep continuity/failure/backhaul guardrails, "
        "then prefer checkpoints with higher mixed-like reward."
    )
    reason["score_breakdown"] = score_breakdown
    reason["graceful_fallback"] = {
        "reference_metrics_available": bool(popularity_metrics),
        "full_metrics_available": False,
        "missing_metrics": score_breakdown.get("missing_metrics", []),
        "reference_missing_metrics": score_breakdown.get("reference_missing_metrics", []),
        "full_missing_metrics": score_breakdown.get("full_missing_metrics", []),
        "full_reference_missing_metrics": score_breakdown.get("full_reference_missing_metrics", []),
    }
    return reason


def advantage_score_priority_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float]:
    score_payload = compute_advantage_checkpoint_score(metrics)
    return (
        float(score_payload["score"]),
        float(metrics.get("workflow_continuity_rate", 0.0)),
        -float(metrics.get("handoff_failure_rate", 0.0)),
        -float(metrics.get("backhaul_traffic_cost", 0.0)),
        float(metrics.get("handoff_ready_ratio", 0.0)),
        float(metrics.get("mechanism_realization_rate", 0.0)),
        float(metrics.get("total_reward", 0.0)),
    )


def build_advantage_selection_reason(
    *,
    candidate_metrics: dict[str, float],
    popularity_metrics: dict[str, float],
    selected: bool,
) -> dict[str, Any]:
    focus_metrics = [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "handoff_ready_ratio",
        "mechanism_realization_rate",
        "adapter_state_migration_overhead",
    ]
    lower_is_better = {
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_state_migration_overhead",
    }
    deltas: dict[str, dict[str, Any]] = {}
    wins = 0
    losses = 0
    ties = 0
    for metric_name in focus_metrics:
        candidate_value = float(candidate_metrics.get(metric_name, 0.0) or 0.0)
        popularity_value = float(popularity_metrics.get(metric_name, 0.0) or 0.0)
        delta = candidate_value - popularity_value
        effective_delta = -delta if metric_name in lower_is_better else delta
        if effective_delta > 1e-6:
            result = "win"
            wins += 1
        elif effective_delta < -1e-6:
            result = "loss"
            losses += 1
        else:
            result = "tie"
            ties += 1
        deltas[metric_name] = {
            "sa_ghmappo": round(candidate_value, 6),
            "popularity_cache_heuristic": round(popularity_value, 6),
            "delta_sa_minus_popularity": round(delta, 6),
            "higher_is_better": metric_name not in lower_is_better,
            "result": result,
        }
    return {
        "selected": bool(selected),
        "selection_rule": "best_by_advantage_score",
        "selection_intent": "balance reward, continuity, mechanism readiness, failure rate, and traffic cost without changing env reward",
        "score_breakdown": compute_advantage_checkpoint_score(candidate_metrics),
        "comparison_against_popularity_cache_heuristic": {
            "available": bool(popularity_metrics),
            "metrics": deltas,
            "win_loss_tie": {"win": wins, "loss": losses, "tie": ties},
        },
    }


def build_mechanism_advantage_selection_reason(
    *,
    candidate_metrics: dict[str, Any],
    popularity_metrics: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    selected: bool,
    current_agent_name: str = "sa_ghmappo",
) -> dict[str, Any]:
    score_breakdown = compute_mechanism_advantage_checkpoint_score(
        candidate_metrics,
        reference_metrics=popularity_metrics,
        rows=rows,
        current_agent_name=current_agent_name,
    )
    reason = build_advantage_selection_reason(
        candidate_metrics=candidate_metrics,
        popularity_metrics=popularity_metrics,
        selected=selected,
    )
    reason["selection_rule"] = "best_by_mechanism_advantage_score"
    reason["selection_intent"] = (
        "v2 mechanism-aware external checkpoint ranking; preserve reward/continuity/failure "
        "while explicitly scoring handoff readiness, mechanism realization, backhaul, and window stability"
    )
    reason["score_breakdown"] = score_breakdown
    reason["graceful_fallback"] = {
        "reference_metrics_available": bool(popularity_metrics),
        "missing_metrics": score_breakdown.get("missing_metrics", []),
        "reference_missing_metrics": score_breakdown.get("reference_missing_metrics", []),
    }
    return reason


def compute_mechanism_score(metrics: dict[str, float]) -> float:
    return float(
        0.35 * float(metrics.get("mechanism_realization_rate", 0.0))
        + 0.2 * float(metrics.get("prefetch_validated_hit_rate", 0.0))
        + 0.2 * float(metrics.get("handoff_ready_rate", 0.0))
        + 0.15 * float(metrics.get("migration_prepare_rate", 0.0))
        + 0.1 * float(metrics.get("prefetch_request_rate", 0.0))
    )


def maybe_apply_anti_collapse_controls(
    *,
    agent: Any,
    args: argparse.Namespace,
    update_index: int,
    learn_info: dict[str, Any],
    current_eval_metrics: dict[str, Any],
    best_mechanism_score_so_far: float,
    stability_control_history: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    current_score = compute_mechanism_score(current_eval_metrics)
    updated_best = max(float(best_mechanism_score_so_far), current_score)
    stability_profiles = {
        "formal_main_stable",
        "top_journal_mechanism_v11_mappo_reward",
        "top_journal_mechanism_v12_learned_option",
        "top_journal_mechanism_v13_prd_option",
        "top_journal_mechanism_v14_net_utility_prd",
        "top_journal_mechanism_v15_terminal_option",
        "top_journal_mechanism_v16_conservative_terminal_option",
        "top_journal_mechanism_v17_dag_aware_option",
        "top_journal_mechanism_v18_counterfactual_option",
        "top_journal_mechanism_v19_handoff_risk_prd",
        "top_journal_mechanism_v20_idle_execution_prd",
        "top_journal_mechanism_v21_efficiency_prd",
        "top_journal_mechanism_v22_validated_utility_prd",
        "top_journal_mechanism_v23_counterfactual_constrained_prd",
        "top_journal_mechanism_v24_tail_risk_constrained_prd",
        "top_journal_mechanism_v25_opportunity_risk_prd",
        "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd",
        "top_journal_mechanism_v27_conservative_advantage_imitation",
        "top_journal_mechanism_v28_credit_focal_mappo",
        "top_journal_mechanism_v29_dt_fused_credit_mappo",
        "top_journal_mechanism_v30_dt_prior_mappo",
        "top_journal_mechanism_v31_handoff_pacing_mappo",
        "top_journal_mechanism_v32_dt_continuation_mappo",
    }
    if args.agent_name != "sa_ghmappo" or args.profile not in stability_profiles:
        return updated_best, None
    if not hasattr(agent, "apply_stability_controls"):
        return updated_best, None
    if len(stability_control_history) >= 3:
        return updated_best, None
    target_kl = float(learn_info.get("target_kl", 0.0) or 0.0)
    approx_kl = float(learn_info.get("approx_kl", 0.0) or 0.0)
    current_realization = float(current_eval_metrics.get("mechanism_realization_rate", 0.0) or 0.0)
    collapse_like = bool(
        update_index >= 3
        and best_mechanism_score_so_far > 1e-6
        and current_score <= 0.65 * float(best_mechanism_score_so_far)
        and current_realization <= 0.5
    )
    high_kl = bool(target_kl > 0.0 and approx_kl >= 1.1 * target_kl)
    if not collapse_like and not high_kl:
        return updated_best, None
    reason = "mechanism_score_drop" if collapse_like else "high_kl"
    applied = agent.apply_stability_controls(
        learning_rate_scale=0.75,
        clip_ratio_scale=0.9,
        entropy_coef_scale=0.85,
        auxiliary_coef_scale=1.15,
        slow_weight_scale=1.06,
        event_weight_scale=1.1,
        mechanism_bias_delta=0.05,
        max_auxiliary_coef=0.45,
        max_mechanism_logit_bias_strength=1.6,
    )
    record = {
        "update_index": update_index,
        "reason": reason,
        "best_mechanism_score_before": round(float(best_mechanism_score_so_far), 6),
        "current_mechanism_score": round(current_score, 6),
        "current_mechanism_realization_rate": round(current_realization, 6),
        "approx_kl": round(approx_kl, 6),
        "target_kl": round(target_kl, 6),
        "applied_controls": applied,
    }
    stability_control_history.append(record)
    return updated_best, record


def build_mechanism_collapse_audit(update_eval_history: list[dict[str, Any]], current_agent_name: str) -> dict[str, Any]:
    per_update: list[dict[str, Any]] = []
    peak_score = -1.0
    peak_update = 0
    collapse_start_update = 0
    for item in update_eval_history:
        metrics = dict(item.get("aggregate_by_agent", {}).get(current_agent_name, {}))
        update_row = {
            "update_index": int(item.get("update_index", 0) or 0),
            "total_reward": float(metrics.get("total_reward", 0.0) or 0.0),
            "workflow_continuity_rate": float(metrics.get("workflow_continuity_rate", 0.0) or 0.0),
            "mechanism_realization_rate": float(metrics.get("mechanism_realization_rate", 0.0) or 0.0),
            "prefetch_request_rate": float(metrics.get("prefetch_request_rate", 0.0) or 0.0),
            "prefetch_validated_hit_rate": float(metrics.get("prefetch_validated_hit_rate", 0.0) or 0.0),
            "migration_prepare_rate": float(metrics.get("migration_prepare_rate", 0.0) or 0.0),
            "handoff_ready_rate": float(metrics.get("handoff_ready_rate", 0.0) or 0.0),
        }
        update_row["mechanism_score"] = round(compute_mechanism_score(update_row), 6)
        per_update.append(update_row)
        if update_row["mechanism_score"] > peak_score:
            peak_score = update_row["mechanism_score"]
            peak_update = update_row["update_index"]
    collapse_threshold = max(0.15, 0.5 * peak_score) if peak_score > 0 else 0.15
    for update_row in per_update:
        if update_row["update_index"] <= peak_update:
            continue
        if update_row["mechanism_score"] <= collapse_threshold and update_row["mechanism_realization_rate"] <= 0.5:
            collapse_start_update = update_row["update_index"]
            break
    high_band = 0.9 * peak_score if peak_score > 0 else 0.0
    best_updates_before_collapse = [
        item["update_index"]
        for item in per_update
        if item["update_index"] < (collapse_start_update or 10 ** 9) and item["mechanism_score"] >= high_band
    ]
    best_range = {
        "start_update": best_updates_before_collapse[0] if best_updates_before_collapse else peak_update,
        "end_update": best_updates_before_collapse[-1] if best_updates_before_collapse else peak_update,
    }
    return {
        "update_metrics": per_update,
        "collapse_detection_rule": {
            "peak_score_ratio_threshold": 0.5,
            "minimum_absolute_score_threshold": 0.15,
            "mechanism_realization_rate_threshold": 0.5,
        },
        "peak_mechanism_score": round(max(peak_score, 0.0), 6),
        "peak_update": peak_update,
        "collapse_detected": bool(collapse_start_update > 0),
        "collapse_start_update": collapse_start_update,
        "pre_collapse_best_update_range": best_range,
    }


def build_checkpoint_family_eval(
    *,
    current_agent_name: str,
    checkpoint_root: Path,
    workflow_states: list[Any],
    eval_windows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    audit_scope = resolve_post_training_audit_scope(args)
    family_results: dict[str, Any] = {}
    for label in [
        "best_by_reward",
        "best_by_continuity",
        "best_by_mechanism_balanced",
        "best_by_advantage_score",
        "best_by_mechanism_advantage_score",
        "best_by_round2_mechanism_score",
        "best_by_retained_mechanism_score",
        "best_by_reward_tiebreak_score",
    ]:
        checkpoint_path = checkpoint_root / f"{label}.pt"
        if not checkpoint_path.exists():
            family_results[label] = {
                "checkpoint_path": str(checkpoint_path),
                "exists": False,
            }
            continue
        try:
            payload = evaluate_checkpoint_protocol(
                current_agent_name=current_agent_name,
                checkpoint_path=checkpoint_path,
                workflow_states=workflow_states,
                eval_windows=eval_windows,
                args=args,
                include_reference_agents=True,
                max_workflows=audit_scope["max_workflows"],
                max_windows=audit_scope["max_windows"],
                protocol_name=f"{label}_family_eval",
            )
            checkpoint_metadata = load_checkpoint_metadata(str(checkpoint_path))
        except Exception as exc:  # Defensive audit path for partially written checkpoints.
            family_results[label] = {
                "checkpoint_path": str(checkpoint_path),
                "exists": True,
                "load_error": f"{type(exc).__name__}: {exc}",
                "checkpoint_metadata": {},
                "aggregate_by_agent": {},
                "aggregate_reward_breakdown_by_agent": {},
                "aggregate_mechanism_diagnostics_by_agent": {},
                "aggregate_policy_diagnostics_by_agent": {},
                "eval_window_ids": [],
                "workflow_ids": [],
            }
            continue
        family_results[label] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "aggregate_by_agent": payload.get("aggregate_by_agent", {}),
            "aggregate_reward_breakdown_by_agent": payload.get("aggregate_reward_breakdown_by_agent", {}),
            "aggregate_mechanism_diagnostics_by_agent": payload.get("aggregate_mechanism_diagnostics_by_agent", {}),
            "aggregate_policy_diagnostics_by_agent": payload.get("aggregate_policy_diagnostics_by_agent", {}),
            "eval_window_ids": payload.get("eval_window_ids", []),
            "workflow_ids": payload.get("workflow_ids", []),
        }
    return {
        "agent_name": current_agent_name,
        "config_profile": args.profile,
        "audit_scope": {
            "mode": audit_scope["mode"],
            "full_protocol": audit_scope["full_protocol"],
            "max_windows": audit_scope["max_windows"],
            "max_workflows": audit_scope["max_workflows"],
        },
        "family_results": family_results,
    }


def maybe_update_best_checkpoint(
    *,
    current_agent_name: str,
    checkpoint_path: Path,
    checkpoint_root: Path,
    update_index: int,
    episode_index: int,
    update_eval: dict[str, Any],
    best_record: dict[str, Any],
    retention_start_update: int = 8,
) -> dict[str, Any]:
    policy_learning_gate = dict(update_eval.get("policy_learning_gate", {}) or {})
    selection_eval = dict(update_eval.get("raw_policy_eval", {}) or update_eval)
    selection_protocol = {
        "protocol_name": selection_eval.get("protocol_name", "update_eval_raw_policy"),
        "deterministic_eval": bool(selection_eval.get("deterministic_eval", True)),
        "policy_evaluation_mode": selection_eval.get("policy_evaluation_mode", "raw_policy"),
        "eval_window_ids": list(selection_eval.get("eval_window_ids", [])),
        "workflow_ids": list(selection_eval.get("workflow_ids", [])),
        "policy_learning_gate": policy_learning_gate,
    }
    best_record["latest_checkpoint_path"] = str(checkpoint_root / "latest.pt")
    best_record["selection_protocol"] = selection_protocol
    best_record["policy_learning_gate"] = policy_learning_gate
    if not bool(policy_learning_gate.get("selection_eligible", False)):
        best_record["selection_blocked"] = {
            "update_index": int(update_index),
            "episode_index": int(episode_index),
            "checkpoint_path": str(checkpoint_path),
            "reason": str(policy_learning_gate.get("reason", "raw_policy_evidence_required")),
            "gate_status": str(policy_learning_gate.get("status", "missing_policy_learning_gate")),
        }
        return best_record
    current_metrics = dict(selection_eval["aggregate_by_agent"].get(current_agent_name, {}))
    reward_candidate = float(current_metrics.get("total_reward", 0.0))
    reward_best = float(best_record.get("best_by_reward", {}).get("score", float("-inf")))
    if reward_candidate > reward_best:
        target_path = checkpoint_root / "best_by_reward.pt"
        shutil.copy2(checkpoint_path, target_path)
        best_record["best_by_reward"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "score": reward_candidate,
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "selection_protocol": selection_protocol,
        }
    continuity_candidate = continuity_priority_tuple(current_metrics)
    continuity_best = tuple(best_record.get("best_by_continuity", {}).get("priority_tuple", (-1.0, -1.0, -1.0, -1.0)))
    if continuity_candidate > continuity_best:
        target_path = checkpoint_root / "best_by_continuity.pt"
        shutil.copy2(checkpoint_path, target_path)
        best_record["best_by_continuity"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(continuity_candidate),
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "selection_protocol": selection_protocol,
        }
    mechanism_balanced_candidate = mechanism_balanced_priority_tuple(current_metrics)
    mechanism_balanced_best = tuple(
        best_record.get(
            "best_by_mechanism_balanced",
            {},
        ).get(
            "priority_tuple",
            (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        )
    )
    current_mechanism_best_update = int(best_record.get("best_by_mechanism_balanced", {}).get("update_index", 0) or 0)
    if mechanism_balanced_candidate > mechanism_balanced_best or (
        mechanism_balanced_candidate == mechanism_balanced_best and update_index > current_mechanism_best_update
    ):
        target_path = checkpoint_root / "best_by_mechanism_balanced.pt"
        shutil.copy2(checkpoint_path, target_path)
        best_record["best_by_mechanism_balanced"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(mechanism_balanced_candidate),
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "selection_protocol": selection_protocol,
        }
    advantage_candidate = advantage_score_priority_tuple(current_metrics)
    advantage_best = tuple(
        best_record.get("best_by_advantage_score", {}).get(
            "priority_tuple",
            (float("-inf"), -1.0, -1.0, float("-inf"), -1.0, -1.0, float("-inf")),
        )
    )
    current_advantage_best_update = int(best_record.get("best_by_advantage_score", {}).get("update_index", 0) or 0)
    if advantage_candidate > advantage_best or (
        advantage_candidate == advantage_best and update_index > current_advantage_best_update
    ):
        target_path = checkpoint_root / "best_by_advantage_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        popularity_metrics = dict(selection_eval.get("aggregate_by_agent", {}).get("popularity_cache_heuristic", {}))
        best_record["best_by_advantage_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(advantage_candidate),
            "score_breakdown": compute_advantage_checkpoint_score(current_metrics),
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "selection_protocol": selection_protocol,
            "selection_reason": build_advantage_selection_reason(
                candidate_metrics=current_metrics,
                popularity_metrics=popularity_metrics,
                selected=True,
            ),
        }
    mechanism_advantage_candidate = mechanism_advantage_score_priority_tuple(
        current_metrics,
        rows=list(selection_eval.get("rows", [])),
        current_agent_name=current_agent_name,
    )
    mechanism_advantage_best = tuple(
        best_record.get("best_by_mechanism_advantage_score", {}).get(
            "priority_tuple",
            (float("-inf"), -1.0, -1.0, -1.0, -1.0, float("-inf"), float("-inf"), float("-inf")),
        )
    )
    current_mechanism_advantage_best_update = int(
        best_record.get("best_by_mechanism_advantage_score", {}).get("update_index", 0) or 0
    )
    if mechanism_advantage_candidate > mechanism_advantage_best or (
        mechanism_advantage_candidate == mechanism_advantage_best and update_index > current_mechanism_advantage_best_update
    ):
        target_path = checkpoint_root / "best_by_mechanism_advantage_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        popularity_metrics = dict(selection_eval.get("aggregate_by_agent", {}).get("popularity_cache_heuristic", {}))
        best_record["best_by_mechanism_advantage_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(mechanism_advantage_candidate),
            "score_breakdown": compute_mechanism_advantage_checkpoint_score(
                current_metrics,
                reference_metrics=popularity_metrics,
                rows=list(selection_eval.get("rows", [])),
                current_agent_name=current_agent_name,
            ),
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "selection_protocol": selection_protocol,
            "selection_reason": build_mechanism_advantage_selection_reason(
                candidate_metrics=current_metrics,
                popularity_metrics=popularity_metrics,
                rows=list(selection_eval.get("rows", [])),
                selected=True,
                current_agent_name=current_agent_name,
            ),
        }
    policy_diagnostics = dict(selection_eval.get("aggregate_policy_diagnostics_by_agent", {}).get(current_agent_name, {}))
    round2_candidate = round2_mechanism_score_priority_tuple(current_metrics, policy_diagnostics)
    round2_best = tuple(
        best_record.get("best_by_round2_mechanism_score", {}).get(
            "priority_tuple",
            (float("-inf"), -1.0, -1.0, -1.0, -1.0, float("-inf"), float("-inf")),
        )
    )
    current_round2_best_update = int(best_record.get("best_by_round2_mechanism_score", {}).get("update_index", 0) or 0)
    if round2_candidate > round2_best or (
        round2_candidate == round2_best and update_index > current_round2_best_update
    ):
        target_path = checkpoint_root / "best_by_round2_mechanism_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        best_record["best_by_round2_mechanism_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(round2_candidate),
            "score_breakdown": compute_round2_mechanism_checkpoint_score(current_metrics, policy_diagnostics),
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "policy_diagnostics": policy_diagnostics,
            "selection_protocol": selection_protocol,
            "selection_reason": {
                "selected": True,
                "selection_rule": "best_by_round2_mechanism_score",
                "selection_intent": (
                    "round2 mechanism-window policy checkpoint score; preserves reward/continuity/failure/backhaul "
                    "while adding prepare probability and guard-to-prepare diagnostics"
                ),
                "score_breakdown": compute_round2_mechanism_checkpoint_score(current_metrics, policy_diagnostics),
            },
        }
    retained_candidate = retained_mechanism_score_priority_tuple(
        current_metrics,
        policy_diagnostics,
        update_index=update_index,
        retention_start_update=retention_start_update,
    )
    retained_best = tuple(
        best_record.get("best_by_retained_mechanism_score", {}).get(
            "priority_tuple",
            (float("-inf"), float("-inf"), -1.0, -1.0, -1.0, float("-inf"), -1.0, float("-inf")),
        )
    )
    current_retained_best_update = int(
        best_record.get("best_by_retained_mechanism_score", {}).get("update_index", 0) or 0
    )
    if retained_candidate > retained_best or (
        retained_candidate == retained_best and update_index > current_retained_best_update
    ):
        target_path = checkpoint_root / "best_by_retained_mechanism_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        score_breakdown = compute_retained_mechanism_checkpoint_score(
            current_metrics,
            policy_diagnostics,
            update_index=update_index,
            retention_start_update=retention_start_update,
        )
        best_record["best_by_retained_mechanism_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(retained_candidate),
            "score_breakdown": score_breakdown,
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "policy_diagnostics": policy_diagnostics,
            "selection_protocol": selection_protocol,
            "selection_reason": {
                "selected": True,
                "selection_rule": "best_by_retained_mechanism_score",
                "selection_intent": (
                    "round3 late-training retained mechanism score; keeps reward/continuity/failure/backhaul "
                    "guardrails while preferring late-stage prepare probability stability"
                ),
                "score_breakdown": score_breakdown,
            },
        }
    popularity_metrics = dict(selection_eval.get("aggregate_by_agent", {}).get("popularity_cache_heuristic", {}))
    reward_tiebreak_candidate = reward_tiebreak_score_priority_tuple(
        current_metrics,
        reference_metrics=popularity_metrics,
        rows=list(selection_eval.get("rows", [])),
        current_agent_name=current_agent_name,
    )
    reward_tiebreak_best = tuple(
        best_record.get("best_by_reward_tiebreak_score", {}).get(
            "priority_tuple",
            (float("-inf"), -1.0, float("-inf"), float("-inf"), float("-inf"), -1.0, -1.0, float("-inf")),
        )
    )
    current_reward_tiebreak_best_update = int(
        best_record.get("best_by_reward_tiebreak_score", {}).get("update_index", 0) or 0
    )
    if reward_tiebreak_candidate > reward_tiebreak_best or (
        reward_tiebreak_candidate == reward_tiebreak_best and update_index > current_reward_tiebreak_best_update
    ):
        target_path = checkpoint_root / "best_by_reward_tiebreak_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        score_breakdown = compute_reward_tiebreak_checkpoint_score(
            current_metrics,
            reference_metrics=popularity_metrics,
            rows=list(selection_eval.get("rows", [])),
            current_agent_name=current_agent_name,
        )
        best_record["best_by_reward_tiebreak_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(reward_tiebreak_candidate),
            "score_breakdown": score_breakdown,
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "policy_diagnostics": policy_diagnostics,
            "selection_protocol": selection_protocol,
            "selection_reason": build_reward_tiebreak_selection_reason(
                candidate_metrics=current_metrics,
                popularity_metrics=popularity_metrics,
                rows=list(selection_eval.get("rows", [])),
                selected=True,
                current_agent_name=current_agent_name,
            ),
        }
    pareto_reference_agent = "ppo" if "ppo" in selection_eval.get("aggregate_by_agent", {}) else "popularity_cache_heuristic"
    pareto_reference_metrics = dict(selection_eval.get("aggregate_by_agent", {}).get(pareto_reference_agent, {}))
    pareto_safe_candidate = pareto_safe_score_priority_tuple(
        current_metrics,
        reference_metrics=pareto_reference_metrics,
        reference_agent=pareto_reference_agent,
        rows=list(selection_eval.get("rows", [])),
        current_agent_name=current_agent_name,
    )
    pareto_safe_best = tuple(
        best_record.get("best_by_pareto_safe_score", {}).get(
            "priority_tuple",
            (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), -1.0, -1.0),
        )
    )
    current_pareto_safe_best_update = int(
        best_record.get("best_by_pareto_safe_score", {}).get("update_index", 0) or 0
    )
    if pareto_safe_candidate > pareto_safe_best or (
        pareto_safe_candidate == pareto_safe_best and update_index > current_pareto_safe_best_update
    ):
        target_path = checkpoint_root / "best_by_pareto_safe_score.pt"
        shutil.copy2(checkpoint_path, target_path)
        score_breakdown = compute_pareto_safe_checkpoint_score(
            current_metrics,
            reference_metrics=pareto_reference_metrics,
            reference_agent=pareto_reference_agent,
            rows=list(selection_eval.get("rows", [])),
            current_agent_name=current_agent_name,
        )
        best_record["best_by_pareto_safe_score"] = {
            "path": str(target_path),
            "source_checkpoint_path": str(checkpoint_path),
            "priority_tuple": list(pareto_safe_candidate),
            "score_breakdown": score_breakdown,
            "update_index": update_index,
            "episode_index": episode_index,
            "metrics": current_metrics,
            "reference_agent": pareto_reference_agent if pareto_reference_metrics else "",
            "reference_metrics": pareto_reference_metrics,
            "policy_diagnostics": policy_diagnostics,
            "selection_protocol": selection_protocol,
            "selection_reason": {
                "selected": True,
                "selection_rule": "best_by_pareto_safe_score",
                "selection_intent": (
                    "v9 Pareto-safe checkpoint ranking; keeps total reward competitive while "
                    "penalizing handoff-failure and backhaul regressions against the available reference"
                ),
                "score_breakdown": score_breakdown,
            },
        }
    return best_record


def main() -> None:
    args = parse_args()
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    rng = random.Random(args.random_seed)
    workflow_states = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
    if not workflow_states:
        raise RuntimeError("???????? workflow states?")

    if args.window_plan_path:
        window_payload = apply_frozen_window_plan({}, args.window_plan_path)
    else:
        _, window_payload = resolve_window_candidates(
            root_dir=ROOT_DIR,
            mobility_source=args.mobility_source,
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root=args.lust_scenario_root,
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout=args.rsu_layout,
            frame_offset=args.frame_offset,
            window_length=args.window_length,
            window_selector=args.window_selector,
            window_count=args.train_window_count,
            window_scan_stride=args.window_scan_stride,
            random_seed=args.random_seed,
            window_mode=args.window_mode,
        )
    train_window_plan = build_training_window_plan(window_payload, args)
    if args.eval_window_plan_path:
        eval_window_payload = apply_frozen_window_plan({}, args.eval_window_plan_path)
    else:
        eval_window_payload = window_payload
    eval_window_plan = [
        dict(item) for item in eval_window_payload["selected_windows"]
    ]
    training_window_sampling_config = {
        "mechanism_window_oversample_ratio": float(args.mechanism_window_oversample_ratio),
        "handoff_imminent_oversample_ratio": float(args.handoff_imminent_oversample_ratio),
        "target_mismatch_sample_weight": float(args.target_mismatch_sample_weight),
        "min_mechanism_activating_windows": int(args.min_mechanism_activating_windows),
        "selected_window_count": len(window_payload["selected_windows"]),
        "expanded_train_window_count": len(train_window_plan),
        "eval_window_count": len(eval_window_plan),
        "frozen_window_plan_path": window_payload.get("frozen_window_plan_path", ""),
        "frozen_window_plan_protocol_version": window_payload.get("frozen_window_plan_protocol_version", ""),
        "frozen_window_plan_split": window_payload.get("frozen_window_plan_split", ""),
        "frozen_eval_window_plan_path": eval_window_payload.get("frozen_window_plan_path", ""),
        "frozen_eval_window_plan_protocol_version": eval_window_payload.get(
            "frozen_window_plan_protocol_version",
            "",
        ),
        "frozen_eval_window_plan_split": eval_window_payload.get(
            "frozen_window_plan_split",
            "",
        ),
        "update_eval_max_windows": args.update_eval_max_windows,
        "update_eval_max_workflows": args.update_eval_max_workflows,
    }

    run_id = datetime.now().strftime(f"{args.agent_name}_train_%Y%m%d_%H%M%S_%f") + f'_seed{args.random_seed}'
    output_root = Path(args.output_root) / args.agent_name / run_id
    episode_root = output_root / "episodes"
    checkpoint_root = output_root / "checkpoints"
    episode_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.agent_name, **build_agent_kwargs(args))
    adapter_catalog = AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json")

    pending_rollout: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    update_eval_history: list[dict[str, Any]] = []
    best_checkpoint_record: dict[str, Any] = {
        "latest_checkpoint_path": "",
        "policy_learning_gate": {},
        "best_by_reward": {},
        "best_by_continuity": {},
        "best_by_mechanism_balanced": {},
        "best_by_advantage_score": {},
        "best_by_mechanism_advantage_score": {},
        "best_by_round2_mechanism_score": {},
        "best_by_retained_mechanism_score": {},
        "best_by_reward_tiebreak_score": {},
        "best_by_pareto_safe_score": {},
    }
    checkpoint_paths: list[str] = []
    stability_control_history: list[dict[str, Any]] = []
    best_mechanism_score_so_far = 0.0
    update_index = 0
    counterfactual_model_query_count_total = 0
    counterfactual_model_transition_count_total = 0
    warm_start_record: dict[str, Any] = {
        "enabled": bool(args.warm_start_checkpoint_path),
        "source_checkpoint_path": args.warm_start_checkpoint_path,
        "loaded": False,
    }
    if args.warm_start_checkpoint_path:
        warm_start_source = Path(args.warm_start_checkpoint_path)
        if not warm_start_source.exists():
            raise FileNotFoundError(f"warm_start_checkpoint_path does not exist: {warm_start_source}")
        agent.load(str(warm_start_source))
        warm_start_target = checkpoint_root / "warm_start.pt"
        agent.save(str(warm_start_target))
        warm_start_metadata = {
            "run_id": run_id,
            "agent_name": args.agent_name,
            "config_profile": args.profile,
            "warm_start_source_checkpoint_path": str(warm_start_source),
            "warm_start_source_metadata": load_checkpoint_metadata(str(warm_start_source)),
            "train_window_mode": args.train_window_mode,
            "training_window_sampling_config": training_window_sampling_config,
            "primary_vehicle_selection": args.primary_vehicle_selection,
            "episodes": args.episodes,
            "update_count": 0,
            "is_smoke_checkpoint": False,
            **build_run_scale_info(args.profile, args.episodes, 0),
        }
        annotate_checkpoint_metadata(warm_start_target, warm_start_metadata)
        checkpoint_paths.append(str(warm_start_target))
        warm_start_eval = run_update_eval(
            current_agent_name=args.agent_name,
            checkpoint_path=warm_start_target,
            workflow_states=workflow_states,
            eval_windows=eval_window_plan,
            args=args,
        )
        warm_start_eval["update_index"] = 0
        warm_start_eval["episode_index"] = 0
        warm_start_eval["warm_start_eval"] = True
        warm_start_eval["current_agent_policy_diagnostics"] = dict(
            warm_start_eval.get("aggregate_policy_diagnostics_by_agent", {}).get(args.agent_name, default_policy_diagnostics())
        )
        update_eval_history.append(warm_start_eval)
        warm_start_eval["policy_learning_gate"] = build_policy_learning_gate(
            update_eval_history=update_eval_history,
            current_agent_name=args.agent_name,
        )
        best_checkpoint_record = maybe_update_best_checkpoint(
            current_agent_name=args.agent_name,
            checkpoint_path=warm_start_target,
            checkpoint_root=checkpoint_root,
            update_index=0,
            episode_index=0,
            update_eval=warm_start_eval,
            best_record=best_checkpoint_record,
            retention_start_update=0,
        )
        warm_start_record = {
            **warm_start_record,
            "loaded": True,
            "target_checkpoint_path": str(warm_start_target),
            "source_metadata": warm_start_metadata["warm_start_source_metadata"],
            "eval_metrics": dict(warm_start_eval.get("aggregate_by_agent", {}).get(args.agent_name, {})),
        }

    for episode_index in range(1, args.episodes + 1):
        workflow_state = workflow_states[(episode_index - 1) % len(workflow_states)]
        selected_window = choose_training_window(train_window_plan, episode_index=episode_index, mode=args.train_window_mode, rng=rng)
        mobility_bundle = load_window_bundle(
            root_dir=ROOT_DIR,
            mobility_source=args.mobility_source,
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root=args.lust_scenario_root,
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout=str(selected_window.get("recommended_rsu_layout", args.rsu_layout)),
            frame_offset=int(selected_window["frame_offset"]),
            window_length=int(selected_window["window_length"]),
            random_seed=args.random_seed + episode_index - 1,
        )
        mobility_bundle.rsu_metadata["window_class"] = selected_window.get("window_class", "unknown")
        recorder = EpisodeRecorder(prefetch_validation_window=6)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=mobility_bundle.provider,
            workflow_state=workflow_state,
            adapter_catalog=adapter_catalog,
            rsu_states=mobility_bundle.rsu_states,
            predictor_manager=PredictorManager(
                **build_predictor_runtime_kwargs(
                    args,
                    random_seed=args.random_seed + episode_index - 1,
                )
            ),
            max_steps=max(args.max_steps + 2, 8),
            mobility_source=args.mobility_source,
            primary_vehicle_selection=args.primary_vehicle_selection,
            reward_positive_offset=args.reward_positive_offset,
        )
        env = GymVecEnv(core_env=core_env, recorder=recorder)
        trainer = MARLOnPolicyTrainer(env=env, agent=agent, recorder=recorder, max_steps=args.max_steps, gamma=args.gamma, gae_lambda=args.gae_lambda)
        summary, rollout = trainer.collect_episode(
            run_metadata={
                "script": "scripts/train_sa_ghmappo_real_sample.py",
                "mainline": mainline_label,
                "agent_name": args.agent_name,
                "episode_index": episode_index,
                "workflow_id": workflow_state.workflow_id,
                "window_id": mobility_bundle.rsu_metadata.get("window_id"),
                "window_class": selected_window.get("window_class", "unknown"),
                "config_profile": args.profile,
                "train_window_mode": args.train_window_mode,
                "primary_vehicle_selection": args.primary_vehicle_selection,
                "reward_positive_offset": args.reward_positive_offset,
            }
        )
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
        summary["retrospective_handoff_auxiliary"] = annotate_retrospective_handoff_auxiliary_targets(
            rollout=rollout,
            env=env,
            agent=agent,
        )
        summary["reward_shaping"] = apply_temporal_reward_shaping_to_rollout(
            rollout=rollout,
            agent=agent,
            args=args,
        )
        pending_rollout.extend(rollout)

        should_update = (episode_index % max(args.update_every, 1) == 0) or (episode_index == args.episodes)
        if should_update:
            update_index += 1
            learn_info = agent.learn(pending_rollout)
            pending_rollout = []
            checkpoint_path = checkpoint_root / f"update_{update_index:04d}.pt"
            latest_checkpoint_path = checkpoint_root / "latest.pt"
            agent.save(str(checkpoint_path))
            agent.save(str(latest_checkpoint_path))
            checkpoint_metadata = {
                "run_id": run_id,
                "agent_name": args.agent_name,
                "config_profile": args.profile,
                "train_window_mode": args.train_window_mode,
                "training_window_sampling_config": training_window_sampling_config,
                "primary_vehicle_selection": args.primary_vehicle_selection,
                "reward_positive_offset": args.reward_positive_offset,
                "episodes": args.episodes,
                "update_count": update_index,
                "is_smoke_checkpoint": args.profile == "smoke",
                **build_run_scale_info(args.profile, args.episodes, update_index),
            }
            annotate_checkpoint_metadata(checkpoint_path, checkpoint_metadata)
            annotate_checkpoint_metadata(latest_checkpoint_path, checkpoint_metadata)
            checkpoint_paths.append(str(checkpoint_path))
            update_eval = run_update_eval(
                current_agent_name=args.agent_name,
                checkpoint_path=checkpoint_path,
                workflow_states=workflow_states,
                eval_windows=eval_window_plan,
                args=args,
            )
            update_eval["update_index"] = update_index
            update_eval["episode_index"] = episode_index
            update_eval["current_agent_policy_diagnostics"] = dict(
                update_eval.get("aggregate_policy_diagnostics_by_agent", {}).get(args.agent_name, default_policy_diagnostics())
            )
            update_eval_history.append(update_eval)
            update_eval["policy_learning_gate"] = build_policy_learning_gate(
                update_eval_history=update_eval_history,
                current_agent_name=args.agent_name,
            )
            best_checkpoint_record = maybe_update_best_checkpoint(
                current_agent_name=args.agent_name,
                checkpoint_path=checkpoint_path,
                checkpoint_root=checkpoint_root,
                update_index=update_index,
                episode_index=episode_index,
                update_eval=update_eval,
                best_record=best_checkpoint_record,
                retention_start_update=8 if args.profile == "sa_mechanism_retention_round3" else 0,
            )
            current_eval_metrics = dict(update_eval.get("aggregate_by_agent", {}).get(args.agent_name, {}))
            best_mechanism_score_so_far, applied_stability_control = maybe_apply_anti_collapse_controls(
                agent=agent,
                args=args,
                update_index=update_index,
                learn_info=learn_info,
                current_eval_metrics=current_eval_metrics,
                best_mechanism_score_so_far=best_mechanism_score_so_far,
                stability_control_history=stability_control_history,
            )
            update_logs.append(
                {
                    "update_index": update_index,
                    "episode_index": episode_index,
                    **learn_info,
                    "current_eval_metrics": current_eval_metrics,
                    "current_eval_mechanism_diagnostics": dict(update_eval.get("aggregate_mechanism_diagnostics_by_agent", {}).get(args.agent_name, {})),
                    "current_eval_policy_diagnostics": dict(update_eval.get("aggregate_policy_diagnostics_by_agent", {}).get(args.agent_name, default_policy_diagnostics())),
                    "applied_stability_control": applied_stability_control,
                }
            )
        else:
            learn_info = {
                "agent_name": agent.agent_name,
                "policy_update_skipped": True,
                "reason": "waiting_for_update_every",
                "pending_rollout_steps": len(pending_rollout),
            }

        summary["agent_info"] = {"agent_name": agent.agent_name, "learn_info": learn_info}
        counterfactual_model_query_count_total += int(
            summary.get("trainer_info", {}).get(
                "counterfactual_model_query_count",
                0,
            )
            or 0
        )
        counterfactual_model_transition_count_total += int(
            summary.get("trainer_info", {}).get(
                "counterfactual_model_transition_count",
                0,
            )
            or 0
        )
        (episode_root / f"episode_{episode_index:04d}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        episode_metric = build_episode_metric(summary=summary, episode_index=episode_index, updated=should_update)
        episode_metrics.append(episode_metric)
        print(
            f"episode={episode_index} workflow={workflow_state.workflow_id} window={mobility_bundle.rsu_metadata.get('window_id')} class={selected_window.get('window_class', 'unknown')} "
            f"reward={episode_metric['total_reward']:.3f} continuity={episode_metric['workflow_continuity_rate']:.3f} "
            f"ready={episode_metric['handoff_ready_ratio']:.3f} prefetch={episode_metric['predictive_prefetch_precision']:.3f} "
            f"update={not learn_info.get('policy_update_skipped', False)}"
        )
        print(
            f"  reward_diag service={episode_metric['service_reward']:.3f} delay={episode_metric['delay_penalty']:.3f} cache_miss={episode_metric['cache_miss_penalty']:.3f} "
            f"migration={episode_metric['migration_cost']:.3f} continuity_bonus={episode_metric['continuity_bonus']:.3f} "
            f"mech_bonus={episode_metric.get('mechanism_exploration_bonus', 0.0):.3f}"
        )
        print(
            f"  mechanism_diag prefetch_req={episode_metric['prefetch_request_rate']:.3f} validated_hit={episode_metric['prefetch_validated_hit_rate']:.3f} "
            f"migration_prepare={episode_metric['migration_prepare_rate']:.3f} handoff_ready={episode_metric['handoff_ready_rate']:.3f} "
            f"realized={episode_metric['mechanism_realization_rate']:.3f}"
        )
        reward_shaping = summary.get("reward_shaping", {})
        if reward_shaping.get("enabled", False):
            print(
                f"  shaping reward_delta_mean={float(reward_shaping.get('reward_delta_mean', 0.0)):.3f} "
                f"reward_delta_sum={float(reward_shaping.get('reward_delta_sum', 0.0)):.3f} "
                f"timing_score={float(reward_shaping.get('mean_timing_window_score', 0.0)):.3f}"
            )
        if should_update and update_eval_history:
            current_eval_metrics = update_eval_history[-1].get("aggregate_by_agent", {}).get(args.agent_name, {})
            current_eval_mechanism = update_eval_history[-1].get("aggregate_mechanism_diagnostics_by_agent", {}).get(args.agent_name, {})
            current_eval_policy_diag = update_eval_history[-1].get("aggregate_policy_diagnostics_by_agent", {}).get(args.agent_name, {})
            current_eval_alignment_samples = update_eval_history[-1].get("policy_alignment_samples_by_agent", {}).get(args.agent_name, [])
            print(
                f"  learn_info entropy={float(learn_info.get('policy_entropy', 0.0) or 0.0):.6f} "
                f"approx_kl={float(learn_info.get('approx_kl', 0.0) or 0.0):.6f} "
                f"clip_fraction={float(learn_info.get('clip_fraction', 0.0) or 0.0):.6f} "
                f"explained_variance={float(learn_info.get('explained_variance', 0.0) or 0.0):.6f} "
                f"target_kl={float(learn_info.get('target_kl', 0.0) or 0.0):.6f}"
            )
            print(
                f"  mechanism_aux loss={float(learn_info.get('mechanism_aux_loss_mean', 0.0) or 0.0):.6f} "
                f"guided={int(learn_info.get('mechanism_guided_action_count', 0) or 0)} "
                f"weighted_ratio={float(learn_info.get('weighted_mechanism_transition_ratio', 0.0) or 0.0):.3f} "
                f"retention={bool(learn_info.get('mechanism_retention_active', False))} "
                f"effective_coef={float(learn_info.get('effective_mechanism_aux_coef', 0.0) or 0.0):.3f} "
                f"effective_weight={float(learn_info.get('effective_mechanism_window_weight', 0.0) or 0.0):.3f} "
                f"prepare_prob_before={float(learn_info.get('mechanism_guided_event_prepare_prob_before_update', 0.0) or 0.0):.3f} "
                f"prepare_prob_after={float(learn_info.get('mechanism_guided_event_prepare_prob_after_update', 0.0) or 0.0):.3f} "
                f"mech_entropy={float(learn_info.get('mechanism_head_entropy', 0.0) or 0.0):.3f}"
            )
            print(
                f"  update_eval reward={float(current_eval_metrics.get('total_reward', 0.0)):.3f} continuity={float(current_eval_metrics.get('workflow_continuity_rate', 0.0)):.3f} "
                f"prefetch_req={float(current_eval_mechanism.get('prefetch_request_rate', 0.0)):.3f} validated_hit={float(current_eval_mechanism.get('prefetch_validated_hit_rate', 0.0)):.3f} "
                f"migration_prepare={float(current_eval_mechanism.get('migration_prepare_rate', 0.0)):.3f} handoff_ready={float(current_eval_mechanism.get('handoff_ready_rate', 0.0)):.3f} "
                f"realized={float(current_eval_mechanism.get('mechanism_realization_rate', 0.0)):.3f}"
            )
            print(
                f"  timing_diag stoch_prepare={float(current_eval_policy_diag.get('stochastic_event_prepare_rate', 0.0)):.3f} "
                f"det_prepare={float(current_eval_policy_diag.get('deterministic_event_prepare_rate', 0.0)):.3f} "
                f"gap={float(current_eval_policy_diag.get('gap_event_prepare_rate', 0.0)):.3f} "
                f"gt_handoff_rate={float(current_eval_policy_diag.get('gt_handoff_opportunity_rate', 0.0)):.3f} "
                f"predictor_invoked_rate={float(current_eval_policy_diag.get('predictor_invoked_rate', 0.0)):.3f} "
                f"valid_target_rate={float(current_eval_policy_diag.get('valid_handoff_target_rate', 0.0)):.3f} "
                f"timing_active={int(round(float(current_eval_policy_diag.get('timing_active_step_count', 0.0))))} "
                f"high_prepare={int(round(float(current_eval_policy_diag.get('high_prepare_step_count', 0.0))))}"
            )
            print(
                f"  primary_align first_match_rate={float(current_eval_policy_diag.get('first_vehicle_matches_primary_rate', 0.0)):.3f} "
                f"policy_current_rsu_rate={float(current_eval_policy_diag.get('policy_current_rsu_non_null_rate', 0.0)):.3f} "
                f"gt_current_rsu_rate={float(current_eval_policy_diag.get('gt_current_rsu_non_null_rate', 0.0)):.3f} "
                f"lookup_fallback_rate={float(current_eval_policy_diag.get('primary_vehicle_lookup_fallback_rate', 0.0)):.3f}"
            )
            print(
                f"  prediction_diag raw_candidate_rate={float(current_eval_policy_diag.get('raw_handoff_candidate_rate', 0.0)):.3f} "
                f"gate_pass_rate={float(current_eval_policy_diag.get('gate_pass_rate', 0.0)):.3f} "
                f"conf_mean={float(current_eval_policy_diag.get('prediction_confidence_mean', 0.0)):.3f} "
                f"unc_mean={float(current_eval_policy_diag.get('prediction_uncertainty_mean', 0.0)):.3f} "
                f"gate_mean={float(current_eval_policy_diag.get('prediction_gate_value_mean', 0.0)):.3f}"
            )
            print(
                f"  prediction_break no_candidate={int(round(float(current_eval_policy_diag.get('invalid_reason_no_candidate_count', 0.0))))} "
                f"missing_state={int(round(float(current_eval_policy_diag.get('candidate_block_reason_missing_prediction_state_count', 0.0))))} "
                f"no_next={int(round(float(current_eval_policy_diag.get('candidate_block_reason_no_next_rsu_count', 0.0))))} "
                f"same_rsu={int(round(float(current_eval_policy_diag.get('candidate_block_reason_same_rsu_count', 0.0))))} "
                f"eta_outside={int(round(float(current_eval_policy_diag.get('candidate_block_reason_eta_outside_window_count', 0.0))))} "
                f"low_conf={int(round(float(current_eval_policy_diag.get('invalid_reason_low_confidence_count', 0.0))))} "
                f"high_unc={int(round(float(current_eval_policy_diag.get('invalid_reason_high_uncertainty_count', 0.0))))} "
                f"gate_below={int(round(float(current_eval_policy_diag.get('invalid_reason_gate_below_threshold_count', 0.0))))}"
            )
            print(
                f"  sequence_diag gt_eta_mean={float(current_eval_policy_diag.get('gt_first_handoff_eta_mean', 0.0)):.3f} "
                f"gt_eta_p25={float(current_eval_policy_diag.get('gt_first_handoff_eta_p25', 0.0)):.3f} "
                f"gt_eta_p75={float(current_eval_policy_diag.get('gt_first_handoff_eta_p75', 0.0)):.3f} "
                f"pred_other_rate={float(current_eval_policy_diag.get('pred_first_non_current_rsu_rate', 0.0)):.3f} "
                f"pred_eta_mean={float(current_eval_policy_diag.get('pred_first_non_current_rsu_eta_mean', 0.0)):.3f} "
                f"all_null={int(round(float(current_eval_policy_diag.get('predicted_sequence_all_null_count', 0.0))))} "
                f"all_current={int(round(float(current_eval_policy_diag.get('predicted_sequence_all_current_rsu_count', 0.0))))} "
                f"contains_other={int(round(float(current_eval_policy_diag.get('predicted_sequence_contains_other_rsu_count', 0.0))))} "
                f"match={int(round(float(current_eval_policy_diag.get('gt_pred_next_rsu_match_count', 0.0))))} "
                f"mismatch={int(round(float(current_eval_policy_diag.get('gt_pred_next_rsu_mismatch_count', 0.0))))}"
            )
            print(
                f"  timing_state urgency_mean={float(current_eval_policy_diag.get('temporal_urgency_mean', 0.0)):.3f} "
                f"urgency_p75={float(current_eval_policy_diag.get('temporal_urgency_p75', 0.0)):.3f} "
                f"countdown_mean={float(current_eval_policy_diag.get('countdown_steps_mean', 0.0)):.3f} "
                f"countdown_p25={float(current_eval_policy_diag.get('countdown_steps_p25', 0.0)):.3f} "
                f"countdown_p75={float(current_eval_policy_diag.get('countdown_steps_p75', 0.0)):.3f}"
            )
            print(
                f"  timing_border prob_mean={float(current_eval_policy_diag.get('event_prepare_prob_mean', 0.0)):.3f} "
                f"prob_p75={float(current_eval_policy_diag.get('event_prepare_prob_p75', 0.0)):.3f} "
                f"margin_mean={float(current_eval_policy_diag.get('event_margin_mean', 0.0)):.3f} "
                f"margin_p75={float(current_eval_policy_diag.get('event_margin_p75', 0.0)):.3f} "
                f"override={int(round(float(current_eval_policy_diag.get('override_trigger_count', 0.0))))} "
                f"smoothing={int(round(float(current_eval_policy_diag.get('borderline_trigger_count', 0.0))))}"
            )
            for sample in list(current_eval_alignment_samples[:5]):
                print(
                    f"  alignment_sample step={int(sample.get('step_index', 0) or 0)} "
                    f"window={sample.get('window_id')} workflow={sample.get('workflow_id')} "
                    f"primary_vehicle={sample.get('primary_vehicle_id')} first_vehicle={sample.get('first_vehicle_id')} "
                    f"gt_current_rsu={sample.get('gt_current_rsu')} policy_current_rsu={sample.get('policy_current_rsu')} "
                    f"gt_next={sample.get('gt_first_next_rsu')} gt_eta={int(sample.get('gt_eta', 0) or 0)} "
                    f"pred_seq={sample.get('predicted_sequence')} "
                    f"pred_next={sample.get('predicted_first_non_current_rsu')} pred_eta={int(sample.get('pred_eta', 0) or 0)}"
                )
            latest_control = update_logs[-1].get("applied_stability_control") if update_logs else None
            if latest_control:
                print(
                    f"  stability_control reason={latest_control.get('reason')} lr={float(latest_control.get('applied_controls', {}).get('learning_rate', 0.0)):.6f} "
                    f"clip={float(latest_control.get('applied_controls', {}).get('clip_ratio', 0.0)):.4f} aux={float(latest_control.get('applied_controls', {}).get('auxiliary_coef', 0.0)):.4f} "
                    f"mech_bias={float(latest_control.get('applied_controls', {}).get('mechanism_logit_bias_strength', 0.0)):.4f}"
                )

    training_audit = build_training_audit(update_logs=update_logs, update_eval_history=update_eval_history, current_agent_name=args.agent_name)
    run_scale_info = build_run_scale_info(args.profile, args.episodes, update_index)
    checkpoint_consistency_audit = run_checkpoint_consistency_audit(
        current_agent_name=args.agent_name,
        checkpoint_root=checkpoint_root,
        workflow_states=workflow_states,
        eval_windows=eval_window_plan,
        args=args,
        best_record=best_checkpoint_record,
    )
    audit_scope = dict(checkpoint_consistency_audit.get("audit_scope", {}) or {})
    if bool(audit_scope.get("repair_best_record_eligible", False)):
        best_checkpoint_record, best_record_repaired = repair_best_checkpoint_record_from_audit(
            checkpoint_root=checkpoint_root,
            best_record=best_checkpoint_record,
            audit_payload=checkpoint_consistency_audit,
        )
        if best_record_repaired:
            checkpoint_consistency_audit = run_checkpoint_consistency_audit(
                current_agent_name=args.agent_name,
                checkpoint_root=checkpoint_root,
                workflow_states=workflow_states,
                eval_windows=eval_window_plan,
                args=args,
                best_record=best_checkpoint_record,
            )
    else:
        best_record_repaired = False
        checkpoint_consistency_audit["best_record_repair_skipped"] = {
            "skipped": True,
            "reason": "post_training_audit_mode_is_not_full_protocol",
            "audit_mode": str(audit_scope.get("mode", "unknown")),
        }
    mechanism_collapse_audit = build_mechanism_collapse_audit(update_eval_history=update_eval_history, current_agent_name=args.agent_name)
    checkpoint_family_eval = build_checkpoint_family_eval(
        current_agent_name=args.agent_name,
        checkpoint_root=checkpoint_root,
        workflow_states=workflow_states,
        eval_windows=eval_window_plan,
        args=args,
    )
    policy_diagnostic_history_summary = summarize_policy_diagnostic_history(update_eval_history=update_eval_history, agent_name=args.agent_name)
    train_summary = {
        "run_id": run_id,
        "mainline": mainline_label,
        "agent_name": args.agent_name,
        "profile": args.profile,
        "config_profile": args.profile,
        "experiment_run_type": run_scale_info["experiment_run_type"],
        "paper_claim_ready": run_scale_info["paper_claim_ready"],
        "scope_note": run_scale_info["scope_note"],
        "train_window_mode": args.train_window_mode,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "reward_protocol": {
            "reward_positive_offset": float(args.reward_positive_offset),
            "offset_free": abs(float(args.reward_positive_offset)) <= 1e-12,
            "offset_adjusted_total_reward_reported": True,
        },
        "window_mode": args.window_mode,
        "training_window_sampling_config": training_window_sampling_config,
        "episodes": args.episodes,
        "update_every": args.update_every,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "auxiliary_coef": args.auxiliary_coef,
        "counterfactual_model_rollout": {
            "enabled": bool(
                getattr(agent, "_env_action_model_rollout_enabled", False)
                or getattr(
                    agent,
                    "_option_counterfactual_model_rollout_enabled",
                    False,
                )
            ),
            "protocol": (
                "digital_twin_multihorizon_env_action_td_v2"
                if getattr(agent, "_env_action_model_rollout_enabled", False)
                else "digital_twin_multistep_option_td_v2"
            ),
            "horizon": int(
                getattr(agent, "_env_action_model_rollout_horizon", 1)
                if getattr(agent, "_env_action_model_rollout_enabled", False)
                else getattr(
                    agent,
                    "_option_counterfactual_model_rollout_horizon",
                    1,
                )
            ),
            "horizons": list(
                getattr(
                    agent,
                    "_env_action_model_rollout_horizons",
                    (
                        getattr(
                            agent,
                            "_env_action_model_rollout_horizon",
                            1,
                        ),
                    ),
                )
            ),
            "imagination_replay_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_imagination_replay_enabled",
                    False,
                )
            ),
            "imagination_replay_depths": list(
                getattr(
                    agent,
                    "_env_action_model_imagination_replay_depths",
                    (),
                )
            ),
            "imagination_replay_horizons": list(
                getattr(
                    agent,
                    "_env_action_model_imagination_replay_horizons",
                    (),
                )
            ),
            "imagination_replay_recovery_only": bool(
                getattr(
                    agent,
                    "_env_action_model_imagination_replay_recovery_only",
                    False,
                )
            ),
            "outcome_memory_fusion_enabled": bool(
                getattr(agent, "_outcome_memory_fusion_enabled", False)
            ),
            "robust_horizon_policy_improvement_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_robust_horizons_enabled",
                    False,
                )
            ),
            "horizon_risk_coef": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_horizon_risk_coef",
                    0.0,
                )
            ),
            "horizon_aggregation_mode": str(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_horizon_aggregation_mode",
                    "mean_std",
                )
            ),
            "horizon_lambda": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_horizon_lambda",
                    0.0,
                )
            ),
            "adaptive_kl_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_adaptive_kl_enabled",
                    False,
                )
            ),
            "policy_improvement_target_kl": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_target_kl",
                    0.0,
                )
            ),
            "regret_adaptive_kl_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_regret_adaptive_kl_enabled",
                    False,
                )
            ),
            "max_policy_improvement_target_kl": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_max_target_kl",
                    0.0,
                )
            ),
            "counterfactual_regret_priority_coef": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_regret_priority_coef",
                    0.0,
                )
            ),
            "tail_distillation_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_distillation_enabled",
                    False,
                )
            ),
            "tail_distillation_quantile": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_quantile",
                    0.0,
                )
            ),
            "tail_distillation_min_regret": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_min_regret",
                    0.0,
                )
            ),
            "tail_distillation_epochs": int(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_epochs",
                    0,
                )
            ),
            "tail_distillation_coef": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_coef",
                    0.0,
                )
            ),
            "tail_distillation_max_policy_kl": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_max_policy_kl",
                    0.0,
                )
            ),
            "tail_residual_optimizer_enabled": bool(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_residual_optimizer_enabled",
                    False,
                )
            ),
            "tail_residual_learning_rate": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_residual_learning_rate",
                    0.0,
                )
            ),
            "tail_residual_backtrack_factor": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_residual_backtrack_factor",
                    0.0,
                )
            ),
            "tail_residual_min_learning_rate": float(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_residual_min_learning_rate",
                    0.0,
                )
            ),
            "tail_residual_max_backtracks": int(
                getattr(
                    agent,
                    "_env_action_model_policy_improvement_tail_residual_max_backtracks",
                    0,
                )
            ),
            "model_query_count": counterfactual_model_query_count_total,
            "model_transition_count": counterfactual_model_transition_count_total,
        },
        "warm_start": warm_start_record,
        "mechanism_aux_config": {
            "mechanism_aux_coef": float(getattr(agent, "_mechanism_aux_coef", 0.0)),
            "mechanism_window_weight": float(getattr(agent, "_mechanism_window_weight", 1.0)),
            "prepare_action_prior_weight": float(getattr(agent, "_prepare_action_prior_weight", 0.0)),
            "mechanism_entropy_coef": float(getattr(agent, "_mechanism_entropy_coef", 0.0)),
            "mechanism_retention_start_update": int(getattr(agent, "_mechanism_retention_start_update", 0)),
            "mechanism_aux_coef_floor_after_update": float(
                getattr(agent, "_mechanism_aux_coef_floor_after_update", 0.0)
            ),
            "mechanism_window_weight_floor_after_update": float(
                getattr(agent, "_mechanism_window_weight_floor_after_update", 1.0)
            ),
            "mechanism_entropy_floor_after_update": float(
                getattr(agent, "_mechanism_entropy_floor_after_update", 0.0)
            ),
            "mechanism_aux_current_cache_fill_enabled": bool(
                getattr(agent, "_mechanism_aux_current_cache_fill_enabled", True)
            ),
            "mechanism_credit_prd_enabled": bool(getattr(agent, "_mechanism_credit_prd_enabled", False)),
            "mechanism_credit_policy_coef": float(getattr(agent, "_mechanism_credit_policy_coef", 0.0)),
            "mechanism_credit_event_coef": float(getattr(agent, "_mechanism_credit_event_coef", 0.0)),
            "mechanism_credit_option_coef": float(getattr(agent, "_mechanism_credit_option_coef", 0.0)),
            "mechanism_credit_clip": float(getattr(agent, "_mechanism_credit_clip", 0.0)),
            "mechanism_credit_success_bonus": float(getattr(agent, "_mechanism_credit_success_bonus", 0.0)),
            "mechanism_credit_prepare_bonus": float(getattr(agent, "_mechanism_credit_prepare_bonus", 0.0)),
            "mechanism_credit_ready_bonus": float(getattr(agent, "_mechanism_credit_ready_bonus", 0.0)),
            "mechanism_credit_prefetch_hit_bonus": float(
                getattr(agent, "_mechanism_credit_prefetch_hit_bonus", 0.0)
            ),
            "mechanism_credit_miss_penalty": float(getattr(agent, "_mechanism_credit_miss_penalty", 0.0)),
            "mechanism_credit_false_positive_penalty": float(
                getattr(agent, "_mechanism_credit_false_positive_penalty", 0.0)
            ),
            "mechanism_credit_min_context": float(getattr(agent, "_mechanism_credit_min_context", 0.0)),
            "mechanism_focal_aux_enabled": bool(getattr(agent, "_mechanism_focal_aux_enabled", False)),
            "mechanism_focal_gamma": float(getattr(agent, "_mechanism_focal_gamma", 0.0)),
            "digital_twin_handoff_fusion_enabled": bool(
                getattr(agent, "_digital_twin_handoff_fusion_enabled", False)
            ),
            "digital_twin_handoff_slow_scale": float(
                getattr(agent, "_digital_twin_handoff_slow_scale", 0.0)
            ),
            "digital_twin_handoff_fast_scale": float(
                getattr(agent, "_digital_twin_handoff_fast_scale", 0.0)
            ),
            "digital_twin_handoff_event_scale": float(
                getattr(agent, "_digital_twin_handoff_event_scale", 0.0)
            ),
            "digital_twin_handoff_critic_scale": float(
                getattr(agent, "_digital_twin_handoff_critic_scale", 0.0)
            ),
            "backhaul_guard_enabled": bool(getattr(agent, "_backhaul_guard_enabled", False)),
            "backhaul_guard_max_reactive_fills_per_adapter": int(
                getattr(agent, "_backhaul_guard_max_reactive_fills_per_adapter", 0)
            ),
            "cache_warm_start_guard_enabled": bool(
                getattr(agent, "_cache_warm_start_guard_enabled", False)
            ),
            "cache_warm_start_guard_min_countdown": float(
                getattr(agent, "_cache_warm_start_guard_min_countdown", 0.0)
            ),
            "cache_warm_start_guard_max_prefetch_countdown": float(
                getattr(agent, "_cache_warm_start_guard_max_prefetch_countdown", 0.0)
            ),
            "predictive_prefetch_admission_guard_enabled": bool(
                getattr(agent, "_predictive_prefetch_admission_guard_enabled", False)
            ),
            "predictive_prefetch_admission_min_confidence": float(
                getattr(agent, "_predictive_prefetch_admission_min_confidence", 0.0)
            ),
            "predictive_prefetch_admission_require_distinct_next": bool(
                getattr(agent, "_predictive_prefetch_admission_require_distinct_next", True)
            ),
        },
        "predictor_runtime_config": build_predictor_runtime_kwargs(args, random_seed=args.random_seed),
        "reward_shaping_config": build_temporal_reward_shaping_config(agent=agent, args=args),
        "update_count": update_index,
        "workflow_ids": [workflow_state.workflow_id for workflow_state in workflow_states],
        "selected_window_plan": train_window_plan,
        "evaluation_window_plan": eval_window_plan,
        "evaluation_protocol": {
            "protocol_name": "all_selected_windows_x_all_selected_workflows_deterministic",
            "deterministic_eval": True,
            "selection_eval_mode": "safety_projected",
            "policy_learning_gate_eval_mode": "raw_policy",
            "raw_policy_definition": (
                "learned_actor_and_model_critic_policy_improvement_plus_action_mask_"
                "without_fixed_policy_adjustments_or_safety_rules"
            ),
            "eval_window_ids": [item.get("window_id") for item in eval_window_plan],
            "workflow_ids": [workflow_state.workflow_id for workflow_state in workflow_states],
        },
        "post_training_audit_config": {
            "mode": str(getattr(args, "post_training_audit_mode", "full")),
            "max_windows": getattr(args, "post_training_audit_max_windows", None),
            "max_workflows": getattr(args, "post_training_audit_max_workflows", None),
            "checkpoint_consistency_audit_scope": checkpoint_consistency_audit.get("audit_scope", {}),
        },
        "output_dir": str(output_root),
        "latest_checkpoint_path": best_checkpoint_record.get("latest_checkpoint_path", ""),
        "latest_checkpoint_metadata": load_checkpoint_metadata(best_checkpoint_record.get("latest_checkpoint_path", "")) if best_checkpoint_record.get("latest_checkpoint_path") else {},
        "best_by_reward_path": best_checkpoint_record.get("best_by_reward", {}).get("path", ""),
        "best_by_continuity_path": best_checkpoint_record.get("best_by_continuity", {}).get("path", ""),
        "best_by_mechanism_balanced_path": best_checkpoint_record.get("best_by_mechanism_balanced", {}).get("path", ""),
        "best_by_advantage_score_path": best_checkpoint_record.get("best_by_advantage_score", {}).get("path", ""),
        "best_by_mechanism_advantage_score_path": best_checkpoint_record.get("best_by_mechanism_advantage_score", {}).get("path", ""),
        "best_by_round2_mechanism_score_path": best_checkpoint_record.get("best_by_round2_mechanism_score", {}).get("path", ""),
        "best_by_retained_mechanism_score_path": best_checkpoint_record.get("best_by_retained_mechanism_score", {}).get("path", ""),
        "best_by_reward_tiebreak_score_path": best_checkpoint_record.get("best_by_reward_tiebreak_score", {}).get("path", ""),
        "best_by_pareto_safe_score_path": best_checkpoint_record.get("best_by_pareto_safe_score", {}).get("path", ""),
        "best_by_advantage_score_selection_reason": best_checkpoint_record.get("best_by_advantage_score", {}).get("selection_reason", {}),
        "best_by_mechanism_advantage_score_selection_reason": best_checkpoint_record.get("best_by_mechanism_advantage_score", {}).get("selection_reason", {}),
        "best_by_round2_mechanism_score_selection_reason": best_checkpoint_record.get("best_by_round2_mechanism_score", {}).get("selection_reason", {}),
        "best_by_retained_mechanism_score_selection_reason": best_checkpoint_record.get("best_by_retained_mechanism_score", {}).get("selection_reason", {}),
        "best_by_reward_tiebreak_score_selection_reason": best_checkpoint_record.get("best_by_reward_tiebreak_score", {}).get("selection_reason", {}),
        "best_by_pareto_safe_score_selection_reason": best_checkpoint_record.get("best_by_pareto_safe_score", {}).get("selection_reason", {}),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_consistency_audit_path": str(output_root / "checkpoint_consistency_audit.json"),
        "mechanism_collapse_audit_path": str(output_root / "mechanism_collapse_audit.json"),
        "checkpoint_family_eval_path": str(output_root / "checkpoint_family_eval.json"),
        "mean_metrics": aggregate_metrics(episode_metrics),
        "mean_reward_breakdown": aggregate_reward_breakdown(episode_metrics),
        "mean_mechanism_diagnostics": {
            field_name: round(fmean(float(item.get(field_name, 0.0)) for item in episode_metrics), 6) if episode_metrics else 0.0
            for field_name in MECHANISM_DIAGNOSTIC_FIELDS
        },
        **policy_diagnostic_history_summary,
        "episode_metrics": episode_metrics,
        "update_logs": update_logs,
        "stability_control_history": stability_control_history,
        "training_effectiveness_audit": training_audit,
        "policy_learning_gate": build_policy_learning_gate(
            update_eval_history=update_eval_history,
            current_agent_name=args.agent_name,
        ),
        "mechanism_collapse_audit": mechanism_collapse_audit,
    }
    (output_root / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "update_eval_history.json").write_text(json.dumps(update_eval_history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "best_checkpoint_record.json").write_text(json.dumps(best_checkpoint_record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "training_effectiveness_audit.json").write_text(json.dumps(training_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "checkpoint_consistency_audit.json").write_text(json.dumps(checkpoint_consistency_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "mechanism_collapse_audit.json").write_text(json.dumps(mechanism_collapse_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "checkpoint_family_eval.json").write_text(json.dumps(checkpoint_family_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    print("????")
    print(f"run_id: {run_id}")
    print(f"profile: {args.profile}")
    print(f"experiment_run_type: {run_scale_info['experiment_run_type']}")
    print(f"paper_claim_ready: {run_scale_info['paper_claim_ready']}")
    print(f"train_window_mode: {args.train_window_mode}")
    print(f"output_dir: {output_root}")
    print(f"best_by_reward_path: {best_checkpoint_record.get('best_by_reward', {}).get('path', '')}")
    print(f"best_by_continuity_path: {best_checkpoint_record.get('best_by_continuity', {}).get('path', '')}")
    print(f"best_by_mechanism_balanced_path: {best_checkpoint_record.get('best_by_mechanism_balanced', {}).get('path', '')}")
    print(f"best_by_advantage_score_path: {best_checkpoint_record.get('best_by_advantage_score', {}).get('path', '')}")
    print(f"best_by_mechanism_advantage_score_path: {best_checkpoint_record.get('best_by_mechanism_advantage_score', {}).get('path', '')}")
    print(f"best_by_round2_mechanism_score_path: {best_checkpoint_record.get('best_by_round2_mechanism_score', {}).get('path', '')}")
    print(f"best_by_retained_mechanism_score_path: {best_checkpoint_record.get('best_by_retained_mechanism_score', {}).get('path', '')}")
    print(f"best_by_reward_tiebreak_score_path: {best_checkpoint_record.get('best_by_reward_tiebreak_score', {}).get('path', '')}")
    print(f"best_by_pareto_safe_score_path: {best_checkpoint_record.get('best_by_pareto_safe_score', {}).get('path', '')}")
    print(f"best_record_repaired: {best_record_repaired}")
    print(f"collapse_detected: {bool(mechanism_collapse_audit.get('collapse_detected', False))}")
    print(f"collapse_start_update: {int(mechanism_collapse_audit.get('collapse_start_update', 0) or 0)}")
    print(f"mean_total_reward: {train_summary['mean_metrics'].get('total_reward', 0.0):.3f}")
    print(f"mean_continuity: {train_summary['mean_metrics'].get('workflow_continuity_rate', 0.0):.3f}")
    print(f"mean_prefetch_request_rate: {train_summary['mean_mechanism_diagnostics'].get('prefetch_request_rate', 0.0):.3f}")
    print(f"mean_prefetch_validated_hit_rate: {train_summary['mean_mechanism_diagnostics'].get('prefetch_validated_hit_rate', 0.0):.3f}")
    print(f"mean_migration_prepare_rate: {train_summary['mean_mechanism_diagnostics'].get('migration_prepare_rate', 0.0):.3f}")
    print(f"mean_handoff_ready_rate: {train_summary['mean_mechanism_diagnostics'].get('handoff_ready_rate', 0.0):.3f}")
    print(f"mean_mechanism_realization_rate: {train_summary['mean_mechanism_diagnostics'].get('mechanism_realization_rate', 0.0):.3f}")


if __name__ == "__main__":
    main()
