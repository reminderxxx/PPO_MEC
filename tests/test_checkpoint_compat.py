"""checkpoint 向后兼容辅助逻辑测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.real_eval_support import (
    _filter_checkpoint_config,
    _infer_prediction_feature_dim_from_payload,
)


class CheckpointCompatTestCase(unittest.TestCase):
    """验证旧 checkpoint 可从权重形状恢复关键 encoder 参数。"""

    def test_infer_legacy_prediction_feature_dim(self) -> None:
        payload = {
            "network_state_dict": {
                "encoder._prediction_projection.0.weight": torch.zeros(64, 8),
            },
        }

        self.assertEqual(_infer_prediction_feature_dim_from_payload(payload), 8)

    def test_missing_prediction_projection_returns_none(self) -> None:
        self.assertIsNone(_infer_prediction_feature_dim_from_payload({"network_state_dict": {}}))

    def test_sa_v29_checkpoint_config_preserves_dt_fusion_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "digital_twin_handoff_fusion_enabled": True,
                "digital_twin_handoff_slow_scale": 0.38,
                "digital_twin_handoff_fast_scale": 0.48,
                "digital_twin_handoff_event_scale": 0.95,
                "digital_twin_handoff_critic_scale": 0.78,
                "digital_twin_policy_prior_enabled": True,
                "digital_twin_policy_prior_logit_bias": 3.2,
                "digital_twin_policy_prior_distill_coef": 0.095,
                "digital_twin_policy_prior_pacing_enabled": True,
                "digital_twin_policy_prior_pacing_fast_scale": 1.35,
                "digital_twin_policy_prior_env_action_bias_enabled": True,
                "digital_twin_policy_prior_env_action_logit_bias": 4.8,
                "digital_twin_policy_prior_continuation_threshold": 0.32,
                "digital_twin_policy_prior_continuation_prepare_scale": 1.35,
                "digital_twin_policy_prior_continuation_wait_scale": 0.95,
                "digital_twin_policy_prior_continuation_steady_suppression": 0.42,
                "digital_twin_policy_prior_adaptive_wait_enabled": True,
                "digital_twin_policy_prior_wait_ready_threshold": 0.5,
                "digital_twin_policy_prior_wait_timing_ceiling": 0.58,
                "digital_twin_policy_prior_wait_cache_ready_scale": 1.42,
                "digital_twin_policy_prior_prepare_not_ready_scale": 0.92,
                "env_action_ppo_enabled": True,
                "env_action_ppo_coef": 0.72,
                "env_action_ppo_advantage_blend": 0.54,
                "env_action_ppo_teacher_coef": 0.34,
                "env_action_ppo_mechanism_focus": 0.65,
                "env_action_ppo_max_weight": 2.25,
                "env_action_ppo_ratio_barrier_coef": 0.045,
                "env_action_ppo_ratio_barrier_margin": 0.32,
                "env_action_counterfactual_margin_enabled": True,
                "env_action_counterfactual_margin_coef": 0.18,
                "env_action_counterfactual_margin_min_gap": 0.04,
                "env_action_counterfactual_margin_max_weight": 2.2,
                "env_action_counterfactual_margin_advantage_gate": 0.12,
                "env_action_counterfactual_margin_advantage_blend": 0.7,
                "argmax_margin_regularization_enabled": True,
                "argmax_margin_coef": 0.76,
                "argmax_margin_min_gap": 0.42,
                "argmax_margin_max_weight": 4.8,
                "argmax_margin_tail_risk_threshold": 0.04,
                "argmax_margin_mechanism_penalty_scale": 1.65,
                "delayed_mechanism_credit_enabled": True,
                "delayed_mechanism_credit_policy_coef": 0.46,
                "delayed_mechanism_credit_event_coef": 1.05,
                "delayed_mechanism_credit_horizon": 5,
                "delayed_mechanism_credit_decay": 0.72,
                "delayed_mechanism_credit_clip": 1.8,
                "delayed_mechanism_credit_ready_bonus": 1.35,
                "delayed_mechanism_credit_success_bonus": 0.88,
                "delayed_mechanism_credit_failure_penalty": 1.02,
                "delayed_mechanism_credit_missed_prepare_scale": 0.62,
                "delayed_mechanism_credit_stale_penalty": 0.42,
                "delayed_mechanism_credit_context_gate": 0.24,
                "advantage_weighted_behavior_regularization_enabled": True,
                "advantage_weighted_behavior_coef": 0.24,
                "advantage_weighted_behavior_positive_coef": 1.08,
                "advantage_weighted_behavior_negative_coef": 0.92,
                "advantage_weighted_behavior_temperature": 0.62,
                "advantage_weighted_behavior_max_weight": 2.2,
                "advantage_weighted_behavior_positive_gate": 0.08,
                "advantage_weighted_behavior_negative_gate": 0.04,
                "advantage_weighted_behavior_mechanism_scale": 1.35,
                "mechanism_credit_prd_enabled": True,
                "mechanism_credit_event_coef": 0.98,
                "mechanism_focal_aux_enabled": True,
                "mechanism_focal_gamma": 1.4,
                "encoder_kind": "graph",
            },
        )

        self.assertTrue(filtered["digital_twin_handoff_fusion_enabled"])
        self.assertEqual(filtered["digital_twin_handoff_event_scale"], 0.95)
        self.assertTrue(filtered["digital_twin_policy_prior_enabled"])
        self.assertEqual(filtered["digital_twin_policy_prior_logit_bias"], 3.2)
        self.assertEqual(filtered["digital_twin_policy_prior_distill_coef"], 0.095)
        self.assertTrue(filtered["digital_twin_policy_prior_pacing_enabled"])
        self.assertEqual(filtered["digital_twin_policy_prior_pacing_fast_scale"], 1.35)
        self.assertTrue(filtered["digital_twin_policy_prior_env_action_bias_enabled"])
        self.assertEqual(filtered["digital_twin_policy_prior_env_action_logit_bias"], 4.8)
        self.assertEqual(filtered["digital_twin_policy_prior_continuation_threshold"], 0.32)
        self.assertEqual(filtered["digital_twin_policy_prior_continuation_prepare_scale"], 1.35)
        self.assertEqual(filtered["digital_twin_policy_prior_continuation_wait_scale"], 0.95)
        self.assertEqual(filtered["digital_twin_policy_prior_continuation_steady_suppression"], 0.42)
        self.assertTrue(filtered["digital_twin_policy_prior_adaptive_wait_enabled"])
        self.assertEqual(filtered["digital_twin_policy_prior_wait_ready_threshold"], 0.5)
        self.assertEqual(filtered["digital_twin_policy_prior_wait_timing_ceiling"], 0.58)
        self.assertEqual(filtered["digital_twin_policy_prior_wait_cache_ready_scale"], 1.42)
        self.assertEqual(filtered["digital_twin_policy_prior_prepare_not_ready_scale"], 0.92)
        self.assertTrue(filtered["env_action_ppo_enabled"])
        self.assertEqual(filtered["env_action_ppo_coef"], 0.72)
        self.assertEqual(filtered["env_action_ppo_advantage_blend"], 0.54)
        self.assertEqual(filtered["env_action_ppo_teacher_coef"], 0.34)
        self.assertEqual(filtered["env_action_ppo_mechanism_focus"], 0.65)
        self.assertEqual(filtered["env_action_ppo_max_weight"], 2.25)
        self.assertEqual(filtered["env_action_ppo_ratio_barrier_coef"], 0.045)
        self.assertEqual(filtered["env_action_ppo_ratio_barrier_margin"], 0.32)
        self.assertTrue(filtered["env_action_counterfactual_margin_enabled"])
        self.assertEqual(filtered["env_action_counterfactual_margin_coef"], 0.18)
        self.assertEqual(filtered["env_action_counterfactual_margin_min_gap"], 0.04)
        self.assertEqual(filtered["env_action_counterfactual_margin_max_weight"], 2.2)
        self.assertEqual(filtered["env_action_counterfactual_margin_advantage_gate"], 0.12)
        self.assertEqual(filtered["env_action_counterfactual_margin_advantage_blend"], 0.7)
        self.assertTrue(filtered["argmax_margin_regularization_enabled"])
        self.assertEqual(filtered["argmax_margin_coef"], 0.76)
        self.assertEqual(filtered["argmax_margin_min_gap"], 0.42)
        self.assertEqual(filtered["argmax_margin_max_weight"], 4.8)
        self.assertEqual(filtered["argmax_margin_tail_risk_threshold"], 0.04)
        self.assertEqual(filtered["argmax_margin_mechanism_penalty_scale"], 1.65)
        self.assertTrue(filtered["delayed_mechanism_credit_enabled"])
        self.assertEqual(filtered["delayed_mechanism_credit_policy_coef"], 0.46)
        self.assertEqual(filtered["delayed_mechanism_credit_event_coef"], 1.05)
        self.assertEqual(filtered["delayed_mechanism_credit_horizon"], 5)
        self.assertEqual(filtered["delayed_mechanism_credit_decay"], 0.72)
        self.assertEqual(filtered["delayed_mechanism_credit_clip"], 1.8)
        self.assertEqual(filtered["delayed_mechanism_credit_ready_bonus"], 1.35)
        self.assertEqual(filtered["delayed_mechanism_credit_success_bonus"], 0.88)
        self.assertEqual(filtered["delayed_mechanism_credit_failure_penalty"], 1.02)
        self.assertEqual(filtered["delayed_mechanism_credit_missed_prepare_scale"], 0.62)
        self.assertEqual(filtered["delayed_mechanism_credit_stale_penalty"], 0.42)
        self.assertEqual(filtered["delayed_mechanism_credit_context_gate"], 0.24)
        self.assertTrue(filtered["advantage_weighted_behavior_regularization_enabled"])
        self.assertEqual(filtered["advantage_weighted_behavior_coef"], 0.24)
        self.assertEqual(filtered["advantage_weighted_behavior_positive_coef"], 1.08)
        self.assertEqual(filtered["advantage_weighted_behavior_negative_coef"], 0.92)
        self.assertEqual(filtered["advantage_weighted_behavior_temperature"], 0.62)
        self.assertEqual(filtered["advantage_weighted_behavior_max_weight"], 2.2)
        self.assertEqual(filtered["advantage_weighted_behavior_positive_gate"], 0.08)
        self.assertEqual(filtered["advantage_weighted_behavior_negative_gate"], 0.04)
        self.assertEqual(filtered["advantage_weighted_behavior_mechanism_scale"], 1.35)
        self.assertEqual(filtered["mechanism_credit_event_coef"], 0.98)
        self.assertTrue(filtered["mechanism_focal_aux_enabled"])

    def test_sa_v64_checkpoint_config_preserves_alignment_barrier_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_sparse_recovery_focus": 0.34,
                "env_action_risk_adjusted_recovery_coef": 2.70,
                "env_action_risk_adjusted_recovery_floor": 0.10,
                "env_action_adapter_miss_counterfactual_coef": 3.45,
                "cache_feasibility_prior_enabled": True,
                "cache_feasibility_cache_fill_bias": 7.20,
                "cache_feasibility_steady_penalty": 6.20,
                "cache_feasibility_prepare_penalty": 1.45,
                "cache_feasibility_prefetch_penalty": 1.65,
                "cache_feasibility_current_miss_prepare_penalty": 8.80,
                "cache_feasibility_current_miss_prefetch_penalty": 5.60,
                "cache_feasibility_min_context": 0.0,
                "handoff_alignment_barrier_enabled": True,
                "handoff_alignment_barrier_prepare_penalty": 12.50,
                "handoff_alignment_barrier_prefetch_penalty": 7.20,
                "handoff_alignment_barrier_current_fill_bias": 8.80,
                "handoff_alignment_barrier_target_mismatch_penalty": 5.60,
                "handoff_alignment_barrier_late_eta_penalty": 3.20,
                "handoff_alignment_barrier_min_context": 0.0,
            },
        )

        self.assertEqual(filtered["env_action_sparse_recovery_focus"], 0.34)
        self.assertEqual(filtered["env_action_risk_adjusted_recovery_coef"], 2.70)
        self.assertEqual(filtered["env_action_risk_adjusted_recovery_floor"], 0.10)
        self.assertEqual(filtered["env_action_adapter_miss_counterfactual_coef"], 3.45)
        self.assertTrue(filtered["cache_feasibility_prior_enabled"])
        self.assertEqual(filtered["cache_feasibility_cache_fill_bias"], 7.20)
        self.assertEqual(filtered["cache_feasibility_steady_penalty"], 6.20)
        self.assertEqual(filtered["cache_feasibility_prepare_penalty"], 1.45)
        self.assertEqual(filtered["cache_feasibility_prefetch_penalty"], 1.65)
        self.assertEqual(filtered["cache_feasibility_current_miss_prepare_penalty"], 8.80)
        self.assertEqual(filtered["cache_feasibility_current_miss_prefetch_penalty"], 5.60)
        self.assertEqual(filtered["cache_feasibility_min_context"], 0.0)
        self.assertTrue(filtered["handoff_alignment_barrier_enabled"])
        self.assertEqual(filtered["handoff_alignment_barrier_prepare_penalty"], 12.50)
        self.assertEqual(filtered["handoff_alignment_barrier_prefetch_penalty"], 7.20)
        self.assertEqual(filtered["handoff_alignment_barrier_current_fill_bias"], 8.80)
        self.assertEqual(filtered["handoff_alignment_barrier_target_mismatch_penalty"], 5.60)
        self.assertEqual(filtered["handoff_alignment_barrier_late_eta_penalty"], 3.20)
        self.assertEqual(filtered["handoff_alignment_barrier_min_context"], 0.0)

    def test_sa_v67_checkpoint_config_preserves_sparse_recovery_prior_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "sparse_handoff_recovery_prior_enabled": True,
                "sparse_handoff_recovery_prefetch_bias": 6.80,
                "sparse_handoff_recovery_prepare_bias": 5.40,
                "sparse_handoff_recovery_current_fill_bias": 2.40,
                "sparse_handoff_recovery_steady_bias": 1.45,
                "sparse_handoff_recovery_local_penalty": 6.20,
                "sparse_handoff_recovery_min_context": 0.16,
                "sparse_handoff_recovery_max_eta": 16,
            },
        )

        self.assertTrue(filtered["sparse_handoff_recovery_prior_enabled"])
        self.assertEqual(filtered["sparse_handoff_recovery_prefetch_bias"], 6.80)
        self.assertEqual(filtered["sparse_handoff_recovery_prepare_bias"], 5.40)
        self.assertEqual(filtered["sparse_handoff_recovery_current_fill_bias"], 2.40)
        self.assertEqual(filtered["sparse_handoff_recovery_steady_bias"], 1.45)
        self.assertEqual(filtered["sparse_handoff_recovery_local_penalty"], 6.20)
        self.assertEqual(filtered["sparse_handoff_recovery_min_context"], 0.16)
        self.assertEqual(filtered["sparse_handoff_recovery_max_eta"], 16)

    def test_sa_v69_checkpoint_config_preserves_sparse_realization_credit_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "sparse_handoff_realization_credit_enabled": True,
                "sparse_handoff_realization_success_bonus": 1.55,
                "sparse_handoff_realization_ready_bonus": 1.25,
                "sparse_handoff_realization_prefetch_bonus": 1.05,
                "sparse_handoff_realization_failed_prepare_penalty": 2.55,
                "sparse_handoff_realization_local_penalty": 1.65,
                "sparse_handoff_realization_min_context": 0.08,
            },
        )

        self.assertTrue(filtered["sparse_handoff_realization_credit_enabled"])
        self.assertEqual(filtered["sparse_handoff_realization_success_bonus"], 1.55)
        self.assertEqual(filtered["sparse_handoff_realization_ready_bonus"], 1.25)
        self.assertEqual(filtered["sparse_handoff_realization_prefetch_bonus"], 1.05)
        self.assertEqual(filtered["sparse_handoff_realization_failed_prepare_penalty"], 2.55)
        self.assertEqual(filtered["sparse_handoff_realization_local_penalty"], 1.65)
        self.assertEqual(filtered["sparse_handoff_realization_min_context"], 0.08)

    def test_sa_v70_checkpoint_config_preserves_sparse_tail_option_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "sparse_handoff_option_prior_enabled": True,
                "sparse_handoff_option_prepare_bias": 7.40,
                "sparse_handoff_option_popularity_penalty": 4.35,
                "sparse_handoff_option_local_penalty": 5.85,
                "sparse_handoff_option_min_context": 0.08,
                "sparse_handoff_option_max_eta": 22,
            },
        )

        self.assertTrue(filtered["sparse_handoff_option_prior_enabled"])
        self.assertEqual(filtered["sparse_handoff_option_prepare_bias"], 7.40)
        self.assertEqual(filtered["sparse_handoff_option_popularity_penalty"], 4.35)
        self.assertEqual(filtered["sparse_handoff_option_local_penalty"], 5.85)
        self.assertEqual(filtered["sparse_handoff_option_min_context"], 0.08)
        self.assertEqual(filtered["sparse_handoff_option_max_eta"], 22)

    def test_sa_v71_checkpoint_config_preserves_counterfactual_option_critic_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "option_counterfactual_critic_enabled": True,
                "option_counterfactual_value_coef": 0.65,
                "option_counterfactual_advantage_coef": 1.20,
                "option_counterfactual_advantage_clip": 2.0,
                "option_counterfactual_warmup_updates": 2,
                "option_counterfactual_tail_weight": 2.25,
                "option_counterfactual_policy_improvement_enabled": True,
                "option_counterfactual_policy_improvement_coef": 1.0,
                "option_counterfactual_policy_improvement_clip": 2.0,
                "option_counterfactual_policy_improvement_deterministic_only": True,
                "option_counterfactual_model_rollout_enabled": True,
                "option_counterfactual_model_rollout_horizon": 4,
            },
        )

        self.assertTrue(filtered["option_counterfactual_critic_enabled"])
        self.assertEqual(filtered["option_counterfactual_value_coef"], 0.65)
        self.assertEqual(filtered["option_counterfactual_advantage_coef"], 1.20)
        self.assertEqual(filtered["option_counterfactual_advantage_clip"], 2.0)
        self.assertEqual(filtered["option_counterfactual_warmup_updates"], 2)
        self.assertEqual(filtered["option_counterfactual_tail_weight"], 2.25)
        self.assertTrue(filtered["option_counterfactual_policy_improvement_enabled"])
        self.assertEqual(filtered["option_counterfactual_policy_improvement_coef"], 1.0)
        self.assertEqual(filtered["option_counterfactual_policy_improvement_clip"], 2.0)
        self.assertTrue(
            filtered["option_counterfactual_policy_improvement_deterministic_only"]
        )
        self.assertTrue(filtered["option_counterfactual_model_rollout_enabled"])
        self.assertEqual(filtered["option_counterfactual_model_rollout_horizon"], 4)

    def test_sa_v74_checkpoint_config_preserves_action_model_critic_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_critic_enabled": True,
                "env_action_model_critic_value_coef": 0.8,
                "env_action_model_critic_advantage_coef": 1.25,
                "env_action_model_critic_policy_improvement_coef": 2.0,
                "env_action_model_critic_advantage_clip": 2.0,
                "env_action_model_critic_warmup_updates": 2,
                "env_action_model_rollout_enabled": True,
                "env_action_model_rollout_horizon": 4,
                "env_action_model_policy_improvement_enabled": True,
                "env_action_model_policy_improvement_coef": 0.35,
                "env_action_model_policy_improvement_temperature": 2.5,
            },
        )

        self.assertTrue(filtered["env_action_model_critic_enabled"])
        self.assertEqual(filtered["env_action_model_critic_value_coef"], 0.8)
        self.assertEqual(filtered["env_action_model_critic_advantage_coef"], 1.25)
        self.assertEqual(
            filtered["env_action_model_critic_policy_improvement_coef"],
            2.0,
        )
        self.assertEqual(filtered["env_action_model_critic_advantage_clip"], 2.0)
        self.assertEqual(filtered["env_action_model_critic_warmup_updates"], 2)
        self.assertTrue(filtered["env_action_model_rollout_enabled"])
        self.assertEqual(filtered["env_action_model_rollout_horizon"], 4)
        self.assertTrue(filtered["env_action_model_policy_improvement_enabled"])
        self.assertEqual(
            filtered["env_action_model_policy_improvement_coef"],
            0.35,
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_temperature"],
            2.5,
        )

    def test_sa_v76_checkpoint_config_preserves_memory_and_robust_model_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "outcome_memory_fusion_enabled": True,
                "outcome_memory_actor_scale": 0.70,
                "outcome_memory_critic_scale": 0.85,
                "env_action_model_rollout_horizons": [1, 2, 4, 8],
                "env_action_model_policy_improvement_robust_horizons_enabled": True,
                "env_action_model_policy_improvement_horizon_risk_coef": 0.75,
                "env_action_model_policy_improvement_adaptive_kl_enabled": True,
                "env_action_model_policy_improvement_target_kl": 0.03,
            },
        )

        self.assertTrue(filtered["outcome_memory_fusion_enabled"])
        self.assertEqual(filtered["outcome_memory_actor_scale"], 0.70)
        self.assertEqual(filtered["outcome_memory_critic_scale"], 0.85)
        self.assertEqual(
            filtered["env_action_model_rollout_horizons"],
            [1, 2, 4, 8],
        )
        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_robust_horizons_enabled"
            ]
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_horizon_risk_coef"
            ],
            0.75,
        )
        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_adaptive_kl_enabled"
            ]
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_target_kl"],
            0.03,
        )

    def test_sa_v77_checkpoint_config_preserves_temporal_downside_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_policy_improvement_horizon_aggregation_mode": (
                    "lambda_downside"
                ),
                "env_action_model_policy_improvement_horizon_lambda": 0.90,
            },
        )

        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_horizon_aggregation_mode"
            ],
            "lambda_downside",
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_horizon_lambda"],
            0.90,
        )

    def test_sa_v78_checkpoint_config_preserves_regret_adaptive_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_policy_improvement_regret_adaptive_kl_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_max_target_kl": 0.35,
                "env_action_model_policy_improvement_regret_priority_coef": 2.0,
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_regret_adaptive_kl_enabled"
            ]
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_max_target_kl"],
            0.35,
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_regret_priority_coef"
            ],
            2.0,
        )

    def test_sa_v79_checkpoint_config_preserves_tail_distillation_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_policy_improvement_tail_distillation_enabled": (
                    True
                ),
                "env_action_model_policy_improvement_tail_quantile": 0.75,
                "env_action_model_policy_improvement_tail_min_regret": 0.50,
                "env_action_model_policy_improvement_tail_epochs": 8,
                "env_action_model_policy_improvement_tail_coef": 1.0,
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_distillation_enabled"
            ]
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_tail_quantile"],
            0.75,
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_tail_min_regret"],
            0.50,
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_tail_epochs"],
            8,
        )
        self.assertEqual(
            filtered["env_action_model_policy_improvement_tail_coef"],
            1.0,
        )

    def test_sa_v80_checkpoint_config_preserves_imagination_replay_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_imagination_replay_enabled": True,
                "env_action_model_imagination_replay_depths": [2, 4, 8],
            },
        )

        self.assertTrue(
            filtered["env_action_model_imagination_replay_enabled"]
        )
        self.assertEqual(
            filtered["env_action_model_imagination_replay_depths"],
            [2, 4, 8],
        )

    def test_sa_v81_checkpoint_config_preserves_imagination_trust_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_imagination_replay_horizons": [1, 2, 4],
                "env_action_model_policy_improvement_tail_max_policy_kl": 0.05,
            },
        )

        self.assertEqual(
            filtered["env_action_model_imagination_replay_horizons"],
            [1, 2, 4],
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_max_policy_kl"
            ],
            0.05,
        )

    def test_sa_v82_checkpoint_config_preserves_recovery_only_field(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_imagination_replay_recovery_only": True,
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_imagination_replay_recovery_only"
            ]
        )

    def test_sa_v83_checkpoint_config_preserves_recovery_residual_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "outcome_recovery_residual_enabled": True,
                "outcome_recovery_residual_scale": 1.0,
                "env_action_model_policy_improvement_tail_recovery_only": True,
                "env_action_model_policy_improvement_tail_adapter_only": True,
            },
        )

        self.assertTrue(filtered["outcome_recovery_residual_enabled"])
        self.assertEqual(filtered["outcome_recovery_residual_scale"], 1.0)
        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_recovery_only"
            ]
        )
        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_adapter_only"
            ]
        )

    def test_sa_v85_checkpoint_config_preserves_residual_optimizer_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
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
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_residual_optimizer_enabled"
            ]
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_residual_learning_rate"
            ],
            0.02,
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_residual_backtrack_factor"
            ],
            0.5,
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_residual_min_learning_rate"
            ],
            0.00015625,
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_residual_max_backtracks"
            ],
            7,
        )

    def test_sa_v86_checkpoint_config_preserves_imagined_beam_field(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_imagination_beam_search_enabled": True,
            },
        )

        self.assertTrue(
            filtered["env_action_model_imagination_beam_search_enabled"]
        )

    def test_sa_v87_checkpoint_config_preserves_logit_projection_field(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_policy_improvement_tail_logit_projection_enabled": (
                    True
                ),
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_logit_projection_enabled"
            ]
        )

    def test_sa_v88_checkpoint_config_preserves_contextual_expert_field(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {"outcome_context_residual_enabled": True},
        )

        self.assertTrue(filtered["outcome_context_residual_enabled"])

    def test_sa_v89_checkpoint_config_preserves_branch_replay_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_imagination_replay_branch_mode": "top_k",
                "env_action_model_imagination_replay_branch_top_k": 2,
            },
        )

        self.assertEqual(
            filtered["env_action_model_imagination_replay_branch_mode"],
            "top_k",
        )
        self.assertEqual(
            filtered["env_action_model_imagination_replay_branch_top_k"],
            2,
        )

    def test_sa_v91_checkpoint_config_preserves_target_balance_fields(
        self,
    ) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
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
            },
        )

        self.assertTrue(
            filtered[
                "env_action_model_policy_improvement_tail_target_balance_enabled"
            ]
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_target_balance_power"
            ],
            0.5,
        )
        self.assertEqual(
            filtered[
                "env_action_model_policy_improvement_tail_target_balance_max_weight"
            ],
            4.0,
        )

    def test_sa_v92_checkpoint_config_preserves_online_planner_fields(self) -> None:
        filtered = _filter_checkpoint_config(
            "sa_ghmappo",
            {
                "env_action_model_online_planner_enabled": True,
                "env_action_model_online_planner_coef": 1.0,
                "env_action_model_online_planner_mechanism_coef": 2.0,
                "env_action_model_online_planner_policy_prior_coef": 0.15,
                "env_action_model_online_planner_min_margin": 0.08,
                "env_action_model_online_planner_prefer_beam_targets": True,
            },
        )

        self.assertTrue(filtered["env_action_model_online_planner_enabled"])
        self.assertEqual(filtered["env_action_model_online_planner_coef"], 1.0)
        self.assertEqual(
            filtered["env_action_model_online_planner_mechanism_coef"],
            2.0,
        )
        self.assertEqual(
            filtered["env_action_model_online_planner_policy_prior_coef"],
            0.15,
        )
        self.assertEqual(filtered["env_action_model_online_planner_min_margin"], 0.08)
        self.assertTrue(filtered["env_action_model_online_planner_prefer_beam_targets"])


if __name__ == "__main__":
    unittest.main()
