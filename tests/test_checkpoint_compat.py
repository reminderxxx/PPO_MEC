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


if __name__ == "__main__":
    unittest.main()
