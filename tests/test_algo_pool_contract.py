"""Algorithm-pool registry and action-schema contract tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import (
    build_agent,
    checkpoint_required_agents,
    get_algo_spec,
    list_evaluable_agents,
    list_registered_agents,
    list_trainable_agents,
)
from src.envs.specs import ActionAdapter, ActionMaskBuilder, ActionSchema


def _minimal_semantic_state() -> dict:
    return {
        "time_index": 1,
        "current_workflow_node": {
            "node_id": "n1",
            "required_adapter": "adapter_tracking",
            "input_size": 10.0,
            "output_size": 5.0,
            "predecessors": [],
            "successors": [],
        },
        "workflow": {
            "nodes": [
                {
                    "node_id": "n1",
                    "predecessors": [],
                    "successors": [],
                }
            ],
            "completed_node_ids": [],
            "execution_order": ["n1"],
            "current_node_id": "n1",
        },
        "vehicles": [
            {
                "vehicle_id": "veh_1",
                "associated_rsu_id": "rsu_a",
                "speed": 10.0,
            }
        ],
        "rsus": [
            {
                "rsu_id": "rsu_a",
                "cached_adapter_ids": [],
                "cache_capacity": 4,
            },
            {
                "rsu_id": "rsu_b",
                "cached_adapter_ids": [],
                "cache_capacity": 4,
            },
        ],
        "predictions": {
            "future_load": {"rsu_a": 1.0, "rsu_b": 2.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.8},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.1},
            "dwell_time": {"veh_1": 3.0},
            "next_rsu_sequence": {"veh_1": ["rsu_b"]},
        },
    }


class AlgoPoolContractTestCase(unittest.TestCase):
    """Validate the direction-matched algorithm pool contract."""

    def test_registry_contains_only_live_trainable_and_heuristic_agents(self) -> None:
        self.assertEqual(
            list_registered_agents(),
            [
                "cache_offload_drl",
                "controller_mat",
                "dag_offload_drl",
                "ddqn",
                "dqn",
                "dt_handoff_drl",
                "dueling_ddqn",
                "dueling_dqn",
                "ippo",
                "mappo",
                "popularity_cache_heuristic",
                "ppo",
                "qmix",
                "reactive_greedy",
                "sa_ghmappo",
            ],
        )
        self.assertEqual(
            list_evaluable_agents(),
            [
                "cache_offload_drl",
                "controller_mat",
                "dag_offload_drl",
                "ddqn",
                "dqn",
                "dt_handoff_drl",
                "dueling_ddqn",
                "dueling_dqn",
                "ippo",
                "mappo",
                "popularity_cache_heuristic",
                "ppo",
                "qmix",
                "reactive_greedy",
                "sa_ghmappo",
            ],
        )
        self.assertEqual(
            list_trainable_agents(),
            [
                "cache_offload_drl",
                "controller_mat",
                "dag_offload_drl",
                "ddqn",
                "dqn",
                "dt_handoff_drl",
                "dueling_ddqn",
                "dueling_dqn",
                "mappo",
                "ppo",
                "qmix",
                "sa_ghmappo",
            ],
        )
        self.assertEqual(
            checkpoint_required_agents(),
            {
                "cache_offload_drl",
                "controller_mat",
                "dag_offload_drl",
                "ddqn",
                "dqn",
                "dt_handoff_drl",
                "dueling_ddqn",
                "dueling_dqn",
                "ippo",
                "mappo",
                "ppo",
                "qmix",
                "sa_ghmappo",
            },
        )

    def test_live_agents_can_be_built(self) -> None:
        self.assertEqual(build_agent("cache_offload_drl", random_seed=1).agent_name, "cache_offload_drl")
        self.assertEqual(build_agent("dqn", random_seed=1).agent_name, "dqn")
        self.assertEqual(build_agent("dag_offload_drl", random_seed=1).agent_name, "dag_offload_drl")
        self.assertEqual(build_agent("ddqn", random_seed=1).agent_name, "ddqn")
        self.assertEqual(build_agent("dt_handoff_drl", random_seed=1).agent_name, "dt_handoff_drl")
        self.assertEqual(build_agent("dueling_dqn", random_seed=1).agent_name, "dueling_dqn")
        self.assertEqual(build_agent("dueling_ddqn", random_seed=1).agent_name, "dueling_ddqn")
        self.assertEqual(build_agent("controller_mat", random_seed=1).agent_name, "controller_mat")
        self.assertEqual(build_agent("ippo", random_seed=1).agent_name, "ippo")
        self.assertEqual(build_agent("ppo", random_seed=1).agent_name, "ppo")
        self.assertEqual(build_agent("qmix", random_seed=1).agent_name, "qmix")
        self.assertEqual(build_agent("mappo", random_seed=1).agent_name, "mappo")
        self.assertEqual(build_agent("reactive_greedy").support_level, "heuristic")
        self.assertEqual(build_agent("popularity_cache_heuristic").support_level, "heuristic")
        self.assertEqual(get_algo_spec("ippo")["support_level"], "diagnostic")
        self.assertEqual(get_algo_spec("mappo")["support_level"], "trainable")

    def test_mappo_uses_controller_level_ctde_contract(self) -> None:
        state = _minimal_semantic_state()
        ppo = build_agent("ppo", random_seed=1)
        mappo = build_agent("mappo", random_seed=1)
        ppo_output = ppo._network.forward_single(state)
        mappo_output = mappo._network.forward_single(state)
        self.assertEqual(ppo_output["critic_mode"], "independent")
        self.assertEqual(ppo_output["critic_context_key"], "critic_context")
        self.assertIn("slow_logits", mappo_output)
        self.assertIn("fast_logits", mappo_output)
        self.assertIn("event_logits", mappo_output)
        self.assertNotIn("flat_logits", mappo_output)
        self.assertEqual(mappo_output["critic_mode"], "centralized")
        self.assertEqual(mappo_output["critic_context_key"], "centralized_critic_context")
        self.assertEqual(mappo.baseline_config["ctde_scope"], "controller_level_cache_execution_handoff")
        self.assertTrue(mappo.baseline_config["paper_grade_independent_baseline"])
        self.assertTrue(mappo.baseline_config["controller_head_credit"])
        self.assertEqual(
            mappo.baseline_config["head_credit_protocol"],
            "aggregation_reason_weighted_controller_ppo_v3",
        )
        self.assertEqual(
            mappo.baseline_config["controller_head_credit_floors"],
            {"slow": 0.25, "fast": 0.10, "event": 0.12},
        )
        self.assertEqual(
            mappo._build_head_credit_weights("event_head_prepare"),
            {"slow": 0.3, "fast": 0.1, "event": 1.0},
        )
        self.assertEqual(
            mappo._build_head_credit_weights("fast_head_steady_offload"),
            {"slow": 0.3, "fast": 1.0, "event": 0.15},
        )
        self.assertEqual(mappo._resolve_actor_weight("slow", 0.0), 0.25)
        self.assertAlmostEqual(mappo._resolve_entropy_weight("event", 0.0), 0.162)

    def test_mappo_action_exposes_three_controller_agents(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("mappo", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
            },
        )
        self.assertIn(action, {0, 1, 2, 3, 4})
        self.assertEqual(set(action_info["head_actions"].keys()), {"slow", "fast", "event"})
        self.assertEqual(action_info["critic_mode"], "centralized")
        self.assertEqual(action_info["critic_context_key"], "centralized_critic_context")
        self.assertEqual(action_info["policy_type"], "mappo_policy")
        self.assertEqual(action_info["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")
        self.assertIn("head_credit_weights", action_info)
        self.assertEqual(action_info["effective_head_credit_floors"]["policy"]["slow"], 0.25)

    def test_mappo_strong_audit_training_profile_sets_v3_protocol(self) -> None:
        from scripts.train_algo_pool_real_sample import agent_profile_kwargs

        kwargs = agent_profile_kwargs("mappo", "mappo_strong_audit")
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")
        self.assertEqual(kwargs["slow_policy_credit_floor"], 0.25)
        self.assertEqual(kwargs["event_advantage_blend"], 0.85)
        self.assertEqual(agent_profile_kwargs("ppo", "mappo_strong_audit"), {})

    def test_sa_v6_profile_is_registered_for_strong_competition(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        self.assertIn("top_journal_mechanism_v6_strong_competition", PROFILE_DEFAULTS)
        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v6_strong_competition"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["train_window_count"], 6)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v6_strong_competition")
        self.assertEqual(kwargs["mechanism_window_weight"], 1.65)
        self.assertEqual(kwargs["mechanism_window_weight_floor_after_update"], 1.60)
        self.assertFalse(kwargs["predictive_prepare_hard_override_enabled"])
        self.assertEqual(kwargs["cache_warm_start_guard_max_prefetch_countdown"], 6.0)
        self.assertTrue(kwargs["predictive_prefetch_admission_guard_enabled"])
        self.assertEqual(kwargs["predictive_prefetch_admission_min_confidence"], 0.55)
        self.assertTrue(kwargs["predictive_prefetch_admission_require_distinct_next"])

    def test_sa_v7_profile_combines_v6_guards_with_latency_fallback(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        self.assertIn("top_journal_mechanism_v7_latency_fallback", PROFILE_DEFAULTS)
        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v7_latency_fallback"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["train_window_count"], 6)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v7_latency_fallback")
        self.assertEqual(kwargs["mechanism_window_weight"], 1.65)
        self.assertEqual(kwargs["cache_warm_start_guard_max_prefetch_countdown"], 6.0)
        self.assertTrue(kwargs["predictive_prefetch_admission_guard_enabled"])
        self.assertTrue(kwargs["latency_fallback_bias_enabled"])
        self.assertEqual(kwargs["latency_fallback_bias_strength"], 1.20)
        self.assertEqual(kwargs["latency_fallback_confidence_floor"], 0.62)
        self.assertEqual(kwargs["latency_fallback_slow_suppression_strength"], 1.20)

    def test_sa_v8_profile_replaces_vehicle_fallback_with_steady_rsu_bias(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v8_strict_full"]
        self.assertEqual(defaults["episodes"], 96)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v8_strict_full")
        self.assertFalse(kwargs["latency_fallback_bias_enabled"])
        self.assertTrue(kwargs["steady_rsu_bias_enabled"])
        self.assertEqual(kwargs["steady_rsu_bias_strength"], 1.20)
        self.assertEqual(kwargs["steady_rsu_confidence_floor"], 0.62)
        self.assertEqual(kwargs["continuity_guard_confidence_threshold"], 0.65)
        self.assertEqual(kwargs["continuity_guard_prepare_score_threshold"], 0.35)
        self.assertEqual(kwargs["predictive_prefetch_admission_min_confidence"], 0.62)

    def test_sa_v9_profile_adds_pareto_safe_guardrails(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v9_pareto_safe"]
        self.assertEqual(defaults["episodes"], 96)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v9_pareto_safe")
        self.assertFalse(kwargs["latency_fallback_bias_enabled"])
        self.assertTrue(kwargs["steady_rsu_bias_enabled"])
        self.assertGreater(kwargs["steady_rsu_bias_strength"], 1.20)
        self.assertTrue(kwargs["backhaul_guard_enabled"])
        self.assertEqual(kwargs["backhaul_guard_max_reactive_fills_per_adapter"], 1)
        self.assertEqual(kwargs["predictive_prefetch_admission_min_confidence"], 0.68)
        self.assertEqual(kwargs["cache_warm_start_guard_max_prefetch_countdown"], 5.0)

    def test_sa_v10_profile_transfers_mappo_controller_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v10_mappo_rl"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v10_mappo_rl")
        self.assertTrue(kwargs["head_credit_enabled"])
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")
        self.assertEqual(kwargs["slow_policy_credit_floor"], 0.25)
        self.assertEqual(kwargs["fast_policy_credit_floor"], 0.10)
        self.assertEqual(kwargs["event_policy_credit_floor"], 0.12)
        self.assertEqual(kwargs["slow_entropy_credit_floor"], 0.20)
        self.assertEqual(kwargs["fast_entropy_credit_floor"], 0.08)
        self.assertEqual(kwargs["event_entropy_credit_floor"], 0.12)
        self.assertEqual(kwargs["event_advantage_blend"], 0.85)
        self.assertLess(kwargs["heuristic_imitation_coef"], 0.10)
        self.assertLess(kwargs["mechanism_aux_coef"], 0.09)
        self.assertFalse(kwargs["mechanism_aux_current_cache_fill_enabled"])
        agent = build_agent("sa_ghmappo", random_seed=1, deterministic_action=True, **kwargs)
        self.assertEqual(
            agent._build_head_credit_weights("event_head_prepare"),
            {"slow": 0.3, "fast": 0.1, "event": 1.0},
        )

    def test_sa_v11_profile_is_reward_first_mappo_rl(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v11_mappo_reward"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        self.assertEqual(defaults["entropy_coef"], PROFILE_DEFAULTS["top_journal_mechanism_v10_mappo_rl"]["entropy_coef"])
        self.assertEqual(defaults["auxiliary_coef"], PROFILE_DEFAULTS["top_journal_mechanism_v8_strict_full"]["auxiliary_coef"])
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v11_mappo_reward")
        self.assertTrue(kwargs["head_credit_enabled"])
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")
        self.assertLess(kwargs["heuristic_imitation_coef"], 0.10)
        self.assertLess(kwargs["mechanism_aux_coef"], 0.09)
        self.assertLess(kwargs["prepare_action_prior_weight"], 0.60)
        self.assertTrue(kwargs["mechanism_aux_current_cache_fill_enabled"])
        self.assertGreaterEqual(kwargs["fast_policy_credit_floor"], 0.12)
        self.assertGreater(kwargs["event_logit_sharpening_final_scale"], 1.55)
        self.assertTrue(kwargs["idle_popularity_fallback_enabled"])
        self.assertTrue(kwargs["idle_popularity_fallback_only_vehicle_fallback"])
        self.assertEqual(kwargs["idle_popularity_prefetch_threshold"], 2)
        self.assertFalse(kwargs["idle_popularity_no_rsu_local_fallback_enabled"])
        self.assertTrue(kwargs["idle_popularity_no_rsu_local_requires_low_context"])

    def test_sa_v11_idle_popularity_fallback_replaces_vehicle_fallback_only(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=3,
            deterministic_action=True,
            idle_popularity_fallback_enabled=True,
            idle_popularity_fallback_only_vehicle_fallback=True,
            idle_popularity_prefetch_threshold=2,
        )

        fallback_info = agent._maybe_apply_idle_popularity_fallback(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            original_env_action=2,
            deterministic=True,
        )

        self.assertTrue(fallback_info["applied"])
        self.assertEqual(fallback_info["fallback_action"], 0)
        self.assertEqual(fallback_info["candidate_reason"], "popular_adapter_reactive_cache_fill")

        non_fallback_info = agent._maybe_apply_idle_popularity_fallback(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            original_env_action=3,
            deterministic=True,
        )

        self.assertFalse(non_fallback_info["applied"])
        self.assertEqual(non_fallback_info["reason"], "original_action_not_vehicle_fallback")

    def test_sa_v11_idle_popularity_fallback_replaces_no_rsu_current_offload(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"] = {
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "next_rsu_sequence": {},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=5,
            deterministic_action=True,
            idle_popularity_fallback_enabled=True,
            idle_popularity_fallback_only_vehicle_fallback=True,
            idle_popularity_prefetch_threshold=2,
            idle_popularity_no_rsu_local_fallback_enabled=True,
        )

        fallback_info = agent._maybe_apply_idle_popularity_fallback(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            original_env_action=3,
            deterministic=True,
        )

        self.assertTrue(fallback_info["applied"])
        self.assertEqual(fallback_info["fallback_action"], 2)
        self.assertEqual(fallback_info["candidate_reason"], "no_associated_rsu_vehicle_fallback")
        self.assertEqual(fallback_info["reason"], "no_rsu_current_offload_replaced_by_local")
        self.assertTrue(fallback_info["low_mechanism_no_rsu_context"])

    def test_sa_v12_profile_enables_learned_option_gate(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v12_learned_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v12_learned_option")
        self.assertTrue(kwargs["head_credit_enabled"])
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertEqual(kwargs["option_gate_count"], 4)
        self.assertGreater(kwargs["option_gate_loss_coef"], 0.0)
        self.assertGreater(kwargs["option_gate_prior_coef"], 0.0)
        self.assertTrue(kwargs["option_gate_context_prior_enabled"])
        self.assertGreater(kwargs["option_gate_deterministic_prior_margin"], 0.0)
        self.assertTrue(kwargs["option_gate_idle_prior_enabled"])
        self.assertFalse(kwargs["idle_popularity_no_rsu_local_fallback_enabled"])

    def test_sa_v12_option_gate_records_policy_choice(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_prior_logit_bias=0.5,
            idle_popularity_fallback_enabled=False,
        )
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
            },
        )

        self.assertIn(action, {0, 1, 2, 3, 4})
        option_info = action_info["option_gate"]
        self.assertTrue(option_info["enabled"])
        self.assertIn(option_info["option_action"], {0, 1, 2, 3})
        self.assertIn("option_log_prob", option_info)
        self.assertEqual(len(option_info["option_mask"]), 4)

    def test_sa_v12_contextual_prior_prefers_popularity_on_idle_sparse(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_deterministic_prior_margin=0.2,
            option_gate_idle_prior_enabled=True,
            option_gate_prior_logit_bias=0.0,
            idle_popularity_fallback_enabled=False,
        )
        option_info = agent._maybe_apply_option_gate(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([0.1, 0.0, -1.0, -1.0])},
            base_env_action=3,
            deterministic=True,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertTrue(option_info["applied"])
        self.assertEqual(option_info["option_label"], "popularity_safe")
        self.assertEqual(option_info["option_env_action"], 0)
        self.assertFalse(option_info["option_mask"][3])
        self.assertEqual(option_info["prior_label"], "popularity_safe")
        self.assertEqual(option_info["selection_reason"], "context_prior_margin")

    def test_sa_v12_can_warm_start_from_v11_without_option_head(self) -> None:
        source_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=False,
        )
        target_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "v11_like.pt"
            source_agent.save(str(checkpoint_path))
            target_agent.load(str(checkpoint_path))

        action, action_info = target_agent.act(
            None,
            {
                "semantic_state": _minimal_semantic_state(),
                "action_mask": [True, True, True, True, True],
            },
        )
        self.assertIn(action, {0, 1, 2, 3, 4})
        self.assertTrue(action_info["option_gate"]["enabled"])

    def test_sa_v12_contextual_option_preserves_mappo_on_mechanism_window(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
        )
        option_info = agent._maybe_apply_option_gate(
            semantic_state=_minimal_semantic_state(),
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([-1.0, -1.0, -1.0, 3.0])},
            base_env_action=4,
            deterministic=True,
            run_metadata={"window_class": "mechanism_activating"},
        )

        self.assertFalse(option_info["enabled"])
        self.assertFalse(option_info["applied"])
        self.assertEqual(option_info["reason"], "mechanism_window_preserve_mappo")
        self.assertEqual(option_info["base_env_action"], 4)

    def test_sa_v13_profile_enables_prd_option_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v13_prd_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v13_prd_option")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["option_gate_prd_enabled"])
        self.assertGreater(kwargs["option_gate_prd_coef"], 0.0)
        self.assertGreater(kwargs["option_gate_prd_clip"], 0.0)
        self.assertTrue(kwargs["option_gate_mechanism_preserve_enabled"])
        self.assertTrue(kwargs["event_prd_advantage_enabled"])
        self.assertGreater(kwargs["event_prd_advantage_coef"], 0.0)
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")

    def test_sa_v13_event_prd_rewards_mechanism_prepare_credit(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            event_prd_advantage_enabled=True,
            event_prd_advantage_coef=0.4,
        )
        positive_row = {
            "action": 4,
            "action_info": {
                "head_actions": {"event": 1},
                "final_env_action": 4,
                "prepare_window_score": 0.8,
                "temporal_urgency": 0.7,
                "prediction_confidence": 0.75,
                "gate_pass": True,
            },
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {"metrics_protocol": {"mechanism_success_rate": 0.5, "handoff_ready_rate": 0.5}},
        }
        negative_row = {
            "action": 3,
            "action_info": {
                "head_actions": {"event": 0},
                "final_env_action": 3,
                "prepare_window_score": 0.8,
                "temporal_urgency": 0.7,
                "prediction_confidence": 0.75,
                "gate_pass": True,
            },
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {"metrics_protocol": {"mechanism_success_rate": 0.0, "handoff_ready_rate": 0.0}},
        }

        self.assertGreater(agent._event_partial_reward_credit(positive_row), 0.0)
        self.assertLess(agent._event_partial_reward_credit(negative_row), 0.0)

    def test_sa_v14_profile_enables_net_utility_prd_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v14_net_utility_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v14_net_utility_prd")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["option_gate_prd_enabled"])
        self.assertTrue(kwargs["event_prd_advantage_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["net_utility_cost_dual_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_enabled"])
        self.assertGreater(kwargs["net_utility_backhaul_coef"], 0.0)
        self.assertGreater(kwargs["net_utility_expired_prefetch_coef"], 0.0)
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")

    def test_sa_v15_profile_keeps_prd_option_and_terminal_fallback(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v15_terminal_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v15_terminal_option")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["option_gate_prd_enabled"])
        self.assertTrue(kwargs["event_prd_advantage_enabled"])
        self.assertFalse(kwargs["net_utility_prd_enabled"])
        self.assertFalse(kwargs["net_utility_cost_dual_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_enabled"])
        self.assertEqual(kwargs["head_credit_protocol"], "aggregation_reason_weighted_controller_ppo_v3")

    def test_sa_v16_profile_enables_conservative_terminal_option(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v16_conservative_terminal_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v16_conservative_terminal_option")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["option_gate_prd_enabled"])
        self.assertFalse(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_conservative_enabled"])
        self.assertGreater(kwargs["net_utility_option_termination_max_timing_support"], 0.0)

    def test_sa_v17_profile_enables_dag_aware_option(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v17_dag_aware_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v17_dag_aware_option")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_enabled"])
        self.assertTrue(kwargs["net_utility_option_termination_conservative_enabled"])
        self.assertTrue(kwargs["dag_aware_option_termination_enabled"])
        self.assertEqual(kwargs["dag_aware_option_min_critical_path"], 6)
        self.assertEqual(kwargs["dag_aware_option_short_workflow_max_nodes"], 12)

    def test_sa_v18_profile_enables_counterfactual_option_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v18_counterfactual_option"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v18_counterfactual_option")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["dag_aware_option_termination_enabled"])
        self.assertTrue(kwargs["option_gate_prd_enabled"])
        self.assertTrue(kwargs["option_gate_counterfactual_prd_enabled"])
        self.assertGreater(kwargs["option_gate_counterfactual_coef"], 0.0)
        self.assertFalse(kwargs["option_gate_mechanism_preserve_enabled"])

    def test_sa_v18_counterfactual_credit_rewards_better_option_than_policy_baseline(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_prd_enabled=False,
            option_gate_counterfactual_prd_enabled=True,
            option_gate_counterfactual_coef=1.0,
            option_gate_counterfactual_clip=2.0,
        )
        row = {
            "reward": 1.0,
            "action": 4,
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "action_info": {
                "prepare_window_score": 0.8,
                "temporal_urgency": 0.7,
                "prediction_confidence": 0.8,
                "gate_pass": True,
                "final_env_action": 4,
                "option_gate": {
                    "enabled": True,
                    "applied": True,
                    "option_action": 3,
                    "option_label": "mechanism_prepare",
                    "option_env_action": 4,
                    "base_env_action": 0,
                    "option_actions": {"0": 0, "1": 2, "2": 2, "3": 4},
                    "option_mask": [True, True, False, True],
                    "window_class": "mechanism_activating",
                },
            },
            "env_info": {
                "metrics_protocol": {
                    "mechanism_success_rate": 1.0,
                    "handoff_ready_rate": 1.0,
                    "handoff_failure_rate": 0.0,
                }
            },
        }

        advantage = agent._option_gate_advantage(
            row=row,
            base_advantage=torch.tensor(0.0),
            option_probs=torch.tensor([0.75, 0.20, 0.0, 0.05]),
            option_mask=[True, True, False, True],
        )

        self.assertGreater(float(advantage.item()), 0.0)

    def test_sa_v14_net_utility_option_terminates_idle_no_rsu_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            net_utility_option_termination_enabled=True,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        semantic_state["vehicles"][0]["associated_rsu_id"] = None

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertTrue(option_info["applied"])
        self.assertEqual(option_info["selection_reason"], "net_utility_idle_prefetch_termination")
        self.assertEqual(option_info["option_label"], "popularity_safe")
        self.assertEqual(option_info["option_env_action"], 2)

    def test_sa_v16_conservative_option_preserves_handoff_candidate(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            net_utility_option_termination_enabled=True,
            net_utility_option_termination_conservative_enabled=True,
            net_utility_option_termination_max_timing_support=1.0,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        semantic_state["vehicles"][0]["associated_rsu_id"] = None

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertFalse(option_info["applied"])
        self.assertEqual(option_info["selection_reason"], "policy_argmax")
        self.assertEqual(option_info["option_label"], "accept_mappo")

    def test_sa_v16_conservative_option_terminates_low_context_idle_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            net_utility_option_termination_enabled=True,
            net_utility_option_termination_conservative_enabled=True,
            net_utility_option_termination_max_timing_support=1.0,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        semantic_state["vehicles"][0]["associated_rsu_id"] = None
        semantic_state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 2.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {},
            "prediction_uncertainty_by_vehicle": {},
            "dwell_time": {},
            "next_rsu_sequence": {"veh_1": []},
        }

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertTrue(option_info["applied"])
        self.assertEqual(
            option_info["selection_reason"],
            "net_utility_conservative_idle_prefetch_termination",
        )
        self.assertEqual(option_info["option_label"], "popularity_safe")
        self.assertEqual(option_info["option_env_action"], 2)

    def test_sa_v17_dag_aware_option_terminates_short_dag_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            dag_aware_option_termination_enabled=True,
            dag_aware_option_min_critical_path=6,
            dag_aware_option_short_workflow_max_nodes=12,
            dag_aware_option_branching_successors=3,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        nodes = [
            {"node_id": f"task_{idx}", "predecessors": [], "successors": []}
            for idx in range(1, 10)
        ]
        nodes[2]["successors"] = ["task_4", "task_5", "task_6", "task_7"]
        for successor_idx in range(4, 8):
            nodes[successor_idx - 1]["predecessors"] = ["task_3"]
        semantic_state["workflow"] = {
            "nodes": nodes,
            "completed_node_ids": ["task_1", "task_2"],
            "execution_order": [node["node_id"] for node in nodes],
            "current_node_id": "task_3",
        }
        semantic_state["current_workflow_node"] = {
            "node_id": "task_3",
            "required_adapter": "adapter_tracking",
            "predecessors": ["task_1", "task_2"],
            "successors": ["task_4", "task_5", "task_6", "task_7"],
        }

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "mechanism_activating"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertTrue(option_info["applied"])
        self.assertEqual(option_info["selection_reason"], "dag_aware_short_dag_prefetch_termination")
        self.assertEqual(option_info["option_label"], "popularity_safe")

    def test_sa_v17_dag_aware_option_preserves_long_dag_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            dag_aware_option_termination_enabled=True,
            dag_aware_option_min_critical_path=6,
            dag_aware_option_short_workflow_max_nodes=12,
            dag_aware_option_branching_successors=3,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        nodes = []
        for idx in range(1, 18):
            successors = [f"task_{idx + 1}"] if idx < 17 else []
            predecessors = [f"task_{idx - 1}"] if idx > 1 else []
            nodes.append({"node_id": f"task_{idx}", "predecessors": predecessors, "successors": successors})
        semantic_state["workflow"] = {
            "nodes": nodes,
            "completed_node_ids": [],
            "execution_order": [node["node_id"] for node in nodes],
            "current_node_id": "task_1",
        }
        semantic_state["current_workflow_node"] = {
            "node_id": "task_1",
            "required_adapter": "adapter_tracking",
            "predecessors": [],
            "successors": ["task_2"],
        }

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "mechanism_activating"},
        )

        self.assertFalse(option_info["enabled"])
        self.assertFalse(option_info["applied"])
        self.assertEqual(option_info["reason"], "mechanism_window_preserve_mappo")

    def test_sa_v17_dag_aware_option_terminates_idle_low_confidence_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_context_prior_enabled=True,
            option_gate_idle_prior_enabled=True,
            net_utility_option_termination_enabled=True,
            net_utility_option_termination_conservative_enabled=True,
            net_utility_option_termination_max_timing_support=1.0,
            dag_aware_option_termination_enabled=True,
            dag_aware_idle_prefetch_confidence_floor=0.65,
        )
        semantic_state = deepcopy(_minimal_semantic_state())
        semantic_state["vehicles"][0]["associated_rsu_id"] = None
        semantic_state["predictions"]["prediction_confidence_by_vehicle"] = {"veh_1": 0.6}

        option_info = agent._maybe_apply_option_gate(
            semantic_state=semantic_state,
            action_mask=[True, True, True, True, True],
            policy_output={"option_logits": torch.tensor([5.0, -2.0, -2.0, -2.0])},
            base_env_action=1,
            deterministic=True,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(option_info["enabled"])
        self.assertTrue(option_info["applied"])
        self.assertEqual(option_info["selection_reason"], "dag_aware_idle_low_confidence_prefetch_termination")
        self.assertEqual(option_info["option_label"], "popularity_safe")

    def test_sa_v14_net_utility_prd_penalizes_idle_expired_prefetch(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            event_prd_advantage_enabled=True,
            option_gate_prd_enabled=True,
            net_utility_prd_enabled=True,
            net_utility_backhaul_coef=0.16,
            net_utility_migration_coef=0.22,
            net_utility_expired_prefetch_coef=0.55,
            net_utility_idle_prefetch_penalty=0.65,
            net_utility_success_bonus=0.16,
            net_utility_backhaul_normalizer=64.0,
        )
        expired_idle_row = {
            "action": 1,
            "reward": -0.2,
            "action_info": {
                "head_actions": {"event": 1},
                "final_env_action": 1,
                "option_gate": {
                    "enabled": True,
                    "option_label": "accept_mappo",
                    "window_class": "idle_or_sparse",
                },
            },
            "decision_info": {"run_metadata": {"window_class": "idle_or_sparse"}},
            "env_info": {
                "metrics_protocol": {
                    "backhaul_traffic_cost": 64.0,
                    "adapter_state_migration_overhead": 0.0,
                    "predictive_prefetch_requested": True,
                    "prefetch_expired_miss": True,
                    "mechanism_success_strict": False,
                }
            },
        }
        success_row = {
            "action": 4,
            "reward": 1.0,
            "action_info": {
                "head_actions": {"event": 1},
                "final_env_action": 4,
                "prepare_window_score": 0.8,
                "temporal_urgency": 0.7,
                "prediction_confidence": 0.75,
                "gate_pass": True,
                "option_gate": {
                    "enabled": True,
                    "option_label": "accept_mappo",
                    "window_class": "mechanism_activating",
                },
            },
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {
                "metrics_protocol": {
                    "backhaul_traffic_cost": 0.0,
                    "adapter_state_migration_overhead": 0.0,
                    "mechanism_success_strict": True,
                    "handoff_ready": True,
                }
            },
        }

        self.assertLess(agent._event_partial_reward_credit(expired_idle_row), -1.0)
        self.assertLess(agent._option_gate_partial_reward_credit(expired_idle_row), -1.0)
        self.assertGreater(agent._event_partial_reward_credit(success_row), 0.0)
        self.assertGreater(agent._option_gate_partial_reward_credit(success_row), 0.0)

    def test_qmix_uses_controller_level_value_decomposition_contract(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("qmix", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
            },
        )
        self.assertIn(action, {0, 1, 2, 3, 4})
        self.assertEqual(set(action_info["head_actions"].keys()), {"slow", "fast", "event"})
        self.assertEqual(action_info["critic_mode"], "centralized_mixer")
        self.assertEqual(action_info["critic_context_key"], "centralized_critic_context")
        self.assertEqual(action_info["mixer"], "qmix")
        self.assertEqual(action_info["policy_type"], "qmix_policy")

    def test_controller_mat_uses_transformer_ctde_contract(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("controller_mat", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
            },
        )
        self.assertIn(action, {0, 1, 2, 3, 4})
        self.assertEqual(set(action_info["head_actions"].keys()), {"slow", "fast", "event"})
        self.assertEqual(action_info["critic_mode"], "controller_transformer_ctde")
        self.assertEqual(action_info["critic_context_key"], "controller_transformer_pooled_context")
        self.assertEqual(action_info["policy_type"], "controller_mat_policy")
        self.assertTrue(agent.baseline_config["controller_attention"])

    def test_domain_baselines_use_independent_domain_contracts(self) -> None:
        state = _minimal_semantic_state()
        expected = {
            "dag_offload_drl": (
                "dag_offload_drl_policy",
                "dag_offload_centralized_critic",
                "flat_semantic_plus_dag_scalars",
                "dag_scalar_features",
            ),
            "cache_offload_drl": (
                "cache_offload_drl_policy",
                "cache_offload_centralized_critic",
                "flat_semantic_plus_cache_scalars",
                "cache_scalar_features",
            ),
            "dt_handoff_drl": (
                "dt_handoff_drl_policy",
                "dt_handoff_centralized_critic",
                "flat_semantic_plus_digital_twin_handoff_scalars",
                "digital_twin_snapshot_features",
            ),
        }
        for agent_name, (policy_type, critic_mode, critic_context_key, config_key) in expected.items():
            agent = build_agent(agent_name, random_seed=1, deterministic_action=True)
            action, action_info = agent.act(
                None,
                {
                    "semantic_state": state,
                    "action_mask": [True, True, True, True, True],
                },
            )
            self.assertIn(action, {0, 1, 2, 3, 4})
            self.assertEqual(set(action_info["head_actions"].keys()), {"slow", "fast", "event"})
            self.assertEqual(action_info["policy_type"], policy_type)
            self.assertEqual(action_info["critic_mode"], critic_mode)
            self.assertEqual(action_info["critic_context_key"], critic_context_key)
            self.assertTrue(agent.baseline_config[config_key])
            self.assertFalse(agent.baseline_config["graph_encoder"])
            self.assertFalse(agent.baseline_config["surrogate_enhanced_head"])

    def test_learned_policy_respects_flat_action_mask(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("ppo", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [False, False, False, True, False],
            },
        )
        self.assertEqual(action, 3)
        self.assertTrue(action_info["action_mask_applied"])
        self.assertEqual(action_info["valid_action_count"], 1)
        self.assertEqual(action_info["action_probs"]["flat"], [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_dqn_policy_respects_action_mask(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("dqn", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [False, False, True, False, False],
            },
        )
        self.assertEqual(action, 2)
        self.assertTrue(action_info["action_mask_applied"])
        self.assertEqual(action_info["valid_action_count"], 1)
        self.assertEqual(action_info["action_probs"]["flat"], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_dueling_dqn_policy_respects_action_mask(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent("dueling_ddqn", random_seed=1, deterministic_action=True)
        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [False, True, False, False, False],
            },
        )
        self.assertEqual(action, 1)
        self.assertTrue(action_info["action_mask_applied"])
        self.assertEqual(action_info["valid_action_count"], 1)
        self.assertEqual(action_info["q_architecture"], "dueling")
        self.assertEqual(action_info["action_probs"]["flat"], [0.0, 1.0, 0.0, 0.0, 0.0])

    def test_top_journal_mechanism_aux_does_not_force_reactive_cache_fill(self) -> None:
        state = _minimal_semantic_state()
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = []
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            mechanism_aux_current_cache_fill_enabled=False,
        )

        targets = agent._build_mechanism_targets(state)

        self.assertEqual(targets["slow_target"], 0)

    def test_top_journal_mechanism_aux_still_targets_predictive_prefetch(self) -> None:
        state = _minimal_semantic_state()
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            mechanism_aux_current_cache_fill_enabled=False,
        )

        targets = agent._build_mechanism_targets(state)

        self.assertEqual(targets["slow_target"], 2)

    def test_latency_fallback_bias_targets_fast_head_only_when_low_risk_and_warm(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_a"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            latency_fallback_bias_enabled=True,
            latency_fallback_bias_strength=1.2,
            latency_fallback_confidence_floor=0.62,
        )

        targets = agent._build_mechanism_targets(state)

        self.assertEqual(targets["slow_target"], 0)
        self.assertEqual(targets["fast_target"], 1)
        self.assertGreaterEqual(targets["confidence_weight"], 0.62)

    def test_latency_fallback_suppresses_slow_mechanism_heads_only_when_candidate(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_a"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            mechanism_logit_bias_strength=1.0,
            latency_fallback_bias_enabled=True,
            latency_fallback_bias_strength=1.2,
            latency_fallback_confidence_floor=0.62,
            latency_fallback_slow_suppression_strength=1.2,
        )
        policy_output = {
            "slow_logits": torch.tensor([0.0, 1.0, 3.0]),
            "fast_logits": torch.tensor([0.0, 0.0]),
            "event_logits": torch.tensor([0.0, 1.0]),
        }

        adjusted = agent._apply_policy_adjustments(policy_output, state)

        self.assertLess(float(adjusted["slow_logits"][2]), float(policy_output["slow_logits"][2]))
        self.assertLess(float(adjusted["slow_logits"][1]), float(policy_output["slow_logits"][1]))
        self.assertGreater(float(adjusted["fast_logits"][1]), float(policy_output["fast_logits"][1]))
        self.assertTrue(adjusted["mechanism_bias_info"]["latency_fallback_candidate"])

    def test_steady_rsu_bias_targets_current_rsu_without_hard_override(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_a"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            mechanism_logit_bias_strength=1.0,
            steady_rsu_bias_enabled=True,
            steady_rsu_bias_strength=1.2,
            steady_rsu_confidence_floor=0.62,
        )
        policy_output = {
            "slow_logits": torch.tensor([0.0, 1.0, 3.0]),
            "fast_logits": torch.tensor([0.0, 0.0]),
            "event_logits": torch.tensor([0.0, 1.0]),
        }

        adjusted = agent._apply_policy_adjustments(policy_output, state)

        self.assertGreater(float(adjusted["fast_logits"][0]), float(policy_output["fast_logits"][0]))
        self.assertEqual(float(adjusted["fast_logits"][1]), float(policy_output["fast_logits"][1]))
        self.assertEqual(float(adjusted["slow_logits"][2]), float(policy_output["slow_logits"][2]))
        self.assertTrue(adjusted["mechanism_bias_info"]["steady_rsu_candidate"])

    def test_continuity_guard_keeps_prefetch_available_until_target_cache_ready(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            continuity_guard_enabled=True,
            handoff_target_alignment_guard_enabled=True,
            continuity_guard_logit_penalty=1.0,
            continuity_guard_prepare_boost=1.0,
            continuity_guard_confidence_threshold=0.5,
        )
        policy_output = {
            "slow_logits": torch.tensor([0.0, 0.0, 3.0]),
            "fast_logits": torch.tensor([0.0, 0.0]),
            "event_logits": torch.tensor([0.0, 0.0]),
        }

        adjusted = agent._apply_continuity_guard(policy_output, state)

        self.assertEqual(float(adjusted["slow_logits"][2]), float(policy_output["slow_logits"][2]))
        self.assertEqual(float(adjusted["event_logits"][1]), float(policy_output["event_logits"][1]))
        self.assertTrue(adjusted["continuity_guard_info"]["guard_triggered"])
        self.assertFalse(adjusted["continuity_guard_info"]["target_cache_ready_for_prepare"])

    def test_continuity_guard_boosts_prepare_after_target_cache_ready(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            continuity_guard_enabled=True,
            handoff_target_alignment_guard_enabled=True,
            continuity_guard_logit_penalty=1.0,
            continuity_guard_prepare_boost=1.0,
            continuity_guard_confidence_threshold=0.5,
        )
        policy_output = {
            "slow_logits": torch.tensor([0.0, 0.0, 3.0]),
            "fast_logits": torch.tensor([0.0, 0.0]),
            "event_logits": torch.tensor([0.0, 0.0]),
        }

        adjusted = agent._apply_continuity_guard(policy_output, state)

        self.assertLess(float(adjusted["slow_logits"][2]), float(policy_output["slow_logits"][2]))
        self.assertGreater(float(adjusted["event_logits"][1]), float(policy_output["event_logits"][1]))
        self.assertTrue(adjusted["continuity_guard_info"]["target_cache_ready_for_prepare"])

    def test_predictive_prepare_hard_override_ignores_low_policy_margin_when_enabled(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            deterministic_temporal_smoothing_enabled=True,
            predictive_prepare_hard_override_enabled=True,
            predictive_prepare_hard_override_score_threshold=0.0,
            predictive_prepare_hard_override_confidence_threshold=0.5,
        )
        policy_output = {
            "event_logits": torch.tensor([5.0, -5.0]),
        }
        selected_actions = {"slow": 0, "fast": 0, "event": 0}

        info = agent._apply_deterministic_temporal_smoothing(
            semantic_state=state,
            policy_output=policy_output,
            selected_actions=selected_actions,
            deterministic=True,
        )

        self.assertEqual(selected_actions["event"], 1)
        self.assertTrue(info["predictive_prepare_override_eligible"])
        self.assertTrue(info["override_triggered"])

    def test_backhaul_guard_caps_reactive_cache_fill_without_prediction_signal(self) -> None:
        state = _minimal_semantic_state()
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = []
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            backhaul_guard_enabled=True,
            backhaul_guard_max_reactive_fills_per_adapter=1,
        )
        first_actions = {"slow": 1, "fast": 0, "event": 0}
        second_actions = {"slow": 1, "fast": 0, "event": 0}

        first_info = agent._apply_backhaul_guard_to_actions(
            semantic_state=state,
            selected_actions=first_actions,
        )
        state["time_index"] = 2
        second_info = agent._apply_backhaul_guard_to_actions(
            semantic_state=state,
            selected_actions=second_actions,
        )

        self.assertFalse(first_info["guarded"])
        self.assertEqual(first_actions["slow"], 1)
        self.assertTrue(second_info["guarded"])
        self.assertEqual(second_actions["slow"], 0)

    def test_cache_warm_start_guard_prioritizes_current_cache_fill(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            cache_warm_start_guard_enabled=True,
        )
        actions = {"slow": 0, "fast": 0, "event": 1}

        guard_info = agent._apply_cache_warm_start_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "current_adapter_not_warm_cache_first")
        self.assertEqual(actions["slow"], 1)
        self.assertEqual(actions["event"], 0)

    def test_backhaul_guard_preserves_cache_warm_current_fill(self) -> None:
        state = _minimal_semantic_state()
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = []
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            backhaul_guard_enabled=True,
            backhaul_guard_max_reactive_fills_per_adapter=1,
            cache_warm_start_guard_enabled=True,
        )
        actions = {"slow": 0, "fast": 0, "event": 1}
        cache_warm_info = agent._apply_cache_warm_start_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )
        state["time_index"] = 2

        backhaul_info = agent._apply_backhaul_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
            cache_warm_guard_info=cache_warm_info,
        )

        self.assertFalse(backhaul_info["guarded"])
        self.assertEqual(backhaul_info["reason"], "cache_warm_guard_allows_current_fill")
        self.assertEqual(actions["slow"], 1)

    def test_cache_warm_start_guard_prefetches_before_prepare_when_current_is_warm(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            cache_warm_start_guard_enabled=True,
            cache_warm_start_guard_min_countdown=0.0,
        )
        actions = {"slow": 0, "fast": 0, "event": 1}

        guard_info = agent._apply_cache_warm_start_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "target_adapter_not_warm_prefetch_first")
        self.assertEqual(actions["slow"], 2)
        self.assertEqual(actions["event"], 0)

    def test_cache_warm_start_guard_defers_prefetch_outside_freshness_window(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a"] * 7 + ["rsu_b"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            cache_warm_start_guard_enabled=True,
            cache_warm_start_guard_min_countdown=0.0,
            cache_warm_start_guard_max_prefetch_countdown=6.0,
        )
        actions = {"slow": 0, "fast": 0, "event": 1}

        guard_info = agent._apply_cache_warm_start_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertFalse(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "target_prefetch_deferred_until_freshness_window")
        self.assertEqual(guard_info["handoff_countdown_steps"], 8.0)
        self.assertEqual(actions["slow"], 0)
        self.assertEqual(actions["event"], 1)

    def test_cache_warm_start_guard_prefetches_inside_freshness_window(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a"] * 5 + ["rsu_b"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            cache_warm_start_guard_enabled=True,
            cache_warm_start_guard_min_countdown=0.0,
            cache_warm_start_guard_max_prefetch_countdown=6.0,
        )
        actions = {"slow": 0, "fast": 0, "event": 1}

        guard_info = agent._apply_cache_warm_start_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "target_adapter_not_warm_prefetch_first")
        self.assertEqual(guard_info["handoff_countdown_steps"], 6.0)
        self.assertEqual(actions["slow"], 2)
        self.assertEqual(actions["event"], 0)

    def test_predictive_prefetch_admission_guard_defers_low_confidence_unaligned_prefetch(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = "rsu_a"
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = "rsu_b"
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_b"]
        state["predictions"]["prediction_confidence_by_vehicle"]["veh_1"] = 0.38
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            predictive_prefetch_admission_guard_enabled=True,
            predictive_prefetch_admission_min_confidence=0.55,
            predictive_prefetch_admission_require_distinct_next=True,
        )
        actions = {"slow": 2, "fast": 0, "event": 0}

        guard_info = agent._apply_predictive_prefetch_admission_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "low_confidence_unaligned_prefetch_deferred_to_prepare")
        self.assertFalse(guard_info["predicted_next_aligned"])
        self.assertEqual(actions["slow"], 0)
        self.assertEqual(actions["event"], 1)

    def test_predictive_prefetch_admission_guard_admits_confident_aligned_prefetch(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = "rsu_b"
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = "rsu_b"
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_b"]
        state["predictions"]["prediction_confidence_by_vehicle"]["veh_1"] = 0.61
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            predictive_prefetch_admission_guard_enabled=True,
            predictive_prefetch_admission_min_confidence=0.55,
            predictive_prefetch_admission_require_distinct_next=True,
        )
        actions = {"slow": 2, "fast": 0, "event": 0}

        guard_info = agent._apply_predictive_prefetch_admission_guard_to_actions(
            semantic_state=state,
            selected_actions=actions,
        )

        self.assertFalse(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "prefetch_admitted")
        self.assertTrue(guard_info["predicted_next_aligned"])
        self.assertEqual(actions["slow"], 2)
        self.assertEqual(actions["event"], 0)

    def test_action_schema_declares_discrete_contract(self) -> None:
        schema = ActionSchema.default_vec_workflow_schema()
        self.assertEqual(schema.discrete_action_count, 5)
        self.assertFalse(schema.supports_continuous_control)
        self.assertIn("semantic_discrete", schema.to_dict()["kind"])
        mask = ActionMaskBuilder(schema).build_mask({"current_workflow_node": {"node_id": "n1"}})
        self.assertEqual(mask, [True, False, True, True, False])

    def test_action_mask_builder_enforces_predictive_preconditions(self) -> None:
        state = _minimal_semantic_state()
        mask_info = ActionMaskBuilder().build_mask_info(state)
        self.assertEqual(mask_info["mask"], [True, True, True, True, True])
        self.assertEqual(mask_info["valid_action_count"], 5)

        warm_target_state = deepcopy(state)
        warm_target_state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        warm_info = ActionMaskBuilder().build_mask_info(warm_target_state)
        self.assertEqual(warm_info["mask"], [True, False, True, True, True])
        self.assertEqual(warm_info["invalid_reasons"]["1"], "target_adapter_already_ready")

        no_handoff_state = deepcopy(state)
        no_handoff_state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        no_handoff_info = ActionMaskBuilder().build_mask_info(no_handoff_state)
        self.assertEqual(no_handoff_info["mask"], [True, True, True, True, False])
        self.assertEqual(no_handoff_info["invalid_reasons"]["4"], "missing_distinct_handoff_target")

    def test_action_mask_builder_uses_first_non_current_rsu_for_prefetch(self) -> None:
        state = _minimal_semantic_state()
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = "rsu_a"
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_b"]

        mask_info = ActionMaskBuilder().build_mask_info(state)
        control = ActionAdapter().decode(1, state)

        self.assertTrue(mask_info["mask"][1])
        self.assertEqual(mask_info["semantic_preconditions"]["predicted_next_rsu_id"], "rsu_b")
        self.assertEqual(control.cache_action["rsu_id"], "rsu_b")
        self.assertFalse(control.metadata["invalid_action"])

    def test_hierarchical_policy_samples_masked_env_actions_without_projection(self) -> None:
        agent = build_agent("sa_ghmappo", random_seed=1)
        policy_output = {
            "slow_logits": torch.tensor([0.0, 0.0, 8.0]),
            "fast_logits": torch.tensor([2.0, 0.0]),
            "event_logits": torch.tensor([0.0, 8.0]),
        }

        actions, _, _, _, projection_info = agent._sample_actions(
            policy_output,
            deterministic=True,
            action_mask=[True, False, True, True, False],
        )

        self.assertIn(projection_info["projected_env_action"], {0, 2, 3})
        self.assertEqual(actions, agent._head_targets_for_env_action(projection_info["projected_env_action"]))
        self.assertFalse(projection_info["projection_applied"])
        self.assertEqual(projection_info["invalid_attempt_count"], 0)
        self.assertTrue(projection_info["masked_hierarchical_env_action_sampling"])

    def test_action_adapter_decodes_core_control_action(self) -> None:
        state = {
            "current_workflow_node": {"required_adapter": "adapter_tracking"},
            "vehicles": [{"vehicle_id": "veh_1", "associated_rsu_id": "rsu_a"}],
            "predictions": {"next_rsu_sequence": {"veh_1": ["rsu_b"]}},
        }
        control = ActionAdapter().decode(1, state)
        self.assertEqual(control.cache_action["strategy"], "predictive_prefetch")
        self.assertEqual(control.cache_action["rsu_id"], "rsu_b")
        self.assertFalse(control.metadata["invalid_action"])

    def test_action_adapter_marks_invalid_predictive_action_without_current_rsu_fallback(self) -> None:
        state = {
            "current_workflow_node": {"required_adapter": "adapter_tracking"},
            "vehicles": [{"vehicle_id": "veh_1", "associated_rsu_id": "rsu_a"}],
            "predictions": {"next_rsu_sequence": {"veh_1": ["rsu_a"]}},
        }

        control = ActionAdapter().decode(1, state)

        self.assertEqual(control.cache_action, {})
        self.assertTrue(control.metadata["invalid_action"])
        self.assertEqual(control.metadata["invalid_reason"], "missing_distinct_predicted_next_rsu")


if __name__ == "__main__":
    unittest.main()
