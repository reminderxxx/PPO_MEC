"""Algorithm-pool registry and action-schema contract tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

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
                "reactive_aging_lfu",
                "reactive_fifo",
                "reactive_greedy",
                "reactive_lfu",
                "reactive_lru",
                "reactive_random",
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
                "reactive_aging_lfu",
                "reactive_fifo",
                "reactive_greedy",
                "reactive_lfu",
                "reactive_lru",
                "reactive_random",
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

    def test_optional_transition_model_preserves_policy_initialization(self) -> None:
        without_model = build_agent(
            "sa_ghmappo",
            random_seed=17,
            deterministic_action=True,
            learned_transition_model_enabled=False,
        )
        with_model = build_agent(
            "sa_ghmappo",
            random_seed=17,
            deterministic_action=True,
            learned_transition_model_enabled=True,
        )

        without_state = without_model._network.state_dict()
        with_state = with_model._network.state_dict()
        self.assertEqual(set(without_state), set(with_state))
        for parameter_name in without_state:
            self.assertTrue(
                torch.equal(without_state[parameter_name], with_state[parameter_name]),
                parameter_name,
            )

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

    def test_sa_v19_profile_enables_handoff_risk_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v19_handoff_risk_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v19_handoff_risk_prd")
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["dag_aware_option_termination_enabled"])
        self.assertTrue(kwargs["handoff_risk_prd_enabled"])
        self.assertGreater(kwargs["handoff_risk_event_coef"], 0.0)
        self.assertGreater(kwargs["handoff_risk_option_coef"], 0.0)
        self.assertTrue(kwargs["handoff_risk_cost_dual_enabled"])

    def test_sa_v19_handoff_risk_credit_rewards_ready_over_failure(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            handoff_risk_prd_enabled=True,
            handoff_risk_event_coef=1.0,
            handoff_risk_option_coef=1.0,
            handoff_risk_failure_penalty=1.0,
            handoff_risk_ready_bonus=0.8,
            handoff_risk_prepare_bonus=0.3,
            handoff_risk_unprepared_penalty=0.4,
        )
        base_row = {
            "action": 4,
            "action_info": {
                "prepare_window_score": 0.8,
                "temporal_urgency": 0.7,
                "prediction_confidence": 0.8,
                "final_env_action": 4,
                "head_actions": {"event": 1},
            },
            "env_info": {
                "metrics_protocol": {
                    "predicted_handoff_signal": True,
                    "handoff_event_count": 1,
                }
            },
        }
        ready_row = deepcopy(base_row)
        ready_row["env_info"]["metrics_protocol"].update(
            {
                "handoff_ready": True,
                "handoff_failed": False,
                "migration_prepare_realized": True,
            }
        )
        failed_row = deepcopy(base_row)
        failed_row["action"] = 0
        failed_row["action_info"]["final_env_action"] = 0
        failed_row["action_info"]["head_actions"] = {"event": 0}
        failed_row["env_info"]["metrics_protocol"].update(
            {
                "handoff_ready": False,
                "handoff_failed": True,
                "migration_prepare_realized": False,
            }
        )

        self.assertGreater(
            agent._handoff_risk_prd_credit(ready_row),
            agent._handoff_risk_prd_credit(failed_row),
        )

    def test_sa_v20_profile_enables_idle_execution_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v20_idle_execution_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v20_idle_execution_prd")
        self.assertTrue(kwargs["handoff_risk_prd_enabled"])
        self.assertTrue(kwargs["idle_execution_prd_enabled"])
        self.assertGreater(kwargs["idle_execution_policy_coef"], 0.0)
        self.assertGreater(kwargs["idle_execution_option_coef"], 0.0)
        self.assertGreater(kwargs["idle_execution_current_rsu_delay_coef"], 0.0)

    def test_sa_v20_idle_execution_credit_prefers_low_risk_local(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            idle_execution_prd_enabled=True,
            idle_execution_policy_coef=1.0,
            idle_execution_option_coef=1.0,
            idle_execution_current_rsu_delay_coef=0.5,
            idle_execution_local_bonus=0.3,
        )
        base_row = {
            "action_info": {
                "prepare_window_score": 0.05,
                "temporal_urgency": 0.04,
                "prediction_confidence": 0.2,
                "option_gate": {"window_class": "idle_or_sparse"},
            },
            "decision_info": {"run_metadata": {"window_class": "idle_or_sparse"}},
            "env_info": {
                "metrics_protocol": {
                    "service_delay_sum": 4.0,
                    "handoff_failure_rate": 0.0,
                    "mechanism_success_rate": 0.0,
                }
            },
        }
        current_row = deepcopy(base_row)
        current_row["action"] = 3
        current_row["action_info"]["final_env_action"] = 3
        local_row = deepcopy(base_row)
        local_row["action"] = 2
        local_row["action_info"]["final_env_action"] = 2

        self.assertGreater(
            agent._idle_execution_prd_credit(local_row),
            agent._idle_execution_prd_credit(current_row),
        )
        self.assertLess(
            float(
                agent._option_gate_advantage(
                    row=current_row,
                    base_advantage=torch.tensor(0.0),
                ).item()
            ),
            0.0,
        )

    def test_sa_v21_profile_reactivates_net_utility_efficiency_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v21_efficiency_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v21_efficiency_prd")
        self.assertTrue(kwargs["idle_execution_prd_enabled"])
        self.assertTrue(kwargs["handoff_risk_prd_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["net_utility_cost_dual_enabled"])
        self.assertGreater(kwargs["idle_execution_policy_coef"], 0.30)
        self.assertGreater(kwargs["idle_execution_option_coef"], 0.42)
        self.assertGreater(kwargs["net_utility_backhaul_coef"], 0.0)
        self.assertLess(kwargs["mechanism_window_weight"], 1.52)

    def test_sa_v22_profile_enables_validated_utility_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v22_validated_utility_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v22_validated_utility_prd")
        self.assertTrue(kwargs["idle_execution_prd_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["net_utility_cost_dual_enabled"])
        self.assertGreater(kwargs["net_utility_failed_mechanism_penalty"], 0.0)
        self.assertGreater(kwargs["net_utility_failed_mechanism_backhaul_coef"], 0.0)
        self.assertGreater(kwargs["idle_execution_policy_coef"], 0.46)
        self.assertLess(kwargs["prepare_action_prior_weight"], 0.50)

    def test_sa_v23_profile_enables_counterfactual_constrained_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v23_counterfactual_constrained_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v23_counterfactual_constrained_prd")
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["option_gate_counterfactual_prd_enabled"])
        self.assertFalse(kwargs["option_gate_mechanism_preserve_enabled"])
        self.assertGreater(kwargs["counterfactual_teacher_event_coef"], 0.0)
        self.assertGreater(kwargs["counterfactual_teacher_option_coef"], 0.0)
        self.assertGreater(kwargs["mechanism_window_weight"], 1.18)
        self.assertLess(kwargs["net_utility_mechanism_window_failed_penalty_scale"], 1.0)

    def test_sa_v24_profile_enables_tail_risk_constrained_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v24_tail_risk_constrained_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v24_tail_risk_constrained_prd")
        self.assertTrue(kwargs["tail_risk_prd_enabled"])
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["option_gate_counterfactual_prd_enabled"])
        self.assertTrue(kwargs["option_gate_mechanism_preserve_enabled"])
        self.assertGreater(kwargs["tail_risk_policy_coef"], 0.0)
        self.assertGreater(kwargs["tail_risk_event_coef"], 0.0)
        self.assertGreater(kwargs["tail_risk_option_coef"], 0.0)
        self.assertGreater(kwargs["net_utility_mechanism_window_failed_penalty_scale"], 0.5)

    def test_sa_v25_profile_enables_opportunity_risk_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v25_opportunity_risk_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v25_opportunity_risk_prd")
        self.assertTrue(kwargs["tail_risk_prd_enabled"])
        self.assertTrue(kwargs["opportunity_prd_enabled"])
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["option_gate_mechanism_preserve_enabled"])
        self.assertGreater(kwargs["opportunity_policy_coef"], 0.0)
        self.assertGreater(kwargs["opportunity_event_coef"], 0.0)
        self.assertGreater(kwargs["opportunity_option_coef"], 0.0)
        self.assertGreater(kwargs["opportunity_reward_surplus_coef"], 0.0)
        self.assertLess(kwargs["tail_risk_policy_coef"], 0.52)

    def test_sa_v26_profile_enables_safe_counterfactual_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v26_mechanism_safe_counterfactual_prd"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v26_mechanism_safe_counterfactual_prd")
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["option_gate_counterfactual_prd_enabled"])
        self.assertTrue(kwargs["option_gate_mechanism_preserve_enabled"])
        self.assertFalse(kwargs["tail_risk_prd_enabled"])
        self.assertFalse(kwargs["opportunity_prd_enabled"])
        self.assertGreater(kwargs["counterfactual_teacher_mechanism_bonus"], 0.0)
        self.assertGreater(kwargs["counterfactual_teacher_invalid_mechanism_penalty"], 0.0)
        self.assertGreater(kwargs["idle_execution_mechanism_penalty"], 0.72)
        self.assertLess(kwargs["counterfactual_teacher_option_coef"], 0.72)

    def test_sa_v27_profile_enables_conservative_advantage_imitation(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v27_conservative_advantage_imitation"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v27_conservative_advantage_imitation")
        self.assertTrue(kwargs["idle_execution_prd_enabled"])
        self.assertTrue(kwargs["handoff_risk_prd_enabled"])
        self.assertTrue(kwargs["conservative_imitation_enabled"])
        self.assertFalse(kwargs["tail_risk_prd_enabled"])
        self.assertFalse(kwargs["opportunity_prd_enabled"])
        self.assertFalse(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertGreater(kwargs["heuristic_imitation_coef"], 0.0)
        self.assertGreater(kwargs["conservative_imitation_failure_coef"], 0.0)
        self.assertLess(kwargs["conservative_imitation_success_decay"], 1.0)

    def test_sa_v27_conservative_imitation_weight_targets_low_reward_failures(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            conservative_imitation_enabled=True,
            conservative_imitation_min_weight=0.08,
            conservative_imitation_max_weight=1.45,
            conservative_imitation_shortfall_coef=0.70,
            conservative_imitation_failure_coef=0.85,
            conservative_imitation_mismatch_coef=0.22,
            conservative_imitation_success_decay=0.20,
        )
        failed_row = {
            "reward": -0.8,
            "action": 4,
            "action_info": {"final_env_action": 4, "head_actions": {"event": 1}},
            "env_info": {
                "metrics_protocol": {
                    "migration_prepare_requested": True,
                    "mechanism_success_strict": False,
                }
            },
        }
        success_row = {
            "reward": 1.2,
            "action": 3,
            "action_info": {"final_env_action": 3, "head_actions": {"event": 0}},
            "env_info": {
                "metrics_protocol": {
                    "handoff_ready": True,
                    "mechanism_success_strict": True,
                }
            },
        }

        failed_weight = agent._conservative_imitation_weight(
            failed_row,
            reward_floor=0.0,
            student_action=4,
            teacher_action=3,
        )
        success_weight = agent._conservative_imitation_weight(
            success_row,
            reward_floor=0.0,
            student_action=3,
            teacher_action=3,
        )

        self.assertGreater(failed_weight, success_weight)
        self.assertLessEqual(failed_weight, 1.45)
        self.assertGreaterEqual(success_weight, 0.08)

    def test_sa_v28_profile_enables_mechanism_credit_and_focal_aux(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v28_credit_focal_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["update_every"], 8)
        self.assertEqual(defaults["train_window_count"], 20)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v28_credit_focal_mappo")
        self.assertTrue(kwargs["idle_execution_prd_enabled"])
        self.assertTrue(kwargs["handoff_risk_prd_enabled"])
        self.assertTrue(kwargs["mechanism_credit_prd_enabled"])
        self.assertTrue(kwargs["mechanism_focal_aux_enabled"])
        self.assertGreater(kwargs["mechanism_credit_event_coef"], kwargs["mechanism_credit_policy_coef"])
        self.assertGreater(kwargs["mechanism_credit_option_coef"], 0.0)
        self.assertFalse(kwargs["tail_risk_prd_enabled"])
        self.assertFalse(kwargs["opportunity_prd_enabled"])
        self.assertFalse(kwargs["counterfactual_teacher_prd_enabled"])

    def test_sa_v28_mechanism_credit_rewards_success_and_penalizes_misses(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_credit_prd_enabled=True,
            mechanism_credit_policy_coef=0.30,
            mechanism_credit_event_coef=0.85,
            mechanism_credit_option_coef=0.42,
            mechanism_credit_clip=1.80,
            mechanism_credit_success_bonus=1.0,
            mechanism_credit_prepare_bonus=0.55,
            mechanism_credit_ready_bonus=0.85,
            mechanism_credit_prefetch_hit_bonus=0.65,
            mechanism_credit_miss_penalty=0.55,
            mechanism_credit_false_positive_penalty=0.38,
        )
        base_action_info = {
            "prepare_window_score": 0.8,
            "temporal_urgency": 0.7,
            "prediction_confidence": 0.9,
            "gate_pass": True,
            "raw_handoff_candidate": True,
            "predicted_handoff_target_valid": True,
        }
        success_row = {
            "action": 4,
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "action_info": {
                **base_action_info,
                "final_env_action": 4,
                "head_actions": {"event": 1},
            },
            "env_info": {
                "metrics_protocol": {
                    "handoff_event_count": 1,
                    "predicted_handoff_signal": True,
                    "has_predicted_handoff_target": True,
                    "migration_prepare_realized": True,
                    "handoff_ready": True,
                    "prefetch_validated_hit": True,
                    "mechanism_success_strict": True,
                }
            },
        }
        missed_row = {
            "action": 2,
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "action_info": {
                **base_action_info,
                "final_env_action": 2,
                "head_actions": {"event": 0},
            },
            "env_info": {
                "metrics_protocol": {
                    "handoff_event_count": 1,
                    "predicted_handoff_signal": True,
                    "has_predicted_handoff_target": True,
                    "handoff_failed": True,
                    "mechanism_success_strict": False,
                }
            },
        }
        false_positive_row = {
            "action": 4,
            "decision_info": {"run_metadata": {"window_class": "idle_or_sparse"}},
            "action_info": {
                "prepare_window_score": 0.0,
                "temporal_urgency": 0.0,
                "prediction_confidence": 0.0,
                "gate_pass": False,
                "final_env_action": 4,
                "head_actions": {"event": 1},
            },
            "env_info": {"metrics_protocol": {"mechanism_success_strict": False}},
        }

        success_credit = agent._mechanism_credit_prd_credit(success_row)
        missed_credit = agent._mechanism_credit_prd_credit(missed_row)
        false_positive_credit = agent._mechanism_credit_prd_credit(false_positive_row)

        self.assertGreater(success_credit, 0.0)
        self.assertLess(missed_credit, 0.0)
        self.assertLess(false_positive_credit, 0.0)
        self.assertGreater(success_credit, abs(missed_credit))

    def test_sa_v29_profile_enables_digital_twin_handoff_fusion(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v29_dt_fused_credit_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v29_dt_fused_credit_mappo")
        self.assertTrue(kwargs["digital_twin_handoff_fusion_enabled"])
        self.assertTrue(kwargs["mechanism_credit_prd_enabled"])
        self.assertTrue(kwargs["mechanism_focal_aux_enabled"])
        self.assertGreater(kwargs["digital_twin_handoff_event_scale"], kwargs["digital_twin_handoff_slow_scale"])
        self.assertGreater(kwargs["mechanism_credit_event_coef"], kwargs["mechanism_credit_policy_coef"])

    def test_sa_v29_digital_twin_handoff_features_reach_network_output(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["position_x"] = 9.5
        state["vehicles"][0]["position_y"] = 0.0
        state["rsus"][0]["position_x"] = 0.0
        state["rsus"][0]["position_y"] = 0.0
        state["rsus"][0]["coverage_radius"] = 10.0
        state["current_node_service_steps_remaining"] = 3.0
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_b", "rsu_b"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_handoff_event_scale=0.95,
        )

        output = agent._forward_policy(state)
        encoded = output["encoded"]

        self.assertIn("digital_twin_handoff_fusion_enabled", encoded)
        self.assertAlmostEqual(float(encoded["digital_twin_handoff_fusion_enabled"].item()), 1.0)
        self.assertGreater(float(encoded["digital_twin_handoff_target_differs"].item()), 0.0)
        self.assertGreater(float(encoded["digital_twin_handoff_boundary_urgency"].item()), 0.9)
        self.assertGreater(float(encoded["digital_twin_handoff_service_pressure"].item()), 0.0)

    def test_sa_v30_profile_enables_dt_policy_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v30_dt_prior_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v30_dt_prior_mappo")
        self.assertTrue(kwargs["digital_twin_handoff_fusion_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_enabled"])
        self.assertGreater(kwargs["digital_twin_policy_prior_logit_bias"], 0.0)
        self.assertGreater(kwargs["digital_twin_policy_prior_distill_coef"], 0.0)
        self.assertGreater(kwargs["mechanism_credit_event_coef"], kwargs["mechanism_credit_policy_coef"])

    def test_sa_v30_dt_policy_prior_boosts_event_prepare_margin(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["position_x"] = 9.5
        state["vehicles"][0]["position_y"] = 0.0
        state["rsus"][0]["position_x"] = 0.0
        state["rsus"][0]["position_y"] = 0.0
        state["rsus"][0]["coverage_radius"] = 10.0
        base_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=False,
        )
        prior_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=True,
            digital_twin_policy_prior_logit_bias=3.0,
            digital_twin_policy_prior_prepare_threshold=0.0,
            digital_twin_policy_prior_confidence_floor=0.1,
        )

        base_output = base_agent._forward_policy(state)
        prior_output = prior_agent._forward_policy(state)
        annotation = prior_agent._build_digital_twin_policy_prior_annotation(state)
        base_margin = float((base_output["event_logits"][1] - base_output["event_logits"][0]).item())
        prior_margin = float((prior_output["event_logits"][1] - prior_output["event_logits"][0]).item())

        self.assertTrue(annotation["apply"])
        self.assertEqual(annotation["event_target"], 1)
        self.assertIn("digital_twin_policy_prior_info", prior_output)
        self.assertGreater(prior_margin, base_margin)

    def test_sa_raw_policy_mode_bypasses_policy_adjustments(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["position_x"] = 9.5
        state["vehicles"][0]["position_y"] = 0.0
        state["rsus"][0]["position_x"] = 0.0
        state["rsus"][0]["position_y"] = 0.0
        state["rsus"][0]["coverage_radius"] = 10.0
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=True,
            digital_twin_policy_prior_logit_bias=3.0,
            digital_twin_policy_prior_prepare_threshold=0.0,
            digital_twin_policy_prior_confidence_floor=0.1,
        )

        safety_output = agent._forward_policy(state)
        raw_output = agent._forward_policy(
            state,
            run_metadata={"policy_evaluation_mode": "raw_policy"},
        )

        self.assertIn("digital_twin_policy_prior_info", safety_output)
        self.assertNotIn("digital_twin_policy_prior_info", raw_output)
        safety_margin = float((safety_output["event_logits"][1] - safety_output["event_logits"][0]).item())
        raw_margin = float((raw_output["event_logits"][1] - raw_output["event_logits"][0]).item())
        self.assertGreater(safety_margin, raw_margin)

    def test_sa_v31_profile_enables_handoff_pacing_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v31_handoff_pacing_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v31_handoff_pacing_mappo")
        self.assertTrue(kwargs["digital_twin_policy_prior_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_pacing_enabled"])
        self.assertGreater(kwargs["digital_twin_policy_prior_pacing_fast_scale"], 1.0)
        self.assertGreater(kwargs["digital_twin_policy_prior_pacing_event_suppression"], 0.0)

    def test_sa_v31_handoff_pacing_prior_boosts_fast_fallback(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["position_x"] = 9.5
        state["vehicles"][0]["position_y"] = 0.0
        state["rsus"][0]["position_x"] = 0.0
        state["rsus"][0]["position_y"] = 0.0
        state["rsus"][0]["coverage_radius"] = 10.0
        base_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=False,
        )
        pacing_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=True,
            digital_twin_policy_prior_logit_bias=3.5,
            digital_twin_policy_prior_pacing_enabled=True,
            digital_twin_policy_prior_pacing_threshold=0.0,
            digital_twin_policy_prior_pacing_fast_scale=1.4,
            digital_twin_policy_prior_pacing_event_suppression=0.8,
            digital_twin_policy_prior_pacing_slow_suppression=0.7,
        )

        base_output = base_agent._forward_policy(state)
        pacing_output = pacing_agent._forward_policy(state)
        annotation = pacing_agent._build_digital_twin_policy_prior_annotation(state)
        base_fast_margin = float((base_output["fast_logits"][1] - base_output["fast_logits"][0]).item())
        pacing_fast_margin = float((pacing_output["fast_logits"][1] - pacing_output["fast_logits"][0]).item())
        base_event_margin = float((base_output["event_logits"][1] - base_output["event_logits"][0]).item())
        pacing_event_margin = float((pacing_output["event_logits"][1] - pacing_output["event_logits"][0]).item())

        self.assertTrue(annotation["apply"])
        self.assertTrue(annotation["pacing_target"])
        self.assertEqual(annotation["fast_target"], 1)
        self.assertGreater(pacing_fast_margin, base_fast_margin)
        self.assertLess(pacing_event_margin, base_event_margin)

    def test_sa_v32_profile_enables_dt_continuation_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v32_dt_continuation_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v32_dt_continuation_mappo")
        self.assertTrue(kwargs["digital_twin_policy_prior_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_pacing_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_env_action_bias_enabled"])
        self.assertGreater(kwargs["digital_twin_policy_prior_env_action_logit_bias"], 0.0)
        self.assertGreater(kwargs["digital_twin_policy_prior_continuation_prepare_scale"], 1.0)

    def test_sa_v32_dt_continuation_prior_biases_env_prepare_score(self) -> None:
        state = _minimal_semantic_state()
        state["vehicles"][0]["position_x"] = 9.5
        state["vehicles"][0]["position_y"] = 0.0
        state["rsus"][0]["position_x"] = 0.0
        state["rsus"][0]["position_y"] = 0.0
        state["rsus"][0]["coverage_radius"] = 10.0
        base_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=False,
        )
        continuation_agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_handoff_fusion_enabled=True,
            digital_twin_policy_prior_enabled=True,
            digital_twin_policy_prior_logit_bias=4.0,
            digital_twin_policy_prior_pacing_enabled=True,
            digital_twin_policy_prior_pacing_threshold=0.0,
            digital_twin_policy_prior_env_action_bias_enabled=True,
            digital_twin_policy_prior_env_action_logit_bias=5.0,
            digital_twin_policy_prior_continuation_threshold=0.0,
            digital_twin_policy_prior_continuation_prepare_scale=1.3,
        )
        run_metadata = {"window_class": "mechanism_activating"}

        base_output = base_agent._forward_policy(state, run_metadata=run_metadata)
        continuation_output = continuation_agent._forward_policy(state, run_metadata=run_metadata)
        annotation = continuation_agent._build_digital_twin_policy_prior_annotation(
            state,
            run_metadata=run_metadata,
        )
        base_scores = base_agent._hierarchical_env_action_scores(base_output)
        continuation_scores = continuation_agent._hierarchical_env_action_scores(continuation_output)

        self.assertTrue(annotation["apply"])
        self.assertTrue(annotation["continuation_target"])
        self.assertEqual(annotation["env_target"], 4)
        self.assertIn("env_action_logits_bias", continuation_output)
        self.assertGreater(float(continuation_output["env_action_logits_bias"][4].item()), 0.0)
        self.assertGreater(float(continuation_scores[4].item()), float(base_scores[4].item()))
        self.assertGreater(float(continuation_scores[4].item()), float(continuation_scores[3].item()))

    def test_sa_v33_profile_enables_env_action_ppo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v33_env_action_ppo_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v33_env_action_ppo_mappo")
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertGreater(kwargs["env_action_ppo_coef"], 0.0)
        self.assertGreater(kwargs["env_action_ppo_teacher_coef"], 0.0)
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertLess(kwargs["digital_twin_policy_prior_env_action_logit_bias"], 4.80)

    def test_sa_v33_env_action_ppo_optimizes_executed_action(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_ppo_enabled=True,
            env_action_ppo_coef=1.0,
            env_action_ppo_advantage_blend=0.5,
            env_action_ppo_teacher_coef=0.0,
            env_action_ppo_mechanism_focus=0.0,
        )
        action_mask = [True, True, True, True, True]
        policy_output = agent._forward_policy(state)
        old_log_prob, _, _ = agent._env_action_distribution_statistics(
            policy_output=policy_output,
            env_action=4,
            action_mask=action_mask,
        )

        loss = agent._compute_env_action_ppo_loss(
            batch_outputs=[policy_output],
            batch_action_masks=[action_mask],
            batch_actions=torch.tensor([4], dtype=torch.long),
            old_env_action_log_probs=old_log_prob.detach().reshape(1),
            base_advantage=torch.tensor([1.0], dtype=torch.float32),
            event_advantage=torch.tensor([1.0], dtype=torch.float32),
            batch_rows=[{"action": 4, "action_info": {"final_env_action": 4}}],
        )

        self.assertLess(float(loss.item()), 0.0)

    def test_sa_v34_profile_enables_adaptive_wait_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v34_adaptive_wait_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v34_adaptive_wait_mappo")
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_adaptive_wait_enabled"])
        self.assertGreater(kwargs["digital_twin_policy_prior_continuation_wait_scale"], 1.0)
        self.assertLess(kwargs["digital_twin_policy_prior_continuation_prepare_scale"], 1.0)
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], 0.0)

    def test_sa_v35_profile_relaxes_hard_guards_and_keeps_action_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v35_guard_relaxed_action_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v35_guard_relaxed_action_mappo")

        self.assertFalse(kwargs["continuity_guard_enabled"])
        self.assertFalse(kwargs["handoff_target_alignment_guard_enabled"])
        self.assertFalse(kwargs["backhaul_guard_enabled"])
        self.assertFalse(kwargs["cache_warm_start_guard_enabled"])
        self.assertFalse(kwargs["predictive_prefetch_admission_guard_enabled"])
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertTrue(kwargs["digital_twin_policy_prior_adaptive_wait_enabled"])
        self.assertGreater(kwargs["env_action_ppo_teacher_coef"], 0.60)
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], 0.0)
        self.assertGreater(kwargs["digital_twin_policy_prior_continuation_wait_scale"], 1.0)
        self.assertLess(kwargs["digital_twin_policy_prior_continuation_prepare_scale"], 1.0)

    def test_sa_v36_profile_enables_counterfactual_margin_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v36_counterfactual_margin_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v36_counterfactual_margin_mappo")

        self.assertFalse(kwargs["continuity_guard_enabled"])
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertGreater(kwargs["env_action_counterfactual_margin_coef"], 0.0)
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], 0.0)

    def test_sa_v37_profile_gates_counterfactual_margin_with_advantage(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v37_advantage_gated_counterfactual_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v37_advantage_gated_counterfactual_mappo")

        self.assertFalse(kwargs["continuity_guard_enabled"])
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertLess(kwargs["env_action_counterfactual_margin_coef"], 0.10)
        self.assertGreater(kwargs["env_action_counterfactual_margin_advantage_gate"], 0.0)
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], 0.0)

    def test_sa_v38_profile_uses_undiscounted_dt_prior_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v38_undiscounted_dt_prior_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        self.assertEqual(defaults["gamma"], 1.0)
        self.assertEqual(defaults["gae_lambda"], 1.0)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v38_undiscounted_dt_prior_mappo")

        self.assertFalse(kwargs["continuity_guard_enabled"])
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertFalse(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertGreater(kwargs["digital_twin_policy_prior_distill_coef"], 0.10)
        self.assertGreater(kwargs["digital_twin_policy_prior_advantage_weight"], 0.70)
        self.assertGreater(kwargs["env_action_ppo_teacher_coef"], 0.70)
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], 0.0)

    def test_sa_v39_profile_uses_delayed_mechanism_credit_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v39_delayed_credit_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v39_delayed_credit_mappo")

        self.assertTrue(kwargs["delayed_mechanism_credit_enabled"])
        self.assertGreater(kwargs["delayed_mechanism_credit_policy_coef"], 0.0)
        self.assertGreater(kwargs["delayed_mechanism_credit_event_coef"], kwargs["delayed_mechanism_credit_policy_coef"])
        self.assertEqual(kwargs["delayed_mechanism_credit_horizon"], 5)
        self.assertLess(kwargs["env_action_ppo_teacher_coef"], 0.60)
        self.assertFalse(kwargs["env_action_counterfactual_margin_enabled"])

    def test_sa_v40_profile_uses_advantage_weighted_behavior_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v40_advantage_weighted_behavior_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v40_advantage_weighted_behavior_mappo")

        self.assertTrue(kwargs["advantage_weighted_behavior_regularization_enabled"])
        self.assertGreater(kwargs["advantage_weighted_behavior_coef"], 0.0)
        self.assertTrue(kwargs["delayed_mechanism_credit_enabled"])
        self.assertLess(kwargs["env_action_ppo_teacher_coef"], 0.46)
        self.assertGreater(kwargs["advantage_weighted_behavior_positive_coef"], 1.0)

    def test_sa_v41_profile_uses_conservative_recovery_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v41_conservative_recovery_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v41_conservative_recovery_mappo")

        self.assertTrue(kwargs["advantage_weighted_behavior_regularization_enabled"])
        self.assertGreater(kwargs["advantage_weighted_behavior_coef"], 0.0)
        self.assertEqual(kwargs["advantage_weighted_behavior_positive_coef"], 0.0)
        self.assertGreater(kwargs["advantage_weighted_behavior_negative_coef"], 1.0)
        self.assertTrue(kwargs["delayed_mechanism_credit_enabled"])
        self.assertLess(kwargs["env_action_ppo_ratio_barrier_margin"], 0.30)

    def test_sa_v42_profile_uses_completion_aligned_offset_free_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v42_completion_aligned_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["max_steps"], 22)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertEqual(defaults["gamma"], 1.0)
        self.assertEqual(defaults["gae_lambda"], 1.0)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v42_completion_aligned_mappo")

        self.assertTrue(kwargs["delayed_mechanism_credit_enabled"])
        self.assertTrue(kwargs["advantage_weighted_behavior_regularization_enabled"])
        self.assertEqual(kwargs["advantage_weighted_behavior_positive_coef"], 0.0)
        self.assertGreater(kwargs["advantage_weighted_behavior_negative_coef"], 1.0)
        self.assertLess(kwargs["env_action_ppo_teacher_coef"], 0.42)
        self.assertGreater(kwargs["delayed_mechanism_credit_failure_penalty"], 1.02)

    def test_sa_v43_profile_uses_strict_opportunity_offset_free_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v43_strict_opportunity_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v43_strict_opportunity_mappo")

        self.assertTrue(kwargs["delayed_mechanism_credit_enabled"])
        self.assertTrue(kwargs["delayed_mechanism_credit_strict_opportunity_enabled"])
        self.assertGreater(kwargs["delayed_mechanism_credit_context_gate"], 0.40)
        self.assertGreater(kwargs["delayed_mechanism_credit_failure_penalty"], 1.5)
        self.assertLess(kwargs["prepare_action_prior_weight"], 0.10)
        self.assertGreater(kwargs["advantage_weighted_behavior_negative_coef"], 1.4)

    def test_sa_v44_profile_uses_opportunity_constrained_policy(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v44_opportunity_constrained_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v44_opportunity_constrained_mappo")

        self.assertTrue(kwargs["delayed_mechanism_credit_strict_opportunity_enabled"])
        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertGreater(kwargs["opportunity_constrained_prepare_penalty"], 6.0)
        self.assertGreater(kwargs["opportunity_constrained_prefetch_penalty"], 3.0)
        self.assertGreaterEqual(kwargs["opportunity_constrained_reliability_floor"], 0.28)
        self.assertLess(kwargs["mechanism_aux_coef"], 0.01)
        self.assertGreater(kwargs["counterfactual_teacher_invalid_mechanism_penalty"], 1.0)

    def test_sa_v45_profile_balances_refresh_after_opportunity_constraint(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v45_balanced_refresh_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v45_balanced_refresh_mappo")

        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertLess(kwargs["opportunity_constrained_reliability_floor"], 0.28)
        self.assertLess(kwargs["opportunity_constrained_prepare_penalty"], 6.0)
        self.assertGreater(kwargs["opportunity_constrained_current_bias"], 2.5)
        self.assertLess(kwargs["opportunity_constrained_local_bias"], 0.2)
        self.assertLess(kwargs["mechanism_aux_coef"], 0.01)

    def test_sa_v46_profile_enables_constrained_net_utility_prd(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v46_net_utility_constrained_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertLess(defaults["clip_ratio"], 0.05)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v46_net_utility_constrained_mappo")

        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["net_utility_cost_dual_enabled"])
        self.assertFalse(kwargs["net_utility_option_termination_enabled"])
        self.assertGreater(kwargs["net_utility_backhaul_coef"], 0.20)
        self.assertGreater(kwargs["net_utility_failed_mechanism_backhaul_coef"], 0.30)
        self.assertGreater(kwargs["opportunity_constrained_prepare_penalty"], 7.0)
        self.assertGreater(kwargs["advantage_weighted_behavior_negative_coef"], 1.30)

    def test_sa_v47_profile_enables_service_backhaul_aware_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v47_service_backhaul_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertLess(defaults["learning_rate"], 2.6e-5)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v47_service_backhaul_mappo")

        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["backhaul_guard_enabled"])
        self.assertTrue(kwargs["backhaul_aware_policy_enabled"])
        self.assertGreater(kwargs["backhaul_aware_service_fill_bias"], 2.0)
        self.assertGreater(kwargs["backhaul_aware_redundant_fill_penalty"], 2.0)
        self.assertGreater(kwargs["backhaul_aware_no_signal_prepare_penalty"], 1.5)
        self.assertGreater(kwargs["net_utility_failed_mechanism_backhaul_coef"], 0.40)

    def test_sa_v48_profile_enables_service_fill_without_no_signal_suppression(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import PROFILE_DEFAULTS, build_sa_ghmappo_profile_kwargs

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v48_service_fill_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertLess(defaults["clip_ratio"], 0.05)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v48_service_fill_mappo")

        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertFalse(kwargs["backhaul_guard_enabled"])
        self.assertTrue(kwargs["backhaul_aware_policy_enabled"])
        self.assertGreater(kwargs["backhaul_aware_service_fill_bias"], 1.0)
        self.assertLess(kwargs["backhaul_aware_redundant_fill_penalty"], 1.0)
        self.assertEqual(kwargs["backhaul_aware_no_signal_prefetch_penalty"], 0.0)
        self.assertEqual(kwargs["backhaul_aware_no_signal_prepare_penalty"], 0.0)
        self.assertGreater(kwargs["net_utility_failed_mechanism_backhaul_coef"], 0.35)

    def test_sa_v49_profile_enables_retrospective_handoff_auxiliary(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        defaults = PROFILE_DEFAULTS["top_journal_mechanism_v49_retrospective_handoff_mappo"]
        self.assertEqual(defaults["episodes"], 128)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertEqual(defaults["prediction_horizon"], 8)
        self.assertIn("top_journal_mechanism_v49_retrospective_handoff_mappo", MECHANISM_COVERAGE_PROFILES)
        kwargs = build_sa_ghmappo_profile_kwargs("top_journal_mechanism_v49_retrospective_handoff_mappo")

        self.assertTrue(kwargs["opportunity_constrained_policy_enabled"])
        self.assertTrue(kwargs["net_utility_prd_enabled"])
        self.assertTrue(kwargs["retrospective_handoff_aux_enabled"])
        self.assertGreater(kwargs["retrospective_handoff_aux_transition_weight"], 1.0)
        self.assertGreater(kwargs["mechanism_aux_coef"], 0.01)
        self.assertFalse(kwargs["backhaul_guard_enabled"])

    def test_sa_v49_training_window_plan_requires_mechanism_coverage(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import build_training_window_plan

        class Args:
            window_mode = "mixed_informative"
            mechanism_window_oversample_ratio = 2.25
            handoff_imminent_oversample_ratio = 1.50
            target_mismatch_sample_weight = 1.50
            min_mechanism_activating_windows = 4

        active_window = {"window_id": "active_0", "window_class": "active_non_mechanism"}
        mechanism_window = {"window_id": "mechanism_0", "window_class": "mechanism_activating"}

        plan = build_training_window_plan(
            {
                "selected_windows": [active_window],
                "mechanism_activating_windows": [mechanism_window],
            },
            Args(),
        )

        self.assertIn("mechanism_activating", [item["window_class"] for item in plan])

        with self.assertRaises(RuntimeError):
            build_training_window_plan(
                {
                    "selected_windows": [active_window],
                    "mechanism_activating_windows": [],
                },
                Args(),
            )

    def test_mechanism_window_classification_is_not_predictor_ratio_gated(self) -> None:
        from inspect import signature

        from src.evaluators.main_results_support import resolve_window_candidates

        default_threshold = signature(resolve_window_candidates).parameters[
            "activating_handoff_prediction_ratio_threshold"
        ].default

        self.assertEqual(default_threshold, 0.0)

    def test_sa_v50_profile_extends_handoff_horizon_and_auxiliary_eta(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v50_long_horizon_handoff_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["retrospective_handoff_aux_enabled"])
        self.assertEqual(kwargs["retrospective_handoff_aux_max_eta"], 16.0)
        self.assertGreater(kwargs["retrospective_handoff_aux_transition_weight"], 1.6)
        self.assertGreater(kwargs["prepare_action_prior_weight"], 0.06)

    def test_sa_v51_profile_uses_physical_transfer_horizon_and_stronger_auxiliary(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v51_physical_transfer_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["retrospective_handoff_aux_enabled"])
        self.assertEqual(kwargs["retrospective_handoff_aux_max_eta"], 16.0)
        self.assertGreater(kwargs["retrospective_handoff_aux_transition_weight"], 2.0)
        self.assertGreater(kwargs["prepare_action_prior_weight"], 0.08)

    def test_sa_v52_profile_enables_net_advantage_gate(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v52_net_advantage_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["net_advantage_prepare_gate_enabled"])
        self.assertGreater(kwargs["net_advantage_prepare_gate_policy_coef"], 0.0)
        self.assertGreater(kwargs["net_advantage_prepare_gate_event_coef"], 0.0)
        self.assertLess(kwargs["mechanism_aux_coef"], 0.020)
        self.assertLess(kwargs["prepare_action_prior_weight"], 0.060)
        self.assertGreater(kwargs["net_utility_failed_mechanism_backhaul_coef"], 0.40)

    def test_sa_v53_profile_enables_service_cache_net_advantage_gate(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v53_service_net_advantage_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["net_advantage_prepare_gate_enabled"])
        self.assertGreater(kwargs["net_advantage_prepare_gate_service_fill_scale"], 0.80)
        self.assertGreater(kwargs["net_advantage_prepare_gate_local_penalty_scale"], 0.60)
        self.assertGreater(kwargs["idle_execution_current_rsu_delay_coef"], 0.45)
        self.assertLess(kwargs["idle_execution_local_bonus"], 0.50)
        self.assertGreater(kwargs["opportunity_current_rsu_efficiency_coef"], 0.30)

    def test_sa_v54_profile_enables_service_completion_gate(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v54_service_completion_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["net_advantage_prepare_gate_enabled"])
        self.assertTrue(kwargs["service_completion_gate_enabled"])
        self.assertGreater(kwargs["service_completion_gate_bias"], 3.0)
        self.assertGreater(kwargs["service_completion_gate_policy_coef"], 0.30)
        self.assertGreater(kwargs["service_completion_gate_fallback_suppression_scale"], 1.0)

    def test_sa_v55_profile_enables_coverage_recovery_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v55_coverage_recovery_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertEqual(defaults["post_training_audit_mode"], "compact")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["net_advantage_prepare_gate_enabled"])
        self.assertTrue(kwargs["service_completion_gate_enabled"])
        self.assertGreater(kwargs["coverage_recovery_gate_bias_scale"], 1.50)
        self.assertGreater(kwargs["coverage_recovery_gate_min_scale"], 0.55)
        self.assertGreater(kwargs["coverage_recovery_gate_fallback_suppression_scale"], 2.0)
        self.assertGreater(kwargs["coverage_recovery_gate_prepare_credit"], 1.0)
        self.assertGreater(kwargs["coverage_recovery_gate_fallback_penalty"], 1.4)
        self.assertTrue(kwargs["coverage_recovery_guard_enabled"])

    def test_sa_v56_profile_enables_partial_observation_handoff_memory(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v56_partial_observation_handoff_memory_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertEqual(defaults["post_training_audit_mode"], "compact")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["coverage_recovery_guard_enabled"])
        self.assertTrue(kwargs["coverage_recovery_final_guard_enabled"])
        self.assertLess(kwargs["coverage_recovery_gate_min_scale"], 0.50)
        self.assertLessEqual(kwargs["coverage_recovery_final_guard_min_scale"], 0.20)
        self.assertGreater(kwargs["coverage_recovery_target_memory_option_credit"], 0.60)
        self.assertGreater(kwargs["coverage_recovery_target_memory_option_penalty"], 1.0)
        self.assertGreater(kwargs["net_advantage_prepare_gate_policy_coef"], 0.45)
        self.assertGreater(kwargs["net_advantage_prepare_gate_event_coef"], 0.60)

    def test_sa_v57_profile_enables_service_continuity_counterfactual(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v57_service_continuity_counterfactual_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["service_continuity_teacher_enabled"])
        self.assertTrue(kwargs["idle_popularity_no_rsu_service_continuity_enabled"])
        self.assertTrue(kwargs["option_gate_counterfactual_prd_enabled"])
        self.assertGreater(kwargs["service_continuity_local_penalty"], 1.0)
        self.assertGreater(kwargs["service_continuity_current_bonus"], 0.9)
        self.assertGreater(kwargs["opportunity_constrained_no_rsu_service_bias"], 3.0)
        self.assertGreater(kwargs["opportunity_constrained_no_rsu_local_penalty"], 3.0)

    def test_sa_v58_profile_enables_ready_aware_counterfactual_margin(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v58_ready_aware_counterfactual_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertTrue(kwargs["idle_popularity_no_rsu_any_action_override_enabled"])
        self.assertTrue(kwargs["option_gate_idle_recovery_mechanism_prior_enabled"])
        self.assertGreater(kwargs["opportunity_constrained_no_rsu_prepare_bias"], 2.0)
        self.assertGreater(kwargs["service_continuity_prepare_bonus"], 1.0)
        self.assertGreater(kwargs["service_continuity_local_penalty"], 1.5)

    def test_sa_v59_profile_enables_sparse_recovery_curriculum(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v59_sparse_recovery_curriculum_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertGreater(kwargs["env_action_sparse_recovery_focus"], 1.0)
        self.assertGreater(kwargs["env_action_ppo_teacher_coef"], 1.0)
        self.assertGreater(kwargs["service_continuity_local_penalty"], 2.0)

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v60_profile_enables_risk_adjusted_recovery(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v60_risk_adjusted_recovery_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertGreater(kwargs["env_action_risk_adjusted_recovery_coef"], 1.0)
        self.assertGreater(
            kwargs["env_action_risk_adjusted_recovery_coef"],
            kwargs["env_action_sparse_recovery_focus"],
        )
        self.assertLess(kwargs["env_action_sparse_recovery_focus"], 1.0)
        self.assertGreaterEqual(kwargs["env_action_risk_adjusted_recovery_floor"], 0.0)
        self.assertLessEqual(kwargs["env_action_risk_adjusted_recovery_floor"], 1.0)

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v61_profile_enables_adapter_miss_counterfactual(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["counterfactual_teacher_prd_enabled"])
        self.assertTrue(kwargs["env_action_counterfactual_margin_enabled"])
        self.assertGreater(kwargs["env_action_adapter_miss_counterfactual_coef"], 2.0)
        self.assertGreater(
            kwargs["env_action_adapter_miss_counterfactual_coef"],
            kwargs["env_action_risk_adjusted_recovery_coef"],
        )
        self.assertLess(kwargs["env_action_sparse_recovery_focus"], 0.5)
        self.assertGreater(kwargs["env_action_counterfactual_margin_coef"], 0.3)

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v62_profile_enables_cache_feasibility_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v62_cache_feasibility_prior_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["cache_feasibility_prior_enabled"])
        self.assertGreater(kwargs["cache_feasibility_cache_fill_bias"], 5.0)
        self.assertGreater(kwargs["cache_feasibility_steady_penalty"], 7.0)
        self.assertGreater(
            kwargs["env_action_adapter_miss_counterfactual_coef"],
            build_sa_ghmappo_profile_kwargs(
                "top_journal_mechanism_v61_adapter_miss_counterfactual_mappo"
            )["env_action_adapter_miss_counterfactual_coef"],
        )
        self.assertLess(kwargs["target_kl"], 0.0035)

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v63_profile_enables_current_ready_two_stage_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v63_current_ready_two_stage_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v62_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v62_cache_feasibility_prior_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["cache_feasibility_prior_enabled"])
        self.assertGreater(kwargs["cache_feasibility_cache_fill_bias"], v62_kwargs["cache_feasibility_cache_fill_bias"])
        self.assertGreater(kwargs["cache_feasibility_current_miss_prepare_penalty"], 8.0)
        self.assertGreater(kwargs["cache_feasibility_current_miss_prefetch_penalty"], 5.0)
        self.assertGreater(
            kwargs["cache_feasibility_current_miss_prepare_penalty"],
            kwargs["cache_feasibility_prepare_penalty"],
        )
        self.assertLess(kwargs["target_kl"], v62_kwargs["target_kl"])

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v64_profile_enables_handoff_alignment_barrier(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v64_handoff_alignment_barrier_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v63_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v63_current_ready_two_stage_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["handoff_alignment_barrier_enabled"])
        self.assertGreater(kwargs["handoff_alignment_barrier_prepare_penalty"], 12.0)
        self.assertGreater(kwargs["handoff_alignment_barrier_prefetch_penalty"], 7.0)
        self.assertGreater(
            kwargs["cache_feasibility_current_miss_prepare_penalty"],
            v63_kwargs["cache_feasibility_current_miss_prepare_penalty"],
        )
        self.assertLess(kwargs["service_continuity_prepare_bonus"], v63_kwargs["service_continuity_prepare_bonus"])
        self.assertLess(kwargs["target_kl"], v63_kwargs["target_kl"])

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v65_profile_enables_argmax_margin_mappo(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v65_argmax_margin_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v64_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v64_handoff_alignment_barrier_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["argmax_margin_regularization_enabled"])
        self.assertGreater(kwargs["argmax_margin_coef"], 0.70)
        self.assertLess(kwargs["event_logit_sharpening_final_scale"], v64_kwargs["event_logit_sharpening_final_scale"])
        self.assertLess(kwargs["digital_twin_policy_prior_logit_bias"], v64_kwargs["digital_twin_policy_prior_logit_bias"])
        self.assertGreater(kwargs["service_continuity_current_bonus"], v64_kwargs["service_continuity_current_bonus"])
        self.assertLess(kwargs["service_continuity_prepare_bonus"], v64_kwargs["service_continuity_prepare_bonus"])
        self.assertLess(kwargs["target_kl"], v64_kwargs["target_kl"])

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v66_profile_restores_sparse_recovery_with_safe_cpi(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v66_sparse_safe_cpi_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v65_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v65_argmax_margin_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["argmax_margin_regularization_enabled"])
        self.assertGreater(kwargs["env_action_sparse_recovery_focus"], v65_kwargs["env_action_sparse_recovery_focus"])
        self.assertGreater(kwargs["digital_twin_policy_prior_distill_coef"], v65_kwargs["digital_twin_policy_prior_distill_coef"])
        self.assertGreater(kwargs["service_continuity_prepare_bonus"], v65_kwargs["service_continuity_prepare_bonus"])
        self.assertGreater(kwargs["env_action_ppo_ratio_barrier_coef"], v65_kwargs["env_action_ppo_ratio_barrier_coef"])
        self.assertLess(kwargs["argmax_margin_coef"], v65_kwargs["argmax_margin_coef"])
        self.assertLess(kwargs["target_kl"], v65_kwargs["target_kl"])

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v67_profile_adds_sparse_handoff_recovery_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v67_sparse_handoff_recovery_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v66_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v66_sparse_safe_cpi_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertEqual(defaults["prediction_horizon"], 16)
        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["sparse_handoff_recovery_prior_enabled"])
        self.assertGreater(kwargs["sparse_handoff_recovery_prefetch_bias"], 0.0)
        self.assertGreater(kwargs["sparse_handoff_recovery_prepare_bias"], 0.0)
        self.assertGreater(kwargs["sparse_handoff_recovery_local_penalty"], 0.0)
        self.assertGreater(kwargs["env_action_sparse_recovery_focus"], v66_kwargs["env_action_sparse_recovery_focus"])
        self.assertGreater(kwargs["digital_twin_policy_prior_distill_coef"], v66_kwargs["digital_twin_policy_prior_distill_coef"])
        self.assertLess(kwargs["target_kl"], v66_kwargs["target_kl"])

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v67_sparse_recovery_prior_boosts_prefetch_and_suppresses_local(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "idle_or_sparse"
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"][1]["cached_adapter_ids"] = []
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            sparse_handoff_recovery_prior_enabled=True,
            sparse_handoff_recovery_prefetch_bias=6.0,
            sparse_handoff_recovery_prepare_bias=4.0,
            sparse_handoff_recovery_current_fill_bias=1.0,
            sparse_handoff_recovery_steady_bias=1.0,
            sparse_handoff_recovery_local_penalty=5.0,
            sparse_handoff_recovery_min_context=0.05,
            sparse_handoff_recovery_max_eta=16,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_sparse_handoff_recovery_prior(
            policy_output,
            state,
            run_metadata={"window_class": "idle_or_sparse"},
        )
        info = adjusted["sparse_handoff_recovery_prior_info"]
        env_bias = adjusted["env_action_logits_bias"]

        self.assertTrue(info["active"])
        self.assertGreater(info["prefetch_bias"], 0.0)
        self.assertGreater(float(adjusted["slow_logits"][2]), float(policy_output["slow_logits"][2]))
        self.assertGreater(float(env_bias[1]), 0.0)
        self.assertLess(float(env_bias[2]), 0.0)

    def test_sa_v68_profile_disables_idle_popularity_fallback_for_actor_native_recovery(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v68_actor_native_sparse_recovery_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v67_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v67_sparse_handoff_recovery_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(v67_kwargs["idle_popularity_fallback_enabled"])
        self.assertFalse(kwargs["idle_popularity_fallback_enabled"])
        self.assertFalse(kwargs["idle_popularity_no_rsu_service_continuity_enabled"])
        self.assertFalse(kwargs["idle_popularity_no_rsu_any_action_override_enabled"])
        self.assertTrue(kwargs["sparse_handoff_recovery_prior_enabled"])
        self.assertGreater(
            kwargs["sparse_handoff_recovery_prepare_bias"],
            v67_kwargs["sparse_handoff_recovery_prepare_bias"],
        )
        self.assertGreater(
            kwargs["sparse_handoff_recovery_local_penalty"],
            v67_kwargs["sparse_handoff_recovery_local_penalty"],
        )
        self.assertGreater(
            kwargs["sparse_handoff_recovery_max_eta"],
            v67_kwargs["sparse_handoff_recovery_max_eta"],
        )

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v69_profile_adds_success_conditioned_sparse_realization_credit(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v69_realization_credit_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v67_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v67_sparse_handoff_recovery_mappo"
        )
        v68_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v68_actor_native_sparse_recovery_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["sparse_handoff_realization_credit_enabled"])
        self.assertGreater(kwargs["sparse_handoff_realization_success_bonus"], 0.0)
        self.assertGreater(kwargs["sparse_handoff_realization_failed_prepare_penalty"], 0.0)
        self.assertTrue(kwargs["idle_popularity_fallback_enabled"])
        self.assertTrue(v67_kwargs["idle_popularity_fallback_enabled"])
        self.assertFalse(v68_kwargs["idle_popularity_fallback_enabled"])
        self.assertTrue(kwargs["sparse_handoff_recovery_prior_enabled"])
        self.assertLess(
            kwargs["sparse_handoff_recovery_prefetch_bias"],
            v68_kwargs["sparse_handoff_recovery_prefetch_bias"],
        )

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v70_profile_adds_sparse_tail_option_prior(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v70_sparse_tail_option_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        v69_kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v69_realization_credit_mappo"
        )

        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertEqual(defaults["train_window_mode"], "rotate")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["sparse_handoff_option_prior_enabled"])
        self.assertGreater(kwargs["sparse_handoff_option_prepare_bias"], 0.0)
        self.assertGreater(kwargs["sparse_handoff_option_popularity_penalty"], 0.0)
        self.assertGreater(kwargs["sparse_handoff_option_local_penalty"], 0.0)
        self.assertTrue(kwargs["sparse_handoff_realization_credit_enabled"])
        self.assertTrue(kwargs["idle_popularity_fallback_enabled"])
        self.assertGreater(
            kwargs["handoff_risk_option_coef"],
            v69_kwargs["handoff_risk_option_coef"],
        )

        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.window_mode, "full_stratified")
        self.assertEqual(args.train_window_mode, "rotate")

    def test_sa_v71_profile_uses_learned_counterfactual_option_credit_only(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v71_tail_counterfactual_option_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertEqual(defaults["window_mode"], "full_stratified")
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["option_gate_enabled"])
        self.assertTrue(kwargs["option_counterfactual_critic_enabled"])
        self.assertGreater(kwargs["option_counterfactual_value_coef"], 0.0)
        self.assertGreater(kwargs["option_counterfactual_advantage_coef"], 0.0)
        self.assertGreater(kwargs["option_counterfactual_tail_weight"], 0.0)
        self.assertTrue(kwargs["option_counterfactual_policy_improvement_enabled"])
        self.assertGreater(kwargs["option_counterfactual_policy_improvement_coef"], 0.0)
        self.assertFalse(kwargs["sparse_handoff_option_prior_enabled"])
        self.assertFalse(kwargs["sparse_handoff_recovery_prior_enabled"])
        self.assertFalse(kwargs["sparse_handoff_realization_credit_enabled"])
        self.assertFalse(kwargs["continuity_guard_enabled"])
        self.assertFalse(kwargs["idle_popularity_fallback_enabled"])
        self.assertEqual(kwargs["heuristic_imitation_coef"], 0.0)
        self.assertEqual(kwargs["option_gate_prior_logit_bias"], 0.0)
        self.assertFalse(defaults["temporal_reward_shaping_enabled"])
        original_argv = sys.argv
        try:
            sys.argv = ["train_sa_ghmappo_real_sample.py", "--profile", profile]
            args = parse_args()
        finally:
            sys.argv = original_argv
        self.assertFalse(args.temporal_reward_shaping_enabled)
        self.assertEqual(args.update_eval_max_windows, 4)
        self.assertEqual(args.update_eval_max_workflows, 1)
        self.assertEqual(args.post_training_audit_mode, "compact")

    def test_sa_v71_counterfactual_option_critic_uses_return_targets(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            option_gate_enabled=True,
            option_counterfactual_critic_enabled=True,
            option_counterfactual_value_coef=0.5,
            option_counterfactual_advantage_coef=1.0,
            option_counterfactual_advantage_clip=2.0,
            option_counterfactual_warmup_updates=0,
            option_counterfactual_tail_weight=2.0,
        )
        option_q_values = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        losses = agent._compute_option_gate_loss(
            batch_outputs=[
                {
                    "option_logits": torch.zeros(4, requires_grad=True),
                    "option_q_values": option_q_values,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "option_gate": {
                            "enabled": True,
                            "option_action": 1,
                            "option_log_prob": -1.386294,
                            "option_mask": [True, True, True, True],
                            "prior_target": 0,
                            "sparse_tail_risk_option_context": {
                                "active": True,
                                "context": 1.0,
                            },
                        }
                    }
                }
            ],
            batch_advantage=torch.tensor([0.0]),
            batch_option_returns=torch.tensor([2.0]),
        )

        self.assertEqual(len(losses), 5)
        self.assertGreater(float(losses[3].detach()), 0.0)
        self.assertGreater(float(losses[4].detach()), 0.0)
        losses[3].backward()
        self.assertIsNotNone(option_q_values.grad)

    def test_sa_v71_counterfactual_critic_improves_option_logits(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            option_gate_enabled=True,
            option_counterfactual_critic_enabled=True,
            option_counterfactual_policy_improvement_enabled=True,
            option_counterfactual_policy_improvement_coef=1.0,
            option_counterfactual_policy_improvement_clip=2.0,
            option_counterfactual_warmup_updates=0,
        )
        improved_logits, normalized_advantage = agent._critic_improved_option_logits(
            option_logits=torch.zeros(4),
            option_q_values=torch.tensor([0.0, 2.0, -1.0, 1.0]),
            option_mask=[True, True, False, True],
        )

        self.assertEqual(int(torch.argmax(improved_logits).item()), 1)
        self.assertGreater(float(normalized_advantage[1]), 0.0)
        self.assertEqual(float(normalized_advantage[2]), 0.0)

    def test_sa_v72_profile_uses_digital_twin_counterfactual_targets(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v72_digital_twin_counterfactual_option_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertEqual(
            defaults["primary_vehicle_selection"],
            "handoff_pressure",
        )
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["option_counterfactual_critic_enabled"])
        self.assertTrue(kwargs["option_counterfactual_model_rollout_enabled"])
        self.assertTrue(
            kwargs[
                "option_counterfactual_policy_improvement_deterministic_only"
            ]
        )
        self.assertFalse(kwargs["sparse_handoff_option_prior_enabled"])
        self.assertFalse(kwargs["sparse_handoff_realization_credit_enabled"])

    def test_sa_v72_counterfactual_targets_train_unselected_options(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            option_gate_enabled=True,
            option_counterfactual_critic_enabled=True,
            option_counterfactual_model_rollout_enabled=True,
            option_counterfactual_warmup_updates=0,
        )
        option_q_values = torch.zeros(4, requires_grad=True)
        losses = agent._compute_option_gate_loss(
            batch_outputs=[
                {
                    "option_logits": torch.zeros(4, requires_grad=True),
                    "option_q_values": option_q_values,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "option_gate": {
                            "enabled": True,
                            "option_action": 0,
                            "option_log_prob": -1.386294,
                            "option_mask": [True, True, True, True],
                            "prior_target": 0,
                            "counterfactual_model_rollout": {
                                "option_td_targets": {
                                    "0": 1.0,
                                    "1": 4.0,
                                    "2": -2.0,
                                    "3": 2.0,
                                }
                            },
                        }
                    }
                }
            ],
            batch_advantage=torch.tensor([0.0]),
            batch_option_returns=torch.tensor([0.0]),
        )

        losses[3].backward()
        self.assertIsNotNone(option_q_values.grad)
        self.assertGreater(float(torch.abs(option_q_values.grad[1])), 0.0)
        self.assertGreater(float(torch.abs(option_q_values.grad[2])), 0.0)

    def test_sa_v73_profile_uses_four_step_counterfactual_rollout(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v73_multistep_counterfactual_option_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["option_counterfactual_model_rollout_enabled"])
        self.assertEqual(kwargs["option_counterfactual_model_rollout_horizon"], 4)
        self.assertTrue(
            kwargs[
                "option_counterfactual_policy_improvement_deterministic_only"
            ]
        )

    def test_sa_v74_profile_uses_native_action_model_critic(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v74_model_based_env_action_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertFalse(kwargs["option_gate_enabled"])
        self.assertFalse(kwargs["option_counterfactual_critic_enabled"])
        self.assertTrue(kwargs["env_action_model_critic_enabled"])
        self.assertTrue(kwargs["env_action_model_rollout_enabled"])
        self.assertEqual(kwargs["env_action_model_rollout_horizon"], 4)
        self.assertGreater(kwargs["env_action_model_critic_value_coef"], 0.0)
        self.assertGreater(kwargs["env_action_model_critic_advantage_coef"], 0.0)
        self.assertGreater(
            kwargs["env_action_model_critic_policy_improvement_coef"],
            0.0,
        )
        self.assertTrue(kwargs["env_action_ppo_enabled"])
        self.assertFalse(defaults["temporal_reward_shaping_enabled"])

    def test_sa_v74_action_model_critic_trains_unselected_actions(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_rollout_enabled=True,
            env_action_model_critic_warmup_updates=0,
        )
        action_q_values = torch.zeros(5, requires_grad=True)
        value_loss, _ = agent._compute_env_action_model_critic_loss(
            batch_outputs=[{"env_action_q_values": action_q_values}],
            batch_rows=[
                {
                    "action_info": {
                        "env_action_model_rollout": {
                            "action_td_targets": {
                                "0": 1.0,
                                "1": 4.0,
                                "2": -2.0,
                                "3": 2.0,
                                "4": 6.0,
                            }
                        }
                    }
                }
            ],
            batch_action_masks=[[True, True, True, True, True]],
        )

        value_loss.backward()
        self.assertIsNotNone(action_q_values.grad)
        self.assertGreater(float(torch.abs(action_q_values.grad[1])), 0.0)
        self.assertGreater(float(torch.abs(action_q_values.grad[2])), 0.0)
        self.assertGreater(float(torch.abs(action_q_values.grad[4])), 0.0)

    def test_sa_v74_action_critic_improves_native_action_scores(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_critic_policy_improvement_coef=2.0,
            env_action_model_critic_warmup_updates=0,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
            "env_action_q_values": torch.tensor([0.0, 1.0, -2.0, 0.5, 4.0]),
        }
        adjusted, diagnostics = agent._apply_env_action_model_critic_improvement(
            policy_output=policy_output,
            action_mask=[True, True, True, True, True],
        )
        improved_scores = agent._hierarchical_env_action_scores(adjusted)

        self.assertTrue(diagnostics["applied"])
        self.assertEqual(int(torch.argmax(improved_scores).item()), 4)

    def test_sa_v75_profile_uses_conservative_model_policy_improvement(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
            parse_args,
        )

        profile = "top_journal_mechanism_v75_conservative_model_policy_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["env_action_model_policy_improvement_enabled"])
        self.assertGreater(
            kwargs["env_action_model_policy_improvement_coef"],
            0.0,
        )
        self.assertGreater(
            kwargs["env_action_model_policy_improvement_temperature"],
            1.0,
        )
        self.assertEqual(kwargs["env_action_model_critic_advantage_coef"], 0.0)
        self.assertEqual(
            kwargs["env_action_model_critic_policy_improvement_coef"],
            0.0,
        )
        with patch.object(
            sys,
            "argv",
            ["train_sa_ghmappo_real_sample.py", "--profile", profile],
        ):
            self.assertEqual(
                parse_args().primary_vehicle_selection,
                "handoff_pressure",
            )
        with patch.object(
            sys,
            "argv",
            [
                "train_sa_ghmappo_real_sample.py",
                "--profile",
                profile,
                "--primary_vehicle_selection",
                "stable_first",
            ],
        ):
            self.assertEqual(
                parse_args().primary_vehicle_selection,
                "stable_first",
            )

    def test_sa_v75_model_policy_loss_updates_actor_logits(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_policy_improvement_enabled=True,
            env_action_model_policy_improvement_coef=0.35,
            env_action_model_policy_improvement_temperature=2.5,
        )
        slow_logits = torch.zeros(3, requires_grad=True)
        fast_logits = torch.zeros(2, requires_grad=True)
        event_logits = torch.zeros(2, requires_grad=True)
        loss, target_kl = agent._compute_env_action_model_policy_improvement_loss(
            batch_outputs=[
                {
                    "slow_logits": slow_logits,
                    "fast_logits": fast_logits,
                    "event_logits": event_logits,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "action_projection": {
                            "masked_env_action_probs": [0.2] * 5,
                        },
                        "env_action_model_rollout": {
                            "action_td_targets": {
                                "0": 1.0,
                                "1": 2.0,
                                "2": 0.0,
                                "3": -1.0,
                                "4": 4.0,
                            }
                        },
                    }
                }
            ],
            batch_action_masks=[[True, True, True, True, True]],
        )

        loss.backward()
        self.assertGreater(float(target_kl), 0.0)
        self.assertIsNotNone(slow_logits.grad)
        self.assertIsNotNone(fast_logits.grad)
        self.assertIsNotNone(event_logits.grad)
        self.assertGreater(float(torch.abs(event_logits.grad).sum()), 0.0)

    def test_sa_v76_profile_enables_memory_robust_horizons_and_kl(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v76_recurrent_robust_model_policy_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertEqual(
            defaults["primary_vehicle_selection"],
            "handoff_pressure",
        )
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["outcome_memory_fusion_enabled"])
        self.assertEqual(
            tuple(kwargs["env_action_model_rollout_horizons"]),
            (1, 2, 4, 8),
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_robust_horizons_enabled"
            ]
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_adaptive_kl_enabled"
            ]
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_target_kl"],
            0.03,
        )

    def test_sa_v76_memory_features_reach_actor_encoder(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            outcome_memory_fusion_enabled=True,
        )
        info = {
            "semantic_state": _minimal_semantic_state(),
            "algorithm_memory": {
                "step_index": 4,
                "last_action_id": 4,
                "same_action_streak": 3,
                "prepare_action_streak": 3,
                "failed_prepare_streak": 2,
                "no_progress_streak": 2,
                "last_reward": -1.25,
                "last_handoff_failed": True,
                "last_stall": True,
            },
        }

        output = agent._forward_policy(agent._extract_semantic_state(info))
        encoded = output["encoded"]

        self.assertIn("outcome_memory_fusion_enabled", encoded)
        self.assertAlmostEqual(
            float(encoded["outcome_memory_same_action_streak"].item()),
            3.0 / 8.0,
        )
        self.assertAlmostEqual(
            float(encoded["outcome_memory_failed_prepare_streak"].item()),
            2.0 / 8.0,
        )
        self.assertAlmostEqual(
            float(encoded["outcome_memory_no_progress_streak"].item()),
            2.0 / 8.0,
        )

    def test_sa_v76_trainer_memory_tracks_repeated_failed_prepare(self) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        memory = MARLOnPolicyTrainer._initial_algorithm_memory()
        decision_info = {
            "semantic_state": {
                "workflow": {
                    "execution_order": ["n1", "n2"],
                    "completed_node_ids": [],
                }
            }
        }
        next_info = {
            "semantic_state": deepcopy(decision_info["semantic_state"]),
            "metrics_protocol": {
                "stall_occurred": True,
                "handoff_failed": True,
                "mechanism_success_strict": False,
                "cache_hit": False,
            },
        }

        memory = MARLOnPolicyTrainer._advance_algorithm_memory(
            memory,
            action=4,
            reward=-1.25,
            decision_info=decision_info,
            next_info=next_info,
        )
        memory = MARLOnPolicyTrainer._advance_algorithm_memory(
            memory,
            action=4,
            reward=-1.25,
            decision_info=decision_info,
            next_info=next_info,
        )

        self.assertEqual(memory["same_action_streak"], 2)
        self.assertEqual(memory["prepare_action_streak"], 2)
        self.assertEqual(memory["failed_prepare_streak"], 2)
        self.assertEqual(memory["no_progress_streak"], 2)
        self.assertTrue(memory["last_handoff_failed"])

    def test_sa_v76_robust_horizon_target_rejects_short_term_prepare_spike(
        self,
    ) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_policy_improvement_enabled=True,
            env_action_model_policy_improvement_robust_horizons_enabled=True,
            env_action_model_policy_improvement_horizon_risk_coef=0.75,
            env_action_model_policy_improvement_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_target_kl=0.03,
        )
        target_maps = [
            {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.0},
            {0: 2.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 5.0},
            {0: 5.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -5.0},
            {0: 8.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -10.0},
        ]
        robust_advantage = agent._build_env_action_model_robust_advantage(
            target_maps=target_maps,
            valid_indices=[0, 1, 2, 3, 4],
        )
        old_probs = torch.full((5,), 0.2)
        improved_target = agent._build_env_action_model_improved_target(
            old_probs=old_probs,
            normalized_advantage=robust_advantage,
        )
        target_kl = torch.sum(
            improved_target
            * (
                torch.log(improved_target)
                - torch.log(old_probs)
            )
        )

        self.assertEqual(int(torch.argmax(improved_target).item()), 0)
        self.assertLessEqual(float(target_kl), 0.03001)
        self.assertLess(
            float(improved_target[4]),
            float(improved_target[0]),
        )

    def test_sa_v77_profile_enables_temporal_downside_lambda_target(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v77_temporal_downside_model_policy_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertEqual(
            kwargs[
                "env_action_model_policy_improvement_horizon_aggregation_mode"
            ],
            "lambda_downside",
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_horizon_lambda"],
            0.90,
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_adaptive_kl_enabled"
            ]
        )

    def test_sa_v77_temporal_downside_preserves_delayed_gain(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_policy_improvement_enabled=True,
            env_action_model_policy_improvement_robust_horizons_enabled=True,
            env_action_model_policy_improvement_horizon_risk_coef=0.75,
            env_action_model_policy_improvement_horizon_aggregation_mode=(
                "lambda_downside"
            ),
            env_action_model_policy_improvement_horizon_lambda=0.90,
            env_action_model_policy_improvement_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_target_kl=0.03,
        )
        target_maps = [
            {0: -10.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.0},
            {0: -5.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 5.0},
            {0: 5.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -5.0},
            {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -10.0},
        ]

        temporal_advantage = agent._build_env_action_model_robust_advantage(
            target_maps=target_maps,
            valid_indices=[0, 1, 2, 3, 4],
        )
        old_probs = torch.full((5,), 0.2)
        improved_target = agent._build_env_action_model_improved_target(
            old_probs=old_probs,
            normalized_advantage=temporal_advantage,
        )
        target_kl = torch.sum(
            improved_target
            * (
                torch.log(improved_target)
                - torch.log(old_probs)
            )
        )

        self.assertEqual(int(torch.argmax(improved_target).item()), 0)
        self.assertGreater(
            float(improved_target[0]),
            float(improved_target[4]),
        )
        self.assertLessEqual(float(target_kl), 0.03001)

    def test_sa_v78_profile_enables_regret_adaptive_long_horizon_target(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v78_regret_adaptive_model_policy_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertEqual(
            tuple(kwargs["env_action_model_rollout_horizons"]),
            (1, 2, 4, 8, 16),
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_regret_adaptive_kl_enabled"
            ]
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_max_target_kl"],
            0.35,
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_regret_priority_coef"],
            2.0,
        )

    def test_sa_v78_counterfactual_regret_can_reverse_bad_policy_mode(
        self,
    ) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_critic_enabled=True,
            env_action_model_policy_improvement_enabled=True,
            env_action_model_policy_improvement_robust_horizons_enabled=True,
            env_action_model_policy_improvement_horizon_risk_coef=0.75,
            env_action_model_policy_improvement_horizon_aggregation_mode=(
                "lambda_downside"
            ),
            env_action_model_policy_improvement_horizon_lambda=0.90,
            env_action_model_policy_improvement_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_target_kl=0.03,
            env_action_model_policy_improvement_regret_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_max_target_kl=0.35,
            env_action_model_policy_improvement_regret_priority_coef=2.0,
        )
        target_maps = [
            {0: 4.15, 2: 0.20, 3: 0.10, 4: 1.10},
            {0: 6.74, 2: -1.06, 3: -1.16, 4: -0.16},
            {0: 9.59, 2: -3.54, 3: -3.64, 4: -2.64},
            {0: 9.59, 2: -8.36, 3: -8.46, 4: -7.46},
            {0: 29.26, 2: -9.27, 3: -9.37, 4: -8.37},
        ]
        normalized_advantage = (
            agent._build_env_action_model_robust_advantage(
                target_maps=target_maps,
                valid_indices=[0, 2, 3, 4],
            )
        )
        old_probs = torch.tensor([0.168305, 0.056089, 0.068270, 0.707335])
        old_probs = old_probs / old_probs.sum()
        normalized_regret = (
            agent._normalized_env_action_counterfactual_regret(
                old_probs=old_probs,
                normalized_advantage=normalized_advantage,
            )
        )
        effective_target_kl = 0.03 + (0.35 - 0.03) * float(
            normalized_regret
        )
        improved_target = agent._build_env_action_model_improved_target(
            old_probs=old_probs,
            normalized_advantage=normalized_advantage,
            target_kl=effective_target_kl,
        )
        target_kl = torch.sum(
            improved_target
            * (
                torch.log(improved_target)
                - torch.log(old_probs)
            )
        )

        self.assertGreater(float(normalized_regret), 0.70)
        self.assertEqual(int(torch.argmax(improved_target).item()), 0)
        self.assertLessEqual(float(target_kl), effective_target_kl + 1e-5)
        self.assertLessEqual(effective_target_kl, 0.35)

    def test_sa_v79_profile_enables_high_regret_tail_distillation(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v79_tail_distilled_model_policy_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_distillation_enabled"
            ]
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_tail_quantile"],
            0.75,
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_tail_min_regret"],
            0.50,
        )
        self.assertEqual(
            kwargs["env_action_model_policy_improvement_tail_epochs"],
            8,
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_tail_coef"],
            1.0,
        )

    def test_sa_v80_profile_enables_training_only_imagination_replay(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v80_imagination_replay_model_policy_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs["env_action_model_imagination_replay_enabled"]
        )
        self.assertEqual(
            tuple(kwargs["env_action_model_imagination_replay_depths"]),
            (2, 4, 8),
        )
        self.assertEqual(
            kwargs["env_action_model_policy_improvement_tail_epochs"],
            4,
        )
        self.assertAlmostEqual(
            kwargs["env_action_model_policy_improvement_tail_coef"],
            0.75,
        )

    def test_sa_v81_profile_enables_multihorizon_joint_trust_region(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v81_trust_region_imagination_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertEqual(
            tuple(kwargs["env_action_model_imagination_replay_horizons"]),
            (1, 2, 4),
        )
        self.assertAlmostEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_max_policy_kl"
            ],
            0.05,
        )

    def test_sa_v82_profile_enables_outcome_conditioned_recovery_imagination(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        profile = "top_journal_mechanism_v82_recovery_imagination_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs["env_action_model_imagination_replay_recovery_only"]
        )
        self.assertFalse(
            MARLOnPolicyTrainer._algorithm_memory_has_recovery_signal(
                MARLOnPolicyTrainer._initial_algorithm_memory()
            )
        )
        self.assertTrue(
            MARLOnPolicyTrainer._algorithm_memory_has_recovery_signal(
                {"no_progress_streak": 2}
            )
        )

    def test_sa_v83_profile_isolates_recovery_residual_updates(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v83_recovery_residual_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["outcome_recovery_residual_enabled"])
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_recovery_only"
            ]
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_adapter_only"
            ]
        )

    def test_sa_v84_profile_enables_search_distilled_dual_residuals(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v84_search_distilled_residual_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["digital_twin_planning_residual_enabled"])
        self.assertTrue(kwargs["env_action_model_beam_search_enabled"])
        self.assertEqual(kwargs["env_action_model_beam_search_horizon"], 6)
        self.assertEqual(kwargs["env_action_model_beam_search_width"], 2)
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_planning_adapter_only"
            ]
        )

    def test_sa_v85_profile_enables_backtracked_residual_optimizer(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = (
            "top_journal_mechanism_v85_trust_region_residual_search_mappo"
        )
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_residual_optimizer_enabled"
            ]
        )
        self.assertEqual(
            kwargs["env_action_model_policy_improvement_tail_epochs"],
            24,
        )
        self.assertAlmostEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_residual_learning_rate"
            ],
            0.02,
        )
        self.assertAlmostEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_residual_backtrack_factor"
            ],
            0.5,
        )
        self.assertEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_residual_max_backtracks"
            ],
            7,
        )
        self.assertAlmostEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_max_policy_kl"
            ],
            kwargs["env_action_model_policy_improvement_max_target_kl"],
        )

    def test_sa_v86_profile_searches_imagined_recovery_states(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v86_recovery_beam_replay_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["env_action_model_imagination_replay_enabled"])
        self.assertTrue(
            kwargs["env_action_model_imagination_replay_recovery_only"]
        )
        self.assertTrue(
            kwargs["env_action_model_imagination_beam_search_enabled"]
        )
        self.assertTrue(
            kwargs["env_action_model_policy_improvement_tail_beam_only"]
        )

    def test_sa_v86_imagined_recovery_sample_carries_beam_targets(self) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        class FakeEnv:
            def __init__(self, state: int = 0) -> None:
                self.state = state
                self._recorder = None

            def step(self, action: int):
                self.state += 1
                reward = float(self.state * 10 + int(action))
                return (
                    [self.state],
                    reward,
                    self.state >= 7,
                    False,
                    {
                        "semantic_state": {},
                        "metrics_protocol": {"stall_occurred": False},
                    },
                )

        class FakeAgent:
            def act(self, observation, info):
                del observation, info
                return 0, {"action_mask": [True] * 5}

            def evaluate_value(self, observation, info=None):
                del observation, info
                return 0.0

        trainer = MARLOnPolicyTrainer(
            env=FakeEnv(state=0),
            agent=FakeAgent(),
            recorder=None,
            gamma=0.99,
        )
        sample, transition_count = (
            trainer._collect_imagined_env_action_sample(
                branch_env=FakeEnv(state=4),
                observation=[4],
                decision_info={"semantic_state": {}},
                algorithm_memory={"no_progress_streak": 1},
                action_info={
                    "action_mask": [True] * 5,
                    "env_action_model_imagination_beam_search_enabled": True,
                    "env_action_model_beam_search_horizon": 2,
                    "env_action_model_beam_search_width": 2,
                },
                run_metadata=None,
                imagination_depth=2,
                imagination_horizons=[1],
            )
        )
        rollout = sample["action_info"]["env_action_model_rollout"]

        self.assertEqual(
            rollout["protocol"],
            "digital_twin_imagined_recovery_beam_td_v3",
        )
        self.assertEqual(
            rollout["beam_search_protocol"],
            "digital_twin_discrete_beam_search_v1",
        )
        self.assertEqual(set(rollout["beam_action_td_targets"]), set("01234"))
        self.assertGreater(
            min(rollout["beam_action_td_targets"].values()),
            50.0,
        )
        self.assertEqual(transition_count, rollout["model_transition_count"])

    def test_sa_v87_profile_enables_mirror_logit_projection(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v87_mirror_residual_projection_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs["env_action_model_imagination_beam_search_enabled"]
        )
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_logit_projection_enabled"
            ]
        )

    def test_sa_v87_logit_projection_pushes_toward_beam_action(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            use_hierarchy=False,
            env_action_model_policy_improvement_enabled=True,
            env_action_model_policy_improvement_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_regret_adaptive_kl_enabled=True,
            env_action_model_policy_improvement_target_kl=0.03,
            env_action_model_policy_improvement_max_target_kl=0.35,
            env_action_model_policy_improvement_prefer_beam_targets=True,
        )
        logits = torch.tensor(
            [-0.5, -1.0, -2.5, -3.0, 2.5],
            dtype=torch.float32,
            requires_grad=True,
        )
        old_probs = torch.softmax(logits.detach(), dim=-1).tolist()
        loss, _ = agent._compute_env_action_model_policy_improvement_loss(
            batch_outputs=[{"flat_logits": logits}],
            batch_rows=[
                {
                    "action_info": {
                        "action_projection": {
                            "masked_env_action_probs": old_probs,
                        },
                        "env_action_model_rollout": {
                            "beam_action_td_targets": {
                                "0": 0.0,
                                "1": -1.0,
                                "2": 10.0,
                                "3": -2.0,
                                "4": -5.0,
                            },
                        },
                    },
                }
            ],
            batch_action_masks=[[True] * 5],
            logit_projection_enabled=True,
        )

        loss.backward()

        self.assertLess(float(logits.grad[2]), 0.0)
        self.assertGreater(float(logits.grad[4]), 0.0)

    def test_sa_v88_profile_enables_contextual_recovery_expert(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v88_contextual_recovery_expert_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["outcome_context_residual_enabled"])
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_logit_projection_enabled"
            ]
        )

    def test_sa_v88_contextual_expert_receives_explicit_state(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            outcome_memory_fusion_enabled=True,
            outcome_recovery_residual_enabled=True,
            digital_twin_planning_residual_enabled=True,
            outcome_context_residual_enabled=True,
        )
        state = deepcopy(_minimal_semantic_state())
        state["algorithm_memory"] = {
            "step_index": 6,
            "last_action_id": 4,
            "failed_prepare_streak": 1,
            "no_progress_streak": 1,
            "last_stall": True,
        }
        output = agent._forward_policy(state)

        self.assertIn("outcome_context_residual_bias", output)
        self.assertIsNotNone(output["outcome_context_residual_bias"])
        self.assertEqual(tuple(output["outcome_context_residual_bias"].shape), (5,))
        self.assertAlmostEqual(
            float(output["outcome_recovery_residual_gate"].item()),
            1.0,
        )

    def test_sa_v89_profile_enables_diverse_recovery_branches(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v89_diverse_recovery_branch_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["outcome_context_residual_enabled"])
        self.assertEqual(
            kwargs["env_action_model_imagination_replay_branch_mode"],
            "top_k",
        )
        self.assertEqual(
            kwargs["env_action_model_imagination_replay_branch_top_k"],
            2,
        )

    def test_sa_v89_top_k_imagination_branches_follow_policy_support(
        self,
    ) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        top_k = MARLOnPolicyTrainer._select_imagination_branch_actions(
            valid_actions=[0, 2, 4],
            root_probs=[0.3, 0.0, 0.1, 0.0, 0.6],
            dominant_action=4,
            branch_mode="top_k",
            branch_top_k=2,
        )
        dominant = MARLOnPolicyTrainer._select_imagination_branch_actions(
            valid_actions=[0, 2, 4],
            root_probs=[0.3, 0.0, 0.1, 0.0, 0.6],
            dominant_action=4,
            branch_mode="dominant",
            branch_top_k=2,
        )

        self.assertEqual(top_k, {0, 4})
        self.assertEqual(dominant, {4})

    def test_sa_v90_profile_scopes_tail_updates_to_recovery_adapters(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v90_causally_scoped_recovery_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["episodes"], 256)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertEqual(
            kwargs["env_action_model_imagination_replay_branch_mode"],
            "top_k",
        )
        self.assertTrue(
            kwargs["env_action_model_policy_improvement_tail_adapter_only"]
        )
        self.assertFalse(
            kwargs[
                "env_action_model_policy_improvement_tail_planning_adapter_only"
            ]
        )

    def test_sa_v91_profile_balances_high_regret_recovery_targets(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v91_balanced_recovery_distillation_mappo"
        defaults = PROFILE_DEFAULTS[profile]
        kwargs = build_sa_ghmappo_profile_kwargs(profile)

        self.assertEqual(defaults["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_policy_improvement_tail_target_balance_enabled"
            ]
        )
        self.assertEqual(
            kwargs[
                "env_action_model_policy_improvement_tail_target_balance_power"
            ],
            0.5,
        )
        self.assertTrue(
            kwargs["env_action_model_policy_improvement_tail_adapter_only"]
        )

    def test_sa_v91_target_balance_preserves_scale_and_upweights_minority(
        self,
    ) -> None:
        agent = build_agent("sa_ghmappo", random_seed=7)
        selected = [
            (index, 0.9, 2, True) for index in range(12)
        ] + [
            (12, 0.95, 0, True),
            (13, 0.94, 0, False),
        ]

        by_index, by_target = agent._build_tail_target_balance_weights(
            selected,
            power=0.5,
            max_weight=4.0,
        )

        self.assertGreater(by_target["0"], by_target["2"])
        self.assertLessEqual(max(by_index.values()), 4.0)
        self.assertAlmostEqual(
            sum(by_index.values()) / len(by_index),
            1.0,
            places=6,
        )

    def test_sa_v92_online_planner_uses_counterfactual_return_with_policy_contract(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import build_sa_ghmappo_profile_kwargs

        kwargs = build_sa_ghmappo_profile_kwargs(
            "top_journal_mechanism_v92_online_counterfactual_mappo"
        )
        kwargs["env_action_model_online_planner_policy_prior_coef"] = 0.0
        agent = build_agent("sa_ghmappo", random_seed=7, **kwargs)
        selected_action, planner_stats = agent.select_env_action_from_model_targets(
            action_info={
                "final_env_action": 0,
                "action_mask": [True, True, True, True, True],
                "action_projection": {
                    "masked_env_action_probs": [0.70, 0.05, 0.10, 0.10, 0.05]
                },
            },
            rollout_info={
                "action_td_targets_by_horizon": {
                    "1": {"0": 0.0, "1": 0.0, "2": 2.0, "3": 0.0, "4": 0.0},
                    "4": {"0": 0.0, "1": 0.0, "2": 4.0, "3": 0.0, "4": 0.0},
                }
            },
        )

        self.assertEqual(selected_action, 2)
        self.assertTrue(planner_stats["enabled"])
        self.assertTrue(planner_stats["applied"])
        self.assertEqual(planner_stats["candidate_count"], 5)
        self.assertEqual(
            planner_stats["protocol"],
            "mappo_counterfactual_policy_improvement_v1",
        )

    def test_sa_v92_profile_enables_online_counterfactual_planner(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            PROFILE_DEFAULTS,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v92_online_counterfactual_mappo"
        self.assertEqual(PROFILE_DEFAULTS[profile]["reward_positive_offset"], 0.0)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertTrue(kwargs["env_action_model_online_planner_enabled"])
        self.assertEqual(kwargs["env_action_model_online_planner_coef"], 1.0)

    def test_sa_v114_profile_distills_selective_teacher_without_online_planner(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v114_selective_teacher_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["env_action_model_teacher_distillation_enabled"])
        self.assertGreater(kwargs["env_action_model_teacher_distillation_coef"], 0.0)
        self.assertFalse(kwargs["env_action_model_online_planner_enabled"])

    def test_sa_v114_teacher_label_is_counterfactual_and_updates_native_logits(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_teacher_distillation_enabled=True,
            env_action_model_teacher_distillation_coef=0.8,
        )
        selected_action, planner_stats = agent.select_env_action_from_model_targets(
            action_info={
                "final_env_action": 0,
                "action_mask": [True] * 5,
                "action_projection": {
                    "masked_env_action_probs": [0.70, 0.05, 0.10, 0.10, 0.05]
                },
            },
            rollout_info={
                "action_td_targets": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 4.0,
                    "3": 0.0,
                    "4": 0.0,
                }
            },
            teacher_only=True,
        )
        self.assertEqual(selected_action, 2)
        self.assertTrue(planner_stats["teacher_only"])
        self.assertTrue(planner_stats["applied"])

        slow_logits = torch.zeros(3, requires_grad=True)
        fast_logits = torch.zeros(2, requires_grad=True)
        event_logits = torch.zeros(2, requires_grad=True)
        loss, support = agent._compute_env_action_model_teacher_distillation_loss(
            batch_outputs=[
                {
                    "slow_logits": slow_logits,
                    "fast_logits": fast_logits,
                    "event_logits": event_logits,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "counterfactual_teacher_planner": planner_stats,
                    }
                }
            ],
            batch_action_masks=[[True] * 5],
        )
        loss.backward()
        self.assertEqual(support, 1)
        self.assertGreater(float(event_logits.grad.abs().sum()), 0.0)

    def test_sa_v118_distills_positive_online_planner_advantage(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v118_conservative_planner_distillation_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["env_action_model_online_planner_enabled"])
        self.assertTrue(kwargs["env_action_model_teacher_distillation_enabled"])
        self.assertTrue(
            kwargs["env_action_model_teacher_distillation_online_planner_enabled"]
        )

        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            **kwargs,
        )
        slow_logits = torch.zeros(3, requires_grad=True)
        fast_logits = torch.zeros(2, requires_grad=True)
        event_logits = torch.zeros(2, requires_grad=True)
        loss, support = agent._compute_env_action_model_teacher_distillation_loss(
            batch_outputs=[
                {
                    "slow_logits": slow_logits,
                    "fast_logits": fast_logits,
                    "event_logits": event_logits,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "action_projection": {
                            "masked_env_action_probs": [0.2] * 5,
                        },
                        "online_counterfactual_planner": {
                            "applied": True,
                            "selected_action": 2,
                            "score_margin": 0.6,
                            "model_advantage_gain": 0.8,
                        },
                    }
                }
            ],
            batch_action_masks=[[True] * 5],
        )
        loss.backward()
        self.assertEqual(support, 1)
        self.assertGreater(float(event_logits.grad.abs().sum()), 0.0)

    def test_sa_v118_rejects_low_advantage_online_planner_label(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_teacher_distillation_enabled=True,
            env_action_model_teacher_distillation_coef=0.45,
            env_action_model_teacher_distillation_online_planner_enabled=True,
            env_action_model_teacher_distillation_min_advantage=0.1,
        )
        loss, support = agent._compute_env_action_model_teacher_distillation_loss(
            batch_outputs=[
                {
                    "slow_logits": torch.zeros(3, requires_grad=True),
                    "fast_logits": torch.zeros(2, requires_grad=True),
                    "event_logits": torch.zeros(2, requires_grad=True),
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "online_counterfactual_planner": {
                            "applied": True,
                            "selected_action": 2,
                            "score_margin": 0.6,
                            "model_advantage_gain": 0.05,
                        }
                    }
                }
            ],
            batch_action_masks=[[True] * 5],
        )
        self.assertEqual(support, 0)
        self.assertEqual(float(loss.item()), 0.0)

    def test_sa_v119_requires_model_and_realized_advantage_agreement(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v119_doubly_validated_planner_distillation_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_teacher_distillation_realized_advantage_gate_enabled"
            ]
        )
        agent = build_agent("sa_ghmappo", random_seed=7, **kwargs)
        outputs = [
            {
                "slow_logits": torch.zeros(3, requires_grad=True),
                "fast_logits": torch.zeros(2, requires_grad=True),
                "event_logits": torch.zeros(2, requires_grad=True),
            }
        ]
        rows = [
            {
                "action_info": {
                    "action_projection": {
                        "masked_env_action_probs": [0.2] * 5,
                    },
                    "online_counterfactual_planner": {
                        "applied": True,
                        "selected_action": 2,
                        "score_margin": 0.6,
                        "model_advantage_gain": 0.8,
                    },
                }
            }
        ]
        rejected_loss, rejected_support = (
            agent._compute_env_action_model_teacher_distillation_loss(
                batch_outputs=outputs,
                batch_rows=rows,
                batch_action_masks=[[True] * 5],
                batch_realized_advantage=torch.tensor([-0.5]),
            )
        )
        accepted_loss, accepted_support = (
            agent._compute_env_action_model_teacher_distillation_loss(
                batch_outputs=outputs,
                batch_rows=rows,
                batch_action_masks=[[True] * 5],
                batch_realized_advantage=torch.tensor([0.5]),
            )
        )
        self.assertEqual(rejected_support, 0)
        self.assertEqual(float(rejected_loss.item()), 0.0)
        self.assertEqual(accepted_support, 1)
        self.assertGreater(float(accepted_loss.item()), 0.0)

    def test_sa_v120_retains_realized_advantage_gate_with_stronger_projection(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v120_doubly_validated_strong_distillation_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_teacher_distillation_realized_advantage_gate_enabled"
            ]
        )
        self.assertEqual(kwargs["env_action_model_teacher_distillation_coef"], 0.45)
        self.assertEqual(kwargs["env_action_model_teacher_distillation_max_weight"], 3.0)
        self.assertEqual(
            kwargs["env_action_model_teacher_distillation_behavior_kl_coef"],
            0.20,
        )

    def test_sa_v121_adds_realized_advantage_gated_logit_margin(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v121_realized_advantage_margin_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(
            kwargs[
                "env_action_model_teacher_distillation_realized_advantage_gate_enabled"
            ]
        )
        self.assertEqual(
            kwargs["env_action_model_teacher_distillation_logit_margin"],
            0.25,
        )
        agent = build_agent("sa_ghmappo", random_seed=7, **kwargs)
        event_logits = torch.zeros(2, requires_grad=True)
        loss, support = agent._compute_env_action_model_teacher_distillation_loss(
            batch_outputs=[
                {
                    "slow_logits": torch.zeros(3, requires_grad=True),
                    "fast_logits": torch.zeros(2, requires_grad=True),
                    "event_logits": event_logits,
                }
            ],
            batch_rows=[
                {
                    "action_info": {
                        "action_projection": {
                            "masked_env_action_probs": [0.2] * 5,
                        },
                        "online_counterfactual_planner": {
                            "applied": True,
                            "selected_action": 2,
                            "score_margin": 0.6,
                            "model_advantage_gain": 0.8,
                        },
                    }
                }
            ],
            batch_action_masks=[[True] * 5],
            batch_realized_advantage=torch.tensor([0.5]),
        )
        loss.backward()
        self.assertEqual(support, 1)
        self.assertGreater(float(event_logits.grad.abs().sum()), 0.0)

    def test_sa_v115_profile_uses_training_only_policy_iteration(self) -> None:
        from scripts.train_sa_ghmappo_real_sample import (
            MECHANISM_COVERAGE_PROFILES,
            build_sa_ghmappo_profile_kwargs,
        )

        profile = "top_journal_mechanism_v115_training_policy_iteration_mappo"
        kwargs = build_sa_ghmappo_profile_kwargs(profile)
        self.assertIn(profile, MECHANISM_COVERAGE_PROFILES)
        self.assertTrue(kwargs["env_action_model_training_planner_enabled"])
        self.assertFalse(kwargs["env_action_model_online_planner_enabled"])

    def test_sa_v115_training_planner_keeps_native_mode_separate(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            env_action_model_training_planner_enabled=True,
        )
        selected_action, planner_stats = agent.select_env_action_from_model_targets(
            action_info={
                "final_env_action": 0,
                "action_mask": [True] * 5,
                "action_projection": {
                    "masked_env_action_probs": [0.20] * 5,
                },
            },
            rollout_info={
                "action_td_targets": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 4.0,
                    "3": 0.0,
                    "4": 0.0,
                }
            },
            training_only=True,
        )
        self.assertEqual(selected_action, 2)
        self.assertTrue(planner_stats["training_only"])
        self.assertFalse(planner_stats["teacher_only"])

    def test_sa_v93_planner_prefers_realized_mechanism_over_return_only_action(
        self,
    ) -> None:
        from scripts.train_sa_ghmappo_real_sample import build_sa_ghmappo_profile_kwargs

        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            **build_sa_ghmappo_profile_kwargs(
                "top_journal_mechanism_v93_mechanism_aware_online_mappo"
            ),
        )
        selected_action, planner_stats = agent.select_env_action_from_model_targets(
            action_info={
                "final_env_action": 0,
                "action_mask": [True, True, True, True, True],
                "action_projection": {
                    "masked_env_action_probs": [0.20, 0.20, 0.20, 0.20, 0.20]
                },
            },
            rollout_info={
                "action_td_targets": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 4.0,
                    "3": 0.0,
                    "4": 0.0,
                },
                "action_mechanism_targets": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 0.0,
                    "3": 0.0,
                    "4": 1.0,
                },
            },
        )

        self.assertEqual(selected_action, 4)
        self.assertEqual(planner_stats["mechanism_coef"], 2.0)
        self.assertGreater(planner_stats["mechanism_advantage"]["4"], 0.0)

    def test_marl_evaluation_skips_training_only_model_targets(self) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        class FakeAgent:
            agent_name = "fake"

            def learn(self, rollout):
                return {"collected_steps": len(rollout)}

        class StubTrainer(MARLOnPolicyTrainer):
            def __init__(self) -> None:
                self._agent = FakeAgent()
                self.collect_model_targets = None

            def collect_episode(
                self,
                run_metadata=None,
                collect_model_targets=True,
            ):
                del run_metadata
                self.collect_model_targets = collect_model_targets
                return {"episode_status": {}}, [{"reward": 1.0}]

        trainer = StubTrainer()
        summary = trainer.run_episode(learn=False)

        self.assertFalse(trainer.collect_model_targets)
        self.assertTrue(
            summary["agent_info"]["learn_info"]["policy_update_skipped"]
        )

    def test_marl_learned_planner_action_reaches_environment(self) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        class FakeEnv:
            def __init__(self) -> None:
                self.actions = []

            def reset(self):
                return [0], {"semantic_state": {}, "action_mask": [True] * 5}

            def step(self, action: int):
                self.actions.append(int(action))
                return (
                    [1],
                    1.0,
                    True,
                    False,
                    {"semantic_state": {}, "action_mask": [True] * 5},
                )

        class FakeRecorder:
            def start_episode(self, run_metadata=None) -> None:
                del run_metadata

            def build_summary(self):
                return {"episode_status": {}}

        class FakeAgent:
            agent_name = "fake"
            _learned_transition_model_planner_enabled = True

            def act(self, observation, info):
                del observation, info
                return 0, {
                    "final_env_action": 0,
                    "action_mask": [True] * 5,
                    "value": 0.0,
                    "log_prob": 0.0,
                }

            def predict_learned_transition_targets(self, observation, action_info):
                del observation, action_info
                return {"source": "learned_transition_ensemble", "action_td_targets": {}}

            def select_env_action_from_model_targets(self, action_info, rollout_info):
                del action_info, rollout_info
                return 2, {"enabled": True, "applied": True, "selected_action": 2}

            def relabel_action_info_for_env_action(
                self, action_info, decision_info, env_action, planner_stats
            ):
                del decision_info
                updated = dict(action_info)
                updated["final_env_action"] = int(env_action)
                updated["online_counterfactual_planner"] = dict(planner_stats)
                return updated

        env = FakeEnv()
        trainer = MARLOnPolicyTrainer(
            env=env,
            agent=FakeAgent(),
            recorder=FakeRecorder(),
            max_steps=1,
        )
        _, rollout = trainer.collect_episode(collect_model_targets=False)

        self.assertEqual(env.actions, [2])
        self.assertEqual(rollout[0]["action"], 2)

    def test_ucc_uses_ucb_during_training_and_lcb_during_evaluation(self) -> None:
        class FakeModel:
            ready = True
            sample_count = 128
            update_count = 3
            uncertainty_scale = 1.0

            def predict(self, observation, action_ids):
                del observation
                return {
                    "ready": True,
                    "td_target_mean": [10.0 for _ in action_ids],
                    "td_target_std": [2.0 for _ in action_ids],
                    "reward_mean": [0.0 for _ in action_ids],
                    "reward_std": [0.0 for _ in action_ids],
                }

        training_agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            learned_transition_model_enabled=True,
            learned_transition_model_planner_enabled=True,
            learned_transition_model_risk_coef=0.65,
            learned_transition_model_exploration_coef=0.80,
        )
        training_agent._learned_transition_model = FakeModel()
        training_agent._update_count = 1
        train_targets = training_agent.predict_learned_transition_targets(
            observation=[0.0] * 9,
            action_info={"action_mask": [True] * 5},
        )

        evaluation_agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            deterministic_action=True,
            learned_transition_model_enabled=True,
            learned_transition_model_planner_enabled=True,
            learned_transition_model_risk_coef=0.65,
            learned_transition_model_exploration_coef=0.80,
        )
        evaluation_agent._learned_transition_model = FakeModel()
        evaluation_agent._update_count = 1
        eval_targets = evaluation_agent.predict_learned_transition_targets(
            observation=[0.0] * 9,
            action_info={"action_mask": [True] * 5},
        )

        self.assertEqual(train_targets["exploration_mode"], "ucb_train")
        self.assertEqual(eval_targets["exploration_mode"], "lcb_eval")
        self.assertAlmostEqual(train_targets["action_td_targets"]["0"], 10.3)
        self.assertAlmostEqual(eval_targets["action_td_targets"]["0"], 8.7)

    def test_counterfactual_calibration_samples_are_emitted(self) -> None:
        from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

        class BranchEnv:
            def __deepcopy__(self, memo):
                del memo
                return self

            def step(self, action: int):
                del action
                return [1.0] * 9, 2.0, True, False, {}

        class FakeRecorder:
            def start_episode(self, run_metadata=None) -> None:
                del run_metadata

            def build_summary(self):
                return {"episode_status": {}}

        trainer = MARLOnPolicyTrainer(
            env=BranchEnv(),
            agent=object(),
            recorder=FakeRecorder(),
            max_steps=1,
        )
        payload = trainer._collect_env_action_counterfactual_targets(
            observation=[0.0] * 9,
            action_info={
                "action_mask": [True] * 5,
                "env_action_model_rollout_enabled": True,
                "env_action_model_rollout_horizon": 1,
                "env_action_model_rollout_horizons": [1],
            },
            algorithm_memory={},
            decision_info={},
            run_metadata={},
        )

        self.assertEqual(len(payload["counterfactual_transition_samples"]), 5)
        self.assertEqual(
            {sample["action"] for sample in payload["counterfactual_transition_samples"]},
            {0, 1, 2, 3, 4},
        )

    def test_sa_v70_sparse_tail_option_prior_boosts_mechanism_option(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "idle_or_sparse"
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"][1]["cached_adapter_ids"] = []
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            option_gate_enabled=True,
            option_gate_context_prior_enabled=True,
            option_gate_idle_recovery_mechanism_prior_enabled=True,
            sparse_handoff_recovery_prior_enabled=True,
            sparse_handoff_recovery_prefetch_bias=1.0,
            sparse_handoff_recovery_prepare_bias=1.0,
            sparse_handoff_recovery_current_fill_bias=0.0,
            sparse_handoff_recovery_steady_bias=0.0,
            sparse_handoff_recovery_local_penalty=1.0,
            sparse_handoff_recovery_min_context=0.0,
            sparse_handoff_recovery_max_eta=16,
            sparse_handoff_option_prior_enabled=True,
            sparse_handoff_option_prepare_bias=6.0,
            sparse_handoff_option_popularity_penalty=3.0,
            sparse_handoff_option_local_penalty=4.0,
            sparse_handoff_option_min_context=0.0,
            sparse_handoff_option_max_eta=16,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
            "option_logits": torch.zeros(4),
        }

        adjusted = agent._apply_sparse_handoff_recovery_prior(
            policy_output,
            state,
            run_metadata={"window_class": "idle_or_sparse"},
        )
        candidate_info = agent._build_option_gate_candidates(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            base_env_action=3,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        info = adjusted["sparse_handoff_recovery_prior_info"]
        self.assertTrue(info["active"])
        self.assertGreater(info["option_prepare_bias"], 0.0)
        self.assertGreater(float(adjusted["option_logits"][3]), 0.0)
        self.assertLess(float(adjusted["option_logits"][1]), 0.0)
        self.assertLess(float(adjusted["option_logits"][2]), 0.0)
        self.assertTrue(candidate_info["idle_recovery_context"])
        self.assertEqual(candidate_info["prior_target"], 3)
        self.assertTrue(candidate_info["sparse_tail_risk_option_context"]["active"])

    def test_sa_v69_sparse_realization_credit_rewards_success_and_penalizes_failure(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            sparse_handoff_realization_credit_enabled=True,
            sparse_handoff_realization_success_bonus=1.5,
            sparse_handoff_realization_ready_bonus=1.2,
            sparse_handoff_realization_prefetch_bonus=1.0,
            sparse_handoff_realization_failed_prepare_penalty=2.4,
            sparse_handoff_realization_local_penalty=1.4,
            sparse_handoff_realization_min_context=0.0,
        )
        base_row = {
            "action": 4,
            "decision_info": {
                "semantic_state": deepcopy(_minimal_semantic_state()),
                "run_metadata": {"window_class": "idle_or_sparse"},
            },
            "action_info": {
                "final_env_action": 4,
                "prepare_window_score": 0.42,
                "temporal_urgency": 0.34,
                "prediction_confidence": 0.70,
                "raw_handoff_candidate": True,
                "option_gate": {
                    "window_class": "idle_or_sparse",
                    "idle_recovery_context": True,
                },
            },
            "reward": 1.0,
        }
        success_row = deepcopy(base_row)
        success_row["env_info"] = {
            "metrics_protocol": {
                "predicted_handoff_signal": True,
                "mechanism_success_rate": 0.75,
                "handoff_ready_rate": 1.0,
                "prefetch_validated_hit_count": 4.0,
                "prefetch_validated_hit_rate": 1.0,
                "migration_success_count": 1.0,
            }
        }
        failed_row = deepcopy(base_row)
        failed_row["env_info"] = {
            "metrics_protocol": {
                "predicted_handoff_signal": True,
                "mechanism_success_rate": 0.0,
                "handoff_ready_rate": 0.0,
                "prefetch_validated_hit_count": 0.0,
                "migration_failed_count": 4.0,
                "migration_prepare_requested": True,
            }
        }

        self.assertGreater(
            agent._sparse_handoff_realization_credit(success_row, env_action=4),
            0.0,
        )
        self.assertLess(
            agent._sparse_handoff_realization_credit(failed_row, env_action=4),
            0.0,
        )
        self.assertLess(
            agent._sparse_handoff_realization_credit(success_row, env_action=2),
            agent._sparse_handoff_realization_credit(success_row, env_action=3),
        )

    def test_sa_v62_policy_prior_boosts_cache_fill_when_current_adapter_missing(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "mechanism_activating"
        state["rsus"][0]["cached_adapter_ids"] = []
        state["rsus"][1]["cached_adapter_ids"] = []
        state["current_workflow_node"]["required_adapter"] = "adapter_tracking"
        state["vehicles"][0]["associated_rsu_id"] = "rsu_a"
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.20},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.80},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_min_context=0.45,
            opportunity_constrained_low_context=0.20,
            opportunity_constrained_current_bias=0.0,
            opportunity_constrained_prepare_penalty=0.0,
            opportunity_constrained_prefetch_penalty=0.0,
            cache_feasibility_prior_enabled=True,
            cache_feasibility_cache_fill_bias=5.70,
            cache_feasibility_steady_penalty=7.40,
            cache_feasibility_prepare_penalty=1.10,
            cache_feasibility_prefetch_penalty=1.35,
            cache_feasibility_min_context=0.0,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_opportunity_constrained_policy(raw, state)

        info = adjusted["opportunity_constrained_policy_info"]
        self.assertTrue(info["cache_feasibility_prior_active"])
        self.assertFalse(info["current_cache_ready"])
        self.assertGreater(float(adjusted["env_action_logits_bias"][0].item()), 5.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][3].item()), -7.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][4].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][1].item()), 0.0)
        self.assertGreater(float(adjusted["slow_logits"][1].item()), float(raw["slow_logits"][1].item()))
        self.assertLess(float(adjusted["fast_logits"][0].item()), float(raw["fast_logits"][0].item()))

    def test_sa_v63_current_ready_prior_suppresses_prepare_under_strong_signal_until_current_cache_ready(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "mechanism_activating"
        state["rsus"][0]["cached_adapter_ids"] = []
        state["rsus"][1]["cached_adapter_ids"] = []
        state["current_workflow_node"]["required_adapter"] = "adapter_tracking"
        state["vehicles"][0]["associated_rsu_id"] = "rsu_a"
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.92},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.04},
            "dwell_time": {"veh_1": 3.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_b", "rsu_b", "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_min_context=0.45,
            opportunity_constrained_low_context=0.20,
            opportunity_constrained_prepare_bias=3.0,
            opportunity_constrained_prefetch_bias=2.0,
            opportunity_constrained_current_bias=0.0,
            opportunity_constrained_prepare_penalty=0.0,
            opportunity_constrained_prefetch_penalty=0.0,
            cache_feasibility_prior_enabled=True,
            cache_feasibility_cache_fill_bias=7.20,
            cache_feasibility_steady_penalty=6.20,
            cache_feasibility_current_miss_prepare_penalty=8.80,
            cache_feasibility_current_miss_prefetch_penalty=5.60,
            cache_feasibility_min_context=0.0,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_opportunity_constrained_policy(raw, state)

        info = adjusted["opportunity_constrained_policy_info"]
        self.assertTrue(info["strong_opportunity"])
        self.assertTrue(info["cache_feasibility_prior_active"])
        self.assertFalse(info["current_cache_ready"])
        self.assertGreater(info["cache_feasibility_current_miss_prepare_penalty"], 8.0)
        self.assertGreater(info["cache_feasibility_current_miss_prefetch_penalty"], 5.0)
        self.assertGreater(float(adjusted["env_action_logits_bias"][0].item()), 6.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][4].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][1].item()), 0.0)
        self.assertLess(float(adjusted["event_logits"][1].item()), float(raw["event_logits"][1].item()))
        self.assertLess(float(adjusted["slow_logits"][2].item()), float(raw["slow_logits"][2].item()))

    def test_sa_v64_handoff_alignment_barrier_suppresses_mismatched_prepare(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "mechanism_activating"
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"].append(
            {
                "rsu_id": "rsu_c",
                "cached_adapter_ids": [],
                "cache_capacity": 4,
            }
        )
        state["current_workflow_node"]["required_adapter"] = "adapter_tracking"
        state["vehicles"][0]["associated_rsu_id"] = "rsu_a"
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0, "rsu_c": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_c"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.94},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.03},
            "dwell_time": {"veh_1": 2.0},
            "next_rsu_sequence": {"veh_1": ["rsu_c", "rsu_b", "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_min_context=0.45,
            opportunity_constrained_low_context=0.20,
            opportunity_constrained_prepare_bias=3.0,
            opportunity_constrained_prefetch_bias=2.0,
            opportunity_constrained_current_bias=0.0,
            opportunity_constrained_prepare_penalty=0.0,
            opportunity_constrained_prefetch_penalty=0.0,
            cache_feasibility_prior_enabled=True,
            cache_feasibility_cache_fill_bias=8.40,
            cache_feasibility_steady_penalty=6.20,
            cache_feasibility_current_miss_prepare_penalty=12.80,
            cache_feasibility_current_miss_prefetch_penalty=7.80,
            cache_feasibility_min_context=0.0,
            handoff_alignment_barrier_enabled=True,
            handoff_alignment_barrier_prepare_penalty=12.50,
            handoff_alignment_barrier_prefetch_penalty=7.20,
            handoff_alignment_barrier_current_fill_bias=8.80,
            handoff_alignment_barrier_target_mismatch_penalty=5.60,
            handoff_alignment_barrier_late_eta_penalty=3.20,
            handoff_alignment_barrier_min_context=0.0,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_opportunity_constrained_policy(raw, state)

        info = adjusted["opportunity_constrained_policy_info"]
        self.assertTrue(info["strong_opportunity"])
        self.assertFalse(info["cache_feasibility_prior_active"])
        self.assertTrue(info["current_cache_ready"])
        self.assertTrue(info["handoff_alignment_barrier_active"])
        self.assertTrue(info["handoff_alignment_target_mismatch"])
        self.assertEqual(info["predicted_first_non_current_rsu_id"], "rsu_c")
        self.assertGreater(info["handoff_alignment_prepare_penalty"], 12.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][4].item()), -10.0)
        self.assertLess(float(adjusted["event_logits"][1].item()), float(raw["event_logits"][1].item()))

    def test_sa_v64_service_credit_penalizes_prepare_after_adapter_miss(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            service_continuity_teacher_enabled=True,
            service_continuity_prepare_bonus=0.34,
            service_continuity_current_bonus=1.78,
            env_action_adapter_miss_counterfactual_coef=4.20,
            handoff_alignment_barrier_enabled=True,
            handoff_alignment_barrier_prepare_penalty=12.50,
            handoff_alignment_barrier_prefetch_penalty=7.20,
        )
        row = {
            "action": 4,
            "decision_info": {
                "semantic_state": deepcopy(_minimal_semantic_state()),
                "run_metadata": {"window_class": "mechanism_activating"},
            },
            "action_info": {
                "final_env_action": 4,
                "prepare_window_score": 0.80,
                "temporal_urgency": 0.70,
                "prediction_confidence": 0.90,
                "predicted_handoff_target_valid": True,
                "raw_handoff_candidate": True,
            },
            "env_info": {
                "metrics_protocol": {
                    "predicted_handoff_signal": True,
                    "has_predicted_handoff_target": True,
                    "mechanism_success_rate": 0.0,
                    "handoff_ready_rate": 0.40,
                    "workflow_continuity_rate": 0.58,
                    "adapter_miss_count": 3.0,
                    "cache_miss_penalty_sum": 1.2,
                    "migration_failed_count": 4.0,
                    "stall_occurred": True,
                }
            },
            "reward": -1.0,
        }

        self.assertLess(
            agent._service_continuity_counterfactual_credit(
                row,
                env_action=4,
                window_class="mechanism_activating",
                context_strength=0.90,
                timing_support=0.80,
                ready_score=0.0,
            ),
            0.0,
        )

    def test_sa_v65_argmax_margin_penalizes_prepare_over_current_service(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["window_class"] = "mechanism_activating"
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            service_continuity_teacher_enabled=True,
            service_continuity_prepare_bonus=0.18,
            service_continuity_current_bonus=2.18,
            env_action_adapter_miss_counterfactual_coef=4.80,
            handoff_alignment_barrier_enabled=True,
            handoff_alignment_barrier_prepare_penalty=12.50,
            handoff_alignment_barrier_prefetch_penalty=7.20,
            argmax_margin_regularization_enabled=True,
            argmax_margin_coef=0.76,
            argmax_margin_min_gap=0.42,
            argmax_margin_tail_risk_threshold=0.04,
        )
        row = {
            "action": 4,
            "decision_info": {
                "semantic_state": deepcopy(state),
                "run_metadata": {"window_class": "mechanism_activating"},
            },
            "action_info": {
                "final_env_action": 4,
                "prepare_window_score": 0.80,
                "temporal_urgency": 0.70,
                "prediction_confidence": 0.90,
                "predicted_handoff_target_valid": True,
                "raw_handoff_candidate": True,
            },
            "env_info": {
                "metrics_protocol": {
                    "predicted_handoff_signal": True,
                    "has_predicted_handoff_target": True,
                    "mechanism_success_rate": 0.0,
                    "handoff_ready_rate": 0.40,
                    "workflow_continuity_rate": 0.58,
                    "adapter_miss_count": 3.0,
                    "cache_miss_penalty_sum": 1.2,
                    "delay_penalty_sum": 8.0,
                    "migration_failed_count": 4.0,
                    "stall_occurred": True,
                }
            },
            "reward": -1.0,
        }
        with torch.no_grad():
            policy_output = agent._forward_policy(
                state,
                run_metadata={"window_class": "mechanism_activating"},
            )
            policy_output = dict(policy_output)
            policy_output["env_action_logits_bias"] = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 3.0],
                dtype=policy_output["event_logits"].dtype,
                device=policy_output["event_logits"].device,
            )

        loss = agent._compute_argmax_margin_regularization_loss(
            batch_outputs=[policy_output],
            batch_action_masks=[[True, True, True, True, True]],
            batch_rows=[row],
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(
            agent._counterfactual_teacher_action_credit(row, 0),
            agent._counterfactual_teacher_action_credit(row, 4),
        )

    def test_sa_v54_policy_path_applies_net_and_service_completion_gates(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["workflow"]["nodes"] = [
            {"node_id": "n1", "predecessors": [], "successors": []}
        ]
        state["workflow"]["completed_node_ids"] = []
        state["current_workflow_node"]["successors"] = []
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.2},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.8},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            opportunity_constrained_policy_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.0,
            net_advantage_prepare_gate_min_score=0.42,
            net_advantage_prepare_gate_margin=0.10,
            service_completion_gate_enabled=True,
            service_completion_gate_bias=3.2,
            service_completion_gate_remaining_nodes_threshold=2,
            service_completion_gate_fallback_suppression_scale=1.1,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._forward_policy(state)

        self.assertIn("net_advantage_prepare_gate_info", adjusted)
        self.assertIn("service_completion_gate_info", adjusted)
        service_info = adjusted["service_completion_gate_info"]
        self.assertTrue(service_info["active"])
        self.assertEqual(service_info["target_action"], 3)
        self.assertGreater(float(adjusted["env_action_logits_bias"][3].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][2].item()), 0.0)
        self.assertGreater(float(adjusted["fast_logits"][0].item()), float(raw["fast_logits"][0].item()))
        self.assertLess(float(adjusted["fast_logits"][1].item()), float(raw["fast_logits"][1].item()))

    def test_sa_v54_net_gate_boosts_coverage_gap_recovery_prepare(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": None},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.55},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.35},
            "dwell_time": {"veh_1": 1.0},
            "next_rsu_sequence": {"veh_1": [None, None, "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.4,
            net_advantage_prepare_gate_min_score=0.50,
            net_advantage_prepare_gate_margin=0.12,
            net_advantage_prepare_gate_current_scale=1.0,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_net_advantage_prepare_gate(raw, state)

        gate_info = adjusted["net_advantage_prepare_gate_info"]
        raw_margin = float((raw["event_logits"][1] - raw["event_logits"][0]).item())
        adjusted_margin = float((adjusted["event_logits"][1] - adjusted["event_logits"][0]).item())

        self.assertTrue(gate_info["target_differs"])
        self.assertIsNone(gate_info["current_rsu_id"])
        self.assertGreater(adjusted_margin, raw_margin)
        self.assertGreater(float(adjusted["env_action_logits_bias"][4].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][2].item()), 0.0)
        self.assertLess(float(adjusted["fast_logits"][1].item()), float(raw["fast_logits"][1].item()))

    def test_sa_v55_policy_path_strongly_suppresses_gap_fallback(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": None},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.48},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.40},
            "dwell_time": {"veh_1": 1.0},
            "next_rsu_sequence": {"veh_1": [None, None, "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_logit_bias_strength=0.0,
            digital_twin_policy_prior_enabled=False,
            opportunity_constrained_policy_enabled=False,
            backhaul_aware_policy_enabled=False,
            continuity_guard_enabled=False,
            event_logit_sharpening_final_scale=1.0,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.65,
            net_advantage_prepare_gate_min_score=0.50,
            net_advantage_prepare_gate_margin=0.12,
            coverage_recovery_gate_bias_scale=1.65,
            coverage_recovery_gate_min_scale=0.62,
            coverage_recovery_gate_fallback_suppression_scale=2.35,
            coverage_recovery_gate_fast_suppression_scale=1.50,
            coverage_recovery_gate_current_suppression_scale=0.36,
            service_completion_gate_enabled=True,
            service_completion_gate_bias=3.35,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._forward_policy(state)

        gate_info = adjusted["net_advantage_prepare_gate_info"]
        raw_margin = float((raw["event_logits"][1] - raw["event_logits"][0]).item())
        adjusted_margin = float((adjusted["event_logits"][1] - adjusted["event_logits"][0]).item())

        self.assertTrue(gate_info["target_differs"])
        self.assertIsNone(gate_info["current_rsu_id"])
        self.assertGreater(gate_info["coverage_recovery_scale"], 0.60)
        self.assertGreater(gate_info["coverage_recovery_bias"], 0.0)
        self.assertGreater(gate_info["coverage_recovery_fallback_suppression"], 3.0)
        self.assertGreater(adjusted_margin, raw_margin)
        self.assertGreater(float(adjusted["env_action_logits_bias"][4].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][2].item()), -3.0)
        self.assertLess(float(adjusted["fast_logits"][1].item()), float(raw["fast_logits"][1].item()))

    def test_sa_v55_no_rsu_option_candidate_uses_coverage_recovery_prepare(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": None},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.60},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.30},
            "dwell_time": {"veh_1": 1.0},
            "next_rsu_sequence": {"veh_1": [None, "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            coverage_recovery_gate_min_scale=0.62,
        )

        candidate_info = agent._build_option_gate_candidates(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            base_env_action=2,
            run_metadata={"window_class": "mechanism_activating"},
        )

        self.assertTrue(candidate_info["no_rsu_available"])
        self.assertTrue(candidate_info["coverage_recovery_no_rsu"])
        self.assertEqual(candidate_info["option_actions"][2], 4)
        self.assertEqual(candidate_info["option_actions"][3], 4)
        self.assertEqual(candidate_info["prior_target"], 3)

    def test_sa_v55_coverage_recovery_guard_converts_gap_fallback_to_prepare(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": None},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.60},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.30},
            "dwell_time": {"veh_1": 1.0},
            "next_rsu_sequence": {"veh_1": [None, "rsu_b"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.65,
            coverage_recovery_gate_bias_scale=1.65,
            coverage_recovery_gate_min_scale=0.62,
            coverage_recovery_guard_enabled=True,
        )

        with torch.no_grad():
            policy_output = agent._forward_policy(state)
        guard_info = agent._apply_coverage_recovery_guard_to_actions(
            semantic_state=state,
            policy_output=policy_output,
            selected_actions={"slow": 0, "fast": 1, "event": 0},
            action_mask=[True, True, True, True, True],
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["original_action"], 2)
        self.assertEqual(guard_info["guarded_action"], 4)
        self.assertGreaterEqual(guard_info["coverage_recovery_scale"], 0.62)

    def test_sa_v56_final_guard_recovers_prepare_after_option_fallback(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            coverage_recovery_final_guard_enabled=True,
            coverage_recovery_final_guard_min_scale=0.18,
            coverage_recovery_final_guard_min_confidence=0.35,
        )
        policy_output = {
            "net_advantage_prepare_gate_info": {
                "current_rsu_id": None,
                "predicted_target_valid": True,
                "target_differs": True,
                "coverage_recovery_scale": 0.05,
                "net_advantage_score": 0.05,
                "prediction_confidence": 0.60,
                "predicted_handoff_target_rsu_id": "rsu_b",
                "predicted_next_rsu_id": None,
            }
        }

        guard_info = agent._apply_coverage_recovery_final_guard_to_env_action(
            semantic_state=deepcopy(_minimal_semantic_state()),
            policy_output=policy_output,
            env_action=2,
            action_mask=[True, True, True, True, True],
            option_gate_info={"option_label": "no_rsu_local", "option_env_action": 2},
        )

        self.assertTrue(guard_info["guarded"])
        self.assertEqual(guard_info["original_action"], 2)
        self.assertEqual(guard_info["guarded_action"], 4)
        self.assertEqual(
            guard_info["reason"],
            "partial_observation_handoff_target_memory_prefers_prepare",
        )

    def test_sa_v56_final_guard_respects_low_memory_confidence(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            coverage_recovery_final_guard_enabled=True,
            coverage_recovery_final_guard_min_scale=0.18,
            coverage_recovery_final_guard_min_confidence=0.35,
        )
        policy_output = {
            "net_advantage_prepare_gate_info": {
                "current_rsu_id": None,
                "predicted_target_valid": True,
                "target_differs": True,
                "coverage_recovery_scale": 0.04,
                "net_advantage_score": 0.04,
                "prediction_confidence": 0.20,
                "predicted_handoff_target_rsu_id": "rsu_b",
                "predicted_next_rsu_id": None,
            }
        }

        guard_info = agent._apply_coverage_recovery_final_guard_to_env_action(
            semantic_state=deepcopy(_minimal_semantic_state()),
            policy_output=policy_output,
            env_action=2,
            action_mask=[True, True, True, True, True],
            option_gate_info={"option_label": "no_rsu_local", "option_env_action": 2},
        )

        self.assertFalse(guard_info["guarded"])
        self.assertEqual(guard_info["reason"], "coverage_recovery_memory_below_min")

    def test_sa_v57_no_rsu_candidate_prefers_service_continuity(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            idle_popularity_no_rsu_service_continuity_enabled=True,
        )
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"]["predicted_handoff_vehicle_ids"] = []
        state["predictions"]["predicted_next_rsu_by_vehicle"] = {}
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"] = {}
        state["predictions"]["next_rsu_sequence"] = {"veh_1": []}

        action, reason, _extra = agent._idle_popularity_candidate_action(
            state,
            [True, True, True, True, True],
        )

        self.assertEqual(action, 3)
        self.assertEqual(reason, "no_associated_rsu_service_continuity")

    def test_sa_v58_no_rsu_service_continuity_can_override_nonlocal_action(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            idle_popularity_fallback_enabled=True,
            idle_popularity_fallback_only_vehicle_fallback=True,
            idle_popularity_no_rsu_service_continuity_enabled=True,
            idle_popularity_no_rsu_any_action_override_enabled=True,
        )
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"]["predicted_handoff_vehicle_ids"] = []
        state["predictions"]["predicted_next_rsu_by_vehicle"] = {}
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"] = {}
        state["predictions"]["next_rsu_sequence"] = {"veh_1": []}

        info = agent._maybe_apply_idle_popularity_fallback(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            original_env_action=4,
            deterministic=True,
        )

        self.assertTrue(info["applied"])
        self.assertEqual(info["fallback_action"], 3)
        self.assertEqual(info["reason"], "no_rsu_service_continuity_replaced_by_popularity_option")

    def test_sa_v58_idle_recovery_keeps_mechanism_option_prior(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            option_gate_context_prior_enabled=True,
            option_gate_idle_recovery_mechanism_prior_enabled=True,
            option_gate_idle_recovery_min_context=0.0,
            idle_popularity_no_rsu_service_continuity_enabled=True,
            net_advantage_prepare_gate_enabled=True,
        )
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"]["predicted_handoff_vehicle_ids"] = ["veh_1"]
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"] = {"veh_1": "rsu_b"}
        state["predictions"]["predicted_next_rsu_by_vehicle"] = {"veh_1": "rsu_b"}
        state["predictions"]["prediction_confidence_by_vehicle"] = {"veh_1": 0.80}
        state["predictions"]["prediction_uncertainty_by_vehicle"] = {"veh_1": 0.20}
        state["predictions"]["next_rsu_sequence"] = {"veh_1": ["rsu_b", "rsu_b"]}

        info = agent._build_option_gate_candidates(
            semantic_state=state,
            action_mask=[True, True, True, True, True],
            base_env_action=2,
            run_metadata={"window_class": "idle_or_sparse"},
        )

        self.assertTrue(info["mechanism_available"])
        self.assertTrue(info["idle_recovery_context"])
        self.assertEqual(info["prior_target"], 3)
        self.assertEqual(info["option_actions"][3], 4)

    def test_sa_v57_service_continuity_teacher_penalizes_local_fallback(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            service_continuity_teacher_enabled=True,
            service_continuity_current_bonus=1.05,
            service_continuity_prepare_bonus=0.78,
            service_continuity_local_penalty=1.45,
            service_continuity_min_prepare_context=0.16,
        )
        row = {
            "action": 2,
            "decision_info": {
                "semantic_state": deepcopy(_minimal_semantic_state()),
                "run_metadata": {"window_class": "idle_or_sparse"},
            },
            "action_info": {
                "final_env_action": 2,
                "prepare_window_score": 0.04,
                "temporal_urgency": 0.05,
                "prediction_confidence": 0.20,
            },
            "env_info": {"metrics_protocol": {"mechanism_success_rate": 0.0}},
            "reward": -1.0,
        }

        self.assertLess(agent._counterfactual_teacher_action_credit(row, 2), 0.0)
        self.assertGreater(
            agent._counterfactual_teacher_action_credit(row, 3),
            agent._counterfactual_teacher_action_credit(row, 2),
        )

    def test_sa_v58_service_teacher_prefers_prepare_over_local_in_idle_recovery(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            service_continuity_teacher_enabled=True,
            service_continuity_current_bonus=1.28,
            service_continuity_prepare_bonus=1.16,
            service_continuity_local_penalty=1.95,
            service_continuity_min_prepare_context=0.06,
        )
        row = {
            "action": 2,
            "decision_info": {
                "semantic_state": deepcopy(_minimal_semantic_state()),
                "run_metadata": {"window_class": "idle_or_sparse"},
            },
            "action_info": {
                "final_env_action": 2,
                "prepare_window_score": 0.08,
                "temporal_urgency": 0.12,
                "prediction_confidence": 0.55,
                "predicted_handoff_target_valid": True,
                "option_gate": {
                    "window_class": "idle_or_sparse",
                    "idle_recovery_context": True,
                },
            },
            "env_info": {
                "metrics_protocol": {
                    "mechanism_success_rate": 0.0,
                    "local_exec_count": 6.0,
                    "current_rsu_exec_count": 1.0,
                }
            },
            "reward": -1.0,
        }

        self.assertLess(agent._counterfactual_teacher_action_credit(row, 2), 0.0)
        self.assertGreater(
            agent._counterfactual_teacher_action_credit(row, 4),
            agent._counterfactual_teacher_action_credit(row, 2),
        )

    def test_sa_v57_no_rsu_policy_prior_biases_service_and_suppresses_local(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_prepare_penalty=1.0,
            opportunity_constrained_prefetch_penalty=1.0,
            opportunity_constrained_no_rsu_service_bias=4.2,
            opportunity_constrained_no_rsu_local_penalty=4.8,
        )
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"]["predicted_handoff_vehicle_ids"] = []
        state["predictions"]["predicted_next_rsu_by_vehicle"] = {}
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"] = {}
        state["predictions"]["next_rsu_sequence"] = {"veh_1": []}
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_opportunity_constrained_policy(policy_output, state)
        env_bias = adjusted["env_action_logits_bias"]

        self.assertGreater(float(env_bias[3]), 0.0)
        self.assertLess(float(env_bias[2]), 0.0)

    def test_sa_v58_no_rsu_policy_prior_biases_prepare_when_target_exists(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_prepare_penalty=1.0,
            opportunity_constrained_prefetch_penalty=1.0,
            opportunity_constrained_no_rsu_service_bias=5.0,
            opportunity_constrained_no_rsu_local_penalty=5.7,
            opportunity_constrained_no_rsu_prepare_bias=3.1,
            opportunity_constrained_no_rsu_prepare_min_context=0.0,
        )
        state = deepcopy(_minimal_semantic_state())
        state["vehicles"][0]["associated_rsu_id"] = None
        state["predictions"]["predicted_handoff_vehicle_ids"] = ["veh_1"]
        state["predictions"]["predicted_next_rsu_by_vehicle"] = {"veh_1": "rsu_b"}
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"] = {"veh_1": "rsu_b"}
        state["predictions"]["prediction_confidence_by_vehicle"] = {"veh_1": 0.75}
        state["predictions"]["prediction_uncertainty_by_vehicle"] = {"veh_1": 0.25}
        state["predictions"]["next_rsu_sequence"] = {"veh_1": ["rsu_b", "rsu_b"]}
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_opportunity_constrained_policy(policy_output, state)
        env_bias = adjusted["env_action_logits_bias"]

        self.assertGreater(float(env_bias[4]), 0.0)
        self.assertLess(float(env_bias[2]), 0.0)

    def test_sa_v52_net_advantage_gate_boosts_prepare_when_net_positive(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.0,
            net_advantage_prepare_gate_min_score=0.42,
            net_advantage_prepare_gate_margin=0.10,
            net_advantage_prepare_gate_policy_coef=0.2,
            net_advantage_prepare_gate_event_coef=0.4,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_net_advantage_prepare_gate(raw, state)

        raw_margin = float((raw["event_logits"][1] - raw["event_logits"][0]).item())
        adjusted_margin = float((adjusted["event_logits"][1] - adjusted["event_logits"][0]).item())
        gate_info = adjusted["net_advantage_prepare_gate_info"]

        self.assertGreater(gate_info["net_advantage_score"], gate_info["min_score"])
        self.assertGreater(gate_info["positive_scale"], 0.0)
        self.assertGreater(adjusted_margin, raw_margin)
        self.assertGreater(float(adjusted["env_action_logits_bias"][4].item()), 0.0)

    def test_sa_v52_net_advantage_gate_suppresses_prepare_without_target(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a", "rsu_a"]},
            "predictor_name": "unit_test",
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.0,
            net_advantage_prepare_gate_min_score=0.50,
            net_advantage_prepare_gate_margin=0.12,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_net_advantage_prepare_gate(raw, state)

        raw_margin = float((raw["event_logits"][1] - raw["event_logits"][0]).item())
        adjusted_margin = float((adjusted["event_logits"][1] - adjusted["event_logits"][0]).item())
        gate_info = adjusted["net_advantage_prepare_gate_info"]

        self.assertFalse(gate_info["target_differs"])
        self.assertGreater(gate_info["negative_scale"], 0.0)
        self.assertLess(adjusted_margin, raw_margin)
        self.assertLess(float(adjusted["env_action_logits_bias"][4].item()), 0.0)

    def test_sa_v53_net_advantage_gate_boosts_service_fill_under_cache_pressure(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["workflow"]["nodes"].append(
            {"node_id": "n2", "predecessors": ["n1"], "successors": ["n3"]}
        )
        state["workflow"]["nodes"].append(
            {"node_id": "n3", "predecessors": ["n2"], "successors": []}
        )
        state["current_workflow_node"]["successors"] = ["n2", "n3"]
        state["current_workflow_node"]["input_size"] = 128.0
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
            "predictor_name": "unit_test",
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.0,
            net_advantage_prepare_gate_service_fill_scale=1.0,
            net_advantage_prepare_gate_local_penalty_scale=0.8,
            net_advantage_prepare_gate_min_score=0.50,
            net_advantage_prepare_gate_margin=0.12,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_net_advantage_prepare_gate(raw, state)

        gate_info = adjusted["net_advantage_prepare_gate_info"]

        self.assertTrue(gate_info["missing_current_adapter"])
        self.assertGreater(gate_info["service_fill_bias"], 0.0)
        self.assertGreater(float(adjusted["env_action_logits_bias"][0].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][2].item()), 0.0)
        self.assertGreater(float(adjusted["slow_logits"][1].item()), float(raw["slow_logits"][1].item()))

    def test_sa_v53_net_advantage_gate_boosts_current_execution_after_cache_warm(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["workflow"]["nodes"].append(
            {"node_id": "n2", "predecessors": ["n1"], "successors": ["n3"]}
        )
        state["workflow"]["nodes"].append(
            {"node_id": "n3", "predecessors": ["n2"], "successors": []}
        )
        state["current_workflow_node"]["successors"] = ["n2", "n3"]
        state["current_workflow_node"]["input_size"] = 128.0
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
            "predictor_name": "unit_test",
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            net_advantage_prepare_gate_enabled=True,
            net_advantage_prepare_gate_bias=2.0,
            net_advantage_prepare_gate_service_fill_scale=1.0,
            net_advantage_prepare_gate_local_penalty_scale=0.8,
            net_advantage_prepare_gate_min_score=0.50,
            net_advantage_prepare_gate_margin=0.12,
        )

        with torch.no_grad():
            raw = agent._network.forward_single(state)
            adjusted = agent._apply_net_advantage_prepare_gate(raw, state)

        gate_info = adjusted["net_advantage_prepare_gate_info"]

        self.assertTrue(gate_info["current_cache_ready"])
        self.assertGreater(gate_info["service_fill_bias"], 0.0)
        self.assertGreater(float(adjusted["env_action_logits_bias"][3].item()), 0.0)
        self.assertLess(float(adjusted["env_action_logits_bias"][2].item()), 0.0)
        self.assertGreater(float(adjusted["fast_logits"][0].item()), float(raw["fast_logits"][0].item()))

    def test_sa_v47_backhaul_aware_policy_boosts_service_fill_without_signal(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            backhaul_aware_policy_enabled=True,
            backhaul_aware_service_fill_bias=3.0,
            backhaul_aware_no_signal_prefetch_penalty=1.2,
            backhaul_aware_no_signal_prepare_penalty=1.6,
            backhaul_aware_service_pressure_floor=0.20,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_backhaul_aware_policy(policy_output, state)

        info = adjusted["backhaul_aware_policy_info"]
        self.assertTrue(info["enabled"])
        self.assertTrue(info["missing_current_adapter"])
        self.assertTrue(info["no_trusted_signal"])
        self.assertGreater(info["service_fill_bias"], 0.0)
        self.assertGreater(adjusted["slow_logits"][1].item(), policy_output["slow_logits"][1].item())
        self.assertLess(adjusted["slow_logits"][2].item(), policy_output["slow_logits"][2].item())
        self.assertLess(adjusted["event_logits"][1].item(), policy_output["event_logits"][1].item())

    def test_sa_v47_backhaul_aware_policy_suppresses_redundant_fill_when_ready(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            backhaul_aware_policy_enabled=True,
            backhaul_aware_redundant_fill_penalty=2.5,
            backhaul_aware_steady_bias=1.0,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_backhaul_aware_policy(policy_output, state)

        info = adjusted["backhaul_aware_policy_info"]
        self.assertTrue(info["current_cache_ready"])
        self.assertGreater(info["redundant_fill_penalty"], 0.0)
        self.assertGreater(info["steady_bias"], 0.0)
        self.assertLess(adjusted["slow_logits"][1].item(), policy_output["slow_logits"][1].item())

    def test_sa_v48_backhaul_aware_policy_preserves_prepare_logits_without_signal_penalty(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a"]},
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            backhaul_aware_policy_enabled=True,
            backhaul_aware_service_fill_bias=1.35,
            backhaul_aware_no_signal_prefetch_penalty=0.0,
            backhaul_aware_no_signal_prepare_penalty=0.0,
            backhaul_aware_service_pressure_floor=0.20,
        )
        policy_output = {
            "slow_logits": torch.zeros(3),
            "fast_logits": torch.zeros(2),
            "event_logits": torch.zeros(2),
        }

        adjusted = agent._apply_backhaul_aware_policy(policy_output, state)

        info = adjusted["backhaul_aware_policy_info"]
        self.assertTrue(info["enabled"])
        self.assertTrue(info["missing_current_adapter"])
        self.assertTrue(info["no_trusted_signal"])
        self.assertGreater(info["service_fill_bias"], 0.0)
        self.assertEqual(info["prepare_penalty"], 0.0)
        self.assertEqual(info["prefetch_penalty"], 0.0)
        self.assertGreater(adjusted["slow_logits"][1].item(), policy_output["slow_logits"][1].item())
        self.assertEqual(adjusted["slow_logits"][2].item(), policy_output["slow_logits"][2].item())
        self.assertEqual(adjusted["event_logits"][1].item(), policy_output["event_logits"][1].item())

    def test_sa_v49_retrospective_aux_guides_event_when_predictor_misses_handoff(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["predictions"]["predicted_handoff_vehicle_ids"] = []
        state["predictions"]["predicted_next_rsu_by_vehicle"]["veh_1"] = "rsu_a"
        state["predictions"]["predicted_first_handoff_rsu_by_vehicle"]["veh_1"] = None
        state["predictions"]["next_rsu_sequence"]["veh_1"] = ["rsu_a", "rsu_a", "rsu_a"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            mechanism_aux_coef=0.1,
            mechanism_aux_current_cache_fill_enabled=False,
            retrospective_handoff_aux_enabled=True,
            retrospective_handoff_aux_max_eta=6.0,
            retrospective_handoff_aux_min_score=0.05,
            retrospective_handoff_aux_prepare_weight=0.72,
            retrospective_handoff_aux_transition_weight=1.6,
        )

        annotation = agent._build_mechanism_guidance_annotation(
            state,
            {
                "decision_info": {
                    "retrospective_handoff_label": {
                        "gt_handoff_opportunity": 1.0,
                        "gt_first_handoff_steps": 2.5,
                        "gt_first_next_rsu": "rsu_b",
                        "current_rsu_id": "rsu_a",
                    }
                },
                "action_info": {
                    "prediction_state_available": True,
                    "raw_handoff_candidate": False,
                    "predicted_handoff_target_valid": False,
                    "next_rsu_non_null_count": 0,
                    "gate_pass": False,
                    "prepare_window_score": 0.0,
                    "temporal_urgency": 0.0,
                    "prediction_confidence": 0.1,
                },
            },
        )

        self.assertTrue(annotation["apply"])
        self.assertTrue(annotation["event_guidance"])
        self.assertFalse(annotation["predicted_event_guidance"])
        self.assertTrue(annotation["retrospective_event_guidance"])
        self.assertFalse(annotation["valid_handoff_target"])
        self.assertTrue(annotation["prepare_aux_legal"])
        self.assertGreater(annotation["event_weight"], 0.0)
        self.assertGreater(annotation["transition_weight"], 1.0)

    def test_sa_v44_opportunity_constraint_suppresses_prepare_without_candidate(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["predictions"] = {
            "future_load": {"rsu_a": 1.0, "rsu_b": 1.0},
            "predicted_handoff_vehicle_ids": [],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_a"},
            "predicted_first_handoff_rsu_by_vehicle": {},
            "prediction_confidence_by_vehicle": {"veh_1": 0.05},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.95},
            "dwell_time": {"veh_1": 12.0},
            "next_rsu_sequence": {"veh_1": ["rsu_a", "rsu_a", "rsu_a"]},
            "predictor_name": "unit_test",
        }
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_prepare_penalty=12.0,
            opportunity_constrained_prefetch_penalty=10.0,
            opportunity_constrained_current_bias=4.0,
            opportunity_constrained_min_context=0.54,
            opportunity_constrained_low_context=0.32,
        )

        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
                "deterministic_policy": True,
                "run_metadata": {"window_class": "active_non_mechanism"},
            },
        )

        self.assertNotIn(action, {1, 4})
        constraint_info = action_info["opportunity_constrained_policy"]
        self.assertTrue(constraint_info["enabled"])
        self.assertFalse(constraint_info["weak_opportunity"])
        self.assertGreater(constraint_info["prepare_penalty"], 0.0)
        self.assertGreater(constraint_info["prefetch_penalty"], 0.0)

    def test_sa_v44_opportunity_constraint_requires_trusted_candidate(self) -> None:
        state = deepcopy(_minimal_semantic_state())
        state["predictions"]["prediction_confidence_by_vehicle"] = {"veh_1": 0.10}
        state["predictions"]["prediction_uncertainty_by_vehicle"] = {"veh_1": 0.90}
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            opportunity_constrained_policy_enabled=True,
            opportunity_constrained_prepare_penalty=12.0,
            opportunity_constrained_prefetch_penalty=10.0,
            opportunity_constrained_current_bias=4.0,
            opportunity_constrained_min_context=0.54,
            opportunity_constrained_low_context=0.32,
            opportunity_constrained_reliability_floor=0.40,
        )

        action, action_info = agent.act(
            None,
            {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
                "deterministic_policy": True,
                "run_metadata": {"window_class": "mechanism_activating"},
            },
        )

        self.assertNotIn(action, {1, 4})
        constraint_info = action_info["opportunity_constrained_policy"]
        self.assertFalse(constraint_info["trusted_candidate"])
        self.assertFalse(constraint_info["weak_opportunity"])
        self.assertGreater(constraint_info["prepare_penalty"], 0.0)

    def test_sa_v43_strict_opportunity_does_not_treat_window_class_as_credit(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            delayed_mechanism_credit_enabled=True,
            delayed_mechanism_credit_context_gate=0.46,
            delayed_mechanism_credit_strict_opportunity_enabled=True,
        )
        row = {
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {
                "metrics_protocol": {
                    "handoff_event_count": 0,
                    "predicted_handoff_signal": False,
                    "has_predicted_handoff_target": False,
                }
            },
            "action_info": {
                "prepare_window_score": 0.0,
                "temporal_urgency": 0.0,
                "prediction_confidence": 0.0,
                "gate_pass": False,
                "raw_handoff_candidate": False,
                "predicted_handoff_target_valid": False,
            },
        }

        self.assertFalse(agent._row_mechanism_credit_opportunity(row))

    def test_sa_v36_counterfactual_margin_loss_prefers_local_on_idle_window(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            env_action_counterfactual_margin_enabled=True,
            env_action_counterfactual_margin_coef=0.22,
            counterfactual_teacher_local_bonus=0.9,
            counterfactual_teacher_current_rsu_penalty=0.03,
            counterfactual_teacher_invalid_mechanism_penalty=0.66,
        )
        row = {
            "action": 3,
            "decision_info": {
                "semantic_state": state,
                "run_metadata": {"window_class": "idle_or_sparse"},
            },
            "action_info": {
                "final_env_action": 3,
                "prepare_window_score": 0.05,
                "temporal_urgency": 0.05,
                "prediction_confidence": 0.2,
            },
        }
        policy_output = agent._forward_policy(state, run_metadata={"window_class": "idle_or_sparse"})
        action_mask = [True, True, True, True, True]

        loss = agent._compute_env_action_counterfactual_margin_loss(
            batch_outputs=[policy_output],
            batch_action_masks=[action_mask],
            batch_rows=[row],
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(
            agent._counterfactual_teacher_action_credit(row, 2),
            agent._counterfactual_teacher_action_credit(row, 3),
        )

    def test_sa_v37_counterfactual_margin_loss_skips_negative_advantage(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            counterfactual_teacher_prd_enabled=True,
            env_action_counterfactual_margin_enabled=True,
            env_action_counterfactual_margin_coef=0.055,
            env_action_counterfactual_margin_advantage_gate=0.12,
            env_action_counterfactual_margin_advantage_blend=0.70,
            counterfactual_teacher_local_bonus=0.9,
            counterfactual_teacher_current_rsu_penalty=0.03,
            counterfactual_teacher_invalid_mechanism_penalty=0.66,
        )
        row = {
            "action": 3,
            "decision_info": {
                "semantic_state": state,
                "run_metadata": {"window_class": "idle_or_sparse"},
            },
            "action_info": {
                "final_env_action": 3,
                "prepare_window_score": 0.05,
                "temporal_urgency": 0.05,
                "prediction_confidence": 0.2,
            },
        }
        policy_output = agent._forward_policy(state, run_metadata={"window_class": "idle_or_sparse"})
        action_mask = [True, True, True, True, True]

        loss = agent._compute_env_action_counterfactual_margin_loss(
            batch_outputs=[policy_output],
            batch_action_masks=[action_mask],
            batch_rows=[row],
            base_advantage=torch.tensor([-0.5]),
            event_advantage=torch.tensor([-0.2]),
        )

        self.assertEqual(float(loss.item()), 0.0)

    def test_sa_v39_delayed_credit_propagates_future_ready_to_prepare(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            delayed_mechanism_credit_enabled=True,
            delayed_mechanism_credit_horizon=3,
            delayed_mechanism_credit_decay=0.7,
            delayed_mechanism_credit_ready_bonus=1.2,
            delayed_mechanism_credit_success_bonus=0.8,
            delayed_mechanism_credit_failure_penalty=0.9,
            delayed_mechanism_credit_stale_penalty=0.0,
        )
        rollout = [
            {
                "action": 4,
                "terminated": False,
                "truncated": False,
                "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
                "action_info": {
                    "final_env_action": 4,
                    "head_actions": {"event": 1, "slow": 0, "fast": 0},
                    "prepare_window_score": 0.8,
                    "temporal_urgency": 0.7,
                    "prediction_confidence": 0.8,
                    "gate_pass": True,
                },
                "env_info": {"metrics_protocol": {"predicted_handoff_signal": True}},
            },
            {
                "action": 3,
                "terminated": False,
                "truncated": False,
                "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
                "action_info": {
                    "final_env_action": 3,
                    "head_actions": {"event": 0, "slow": 0, "fast": 1},
                    "prepare_window_score": 0.4,
                    "temporal_urgency": 0.6,
                    "prediction_confidence": 0.8,
                },
                "env_info": {"metrics_protocol": {"predicted_handoff_signal": True}},
            },
            {
                "action": 3,
                "terminated": True,
                "truncated": False,
                "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
                "action_info": {
                    "final_env_action": 3,
                    "head_actions": {"event": 0, "slow": 0, "fast": 1},
                },
                "env_info": {
                    "metrics_protocol": {
                        "handoff_ready": True,
                        "mechanism_success_strict": True,
                        "handoff_event_count": 1,
                    }
                },
            },
        ]

        credits = agent._delayed_mechanism_credit_values(rollout)

        self.assertGreater(float(credits[0]), 0.0)
        self.assertGreater(float(credits[0]), float(credits[1]))

    def test_sa_v40_advantage_weighted_behavior_keeps_positive_deviation(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            advantage_weighted_behavior_regularization_enabled=True,
            advantage_weighted_behavior_coef=0.24,
            advantage_weighted_behavior_positive_gate=0.08,
            advantage_weighted_behavior_negative_gate=0.04,
        )
        row = {
            "action": 4,
            "decision_info": {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
                "run_metadata": {"window_class": "mechanism_activating"},
            },
            "action_info": {"final_env_action": 4},
            "env_info": {"metrics_protocol": {"predicted_handoff_signal": True}},
        }

        stats = agent._annotate_advantage_weighted_behavior_targets(
            [row],
            advantage_values=torch.tensor([0.6]).numpy(),
        )
        policy_output = agent._forward_policy(state, run_metadata={"window_class": "mechanism_activating"})
        loss = agent._compute_advantage_weighted_behavior_loss(
            batch_outputs=[policy_output],
            batch_action_masks=[[True, True, True, True, True]],
            batch_rows=[row],
        )

        self.assertEqual(stats["positive_count"], 1)
        self.assertEqual(row["advantage_weighted_behavior_mode"], "positive_deviation")
        self.assertEqual(row["advantage_weighted_behavior_target_action"], 4)
        self.assertGreater(float(loss.item()), 0.0)

    def test_sa_v40_advantage_weighted_behavior_recovers_negative_deviation(self) -> None:
        state = _minimal_semantic_state()
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            advantage_weighted_behavior_regularization_enabled=True,
            advantage_weighted_behavior_coef=0.24,
            advantage_weighted_behavior_positive_gate=0.08,
            advantage_weighted_behavior_negative_gate=0.04,
        )
        row = {
            "action": 4,
            "decision_info": {
                "semantic_state": state,
                "action_mask": [True, True, True, True, True],
                "run_metadata": {"window_class": "mechanism_activating"},
            },
            "action_info": {"final_env_action": 4},
            "env_info": {"metrics_protocol": {"predicted_handoff_signal": True}},
        }

        stats = agent._annotate_advantage_weighted_behavior_targets(
            [row],
            advantage_values=torch.tensor([-0.6]).numpy(),
        )

        self.assertEqual(stats["negative_count"], 1)
        self.assertEqual(row["advantage_weighted_behavior_mode"], "negative_recovery")
        self.assertEqual(row["advantage_weighted_behavior_target_action"], 0)

    def test_sa_v34_adaptive_wait_prior_prefers_local_when_target_ready(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            digital_twin_policy_prior_enabled=True,
            digital_twin_policy_prior_logit_bias=5.0,
            digital_twin_policy_prior_env_action_bias_enabled=True,
            digital_twin_policy_prior_env_action_logit_bias=5.0,
            digital_twin_policy_prior_continuation_threshold=0.0,
            digital_twin_policy_prior_continuation_prepare_scale=0.8,
            digital_twin_policy_prior_continuation_wait_scale=1.5,
            digital_twin_policy_prior_adaptive_wait_enabled=True,
            digital_twin_policy_prior_wait_ready_threshold=0.5,
            digital_twin_policy_prior_wait_timing_ceiling=1.0,
            digital_twin_policy_prior_wait_cache_ready_scale=1.4,
            counterfactual_teacher_prd_enabled=True,
            counterfactual_teacher_local_bonus=0.6,
            counterfactual_teacher_invalid_mechanism_penalty=0.7,
        )
        run_metadata = {"window_class": "mechanism_activating"}

        annotation = agent._build_digital_twin_policy_prior_annotation(
            state,
            run_metadata=run_metadata,
        )
        policy_output = agent._forward_policy(state, run_metadata=run_metadata)
        scores = agent._hierarchical_env_action_scores(policy_output)
        wait_bias = policy_output["env_action_logits_bias"]
        row = {
            "decision_info": {
                "semantic_state": state,
                "run_metadata": run_metadata,
            },
            "action_info": {
                "prepare_window_score": 0.05,
                "temporal_urgency": 0.05,
                "prediction_confidence": 0.8,
                "gate_pass": True,
                "predicted_handoff_target_valid": True,
            },
        }
        local_credit = agent._counterfactual_teacher_action_credit(row, 2)
        prepare_credit = agent._counterfactual_teacher_action_credit(row, 4)

        self.assertTrue(annotation["apply"])
        self.assertTrue(annotation["adaptive_wait_preferred"])
        self.assertTrue(annotation["continuation_wait_target"])
        self.assertEqual(annotation["env_target"], 2)
        self.assertGreater(float(wait_bias[2].item()), float(wait_bias[4].item()))
        self.assertGreater(float(scores[2].item()), float(scores[4].item()))
        self.assertGreater(local_credit, 0.0)
        self.assertGreater(local_credit, prepare_credit)

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

    def test_sa_v22_net_utility_penalizes_unvalidated_mechanism_attempt(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            event_prd_advantage_enabled=True,
            option_gate_prd_enabled=True,
            net_utility_prd_enabled=True,
            net_utility_backhaul_coef=0.22,
            net_utility_migration_coef=0.24,
            net_utility_failed_mechanism_penalty=0.84,
            net_utility_failed_mechanism_backhaul_coef=0.36,
            net_utility_success_bonus=0.28,
            net_utility_backhaul_normalizer=64.0,
        )
        failed_row = {
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
                    "backhaul_traffic_cost": 64.0,
                    "adapter_state_migration_overhead": 0.0,
                    "mechanism_attempt_selected": True,
                    "migration_prepare_requested": True,
                    "mechanism_success_strict": False,
                    "handoff_ready": False,
                }
            },
        }
        success_row = deepcopy(failed_row)
        success_row["env_info"]["metrics_protocol"].update(
            {
                "backhaul_traffic_cost": 0.0,
                "mechanism_success_strict": True,
                "handoff_ready": True,
            }
        )

        self.assertLess(agent._event_partial_reward_credit(failed_row), 0.0)
        self.assertLess(agent._option_gate_partial_reward_credit(failed_row), 0.0)
        self.assertGreater(agent._event_partial_reward_credit(success_row), 0.0)
        self.assertGreater(agent._option_gate_partial_reward_credit(success_row), 0.0)

    def test_sa_v23_counterfactual_teacher_rewards_mechanism_and_penalizes_invalid_prepare(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            event_prd_advantage_enabled=True,
            option_gate_enabled=True,
            option_gate_count=4,
            option_gate_prd_enabled=True,
            option_gate_counterfactual_prd_enabled=True,
            option_gate_counterfactual_coef=0.46,
            net_utility_prd_enabled=True,
            counterfactual_teacher_prd_enabled=True,
            counterfactual_teacher_event_coef=0.58,
            counterfactual_teacher_option_coef=0.72,
            counterfactual_teacher_mechanism_bonus=0.92,
            counterfactual_teacher_missed_prepare_penalty=0.78,
            counterfactual_teacher_local_bonus=0.42,
            counterfactual_teacher_current_rsu_penalty=0.16,
            counterfactual_teacher_invalid_mechanism_penalty=0.82,
        )
        mechanism_row = {
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
                    "option_action": 3,
                    "option_label": "mechanism_prepare",
                    "option_actions": {"0": 3, "1": 3, "2": 2, "3": 4},
                    "option_mask": [True, True, True, True],
                    "window_class": "mechanism_activating",
                },
            },
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {"metrics_protocol": {"mechanism_success_strict": False}},
        }
        missed_prepare_row = deepcopy(mechanism_row)
        missed_prepare_row["action"] = 3
        missed_prepare_row["action_info"]["head_actions"] = {"event": 0}
        missed_prepare_row["action_info"]["final_env_action"] = 3

        invalid_prepare_row = deepcopy(mechanism_row)
        invalid_prepare_row["decision_info"]["run_metadata"]["window_class"] = "active_non_mechanism"
        invalid_prepare_row["action_info"]["option_gate"]["window_class"] = "active_non_mechanism"

        local_row = deepcopy(invalid_prepare_row)
        local_row["action"] = 2
        local_row["action_info"]["head_actions"] = {"event": 0}
        local_row["action_info"]["final_env_action"] = 2

        mechanism_credit = agent._event_partial_reward_credit(mechanism_row)
        missed_credit = agent._event_partial_reward_credit(missed_prepare_row)
        invalid_credit = agent._event_partial_reward_credit(invalid_prepare_row)
        local_credit = agent._event_partial_reward_credit(local_row)
        self.assertGreater(mechanism_credit, missed_credit)
        self.assertGreater(local_credit, invalid_credit)

        option_advantage = agent._option_gate_advantage(
            row=mechanism_row,
            base_advantage=torch.tensor(0.0),
            option_probs=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            option_mask=[True, True, True, True],
        )
        invalid_option_advantage = agent._option_gate_advantage(
            row=invalid_prepare_row,
            base_advantage=torch.tensor(0.0),
            option_probs=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            option_mask=[True, True, True, True],
        )
        self.assertGreater(float(option_advantage.item()), 0.0)
        self.assertLess(float(invalid_option_advantage.item()), 0.0)

    def test_sa_v24_tail_risk_credit_penalizes_failed_redundant_mechanism(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            tail_risk_prd_enabled=True,
            tail_risk_policy_coef=0.52,
            tail_risk_event_coef=0.36,
            tail_risk_option_coef=0.46,
            tail_risk_reward_shortfall_coef=0.72,
            tail_risk_service_coef=0.80,
            tail_risk_continuity_coef=1.10,
            tail_risk_handoff_failure_coef=1.35,
            tail_risk_failed_mechanism_coef=0.82,
            tail_risk_redundant_mechanism_coef=0.70,
            tail_risk_success_credit=0.18,
        )
        failed_row = {
            "action": 4,
            "reward": -1.2,
            "action_info": {
                "head_actions": {"event": 1},
                "final_env_action": 4,
                "option_gate": {
                    "enabled": True,
                    "option_action": 3,
                    "option_label": "mechanism_prepare",
                    "option_actions": {"0": 3, "1": 3, "2": 2, "3": 4},
                    "option_mask": [True, True, True, True],
                    "window_class": "mechanism_activating",
                },
            },
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {
                "metrics_protocol": {
                    "handoff_failed": True,
                    "handoff_failure_rate": 1.0,
                    "workflow_continuity_rate": 0.25,
                    "service_delay_sum": 4.0,
                    "cache_miss_penalty_sum": 2.4,
                    "migration_prepare_requested": True,
                    "mechanism_success_strict": False,
                }
            },
        }
        success_row = deepcopy(failed_row)
        success_row["reward"] = 1.0
        success_row["env_info"]["metrics_protocol"].update(
            {
                "handoff_failed": False,
                "handoff_failure_rate": 0.0,
                "workflow_continuity_rate": 1.0,
                "service_delay_sum": 0.0,
                "cache_miss_penalty_sum": 0.0,
                "mechanism_success_strict": True,
                "handoff_ready": True,
            }
        )

        failed_credit = agent._tail_risk_prd_credit(failed_row, reward_floor=0.0)
        success_credit = agent._tail_risk_prd_credit(success_row, reward_floor=0.0)
        self.assertLess(failed_credit, -1.0)
        self.assertGreater(success_credit, failed_credit)

        option_advantage = agent._option_gate_advantage(
            row=failed_row,
            base_advantage=torch.tensor(0.0),
            option_probs=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            option_mask=[True, True, True, True],
        )
        self.assertLess(float(option_advantage.item()), 0.0)

    def test_sa_v25_opportunity_credit_rewards_efficient_service_and_penalizes_waste(self) -> None:
        agent = build_agent(
            "sa_ghmappo",
            random_seed=7,
            deterministic_action=True,
            option_gate_enabled=True,
            option_gate_count=4,
            opportunity_prd_enabled=True,
            opportunity_policy_coef=0.42,
            opportunity_event_coef=0.24,
            opportunity_option_coef=0.36,
            opportunity_reward_surplus_coef=0.72,
            opportunity_service_success_coef=0.54,
            opportunity_cache_hit_coef=0.36,
            opportunity_continuity_coef=0.42,
            opportunity_current_rsu_efficiency_coef=0.36,
            opportunity_local_fallback_coef=0.30,
            opportunity_backhaul_penalty_coef=0.34,
            opportunity_delay_penalty_coef=0.24,
            opportunity_failed_service_penalty_coef=0.44,
            opportunity_mechanism_success_bonus=0.32,
        )
        efficient_row = {
            "action": 0,
            "reward": 2.4,
            "action_info": {"final_env_action": 0, "head_actions": {"event": 0}},
            "decision_info": {"run_metadata": {"window_class": "mechanism_activating"}},
            "env_info": {
                "metrics_protocol": {
                    "service_success_count": 8,
                    "workflow_completed_count": 1,
                    "workflow_unfinished_count": 0,
                    "service_delay_sum": 3.2,
                    "service_wait_sum": 0,
                    "adapter_hit_count": 8,
                    "adapter_miss_count": 0,
                    "adapter_warm_hit_count": 8,
                    "workflow_continuity_rate": 1.0,
                    "current_rsu_exec_count": 8,
                    "local_exec_count": 0,
                    "backhaul_traffic_cost": 64.0,
                    "cache_admission_count": 1,
                    "mechanism_success_strict": True,
                    "handoff_failed": False,
                }
            },
        }
        waste_row = deepcopy(efficient_row)
        waste_row["action"] = 4
        waste_row["reward"] = -1.6
        waste_row["action_info"] = {"final_env_action": 4, "head_actions": {"event": 1}}
        waste_row["env_info"]["metrics_protocol"].update(
            {
                "service_success_count": 2,
                "workflow_completed_count": 0,
                "workflow_unfinished_count": 1,
                "service_delay_sum": 10.0,
                "service_wait_sum": 4,
                "adapter_hit_count": 2,
                "adapter_miss_count": 6,
                "adapter_warm_hit_count": 2,
                "workflow_continuity_rate": 0.375,
                "current_rsu_exec_count": 8,
                "backhaul_traffic_cost": 192.0,
                "cache_admission_count": 3,
                "migration_prepare_requested": True,
                "mechanism_success_strict": False,
            }
        )

        efficient_credit = agent._opportunity_prd_credit(efficient_row, reward_floor=0.0)
        waste_credit = agent._opportunity_prd_credit(waste_row, reward_floor=0.0)
        self.assertGreater(efficient_credit, 0.5)
        self.assertLess(waste_credit, 0.0)

        option_advantage = agent._option_gate_advantage(
            row=efficient_row,
            base_advantage=torch.tensor(0.0),
            option_probs=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            option_mask=[True, True, True, True],
        )
        self.assertGreater(float(option_advantage.item()), 0.0)

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

    def test_top_journal_mechanism_aux_guides_cached_handoff_migration_prepare(self) -> None:
        state = _minimal_semantic_state()
        state["rsus"][0]["cached_adapter_ids"] = ["adapter_tracking"]
        state["rsus"][1]["cached_adapter_ids"] = ["adapter_tracking"]
        agent = build_agent(
            "sa_ghmappo",
            random_seed=1,
            mechanism_aux_coef=0.1,
            mechanism_aux_current_cache_fill_enabled=False,
        )

        annotation = agent._build_mechanism_guidance_annotation(
            state,
            {
                "action_info": {
                    "prediction_state_available": True,
                    "raw_handoff_candidate": True,
                    "predicted_handoff_target_valid": True,
                    "next_rsu_non_null_count": 1,
                    "gate_pass": True,
                    "prepare_window_score": 0.8,
                    "temporal_urgency": 0.7,
                    "prediction_confidence": 0.9,
                }
            },
        )

        self.assertTrue(annotation["apply"])
        self.assertTrue(annotation["event_guidance"])
        self.assertFalse(annotation["prefetch_guidance"])
        self.assertTrue(annotation["cache_ready"])

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

    def test_action_adapter_decodes_handoff_prepare_as_target_prefetch(self) -> None:
        state = {
            "current_workflow_node": {"required_adapter": "adapter_tracking"},
            "vehicles": [{"vehicle_id": "veh_1", "associated_rsu_id": "rsu_a"}],
            "predictions": {
                "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
                "predicted_handoff_target_rsu_id_by_vehicle": {"veh_1": "rsu_b"},
            },
        }

        control = ActionAdapter().decode(4, state)

        self.assertEqual(control.cache_action["strategy"], "handoff_prepare_prefetch")
        self.assertEqual(control.cache_action["rsu_id"], "rsu_b")
        self.assertEqual(control.migration_action["mode"], "prepare")
        self.assertEqual(control.migration_action["expected_target_rsu_id"], "rsu_b")
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
