"""Algorithm-pool registry and action-schema contract tests."""

from __future__ import annotations

import sys
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
