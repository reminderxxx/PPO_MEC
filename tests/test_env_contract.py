"""核心环境最小接口合同测试。"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv, make_toy_vec_env
from src.envs.specs import ControlAction, RSUState
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.predictors import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    SupervisedHandoffPredictorNetwork,
)


def 构造控制动作(state: dict) -> ControlAction:
    """基于当前状态构造最小动作。"""
    current_node = state.get("current_workflow_node") or {}
    return ControlAction(
        cache_action={
            "operation": "cache",
            "adapter_id": current_node.get("required_adapter"),
        }
        if current_node
        else {},
        offload_action={"mode": "hybrid"},
        migration_action={"mode": "migrate"},
    )


class EnvContractTestCase(unittest.TestCase):
    """核心环境合同测试。"""

    def setUp(self) -> None:
        self.env = make_toy_vec_env()

    def test_reset_returns_dict(self) -> None:
        state, info = self.env.reset()
        self.assertIsInstance(state, dict)
        self.assertIn("workflow", state)
        self.assertIn("handoff_events", state)
        self.assertIn("handoff_events", info)

    def test_step_contract(self) -> None:
        state, _ = self.env.reset()
        control = 构造控制动作(state)
        next_state, reward, terminated, truncated, info = self.env.step(control)
        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        for field_name in [
            "total",
            "service_reward",
            "delay_penalty",
            "cache_miss_penalty",
            "migration_cost",
            "continuity_bonus",
            "mechanism_exploration_bonus",
            "constraint_penalty",
        ]:
            self.assertTrue(hasattr(reward, field_name))
        self.assertIn("handoff_events", info)
        self.assertIn("cache_hit", info)
        self.assertIn("stall_occurred", info)
        metrics_protocol = info["metrics_protocol"]
        self.assertIn("action_invalid", metrics_protocol)
        self.assertIn("mechanism_success_strict", metrics_protocol)
        self.assertIn("dag_frontier_size", metrics_protocol)
        self.assertIn("dag_critical_path_pressure", metrics_protocol)

    def test_predictor_manager_exposes_kind_and_quality_audit(self) -> None:
        env = VecWorkflowCoreEnv(
            predictor_manager=PredictorManager(predictor_kind="learned_or_calibrated")
        )

        state, _ = env.reset()

        predictions = state["predictions"]
        self.assertEqual(predictions["predictor_kind"], "learned_or_calibrated")
        self.assertEqual(predictions["predictor_name"], "calibrated_baseline_surrogate_v1")
        self.assertFalse(predictions["learned_predictor_attached"])
        self.assertIn("prediction_quality_audit", predictions)
        self.assertIn("brier_score_proxy", predictions["prediction_quality_audit"])

    def test_predictor_manager_audits_oracle_fallback(self) -> None:
        env = VecWorkflowCoreEnv(
            predictor_manager=PredictorManager(
                predictor_kind="oracle",
                oracle_prediction_enabled=True,
            )
        )

        state, _ = env.reset()

        predictions = state["predictions"]
        self.assertEqual(predictions["requested_predictor_kind"], "oracle")
        self.assertTrue(predictions["oracle_requested"])
        self.assertFalse(predictions["oracle_available"])
        self.assertTrue(predictions["oracle_fallback_to_baseline"])
        self.assertEqual(predictions["predictor_kind"], "baseline")

    def test_supervised_predictor_checkpoint_fills_prediction_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "predictor.pt"
            rsu_ids = ["rsu_a", "rsu_b", "rsu_c"]
            input_dim = 10 + 7 * len(rsu_ids)
            network = SupervisedHandoffPredictorNetwork(
                input_dim=input_dim,
                rsu_class_count=len(rsu_ids) + 1,
                hidden_dim=8,
            )
            torch.save(
                {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "horizon": 3,
                    "input_dim": input_dim,
                    "hidden_dim": 8,
                    "feature_schema": {
                        "schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_names": [f"feature_{index}" for index in range(input_dim)],
                    },
                    "rsu_label_map": {"rsu_ids": rsu_ids, "none_index": len(rsu_ids)},
                    "model_state_dict": network.state_dict(),
                    "metrics": {},
                },
                checkpoint_path,
            )
            env = VecWorkflowCoreEnv(
                predictor_manager=PredictorManager(
                    predictor_kind="supervised",
                    predictor_checkpoint_path=str(checkpoint_path),
                )
            )

            state, _ = env.reset()

        predictions = state["predictions"]
        self.assertEqual(predictions["predictor_kind"], "supervised")
        self.assertEqual(predictions["predictor_name"], "supervised_handoff_predictor_v1")
        self.assertTrue(predictions["learned_predictor_attached"])
        self.assertIn("next_rsu_sequence", predictions)
        self.assertIn("predicted_handoff_eta_steps_by_vehicle", predictions)
        self.assertIn("supervised_predictor_checkpoint", predictions)

    def test_supervised_predictor_requires_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            PredictorManager(predictor_kind="supervised")

    def test_supervised_predictor_rejects_rsu_map_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "predictor.pt"
            rsu_ids = ["rsu_a"]
            input_dim = 10 + 7 * len(rsu_ids)
            network = SupervisedHandoffPredictorNetwork(
                input_dim=input_dim,
                rsu_class_count=len(rsu_ids) + 1,
                hidden_dim=8,
            )
            torch.save(
                {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "horizon": 3,
                    "input_dim": input_dim,
                    "hidden_dim": 8,
                    "feature_schema": {
                        "schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_names": [f"feature_{index}" for index in range(input_dim)],
                    },
                    "rsu_label_map": {"rsu_ids": rsu_ids, "none_index": len(rsu_ids)},
                    "model_state_dict": network.state_dict(),
                    "metrics": {},
                },
                checkpoint_path,
            )
            env = VecWorkflowCoreEnv(
                predictor_manager=PredictorManager(
                    predictor_kind="supervised",
                    predictor_checkpoint_path=str(checkpoint_path),
                )
            )

            with self.assertRaises(ValueError):
                env.reset()

    def test_predictor_manager_trims_prediction_history(self) -> None:
        predictor_manager = PredictorManager(prediction_delay_steps=2)
        env = VecWorkflowCoreEnv(predictor_manager=predictor_manager, max_steps=5)

        state, _ = env.reset()
        for _ in range(4):
            state, _, terminated, _, _ = env.step(构造控制动作(state))
            if terminated:
                break

        self.assertLessEqual(len(predictor_manager._prediction_history), 3)

    def test_gym_observation_uses_primary_vehicle_cache_context(self) -> None:
        env = GymVecEnv(core_env=make_toy_vec_env())
        state = {
            "time_index": 0,
            "workflow": {"completed_node_ids": [], "execution_order": ["n1"]},
            "vehicles": [
                {"vehicle_id": "veh_a", "associated_rsu_id": "rsu_a"},
                {"vehicle_id": "veh_b", "associated_rsu_id": "rsu_b"},
            ],
            "primary_vehicle_id": "veh_b",
            "rsus": [
                {"rsu_id": "rsu_a", "cached_adapter_ids": ["adapter_a"]},
                {
                    "rsu_id": "rsu_b",
                    "cached_adapter_ids": ["adapter_b1", "adapter_b2", "adapter_b3"],
                },
            ],
            "predictions": {
                "predicted_handoff_vehicle_ids": [],
                "future_load": {},
            },
            "current_workflow_node": {"node_id": "n1"},
            "handoff_events": [],
        }

        env._normalizer.reset(state)
        observation = env._encode_observation(state)

        self.assertAlmostEqual(observation[7], math.tanh(3.0 / 8.0), places=6)

    def test_handoff_pressure_primary_vehicle_selection(self) -> None:
        frames = [
            {
                "time_index": 0,
                "vehicles": [
                    {
                        "vehicle_id": "veh_a",
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "speed": 1.0,
                        "base_model_id": "veh_base_v1",
                    },
                    {
                        "vehicle_id": "veh_b",
                        "position_x": 2.0,
                        "position_y": 0.0,
                        "speed": 1.0,
                        "base_model_id": "veh_base_v1",
                    },
                ],
            },
            {
                "time_index": 1,
                "vehicles": [
                    {
                        "vehicle_id": "veh_a",
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "speed": 1.0,
                        "base_model_id": "veh_base_v1",
                    },
                    {
                        "vehicle_id": "veh_b",
                        "position_x": 20.0,
                        "position_y": 0.0,
                        "speed": 1.0,
                        "base_model_id": "veh_base_v1",
                    },
                ],
            },
        ]
        rsus = [
            RSUState(rsu_id="rsu_a", position_x=0.0, position_y=0.0, coverage_radius=12.0),
            RSUState(rsu_id="rsu_b", position_x=20.0, position_y=0.0, coverage_radius=12.0),
        ]
        env = VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(trajectory_frames=frames),
            rsu_states=rsus,
            primary_vehicle_selection="handoff_pressure",
        )

        state, _ = env.reset()

        self.assertEqual(state["primary_vehicle_id"], "veh_b")
        self.assertEqual(state["vehicles"][0]["vehicle_id"], "veh_b")
        self.assertEqual(state["primary_vehicle_selection"], "handoff_pressure")
        self.assertTrue(state["primary_vehicle_handoff_pressure_enabled"])


if __name__ == "__main__":
    unittest.main()
