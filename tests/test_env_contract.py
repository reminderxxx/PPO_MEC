"""核心环境最小接口合同测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv, make_toy_vec_env
from src.envs.specs import ControlAction, RSUState


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
