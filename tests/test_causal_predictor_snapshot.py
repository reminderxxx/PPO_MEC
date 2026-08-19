"""G12 causal calibration, snapshot, runtime, and trace contract tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from scripts.audit_predictor_calibration import hard_classification_metrics
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import summary_to_row
from src.metrics.recorder import EpisodeRecorder, validate_decision_observation_trace_record
from src.predictors import CHECKPOINT_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION
from src.predictors.calibration import (
    apply_binary_temperature,
    audit_predictor_splits,
    binary_calibration_metrics,
    canonical_sha256,
    eta_regression_metrics,
    fit_binary_temperature,
    identity_calibration,
    multiclass_calibration_metrics,
    reliability_bins,
    risk_coverage_curve,
    select_abstention_threshold,
    select_calibration_method,
    validate_probability_simplex,
)
from src.predictors.causal_snapshot import (
    CALIBRATION_ARTIFACT_CONTRACT_VERSION,
    build_causal_predictor_snapshot,
    calibration_artifact_round_trip,
    consume_snapshot,
    load_calibration_artifact,
    validate_causal_predictor_snapshot,
)
from src.predictors.supervised_handoff_predictor import (
    SupervisedHandoffPredictorNetwork,
    SupervisedHandoffPredictorRuntime,
)


class CausalPredictorSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.calibration_path = self.root / "calibration.json"
        self.calibration_payload = {
            "calibration_artifact_contract_version": CALIBRATION_ARTIFACT_CONTRACT_VERSION,
            "calibration_method_version": "test_temperature_v1",
            "fit_split": "calibration",
            "evaluation_labels_used_for_fit": False,
            "rl_reward_used_for_selection": False,
            "parameters": {
                "handoff": {"method": "identity", "temperature": 1.0},
                "next_rsu": {"method": "identity", "temperature": 1.0},
                "handoff_target": {"method": "identity", "temperature": 1.0},
            },
            "abstention": {"selected_threshold": 0.6},
        }
        self.calibration_path.write_text(json.dumps(self.calibration_payload), encoding="utf-8")
        self.calibration = load_calibration_artifact(self.calibration_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def snapshot(self, **overrides):
        arguments = {
            "vehicle_id": "veh_1",
            "predictor_kind": "supervised",
            "model_identity": "fixture",
            "checkpoint_identity": {"sha256": "abc", "horizon": 3},
            "predictor_config_hash": canonical_sha256({"config": 1}),
            "source_dataset_identity": {"dataset": "fixture", "cross_source": False},
            "source_window_plan_identity": {"window_id": "w1"},
            "git_commit": "abc123",
            "generated_at_step": 2,
            "generated_at_time": 20,
            "observation_as_of_step": 2,
            "observation_as_of_time": 20,
            "consumed_at_step": 2,
            "consumed_at_time": 20,
            "label_horizon": 3,
            "valid_for_steps": 3,
            "update_interval_steps": 3,
            "current_rsu_id": "rsu_a",
            "class_ids": ["rsu_a", "rsu_b", None],
            "runtime_rsu_ids": ["rsu_a", "rsu_b"],
            "raw_next_rsu_logits": [-5.0, 8.0, -6.0],
            "raw_handoff_logit": 10.0,
            "raw_handoff_target_logits": [-5.0, 8.0, -6.0],
            "eta_point_estimate": 2.0,
            "calibration_artifact": self.calibration,
            "feature_availability_mask": {"current": True, "history": True},
            "normalization_version": "fixed_v1",
            "history_start_step": 1,
            "history_end_step": 2,
            "history_start_time": 10,
            "history_end_time": 20,
            "source_frame_interval": [1, 2],
            "source_time_interval": [10, 20],
            "causal_cutoff_step": 2,
            "causal_cutoff_time": 20,
        }
        arguments.update(overrides)
        return build_causal_predictor_snapshot(**arguments)

    def make_checkpoint(self) -> Path:
        path = self.root / "predictor.pt"
        rsu_ids = ["rsu_a", "rsu_b", "rsu_c", "rsu_d"]
        input_dim = 10 + 7 * len(rsu_ids)
        network = SupervisedHandoffPredictorNetwork(input_dim, len(rsu_ids) + 1, hidden_dim=8)
        for parameter in network.parameters():
            parameter.data.zero_()
        network.next_rsu_head.bias.data[1] = 8.0
        network.handoff_target_head.bias.data[1] = 8.0
        network.handoff_logit_head.bias.data[0] = 8.0
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
                "calibration": {"handoff_decision_threshold": 0.5, "target_selection_mode": "soft"},
                "model_state_dict": network.state_dict(),
                "metrics": {},
            },
            path,
        )
        return path

    def test_valid_causal_time_and_expiry(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(validate_causal_predictor_snapshot(snapshot)["status"], "pass")
        stale = consume_snapshot(snapshot, consumed_at_step=6, consumed_at_time=60)
        self.assertEqual(stale["causal_time"]["age_steps"], 4)
        self.assertIn("snapshot_stale", stale["predictions"]["abstention_reasons"])
        self.assertIn("prediction_expired", stale["predictions"]["abstention_reasons"])
        self.assertEqual(stale["predictions"]["availability_mask"], 0)

    def test_future_as_of_and_history_are_rejected(self) -> None:
        snapshot = self.snapshot()
        snapshot["causal_time"]["observation_as_of_step"] = 3
        snapshot["causal_time"]["history_end_step"] = 4
        validation = validate_causal_predictor_snapshot(snapshot)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("future" in error for error in validation["errors"]))

    def test_cold_start_and_target_not_distinct_are_abstentions(self) -> None:
        cold = self.snapshot(insufficient_history=True)
        self.assertIn("insufficient_history", cold["predictions"]["abstention_reasons"])
        same = self.snapshot(
            raw_handoff_target_logits=[8.0, -5.0, -6.0],
            raw_next_rsu_logits=[8.0, -5.0, -6.0],
        )
        self.assertIn("target_not_distinct", same["predictions"]["abstention_reasons"])

    def test_oracle_identity_and_runtime_labels_are_isolated(self) -> None:
        snapshot = self.snapshot()
        snapshot["identity"]["oracle"] = True
        validation = validate_causal_predictor_snapshot(snapshot)
        self.assertEqual(validation["status"], "fail")
        leaked = self.snapshot()
        leaked["predictions"]["future_label"] = 1
        self.assertEqual(validate_causal_predictor_snapshot(leaked)["status"], "fail")

    def test_unseen_rsu_and_unavailable_null_are_distinct_from_zero(self) -> None:
        unseen = self.snapshot(runtime_rsu_ids=["rsu_a"])
        self.assertIn("unseen_rsu_or_class", unseen["predictions"]["abstention_reasons"])
        unavailable = self.snapshot(
            predictor_available=False,
            raw_next_rsu_logits=None,
            raw_handoff_logit=None,
            raw_handoff_target_logits=None,
        )
        self.assertIsNone(unavailable["predictions"]["handoff_probability"])
        self.assertNotEqual(unavailable["predictions"]["handoff_probability"], 0.0)

    def test_probability_range_simplex_nan_and_unknown_class(self) -> None:
        self.assertTrue(validate_probability_simplex([0.25, 0.75])["valid"])
        self.assertFalse(validate_probability_simplex([0.25, 0.5])["valid"])
        with self.assertRaises(ValueError):
            validate_probability_simplex([math.nan, 1.0])
        with self.assertRaises(ValueError):
            binary_calibration_metrics([1], [1.1])
        with self.assertRaises(ValueError):
            multiclass_calibration_metrics([2], [[0.5, 0.5]])

    def test_binary_metrics_match_manual_case(self) -> None:
        metrics = binary_calibration_metrics([0, 1], [0.25, 0.75], threshold=0.5)
        self.assertAlmostEqual(metrics["threshold_independent"]["brier_score"], 0.0625)
        self.assertAlmostEqual(metrics["threshold_independent"]["negative_log_likelihood"], -math.log(0.75))
        self.assertAlmostEqual(metrics["threshold_independent"]["ece"], 0.25)
        self.assertEqual(metrics["threshold_dependent"]["f1"], 1.0)

    def test_multiclass_metrics_and_classwise_calibration(self) -> None:
        metrics = multiclass_calibration_metrics([0, 1], [[0.8, 0.2], [0.1, 0.9]], class_names=["a", "b"])
        self.assertEqual(metrics["top_1_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["multiclass_brier_score"], 0.05)
        self.assertAlmostEqual(metrics["negative_log_likelihood"], -(math.log(0.8) + math.log(0.9)) / 2)
        self.assertIn("a", metrics["classwise_calibration"])
        self.assertTrue(metrics["probability_simplex_validation"]["passed"])

    def test_historical_empty_next_label_maps_to_none_class(self) -> None:
        metrics = hard_classification_metrics(
            [
                {"next": "rsu_a", "prediction": "0"},
                {"next": "", "prediction": "1"},
            ],
            label_key="next",
            prediction_key="prediction",
            class_names=["rsu_a", "__none__"],
            empty_label_class="__none__",
        )
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["empty_label_count"], 1)
        self.assertEqual(metrics["unknown_class_count"], 0)
        self.assertEqual(metrics["top_1_accuracy"], 1.0)

    def test_empty_bins_zero_denominators_and_missing_tasks_are_null(self) -> None:
        bins = reliability_bins([1], [0.95])
        self.assertEqual(bins["empty_bin_count"], 9)
        empty = binary_calibration_metrics([], [])
        self.assertEqual(empty["availability"], "unavailable")
        self.assertIsNone(empty["threshold_independent"]["brier_score"])
        eta = eta_regression_metrics([None], [None])
        self.assertEqual(eta["availability"], "unavailable")
        self.assertIsNone(eta["mae"])

    def test_eta_metrics_match_manual_case(self) -> None:
        metrics = eta_regression_metrics([1.0, 3.0], [2.0, 1.0])
        self.assertAlmostEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.5))
        self.assertAlmostEqual(metrics["median_absolute_error"], 1.5)
        self.assertIsNone(metrics["interval_coverage"])
        self.assertFalse(metrics["interval_is_classification_confidence"])

    def test_temperature_fit_is_deterministic_and_calibration_only(self) -> None:
        labels = [0, 0, 1, 1]
        logits = [-2.0, -1.0, 1.0, 2.0]
        first = fit_binary_temperature(labels, logits, seed=7)
        second = fit_binary_temperature(labels, logits, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["fit_split"], "calibration")
        self.assertNotIn("evaluation", first)
        self.assertEqual(len(apply_binary_temperature(logits, first["temperature"])), 4)

    def test_method_selection_reports_degradation_and_uses_lexicographic_rule(self) -> None:
        selection = select_calibration_method(
            [
                {"method": "binary_temperature_scaling", "metrics": {"negative_log_likelihood": 0.4, "brier_score": 0.2}},
                {"method": "identity", "metrics": {"negative_log_likelihood": 0.4, "brier_score": 0.2}},
            ]
        )
        self.assertEqual(selection["selected_method"], "identity")
        worse = binary_calibration_metrics([0, 1], [0.49, 0.51])
        better = binary_calibration_metrics([0, 1], [0.1, 0.9])
        self.assertGreater(worse["threshold_independent"]["brier_score"], better["threshold_independent"]["brier_score"])

    def test_calibration_parameter_round_trip(self) -> None:
        round_trip = calibration_artifact_round_trip(self.calibration_payload)
        self.assertEqual(round_trip, self.calibration_payload)

    def test_split_frame_time_overlap_and_hidden_are_rejected(self) -> None:
        def window(name, start, time_start, split=""):
            return {
                "window_id": name,
                "source_segment_id": "segment",
                "frame_offset": start,
                "window_length": 4,
                "time_index_start": time_start,
                "time_index_end": time_start + 3,
                "split": split,
            }
        clean = {
            "predictor_train": [window("a", 0, 0)],
            "calibration": [window("b", 10, 10)],
            "evaluation_dev": [window("c", 20, 20)],
        }
        audit = audit_predictor_splits(clean)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["split_manifest_sha256"], audit_predictor_splits(clean)["split_manifest_sha256"])
        overlapping = deepcopy(clean)
        overlapping["calibration"] = [window("different_id", 2, 2)]
        failed = audit_predictor_splits(overlapping)
        self.assertFalse(failed["passed"])
        self.assertTrue(any(row["interval_kind"] == "frame_interval" for row in failed["overlap_conflicts"]))
        self.assertTrue(any(row["interval_kind"] == "time_interval" for row in failed["overlap_conflicts"]))
        hidden = deepcopy(clean)
        hidden["evaluation_dev"][0]["split"] = "hidden"
        with self.assertRaises(ValueError):
            audit_predictor_splits(hidden)

    def test_risk_coverage_threshold_is_reward_independent(self) -> None:
        curve = risk_coverage_curve([0, 1, 1], [0.1, 0.6, 0.9], candidate_thresholds=[0.0, 0.8])
        self.assertEqual(len(curve), 2)
        selected = select_abstention_threshold(
            [0, 1, 1], [0.1, 0.6, 0.9], minimum_coverage=0.5, candidate_thresholds=[0.0, 0.6, 0.8]
        )
        self.assertFalse(selected["uses_rl_reward"])
        self.assertEqual(
            selected["selection_rule"],
            "calibration_only_minimum_accepted_Brier_subject_to_minimum_coverage",
        )

    def test_low_confidence_and_invalid_artifact_fail_safe(self) -> None:
        low = self.snapshot(raw_handoff_logit=0.0)
        self.assertIn("confidence_below_threshold", low["predictions"]["abstention_reasons"])
        invalid = deepcopy(self.calibration_payload)
        invalid["fit_split"] = "evaluation_dev"
        path = self.root / "invalid.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_calibration_artifact(path)

    def test_feature_flag_default_is_output_compatible(self) -> None:
        implicit = VecWorkflowCoreEnv(predictor_manager=PredictorManager())
        explicit = VecWorkflowCoreEnv(
            predictor_manager=PredictorManager(causal_calibrated_snapshot_enabled=False)
        )
        implicit_state, _ = implicit.reset()
        explicit_state, _ = explicit.reset()
        self.assertEqual(implicit_state["predictions"], explicit_state["predictions"])
        self.assertNotIn("causal_predictor_snapshots_by_vehicle", implicit_state["predictions"])

        implicit_gym = GymVecEnv(core_env=VecWorkflowCoreEnv(predictor_manager=PredictorManager()))
        explicit_gym = GymVecEnv(
            core_env=VecWorkflowCoreEnv(
                predictor_manager=PredictorManager(causal_calibrated_snapshot_enabled=False)
            )
        )
        implicit_observation, implicit_info = implicit_gym.reset(seed=7)
        explicit_observation, explicit_info = explicit_gym.reset(seed=7)
        self.assertEqual(implicit_observation, explicit_observation)
        self.assertEqual(implicit_info["action_schema"], explicit_info["action_schema"])
        action = next(index for index, allowed in enumerate(implicit_info["action_mask"]) if allowed)
        implicit_step = implicit_gym.step(action)
        explicit_step = explicit_gym.step(action)
        self.assertEqual(implicit_step[0], explicit_step[0])
        self.assertEqual(implicit_step[1], explicit_step[1])
        self.assertEqual(
            implicit_step[4]["control_action"], explicit_step[4]["control_action"]
        )

    def test_checkpoint_feature_order_hash_is_validated(self) -> None:
        checkpoint = self.make_checkpoint()
        payload = torch.load(checkpoint, map_location="cpu")
        payload["feature_schema"]["feature_order_sha256"] = "wrong"
        torch.save(payload, checkpoint)
        with self.assertRaises(ValueError):
            SupervisedHandoffPredictorRuntime(checkpoint)

    def test_runtime_rejects_oracle_or_legacy_perturbation_mixing(self) -> None:
        checkpoint = self.make_checkpoint()
        common = {
            "predictor_kind": "supervised",
            "predictor_checkpoint_path": str(checkpoint),
            "causal_calibrated_snapshot_enabled": True,
        }
        with self.assertRaises(ValueError):
            PredictorManager(**common, oracle_prediction_enabled=True)
        with self.assertRaises(ValueError):
            PredictorManager(**common, prediction_noise_std=0.1)

    def test_runtime_accepts_masks_delays_and_resets_historical_snapshots(self) -> None:
        checkpoint = self.make_checkpoint()
        manager = PredictorManager(
            predictor_kind="supervised",
            predictor_checkpoint_path=str(checkpoint),
            causal_calibrated_snapshot_enabled=True,
            causal_snapshot_calibration_artifact_path=str(self.calibration_path),
            causal_snapshot_update_interval_steps=1,
            causal_snapshot_valid_for_steps=3,
            causal_snapshot_min_history_steps=0,
            prediction_delay_steps=1,
        )
        env = VecWorkflowCoreEnv(predictor_manager=manager, max_steps=5)
        initial, _ = env.reset()
        self.assertEqual(initial["predictions"]["causal_predictor_snapshots_by_vehicle"], {})
        control = ControlAction(
            cache_action={},
            offload_action={"mode": "hybrid"},
            migration_action={"mode": "migrate"},
        )
        state, _, _, _, _ = env.step(control)
        snapshots = state["predictions"]["causal_predictor_snapshots_by_vehicle"]
        self.assertTrue(snapshots)
        self.assertTrue(all(snapshot["causal_time"]["generated_at_step"] == 0 for snapshot in snapshots.values()))
        self.assertTrue(all(snapshot["causal_time"]["consumed_at_step"] == 1 for snapshot in snapshots.values()))
        self.assertGreaterEqual(sum(state["predictions"]["causal_snapshot_availability_by_vehicle"].values()), 1)
        reset_state, _ = env.reset()
        self.assertEqual(reset_state["predictions"]["causal_predictor_snapshots_by_vehicle"], {})
        self.assertFalse(reset_state["predictions"]["causal_snapshot_runtime_audit"]["oracle_fallback_allowed"])

    def test_abstained_runtime_uses_mask_not_no_handoff(self) -> None:
        checkpoint = self.make_checkpoint()
        payload = deepcopy(self.calibration_payload)
        payload["abstention"]["selected_threshold"] = 1.0
        high_path = self.root / "high_threshold.json"
        high_path.write_text(json.dumps(payload), encoding="utf-8")
        env = VecWorkflowCoreEnv(
            predictor_manager=PredictorManager(
                predictor_kind="supervised",
                predictor_checkpoint_path=str(checkpoint),
                causal_calibrated_snapshot_enabled=True,
                causal_snapshot_calibration_artifact_path=str(high_path),
                causal_snapshot_min_history_steps=0,
            )
        )
        state, _ = env.reset()
        predictions = state["predictions"]
        self.assertTrue(all(value == 0 for value in predictions["causal_snapshot_availability_by_vehicle"].values()))
        self.assertTrue(all(value is None for value in predictions["predicted_next_rsu_by_vehicle"].values()))
        self.assertTrue(all(len(snapshot["predictions"]["abstention_reasons"]) > 0 for snapshot in predictions["causal_predictor_snapshots_by_vehicle"].values()))

    def test_trace_is_pre_action_aligned_json_safe_and_provenance_reaches_row(self) -> None:
        checkpoint = self.make_checkpoint()
        recorder = EpisodeRecorder()
        core = VecWorkflowCoreEnv(
            predictor_manager=PredictorManager(
                predictor_kind="supervised",
                predictor_checkpoint_path=str(checkpoint),
                causal_calibrated_snapshot_enabled=True,
                causal_snapshot_calibration_artifact_path=str(self.calibration_path),
                causal_snapshot_min_history_steps=0,
            ),
            max_steps=4,
        )
        env = GymVecEnv(core_env=core, recorder=recorder)
        recorder.start_episode({"evaluation_unit_id": "unit_1"})
        observation, info = env.reset(seed=7)
        self.assertGreater(observation[5], 0.0)
        action = next(index for index, allowed in enumerate(info["action_mask"]) if allowed)
        _, _, _, _, step_info = env.step(action)
        trace = step_info["decision_observation_trace_record"]
        self.assertEqual(trace["captured_phase"], "pre_action")
        self.assertEqual(trace["request_id"], "unit_1/request_000001")
        self.assertEqual(len(trace["flattened_observation"]), len(trace["feature_name_to_index"]))
        for name, index in trace["feature_name_to_index"].items():
            self.assertEqual(trace["flattened_feature_values"][name], trace["flattened_observation"][index])
        self.assertFalse(trace["future_or_outcome_fields_present"])
        self.assertNotIn("reward", trace)
        self.assertIsNotNone(trace["predictor_snapshot_provenance"]["snapshot_id"])
        self.assertEqual(trace["predictor_snapshot_provenance"]["availability_mask"], 1)
        validate_decision_observation_trace_record(trace)
        leaked_trace = deepcopy(trace)
        leaked_trace["raw_semantic_fields"]["future_label"] = 1
        with self.assertRaises(ValueError):
            validate_decision_observation_trace_record(leaked_trace)
        self.assertIsInstance(json.loads(json.dumps(trace, allow_nan=False)), dict)
        summary = recorder.build_summary()
        self.assertEqual(summary["decision_observation_trace"]["records"][0]["request_id"], trace["request_id"])
        row = summary_to_row(summary)
        self.assertIn("causal_predictor_snapshot_contract_versions", row)
        self.assertIn("causal_predictor_snapshot_ids", row)


if __name__ == "__main__":
    unittest.main()
