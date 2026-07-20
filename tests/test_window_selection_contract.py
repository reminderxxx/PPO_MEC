"""Window-selection contract tests for formal and holdout benchmark splits."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators import main_results_support


def _window_payload(window_id: str, frame_offset: int, window_class: str) -> dict:
    if window_class == "mechanism_activating":
        handoff_count = 2
        association_change_count = 2
        vehicle_mean = 3.0
    elif window_class == "active_non_mechanism":
        handoff_count = 0
        association_change_count = 1
        vehicle_mean = 3.0
    else:
        handoff_count = 0
        association_change_count = 0
        vehicle_mean = 1.0
    return {
        "window_id": window_id,
        "frame_offset": frame_offset,
        "window_length": 2,
        "time_index_start": frame_offset,
        "time_index_end": frame_offset + 1,
        "dominant_axis": "x",
        "recommended_rsu_layout": "auto_dominant_tight",
        "chosen_rsu_axis": "x",
        "coverage_radius": 10.0,
        "spacing": 12.0,
        "estimated_association_change_count": association_change_count,
        "estimated_handoff_count": handoff_count,
        "active_vehicle_count_mean": vehicle_mean,
        "active_vehicle_count_max": int(vehicle_mean),
        "unique_vehicle_count": int(vehicle_mean),
    }


class WindowSelectionContractTestCase(unittest.TestCase):
    def _resolve_with_offset(self, mode: str) -> dict:
        scan_results = []
        for window_class, prefix in [
            ("mechanism_activating", "m"),
            ("active_non_mechanism", "a"),
            ("idle_or_sparse", "i"),
        ]:
            for index in range(4):
                scan_results.append(_window_payload(f"{prefix}{index}", len(scan_results), window_class))
        with (
            patch.object(main_results_support, "load_real_source_frames", return_value=([{}] * 16, "fake.csv")),
            patch.object(main_results_support, "scan_mobility_windows", return_value=scan_results),
            patch.object(main_results_support, "_build_window_rsus", return_value=([], {})),
            patch.object(
                main_results_support,
                "_estimate_prediction_activity",
                return_value={
                    "predicted_next_rsu_non_null_ratio": 0.0,
                    "predicted_handoff_target_non_null_ratio": 0.0,
                },
            ),
        ):
            _, payload = main_results_support.resolve_window_candidates(
                root_dir=ROOT_DIR,
                mobility_csv_path="",
                max_mobility_rows=16,
                rsu_layout="auto_dominant_tight",
                frame_offset=0,
                window_length=2,
                window_selector="max_handoff_candidate",
                window_count=2,
                window_scan_stride=1,
                random_seed=7,
                window_mode=mode,
                window_rank_offset=1,
                activating_predicted_next_ratio_threshold=0.0,
                activating_handoff_prediction_ratio_threshold=0.0,
            )
        return payload

    def test_full_stratified_rank_offset_skips_each_stratum_prefix(self) -> None:
        payload = self._resolve_with_offset("full_stratified")

        self.assertEqual(payload["window_rank_offset"], 1)
        self.assertEqual(
            [item["window_id"] for item in payload["selected_window_plan_by_strata"]["mechanism_activating"]],
            ["m1", "m2"],
        )
        self.assertEqual(
            [item["window_id"] for item in payload["selected_window_plan_by_strata"]["active_non_mechanism"]],
            ["a1", "a2"],
        )
        self.assertEqual(
            [item["window_id"] for item in payload["selected_window_plan_by_strata"]["idle_or_sparse"]],
            ["i1", "i2"],
        )

    def test_mixed_rank_offset_skips_each_selected_stratum_prefix(self) -> None:
        payload = self._resolve_with_offset("mixed_informative")

        self.assertEqual(
            [item["window_id"] for item in payload["selected_window_plan_by_strata"]["mechanism_activating"]],
            ["m1"],
        )
        self.assertEqual(
            [item["window_id"] for item in payload["selected_window_plan_by_strata"]["active_non_mechanism"]],
            ["a1"],
        )

    def test_high_prediction_without_real_handoff_is_not_mechanism_window(self) -> None:
        busy_no_handoff = _window_payload("busy_no_handoff", 0, "active_non_mechanism")
        busy_no_handoff["estimated_association_change_count"] = 10
        busy_no_handoff["active_vehicle_count_mean"] = 8.0
        mechanism = _window_payload("real_handoff", 1, "mechanism_activating")
        with (
            patch.object(main_results_support, "load_real_source_frames", return_value=([{}] * 8, "fake.csv")),
            patch.object(main_results_support, "scan_mobility_windows", return_value=[busy_no_handoff, mechanism]),
            patch.object(main_results_support, "_build_window_rsus", return_value=([], {})),
            patch.object(
                main_results_support,
                "_estimate_prediction_activity",
                return_value={
                    "predicted_next_rsu_non_null_ratio": 0.9,
                    "predicted_handoff_target_non_null_ratio": 0.9,
                },
            ),
        ):
            _, payload = main_results_support.resolve_window_candidates(
                root_dir=ROOT_DIR,
                mobility_csv_path="",
                max_mobility_rows=8,
                rsu_layout="auto_dominant_tight",
                frame_offset=0,
                window_length=2,
                window_selector="max_handoff_candidate",
                window_count=1,
                window_scan_stride=1,
                random_seed=7,
                window_mode="full_stratified",
            )

        mechanism_ids = [item["window_id"] for item in payload["mechanism_activating_windows"]]
        self.assertEqual(mechanism_ids, ["real_handoff"])
        self.assertNotIn("busy_no_handoff", mechanism_ids)

    def test_layout_candidates_are_forwarded_to_window_scan(self) -> None:
        mechanism = _window_payload("real_handoff", 0, "mechanism_activating")
        with (
            patch.object(main_results_support, "load_real_source_frames", return_value=([{}] * 4, "fake.csv")),
            patch.object(main_results_support, "scan_mobility_windows", return_value=[mechanism]) as scan_mock,
            patch.object(main_results_support, "_build_window_rsus", return_value=([], {})),
            patch.object(
                main_results_support,
                "_estimate_prediction_activity",
                return_value={
                    "predicted_next_rsu_non_null_ratio": 0.9,
                    "predicted_handoff_target_non_null_ratio": 0.9,
                },
            ),
        ):
            main_results_support.resolve_window_candidates(
                root_dir=ROOT_DIR,
                mobility_csv_path="",
                max_mobility_rows=4,
                rsu_layout="auto_dominant_tight",
                layout_candidates=["lust_micro", "tight_y"],
                frame_offset=0,
                window_length=2,
                window_selector="max_handoff_candidate",
                window_count=1,
                window_scan_stride=1,
                random_seed=7,
                window_mode="activating_only",
            )

        self.assertEqual(scan_mock.call_args.kwargs["layout_candidates"], ["lust_micro", "tight_y"])

    def test_load_checkpoint_metadata_uses_checkpoint_update_count_without_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "run" / "checkpoints" / "latest.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "update_count": 4,
                    "training_metadata": {
                        "run_id": "r1",
                        "config_profile": "top_journal_mechanism_v28_credit_focal_mappo",
                        "train_window_mode": "rotate",
                        "episodes": 32,
                        "update_count": 4,
                        "is_smoke_checkpoint": False,
                    },
                    "config": {"mechanism_credit_prd_enabled": True},
                },
                checkpoint_path,
            )

            metadata = main_results_support.load_checkpoint_metadata(str(checkpoint_path))

        self.assertEqual(metadata["metadata_source"], "checkpoint_metadata")
        self.assertEqual(metadata["run_update_count"], 4)
        self.assertEqual(metadata["update_count"], 4)
        self.assertEqual(metadata["checkpoint_source_update_index"], 4)
        self.assertEqual(metadata["experiment_run_type"], "baseline")


if __name__ == "__main__":
    unittest.main()
