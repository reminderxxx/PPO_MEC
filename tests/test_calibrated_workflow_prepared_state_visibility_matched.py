"""Guard the matched runner and paired analysis without scientific execution."""

from __future__ import annotations

from pathlib import Path

from scripts.analyze_calibrated_workflow_prepared_state_visibility_matched import _paired
from scripts.run_calibrated_workflow_prepared_state_visibility_matched import (
    DEFAULT_HISTORICAL_ROOT,
    DEFAULT_PROTOCOL,
    DEFAULT_RULE_ROOT,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_binds_sources_without_training_or_evaluation() -> None:
    receipt = preflight(ROOT / DEFAULT_PROTOCOL, DEFAULT_HISTORICAL_ROOT, DEFAULT_RULE_ROOT)
    assert receipt["status"] == "preflight_passed"
    assert receipt["execution_authorized"] is False
    assert receipt["scientific_steps"] == receipt["new_evaluation_episodes"] == 0
    assert receipt["historical_selected_rows"] == 400
    assert receipt["historical_rule_rows"] == 40
    assert receipt["historical_update96_checkpoints"] == 20
    assert receipt["source_identity"]["public_prefix_suffix_tamper_comparisons"] == 474
    assert {row["delta"] for row in receipt["parameter_counts"]} == {192, 448}


def test_paired_analysis_requires_both_predeclared_views_and_common_completion() -> None:
    rows = []
    for view in ("selected", "update96"):
        for index in range(400):
            common = {
                "checkpoint_view": view,
                "method": ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl")[index // 100],
                "seed": str((7, 17, 29, 43, 61)[(index // 20) % 5]),
                "split": "regression" if index % 20 < 12 else "frozen_check",
                "design_id": f"design_{index % 20}",
                "window_id": f"window_{index % 20}",
                "workflow_id": f"workflow_{index % 20}",
                "source_segment_id": f"source_{index % 20}",
                "workflow_completion_rate": "1",
                "on_time_workflow_completion_rate": "0",
                "service_failure_rate": "0",
                "completed_sample_elapsed_coverage": "1",
                "recompute_seconds": "2",
                "model_prepare_mb": "3",
                "state_transfer_mb": "4",
                "input_transfer_mb": "5",
                "total_transfer_mb": "12",
                "reward": "6",
                "completed_sample_elapsed_seconds": "20",
            }
            rows.append({**common, "observation_arm": "prepared_state_hidden_v3"})
            rows.append({
                **common,
                "observation_arm": "prepared_state_visible_v4",
                "recompute_seconds": "1",
                "completed_sample_elapsed_seconds": "18",
            })
    paired, seed_rows = _paired(rows)
    assert len(paired) == 800
    assert all(row["delta_recompute_seconds"] == -1.0 for row in paired)
    assert all(row["delta_completed_elapsed_seconds"] == -2.0 for row in paired)
    assert len(seed_rows) == 40
