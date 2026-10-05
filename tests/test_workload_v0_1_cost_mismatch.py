from __future__ import annotations

import json
from pathlib import Path

from scripts.run_workload_v0_1_cost_mismatch_robustness import evaluate_design


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_frozen_cost_mismatch_matrix_separates_estimate_and_realization() -> None:
    robustness = _load(
        "configs/experiment/workload_v0_1_cost_mismatch_robustness_v1.json"
    )
    baseline = _load(
        "configs/experiment/measurement_calibrated_vec_workload_v0_1.json"
    )
    assert len(robustness["design_points"]) == 12
    rows = evaluate_design(robustness, baseline)
    assert len(rows) == 36
    local = [
        row
        for row in rows
        if row["method"] == "mechanism_aware_local_incremental_cost"
    ]
    oracle = [row for row in rows if row["method"] == "post_hoc_exact_oracle"]
    assert all(row["decisions"] == ["recover"] * 3 for row in local)
    assert any(row["decisions"] == ["recover"] * 3 for row in oracle)
    assert any(row["decisions"] == ["restart"] * 3 for row in oracle)
    assert all(row["completed_workflows"] == 3 for row in rows)
    assert all(
        row["oracle_information_boundary"] == "post_hoc_realized_cost_oracle"
        for row in oracle
    )


def test_target_ready_information_is_common_cost_under_current_contract() -> None:
    robustness = _load(
        "configs/experiment/workload_v0_1_cost_mismatch_robustness_v1.json"
    )
    baseline = _load(
        "configs/experiment/measurement_calibrated_vec_workload_v0_1.json"
    )
    rows = evaluate_design(robustness, baseline)
    local = {
        row["design_id"]: row
        for row in rows
        if row["method"] == "mechanism_aware_local_incremental_cost"
    }
    assert local["r03"]["estimated_common_model_prepare_seconds"] > 0.0
    assert local["r04"]["estimated_common_model_prepare_seconds"] == 0.0
    assert local["r05"]["estimated_common_model_prepare_seconds"] > 0.0
    assert local["r03"]["decisions"] == local["r04"]["decisions"] == local["r05"]["decisions"]
