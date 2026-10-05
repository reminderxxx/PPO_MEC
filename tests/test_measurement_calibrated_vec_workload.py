from __future__ import annotations

import json
from pathlib import Path

from src.data.workflow.measurement_calibrated_vec_workload import (
    compare_methods,
    generate_workloads,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads(
        (ROOT / "configs/experiment/measurement_calibrated_vec_workload_v0_1.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_workload_has_all_cells_and_provenance() -> None:
    value = config()
    workloads = generate_workloads(value)
    assert len(workloads) == 24
    assert {item["design_id"] for item in workloads} == {
        f"d{index:02d}" for index in range(1, 9)
    }
    assert {item["seed"] for item in workloads} == {17, 29, 43}
    for item in workloads:
        assert item["provenance"]["old_holdout_used"] is False
        assert item["provenance"]["typed_model_objects_are_abstract"] is True
        assert item["integrity"] == {
            "workflow_count": 3,
            "node_count": 6,
            "seed": item["seed"],
        }


def test_local_rule_matches_exact_and_keeps_unfavorable_control() -> None:
    value = config()
    rows = [
        row
        for workload in generate_workloads(value)
        for row in compare_methods(workload, value)
    ]
    assert len(rows) == 72
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["design_id"], row["seed"]), {})[row["method"]] = row
    for methods in by_cell.values():
        local = methods["mechanism_aware_local_incremental_cost"]
        exact = methods["offline_exact_enumeration"]
        assert local["decisions"] == exact["decisions"]
        assert local["completed_workflows"] == exact["completed_workflows"] == 3
    assert any(
        row["method"] == "mechanism_aware_local_incremental_cost"
        and row["restore_cost"] == "low"
        and row["decisions"] == ["recover", "recover", "recover"]
        for row in rows
    )
    assert any(
        row["method"] == "mechanism_aware_local_incremental_cost"
        and row["restore_cost"] == "high"
        and row["decisions"] == ["restart", "restart", "restart"]
        for row in rows
    )
