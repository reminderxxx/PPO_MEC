from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_independent_recovery_cost_measurement import validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/acceptance/independent_recovery_cost_measurement_v1.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_frozen_plan_has_exact_24_call_budget_and_unique_repeats() -> None:
    plan = _plan()
    validation = validate_plan(plan)
    assert validation == {
        "planned_generate_calls": 24,
        "condition_repeat_count": 6,
    }
    keys = {
        (item["condition_id"], item["repeat_id"])
        for item in plan["execution_order"]
    }
    assert len(keys) == 6


def test_predictions_are_frozen_before_results_and_select_recovery() -> None:
    plan = _plan()
    for condition in plan["conditions"]:
        predictions = condition["predictions"]
        assert predictions["predicted_cheaper_action"] == "recovery"
        assert (
            predictions["recovery_scored_completion_seconds"]
            < predictions["restart_scored_completion_seconds"]
        )
        assert predictions["eviction_aware_recovery_incremental_seconds"] == pytest.approx(
            0.0263258
        )
        assert predictions["eviction_aware_restart_incremental_seconds"] == pytest.approx(
            11.07072456
        )


def test_missing_local_model_preparation_is_common_to_both_arms() -> None:
    plan = _plan()
    conditions = {item["condition_id"]: item for item in plan["conditions"]}
    prepared = conditions["target_model_prepared"]["predictions"]
    missing = conditions["target_model_requires_local_preparation"]["predictions"]
    assert missing["restart_local_action_wall_seconds"] > prepared[
        "restart_local_action_wall_seconds"
    ]
    assert missing["recovery_local_action_wall_seconds"] > prepared[
        "recovery_local_action_wall_seconds"
    ]
    assert "both full costs" in missing["common_model_preparation_reason"].lower()
    assert missing["eviction_aware_restart_incremental_seconds"] == prepared[
        "eviction_aware_restart_incremental_seconds"
    ]
    assert missing["eviction_aware_recovery_incremental_seconds"] == prepared[
        "eviction_aware_recovery_incremental_seconds"
    ]


def test_unexecutable_real_eviction_reload_remains_uncovered() -> None:
    plan = _plan()
    uncovered = {item["condition"]: item for item in plan["uncovered_conditions"]}
    victim = uncovered["legal cache replacement followed by real later model reload"]
    assert victim["treatment"].startswith("uncovered")
    assert plan["scope"]["new_algorithm_or_system_feature_forbidden"] is True
    assert plan["scope"]["post_result_retuning_forbidden"] is True
