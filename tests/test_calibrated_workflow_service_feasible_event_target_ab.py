"""Guard the target-only candidate protocol without scientific execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_calibrated_workflow_service_feasible_event_target_ab import (
    _behavior_summary,
    _verdict,
)
from scripts.run_calibrated_workflow_service_feasible_event_target_ab import (
    DEFAULT_CONTROL_ROOT,
    DEFAULT_PROTOCOL,
    execute,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary(on_time: float = 0.1, failure: float = -0.1, completion: float = 0.0) -> list[dict]:
    return [
        {
            "checkpoint_view": view,
            "split": split,
            "mean_delta_workflow_completion_rate": completion,
            "mean_delta_on_time_workflow_completion_rate": on_time,
            "mean_delta_service_failure_rate": failure,
        }
        for view in ("selected", "update96")
        for split in ("regression", "frozen_check", "combined")
    ]


def _seeds(on_time: float = 0.1, failure: float = -0.1) -> list[dict]:
    return [
        {
            "checkpoint_view": view,
            "split": "combined",
            "seed": seed,
            "mean_delta_on_time_workflow_completion_rate": on_time,
            "mean_delta_service_failure_rate": failure,
        }
        for view in ("selected", "update96")
        for seed in (7, 17, 29, 43, 61)
    ]


def test_pending_preflight_is_target_only_and_does_no_scientific_work() -> None:
    receipt = preflight(ROOT / DEFAULT_PROTOCOL, DEFAULT_CONTROL_ROOT)
    assert receipt["status"] == "preflight_passed"
    assert receipt["execution_authorized"] is False
    assert receipt["authorization_state"] == "awaiting_symmetric_action_branch_gate"
    assert receipt["branch_gate_status"] == "pending_a_handoff"
    assert receipt["scientific_steps"] == receipt["new_evaluation_episodes"] == 0
    assert receipt["control_rows"] == {"selected": 400, "update96": 400, "rules": 40}
    assert receipt["network_identity"] == {
        "legacy_parameter_count": 165512,
        "candidate_parameter_count": 165512,
        "state_dict_keys": 60,
        "initialized_tensors_equal": True,
    }


def test_pending_protocol_refuses_scientific_execution(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not authorized by a PASS branch gate"):
        execute(ROOT / DEFAULT_PROTOCOL, DEFAULT_CONTROL_ROOT, tmp_path / "run", ["test"])
    assert not (tmp_path / "run").exists()


def test_verdict_enforces_joint_service_and_seed_direction_gates() -> None:
    assert _verdict(_summary(), _seeds())["status"] == "PASS"
    assert _verdict(_summary(completion=-0.01), _seeds())["status"] == "FAIL"

    tradeoff = _seeds()
    for row in tradeoff[:3]:
        row["mean_delta_service_failure_rate"] = 0.1
    verdict = _verdict(_summary(), tradeoff)
    assert verdict["status"] == "MIXED"
    assert verdict["seed_direction"]["selected"]["improved"] == 2
    assert verdict["seed_direction"]["selected"]["worsened_or_tradeoff"] == 3


def test_behavior_summary_excludes_non_sa_control_rows() -> None:
    common = {
        "split": "regression",
        "seed": "7",
        "design_id": "case",
        "step_index": "0",
        "executed_action": "4",
        "current_bundle_ready": "False",
        "migration_success": "True",
        "current_rsu_id": "rsu_0",
        "service_completed": "True",
        "state_ready": "False",
        "current_prepared_exists": "False",
        "current_prepared_valid": "False",
    }
    result = _behavior_summary(
        [{**common, "method": "sa_ghmappo"}, {**common, "method": "ppo"}],
        "legacy_event_target_v1",
        "selected",
    )
    assert result["episode_n"] == 1
    assert result["behavior_step_n"] == 1
    assert result["current_missing_action4"] == 1
    assert result["state_commits"] == 1
