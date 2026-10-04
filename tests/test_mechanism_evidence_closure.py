from __future__ import annotations

import json
from pathlib import Path

from scripts.run_mechanism_evidence_closure import (
    build_experiment_a,
    current_selector,
    run_policy,
    two_step_selector,
)
from scripts.run_two_node_workflow_reexecution_comparison import compare_receipts


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/experiment/mechanism_evidence_closure_v1.json"


def plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_frozen_matrix_has_twelve_scenarios_and_four_axes() -> None:
    spec = plan()["experiment_c"]
    assert len(spec["scenarios"]) == spec["scenario_count"] == 12
    assert {row["sharing"] for row in spec["scenarios"]} == {True, False}
    assert {row["capacity"] for row in spec["scenarios"]} == {"tight", "ample"}
    assert {row["tail"] for row in spec["scenarios"]} == {"short", "long"}
    assert {row["state_cost"] for row in spec["scenarios"]} == {"low", "high"}


def test_experiment_a_preserves_equal_candidate_completion_and_rejection_guard() -> None:
    result = build_experiment_a(plan())
    candidate = [row for row in result["ledgers"] if row["semantics_label"] == "candidate"]
    old = [row for row in result["ledgers"] if row["semantics_label"] == "old"]
    assert {row["completed_request_count"] for row in candidate} == {72}
    assert {row["service_failure_count"] for row in candidate} == {0}
    assert sorted(row["service_failure_count"] for row in old) == [36, 36, 60, 60]
    assert result["equal_completion_comparison"]["total_transfer_bytes_saved"] == 5637144576


def test_current_and_two_step_use_only_legal_native_actions() -> None:
    frozen = plan()
    scenario = frozen["experiment_c"]["scenarios"][1]
    for selector in (current_selector, two_step_selector):
        episode = run_policy(scenario, frozen, selector)
        assert episode["feasible"] is True
        assert all(step["action_allowed"] and not step["action_invalid"] for step in episode["steps"])


def test_comparison_requires_restart_and_recovery_fidelity() -> None:
    base = {
        "identity": {"x": 1},
        "adapter_status": {"active_adapters": ["a"]},
        "n0_output": {"decoded_text": "x", "token_ids": [1]},
        "n1_input": {"prompt": "p", "rendered_prompt": "r", "content_sha256": "h", "input_ids": [2]},
        "n1_output": {"token_ids": [3]},
    }
    continuous = {**base, "node_call_counts": {"n0": 1, "n1": 1}}
    source = {**base, "node_call_counts": {"n0": 1, "n1": 0}}
    restart = {
        **base,
        "node_call_counts": {"n0": 1, "n1": 1},
        "target_access": {"source_image_access_count": 1},
    }
    target = {
        **base,
        "node_call_counts": {"n0": 0, "n1": 1},
        "restored_n0_output": {"raw_text": "x", "token_ids": [1]},
        "target_access": {"source_image_access_count": 0},
    }
    assert compare_receipts(continuous, source, restart, target)["status"] == "PASS"
