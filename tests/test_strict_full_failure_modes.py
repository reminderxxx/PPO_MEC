from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_strict_full_failure_modes",
    ROOT_DIR / "scripts" / "analyze_strict_full_failure_modes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(agent: str, reward: float, continuity: float, failure: float) -> dict[str, str]:
    return {
        "agent_name": agent,
        "seed": "7",
        "window_id": "w1",
        "workflow_id": "j1",
        "window_class": "mechanism_activating",
        "total_reward": str(reward),
        "workflow_continuity_rate": str(continuity),
        "handoff_failure_rate": str(failure),
    }


def test_build_delta_rows_preserves_raw_metric_direction() -> None:
    pairs = [(row("sa_ghmappo", 5.0, 0.7, 0.2), row("dt_handoff_drl", 4.0, 0.9, 0.1))]
    delta = MODULE.build_delta_rows("formal", pairs)[0]
    assert delta["delta_total_reward"] == 1.0
    assert delta["delta_workflow_continuity_rate"] == pytest.approx(-0.2)
    assert delta["delta_handoff_failure_rate"] == pytest.approx(0.1)


def test_report_flags_joint_continuity_and_failure_regression() -> None:
    pairs = [(row("sa_ghmappo", 5.0, 0.7, 0.2), row("dt_handoff_drl", 4.0, 0.9, 0.1))]
    report = MODULE.build_report(MODULE.build_delta_rows("formal", pairs))
    assert report["failure_counts"] == {
        "continuity_worse": 1,
        "handoff_failure_worse": 1,
        "both_worse": 1,
    }


def test_hidden_label_is_rejected() -> None:
    try:
        MODULE.parse_labeled_path("hidden_holdout=/tmp/rows.csv")
    except ValueError as error:
        assert "hidden holdout" in str(error)
    else:
        raise AssertionError("hidden label should be rejected")
