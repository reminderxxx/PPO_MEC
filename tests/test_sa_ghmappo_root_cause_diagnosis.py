from __future__ import annotations

import math

import pytest

from scripts.diagnose_sa_ghmappo_completion_root_cause import (
    _head_probability_pushforward,
    _original_score,
)


def test_head_probability_pushforward_sums_all_aliases() -> None:
    probabilities = {
        "slow": [0.2, 0.3, 0.5],
        "fast": [0.4, 0.6],
        "event": [0.7, 0.3],
    }
    result = _head_probability_pushforward(probabilities, [True] * 5)
    actual = result["masked_env_probabilities"]
    expected = [0.7 * 0.3, 0.7 * 0.5, 0.7 * 0.2 * 0.6, 0.7 * 0.2 * 0.4, 0.3]
    assert math.isclose(sum(actual), 1.0)
    assert all(math.isclose(left, right) for left, right in zip(actual, expected))
    assert len(result["contributions"]["4"]) == 6


def test_original_reward_has_no_external_truncation_or_incomplete_deadline_cost() -> None:
    objective = {
        "node_completion_reward": 2.0,
        "workflow_completion_reward": 4.0,
        "time_weight": 0.08,
        "transfer_gib_weight": 0.5,
        "failure_penalty": 3.0,
        "deadline_penalty": 4.0,
    }
    case = {
        "case": "overdue_truncated",
        "completed_nodes": 4,
        "total_nodes": 5,
        "workflow_completed": False,
        "elapsed_seconds": 80.0,
        "deadline_seconds": 60.0,
        "transfer_bytes": 0,
        "service_failures": 0,
        "external_truncation": True,
    }
    scored = _original_score(case, objective)
    assert scored["late"] is False
    assert scored["reward_components"]["reward_deadline_penalty"] == 0.0
    assert scored["original_reward"] == pytest.approx(1.6)
