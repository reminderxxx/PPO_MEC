"""Conditional reward plots must not turn selective denominators into gains."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "reward_difference_parts", Path(__file__).resolve().parents[1] / "scripts/plot_reward_difference_parts.py"
)
parts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parts)


def rows(deltas, flags=None):
    return [{"delta": delta, "different_actions": flag} for delta, flag in zip(deltas, flags or [True] * len(deltas))]


def test_conditional_amplification_retains_population_and_sign():
    result = parts.decompose(rows([0, -4, 0, 0]))
    assert result["nonzero_reward_conditional_gap"] == -4
    assert result["different_reward_fraction"] == .25
    assert result["full_gap"] == result["nonzero_reward_weighted_contribution"] == -1
    assert result["negative_units"] == 1


def test_positive_and_negative_outcomes_both_retained():
    result = parts.decompose(rows([8, -12, 0, 0]))
    assert result["different_reward_units"] == 2
    assert result["nonzero_reward_conditional_gap"] == -2
    assert result["full_gap"] == -1
    assert result["positive_units"] == result["negative_units"] == 1


def test_behavior_difference_is_not_a_reward_advantage():
    result = parts.decompose(rows([0, 0], [True, False]))
    assert result["different_action_units"] == 1
    assert result["different_actions_equal_reward_units"] == 1
    assert result["nonzero_reward_conditional_gap"] is None
    assert result["different_reward_fraction"] == 0


def test_same_actions_with_different_rewards_not_dropped():
    result = parts.decompose(rows([3, 0], [False, False]))
    assert result["same_actions_different_reward_units"] == 1
    assert result["different_action_conditional_gap"] is None
    assert result["full_gap"] == 1.5


def test_near_zero_remainder_preserves_decomposition():
    result = parts.decompose(rows([1e-9, -2]))
    assert result["different_reward_units"] == 1
    assert result["remaining_weighted_contribution"] == 5e-10
    assert result["full_gap"] == pytest.approx(result["nonzero_reward_weighted_contribution"] + result["remaining_weighted_contribution"])


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_missing_or_nonfinite_population_rejected(values):
    with pytest.raises(ValueError):
        parts.decompose(rows(values))


def test_more_negative_gap_and_plateau_do_not_prove_excellence():
    result = parts.trend_summary([-.1, -.3, -.5, -.7, -.7, -.7], [18.5, 18.4, 18.3, 18.2, 18.2, 18.2])
    assert result["final_gap_positive"] is False
    assert result["gap_never_decreased_at_recorded_points"] is False
    assert result["main_reward_improved"] is False
    assert result["policy_convergence_proven"] is False


def test_grouping_requires_complete_equal_signature():
    signatures = {"a": (1, 2, 3), "b": (1, 2, 3), "c": (0, 3, 3)}
    assert parts.reward_groups(["a", "b", "c"], signatures) == [["a", "b"], ["c"]]
