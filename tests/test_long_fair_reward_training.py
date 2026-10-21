"""Matched repeated exposure, real-return plotting, and action attribution."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_long_fair_reward_training as long_run
import plot_long_fair_reward_training as plots
from diagnose_sagh_flat_reward import compare_records


def plan():
    return json.loads((ROOT / "configs/experiment/long_fair_reward_training_v1.json").read_text())


def test_long_plan_matches_every_unit_in_every_cycle():
    p = plan()
    long_run.check_plan(p)
    assert p["training_steps"] == 32*6144 == 196608
    assert p["training_steps"]//p["steps_per_update"] == 768
    assert p["training_steps"]//(p["evaluate_every_updates"]*p["steps_per_update"])+1 == 33
    assert p["training_steps"] == p["training_cycles"]*p["train_window_count"]*p["max_workflows"]*p["steps_per_training_unit"]


@pytest.mark.parametrize("field,value", [("training_cycles", 31), ("training_steps", 196607),
                                        ("training_cycles", 1), ("evaluate_every_updates", 25),
                                        ("external_reward_shaping", True)])
def test_incomplete_or_unfair_long_protocol_rejected(field, value):
    p = plan()
    p[field] = value
    with pytest.raises(ValueError):
        long_run.check_plan(p)


def test_full_actual_trailing_window_not_cumulative_maximum_or_synthetic_tail():
    np.testing.assert_allclose(plots.trailing_mean([10, -10, -20, 40], 2), [10, 0, -15, 10])
    assert len(plots.trailing_mean([5, 4, 3], 100)) == 3
    assert plots.trailing_mean([5, 4, 3], 100)[-1] == 4


def test_changed_network_preference_does_not_imply_changed_executed_action():
    initial = {"action": 3, "head_actions": {"fast": 0}, "env_probs": [.1, .9],
               "stages": [{"env_argmax": 0, "env_probs": [.8, .2]}]}
    final = {"action": 3, "head_actions": {"fast": 0}, "env_probs": [.2, .8],
             "stages": [{"env_argmax": 1, "env_probs": [.3, .7]}]}
    result = compare_records(initial, final)
    assert result["network_argmax_changed"] and not result["actual_action_changed"]
    assert result["network_probability_l1"] == pytest.approx(1.)


def test_preparer_temporarily_uses_long_validator_without_leaking_base_override():
    base = long_run.fair.check_plan
    wrapper = lambda value: long_run.check_plan_with_base(value, base)
    long_run.fair.check_plan = wrapper
    try:
        wrapper(plan())
        assert long_run.fair.check_plan is wrapper
    finally:
        long_run.fair.check_plan = base
