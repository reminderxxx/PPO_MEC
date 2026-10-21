"""Pre-execution invariants for the isolated, fair-budget experiment."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fair_reward", ROOT / "scripts/run_fair_reward_training.py")
fair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fair)


def plan():
    return json.loads((ROOT / "configs/experiment/fair_reward_training_v1.json").read_text())


def test_exact_shared_interaction_and_update_budget():
    p = plan()
    fair.check_plan(p)
    assert p["training_steps"] == 6144
    assert p["training_steps"] // p["steps_per_update"] == 24
    assert p["train_window_count"] * p["max_workflows"] == 48
    assert p["training_steps"] // p["steps_per_update"] // p["evaluate_every_updates"] + 1 == 13


@pytest.mark.parametrize("field,value", [("training_steps", 6145), ("steps_per_training_unit", 127),
                                        ("external_reward_shaping", True), ("initialization", "resume")])
def test_unfair_budget_or_reward_rejected(field, value):
    p = plan(); p[field] = value
    with pytest.raises(ValueError):
        fair.check_plan(p)


def test_resource_isolation_required():
    p = plan(); p["resources"]["device"] = "mps"
    with pytest.raises(ValueError):
        fair.check_plan(p)


def test_intervals_use_real_time_and_segment_not_window_id():
    a = {"window_id": "different_id_a", "source_segment_id": "peachtree", "time_index_start": 100, "time_index_end": 200}
    b = {**a, "window_id": "different_id_b", "time_index_start": 180, "time_index_end": 300}
    with pytest.raises(ValueError):
        fair.intervals_disjoint([a, b])
    fair.intervals_disjoint([a, {**b, "time_index_start": 201}])
    fair.intervals_disjoint([a, {**b, "source_segment_id": "lankershim"}])


def test_guard_detects_other_training_but_not_shell_text_or_itself():
    processes = """1 /bin/zsh /bin/zsh -c python train_big.py
2 /path/Python /path/Python scripts/train_algo_pool_real_sample.py --episodes 100000
3 /path/Python /path/Python scripts/run_fair_reward_training.py run
4 /path/Python /path/Python scripts/check_data_ready.py
5 /path/Python /path/Python scripts/benchmark_main_results.py
"""
    assert [p["pid"] for p in fair.foreign_training(processes, 3)] == [2, 5]


def test_frozen_plan_hash_detects_semantic_changes():
    p = plan(); expected = fair.semantic_hash(p)
    assert fair.semantic_hash(json.loads(json.dumps(p, sort_keys=True))) == expected
    p["reward_positive_offset"] = 5
    assert fair.semantic_hash(p) != expected


def test_plateau_is_distinct_from_quality_or_policy_convergence():
    spec = importlib.util.spec_from_file_location("fair_plots", ROOT / "scripts/plot_fair_reward_training.py")
    plots = importlib.util.module_from_spec(spec); spec.loader.exec_module(plots)
    rules = plan()["convergence"]
    result = plots.tail_diagnostic([10.] * 7, [0.] * 7, rules)
    assert result["reward_plateau"] is True
    assert result["completion_floor_met"] is False
    assert result["policy_convergence_proven"] is False
    assert plots.tail_diagnostic([10.] * 3, [1.] * 3, rules)["reward_plateau"] is False
    assert plots.tail_diagnostic([1., 2., 3., 4., 5.], [1.] * 5, rules)["reward_plateau"] is False
