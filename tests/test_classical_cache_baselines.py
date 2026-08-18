from __future__ import annotations

import json

import pytest

from src.agents.registry import build_agent, checkpoint_required_agents, get_algo_spec, list_evaluable_agents, validate_agent_eviction_binding
from src.evaluators.main_results_support import aggregate_rows, build_pairwise_comparison, build_win_tie_loss_summary


NAMES = ["reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"]


def semantic(cached=False):
    return {"current_workflow_node": {"required_adapter": "a"}, "primary_vehicle_id": "v", "vehicles": [{"vehicle_id": "v", "associated_rsu_id": "r"}], "rsus": [{"rsu_id": "r", "cached_adapter_ids": ["a"] if cached else []}]}


def test_registry_metadata_identity_and_checkpoint_free() -> None:
    assert set(NAMES) <= set(list_evaluable_agents())
    assert not (set(NAMES) & checkpoint_required_agents())
    for name in NAMES:
        spec = get_algo_spec(name)
        assert spec["family"] == "classical_cache"
        assert spec["control_policy"] == "reactive_current_rsu_admission_v1"
        assert spec["baseline_scope"] == "reactive placement/admission + selected eviction policy"
        assert spec["uses_prediction"] is False and spec["uses_learning"] is False


def test_all_baselines_share_exact_reactive_actions_but_keep_identity() -> None:
    for cached, expected in ((False, 0), (True, 3)):
        outputs = []
        for name in NAMES:
            agent = build_agent(name, random_seed=7)
            action, info = agent.act(None, {"semantic_state": semantic(cached), "action_mask": [True] * 5})
            outputs.append(action)
            assert info["policy_type"] == name
            assert info["required_eviction_policy"] == get_algo_spec(name)["required_eviction_policy"]
        assert outputs == [expected] * 5


def test_binding_mismatch_fails_and_random_seed_is_run_seed() -> None:
    assert validate_agent_eviction_binding("reactive_lfu", {"eviction_policy": "lfu"})["eviction_policy"] == "lfu"
    with pytest.raises(ValueError, match="requires eviction_policy=lfu"):
        validate_agent_eviction_binding("reactive_lfu", {"eviction_policy": "lru"})
    bound = validate_agent_eviction_binding("reactive_random", {"eviction_policy": "random"}, run_seed=29)
    assert bound["eviction_policy_seed"] == 29
    with pytest.raises(ValueError, match="must equal"):
        validate_agent_eviction_binding("reactive_random", {"eviction_policy": "random", "eviction_policy_seed": 7}, run_seed=29)


def test_reactive_greedy_historical_identity_unchanged() -> None:
    agent = build_agent("reactive_greedy")
    assert agent.agent_name == "reactive_greedy"
    assert "required_eviction_policy" not in get_algo_spec("reactive_greedy")


def _metric(aggregate, agent_name, metric_name):
    return aggregate[agent_name]["metrics"][metric_name]


@pytest.mark.parametrize("row", [{"cache_used_size": None}, {}])
def test_benchmark_aggregation_preserves_unavailable_capacity_snapshot(row) -> None:
    aggregate = aggregate_rows(
        [{"agent_name": "reactive_fifo", **row}],
        group_keys=["agent_name"],
        metrics=["cache_used_size"],
    )
    assert _metric(aggregate, "reactive_fifo", "cache_used_size") == {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "available_count": 0,
        "unavailable_count": 1,
    }


def test_benchmark_aggregation_distinguishes_real_zero_and_mixed_availability() -> None:
    real_zero = aggregate_rows(
        [{"agent_name": "reactive_fifo", "cache_used_size": 0.0}],
        group_keys=["agent_name"],
        metrics=["cache_used_size"],
    )
    assert _metric(real_zero, "reactive_fifo", "cache_used_size") == {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "available_count": 1,
        "unavailable_count": 0,
    }

    mixed = aggregate_rows(
        [
            {"agent_name": "reactive_fifo", "cache_used_size": None},
            {"agent_name": "reactive_fifo", "cache_used_size": 6.0},
        ],
        group_keys=["agent_name"],
        metrics=["cache_used_size"],
    )
    assert _metric(mixed, "reactive_fifo", "cache_used_size") == {
        "mean": 6.0,
        "std": 0.0,
        "min": 6.0,
        "max": 6.0,
        "available_count": 1,
        "unavailable_count": 1,
    }


def test_all_capacity_snapshot_metrics_keep_null_and_json_round_trip() -> None:
    nullable_metrics = [
        "cache_capacity",
        "cache_used_size",
        "cache_remaining_size",
        "cache_occupancy_rate",
    ]
    aggregate = aggregate_rows(
        [{"agent_name": "reactive_lru", "cache_capacity_enabled": 0.0}],
        group_keys=["agent_name"],
        metrics=["cache_capacity_enabled", *nullable_metrics],
    )
    restored = json.loads(json.dumps(aggregate, allow_nan=False))
    assert _metric(restored, "reactive_lru", "cache_capacity_enabled")["mean"] == 0.0
    assert all(_metric(restored, "reactive_lru", name)["mean"] is None for name in nullable_metrics)


def test_normal_metric_aggregation_and_all_classical_baseline_groups_remain_compatible() -> None:
    rows = [
        {"agent_name": name, "total_reward": value, "cache_capacity_enabled": 0.0}
        for value, name in enumerate(NAMES, start=1)
    ]
    aggregate = aggregate_rows(
        rows,
        group_keys=["agent_name"],
        metrics=["total_reward", "cache_capacity_enabled", "cache_used_size"],
    )
    assert set(aggregate) == set(NAMES)
    for value, name in enumerate(NAMES, start=1):
        assert _metric(aggregate, name, "total_reward")["mean"] == float(value)
        assert _metric(aggregate, name, "cache_capacity_enabled")["mean"] == 0.0
        assert _metric(aggregate, name, "cache_used_size")["mean"] is None


def test_summary_consumers_accept_nullable_aggregate_metrics() -> None:
    rows = [
        {"window_id": "w1", "workflow_id": "wf1", "agent_name": "sa_ghmappo", "total_reward": 2.0},
        {"window_id": "w1", "workflow_id": "wf1", "agent_name": "reactive_fifo", "total_reward": 1.0},
    ]
    metrics = ["total_reward", "cache_used_size"]
    by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=metrics)
    comparison = build_pairwise_comparison(by_agent, baseline_agent="sa_ghmappo", metrics=metrics)
    assert comparison["reactive_fifo"]["delta_vs_baseline"]["cache_used_size"] is None
    assert comparison["reactive_fifo"]["result_by_metric"]["cache_used_size"] == "unavailable"

    by_window = aggregate_rows(rows, group_keys=["window_id", "agent_name"], metrics=metrics)
    by_workflow = aggregate_rows(rows, group_keys=["workflow_id", "agent_name"], metrics=metrics)
    summary = build_win_tie_loss_summary(by_window, by_workflow, metrics=metrics)
    assert summary["window_level"]["per_agent_metric_counts"]["reactive_fifo"]["cache_used_size"]["unavailable"] == 1
