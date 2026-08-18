from __future__ import annotations

import pytest

from src.agents.registry import build_agent, checkpoint_required_agents, get_algo_spec, list_evaluable_agents, validate_agent_eviction_binding
from src.evaluators.main_results_support import aggregate_rows


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


def test_benchmark_aggregation_accepts_unavailable_capacity_snapshot() -> None:
    aggregate = aggregate_rows(
        [{"agent_name": "reactive_fifo", "cache_used_size": None}],
        group_keys=["agent_name"],
        metrics=["cache_used_size"],
    )
    assert aggregate["reactive_fifo"]["metrics"]["cache_used_size"]["mean"] == 0.0
