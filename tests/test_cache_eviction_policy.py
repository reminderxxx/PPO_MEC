from __future__ import annotations

import json

import pytest

from src.envs.core.cache_eviction import LRUEvictionPolicy, build_eviction_policy


def _reset(policy: LRUEvictionPolicy, rsu_id: str, residents: list[str]) -> None:
    policy.reset(rsu_id=rsu_id, initial_resident_ids=residents, current_step=0)


def _plan(
    policy: LRUEvictionPolicy,
    *,
    rsu_id: str = "rsu_a",
    residents: list[str] | None = None,
    sizes: dict[str, float] | None = None,
    required: float = 1.0,
    protected: str | None = "incoming",
    unit: str = "adapter_slots",
):
    residents = residents or []
    sizes = sizes or {item: 1.0 for item in residents}
    return policy.plan_victims(
        rsu_id=rsu_id,
        resident_ids=residents,
        resident_sizes=sizes,
        required_free_capacity=required,
        protected_object_id=protected,
        capacity_unit=unit,
        current_step=7,
    )


def test_identity_factory_and_unknown_policy() -> None:
    policy = build_eviction_policy("LRU", seed=17)
    assert policy.policy_name == "lru"
    assert policy.policy_version == "1.0.0"
    assert policy.deterministic is True
    assert policy.requires_seed is False
    assert policy.capacity_units_supported == frozenset({"adapter_slots", "mb"})
    assert policy.export_state()["seed_consumed"] is False
    with pytest.raises(ValueError, match="registered policies: lru"):
        build_eviction_policy("fifo")


def test_reset_initial_order_rsu_isolation_and_episode_isolation() -> None:
    policy = LRUEvictionPolicy()
    _reset(policy, "rsu_a", ["z", "a"])
    _reset(policy, "rsu_b", ["b"])
    state = policy.export_state()
    assert state["rsus"]["rsu_a"]["lru_order_oldest_first"] == ["z", "a"]
    assert state["rsus"]["rsu_b"]["lru_order_oldest_first"] == ["b"]
    policy.on_hit(rsu_id="rsu_a", object_id="z", current_step=2)
    assert policy.export_state()["rsus"]["rsu_b"]["lru_order_oldest_first"] == ["b"]
    policy.reset()
    _reset(policy, "rsu_a", ["fresh"])
    assert set(policy.export_state()["rsus"]) == {"rsu_a"}


def test_admission_and_hit_update_recency_with_g03_tie_break() -> None:
    policy = LRUEvictionPolicy()
    _reset(policy, "rsu_a", ["old", "newer"])
    policy.on_hit(rsu_id="rsu_a", object_id="old", current_step=3)
    policy.on_admission(rsu_id="rsu_a", object_id="z", current_step=4)
    assert _plan(policy, residents=["old", "newer", "z"]).ordered_candidates == ["newer", "old", "z"]
    policy.on_hit(rsu_id="rsu_a", object_id="old", current_step=5)
    policy.on_hit(rsu_id="rsu_a", object_id="z", current_step=5)
    assert _plan(policy, residents=["old", "z"]).ordered_candidates == ["old", "z"]


@pytest.mark.parametrize("unit", ["adapter_slots", "mb"])
def test_single_multi_exact_minimum_prefix_and_determinism(unit: str) -> None:
    policy = LRUEvictionPolicy()
    residents = ["a", "b", "c"]
    sizes = {"a": 3.0, "b": 3.0, "c": 4.0} if unit == "mb" else {item: 1.0 for item in residents}
    _reset(policy, "rsu_a", residents)
    required = 6.0 if unit == "mb" else 2.0
    first = _plan(policy, residents=residents, sizes=sizes, required=required, unit=unit)
    second = _plan(policy, residents=list(residents), sizes=dict(sizes), required=required, unit=unit)
    assert first.ordered_victim_ids == ["a", "b"]
    assert first.cumulative_freed_capacity == required
    assert first.sufficient is True
    assert first.to_dict() == second.to_dict()
    assert residents == ["a", "b", "c"]


def test_protected_empty_and_insufficient_plans() -> None:
    policy = LRUEvictionPolicy()
    _reset(policy, "rsu_a", ["incoming", "a"])
    protected = _plan(
        policy,
        residents=["incoming", "a"],
        sizes={"incoming": 1.0, "a": 1.0},
        required=1.0,
    )
    assert protected.ordered_victim_ids == ["a"]
    empty = _plan(policy, residents=[], sizes={}, required=1.0)
    assert empty.sufficient is False
    insufficient = _plan(policy, residents=["a"], sizes={"a": 1.0}, required=2.0)
    assert insufficient.ordered_victim_ids == ["a"]
    assert insufficient.sufficient is False


def test_plan_is_read_only_and_eviction_cleanup_requires_callback() -> None:
    policy = LRUEvictionPolicy()
    _reset(policy, "rsu_a", ["a", "b"])
    before = policy.export_state()
    plan = _plan(policy, residents=["a", "b"])
    assert policy.export_state() == before
    policy.on_eviction(rsu_id="rsu_a", object_id=plan.ordered_victim_ids[0], current_step=1)
    state = policy.export_state()
    assert "a" not in state["rsus"]["rsu_a"]["resident_metadata"]
    json.dumps(state)


def test_no_eviction_required_and_invalid_inputs() -> None:
    policy = LRUEvictionPolicy()
    _reset(policy, "rsu_a", ["a"])
    plan = _plan(policy, residents=["a"], required=0.0)
    assert plan.ordered_victim_ids == []
    assert plan.selection_reason == "no_eviction_required"
    with pytest.raises(ValueError, match="does not support"):
        _plan(policy, residents=["a"], unit="bytes")
    with pytest.raises(ValueError, match="invalid resident size"):
        _plan(policy, residents=["a"], sizes={"a": 0.0})
