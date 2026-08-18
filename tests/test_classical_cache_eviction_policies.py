from __future__ import annotations

import json
import random

import pytest

from src.envs.core.cache_eviction import (
    AgingLFUEvictionPolicy,
    FIFOEvictionPolicy,
    LFUEvictionPolicy,
    RandomEvictionPolicy,
    build_eviction_policy,
)


def plan(policy, residents, required=1.0, sizes=None, protected="incoming"):
    return policy.plan_victims(rsu_id="r", resident_ids=residents, resident_sizes=sizes or {x: 1.0 for x in residents}, required_free_capacity=required, protected_object_id=protected, capacity_unit="adapter_slots", current_step=20)


def test_factory_registers_exact_suite_and_validates_config() -> None:
    assert [build_eviction_policy(name, seed=3).policy_name for name in ("lru", "fifo", "lfu", "aging_lfu", "random")] == ["lru", "fifo", "lfu", "aging_lfu", "random"]
    with pytest.raises(ValueError, match="requires an explicit seed"):
        build_eviction_policy("random")
    with pytest.raises(ValueError, match="does not accept config"):
        build_eviction_policy("fifo", foo=1)


def test_fifo_hit_does_not_touch_and_readmission_is_newest() -> None:
    fifo = FIFOEvictionPolicy(); fifo.reset(rsu_id="r", initial_resident_ids=["a", "b"])
    fifo.on_hit(rsu_id="r", object_id="a", current_step=2)
    assert plan(fifo, ["a", "b"]).ordered_victim_ids == ["a"]
    fifo.on_eviction(rsu_id="r", object_id="a", current_step=3)
    fifo.on_admission(rsu_id="r", object_id="a", current_step=4)
    assert plan(fifo, ["a", "b"]).ordered_victim_ids == ["b"]
    assert plan(fifo, ["a", "b"], required=2).ordered_victim_ids == ["b", "a"]


def test_lfu_frequency_tie_break_and_no_aging() -> None:
    lfu = LFUEvictionPolicy(); lfu.reset(rsu_id="r", initial_resident_ids=["b", "a"])
    assert plan(lfu, ["b", "a"]).ordered_candidates == ["a", "b"]
    lfu.on_hit(rsu_id="r", object_id="a", current_step=1)
    assert plan(lfu, ["a", "b"]).ordered_victim_ids == ["b"]
    for step in range(2, 20): lfu.on_hit(rsu_id="r", object_id="a", current_step=step)
    assert lfu.export_state()["rsus"]["r"]["resident_metadata"]["a"]["frequency"] == 19


def test_aging_lfu_exact_decay_and_rsu_isolation() -> None:
    policy = AgingLFUEvictionPolicy(aging_interval=2, aging_factor=0.5)
    policy.reset(rsu_id="r", initial_resident_ids=["old", "new"])
    policy.reset(rsu_id="other", initial_resident_ids=["x"])
    policy.on_hit(rsu_id="r", object_id="old", current_step=1)
    policy.on_hit(rsu_id="r", object_id="old", current_step=2)
    state = policy.export_state()
    assert state["rsus"]["r"]["resident_metadata"]["old"]["frequency"] == 1
    assert state["rsu_clocks"] == {"other": 0, "r": 2}
    assert plan(policy, ["old", "new"]).ordered_victim_ids == ["new"]
    with pytest.raises(ValueError): AgingLFUEvictionPolicy(aging_interval=0)
    with pytest.raises(ValueError): AgingLFUEvictionPolicy(aging_interval=1.5)
    with pytest.raises(ValueError): AgingLFUEvictionPolicy(aging_factor=1.0)


def test_random_private_seed_reproducibility_reset_and_protection() -> None:
    global_before = random.getstate()
    residents = list("abcdef")
    p1, p2, p3 = RandomEvictionPolicy(seed=7), RandomEvictionPolicy(seed=7), RandomEvictionPolicy(seed=8)
    for policy in (p1, p2, p3): policy.reset(rsu_id="r", initial_resident_ids=residents)
    a = plan(p1, residents, required=3, protected="a")
    b = plan(p2, residents, required=3, protected="a")
    c = plan(p3, residents, required=3, protected="a")
    assert a.to_dict() == b.to_dict()
    assert a.ordered_victim_ids != c.ordered_victim_ids
    assert len(a.ordered_victim_ids) == len(set(a.ordered_victim_ids)) and "a" not in a.ordered_victim_ids
    assert random.getstate() == global_before
    p1.reset(); p1.reset(rsu_id="r", initial_resident_ids=residents)
    assert plan(p1, residents, required=3, protected="a").to_dict() == a.to_dict()
    json.dumps(p1.export_state())


@pytest.mark.parametrize("policy", [FIFOEvictionPolicy(), LFUEvictionPolicy(), AgingLFUEvictionPolicy()])
def test_mb_multi_victim_and_detached_state(policy) -> None:
    policy.reset(rsu_id="r", initial_resident_ids=["a", "b", "c"])
    result = policy.plan_victims(rsu_id="r", resident_ids=["a", "b", "c"], resident_sizes={"a": 2.0, "b": 3.0, "c": 4.0}, required_free_capacity=5.0, protected_object_id=None, capacity_unit="mb", current_step=1)
    assert result.sufficient and len(result.ordered_victim_ids) == 2
    state = policy.export_state(); state["rsus"].clear()
    assert policy.export_state()["rsus"]
