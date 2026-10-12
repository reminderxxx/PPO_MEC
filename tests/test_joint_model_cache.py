"""Physical invariants for controlled joint cache admission and transport."""

import pytest

from src.envs.core.joint_model_cache import JointModelCache, ModelCacheObject


def _objects():
    return [
        ModelCacheObject("base:b", "base_model", 100, 100, base_model_id="b"),
        ModelCacheObject("adapter:a", "adapter", 30, 30, ("base:b",), adapter_id="a"),
        ModelCacheObject("adapter:c", "adapter", 40, 40, ("base:b",), adapter_id="c"),
        ModelCacheObject("base:d", "base_model", 80, 80, base_model_id="d"),
        ModelCacheObject("adapter:e", "adapter", 20, 20, ("base:d",), adapter_id="e"),
    ]


def _cache(capacity=170, slots=1, rate=100):
    return JointModelCache(["r"], capacity_bytes=capacity, objects=_objects(),
                           link_slots=slots, transfer_bytes_per_frame=rate)


def _load(cache, adapter="a", base="b"):
    cache.begin_frame()
    result = cache.request_adapter("r", adapter, base, available_link_slots=1)
    cache.advance()
    for _ in range(20):
        if cache.is_ready("r", adapter, base):
            return result
        cache.begin_frame()
        cache.advance()
    raise AssertionError("admitted object failed to finish")


def test_base_and_adapter_reserve_capacity_but_become_ready_in_dependency_order():
    cache = _cache()
    assert cache.begin_frame() == {"r": 0}
    result = cache.request_adapter("r", "a", "b", available_link_slots=1)
    assert result["link_slots_consumed"] == 1
    before = cache.snapshot("r")
    assert before["occupied_bytes"] == before["reserved_bytes"] == 130
    assert before["resident_bytes"] == 0
    assert not before["dependency_ready"]["adapter:a"]
    assert not cache.is_ready("r", "a", "b")
    first = cache.advance()
    assert [(row["object_id"], row["transferred_bytes"]) for row in first] == [("base:b", 100)]
    assert cache.snapshot("r")["ready_base_model_ids"] == ["b"]
    assert not cache.is_ready("r", "a", "b")
    assert cache.begin_frame() == {"r": 1}
    last = cache.advance()
    assert last[0]["completed"] and last[0]["transferred_bytes"] == 30
    assert cache.is_ready("r", "a", "b")
    assert cache.snapshot("r")["transferred_bytes"] == 130


def test_shared_base_deduplicates_pending_transfer_and_residency():
    cache = _cache(slots=2)
    cache.begin_frame()
    assert cache.request_adapter("r", "a", "b")["link_slots_consumed"] == 1
    assert cache.request_adapter("r", "a", "b")["status"] == "already_loading"
    second = cache.request_adapter("r", "c", "b")
    assert second["admitted_object_ids"] == ["adapter:c"]
    assert second["link_slots_consumed"] == 0
    assert cache.snapshot("r")["occupied_bytes"] == 170
    cache.advance()
    assert cache.begin_frame() == {"r": 2}
    cache.advance()
    state = cache.snapshot("r")
    assert set(state["cached_adapter_ids"]) == {"a", "c"}
    assert state["transferred_bytes"] == state["resident_bytes"] == 170


def test_capacity_infeasible_or_blocked_never_partially_evicts_or_admits():
    impossible = _cache(capacity=129)
    impossible.begin_frame()
    assert impossible.request_adapter("r", "a", "b")["status"] == "capacity_infeasible"
    assert impossible.snapshot("r")["occupied_bytes"] == 0
    impossible.advance()
    cache = _cache(capacity=170, rate=10)
    cache.begin_frame()
    cache.request_adapter("r", "a", "b")
    before = cache.snapshot("r")
    assert cache.request_adapter("r", "e", "d")["status"] == "capacity_blocked"
    assert cache.snapshot("r") == before
    cache.advance()


def test_automatic_lru_eviction_is_dependency_safe_and_leaves_pending_base_intact():
    cache = _cache(capacity=170)
    _load(cache)
    _load(cache, "c")
    cache.touch("r", "a", "b")
    cache.begin_frame()
    result = cache.request_adapter("r", "e", "d")
    assert result["evicted_object_ids"] == ["adapter:c", "adapter:a", "base:b"]
    assert cache.snapshot("r")["cached_adapter_ids"] == []
    cache.advance()
    cache.begin_frame()
    cache.advance()
    assert cache.is_ready("r", "e", "d")
    assert cache.snapshot("r")["evicted_bytes"] == 170
    assert cache.snapshot("r")["occupied_bytes"] == 100


def test_shared_link_budget_and_frame_ledger_include_final_partial_transfer():
    cache = _cache(capacity=270, slots=2, rate=60)
    cache.begin_frame()
    before = cache.snapshot("r")
    assert cache.request_adapter("r", "a", "b", available_link_slots=0)["status"] == "link_queue"
    assert cache.snapshot("r") == before
    cache.request_adapter("r", "a", "b", available_link_slots=1)
    assert sum(event["transferred_bytes"] for event in cache.advance()) == 60
    assert cache.begin_frame() == {"r": 1}
    cache.request_adapter("r", "e", "d", available_link_slots=0)
    final_base = cache.advance()
    assert len(final_base) == 1 and final_base[0]["transferred_bytes"] == 40
    assert final_base[0]["completed"]
    assert cache.snapshot("r")["last_frame_transfer_bytes"] == 40
    cache.begin_frame()
    cache.advance()
    assert cache.snapshot("r")["transferred_bytes"] == 130
    assert cache.snapshot("r")["occupied_bytes"] == 130


def test_pending_adapter_protects_its_ready_base_against_replacement():
    cache = _cache(capacity=170, rate=100)
    cache.begin_frame()
    cache.request_adapter("r", "a", "b")
    cache.advance()
    cache.begin_frame()
    before = cache.snapshot("r")
    assert cache.request_adapter("r", "e", "d")["status"] == "capacity_blocked"
    assert cache.snapshot("r") == before
    cache.advance()
    assert cache.is_ready("r", "a", "b")


def test_invalid_catalog_and_incompatible_base_are_rejected():
    with pytest.raises(ValueError, match="missing or invalid"):
        JointModelCache(["r"], capacity_bytes=100, objects=_objects()[1:2], link_slots=1,
                        transfer_bytes_per_frame=100)
    cache = _cache()
    cache.begin_frame()
    assert cache.request_adapter("r", "a", "d")["status"] == "unknown_or_incompatible_model"
    assert cache.snapshot("r")["occupied_bytes"] == 0
    cache.advance()
    with pytest.raises(ValueError, match="before adapter"):
        cache.touch("r", "a", "b")


def test_frames_cannot_advance_twice_or_admit_after_advance():
    cache = _cache()
    with pytest.raises(RuntimeError, match="begin_frame"):
        cache.advance()
    cache.begin_frame()
    with pytest.raises(RuntimeError, match="previous cache frame"):
        cache.begin_frame()
    cache.advance()
    with pytest.raises(RuntimeError, match="begin_frame"):
        cache.request_adapter("r", "a", "b")
