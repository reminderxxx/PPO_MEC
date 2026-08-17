from __future__ import annotations

import math

import pytest

from src.data.model_catalog.adapter_catalog import AdapterCatalog, RSUAdapterCacheProfile
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import CacheObject, ControlAction, RSUState


def _env(*, cached: list[str], capacity: float, sizes: dict[str, float]) -> VecWorkflowCoreEnv:
    catalog = AdapterCatalog(
        vehicle_base_models=[],
        rsu_adapter_caches=[RSUAdapterCacheProfile("rsu_a", list(cached))],
        adapter_state_bundles=[],
        cache_objects=[
            CacheObject(f"obj:{adapter_id}", adapter_id, size_mb, "test")
            for adapter_id, size_mb in sizes.items()
        ],
    )
    env = VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 1000.0, list(cached))],
        cache_capacity_profile={"enabled": True, "unit": "mb", "capacity_mb": capacity},
    )
    env.reset()
    return env


def _admit(env: VecWorkflowCoreEnv, adapter_id: str) -> dict:
    return env._apply_cache_action(
        ControlAction(cache_action={"operation": "cache", "rsu_id": "rsu_a", "adapter_id": adapter_id}),
        None,
        "node",
        adapter_id,
    )


@pytest.mark.parametrize("capacity", [0.0, -1.0, math.nan, math.inf])
def test_mb_capacity_must_be_finite_positive(capacity: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        _env(cached=[], capacity=capacity, sizes={"a": 1.0})


def test_configuration_unit_and_missing_mb_fail_fast() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        VecWorkflowCoreEnv(cache_capacity_profile={"enabled": True, "unit": "bytes"})
    with pytest.raises(ValueError, match="capacity_mb is required"):
        VecWorkflowCoreEnv(cache_capacity_profile={"enabled": True, "unit": "mb"})


def test_size_resolver_explicit_fallback_and_invalid() -> None:
    env = _env(cached=[], capacity=128.0, sizes={"explicit": 12.5})
    explicit = env.adapter_catalog.resolve_adapter_resident_size_mb("explicit")
    fallback = env.adapter_catalog.resolve_adapter_resident_size_mb("missing")
    assert (explicit.size_mb, explicit.source) == (12.5, "catalog_cache_object")
    assert (fallback.size_mb, fallback.source) == (64.0, "catalog_fallback")
    env.adapter_catalog.cache_objects[0].size_mb = 0.0
    with pytest.raises(ValueError, match="invalid resident"):
        env.adapter_catalog.resolve_adapter_resident_size_mb("explicit")


def test_mb_direct_exact_single_and_multi_victim_admission() -> None:
    direct = _env(cached=["a"], capacity=10.0, sizes={"a": 3.0, "b": 2.0})
    assert _admit(direct, "b")["cache_used_size"] == 5.0

    exact = _env(cached=["a"], capacity=5.0, sizes={"a": 3.0, "b": 2.0})
    assert _admit(exact, "b")["cache_remaining_size"] == 0.0

    single = _env(cached=["a", "b"], capacity=7.0, sizes={"a": 3.0, "b": 2.0, "c": 4.0})
    result = _admit(single, "c")
    assert result["evicted_adapter_ids"] == ["a"]
    assert result["cache_used_size"] == 6.0

    multi = _env(cached=["a", "b", "c"], capacity=10.0, sizes={"a": 3.0, "b": 3.0, "c": 4.0, "d": 6.0})
    result = _admit(multi, "d")
    assert result["evicted_adapter_ids"] == ["a", "b"]
    assert result["eviction_count"] == 2
    assert result["cache_used_size"] == 10.0
    assert result["cache_used_size"] <= result["cache_capacity"]


def test_oversized_and_already_cached_are_atomic_noops() -> None:
    env = _env(cached=["a", "b"], capacity=5.0, sizes={"a": 2.0, "b": 3.0, "huge": 6.0})
    before = list(env.rsu_states[0].cached_adapter_ids)
    oversized = _admit(env, "huge")
    assert oversized["capacity_rejection_reason"] == "object_exceeds_total_capacity"
    assert oversized["eviction_count"] == 0
    assert env.rsu_states[0].cached_adapter_ids == before
    repeated = _admit(env, "a")
    assert repeated["added_new_adapter"] is False
    assert repeated["cache_used_size"] == 5.0


def test_initial_cache_enforcement_uses_sizes_and_lru_order() -> None:
    env = _env(cached=["a", "b", "c"], capacity=7.0, sizes={"a": 2.0, "b": 3.0, "c": 4.0})
    assert env.rsu_states[0].cached_adapter_ids == ["b", "c"]
    snapshot = env._cache_capacity_snapshot("rsu_a")
    assert snapshot["cache_used_size"] == 7.0
    assert snapshot["cache_remaining_size"] == 0.0


def test_disabled_and_slot_legacy_snapshots() -> None:
    disabled = VecWorkflowCoreEnv(cache_capacity_profile={"enabled": False, "unit": "mb"})
    disabled.reset()
    assert disabled._cache_capacity_snapshot("rsu_a")["cache_capacity"] is None
    slot = VecWorkflowCoreEnv(cache_capacity_profile={"enabled": True, "unit": "adapter_slots", "rsu_adapter_slots": 1})
    slot.reset()
    snapshot = slot._cache_capacity_snapshot("rsu_a")
    assert snapshot["cache_capacity_unit"] == "adapter_slots"
    assert snapshot["cache_used_size"] == 1
