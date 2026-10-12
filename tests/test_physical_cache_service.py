"""Physical typed-cache admission, dependency, and cancellation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.physical_cache_service import MIB, PhysicalTypedCache
from src.envs.core.physical_runtime import PhysicalRuntime


CATALOG_PATH = Path(__file__).resolve().parents[1] / "src/data/model_catalog/typed_model_cache_controlled.json"


def setup(capacity_mib: float = 320.0, *, initial: list[str] | None = None):
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    cache = PhysicalTypedCache(
        catalog, ["rsu-a"], {"rsu-a": capacity_mib * MIB},
        initial_residents={"rsu-a": initial or []},
    )
    clock = PhysicalRuntime(0, {"backhaul": 10 * MIB}, {}, {})
    submit = lambda size, metadata: clock.submit_transfer("backhaul", size, "model", metadata)
    return cache, clock, submit


def objects(cache: PhysicalTypedCache) -> dict[str, dict]:
    return {row["object_id"]: row for row in cache.snapshot()["rsus"]["rsu-a"]["objects"]}


def test_reservation_loading_dependency_and_no_double_count_on_promotion() -> None:
    cache, clock, submit = setup()
    result = cache.request("rsu-a", "adapter_tracking", 0, submit)
    assert result["status"] == "loading"
    assert result["missing_object_ids"] == ["base:veh_base_v1", "adapter:tracking"]
    assert len(result["job_ids"]) == 2
    before = cache.snapshot()["rsus"]["rsu-a"]["used_bytes"]
    assert before == (180 + 56) * MIB
    assert not cache.is_ready("rsu-a", "adapter_tracking")

    # The adapter is smaller, so it finishes first but cannot become ready
    # until the base dependency arrives.
    first_event = clock.advance_to(9.6)[0]
    assert first_event["metadata"]["object_id"] == "adapter:tracking"
    change = cache.on_transfer_complete(first_event["job_id"], first_event["time_s"])
    assert change["status"] == "loaded_waiting_dependency"
    assert not cache.is_ready("rsu-a", "adapter_tracking")
    assert cache.snapshot()["rsus"]["rsu-a"]["used_bytes"] == before
    second_event = clock.advance_to(20.8)[0]
    change = cache.on_transfer_complete(second_event["job_id"], second_event["time_s"])
    assert set(change["became_resident_object_ids"]) == {"base:veh_base_v1", "adapter:tracking"}
    assert cache.is_ready("rsu-a", "adapter_tracking")
    assert cache.snapshot()["rsus"]["rsu-a"]["used_bytes"] == before


def test_request_while_loading_reuses_jobs_and_space() -> None:
    cache, clock, submit = setup()
    first = cache.request("rsu-a", "adapter_tracking", 0, submit)
    second = cache.request("rsu-a", "adapter_tracking", 0, submit)
    assert second["status"] == "loading"
    assert second["job_ids"] == []
    assert second["missing_object_ids"] == []
    assert cache.snapshot()["rsus"]["rsu-a"]["used_bytes"] == 236 * MIB
    assert len(clock.snapshot()["transfers"]) == len(first["job_ids"]) == 2


def test_capacity_rejection_is_atomic_and_dependency_safe_eviction() -> None:
    cache, clock, submit = setup(236, initial=["base:veh_base_v1", "adapter:tracking"])
    before = cache.snapshot()
    rejected = cache.request("rsu-a", "adapter_reasoning", 0, submit)
    assert rejected["status"] == "rejected"
    assert cache.snapshot() == before
    assert clock.snapshot()["transfers"] == []

    # In a larger cache, an old adapter can be evicted while the pinned base
    # remains. The new adapter reserves its bytes immediately.
    cache, clock, submit = setup(260, initial=["base:veh_base_v1", "adapter:tracking"])
    admitted = cache.request("rsu-a", "adapter_perception", 0, submit)
    assert admitted["status"] == "loading"
    assert admitted["evicted_object_ids"] == ["adapter:tracking"]
    assert set(objects(cache)) == {"base:veh_base_v1", "adapter:perception"}
    assert cache.snapshot()["rsus"]["rsu-a"]["used_bytes"] == 220 * MIB


def test_pin_prevents_eviction_until_unpinned() -> None:
    cache, clock, submit = setup(260, initial=["base:veh_base_v1", "adapter:tracking"])
    cache.pin("rsu-a", "adapter_tracking")
    assert objects(cache)["adapter:tracking"]["pin_count"] == 1
    rejected = cache.request("rsu-a", "adapter_perception", 0, submit)
    assert rejected["status"] == "rejected"
    cache.unpin("rsu-a", "adapter_tracking")
    accepted = cache.request("rsu-a", "adapter_perception", 0, submit)
    assert accepted["status"] == "loading"
    with pytest.raises(ValueError):
        cache.unpin("rsu-a", "adapter_tracking")


def test_cancel_base_cascades_to_adapter_and_keeps_transfer_cost_events() -> None:
    cache, clock, submit = setup()
    cache.request("rsu-a", "adapter_tracking", 0, submit)
    base_job = objects(cache)["base:veh_base_v1"]["job_id"]
    clock.advance_to(3.0)
    result = cache.cancel(base_job, clock.cancel_transfer)
    assert result["status"] == "cancelled"
    assert set(result["released_object_ids"]) == {"base:veh_base_v1", "adapter:tracking"}
    assert sum(event["completed_bytes"] for event in result["cancelled_events"]) == pytest.approx(30 * MIB)
    assert result["used_bytes"] == 0
    assert clock.snapshot()["transfers"] == []
    assert not cache.is_ready("rsu-a", "adapter_tracking")


def test_cancel_adapter_keeps_shared_base_transfer() -> None:
    cache, clock, submit = setup()
    cache.request("rsu-a", "adapter_tracking", 0, submit)
    adapter_job = objects(cache)["adapter:tracking"]["job_id"]
    result = cache.cancel(adapter_job, clock.cancel_transfer)
    assert result["released_object_ids"] == ["adapter:tracking"]
    assert set(objects(cache)) == {"base:veh_base_v1"}
    assert len(clock.snapshot()["transfers"]) == 1
    base_event = clock.advance_to(16.0)[0]
    cache.on_transfer_complete(base_event["job_id"], base_event["time_s"])
    assert objects(cache)["base:veh_base_v1"]["state"] == "resident"
    assert cache.snapshot()["rsus"]["rsu-a"]["used_bytes"] == 180 * MIB


def test_invalid_initial_dependency_and_capacity_are_rejected() -> None:
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    with pytest.raises(ValueError, match="missing dependency"):
        PhysicalTypedCache(
            catalog, ["rsu-a"], {"rsu-a": 320 * MIB},
            initial_residents={"rsu-a": ["adapter:tracking"]},
        )
    with pytest.raises(ValueError, match="exceed capacity"):
        PhysicalTypedCache(
            catalog, ["rsu-a"], {"rsu-a": 100 * MIB},
            initial_residents={"rsu-a": ["base:veh_base_v1"]},
        )
