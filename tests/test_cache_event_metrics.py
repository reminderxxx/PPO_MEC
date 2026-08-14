from __future__ import annotations

from copy import deepcopy

import pytest

from src.metrics.cache_event_metrics import (
    audit_cache_event_telemetry,
    reduce_cache_event_summary,
    reduce_cache_events,
)


def event(event_id: str, **updates):
    payload = {
        "event_id": event_id, "event_schema_version": "1.0.0", "event_type": "request",
        "time_index": 1, "episode_step_index": 1, "vehicle_id": "v1", "workflow_id": "w1",
        "node_id": "n1", "object_id": "obj:a", "adapter_id": "a", "object_type": "adapter",
        "size_mb": 3.0, "request_rsu_id": "r1", "selected_target_rsu_id": "r1",
        "served_rsu_id": "r1", "predicted_next_rsu_id": "r2", "predicted_handoff_target_rsu_id": "r2",
        "hit_source": "current_rsu", "cache_lookup_performed": True, "cache_hit": True,
        "was_cached_before": True, "admission_requested": False, "admission_added": False,
        "admission_reason": "not_requested", "cache_target_rsu_id": None,
        "eviction_occurred": False, "eviction_policy": "not_applicable", "evicted_object_id": None,
        "evicted_adapter_id": None, "eviction_reason": "not_occurred",
        "adapter_transfer_size_mb": 0.0, "state_migration_size_mb": 0.0,
        "transfer_source": "catalog_cache_object", "migration_requested": False,
        "migration_realized": False, "cache_capacity_enabled": False,
        "cache_capacity_unit": "adapter_slots", "cache_capacity_before": None,
        "cache_used_before": None, "cache_remaining_before": None, "cache_capacity_after": None,
        "cache_used_after": None, "cache_remaining_after": None, "action_id": 0,
        "action_name": "no_op", "cache_strategy": "none", "offload_mode": "rsu",
        "service_success": True, "stall_occurred": False, "handoff_event_count": 0,
    }
    payload.update(updates)
    return payload


def mixed_trace():
    return [
        event("e1"),
        event("e2", hit_source="target_rsu", selected_target_rsu_id="r2", served_rsu_id="r2"),
        event("e3", hit_source="neighbor_rsu", selected_target_rsu_id="r3", served_rsu_id="r3"),
        event("e4", hit_source="vehicle_local", cache_lookup_performed=False, served_rsu_id=None,
              offload_mode="vehicle", size_mb=4.0),
        event("e5", hit_source="cloud", cache_hit=False, cache_lookup_performed=False, served_rsu_id=None,
              selected_target_rsu_id=None, offload_mode="cloud", size_mb=5.0),
        event("e6", hit_source="unserved", cache_hit=False, served_rsu_id=None, service_success=False,
              stall_occurred=True, size_mb=6.0),
        event("e7", admission_requested=True, admission_added=True, was_cached_before=False,
              admission_reason="reactive_cache_fill", cache_target_rsu_id="r1",
              adapter_transfer_size_mb=7.0, size_mb=7.0),
        event("e8", admission_requested=True, admission_added=False, admission_reason="already_cached",
              cache_target_rsu_id="r1", size_mb=8.0),
        event("e9", admission_requested=True, admission_added=True, was_cached_before=False,
              admission_reason="capacity_fill", cache_target_rsu_id="r1", eviction_occurred=True,
              eviction_policy="lru", evicted_object_id="obj:old", evicted_adapter_id="old",
              eviction_reason="capacity_limit", adapter_transfer_size_mb=9.0, size_mb=9.0,
              cache_capacity_enabled=True, cache_capacity_before=1.0, cache_used_before=1.0,
              cache_remaining_before=0.0, cache_capacity_after=1.0, cache_used_after=1.0,
              cache_remaining_after=0.0),
        event("e10", migration_requested=True, migration_realized=True,
              state_migration_size_mb=1.5, transfer_source="catalog_fallback", size_mb=10.0),
        event("e11", migration_requested=True, migration_realized=False,
              state_migration_size_mb=2.5, transfer_source="catalog_fallback", size_mb=11.0),
        event("e12", event_type="not_applicable", node_id=None, object_id=None, adapter_id=None,
              object_type="not_applicable", size_mb=None, request_rsu_id=None, selected_target_rsu_id=None,
              served_rsu_id=None, hit_source="not_applicable", cache_lookup_performed=False,
              cache_hit=False, was_cached_before=False, offload_mode="none", service_success=False),
    ]


def test_empty_and_only_not_applicable_traces() -> None:
    empty = reduce_cache_events([])
    assert empty.availability == "available"
    assert empty.total_event_count == 0
    assert empty.cache_hit_rate is None
    only_na = reduce_cache_events([mixed_trace()[-1]])
    assert only_na.request_event_count == 0
    assert only_na.not_applicable_event_count == 1
    assert only_na.cache_miss_count == 0


def test_consumer_safe_1_x_optional_field_is_accepted() -> None:
    payload = event("future-minor", event_schema_version="1.1.0", optional_note="ignored")
    result = reduce_cache_events([payload], schema_version="1.1.0")
    assert result.cache_event_schema_version == "1.1.0"
    assert result.request_event_count == 1


def test_hand_calculated_mixed_trace() -> None:
    result = reduce_cache_events(mixed_trace()).to_dict()
    assert (result["request_event_count"], result["not_applicable_event_count"], result["total_event_count"]) == (11, 1, 12)
    assert (result["cache_hit_count"], result["cache_miss_count"], result["cache_hit_rate"]) == (9, 2, 0.818182)
    assert [result[key] for key in ("vehicle_local_hit_count", "current_rsu_hit_count", "target_rsu_hit_count", "neighbor_rsu_hit_count", "cloud_served_count", "unserved_count", "not_applicable_hit_source_count")] == [1, 6, 1, 1, 1, 1, 1]
    assert (result["admission_request_count"], result["admission_added_count"], result["admission_noop_or_existing_count"]) == (3, 2, 1)
    assert result["admission_reason_counts"] == {"already_cached": 1, "capacity_fill": 1, "reactive_cache_fill": 1}
    assert result["eviction_count"] == result["evicted_object_count"] == 1
    assert result["eviction_policy_counts"] == {"lru": 1}
    assert result["adapter_transfer_size_mb_sum"] == 16.0
    assert result["state_migration_size_mb_sum"] == 4.0
    assert (result["migration_request_count"], result["migration_realized_count"], result["migration_realization_rate"]) == (2, 1, 0.5)
    assert result["transfer_source_counts"] == {"catalog_cache_object": 10, "catalog_fallback": 2}
    assert result["transfer_source_size_mb_sums"] == {"catalog_cache_object": 16.0, "catalog_fallback": 4.0}
    assert (result["service_success_count"], result["service_failure_count"], result["stall_count"]) == (10, 1, 1)
    assert [result[key] for key in ("vehicle_execution_count", "current_rsu_execution_count", "target_rsu_execution_count", "neighbor_rsu_execution_count", "cloud_execution_count", "unserved_execution_count")] == [1, 6, 1, 1, 1, 1]


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x.__setitem__("event_schema_version", "2.0.0"), "schema major"),
    (lambda x: x.__setitem__("hit_source", "moon"), "hit source"),
    (lambda x: x.__setitem__("cache_hit", False), "RSU hit_source"),
    (lambda x: (x.__setitem__("admission_added", True), x.__setitem__("cache_target_rsu_id", "r1")), "admission_requested"),
    (lambda x: (x.__setitem__("eviction_occurred", True), x.__setitem__("evicted_object_id", "old")), "added admission"),
])
def test_invalid_events_fail_fast(mutation, match) -> None:
    payload = event("bad")
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        reduce_cache_events([payload])


def test_duplicate_id_and_non_mapping_fail_fast() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reduce_cache_events([event("same"), event("same")])
    with pytest.raises(ValueError, match="not a mapping"):
        reduce_cache_events([object()])


def test_capacity_disabled_is_not_zero_capacity() -> None:
    bad = event("bad-capacity", cache_capacity_before=0.0)
    with pytest.raises(ValueError, match="capacity-disabled"):
        reduce_cache_events([bad])


def test_missing_trace_is_unavailable_not_zero() -> None:
    result = reduce_cache_event_summary({})
    assert result.availability == "unavailable"
    assert result.cache_event_schema_version is None
    audit = audit_cache_event_telemetry({})
    assert audit["availability"] == "unavailable"


def test_audit_exact_different_scope_unavailable_and_mismatch() -> None:
    trace = mixed_trace()
    steps = []
    for item in trace:
        steps.append({
            "current_node_id": item["node_id"], "cache_hit": item["cache_hit"],
            "cache_applied": item["admission_requested"],
            "cache_admission_added_new_adapter": item["admission_added"],
            "cache_eviction": item["eviction_occurred"], "eviction_count": int(item["eviction_occurred"]),
            "migration_prepare_requested": item["migration_requested"], "migration_prepare_realized": item["migration_realized"],
            "migration_mode": "prepare" if item["migration_requested"] else "keep",
            "migration_during_handoff": False,
        })
    summary = {"cache_event_schema_version": "1.0.0", "cache_event_trace": trace, "step_trace": steps,
               "system_metrics": {"backhaul_traffic_cost": 16.0, "adapter_warm_hit_ratio": 0.25, "workflow_continuity_rate": 0.9},
               "handoff_summary": {"migration_prepare_count": 2, "migration_during_handoff_count": 0}}
    audit = audit_cache_event_telemetry(summary)
    comparisons = audit["comparisons"]
    assert comparisons["cache_hit_count__adapter_hit_count"]["match_status"] == "match"
    assert comparisons["request_event_count__episode_step_count"]["mapping_class"] == "compatible_but_different_scope"
    assert comparisons["cache_hit_rate__adapter_warm_hit_ratio"]["mapping_class"] == "not_equivalent"
    assert comparisons["hit_source_distribution__legacy"]["match_status"] == "unavailable"
    broken = deepcopy(summary)
    broken["step_trace"][0]["cache_hit"] = False
    assert audit_cache_event_telemetry(broken)["comparisons"]["cache_hit_count__adapter_hit_count"]["match_status"] == "mismatch"
