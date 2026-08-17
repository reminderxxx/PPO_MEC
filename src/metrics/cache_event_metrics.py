"""Pure reduction and legacy reconciliation for CacheEvent telemetry."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from src.envs.specs import CACHE_EVENT_SCHEMA_VERSION, CacheEvent


@dataclass(frozen=True)
class CacheEventMetricSummary:
    availability: str
    cache_event_schema_version: str | None
    request_event_count: int = 0
    not_applicable_event_count: int = 0
    total_event_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_hit_rate: float | None = None
    vehicle_local_hit_count: int = 0
    current_rsu_hit_count: int = 0
    target_rsu_hit_count: int = 0
    neighbor_rsu_hit_count: int = 0
    cloud_served_count: int = 0
    unserved_count: int = 0
    not_applicable_hit_source_count: int = 0
    admission_request_count: int = 0
    admission_added_count: int = 0
    admission_noop_or_existing_count: int = 0
    admission_reason_counts: dict[str, int] | None = None
    eviction_count: int = 0
    evicted_object_count: int = 0
    eviction_policy_counts: dict[str, int] | None = None
    adapter_transfer_size_mb_sum: float = 0.0
    state_migration_size_mb_sum: float = 0.0
    migration_request_count: int = 0
    migration_realized_count: int = 0
    migration_realization_rate: float | None = None
    transfer_source_counts: dict[str, int] | None = None
    transfer_source_size_mb_sums: dict[str, float] | None = None
    service_success_count: int = 0
    service_failure_count: int = 0
    stall_count: int = 0
    vehicle_execution_count: int = 0
    current_rsu_execution_count: int = 0
    target_rsu_execution_count: int = 0
    neighbor_rsu_execution_count: int = 0
    cloud_execution_count: int = 0
    unserved_execution_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _schema_major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid cache event schema version: {version!r}") from exc


def _event_dict(raw: CacheEvent | Mapping[str, Any], index: int) -> dict[str, Any]:
    if isinstance(raw, CacheEvent):
        return raw.to_dict()
    if not isinstance(raw, Mapping):
        raise ValueError(f"cache event at index {index} is not a mapping or CacheEvent")
    return dict(raw)


def _validate_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def reduce_cache_events(
    events: Iterable[CacheEvent | Mapping[str, Any]],
    *,
    schema_version: str = CACHE_EVENT_SCHEMA_VERSION,
) -> CacheEventMetricSummary:
    """Reduce raw events without consulting step/system/aggregate telemetry."""
    if _schema_major(schema_version) != 1:
        raise ValueError(f"unsupported cache event schema major: {schema_version}")
    raw_events = list(events)
    counters: Counter[str] = Counter()
    admission_reasons: Counter[str] = Counter()
    eviction_policies: Counter[str] = Counter()
    transfer_counts: Counter[str] = Counter()
    transfer_sizes: defaultdict[str, float] = defaultdict(float)
    seen_ids: set[str] = set()
    adapter_transfer = state_transfer = 0.0

    for index, raw in enumerate(raw_events):
        payload = _event_dict(raw, index)
        event_version = payload.get("event_schema_version")
        if _schema_major(event_version) != 1:
            raise ValueError(f"unsupported cache event schema major: {event_version}")
        # G01 has no optional fields yet. Validate the frozen v1 fields with the
        # current dataclass while allowing consumer-safe optional fields in 1.x.
        validation_payload = dict(payload)
        validation_payload["event_schema_version"] = CACHE_EVENT_SCHEMA_VERSION
        event = CacheEvent.from_dict(validation_payload)
        if event.event_id in seen_ids:
            raise ValueError(f"duplicate cache event_id: {event.event_id}")
        seen_ids.add(event.event_id)
        if event.admission_added and not event.admission_requested:
            raise ValueError("admission_added requires admission_requested=true")
        if event.eviction_occurred and not event.admission_added:
            raise ValueError("eviction requires an added admission in G01 lifecycle")
        if event.migration_realized and not event.migration_requested:
            raise ValueError("migration_realized requires migration_requested=true")
        a_size = _validate_number(event.adapter_transfer_size_mb, "adapter_transfer_size_mb")
        m_size = _validate_number(event.state_migration_size_mb, "state_migration_size_mb")
        adapter_transfer += a_size
        state_transfer += m_size
        transfer_counts[event.transfer_source] += 1
        transfer_sizes[event.transfer_source] += a_size + m_size
        counters["total"] += 1

        if event.event_type == "not_applicable":
            counters["not_applicable"] += 1
            counters["not_applicable_hit_source"] += int(event.hit_source == "not_applicable")
            continue
        counters["request"] += 1
        counters["hit"] += int(event.cache_hit)
        counters["miss"] += int(not event.cache_hit)
        counters[f"source:{event.hit_source}"] += 1
        counters["admission_request"] += int(event.admission_requested)
        counters["admission_added"] += int(event.admission_added)
        if event.admission_requested:
            admission_reasons[event.admission_reason] += 1
            counters["admission_noop"] += int(not event.admission_added)
        if event.eviction_occurred:
            counters["eviction"] += event.eviction_count
            counters["evicted_object"] += event.eviction_count
            eviction_policies[event.eviction_policy] += 1
        counters["migration_request"] += int(event.migration_requested)
        counters["migration_realized"] += int(event.migration_realized)
        counters["service_success"] += int(event.service_success)
        counters["service_failure"] += int(not event.service_success)
        counters["stall"] += int(event.stall_occurred)

        if event.service_success:
            execution_source = event.hit_source
            if execution_source == "vehicle_local" and event.offload_mode != "vehicle":
                raise ValueError("vehicle_local execution requires offload_mode=vehicle")
            if execution_source == "cloud" and event.offload_mode != "cloud":
                raise ValueError("cloud execution requires offload_mode=cloud")
            if execution_source in {"current_rsu", "target_rsu", "neighbor_rsu"}:
                if event.offload_mode != "rsu" or event.served_rsu_id is None:
                    raise ValueError("RSU execution requires offload_mode=rsu and served_rsu_id")
            if execution_source == "unserved":
                raise ValueError("successful service cannot have unserved hit_source")
            counters[f"execution:{execution_source}"] += 1
        else:
            if not event.stall_occurred or event.hit_source != "unserved":
                raise ValueError("failed request must be a stalled unserved execution")
            counters["execution:unserved"] += 1

    request_count = counters["request"]
    migration_count = counters["migration_request"]
    return CacheEventMetricSummary(
        availability="available",
        cache_event_schema_version=schema_version,
        request_event_count=request_count,
        not_applicable_event_count=counters["not_applicable"],
        total_event_count=counters["total"],
        cache_hit_count=counters["hit"],
        cache_miss_count=counters["miss"],
        cache_hit_rate=round(counters["hit"] / request_count, 6) if request_count else None,
        vehicle_local_hit_count=counters["source:vehicle_local"],
        current_rsu_hit_count=counters["source:current_rsu"],
        target_rsu_hit_count=counters["source:target_rsu"],
        neighbor_rsu_hit_count=counters["source:neighbor_rsu"],
        cloud_served_count=counters["source:cloud"],
        unserved_count=counters["source:unserved"],
        not_applicable_hit_source_count=counters["not_applicable_hit_source"],
        admission_request_count=counters["admission_request"],
        admission_added_count=counters["admission_added"],
        admission_noop_or_existing_count=counters["admission_noop"],
        admission_reason_counts=dict(sorted(admission_reasons.items())),
        eviction_count=counters["eviction"],
        evicted_object_count=counters["evicted_object"],
        eviction_policy_counts=dict(sorted(eviction_policies.items())),
        adapter_transfer_size_mb_sum=round(adapter_transfer, 6),
        state_migration_size_mb_sum=round(state_transfer, 6),
        migration_request_count=migration_count,
        migration_realized_count=counters["migration_realized"],
        migration_realization_rate=(round(counters["migration_realized"] / migration_count, 6) if migration_count else None),
        transfer_source_counts=dict(sorted(transfer_counts.items())),
        transfer_source_size_mb_sums={key: round(value, 6) for key, value in sorted(transfer_sizes.items())},
        service_success_count=counters["service_success"],
        service_failure_count=counters["service_failure"],
        stall_count=counters["stall"],
        vehicle_execution_count=counters["execution:vehicle_local"],
        current_rsu_execution_count=counters["execution:current_rsu"],
        target_rsu_execution_count=counters["execution:target_rsu"],
        neighbor_rsu_execution_count=counters["execution:neighbor_rsu"],
        cloud_execution_count=counters["execution:cloud"],
        unserved_execution_count=counters["execution:unserved"],
    )


def reduce_cache_event_summary(summary: Mapping[str, Any]) -> CacheEventMetricSummary:
    """Compatibility wrapper that distinguishes a missing trace from an empty trace."""
    if "cache_event_trace" not in summary:
        return CacheEventMetricSummary(availability="unavailable", cache_event_schema_version=None)
    trace = summary["cache_event_trace"]
    if not isinstance(trace, list):
        raise ValueError("cache_event_trace must be a list")
    version = summary.get("cache_event_schema_version")
    if version is None:
        raise ValueError("cache_event_schema_version is required when cache_event_trace is present")
    return reduce_cache_events(trace, schema_version=str(version))


def _comparison(event_value: Any, legacy_value: Any, mapping_class: str, note: str) -> dict[str, Any]:
    if legacy_value is None:
        absolute = relative = None
        status = "unavailable"
    elif isinstance(event_value, (int, float)) and isinstance(legacy_value, (int, float)):
        absolute = round(abs(float(event_value) - float(legacy_value)), 6)
        relative = round(absolute / abs(float(legacy_value)), 6) if legacy_value else (0.0 if absolute == 0 else None)
        status = "match" if mapping_class == "exact" and absolute == 0 else ("mismatch" if mapping_class == "exact" else "informational")
    else:
        absolute = relative = None
        status = "match" if mapping_class == "exact" and event_value == legacy_value else ("mismatch" if mapping_class == "exact" else "informational")
    return {"event_derived_value": event_value, "legacy_value": legacy_value, "absolute_difference": absolute,
            "relative_difference": relative, "mapping_class": mapping_class, "match_status": status, "note": note}


def audit_cache_event_telemetry(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compare event-derived values with legacy episode telemetry without coercing scopes."""
    reduced = reduce_cache_event_summary(summary)
    if reduced.availability != "available":
        return {"availability": "unavailable", "event_metrics": reduced.to_dict(), "comparisons": {}}
    event = reduced.to_dict()
    steps = [step for step in summary.get("step_trace", []) if isinstance(step, Mapping)]
    executable = [step for step in steps if step.get("current_node_id") is not None]
    legacy = {
        "episode_step_count": len(steps) if "step_trace" in summary else None,
        "adapter_hit_count": sum(bool(step.get("cache_hit")) for step in executable) if "step_trace" in summary else None,
        "adapter_miss_count": sum(not bool(step.get("cache_hit")) for step in executable) if "step_trace" in summary else None,
        "cache_admission_count": sum(bool(step.get("cache_applied")) for step in steps) if "step_trace" in summary else None,
        "cache_admission_added_new_adapter_count": sum(bool(step.get("cache_admission_added_new_adapter")) for step in steps) if "step_trace" in summary else None,
        "cache_eviction_count": sum(bool(step.get("cache_eviction")) for step in steps) if "step_trace" in summary else None,
        "eviction_count": sum(int(step.get("eviction_count", 0) or 0) for step in steps) if "step_trace" in summary else None,
        "migration_attempt_count": sum(bool(step.get("migration_prepare_requested")) or str(step.get("migration_mode", "")).lower() in {"prepare", "migrate"} for step in steps) if "step_trace" in summary else None,
        "migration_success_count": sum(bool(step.get("migration_prepare_realized")) or bool(step.get("migration_during_handoff")) for step in steps) if "step_trace" in summary else None,
        "migration_prepare_count": summary.get("handoff_summary", {}).get("migration_prepare_count"),
        "migration_during_handoff_count": summary.get("handoff_summary", {}).get("migration_during_handoff_count"),
        "backhaul_traffic_cost": summary.get("system_metrics", {}).get("backhaul_traffic_cost"),
        "workflow_continuity_rate": summary.get("system_metrics", {}).get("workflow_continuity_rate"),
        "adapter_warm_hit_ratio": summary.get("system_metrics", {}).get("adapter_warm_hit_ratio"),
    }
    comparisons = {
        "request_event_count__episode_step_count": _comparison(event["request_event_count"], legacy["episode_step_count"], "compatible_but_different_scope", "steps may include not_applicable events"),
        "cache_hit_count__adapter_hit_count": _comparison(event["cache_hit_count"], legacy["adapter_hit_count"], "exact", "both count executable request cache_hit=true under G01"),
        "cache_miss_count__adapter_miss_count": _comparison(event["cache_miss_count"], legacy["adapter_miss_count"], "exact", "both count executable request cache_hit=false under G01"),
        "admission_request_count__cache_admission_count": _comparison(event["admission_request_count"], legacy["cache_admission_count"], "exact", "request lifecycle admission flag"),
        "admission_added_count__cache_admission_added_new_adapter_count": _comparison(event["admission_added_count"], legacy["cache_admission_added_new_adapter_count"], "exact", "new adapter admission"),
        "eviction_count__cache_eviction_count": _comparison(event["eviction_count"], legacy["cache_eviction_count"], "compatible_but_different_scope", "legacy cache_eviction_count counts request events; CacheEvent 1.1 eviction_count counts victims"),
        "eviction_count__eviction_count": _comparison(event["eviction_count"], legacy["eviction_count"], "exact", "both count victims; CacheEvent 1.0 implies one victim"),
        "migration_request_count__migration_attempt_count": _comparison(event["migration_request_count"], legacy["migration_attempt_count"], "exact", "prepare or migrate request"),
        "migration_request_count__migration_prepare_count": _comparison(event["migration_request_count"], legacy["migration_prepare_count"], "compatible_but_different_scope", "handoff summary counts prepare mode only"),
        "migration_realized_count__migration_success_count": _comparison(event["migration_realized_count"], legacy["migration_success_count"], "exact", "realized prepare or migration during handoff"),
        "migration_realized_count__migration_during_handoff_count": _comparison(event["migration_realized_count"], legacy["migration_during_handoff_count"], "compatible_but_different_scope", "handoff-only subset may differ from realized prepare"),
        "total_transfer_size_mb__backhaul_traffic_cost": _comparison(round(event["adapter_transfer_size_mb_sum"] + event["state_migration_size_mb_sum"], 6), legacy["backhaul_traffic_cost"], "compatible_but_different_scope", "legacy backhaul books migration at handoff; events book requested/realized migration bundle semantics"),
        "cache_hit_rate__adapter_warm_hit_ratio": _comparison(event["cache_hit_rate"], legacy["adapter_warm_hit_ratio"], "not_equivalent", "warm readiness is step-level, not request cache hit rate"),
        "stall_count__workflow_continuity_rate": _comparison(event["stall_count"], legacy["workflow_continuity_rate"], "not_equivalent", "count and episode step-level rate have different units"),
        "hit_source_distribution__legacy": _comparison({k: event[k] for k in ("vehicle_local_hit_count", "current_rsu_hit_count", "target_rsu_hit_count", "neighbor_rsu_hit_count", "cloud_served_count", "unserved_count")}, None, "unavailable", "legacy summary has no complete request hit-source distribution"),
    }
    return {"availability": "available", "event_metrics": event, "legacy_values": legacy, "comparisons": comparisons}
