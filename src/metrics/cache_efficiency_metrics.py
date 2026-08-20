"""Pure, request-level CacheEvent reducers for the G06 efficiency contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from math import isfinite
from statistics import fmean
from typing import Any, Iterable, Mapping

from src.envs.specs import CACHE_EVENT_SCHEMA_VERSION, CacheEvent
from src.metrics.cache_event_metrics import reduce_cache_events


CACHE_EFFICIENCY_METRICS_VERSION = "1.2.0"
DEFAULT_REUSE_HORIZONS = (1, 3, 6, 12)
HIT_SOURCES = (
    "vehicle_local", "current_rsu", "target_rsu", "neighbor_rsu", "cloud", "unserved"
)


def _group(availability: str, reason: str | None, required: list[str], available: int, unavailable: int) -> dict[str, Any]:
    return {
        "availability": availability,
        "unavailable_reason": reason,
        "required_fields": required,
        "available_event_count": available,
        "unavailable_event_count": unavailable,
    }


def _number(value: Any, field_name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative number") from exc
    if not isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator > 0 else None


@dataclass(frozen=True)
class CacheEfficiencyMetricSummary:
    availability: str
    cache_event_schema_version: str | None
    cache_efficiency_metrics_version: str = CACHE_EFFICIENCY_METRICS_VERSION
    request_metrics: dict[str, Any] = field(default_factory=dict)
    byte_metrics: dict[str, Any] = field(default_factory=dict)
    lifecycle_metrics: dict[str, Any] = field(default_factory=dict)
    capacity_metrics: dict[str, Any] = field(default_factory=dict)
    pollution_metrics: dict[str, Any] = field(default_factory=dict)
    future_reuse_proxy_metrics: dict[str, Any] = field(default_factory=dict)
    latency_saved_metrics: dict[str, Any] = field(default_factory=dict)
    type_aware_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payloads(events: Iterable[CacheEvent | Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, raw in enumerate(events):
        if isinstance(raw, CacheEvent):
            result.append(raw.to_dict())
        elif isinstance(raw, Mapping):
            result.append(dict(raw))
        else:
            raise ValueError(f"cache event at index {index} is not a mapping or CacheEvent")
    return result


def _snapshot_residents(snapshot: Mapping[str, Any], label: str) -> tuple[dict[tuple[str, str], dict[str, Any]], set[tuple[bool, str, float | None]]]:
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("rsus"), list):
        raise ValueError(f"{label} cache snapshot must contain rsus list")
    residents: dict[tuple[str, str], dict[str, Any]] = {}
    contracts: set[tuple[bool, str, float | None]] = set()
    for rsu in snapshot["rsus"]:
        if not isinstance(rsu, Mapping) or not rsu.get("rsu_id") or not isinstance(rsu.get("residents"), list):
            raise ValueError(f"malformed {label} RSU snapshot")
        enabled = bool(rsu.get("capacity_enabled", False))
        unit = str(rsu.get("capacity_unit", "adapter_slots"))
        capacity = _number(rsu.get("capacity"), f"{label}.capacity", nullable=True)
        if not enabled and capacity is not None:
            raise ValueError("capacity-disabled trace context must have null capacity")
        contracts.add((enabled, unit, capacity))
        for item in rsu["residents"]:
            if not isinstance(item, Mapping) or not item.get("object_id"):
                raise ValueError(f"malformed {label} resident")
            size = _number(item.get("size_mb"), f"{label}.resident.size_mb")
            object_type = str(item.get("object_type") or "adapter")
            adapter_id = item.get("adapter_id")
            if object_type == "adapter" and not adapter_id:
                raise ValueError(f"malformed {label} adapter resident")
            key = (str(rsu["rsu_id"]), str(item["object_id"]))
            if key in residents:
                raise ValueError(f"duplicate resident in {label} snapshot: {key}")
            residents[key] = {
                "adapter_id": str(adapter_id) if adapter_id else None,
                "object_type": object_type,
                "required_base_model_id": item.get("required_base_model_id"),
                "evictability": item.get("evictability"),
                "size_mb": size,
            }
    return residents, contracts


def _pollution_metrics(events: list[dict[str, Any]], context: Mapping[str, Any] | None) -> dict[str, Any]:
    required = [
        "cache_trace_context.initial_snapshot", "cache_trace_context.final_snapshot",
        "admitted_object_id", "admitted_size_mb", "evicted_object_ids",
    ]
    if not isinstance(context, Mapping):
        return {**_group("unavailable", "missing_cache_trace_context", required, 0, len(events))}
    if str(context.get("context_schema_version", "")).split(".", 1)[0] != "1":
        raise ValueError("unsupported cache trace context schema major")
    initial, initial_contracts = _snapshot_residents(context.get("initial_snapshot"), "initial")
    final, final_contracts = _snapshot_residents(context.get("final_snapshot"), "final")
    if initial_contracts != final_contracts:
        raise ValueError("initial/final cache capacity contract mismatch")
    end_step = context.get("episode_end_step_index")
    if not isinstance(end_step, int) or end_step < 0:
        raise ValueError("episode_end_step_index must be a non-negative integer")

    active: dict[tuple[str, str], dict[str, Any]] = {
        key: {**value, "start": 0, "origin": "initial", "hit": False}
        for key, value in initial.items()
    }
    total_mb_steps = polluted_mb_steps = 0.0
    confirmed_count = censored_count = 0
    confirmed_mb = censored_mb = 0.0

    for event in events:
        if event.get("event_type") != "request":
            continue
        step = event.get("episode_step_index")
        if not isinstance(step, int) or step < 0 or step > end_step:
            raise ValueError("event step is outside cache trace context horizon")
        if event.get("eviction_occurred"):
            rsu_id = event.get("cache_target_rsu_id")
            if not rsu_id:
                raise ValueError("eviction requires cache_target_rsu_id for residency reconstruction")
            for object_id in event.get("evicted_object_ids") or [event.get("evicted_object_id")]:
                key = (str(rsu_id), str(object_id))
                interval = active.pop(key, None)
                if interval is None:
                    raise ValueError(f"eviction references non-resident object: {key}")
                duration = step - int(interval["start"])
                total_mb_steps += float(interval["size_mb"]) * duration
                if interval["origin"] == "admission" and not interval["hit"]:
                    confirmed_count += 1
                    confirmed_mb += float(interval["size_mb"])
                    polluted_mb_steps += float(interval["size_mb"]) * duration
        typed_admissions = list(event.get("admitted_typed_objects") or [])
        if typed_admissions:
            rsu_id = event.get("cache_target_rsu_id")
            if not rsu_id:
                return {**_group("unavailable", "admission_identity_or_size_missing", required, 0, len(events))}
            for row in typed_admissions:
                object_id = row.get("object_id")
                size = _number(row.get("resident_size_mb"), "resident_size_mb", nullable=True)
                if not object_id or size is None:
                    return {**_group("unavailable", "admission_identity_or_size_missing", required, 0, len(events))}
                key = (str(rsu_id), str(object_id))
                if key in active:
                    raise ValueError(f"duplicate admission of resident object: {key}")
                active[key] = {
                    "adapter_id": row.get("adapter_id"),
                    "object_type": str(row.get("object_type")),
                    "required_base_model_id": row.get("required_base_model_id"),
                    "evictability": row.get("evictability"),
                    "size_mb": size,
                    "start": step,
                    "origin": "admission",
                    "hit": False,
                }
        elif event.get("admission_added"):
            rsu_id = event.get("cache_target_rsu_id")
            object_id = event.get("admitted_object_id")
            adapter_id = event.get("admitted_adapter_id")
            size = _number(event.get("admitted_size_mb"), "admitted_size_mb", nullable=True)
            if not rsu_id or not object_id or not adapter_id or size is None:
                return {**_group("unavailable", "admission_identity_or_size_missing", required, 0, len(events))}
            key = (str(rsu_id), str(object_id))
            if key in active:
                raise ValueError(f"duplicate admission of resident object: {key}")
            active[key] = {"adapter_id": str(adapter_id), "object_type": "adapter", "required_base_model_id": None, "evictability": None, "size_mb": size, "start": step, "origin": "admission", "hit": False}
        if event.get("cache_hit") and event.get("hit_source") in {"current_rsu", "target_rsu", "neighbor_rsu"}:
            lookup_ids = [
                row.get("object_id")
                for row in event.get("per_object_lookup_results") or []
                if row.get("resident")
            ] or [event.get("object_id")]
            for object_id in lookup_ids:
                key = (str(event.get("served_rsu_id")), str(object_id))
                if key not in active:
                    raise ValueError(f"cache hit references non-resident object: {key}")
                active[key]["hit"] = True

    closure_step = end_step + 1
    for interval in active.values():
        duration = closure_step - int(interval["start"])
        total_mb_steps += float(interval["size_mb"]) * duration
        if interval["origin"] == "admission" and not interval["hit"]:
            censored_count += 1
            censored_mb += float(interval["size_mb"])
    reconstructed = {
        (rsu, obj): {
            "adapter_id": value.get("adapter_id"),
            "object_type": value.get("object_type", "adapter"),
            "required_base_model_id": value.get("required_base_model_id"),
            "evictability": value.get("evictability"),
            "size_mb": value["size_mb"],
        }
        for (rsu, obj), value in active.items()
    }
    if reconstructed != final:
        raise ValueError("reconstructed final cache does not match final snapshot")
    return {
        **_group("available", None, required, len(events), 0),
        "unused_admitted_object_count": confirmed_count,
        "unused_admitted_size_mb": round(confirmed_mb, 6),
        "right_censored_unused_admitted_object_count": censored_count,
        "right_censored_unused_admitted_size_mb": round(censored_mb, 6),
        "polluted_resident_mb_steps": round(polluted_mb_steps, 6),
        "total_resident_mb_steps": round(total_mb_steps, 6),
        "cache_pollution_ratio": _rate(polluted_mb_steps, total_mb_steps),
        "right_censored_unused_by_object_type": dict(sorted(Counter(
            interval.get("object_type", "adapter")
            for interval in active.values()
            if interval["origin"] == "admission" and not interval["hit"]
        ).items())),
        "censoring_semantics": "episode-end un-reused admissions are right-censored and excluded from pollution numerator",
    }


def _future_reuse(events: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    victims: list[dict[str, Any]] = []
    requests = [item for item in events if item.get("event_type") == "request"]
    for event in requests:
        if not event.get("eviction_occurred"):
            continue
        ids = list(event.get("evicted_object_ids") or [event.get("evicted_object_id")])
        sizes = event.get("evicted_sizes_mb")
        if sizes is not None and not isinstance(sizes, list):
            raise ValueError("evicted_sizes_mb must be a list")
        if sizes and len(sizes) != len(ids):
            raise ValueError("evicted_sizes_mb must align with victims")
        for index, object_id in enumerate(ids):
            size = _number(sizes[index], "evicted_sizes_mb", nullable=True) if sizes and index < len(sizes) else None
            next_gap = next(
                (
                    int(future["episode_step_index"]) - int(event["episode_step_index"])
                    for future in requests
                    if int(future["episode_step_index"]) > int(event["episode_step_index"])
                    and future.get("object_id") == object_id
                ),
                None,
            )
            victims.append({"size_mb": size, "next_gap": next_gap})
    result: dict[str, Any] = {
        **_group("available", None, ["evicted_object_ids", "episode_step_index", "object_id"], len(victims), 0),
        "evicted_victim_count": len(victims),
        "time_to_next_request_after_eviction_steps_mean": (
            round(fmean(item["next_gap"] for item in victims if item["next_gap"] is not None), 6)
            if any(item["next_gap"] is not None for item in victims) else None
        ),
        "horizons": {},
        "claim_boundary": "future-request reuse proxy; not causal eviction regret or oracle gap",
    }
    for horizon in horizons:
        reused = [item for item in victims if item["next_gap"] is not None and item["next_gap"] <= horizon]
        size_available = all(item["size_mb"] is not None for item in reused)
        result["horizons"][str(horizon)] = {
            "evicted_then_requested_within_h_count": len(reused),
            "evicted_then_requested_within_h_mb": round(sum(item["size_mb"] for item in reused), 6) if size_available else None,
            "eviction_future_reuse_rate_h": _rate(len(reused), len(victims)),
            "mb_availability": "available" if size_available else "partial",
        }
    return result


def _sum_type_maps(events: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        value = event.get(field_name) or {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        for object_type, raw in value.items():
            totals[str(object_type)] = totals.get(str(object_type), 0.0) + float(
                _number(raw, f"{field_name}.{object_type}") or 0.0
            )
    return {key: round(value, 6) for key, value in sorted(totals.items())}


def _type_aware_metrics(
    requests: list[dict[str, Any]], context: Mapping[str, Any] | None
) -> dict[str, Any]:
    typed = [
        item
        for item in requests
        if item.get("model_cache_profile_id") == "typed_base_adapter_state_v1"
    ]
    if not typed:
        return {
            **_group(
                "unavailable",
                "legacy_trace_has_no_typed_model_cache_evidence",
                ["model_cache_profile_id", "base_model_hit", "adapter_hit"],
                0,
                len(requests),
            ),
            "request_count": len(requests),
        }
    if len(typed) != len(requests):
        return {
            **_group(
                "partial",
                "mixed_legacy_and_typed_trace_not_aggregated",
                ["model_cache_profile_id"],
                len(typed),
                len(requests) - len(typed),
            ),
            "request_count": len(requests),
        }
    count = len(typed)
    for item in typed:
        if item.get("typed_model_cache_contract_version") != "1.0.0":
            raise ValueError("typed event contract version mismatch")
        if item.get("orphan_count") != 0:
            raise ValueError("typed event reports an orphan resident")
    base_hits = sum(bool(item.get("base_model_hit")) for item in typed)
    adapter_hits = sum(bool(item.get("adapter_hit")) for item in typed)
    joint_hits = sum(bool(item.get("joint_model_hit")) for item in typed)
    state_ready = sum(bool(item.get("workflow_state_ready")) for item in typed)
    full_ready = sum(bool(item.get("full_service_ready")) for item in typed)
    missing_types = Counter(
        object_type
        for item in typed
        for object_type in item.get("missing_object_types") or []
    )
    compatibility_failures = sum(
        item.get("compatibility_result") == "incompatible" for item in typed
    )
    requested_mb: dict[str, float] = {}
    hit_mb: dict[str, float] = {}
    dependency_bytes_complete_count = 0
    dependency_bytes_missing_count = 0
    requested_service_dependency_mb = 0.0
    full_service_ready_dependency_mb = 0.0
    sharing_hits = 0
    avoided_base_transfer_mb = 0.0
    base_reuse_count = 0
    seen_base_requests: Counter[str] = Counter()
    base_to_adapters: dict[str, set[str]] = {}
    for item in typed:
        rows = list(item.get("requested_typed_objects") or [])
        object_ids = [str(row.get("object_id") or "") for row in rows]
        if not rows or any(not object_id for object_id in object_ids):
            dependency_bytes_missing_count += 1
        elif len(object_ids) != len(set(object_ids)):
            raise ValueError(
                "requested_typed_objects must contain unique dependency objects per request"
            )
        else:
            sizes = [
                _number(row.get("resident_size_mb"), "resident_size_mb", nullable=True)
                for row in rows
            ]
            if any(size is None for size in sizes):
                dependency_bytes_missing_count += 1
            else:
                dependency_bytes_complete_count += 1
                event_dependency_mb = sum(float(size) for size in sizes if size is not None)
                requested_service_dependency_mb += event_dependency_mb
                if item.get("full_service_ready") is True:
                    full_service_ready_dependency_mb += event_dependency_mb
        row_by_id = {str(row.get("object_id")): row for row in rows}
        lookup = list(item.get("per_object_lookup_results") or [])
        for row in rows:
            object_type = str(row.get("object_type"))
            resident_size_mb = _number(
                row.get("resident_size_mb"), "resident_size_mb", nullable=True
            )
            if resident_size_mb is not None:
                requested_mb[object_type] = (
                    requested_mb.get(object_type, 0.0) + float(resident_size_mb)
                )
        for row in lookup:
            if not row.get("resident"):
                continue
            source = row_by_id.get(str(row.get("object_id")))
            if source:
                object_type = str(source.get("object_type"))
                resident_size_mb = _number(
                    source.get("resident_size_mb"),
                    "resident_size_mb",
                    nullable=True,
                )
                if resident_size_mb is not None:
                    hit_mb[object_type] = (
                        hit_mb.get(object_type, 0.0) + float(resident_size_mb)
                    )
        base_row = next((row for row in rows if row.get("object_type") == "base_model"), None)
        adapter_row = next((row for row in rows if row.get("object_type") == "adapter"), None)
        if base_row:
            base_id = str(base_row.get("object_id"))
            base_reuse_count += int(seen_base_requests[base_id] > 0 and bool(item.get("base_model_hit")))
            seen_base_requests[base_id] += 1
            if adapter_row:
                previous_adapters = base_to_adapters.setdefault(base_id, set())
                adapter_id = str(adapter_row.get("adapter_id"))
                if bool(item.get("base_model_hit")) and adapter_id not in previous_adapters and previous_adapters:
                    sharing_hits += 1
                previous_adapters.add(adapter_id)
            if bool(item.get("base_model_hit")) and not bool(item.get("adapter_hit")):
                avoided_base_transfer_mb += float(
                    _number(base_row.get("transfer_size_mb"), "transfer_size_mb") or 0.0
                )
    admitted_by_type = _sum_type_maps(typed, "admitted_mb_by_type")
    evicted_by_type = _sum_type_maps(typed, "evicted_mb_by_type")
    transfer_by_type = _sum_type_maps(typed, "transfer_mb_by_type")
    resident_by_type: dict[str, float] | None = None
    pinned_mb = None
    adapters_per_base: dict[str, int] | None = None
    if isinstance(context, Mapping) and isinstance(context.get("final_snapshot"), Mapping):
        resident_by_type = {}
        pinned_mb = 0.0
        base_adapter_sets: dict[str, set[str]] = {}
        for rsu in context["final_snapshot"].get("rsus", []):
            residents = list(rsu.get("residents") or [])
            base_objects = {
                row.get("base_model_id"): row.get("object_id")
                for row in residents
                if row.get("object_type") == "base_model"
            }
            for row in residents:
                object_type = str(row.get("object_type") or "adapter")
                size = float(_number(row.get("size_mb"), "resident.size_mb") or 0.0)
                resident_by_type[object_type] = resident_by_type.get(object_type, 0.0) + size
                if row.get("evictability") != "evictable":
                    pinned_mb += size
                required = row.get("required_base_model_id")
                if object_type == "adapter" and required in base_objects:
                    key = f"{rsu.get('rsu_id')}/{base_objects[required]}"
                    base_adapter_sets.setdefault(key, set()).add(str(row.get("adapter_id")))
        resident_by_type = {
            key: round(value, 6) for key, value in sorted(resident_by_type.items())
        }
        adapters_per_base = {
            key: len(value) for key, value in sorted(base_adapter_sets.items())
        }
        pinned_mb = round(pinned_mb, 6)
    dependency_bytes_complete = dependency_bytes_complete_count == count
    dependency_byte_hit_rate = (
        _rate(full_service_ready_dependency_mb, requested_service_dependency_mb)
        if dependency_bytes_complete
        else None
    )
    base_transfer_mb = float(transfer_by_type.get("base_model", 0.0))
    adapter_transfer_mb = float(transfer_by_type.get("adapter", 0.0))
    workflow_state_transfer_mb = float(transfer_by_type.get("workflow_state", 0.0))
    other_transfer_mb = sum(
        float(value)
        for object_type, value in transfer_by_type.items()
        if object_type not in {"base_model", "adapter", "workflow_state"}
    )
    primary_transfer_mb = (
        base_transfer_mb + adapter_transfer_mb + workflow_state_transfer_mb
    )
    total_transfer = primary_transfer_mb + other_transfer_mb
    return {
        **_group("available", None, ["CacheEvent 1.3 typed fields"], count, 0),
        "typed_model_cache_contract_version": "1.0.0",
        "request_count": count,
        "base_hit_count": base_hits,
        "base_hit_rate": _rate(base_hits, count),
        "adapter_hit_count": adapter_hits,
        "adapter_hit_rate": _rate(adapter_hits, count),
        "joint_base_adapter_hit_count": joint_hits,
        "joint_base_adapter_hit_rate": _rate(joint_hits, count),
        "workflow_state_ready_count": state_ready,
        "workflow_state_ready_rate": _rate(state_ready, count),
        "full_service_ready_count": full_ready,
        "full_service_ready_rate": _rate(full_ready, count),
        "full_service_ready_request_rate": _rate(full_ready, count),
        "requested_dependency_byte_coverage_count": dependency_bytes_complete_count,
        "requested_dependency_byte_missing_count": dependency_bytes_missing_count,
        "requested_dependency_byte_coverage_rate": _rate(
            dependency_bytes_complete_count, count
        ),
        "requested_service_dependency_mb": (
            round(requested_service_dependency_mb, 6)
            if dependency_bytes_complete
            else None
        ),
        "full_service_ready_dependency_mb": (
            round(full_service_ready_dependency_mb, 6)
            if dependency_bytes_complete
            else None
        ),
        "full_service_ready_byte_hit_rate": dependency_byte_hit_rate,
        "full_service_ready_byte_hit_rate_availability": (
            "available" if dependency_bytes_complete else "partial"
        ),
        "dependency_byte_denominator_semantics": (
            "sum unique base_model+adapter resident bytes once within each eligible typed request; "
            "the same shared base is counted again only for a distinct request event because it is "
            "a requested service dependency, never once per lookup row or resident inventory entry"
        ),
        "miss_count_by_missing_object_type": dict(sorted(missing_types.items())),
        "compatibility_failure_count": compatibility_failures,
        "requested_mb_by_type": {key: round(value, 6) for key, value in sorted(requested_mb.items())},
        "hit_mb_by_type": {key: round(value, 6) for key, value in sorted(hit_mb.items())},
        "resident_mb_by_type": resident_by_type,
        "admitted_mb_by_type": admitted_by_type,
        "evicted_mb_by_type": evicted_by_type,
        "transfer_mb_by_type": transfer_by_type,
        "base_occupancy_share": (
            _rate(resident_by_type.get("base_model", 0.0), sum(resident_by_type.values()))
            if resident_by_type
            else None
        ),
        "adapter_occupancy_share": (
            _rate(resident_by_type.get("adapter", 0.0), sum(resident_by_type.values()))
            if resident_by_type
            else None
        ),
        "pinned_or_unavailable_capacity_mb": pinned_mb,
        "dependency_bundle_rejection_count": sum(
            bool(item.get("capacity_rejection_reason")) for item in typed
        ),
        "adapters_per_resident_base": adapters_per_base,
        "base_reuse_count": base_reuse_count,
        "base_sharing_hit_count": sharing_hits,
        "avoided_duplicate_base_transfer_mb": round(avoided_base_transfer_mb, 6),
        "orphan_count": 0,
        "dependency_bundle_churn_mb": round(
            sum(admitted_by_type.values()) + sum(evicted_by_type.values()), 6
        ),
        "total_transfer_mb": round(total_transfer, 6),
        "primary_transfer_mb": round(primary_transfer_mb, 6),
        "base_model_transfer_mb": round(base_transfer_mb, 6),
        "adapter_transfer_mb": round(adapter_transfer_mb, 6),
        "workflow_state_migration_transfer_mb": round(workflow_state_transfer_mb, 6),
        "other_typed_transfer_mb": round(other_transfer_mb, 6),
        "transfer_mb_per_request": _rate(primary_transfer_mb, count),
        "transfer_mb_per_request_availability": "available",
        "primary_transfer_composition": [
            "base_model_transfer_mb",
            "adapter_transfer_mb",
            "workflow_state_migration_transfer_mb",
        ],
        "other_typed_transfer_excluded_from_primary": True,
        "transfer_amplification": _rate(
            primary_transfer_mb, requested_service_dependency_mb
        ) if dependency_bytes_complete else None,
        "latency_saved": {"availability": "unavailable", "value": None},
    }


def reduce_cache_efficiency_events(
    events: Iterable[CacheEvent | Mapping[str, Any]], *, schema_version: str = CACHE_EVENT_SCHEMA_VERSION,
    trace_context: Mapping[str, Any] | None = None, reuse_horizons: Iterable[int] = DEFAULT_REUSE_HORIZONS,
) -> CacheEfficiencyMetricSummary:
    payloads = _payloads(events)
    base = reduce_cache_events(payloads, schema_version=schema_version)
    horizons = tuple(int(item) for item in reuse_horizons)
    if not horizons or any(item <= 0 for item in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("reuse horizons must be unique positive integers")
    requests = [item for item in payloads if item.get("event_type") == "request"]

    source_counts = Counter(str(item.get("hit_source")) for item in requests)
    request_count = len(requests)
    request_metrics = {
        **_group("available", None, ["event_type", "cache_hit", "hit_source"], request_count, 0),
        "request_count": request_count,
        "object_hit_count": base.cache_hit_count,
        "object_miss_count": base.cache_miss_count,
        "object_hit_rate": base.cache_hit_rate,
        "hit_source_request_counts": {source: source_counts[source] for source in HIT_SOURCES},
        "hit_source_request_rates": {source: _rate(source_counts[source], request_count) for source in HIT_SOURCES},
    }

    sizes: list[float | None] = [_number(item.get("size_mb"), "size_mb", nullable=True) for item in requests]
    size_available = sum(item is not None for item in sizes)
    byte_complete = size_available == request_count
    requested_sum = sum(item for item in sizes if item is not None)
    hit_sum = sum(size for item, size in zip(requests, sizes) if size is not None and bool(item.get("cache_hit")))
    miss_sum = sum(size for item, size in zip(requests, sizes) if size is not None and not bool(item.get("cache_hit")))
    source_mb = {
        source: round(sum(size for item, size in zip(requests, sizes) if size is not None and item.get("hit_source") == source), 6)
        for source in HIT_SOURCES
    }
    byte_metrics = {
        **_group("available" if byte_complete else ("partial" if size_available else "unavailable"),
                 None if byte_complete else "request_size_mb_missing", ["size_mb"], size_available, request_count - size_available),
        "byte_denominator_coverage_count": size_available,
        "byte_denominator_request_count": request_count,
        "byte_denominator_coverage_rate": _rate(size_available, request_count),
        "requested_size_mb_sum": round(requested_sum, 6) if byte_complete else None,
        "hit_size_mb_sum": round(hit_sum, 6) if byte_complete else None,
        "miss_size_mb_sum": round(miss_sum, 6) if byte_complete else None,
        "byte_hit_rate": _rate(hit_sum, requested_sum) if byte_complete else None,
        "hit_source_requested_size_mb_sums": source_mb if byte_complete else {key: None for key in HIT_SOURCES},
        "hit_source_byte_rates": {key: _rate(value, requested_sum) for key, value in source_mb.items()} if byte_complete else {key: None for key in HIT_SOURCES},
    }

    admission_requested = [item for item in requests if item.get("admission_requested")]
    admissions = [item for item in requests if item.get("admission_added")]
    admission_sizes = [_number(item.get("admitted_size_mb"), "admitted_size_mb", nullable=True) for item in admissions]
    rejected = [item for item in admission_requested if not item.get("admission_added")]
    oversized = [item for item in rejected if item.get("capacity_rejection_reason") == "object_exceeds_total_capacity"]
    oversized_sizes = [_number(item.get("requested_object_size_mb"), "requested_object_size_mb", nullable=True) for item in oversized]
    evictions = [item for item in requests if item.get("eviction_occurred")]
    victim_count = sum(int(item.get("eviction_count", len(item.get("evicted_object_ids") or [])) or 0) for item in evictions)
    eviction_mb_values = [_number(item.get("evicted_size_mb_sum"), "evicted_size_mb_sum", nullable=True) for item in evictions]
    admission_mb_complete = all(item is not None for item in admission_sizes)
    eviction_mb_complete = all(item is not None for item in eviction_mb_values)
    admitted_mb = sum(item for item in admission_sizes if item is not None)
    evicted_mb = sum(item for item in eviction_mb_values if item is not None)
    total_transfer = base.adapter_transfer_size_mb_sum + base.state_migration_size_mb_sum
    lifecycle_metrics = {
        **_group("available" if admission_mb_complete and eviction_mb_complete else "partial", None if admission_mb_complete and eviction_mb_complete else "lifecycle_size_mb_missing", ["admitted_size_mb", "evicted_size_mb_sum"], len(admissions) + len(evictions), 0),
        "admission_requested_count": len(admission_requested), "admission_added_count": len(admissions),
        "admitted_size_mb_sum": round(admitted_mb, 6) if admission_mb_complete else None,
        "admission_rejection_count": len(rejected),
        "admission_rejection_reason_counts": dict(sorted(Counter(str(item.get("capacity_rejection_reason") or item.get("admission_reason")) for item in rejected).items())),
        "oversized_rejection_count": len(oversized),
        "oversized_rejection_size_mb_sum": round(sum(item for item in oversized_sizes if item is not None), 6) if all(item is not None for item in oversized_sizes) else None,
        "eviction_event_count": len(evictions), "eviction_victim_count": victim_count,
        "evicted_size_mb_sum": round(evicted_mb, 6) if eviction_mb_complete else None,
        "adapter_transfer_size_mb_sum": base.adapter_transfer_size_mb_sum,
        "state_migration_size_mb_sum": base.state_migration_size_mb_sum,
        "total_transfer_size_mb_sum": round(total_transfer, 6),
        "cache_churn_mb": round(admitted_mb + evicted_mb, 6) if admission_mb_complete and eviction_mb_complete else None,
        "transfer_per_hit_mb": _rate(total_transfer, base.cache_hit_count),
        "transfer_amplification_ratio": _rate(total_transfer, hit_sum) if byte_complete else None,
    }

    capacity_flags = {bool(item.get("cache_capacity_enabled")) for item in requests}
    if len(capacity_flags) > 1:
        raise ValueError("cache capacity enabled state changes within episode")
    if not requests or capacity_flags == {False}:
        capacity_metrics = {**_group("not_applicable", "cache_capacity_disabled", ["cache_capacity_*"], 0, request_count),
                            "observed_capacity_unit": None, "observation_count": 0, "mean_occupancy": None,
                            "peak_occupancy": None, "capacity_saturation_event_count": None,
                            "capacity_saturation_event_rate": None, "rejected_due_to_capacity_count": None}
    else:
        units = {str(item.get("cache_capacity_unit")) for item in requests}
        capacities = {
            _number(item.get(field_name), field_name)
            for item in requests
            for field_name in ("cache_capacity_before", "cache_capacity_after")
            if item.get(field_name) is not None
        }
        if not capacities and isinstance(trace_context, Mapping):
            initial_snapshot = trace_context.get("initial_snapshot")
            if isinstance(initial_snapshot, Mapping):
                capacities = {
                    _number(rsu.get("capacity"), "trace_context.capacity")
                    for rsu in initial_snapshot.get("rsus", [])
                    if isinstance(rsu, Mapping) and rsu.get("capacity") is not None
                }
        if len(units) != 1 or len(capacities) != 1:
            raise ValueError("cache capacity unit or value changes within episode")
        capacity = next(iter(capacities))
        if capacity <= 0:
            raise ValueError("enabled cache capacity must be positive")
        observations = []
        saturated_events = 0
        observed_events = 0
        for item in requests:
            event_saturated = False
            for suffix in ("before", "after"):
                values = (item.get(f"cache_capacity_{suffix}"), item.get(f"cache_used_{suffix}"), item.get(f"cache_remaining_{suffix}"))
                if values == (None, None, None):
                    continue
                if values[0] is not None and values[1] is None and values[2] is None:
                    observed_capacity = _number(values[0], f"cache_capacity_{suffix}")
                    if abs(observed_capacity - capacity) > 1e-6:
                        raise ValueError("cache capacity value changes within episode")
                    continue
                if any(value is None for value in values):
                    raise ValueError("partial capacity snapshot is contradictory")
                observed_capacity = _number(values[0], f"cache_capacity_{suffix}")
                used = _number(values[1], f"cache_used_{suffix}")
                remaining = _number(values[2], f"cache_remaining_{suffix}")
                if abs(observed_capacity - capacity) > 1e-6:
                    raise ValueError("cache capacity value changes within episode")
                if abs((used + remaining) - capacity) > 1e-6:
                    raise ValueError("capacity used/remaining snapshot is contradictory")
                observations.append(used / capacity)
                event_saturated |= remaining <= 1e-9
            observed_events += int(any(
                item.get(f"cache_used_{suffix}") is not None and item.get(f"cache_remaining_{suffix}") is not None
                for suffix in ("before", "after")
            ))
            saturated_events += int(event_saturated)
        rejected_capacity = sum(bool(item.get("capacity_rejection_reason")) for item in rejected)
        capacity_metrics = {
            **_group("available" if observed_events == request_count else "partial",
                     None if observed_events == request_count else "capacity_snapshot_not_observed_for_all_requests",
                     ["cache_capacity_*", "cache_used_*", "cache_remaining_*"], len(observations), 2 * request_count - len(observations)),
            "observed_capacity_unit": next(iter(units)), "observed_capacity": capacity,
            "observation_count": len(observations), "mean_occupancy": round(fmean(observations), 6) if observations else None,
            "peak_occupancy": round(max(observations), 6) if observations else None,
            "capacity_saturation_event_count": saturated_events,
            "capacity_saturation_event_rate": _rate(saturated_events, observed_events),
            "rejected_due_to_capacity_count": rejected_capacity,
        }

    latency = {
        **_group("unavailable", "missing_request_aligned_observed_and_cold_counterfactual_latency", [
            "observed_service_latency_ms", "cold_counterfactual_service_latency_ms",
            "transfer_latency_ms", "stall_or_restart_latency_ms",
        ], 0, request_count),
        "latency_saved_sum_ms": None, "latency_saved_per_request_ms": None,
        "latency_saved_per_hit_ms": None, "latency_saved_per_resident_mb_ms": None,
    }
    return CacheEfficiencyMetricSummary(
        availability="available", cache_event_schema_version=schema_version,
        request_metrics=request_metrics, byte_metrics=byte_metrics, lifecycle_metrics=lifecycle_metrics,
        capacity_metrics=capacity_metrics, pollution_metrics=_pollution_metrics(requests, trace_context),
        future_reuse_proxy_metrics=_future_reuse(requests, horizons), latency_saved_metrics=latency,
        type_aware_metrics=_type_aware_metrics(requests, trace_context),
    )


def reduce_cache_efficiency_summary(summary: Mapping[str, Any], *, reuse_horizons: Iterable[int] = DEFAULT_REUSE_HORIZONS) -> CacheEfficiencyMetricSummary:
    if "cache_event_trace" not in summary:
        missing = _group("unavailable", "missing_cache_event_trace", ["cache_event_trace"], 0, 0)
        return CacheEfficiencyMetricSummary(
            availability="unavailable", cache_event_schema_version=None,
            request_metrics=dict(missing), byte_metrics=dict(missing), lifecycle_metrics=dict(missing),
            capacity_metrics=dict(missing), pollution_metrics=dict(missing),
            future_reuse_proxy_metrics=dict(missing), latency_saved_metrics=dict(missing),
            type_aware_metrics=dict(missing),
        )
    trace = summary["cache_event_trace"]
    if not isinstance(trace, list):
        raise ValueError("cache_event_trace must be a list")
    version = summary.get("cache_event_schema_version")
    if version is None:
        raise ValueError("cache_event_schema_version is required when cache_event_trace is present")
    return reduce_cache_efficiency_events(trace, schema_version=str(version), trace_context=summary.get("cache_trace_context"), reuse_horizons=reuse_horizons)


def cache_efficiency_row_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return lightweight nullable fields for benchmark rows and generic aggregates."""
    result = reduce_cache_efficiency_summary(summary)
    stored = summary.get("cache_efficiency_metrics")
    if stored is not None and stored != result.to_dict():
        raise ValueError("stored cache efficiency metrics do not reconcile with raw CacheEvent")
    if result.availability != "available":
        return {"cache_efficiency_availability": "unavailable"}
    type_aware = result.type_aware_metrics
    return {
        "cache_efficiency_availability": "available",
        "cache_object_hit_rate": result.request_metrics.get("object_hit_rate"),
        "cache_byte_hit_rate": result.byte_metrics.get("byte_hit_rate"),
        "cache_churn_mb": result.lifecycle_metrics.get("cache_churn_mb"),
        "cache_pollution_ratio": result.pollution_metrics.get("cache_pollution_ratio"),
        "cache_transfer_amplification_ratio": result.lifecycle_metrics.get("transfer_amplification_ratio"),
        "cache_capacity_mean_occupancy": result.capacity_metrics.get("mean_occupancy"),
        "cache_latency_saved_sum_ms": result.latency_saved_metrics.get("latency_saved_sum_ms"),
        "cache_base_model_hit_rate": type_aware.get("base_hit_rate"),
        "cache_adapter_hit_rate": type_aware.get("adapter_hit_rate"),
        "cache_joint_model_hit_rate": type_aware.get("joint_base_adapter_hit_rate"),
        "cache_full_service_ready_rate": type_aware.get("full_service_ready_rate"),
        "cache_base_transfer_mb": type_aware.get("base_model_transfer_mb"),
        "full_service_ready_byte_hit_rate": type_aware.get(
            "full_service_ready_byte_hit_rate"
        ),
        "joint_base_adapter_hit_rate": type_aware.get(
            "joint_base_adapter_hit_rate"
        ),
        "full_service_ready_request_rate": type_aware.get(
            "full_service_ready_request_rate"
        ),
        "transfer_mb_per_request": type_aware.get("transfer_mb_per_request"),
        "requested_dependency_byte_coverage_rate": type_aware.get(
            "requested_dependency_byte_coverage_rate"
        ),
        "requested_service_dependency_mb": type_aware.get(
            "requested_service_dependency_mb"
        ),
        "full_service_ready_dependency_mb": type_aware.get(
            "full_service_ready_dependency_mb"
        ),
        "base_model_transfer_mb": type_aware.get("base_model_transfer_mb"),
        "adapter_transfer_mb": type_aware.get("adapter_transfer_mb"),
        "workflow_state_migration_transfer_mb": type_aware.get(
            "workflow_state_migration_transfer_mb"
        ),
        "other_typed_transfer_mb": type_aware.get("other_typed_transfer_mb"),
        "primary_transfer_mb": type_aware.get("primary_transfer_mb"),
    }
