"""Policy-neutral request exposure contract for formal typed-cache execution.

The exposure trace is created before an agent acts.  It contains workload and
mobility identities only; cache, service, reward, eviction, transfer, and
workflow outcomes are recorded separately after execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable, Mapping


FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION = "1.0.0"
FORMAL_REQUEST_EXPOSURE_TRACE_VERSION = "1.0.0"
FORMAL_ENDPOINT_METRICS_CONTRACT_VERSION = "2.0.0"
FORMAL_OUTCOME_TRACE_VERSION = "1.0.0"

OUTCOME_FIELDS = frozenset(
    {
        "action_id",
        "action_name",
        "admission_added",
        "admission_requested",
        "cache_hit",
        "cache_state",
        "evicted_object_id",
        "evicted_object_ids",
        "hit_source",
        "outcome",
        "reward",
        "service_result",
        "service_success",
        "stall_occurred",
        "transfer_mb",
        "victim",
        "workflow_completed",
    }
)
FUTURE_FIELDS = frozenset(
    {
        "actual_next_rsu_id",
        "future_associations",
        "future_frame",
        "future_topology",
        "oracle_action",
        "oracle_future",
    }
)
REQUIRED_REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "request_kind",
        "request_order",
        "step_index",
        "time_index",
        "vehicle_id",
        "workflow_id",
        "node_id",
        "required_base_model",
        "adapter_id",
        "object_id",
        "object_size_mb",
        "request_rsu_id",
        "current_service_rsu_id",
        "eligible_service_rsu_ids",
        "eligible_cache_target_rsu_ids",
        "requested_typed_objects",
        "dependency_bundle",
        "dag_provenance",
        "oracle_only_future_topology",
    }
)


class FormalRequestExposureError(ValueError):
    """Raised when exposure execution is ambiguous, endogenous, or leaky."""


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormalRequestExposureError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FormalRequestExposureError(f"non-string JSON key at {path}")
            _validate_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json(child, f"{path}[{index}]")
        return
    raise FormalRequestExposureError(f"non-JSON value at {path}")


def _walk_forbidden(value: Any, *, path: str, oracle_scope: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            next_oracle = oracle_scope or key == "oracle_only_future_topology"
            if key in OUTCOME_FIELDS:
                raise FormalRequestExposureError(
                    f"outcome field contaminates request exposure: {path}.{key}"
                )
            if key in FUTURE_FIELDS and not next_oracle:
                raise FormalRequestExposureError(
                    f"future field is outside oracle-only scope: {path}.{key}"
                )
            _walk_forbidden(child, path=f"{path}.{key}", oracle_scope=next_oracle)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]", oracle_scope=oracle_scope)


def request_exposure_fingerprint(trace: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(trace))
    payload.pop("request_exposure_fingerprint", None)
    payload.pop("validation", None)
    return canonical_sha256(payload)


def build_formal_request_exposure_trace(
    *,
    evaluation_unit: Mapping[str, Any],
    workflow_state: Any,
    mobility_frames: Iterable[Mapping[str, Any]],
    rsu_mapper: Any,
    adapter_catalog: Any,
    primary_vehicle_id: str,
    primary_vehicle_selection: str,
    max_steps: int,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one immutable trace from frozen workload and mobility inputs."""

    frames = [deepcopy(dict(frame)) for frame in mobility_frames]
    if len(frames) < 2:
        raise FormalRequestExposureError("request exposure needs reset plus one step frame")
    node_map = workflow_state.node_map()
    request_count = min(len(workflow_state.execution_order), int(max_steps), len(frames) - 1)
    associations = [
        rsu_mapper.associate(frame.get("vehicles", []))
        for frame in frames[: request_count + 1]
    ]
    requests: list[dict[str, Any]] = []
    unit_id = str(evaluation_unit["evaluation_unit_id"])
    for offset, node_id in enumerate(workflow_state.execution_order[:request_count], start=1):
        node = node_map[str(node_id)]
        adapter = adapter_catalog.get_typed_adapter(node.required_adapter)
        placement = adapter_catalog.resolve_typed_placement_plan(
            adapter_id=node.required_adapter,
            resident_object_ids=[],
        )
        typed_rows = []
        for object_id in placement.ordered_object_ids:
            item = adapter_catalog.get_typed_object(object_id)
            typed_rows.append(
                {
                    "object_id": item.object_id,
                    "object_type": item.object_type,
                    "adapter_id": item.adapter_id,
                    "base_model_id": item.base_model_id,
                    "required_base_model_id": item.required_base_model_id,
                    "resident_size_mb": float(item.resident_size_mb),
                    "transfer_size_mb": float(item.transfer_size_mb),
                }
            )
        request_rsu = associations[offset - 1].get(primary_vehicle_id)
        current_rsu = associations[offset].get(primary_vehicle_id)
        next_rsu = (
            associations[offset + 1].get(primary_vehicle_id)
            if offset + 1 < len(associations)
            else None
        )
        eligible = sorted([current_rsu] if current_rsu is not None else [])
        requests.append(
            {
                "request_id": f"{unit_id}/request_{offset:06d}",
                "request_kind": "service_request",
                "request_order": offset - 1,
                "step_index": offset,
                "time_index": int(frames[offset]["time_index"]),
                "vehicle_id": str(primary_vehicle_id),
                "workflow_id": str(workflow_state.workflow_id),
                "node_id": str(node.node_id),
                "required_base_model": str(node.required_base_model),
                "adapter_id": str(node.required_adapter),
                "object_id": str(adapter.object_id),
                "object_size_mb": float(adapter.resident_size_mb),
                "request_rsu_id": request_rsu,
                "current_service_rsu_id": current_rsu,
                "eligible_service_rsu_ids": eligible,
                "eligible_cache_target_rsu_ids": eligible,
                "model_cache_profile_id": "typed_base_adapter_state_v1",
                "typed_model_cache_contract_version": "1.0.0",
                "catalog_fingerprint": str(adapter_catalog.canonical_fingerprint()),
                "requested_typed_objects": typed_rows,
                "dependency_bundle": {
                    **placement.to_dict(),
                    "ordered_object_ids": list(placement.ordered_object_ids),
                },
                "dag_provenance": {
                    "execution_order_index": offset - 1,
                    "predecessors": sorted(node.predecessors),
                    "successors": sorted(node.successors),
                    "predecessor_failure_semantics": "expose_successor_without_retry_or_suppression",
                },
                "oracle_only_future_topology": {
                    "actual_next_rsu_id": next_rsu,
                    "actor_visible": False,
                    "controller_visible": False,
                },
            }
        )
    trace = {
        "formal_request_exposure_trace_version": FORMAL_REQUEST_EXPOSURE_TRACE_VERSION,
        "formal_exogenous_request_execution_contract_version": (
            FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION
        ),
        "producer": {
            "identity": "formal_request_exposure_producer_v1.0.0",
            "policy_neutral": True,
            "executes_agent": False,
            "executes_cache_service_or_workflow_outcome": False,
        },
        "evaluation_unit": deepcopy(dict(evaluation_unit)),
        "source_provenance": deepcopy(dict(source_provenance)),
        "execution_semantics": {
            "mode": "replay_driven_exogenous_request_exposure",
            "default_enabled": False,
            "one_request_per_step": True,
            "action_before_lookup": True,
            "service_failure_changes_future_exposure": False,
            "retry_policy": "no_endogenous_retry",
            "primary_vehicle_selection": str(primary_vehicle_selection),
            "request_denominator": "all_exposed_event_type_request_rows",
            "workflow_outcome_source": "request_outcomes_only_not_exposure_progression",
        },
        "requests": requests,
        "exposure_censoring": {
            "right_censored": request_count < len(workflow_state.execution_order),
            "planned_dag_node_count": len(workflow_state.execution_order),
            "exposed_request_count": request_count,
            "reason": (
                "max_steps_or_mobility_horizon"
                if request_count < len(workflow_state.execution_order)
                else "not_censored"
            ),
        },
    }
    trace["request_exposure_fingerprint"] = request_exposure_fingerprint(trace)
    validate_formal_request_exposure_trace(trace)
    return trace


def validate_formal_request_exposure_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    _validate_json(trace)
    _walk_forbidden(trace, path="trace")
    if trace.get("formal_request_exposure_trace_version") != FORMAL_REQUEST_EXPOSURE_TRACE_VERSION:
        raise FormalRequestExposureError("unsupported formal request exposure trace version")
    if trace.get("formal_exogenous_request_execution_contract_version") != (
        FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION
    ):
        raise FormalRequestExposureError("formal execution contract version mismatch")
    producer = trace.get("producer") or {}
    if producer.get("policy_neutral") is not True or producer.get("executes_agent") is not False:
        raise FormalRequestExposureError("request exposure producer is not policy-neutral")
    semantics = trace.get("execution_semantics") or {}
    if semantics.get("mode") != "replay_driven_exogenous_request_exposure":
        raise FormalRequestExposureError("formal mode cannot fall back to endogenous progression")
    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise FormalRequestExposureError("formal request exposure is missing or empty")
    seen: set[str] = set()
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise FormalRequestExposureError(f"request[{index}] is not an object")
        missing = sorted(REQUIRED_REQUEST_FIELDS - set(request))
        if missing:
            raise FormalRequestExposureError(f"request[{index}] missing fields: {missing}")
        request_id = str(request["request_id"])
        if request_id in seen:
            raise FormalRequestExposureError(f"duplicate request exposure: {request_id}")
        seen.add(request_id)
        if int(request["request_order"]) != index or int(request["step_index"]) != index + 1:
            raise FormalRequestExposureError("request exposure is missing or out of order")
        for field in ("eligible_service_rsu_ids", "eligible_cache_target_rsu_ids"):
            rows = request[field]
            if not isinstance(rows, list) or rows != sorted(set(rows)):
                raise FormalRequestExposureError(f"request[{index}].{field} is not canonical")
        if request.get("request_kind") not in {"service_request", "not_applicable"}:
            raise FormalRequestExposureError("unsupported request_kind")
        if request.get("request_kind") == "not_applicable":
            if any(
                request.get(field) is not None
                for field in ("node_id", "adapter_id", "object_id", "object_size_mb")
            ):
                raise FormalRequestExposureError("not-applicable exposure carries object identity")
            if request.get("requested_typed_objects") or request.get("dependency_bundle"):
                raise FormalRequestExposureError("not-applicable exposure carries dependency")
            continue
        typed = request["requested_typed_objects"]
        ordered = list((request["dependency_bundle"] or {}).get("ordered_object_ids") or [])
        if not isinstance(typed, list) or not 1 <= len(typed) <= 2:
            raise FormalRequestExposureError("typed dependency bundle must contain one or two objects")
        if ordered != [row.get("object_id") for row in typed]:
            raise FormalRequestExposureError("typed dependency order mismatch")
        if typed[-1].get("object_type") != "adapter":
            raise FormalRequestExposureError("typed dependency bundle must end with adapter")
        if len(typed) == 2 and typed[0].get("object_type") != "base_model":
            raise FormalRequestExposureError("typed dependency bundle must be base then adapter")
    expected = trace.get("request_exposure_fingerprint")
    if not isinstance(expected, str) or expected != request_exposure_fingerprint(trace):
        raise FormalRequestExposureError("request exposure canonical fingerprint mismatch")
    round_trip = json.loads(json.dumps(trace, ensure_ascii=False, allow_nan=False))
    if round_trip != trace:
        raise FormalRequestExposureError("request exposure JSON round-trip drift")
    return {
        "status": "pass",
        "request_count": len(requests),
        "request_exposure_fingerprint": expected,
        "future_fields_actor_visible": False,
        "outcome_fields_present": False,
    }


def align_cache_event_to_request(
    event: Mapping[str, Any], request: Mapping[str, Any], *, trace_fingerprint: str
) -> dict[str, Any]:
    """Return a CacheEvent 1.3 companion row after strict identity alignment."""

    checks = {
        "episode_step_index": (event.get("episode_step_index"), request.get("step_index")),
        "time_index": (event.get("time_index"), request.get("time_index")),
        "vehicle_id": (event.get("vehicle_id"), request.get("vehicle_id")),
        "workflow_id": (event.get("workflow_id"), request.get("workflow_id")),
        "node_id": (event.get("node_id"), request.get("node_id")),
        "object_id": (event.get("object_id"), request.get("object_id")),
        "adapter_id": (event.get("adapter_id"), request.get("adapter_id")),
        "size_mb": (event.get("size_mb"), request.get("object_size_mb")),
        "request_rsu_id": (event.get("request_rsu_id"), request.get("request_rsu_id")),
        "requested_typed_objects": (
            event.get("requested_typed_objects"),
            request.get("requested_typed_objects"),
        ),
        "dependency_bundle": (
            event.get("dependency_bundle"),
            request.get("dependency_bundle"),
        ),
    }
    drift = [name for name, (observed, expected) in checks.items() if observed != expected]
    if drift:
        raise FormalRequestExposureError(
            "observed CacheEvent does not match request exposure: " + ", ".join(drift)
        )
    row = deepcopy(dict(event))
    row.update(
        formal_request_alignment_version="1.0.0",
        formal_request_id=str(request["request_id"]),
        formal_request_order=int(request["request_order"]),
        request_exposure_fingerprint=str(trace_fingerprint),
        request_alignment_status="matched_exactly_once",
    )
    return row


def build_outcome_audit(events: Iterable[Mapping[str, Any]], trace: Mapping[str, Any]) -> dict[str, Any]:
    requests = list(trace["requests"])
    rows = [deepcopy(dict(event)) for event in events]
    if len(rows) != len(requests):
        raise FormalRequestExposureError("missing or extra request-level CacheEvent")
    aligned = [
        align_cache_event_to_request(
            event,
            request,
            trace_fingerprint=str(trace["request_exposure_fingerprint"]),
        )
        for event, request in zip(rows, requests)
    ]
    request_ids = [row["formal_request_id"] for row in aligned]
    if len(request_ids) != len(set(request_ids)):
        raise FormalRequestExposureError("duplicate request-level CacheEvent")
    outcome_projection = [
        {
            "formal_request_id": row["formal_request_id"],
            "action_id": row.get("action_id"),
            "cache_hit": row.get("cache_hit"),
            "full_service_ready": row.get("full_service_ready"),
            "service_success": row.get("service_success"),
            "stall_occurred": row.get("stall_occurred"),
            "evicted_object_ids": row.get("evicted_object_ids"),
            "transfer_mb_by_type": row.get("transfer_mb_by_type"),
        }
        for row in aligned
    ]
    return {
        "formal_outcome_trace_version": FORMAL_OUTCOME_TRACE_VERSION,
        "request_exposure_fingerprint": trace["request_exposure_fingerprint"],
        "outcome_fingerprint": canonical_sha256(outcome_projection),
        "request_count": len(requests),
        "cache_event_count": len(aligned),
        "alignment_status": "pass",
        "aligned_events": aligned,
    }


def compute_formal_endpoint_metrics(
    events: Iterable[Mapping[str, Any]], trace: Mapping[str, Any], *, truncated: bool
) -> dict[str, Any]:
    """Compute v2 endpoints on the common exogenous request denominator."""

    audit = build_outcome_audit(events, trace)
    rows = audit["aligned_events"]
    request_rows = [row for row in rows if row.get("event_type") == "request"]
    denominator = len(request_rows)
    if denominator == 0:
        raise FormalRequestExposureError("formal endpoint denominator is zero")
    ready_count = sum(bool(row.get("full_service_ready")) for row in request_rows)
    joint_count = sum(bool(row.get("joint_model_hit")) for row in request_rows)
    transfer = sum(
        sum(
            float(value)
            for object_type, value in dict(row.get("transfer_mb_by_type") or {}).items()
            if object_type in {"base_model", "adapter", "workflow_state"}
        )
        for row in request_rows
    )
    byte_denominator = 0.0
    byte_numerator = 0.0
    for row in request_rows:
        request_bytes = sum(
            float(item["resident_size_mb"])
            for item in row.get("requested_typed_objects") or []
            if item.get("object_type") in {"base_model", "adapter"}
        )
        byte_denominator += request_bytes
        if row.get("full_service_ready"):
            byte_numerator += request_bytes
    all_success = all(bool(row.get("service_success")) for row in request_rows)
    workflow_complete = bool(
        not truncated
        and not (trace.get("exposure_censoring") or {}).get("right_censored", False)
        and len(rows) == len(trace["requests"])
        and all_success
    )
    delay = None
    delay_availability = "unavailable_failed_or_incomplete_workflow"
    if workflow_complete:
        delay = float(request_rows[-1]["time_index"] - request_rows[0]["time_index"] + 1)
        delay_availability = "available_completed_workflow"
    return {
        "formal_endpoint_metrics_contract_version": FORMAL_ENDPOINT_METRICS_CONTRACT_VERSION,
        "external_request_denominator": denominator,
        "request_alignment_status": audit["alignment_status"],
        "full_service_ready_byte_hit_rate": (
            byte_numerator / byte_denominator if byte_denominator > 0 else None
        ),
        "joint_base_adapter_hit_rate": joint_count / denominator,
        "full_service_ready_request_rate": ready_count / denominator,
        "transfer_mb_per_request": transfer / denominator,
        "workflow_continuity_rate": sum(
            bool(row.get("service_success")) for row in request_rows
        )
        / denominator,
        "end_to_end_workflow_delay": delay,
        "end_to_end_workflow_delay_availability": delay_availability,
        "workflow_completed_under_exogenous_execution": workflow_complete,
        "right_censored": bool(
            truncated
            or len(rows) < len(trace["requests"])
            or (trace.get("exposure_censoring") or {}).get("right_censored", False)
        ),
        "failed_workflow_in_delay_denominator": True,
        "failed_workflow_delay_value": None if not workflow_complete else delay,
        "request_exposure_fingerprint": trace["request_exposure_fingerprint"],
        "outcome_fingerprint": audit["outcome_fingerprint"],
    }


__all__ = [
    "FORMAL_ENDPOINT_METRICS_CONTRACT_VERSION",
    "FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION",
    "FORMAL_REQUEST_EXPOSURE_TRACE_VERSION",
    "FormalRequestExposureError",
    "align_cache_event_to_request",
    "build_formal_request_exposure_trace",
    "build_outcome_audit",
    "canonical_sha256",
    "compute_formal_endpoint_metrics",
    "request_exposure_fingerprint",
    "validate_formal_request_exposure_trace",
]
