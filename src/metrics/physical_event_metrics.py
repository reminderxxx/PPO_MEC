"""Independent, strict reduction of physical execution events.

The reducer reads only event records, never environment state or step metrics.
Every transfer reports its expected total at start and its actual byte deltas
in progress/completion records. Completed requests require a delivery event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from typing import Any, Iterable, Mapping


PHYSICAL_EVENT_SCHEMA_VERSION = "1.0.0"
_KINDS = {
    "request_created", "request_completed", "request_failed",
    "wait_started", "wait_ended", "wait_cancelled",
    "transfer_started", "transfer_progress", "transfer_completed", "transfer_cancelled",
    "compute_started", "compute_progress", "compute_completed", "compute_cancelled",
    "delivery_started", "delivery_completed", "delivery_failed",
}
_START_KIND = {
    "wait_started": "wait", "transfer_started": "transfer",
    "compute_started": "compute", "delivery_started": "delivery",
}
_END_KIND = {
    "wait_ended": "wait", "wait_cancelled": "wait",
    "transfer_completed": "transfer", "transfer_cancelled": "transfer",
    "compute_completed": "compute", "compute_cancelled": "compute",
    "delivery_completed": "delivery", "delivery_failed": "delivery",
}
_CANCEL_KINDS = {"wait_cancelled", "transfer_cancelled", "compute_cancelled", "delivery_failed"}


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class PhysicalEvent:
    event_id: str
    event_type: str
    time_s: float
    request_id: str
    workflow_id: str | None = None
    node_id: str | None = None
    operation_id: str | None = None
    object_id: str | None = None
    resource_id: str | None = None
    wait_reason: str | None = None
    transfer_kind: str | None = None
    bytes_delta: int | None = None
    total_bytes: int | None = None
    work_units_delta: float | None = None
    total_work_units: float | None = None
    schema_version: str = PHYSICAL_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestPhysicalMetrics:
    request_id: str
    workflow_id: str | None
    availability: str
    created_at_s: float
    completed_at_s: float | None
    latency_s: float | None
    wait_s: float | None
    wait_by_reason_s: dict[str, float] | None
    transfer_s: float | None
    transfer_bytes: int | None
    compute_s: float | None
    compute_work_units: float | None
    delivery_s: float | None
    observed_transfer_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalEventMetricSummary:
    schema_version: str
    availability: str
    requests: tuple[RequestPhysicalMetrics, ...]
    total_latency_s: float | None
    total_transfer_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Operation:
    kind: str
    event: PhysicalEvent
    moved_bytes: int = 0
    work_units: float = 0.0
    finished: bool = False
    finished_at_s: float | None = None
    cancelled: bool = False


@dataclass
class _Request:
    created: PhysicalEvent
    terminal: PhysicalEvent | None = None
    operations: dict[str, _Operation] = field(default_factory=dict)
    delivery_completed: bool = False


def _event(raw: PhysicalEvent | Mapping[str, Any], index: int) -> PhysicalEvent:
    if isinstance(raw, PhysicalEvent):
        event = raw
    elif isinstance(raw, Mapping):
        try:
            event = PhysicalEvent(**{name: raw[name] for name in PhysicalEvent.__dataclass_fields__ if name in raw})
        except (TypeError, KeyError) as exc:
            raise ValueError(f"invalid physical event at index {index}") from exc
    else:
        raise ValueError(f"physical event at index {index} is not a mapping or PhysicalEvent")
    if event.schema_version != PHYSICAL_EVENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported physical event schema: {event.schema_version}")
    if event.event_type not in _KINDS:
        raise ValueError(f"unknown physical event_type: {event.event_type}")
    for name in ("event_id", "request_id"):
        value = getattr(event, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    _nonnegative_number(event.time_s, "time_s")
    for name in ("bytes_delta", "total_bytes"):
        value = getattr(event, name)
        if value is not None:
            _nonnegative_int(value, name)
    for name in ("work_units_delta", "total_work_units"):
        value = getattr(event, name)
        if value is not None:
            _nonnegative_number(value, name)
    if event.event_type in _START_KIND or event.event_type in _END_KIND or event.event_type in ("transfer_progress", "compute_progress"):
        if not isinstance(event.operation_id, str) or not event.operation_id.strip():
            raise ValueError(f"{event.event_type} requires operation_id")
    return event


def _same_operation_identity(start: PhysicalEvent, event: PhysicalEvent) -> None:
    for name in ("request_id", "workflow_id", "node_id", "object_id", "resource_id", "wait_reason", "transfer_kind"):
        if getattr(start, name) != getattr(event, name):
            raise ValueError(f"operation {start.operation_id} changed {name}")
    for name in ("total_bytes", "total_work_units"):
        value = getattr(event, name)
        if value is not None and value != getattr(start, name):
            raise ValueError(f"operation {start.operation_id} changed {name}")


def reduce_physical_events(events: Iterable[PhysicalEvent | Mapping[str, Any]]) -> PhysicalEventMetricSummary:
    """Recompute physical metrics, rejecting corrupt or causally invalid ledgers."""
    requests: dict[str, _Request] = {}
    operation_owner: dict[str, str] = {}
    event_ids: set[str] = set()
    previous_time = 0.0
    for index, raw in enumerate(events):
        event = _event(raw, index)
        if event.event_id in event_ids:
            raise ValueError(f"duplicate physical event_id: {event.event_id}")
        event_ids.add(event.event_id)
        if event.time_s < previous_time:
            raise ValueError("physical events are out of time order")
        previous_time = event.time_s
        if event.event_type == "request_created":
            if event.request_id in requests:
                raise ValueError(f"duplicate request_created: {event.request_id}")
            requests[event.request_id] = _Request(created=event)
            continue
        request = requests.get(event.request_id)
        if request is None:
            raise ValueError(f"event before request_created: {event.request_id}")
        if request.terminal is not None:
            raise ValueError(f"event after terminal request: {event.request_id}")
        if event.workflow_id != request.created.workflow_id:
            raise ValueError(f"request {event.request_id} changed workflow_id")
        if event.event_type in ("request_completed", "request_failed"):
            if any(not operation.finished for operation in request.operations.values()):
                raise ValueError("terminal request has unfinished operations")
            if event.event_type == "request_completed" and any(operation.cancelled for operation in request.operations.values()):
                raise ValueError("completed request contains cancelled operations")
            if event.event_type == "request_completed" and not request.delivery_completed:
                raise ValueError("completed request requires delivery_completed")
            request.terminal = event
            continue
        assert event.operation_id is not None
        if event.event_type in _START_KIND:
            if event.operation_id in operation_owner:
                raise ValueError(f"duplicate operation_id: {event.operation_id}")
            kind = _START_KIND[event.event_type]
            if kind == "transfer":
                if event.total_bytes is None or event.bytes_delta not in (None, 0):
                    raise ValueError("transfer_started requires total_bytes and zero initial bytes")
                if not event.object_id or not event.resource_id or not event.transfer_kind:
                    raise ValueError("transfer_started requires object_id, resource_id and transfer_kind")
            if kind == "compute":
                if event.total_work_units is None or event.work_units_delta not in (None, 0):
                    raise ValueError("compute_started requires total_work_units and zero initial work")
                if not event.node_id or not event.resource_id:
                    raise ValueError("compute_started requires node_id and resource_id")
            if kind == "wait" and not event.wait_reason:
                raise ValueError("wait_started requires wait_reason")
            operation_owner[event.operation_id] = event.request_id
            request.operations[event.operation_id] = _Operation(kind, event)
            continue
        if operation_owner.get(event.operation_id) != event.request_id:
            raise ValueError(f"unknown or foreign operation_id: {event.operation_id}")
        operation = request.operations[event.operation_id]
        if operation.finished:
            raise ValueError(f"event after operation completion: {event.operation_id}")
        if event.time_s < operation.event.time_s:
            raise ValueError("operation completed before it started")
        _same_operation_identity(operation.event, event)
        if event.event_type == "transfer_progress":
            if operation.kind != "transfer" or event.bytes_delta is None:
                raise ValueError("transfer_progress requires active transfer and bytes_delta")
            operation.moved_bytes += event.bytes_delta
            if operation.moved_bytes > operation.event.total_bytes:
                raise ValueError("transfer bytes exceed total_bytes")
            continue
        if event.event_type == "compute_progress":
            if operation.kind != "compute" or event.work_units_delta is None:
                raise ValueError("compute_progress requires active compute and work_units_delta")
            operation.work_units += event.work_units_delta
            if operation.work_units > operation.event.total_work_units + 1e-9:
                raise ValueError("compute work units exceed total_work_units")
            continue
        if _END_KIND[event.event_type] != operation.kind:
            raise ValueError("operation completion type mismatch")
        if operation.kind == "transfer":
            if event.bytes_delta is None:
                raise ValueError("transfer end requires bytes_delta")
            operation.moved_bytes += event.bytes_delta
            if operation.moved_bytes > operation.event.total_bytes:
                raise ValueError("transfer bytes exceed total_bytes")
            if event.event_type == "transfer_completed" and operation.moved_bytes != operation.event.total_bytes:
                raise ValueError("transfer byte conservation failed")
        elif operation.kind == "compute":
            if event.work_units_delta is None:
                raise ValueError("compute end requires work_units_delta")
            operation.work_units += event.work_units_delta
            if operation.work_units > operation.event.total_work_units + 1e-9:
                raise ValueError("compute work units exceed total_work_units")
            if event.event_type == "compute_completed" and not isclose(operation.work_units, operation.event.total_work_units, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("compute work-unit conservation failed")
        if operation.kind == "delivery":
            request.delivery_completed = True
        operation.finished = True
        operation.cancelled = event.event_type in _CANCEL_KINDS
        # Finish time is retained separately from the start event.
        operation.finished_at_s = event.time_s

    summaries: list[RequestPhysicalMetrics] = []
    for request_id, request in requests.items():
        operations = tuple(request.operations.values())
        observed_bytes = sum(op.moved_bytes for op in operations if op.kind == "transfer")
        terminal = request.terminal
        completed = terminal is not None and terminal.event_type == "request_completed"
        def elapsed(kind: str) -> float:
            return sum(op.finished_at_s - op.event.time_s for op in operations if op.kind == kind and op.finished_at_s is not None)
        wait_by_reason: dict[str, float] = {}
        if completed:
            for op in operations:
                if op.kind == "wait" and op.finished_at_s is not None:
                    reason = op.event.wait_reason
                    assert reason is not None
                    wait_by_reason[reason] = wait_by_reason.get(reason, 0.0) + op.finished_at_s - op.event.time_s
        summaries.append(RequestPhysicalMetrics(
            request_id=request_id,
            workflow_id=request.created.workflow_id,
            availability="available" if completed else "unavailable",
            created_at_s=request.created.time_s,
            completed_at_s=terminal.time_s if completed else None,
            latency_s=terminal.time_s - request.created.time_s if completed else None,
            wait_s=elapsed("wait") if completed else None,
            wait_by_reason_s=dict(sorted(wait_by_reason.items())) if completed else None,
            transfer_s=elapsed("transfer") if completed else None,
            transfer_bytes=observed_bytes if completed else None,
            compute_s=elapsed("compute") if completed else None,
            compute_work_units=sum(op.work_units for op in operations if op.kind == "compute") if completed else None,
            delivery_s=elapsed("delivery") if completed else None,
            observed_transfer_bytes=observed_bytes,
        ))
    all_complete = bool(summaries) and all(row.availability == "available" for row in summaries)
    return PhysicalEventMetricSummary(
        schema_version=PHYSICAL_EVENT_SCHEMA_VERSION,
        availability="available" if all_complete else "unavailable",
        requests=tuple(summaries),
        total_latency_s=sum(row.latency_s for row in summaries if row.latency_s is not None) if all_complete else None,
        total_transfer_bytes=sum(row.transfer_bytes for row in summaries if row.transfer_bytes is not None) if all_complete else None,
    )
