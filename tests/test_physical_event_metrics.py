"""Contract checks for cost replay and an independent physical event ledger."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.metrics.physical_event_metrics import PhysicalEvent, reduce_physical_events
from src.runtime.physical_cost_profile import (
    CostParameter,
    PhysicalCostProfile,
    ReplayIdentity,
    ReplayOutputGate,
)


def _profile() -> PhysicalCostProfile:
    identity = ReplayIdentity.from_components(
        model_identity="model@revision-a",
        input_value={"prompt": "upstream output"},
        generation_config={"max_tokens": 16, "temperature": 0},
    )
    return PhysicalCostProfile(
        profile_id="fixture-profile",
        replay_identity=identity,
        cold_load=CostParameter(4.0, "synthetic", "test fixture"),
        hot_call=CostParameter(1.0, "synthetic", "test fixture"),
        adapter_switch=CostParameter(0.25, "synthetic", "test fixture"),
        state_restore=CostParameter(0.5, "synthetic", "test fixture"),
    )


def test_cost_profile_has_explicit_units_provenance_and_replay_identity() -> None:
    profile = _profile()
    assert (profile.schema_version, profile.time_unit, profile.size_unit, profile.work_unit) == (
        "1.0.0", "s", "byte", "work_unit"
    )
    assert profile.evidence == "synthetic"
    assert profile.replay_identity == ReplayIdentity.from_components(
        model_identity="model@revision-a",
        input_value={"prompt": "upstream output"},
        generation_config={"temperature": 0, "max_tokens": 16},
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        profile.validate_replay_identity(replace(profile.replay_identity, input_identity="other"))
    with pytest.raises(ValueError, match="finite"):
        CostParameter(float("nan"), "measured", "sample")
    with pytest.raises(ValueError, match="source"):
        CostParameter(1, "measured", "")
    with pytest.raises(ValueError, match="units"):
        replace(profile, size_unit="MiB")


def test_replay_output_requires_matching_identity_and_completion_event() -> None:
    profile = _profile()
    identity = profile.replay_identity
    output = {"text": "result"}
    gate = ReplayOutputGate(profile=profile, output=output, earliest_completion_s=5)
    output["text"] = "mutated after scheduling"
    with pytest.raises(ValueError, match="unavailable"):
        gate.read(time_s=50, identity=identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        gate.mark_completed(time_s=5, identity=replace(identity, generation_identity="other"))
    with pytest.raises(ValueError, match="before simulated"):
        gate.mark_completed(time_s=4, identity=identity)
    gate.mark_completed(time_s=5, identity=identity)
    with pytest.raises(ValueError, match="unavailable"):
        gate.read(time_s=4.9, identity=identity)
    assert gate.read(time_s=5, identity=identity) == {"text": "result"}
    revealed = gate.read(time_s=5, identity=identity)
    revealed["text"] = "mutated after release"
    assert gate.read(time_s=5, identity=identity) == {"text": "result"}
    with pytest.raises(ValueError, match="already recorded"):
        gate.mark_completed(time_s=6, identity=identity)


def _valid_events() -> list[PhysicalEvent]:
    common = {"request_id": "request-1", "workflow_id": "workflow-1"}
    return [
        PhysicalEvent("e1", "request_created", 0, **common),
        PhysicalEvent("e2", "wait_started", 0, operation_id="wait-1", wait_reason="model", **common),
        PhysicalEvent("e3", "transfer_started", 1, operation_id="tx-1", object_id="adapter-1", resource_id="backhaul-1", transfer_kind="model", total_bytes=100, **common),
        PhysicalEvent("e4", "transfer_progress", 2, operation_id="tx-1", object_id="adapter-1", resource_id="backhaul-1", transfer_kind="model", bytes_delta=60, **common),
        PhysicalEvent("e5", "transfer_completed", 3, operation_id="tx-1", object_id="adapter-1", resource_id="backhaul-1", transfer_kind="model", bytes_delta=40, **common),
        PhysicalEvent("e6", "wait_ended", 3, operation_id="wait-1", wait_reason="model", **common),
        PhysicalEvent("e7", "compute_started", 4, operation_id="compute-1", node_id="n0", resource_id="rsu-1", total_work_units=2, **common),
        PhysicalEvent("e8", "compute_completed", 6, operation_id="compute-1", node_id="n0", resource_id="rsu-1", work_units_delta=2, **common),
        PhysicalEvent("e9", "delivery_started", 6, operation_id="delivery-1", object_id="result-1", resource_id="wireless-1", **common),
        PhysicalEvent("e10", "delivery_completed", 7, operation_id="delivery-1", object_id="result-1", resource_id="wireless-1", **common),
        PhysicalEvent("e11", "request_completed", 7, **common),
    ]


def test_reducer_reconstructs_request_physical_metrics_from_events_only() -> None:
    summary = reduce_physical_events(_valid_events())
    row = summary.requests[0]
    assert summary.availability == "available"
    assert (row.latency_s, row.wait_s, row.transfer_s, row.compute_s, row.delivery_s) == (7, 3, 2, 2, 1)
    assert row.wait_by_reason_s == {"model": 3}
    assert (row.transfer_bytes, row.observed_transfer_bytes, row.compute_work_units) == (100, 100, 2)
    assert (summary.total_latency_s, summary.total_transfer_bytes) == (7, 100)
    assert reduce_physical_events([event.to_dict() for event in _valid_events()]) == summary


def test_incomplete_request_is_unavailable_but_observed_bytes_remain_auditable() -> None:
    summary = reduce_physical_events(_valid_events()[:4])
    row = summary.requests[0]
    assert summary.availability == "unavailable"
    assert row.latency_s is None
    assert row.transfer_bytes is None
    assert row.observed_transfer_bytes == 60
    assert summary.total_latency_s is None
    assert reduce_physical_events([]).availability == "unavailable"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(3, replace(rows[3], event_id="e3")), "duplicate physical event_id"),
        (lambda rows: rows.__setitem__(4, replace(rows[4], time_s=1.5)), "out of time order"),
        (lambda rows: rows.__setitem__(4, replace(rows[4], bytes_delta=39)), "byte conservation"),
        (lambda rows: rows.__setitem__(3, replace(rows[3], bytes_delta=-1)), "non-negative integer"),
        (lambda rows: rows.__setitem__(4, replace(rows[4], request_id="foreign")), "event before request_created"),
        (lambda rows: rows.__setitem__(4, replace(rows[4], object_id="another")), "changed object_id"),
        (lambda rows: rows.__setitem__(4, replace(rows[4], total_bytes=101)), "changed total_bytes"),
        (lambda rows: rows.__setitem__(7, replace(rows[7], work_units_delta=1)), "work-unit conservation"),
    ],
)
def test_reducer_rejects_corrupt_or_inconsistent_events(mutate, message: str) -> None:
    rows = _valid_events()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        reduce_physical_events(rows)


def test_request_completion_requires_delivery_and_closed_operations() -> None:
    rows = _valid_events()
    with pytest.raises(ValueError, match="unfinished operations"):
        reduce_physical_events(rows[:4] + [replace(rows[-1], event_id="early", time_s=3)])
    with pytest.raises(ValueError, match="requires delivery_completed"):
        reduce_physical_events(rows[:8] + [replace(rows[-1], event_id="no-delivery", time_s=6)])
    with pytest.raises(ValueError, match="after terminal request"):
        reduce_physical_events(rows + [replace(rows[3], event_id="late", time_s=8)])
    with pytest.raises(ValueError, match="duplicate operation_id"):
        reduce_physical_events(rows[:3] + [replace(rows[2], event_id="duplicate", time_s=1)])


def test_compute_progress_and_cancelled_transfer_preserve_partial_cost() -> None:
    rows = _valid_events()
    progress = replace(rows[7], event_id="compute-progress", event_type="compute_progress", time_s=5, work_units_delta=0.75)
    rows[7] = replace(rows[7], work_units_delta=1.25)
    rows.insert(7, progress)
    assert reduce_physical_events(rows).requests[0].compute_work_units == 2

    cancelled = [
        PhysicalEvent("a", "request_created", 0, request_id="r"),
        PhysicalEvent("b", "transfer_started", 0, request_id="r", operation_id="t", object_id="o", resource_id="link", transfer_kind="model", total_bytes=100),
        PhysicalEvent("c", "transfer_progress", 1, request_id="r", operation_id="t", object_id="o", resource_id="link", transfer_kind="model", bytes_delta=40),
        PhysicalEvent("d", "transfer_cancelled", 2, request_id="r", operation_id="t", object_id="o", resource_id="link", transfer_kind="model", bytes_delta=0),
        PhysicalEvent("e", "request_failed", 2, request_id="r"),
    ]
    failed = reduce_physical_events(cancelled).requests[0]
    assert failed.availability == "unavailable"
    assert failed.observed_transfer_bytes == 40
    assert failed.transfer_bytes is None
    with pytest.raises(ValueError, match="cancelled operations"):
        reduce_physical_events(cancelled[:-1] + [replace(cancelled[-1], event_type="request_completed")])
