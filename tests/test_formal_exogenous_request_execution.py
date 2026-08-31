from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction
from src.evaluators.main_results_support import build_episode_formal_request_exposure
from src.runtime.formal_exogenous_request_execution import (
    FormalRequestExposureError,
    align_cache_event_to_request,
    build_outcome_audit,
    compute_formal_endpoint_metrics,
    request_exposure_fingerprint,
    validate_formal_request_exposure_trace,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
PROFILE = {
    "model_cache_profile_id": "typed_base_adapter_state_v1",
    "enabled": True,
    "unit": "mb",
    "capacity_mb": 288.0,
    "eviction_policy": "lru",
}


def fixture() -> tuple[dict, object, AdapterCatalog, list, list]:
    catalog = AdapterCatalog.from_json(CATALOG)
    for row in catalog.rsu_typed_cache_profiles:
        row.resident_object_ids = []
    workflow = ToyWorkflowGenerator().generate(node_count=5)
    adapters = [
        "adapter_perception",
        "adapter_tracking",
        "adapter_fusion",
        "adapter_intent",
        "adapter_control",
    ]
    for node, adapter_id in zip(workflow.nodes, adapters):
        node.required_adapter = adapter_id
    raw = ReplayProvider()
    frames = [
        {
            "time_index": frame["time_index"],
            "vehicles": raw._frame_to_vehicle_states(frame),
        }
        for frame in raw._trajectory_frames
    ]
    probe = VecWorkflowCoreEnv(
        workflow_state=workflow,
        adapter_catalog=catalog,
        cache_capacity_profile=PROFILE,
    )
    rsus = deepcopy(probe._rsu_template)
    trace = build_episode_formal_request_exposure(
        workflow_state=workflow,
        mobility_bundle=SimpleNamespace(
            frames=frames,
            rsu_states=rsus,
            rsu_metadata={"window_id": "w"},
        ),
        adapter_catalog=catalog,
        max_steps=3,
        mobility_source="ngsim",
        primary_vehicle_selection="handoff_pressure",
        cache_capacity_profile=PROFILE,
        evaluation_unit={
            "evaluation_unit_id": "seed_7/w/wf_toy_1",
            "workflow_id": "wf_toy_1",
            "benchmark_run_seed": 7,
            "window_id": "w",
        },
        source_provenance={"phase": "synthetic", "formal": False},
    )
    return trace, workflow, catalog, frames, rsus


def run_actions(trace: dict, workflow: object, catalog: AdapterCatalog, frames: list, rsus: list, *, cache: bool) -> list[dict]:
    env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(deepcopy(frames)),
        workflow_state=deepcopy(workflow),
        adapter_catalog=deepcopy(catalog),
        rsu_states=deepcopy(rsus),
        max_steps=5,
        primary_vehicle_selection="handoff_pressure",
        cache_capacity_profile=PROFILE,
        formal_request_exposure_trace=trace,
    )
    env.reset()
    events = []
    for _ in trace["requests"]:
        control = ControlAction(
            cache_action=(
                {"operation": "cache", "strategy": "reactive_cache_fill"}
                if cache
                else {}
            ),
            offload_action={"mode": "rsu"},
        )
        _, _, _, _, info = env.step(control)
        events.append(info["cache_event"])
    return events


def test_two_policies_share_exposure_but_have_distinct_outcomes() -> None:
    trace, workflow, catalog, frames, rsus = fixture()
    cached = run_actions(trace, workflow, catalog, frames, rsus, cache=True)
    uncached = run_actions(trace, workflow, catalog, frames, rsus, cache=False)
    a = build_outcome_audit(cached, trace)
    b = build_outcome_audit(uncached, trace)
    assert a["alignment_status"] == b["alignment_status"] == "pass"
    assert a["request_count"] == b["request_count"] == 3
    assert a["outcome_fingerprint"] != b["outcome_fingerprint"]
    assert [row["formal_request_id"] for row in cached] == [
        row["formal_request_id"] for row in uncached
    ]
    assert any(row["service_success"] for row in cached)
    assert not any(row["service_success"] for row in uncached)


def test_service_failure_does_not_change_next_exposure_and_delay_is_unavailable() -> None:
    trace, workflow, catalog, frames, rsus = fixture()
    events = run_actions(trace, workflow, catalog, frames, rsus, cache=False)
    assert [row["node_id"] for row in events] == [row["node_id"] for row in trace["requests"]]
    metrics = compute_formal_endpoint_metrics(events, trace, truncated=False)
    assert metrics["external_request_denominator"] == 3
    assert metrics["workflow_continuity_rate"] == 0.0
    assert metrics["end_to_end_workflow_delay"] is None
    assert metrics["end_to_end_workflow_delay_availability"] == (
        "unavailable_failed_or_incomplete_workflow"
    )


def test_partial_readiness_multi_victim_oversized_and_cross_rsu_are_outcomes_only() -> None:
    trace, workflow, catalog, frames, rsus = fixture()
    events = run_actions(trace, workflow, catalog, frames, rsus, cache=True)
    mutated = deepcopy(events)
    mutated[0].update(
        base_model_hit=True,
        adapter_hit=False,
        joint_model_hit=False,
        full_service_ready=False,
        missing_object_types=["adapter"],
    )
    mutated[1].update(
        eviction_occurred=True,
        eviction_count=2,
        evicted_object_id="adapter:a",
        evicted_object_ids=["adapter:a", "adapter:b"],
        evicted_adapter_id="a",
        evicted_adapter_ids=["a", "b"],
        capacity_rejection_reason=None,
    )
    mutated[2].update(
        admission_added=False,
        capacity_rejection_reason="oversized_dependency_bundle",
        selected_target_rsu_id="rsu_b",
    )
    audit = build_outcome_audit(mutated, trace)
    assert audit["alignment_status"] == "pass"
    assert audit["request_exposure_fingerprint"] == trace["request_exposure_fingerprint"]
    assert audit["outcome_fingerprint"] != build_outcome_audit(events, trace)["outcome_fingerprint"]


def test_not_applicable_alignment_and_right_censoring_semantics() -> None:
    trace, *_ = fixture()
    request = deepcopy(trace["requests"][0])
    request.update(
        request_kind="not_applicable",
        node_id=None,
        adapter_id=None,
        object_id=None,
        object_size_mb=None,
        requested_typed_objects=[],
        dependency_bundle=None,
    )
    event = {
        "episode_step_index": 1,
        "time_index": request["time_index"],
        "vehicle_id": request["vehicle_id"],
        "workflow_id": request["workflow_id"],
        "node_id": None,
        "object_id": None,
        "adapter_id": None,
        "size_mb": None,
        "request_rsu_id": request["request_rsu_id"],
        "requested_typed_objects": [],
        "dependency_bundle": None,
        "event_type": "not_applicable",
    }
    aligned = align_cache_event_to_request(
        event, request, trace_fingerprint=trace["request_exposure_fingerprint"]
    )
    assert aligned["request_alignment_status"] == "matched_exactly_once"
    assert trace["exposure_censoring"]["right_censored"] is True


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "out_of_order"])
def test_missing_duplicate_extra_and_out_of_order_fail_fast(mode: str) -> None:
    trace, workflow, catalog, frames, rsus = fixture()
    events = run_actions(trace, workflow, catalog, frames, rsus, cache=True)
    if mode == "missing":
        events.pop()
    elif mode == "duplicate":
        events[1] = deepcopy(events[0])
    elif mode == "extra":
        events.append(deepcopy(events[-1]))
    else:
        events[0], events[1] = events[1], events[0]
    with pytest.raises(FormalRequestExposureError):
        build_outcome_audit(events, trace)


def test_future_leak_outcome_pollution_and_identity_drift_fail_fast() -> None:
    trace, *_ = fixture()
    leaked = deepcopy(trace)
    leaked["requests"][0]["future_topology"] = {"rsu": "oracle"}
    leaked["request_exposure_fingerprint"] = request_exposure_fingerprint(leaked)
    with pytest.raises(FormalRequestExposureError, match="future field"):
        validate_formal_request_exposure_trace(leaked)
    polluted = deepcopy(trace)
    polluted["requests"][0]["reward"] = 1.0
    polluted["request_exposure_fingerprint"] = request_exposure_fingerprint(polluted)
    with pytest.raises(FormalRequestExposureError, match="outcome field"):
        validate_formal_request_exposure_trace(polluted)


def test_json_round_trip_and_deterministic_reproduction() -> None:
    first, *_ = fixture()
    second, *_ = fixture()
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert validate_formal_request_exposure_trace(first)["status"] == "pass"
