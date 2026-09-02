from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction, RSUState, VehicleState
from src.evaluators.main_results_support import build_episode_formal_request_exposure
from src.runtime.formal_exogenous_request_execution import (
    FormalRequestExposureError,
    request_exposure_fingerprint,
    validate_formal_request_exposure_trace,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
PROFILE = {
    "model_cache_profile_id": "typed_base_adapter_state_v1",
    "enabled": True,
    "unit": "mb",
    "capacity_mb": 288.0,
    "eviction_policy": "lru",
    "eviction_policy_seed": 7,
}


def vehicle(vehicle_id: str, x: float, *, speed: float = 40.0) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        position_x=x,
        position_y=0.0,
        speed=speed,
        base_model_id="base_perception_v1",
    )


def frames(rows: list[list[VehicleState]]) -> list[dict]:
    return [
        {
            "time_index": 1000 + index,
            "source_segment_run_id": "synthetic_run_1",
            "vehicles": row,
        }
        for index, row in enumerate(rows)
    ]


def fixture_parts(*, coverage: float = 60.0):
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
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    rsus = [
        RSUState("rsu_a", 0.0, 0.0, coverage),
        RSUState("rsu_b", 100.0, 0.0, coverage),
    ]
    return workflow, catalog, rsus


def build_trace(
    mobility_frames: list[dict],
    *,
    coverage: float = 60.0,
    source_provenance: dict | None = None,
) -> tuple[dict, object, AdapterCatalog, list[RSUState]]:
    workflow, catalog, rsus = fixture_parts(coverage=coverage)
    trace = build_episode_formal_request_exposure(
        workflow_state=workflow,
        mobility_bundle=SimpleNamespace(
            frames=mobility_frames,
            rsu_states=rsus,
            rsu_metadata={"window_id": "subject_window"},
        ),
        adapter_catalog=catalog,
        max_steps=5,
        mobility_source="ngsim",
        primary_vehicle_selection="handoff_pressure",
        cache_capacity_profile=PROFILE,
        evaluation_unit={
            "evaluation_unit_id": "seed_7/subject_window/wf_toy_1",
            "workflow_id": "wf_toy_1",
            "benchmark_run_seed": 7,
            "window_id": "subject_window",
        },
        source_provenance=source_provenance
        or {"phase": "test", "formal": False, "performance_evidence": False},
    )
    return trace, workflow, catalog, rsus


def run_trace(
    trace: dict,
    workflow: object,
    catalog: AdapterCatalog,
    rsus: list[RSUState],
    mobility_frames: list[dict],
) -> tuple[list[dict], list[dict]]:
    env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(deepcopy(mobility_frames)),
        workflow_state=deepcopy(workflow),
        adapter_catalog=deepcopy(catalog),
        rsu_states=deepcopy(rsus),
        max_steps=7,
        primary_vehicle_selection="handoff_pressure",
        cache_capacity_profile=PROFILE,
        formal_request_exposure_trace=trace,
    )
    state, _ = env.reset()
    states = [state]
    events = []
    for _ in trace["requests"]:
        next_state, _, _, _, info = env.step(ControlAction())
        states.append(next_state)
        events.append(info["cache_event"])
    return states, events


def persistent_frames() -> list[dict]:
    return frames(
        [
            [vehicle("a", x), vehicle("b", 10.0)]
            for x in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
        ]
    )


def test_disappearing_reset_candidate_is_filtered_and_runtime_never_reselects() -> None:
    mobility = persistent_frames()
    mobility[3]["vehicles"].append(vehicle("z", 0.0))
    mobility[0]["vehicles"].append(vehicle("c", 0.0))
    mobility[1]["vehicles"].append(vehicle("c", 20.0))
    trace, workflow, catalog, rsus = build_trace(mobility)
    selected = trace["subject_lifecycle"]["selected_primary_vehicle_id"]
    assert selected in {"a", "b"}
    assert selected != "c"
    states, events = run_trace(trace, workflow, catalog, rsus, mobility)
    assert {state["primary_vehicle_id"] for state in states} == {selected}
    assert {event["vehicle_id"] for event in events} == {selected}


def test_pressure_ranking_and_tie_break_are_deterministic_for_persistent_candidates() -> None:
    pressure_trace, *_ = build_trace(persistent_frames())
    assert pressure_trace["subject_lifecycle"]["selected_primary_vehicle_id"] == "a"
    tie_mobility = frames(
        [[vehicle("a", 10.0), vehicle("b", 20.0)] for _ in range(6)]
    )
    first, *_ = build_trace(tie_mobility)
    second, *_ = build_trace(deepcopy(tie_mobility))
    assert first == second
    assert first["subject_lifecycle"]["selected_primary_vehicle_id"] == "a"
    assert first["subject_lifecycle"]["eligible_candidate_count"] == 2


def test_no_horizon_persistent_vehicle_fails_before_execution() -> None:
    mobility = frames(
        [
            [vehicle("a", 0.0)],
            [vehicle("a", 10.0)],
            [vehicle("b", 20.0)],
            [vehicle("b", 30.0)],
            [vehicle("c", 40.0)],
            [vehicle("c", 50.0)],
        ]
    )
    with pytest.raises(
        FormalRequestExposureError,
        match="BLOCKED_BY_FORMAL_REQUEST_SUBJECT_ELIGIBILITY",
    ):
        build_trace(mobility)


def test_reused_vehicle_id_with_nonphysical_step_is_rejected() -> None:
    mobility = frames(
        [[vehicle("a", x, speed=1.0)] for x in (0.0, 1.0, 2.0, 500.0, 501.0, 502.0)]
    )
    with pytest.raises(FormalRequestExposureError, match="BLOCKED_BY"):
        build_trace(mobility)


def test_null_association_is_preserved_without_subject_substitution() -> None:
    mobility = frames(
        [[vehicle("a", x)] for x in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)]
    )
    trace, workflow, catalog, rsus = build_trace(mobility, coverage=15.0)
    assert any(
        request["request_rsu_id"] is None
        or request["current_service_rsu_id"] is None
        for request in trace["requests"]
    )
    states, events = run_trace(trace, workflow, catalog, rsus, mobility)
    assert {state["primary_vehicle_id"] for state in states} == {"a"}
    assert {event["vehicle_id"] for event in events} == {"a"}


def test_request_and_current_rsu_use_previous_and_current_frame_of_same_vehicle() -> None:
    mobility = persistent_frames()
    trace, *_ = build_trace(mobility)
    selected = trace["subject_lifecycle"]["selected_primary_vehicle_id"]
    assert selected == "a"
    assert [row["request_rsu_id"] for row in trace["requests"]] == [
        "rsu_a",
        "rsu_a",
        "rsu_a",
        "rsu_b",
        "rsu_b",
    ]
    assert [row["current_service_rsu_id"] for row in trace["requests"]] == [
        "rsu_a",
        "rsu_a",
        "rsu_b",
        "rsu_b",
        "rsu_b",
    ]


def test_trace_runtime_and_cache_event_vehicle_identity_align_exactly() -> None:
    mobility = persistent_frames()
    trace, workflow, catalog, rsus = build_trace(mobility)
    states, events = run_trace(trace, workflow, catalog, rsus, mobility)
    selected = trace["subject_lifecycle"]["selected_primary_vehicle_id"]
    assert [request["vehicle_id"] for request in trace["requests"]] == [selected] * 5
    assert [event["vehicle_id"] for event in events] == [selected] * 5
    assert [event["current_service_rsu_id"] for event in events] == [
        request["current_service_rsu_id"] for request in trace["requests"]
    ]
    assert [state["primary_vehicle_id"] for state in states] == [selected] * 6


def test_exposure_fingerprint_is_agent_capacity_phase_and_repeat_invariant() -> None:
    mobility = persistent_frames()
    fingerprints = set()
    for agent in ("sa_ghmappo", "ppo", "reactive_lru"):
        for capacity in (288, 576, 864):
            trace, *_ = build_trace(
                deepcopy(mobility),
                source_provenance={
                    "phase": "train" if agent == "sa_ghmappo" else "formal",
                    "agent": agent,
                    "capacity_mb": capacity,
                    "runtime_contract_sha256": f"runtime-{capacity}",
                },
            )
            fingerprints.add(trace["request_exposure_fingerprint"])
    assert len(fingerprints) == 1


def test_selection_evidence_is_not_projected_into_actor_or_controller_state() -> None:
    mobility = persistent_frames()
    trace, workflow, catalog, rsus = build_trace(mobility)
    states, _ = run_trace(trace, workflow, catalog, rsus, mobility)
    for state in states:
        projection = state["formal_request_exposure"]
        assert "subject_lifecycle" not in projection
        assert "eligible_candidate_count" not in projection
        assert "eligible_candidate_canonical_fingerprint" not in projection
        assert "request_exposure_fingerprint" not in projection
        assert projection["selection_evidence_actor_visible"] is False
        assert projection["selection_evidence_controller_visible"] is False


def test_legacy_nonformal_path_retains_dynamic_vehicle_reselection() -> None:
    workflow, catalog, rsus = fixture_parts()
    mobility = frames(
        [
            [vehicle("a", 0.0), vehicle("b", 10.0)],
            [vehicle("b", 20.0)],
            [vehicle("b", 30.0)],
        ]
    )
    env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(mobility),
        workflow_state=workflow,
        adapter_catalog=catalog,
        rsu_states=rsus,
        max_steps=2,
        primary_vehicle_selection="stable_first",
        cache_capacity_profile=PROFILE,
    )
    reset_state, _ = env.reset()
    next_state, *_ = env.step(ControlAction())
    assert reset_state["primary_vehicle_id"] == "a"
    assert next_state["primary_vehicle_id"] == "b"


@pytest.mark.parametrize("mutation", ["missing", "extra", "version", "tampered"])
def test_lifecycle_schema_missing_extra_version_and_tamper_are_rejected(mutation: str) -> None:
    trace, *_ = build_trace(persistent_frames())
    broken = deepcopy(trace)
    lifecycle = broken["subject_lifecycle"]
    if mutation == "missing":
        lifecycle.pop("eligible_candidate_count")
    elif mutation == "extra":
        lifecycle["unexpected"] = True
    elif mutation == "version":
        lifecycle["contract_version"] = "9.0.0"
    else:
        lifecycle["selected_primary_vehicle_id"] = "not-selected"
    broken["request_exposure_fingerprint"] = request_exposure_fingerprint(broken)
    with pytest.raises(FormalRequestExposureError):
        validate_formal_request_exposure_trace(broken)


@pytest.mark.parametrize(
    "reference",
    [
        "typed_model_cache_formal_20260901_155201_g14c_v11",
        "/private/tmp/ppo_mec_g14c_v11_e19108a_20260901_155201/checkpoints/latest.pt",
    ],
)
def test_historical_v11_run_checkpoint_and_root_references_are_rejected(reference: str) -> None:
    trace, *_ = build_trace(persistent_frames())
    broken = deepcopy(trace)
    broken["source_provenance"]["historical_source"] = reference
    with pytest.raises(FormalRequestExposureError, match="historical invalid G14C v11"):
        validate_formal_request_exposure_trace(broken)


def test_canonical_finite_round_trip_and_sha256_reproduction() -> None:
    first, *_ = build_trace(persistent_frames())
    second, *_ = build_trace(deepcopy(persistent_frames()))
    assert first["request_exposure_fingerprint"] == second["request_exposure_fingerprint"]
    assert validate_formal_request_exposure_trace(first)["status"] == "pass"
    assert request_exposure_fingerprint(first) == first["request_exposure_fingerprint"]
