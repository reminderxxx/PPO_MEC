from __future__ import annotations

import json

import pytest

from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import (
    CACHE_EVENT_SCHEMA_VERSION,
    CACHE_HIT_SOURCES,
    CacheEvent,
    ControlAction,
)
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder


def _request_event(env: VecWorkflowCoreEnv, control: ControlAction) -> dict:
    env.reset()
    return env.step(control)[4]["cache_event"]


def test_cache_event_required_fields_and_json_round_trip() -> None:
    event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(offload_action={"mode": "rsu"}),
    )

    assert set(CacheEvent.REQUIRED_FIELDS) == set(event)
    assert event["event_schema_version"] == CACHE_EVENT_SCHEMA_VERSION
    restored = CacheEvent.from_dict(json.loads(json.dumps(event)))
    assert restored.to_dict() == event
    assert event["hit_source"] in CACHE_HIT_SOURCES


def test_current_rsu_warm_hit_has_one_final_source() -> None:
    event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(offload_action={"mode": "rsu"}),
    )

    assert event["cache_lookup_performed"] is True
    assert event["cache_hit"] is True
    assert event["hit_source"] == "current_rsu"
    assert event["admission_requested"] is False


def test_reactive_miss_admission_is_one_request() -> None:
    event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(
            cache_action={
                "operation": "cache",
                "adapter_id": "adapter_lane",
                "strategy": "reactive_cache_fill",
            },
            offload_action={"mode": "rsu"},
        ),
    )

    assert event["event_type"] == "request"
    assert event["admission_requested"] is True
    assert event["admission_added"] is True
    assert event["event_id"] == "cache-event-000001"


def test_predictive_target_prefetch_records_target_and_transfer() -> None:
    event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(
            cache_action={
                "operation": "cache",
                "rsu_id": "rsu_b",
                "adapter_id": "adapter_perception",
                "strategy": "predictive_prefetch",
                "prediction_driven": True,
            },
            offload_action={"mode": "rsu", "target_rsu_id": "rsu_a"},
        ),
    )

    assert event["cache_target_rsu_id"] == "rsu_b"
    assert event["cache_strategy"] == "predictive_prefetch"
    assert event["admission_added"] is True
    assert event["adapter_transfer_size_mb"] > 0.0


def test_capacity_disabled_is_not_zero_occupancy() -> None:
    event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(offload_action={"mode": "rsu"}),
    )

    assert event["cache_capacity_enabled"] is False
    for field_name in (
        "cache_capacity_before",
        "cache_used_before",
        "cache_remaining_before",
        "cache_capacity_after",
        "cache_used_after",
        "cache_remaining_after",
    ):
        assert event[field_name] is None


def test_capacity_enabled_eviction_references_victim() -> None:
    env = VecWorkflowCoreEnv(
        cache_capacity_profile={
            "enabled": True,
            "unit": "adapter_slots",
            "rsu_adapter_slots": 1,
            "eviction_policy": "lru",
        }
    )
    event = _request_event(
        env,
        ControlAction(
            cache_action={
                "operation": "cache",
                "adapter_id": "adapter_lane",
                "strategy": "reactive_cache_fill",
            },
            offload_action={"mode": "rsu"},
        ),
    )

    assert event["eviction_occurred"] is True
    assert event["evicted_adapter_id"] == "adapter_perception"
    assert event["evicted_object_id"]
    assert event["cache_used_before"] == 1
    assert event["cache_used_after"] == 1


def test_vehicle_fallback_and_migration_prepare_are_auditable() -> None:
    vehicle_event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(offload_action={"mode": "vehicle"}),
    )
    migration_event = _request_event(
        VecWorkflowCoreEnv(),
        ControlAction(
            offload_action={"mode": "rsu"},
            migration_action={"mode": "prepare", "expected_target_rsu_id": "rsu_b"},
        ),
    )

    assert vehicle_event["hit_source"] == "vehicle_local"
    assert vehicle_event["served_rsu_id"] is None
    assert migration_event["migration_requested"] is True
    assert migration_event["state_migration_size_mb"] > 0.0


def test_empty_current_node_uses_not_applicable_semantics() -> None:
    workflow = ToyWorkflowGenerator().generate()
    workflow.current_node_id = None
    workflow.is_completed = True
    event = _request_event(
        VecWorkflowCoreEnv(workflow_state=workflow),
        ControlAction(),
    )

    assert event["event_type"] == "not_applicable"
    assert event["object_type"] == "not_applicable"
    assert event["hit_source"] == "not_applicable"


def test_recorder_exports_unique_events_without_changing_request_denominator() -> None:
    recorder = EpisodeRecorder()
    recorder.start_episode({"run_id": "cache-event-test"})
    env = GymVecEnv(core_env=VecWorkflowCoreEnv(max_steps=2), recorder=recorder)
    env.reset()
    env.step(1)
    env.step(0)
    summary = recorder.build_summary()

    events = summary["cache_event_trace"]
    assert summary["cache_event_schema_version"] == CACHE_EVENT_SCHEMA_VERSION
    assert len(events) == len(summary["step_trace"]) == 2
    assert len({event["event_id"] for event in events}) == 2
    assert sum(event["event_type"] == "request" for event in events) <= 2


def test_contract_rejects_invalid_invariants() -> None:
    payload = _request_event(VecWorkflowCoreEnv(), ControlAction(offload_action={"mode": "rsu"}))
    payload["cache_hit"] = False
    payload["hit_source"] = "current_rsu"
    with pytest.raises(ValueError, match="RSU hit_source"):
        CacheEvent.from_dict(payload)
