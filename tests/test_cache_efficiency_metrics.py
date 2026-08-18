from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.agents.registry import build_agent, get_algo_spec, validate_agent_eviction_binding
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import aggregate_rows
from src.metrics.cache_efficiency_metrics import reduce_cache_efficiency_events, reduce_cache_efficiency_summary
from src.metrics.recorder import EpisodeRecorder


def _event(event_id: str, step: int, **updates):
    payload = {
        "event_id": event_id, "event_schema_version": "1.2.0", "event_type": "request",
        "time_index": step, "episode_step_index": step, "vehicle_id": "v", "workflow_id": "w",
        "node_id": f"n{step}", "object_id": "obj:a", "adapter_id": "a", "object_type": "adapter",
        "size_mb": 1.0, "request_rsu_id": "r", "selected_target_rsu_id": "r", "served_rsu_id": "r",
        "predicted_next_rsu_id": None, "predicted_handoff_target_rsu_id": None,
        "hit_source": "current_rsu", "cache_lookup_performed": True, "cache_hit": True,
        "was_cached_before": True, "admission_requested": False, "admission_added": False,
        "admission_reason": "not_requested", "cache_target_rsu_id": None,
        "eviction_occurred": False, "eviction_policy": "lru", "evicted_object_id": None,
        "evicted_adapter_id": None, "eviction_reason": "not_occurred",
        "adapter_transfer_size_mb": 0.0, "state_migration_size_mb": 0.0,
        "transfer_source": "catalog_cache_object", "migration_requested": False, "migration_realized": False,
        "cache_capacity_enabled": True, "cache_capacity_unit": "mb", "cache_capacity_before": 10.0,
        "cache_used_before": 3.0, "cache_remaining_before": 7.0, "cache_capacity_after": 10.0,
        "cache_used_after": 3.0, "cache_remaining_after": 7.0, "action_id": 3,
        "action_name": "steady", "cache_strategy": "none", "offload_mode": "rsu",
        "service_success": True, "stall_occurred": False, "handoff_event_count": 0,
        "eviction_count": 0, "evicted_object_ids": [], "evicted_adapter_ids": [],
        "evicted_size_mb_sum": 0.0, "requested_object_size_mb": 1.0,
        "capacity_rejection_reason": None, "admitted_object_id": None, "admitted_adapter_id": None,
        "admitted_size_mb": None, "evicted_sizes_mb": [],
    }
    payload.update(updates)
    return payload


def _miss(event_id: str, step: int, object_id: str, size: float, before: float, after: float, **updates):
    return _event(
        event_id, step, object_id=object_id, adapter_id=object_id.split(":", 1)[1], size_mb=size,
        hit_source="cloud", cache_hit=False, cache_lookup_performed=False, served_rsu_id=None,
        selected_target_rsu_id=None, offload_mode="cloud", cache_used_before=before,
        cache_remaining_before=10.0 - before, cache_used_after=after, cache_remaining_after=10.0 - after,
        requested_object_size_mb=size, **updates,
    )


def _snapshot(step: int, residents: list[tuple[str, float]]):
    return {"snapshot_step_index": step, "rsus": [{
        "rsu_id": "r", "capacity_enabled": True, "capacity_unit": "mb", "capacity": 10.0,
        "residents": [{"object_id": f"obj:{name}", "adapter_id": name, "size_mb": size} for name, size in residents],
    }]}


def _trace_and_context():
    trace = [
        _event("e1", 1),
        _miss("e2", 2, "obj:c", 4.0, 3.0, 7.0, admission_requested=True, admission_added=True,
              admission_reason="fill", cache_target_rsu_id="r", admitted_object_id="obj:c",
              admitted_adapter_id="c", admitted_size_mb=4.0, adapter_transfer_size_mb=4.0),
        _miss("e3", 3, "obj:d", 5.0, 7.0, 9.0, admission_requested=True, admission_added=True,
              admission_reason="fill", cache_target_rsu_id="r", admitted_object_id="obj:d",
              admitted_adapter_id="d", admitted_size_mb=5.0, adapter_transfer_size_mb=5.0,
              eviction_occurred=True, eviction_count=2, evicted_object_id="obj:b", evicted_adapter_id="b",
              evicted_object_ids=["obj:b", "obj:a"], evicted_adapter_ids=["b", "a"],
              evicted_sizes_mb=[2.0, 1.0], evicted_size_mb_sum=3.0, eviction_reason="capacity_limit"),
        _miss("e4", 4, "obj:a", 1.0, 9.0, 9.0),
        _miss("e5", 5, "obj:e", 5.0, 9.0, 10.0, admission_requested=True, admission_added=True,
              admission_reason="fill", cache_target_rsu_id="r", admitted_object_id="obj:e",
              admitted_adapter_id="e", admitted_size_mb=5.0, adapter_transfer_size_mb=5.0,
              eviction_occurred=True, eviction_count=1, evicted_object_id="obj:c", evicted_adapter_id="c",
              evicted_object_ids=["obj:c"], evicted_adapter_ids=["c"], evicted_sizes_mb=[4.0],
              evicted_size_mb_sum=4.0, eviction_reason="capacity_limit"),
        _miss("e6", 6, "obj:c", 4.0, 10.0, 10.0),
    ]
    context = {"context_schema_version": "1.0.0", "initial_snapshot": _snapshot(0, [("a", 1.0), ("b", 2.0)]),
               "final_snapshot": _snapshot(6, [("d", 5.0), ("e", 5.0)]), "episode_end_step_index": 6}
    return trace, context


def test_object_byte_pollution_multi_victim_and_future_reuse() -> None:
    trace, context = _trace_and_context()
    result = reduce_cache_efficiency_events(trace, schema_version="1.2.0", trace_context=context).to_dict()
    assert result["request_metrics"]["object_hit_rate"] == pytest.approx(1 / 6, abs=1e-6)
    assert result["byte_metrics"]["byte_hit_rate"] == 0.05
    assert result["lifecycle_metrics"]["eviction_victim_count"] == 3
    assert result["pollution_metrics"]["unused_admitted_object_count"] == 1
    assert result["pollution_metrics"]["right_censored_unused_admitted_object_count"] == 2
    assert result["pollution_metrics"]["polluted_resident_mb_steps"] == 12.0
    assert result["pollution_metrics"]["total_resident_mb_steps"] == 51.0
    horizon = result["future_reuse_proxy_metrics"]["horizons"]["1"]
    assert horizon["evicted_then_requested_within_h_count"] == 2
    assert horizon["evicted_then_requested_within_h_mb"] == 5.0
    assert result["latency_saved_metrics"]["availability"] == "unavailable"


def test_vehicle_local_is_byte_hit_and_cloud_unserved_are_misses() -> None:
    vehicle = _event("v", 1, hit_source="vehicle_local", offload_mode="vehicle", served_rsu_id=None,
                     cache_lookup_performed=False, cache_capacity_enabled=False, cache_capacity_before=None,
                     cache_used_before=None, cache_remaining_before=None, cache_capacity_after=None,
                     cache_used_after=None, cache_remaining_after=None, size_mb=2.0)
    cloud = deepcopy(vehicle); cloud.update(event_id="c", episode_step_index=2, time_index=2, hit_source="cloud", cache_hit=False, offload_mode="cloud", size_mb=8.0)
    failed = deepcopy(vehicle); failed.update(event_id="u", episode_step_index=3, time_index=3, hit_source="unserved", cache_hit=False, offload_mode="rsu", service_success=False, stall_occurred=True, size_mb=10.0)
    result = reduce_cache_efficiency_events([vehicle, cloud, failed], schema_version="1.2.0")
    assert result.request_metrics["object_hit_count"] == 1
    assert result.byte_metrics["byte_hit_rate"] == 0.1
    assert result.capacity_metrics["availability"] == "not_applicable"


@pytest.mark.parametrize("bad_size", [float("nan"), float("inf"), -1.0])
def test_invalid_sizes_fail_fast(bad_size) -> None:
    with pytest.raises(ValueError, match="size_mb"):
        reduce_cache_efficiency_events([_event("bad", 1, size_mb=bad_size)], schema_version="1.2.0")


def test_missing_size_is_partial_not_zero_and_zero_denominators_are_null() -> None:
    result = reduce_cache_efficiency_events([_event("missing", 1, size_mb=None)], schema_version="1.2.0")
    assert result.byte_metrics["availability"] == "unavailable"
    assert result.byte_metrics["requested_size_mb_sum"] is None
    empty = reduce_cache_efficiency_events([], schema_version="1.1.0")
    assert empty.request_metrics["object_hit_rate"] is None
    assert empty.byte_metrics["byte_hit_rate"] is None


def test_old_trace_and_missing_context_are_compatible_but_pollution_unavailable() -> None:
    old = _event("old", 1, event_schema_version="1.1.0")
    for key in ("admitted_object_id", "admitted_adapter_id", "admitted_size_mb", "evicted_sizes_mb"):
        old.pop(key)
    result = reduce_cache_efficiency_events([old], schema_version="1.1.0")
    assert result.request_metrics["availability"] == "available"
    assert result.pollution_metrics["availability"] == "unavailable"


def test_summary_json_round_trip_and_nullable_aggregate() -> None:
    trace, context = _trace_and_context()
    summary = json.loads(json.dumps({"cache_event_schema_version": "1.2.0", "cache_event_trace": trace, "cache_trace_context": context}))
    result = reduce_cache_efficiency_summary(summary)
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    aggregate = aggregate_rows(
        [{"agent": "a", "metric": None}, {"agent": "a", "metric": 0.0}], ["agent"], ["metric"]
    )
    assert aggregate["a"]["metrics"]["metric"]["mean"] == 0.0
    assert aggregate["a"]["metrics"]["metric"]["available_count"] == 1


def test_five_classical_baselines_emit_the_same_efficiency_schema() -> None:
    schemas = []
    for name in ("reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"):
        spec = get_algo_spec(name)
        profile = validate_agent_eviction_binding(
            name, {"enabled": True, "unit": "adapter_slots", "rsu_adapter_slots": 1,
                   "eviction_policy": spec["required_eviction_policy"]}, run_seed=29,
        )
        recorder = EpisodeRecorder(); recorder.start_episode({"agent_name": name})
        env = GymVecEnv(VecWorkflowCoreEnv(max_steps=1, cache_capacity_profile=profile), recorder)
        observation, info = env.reset(seed=29)
        action, _ = build_agent(name, random_seed=29).act(observation, info)
        env.step(action)
        result = reduce_cache_efficiency_summary(recorder.build_summary()).to_dict()
        schemas.append(set(result))
        assert result["availability"] == "available"
    assert all(schema == schemas[0] for schema in schemas)
