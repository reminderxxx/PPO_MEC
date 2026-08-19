from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.data.model_catalog.adapter_catalog import (
    AdapterCatalog,
    TYPED_MODEL_CACHE_PROFILE_ID,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import CacheEvent, ControlAction
from src.evaluators.cache_baseline_fairness import (
    build_typed_cache_fairness_binding,
    sha256_value,
    validate_typed_cache_fairness_binding,
)
from src.metrics.cache_efficiency_metrics import reduce_cache_efficiency_events
from src.oracles.cache_request_replay import build_request_replay
from src.oracles.future_horizon_cache_oracle import solve_future_horizon_cache_oracle


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
LEGACY = ROOT / "src/data/model_catalog/sample_model_catalog.json"
HF_DIAGNOSTIC = ROOT / "src/data/model_catalog/hf_metadata_diagnostic_model_profile.json"


def _catalog() -> AdapterCatalog:
    return AdapterCatalog.from_json(CONTROLLED)


def _env(catalog: AdapterCatalog | None = None, *, capacity_mb: float = 320.0, policy: str = "lru") -> VecWorkflowCoreEnv:
    return VecWorkflowCoreEnv(
        adapter_catalog=catalog or _catalog(),
        max_steps=3,
        cache_capacity_profile={
            "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
            "enabled": True,
            "unit": "mb",
            "capacity_mb": capacity_mb,
            "eviction_policy": policy,
            "eviction_policy_seed": 7,
        },
    )


def _cache_control(rsu_id: str | None = None, *, migration: str = "keep") -> ControlAction:
    return ControlAction(
        cache_action={"operation": "cache", "rsu_id": rsu_id, "strategy": "manual_cache"},
        offload_action={"mode": "rsu", "target_rsu_id": rsu_id},
        migration_action={"mode": migration},
        metadata={"action_id": 1, "action_name": "cache_current"},
    )


def _replace_object(catalog: AdapterCatalog, object_id: str, **changes) -> None:
    index = next(i for i, item in enumerate(catalog.typed_cache_objects) if item.object_id == object_id)
    payload = catalog.typed_cache_objects[index].to_dict()
    payload.update(changes)
    payload["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(payload)
    catalog.typed_cache_objects[index] = type(catalog.typed_cache_objects[index])(**payload)


def test_legacy_catalog_defaults_to_adapter_only_and_projects_explicitly() -> None:
    catalog = AdapterCatalog.from_json(LEGACY)
    assert catalog.model_cache_profile_id == "legacy_adapter_only_v1"
    projected = catalog.typed_objects_with_legacy_projection()
    assert projected and {item.object_type for item in projected} == {"adapter"}
    assert {item.required_base_model_id for item in projected} == {"veh_base_v1"}


def test_typed_catalog_round_trip_fingerprint_and_hf_nonformal_boundary() -> None:
    catalog = _catalog()
    assert all(
        len(adapter_ids) >= 2
        for adapter_ids in catalog.compatibility_map.values()
    )
    restored = AdapterCatalog.from_dict(json.loads(json.dumps(catalog.to_dict())))
    assert restored.canonical_fingerprint() == catalog.canonical_fingerprint()
    assert restored.validate_typed_catalog()["status"] == "pass"
    hf = AdapterCatalog.from_json(HF_DIAGNOSTIC)
    assert all(item.license_status == "unknown" for item in hf.typed_cache_objects)
    assert all("non_formal" in item.formal_use_status for item in hf.typed_cache_objects)
    assert next(item for item in hf.typed_cache_objects if "bert" in item.object_id).availability == "blocked_provenance_anomaly"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda raw: raw["typed_cache_objects"].append(deepcopy(raw["typed_cache_objects"][0])), "duplicate"),
        (lambda raw: raw["typed_cache_objects"][2].update(dependency_ids=["base:missing"]), "base object dependency"),
        (lambda raw: raw["typed_cache_objects"][0].update(dependency_ids=["adapter:perception"]), "cycle"),
        (lambda raw: raw["typed_cache_objects"][2].update(base_model_family="wrong"), "family mismatch"),
        (lambda raw: raw["typed_cache_objects"][2].update(resident_size_mb=0.0), "finite and positive"),
    ],
)
def test_typed_catalog_fail_fast_mutations(mutation, error: str) -> None:
    raw = json.loads(CONTROLLED.read_text())
    mutation(raw)
    for item in raw["typed_cache_objects"]:
        item["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(item)
    with pytest.raises(ValueError, match=error):
        AdapterCatalog.from_dict(raw)


def test_readiness_layers_cover_full_partial_incompatible_and_vehicle_boundary() -> None:
    env = _env()
    env.reset()
    node = env.workflow_state.current_node()
    assert node is not None
    full = env._typed_service_readiness(current_node=node, primary_vehicle=None, offload_mode="rsu", service_rsu_id="rsu_a", state_required=False, state_ready=True)
    assert full["base_ready"] and full["adapter_ready"] and full["full_service_ready"]
    env._typed_resident_object_ids["rsu_a"] = ["base:veh_base_v1"]
    base_only = env._typed_service_readiness(current_node=node, primary_vehicle=None, offload_mode="rsu", service_rsu_id="rsu_a", state_required=False, state_ready=True)
    assert base_only["base_ready"] and not base_only["adapter_ready"] and base_only["missing_object_types"] == ["adapter"]
    env._typed_resident_object_ids["rsu_a"] = ["adapter:perception"]
    adapter_only = env._typed_service_readiness(current_node=node, primary_vehicle=None, offload_mode="rsu", service_rsu_id="rsu_a", state_required=True, state_ready=False)
    assert adapter_only["adapter_ready"] and not adapter_only["base_ready"]
    assert adapter_only["missing_object_types"] == ["base_model", "workflow_state"]
    vehicle = env._extract_primary_vehicle_from_state(env._last_state)
    from src.envs.specs import VehicleState
    vehicle_state = VehicleState(**{key: vehicle[key] for key in ("vehicle_id", "position_x", "position_y", "speed", "base_model_id", "associated_rsu_id", "active_workflow_id")})
    local = env._typed_service_readiness(current_node=node, primary_vehicle=vehicle_state, offload_mode="vehicle", service_rsu_id=None, state_required=False, state_ready=True)
    assert local["base_ready"] and not local["adapter_ready"] and not local["full_service_ready"]


def test_atomic_base_adapter_admission_and_transfer_accounting() -> None:
    catalog = _catalog()
    next(item for item in catalog.rsu_typed_cache_profiles if item.rsu_id == "rsu_b").resident_object_ids = []
    env = _env(catalog)
    env.reset()
    result = env._apply_typed_cache_action(control=_cache_control("rsu_b"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_tracking")
    assert result["atomic_transaction_status"] == "committed"
    assert [item["object_type"] for item in result["admitted_typed_objects"]] == ["base_model", "adapter"]
    assert result["transfer_mb_by_type"] == {"adapter": 48.0, "base_model": 160.0}
    assert env._typed_resident_object_ids["rsu_b"] == ["base:veh_base_v1", "adapter:tracking"]
    assert result["orphan_count"] == 0


def test_base_present_only_admits_adapter_and_multi_victim_is_atomic() -> None:
    catalog = _catalog()
    next(item for item in catalog.rsu_typed_cache_profiles if item.rsu_id == "rsu_a").resident_object_ids = ["base:veh_base_v1", "adapter:perception", "adapter:tracking"]
    env = _env(catalog, capacity_mb=300.0)
    env.reset()
    result = env._apply_typed_cache_action(control=_cache_control("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_control")
    assert result["atomic_transaction_status"] == "committed"
    assert [item["object_id"] for item in result["admitted_typed_objects"]] == ["adapter:control"]
    assert result["evicted_object_ids"] == ["adapter:perception", "adapter:tracking"]
    assert env._typed_resident_object_ids["rsu_a"] == ["base:veh_base_v1", "adapter:control"]


@pytest.mark.parametrize("policy", ["lru", "fifo", "lfu", "aging_lfu", "random"])
def test_all_five_policies_share_typed_atomic_contract(policy: str) -> None:
    env = _env(capacity_mb=300.0, policy=policy)
    env.reset()
    result = env._apply_typed_cache_action(control=_cache_control("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_control")
    assert result["atomic_transaction_status"] == "committed"
    assert set(result["evicted_object_ids"]) == {"adapter:perception", "adapter:tracking"}
    assert result["orphan_count"] == 0


def test_oversized_or_dependency_unsafe_bundle_rolls_back_without_orphan() -> None:
    env = _env()
    env.reset()
    before = deepcopy(env._typed_resident_object_ids["rsu_a"])
    policy_before = env.export_cache_eviction_policy_state()
    result = env._apply_typed_cache_action(control=_cache_control("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_reasoning")
    assert result["atomic_transaction_status"] == "rolled_back_no_mutation"
    assert result["capacity_rejection_reason"] == "insufficient_dependency_safe_evictable_capacity"
    assert env._typed_resident_object_ids["rsu_a"] == before
    assert env.export_cache_eviction_policy_state() == policy_before
    assert "base:veh_base_v1" not in env._typed_evictable_residents("rsu_a")


def test_typed_slot_mode_fails_fast_and_reset_isolated() -> None:
    with pytest.raises(ValueError, match="requires enabled MB"):
        VecWorkflowCoreEnv(adapter_catalog=_catalog(), cache_capacity_profile={"model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID, "enabled": True, "unit": "adapter_slots", "rsu_adapter_slots": 4})
    env = _env()
    env.reset()
    env._typed_resident_object_ids["rsu_a"].remove("adapter:tracking")
    env.reset()
    assert "adapter:tracking" in env._typed_resident_object_ids["rsu_a"]


def test_typed_event_round_trip_metrics_and_state_transfer_are_independent() -> None:
    env = _env()
    env.reset()
    initial = env.export_cache_trace_snapshot()
    _, _, _, _, info = env.step(_cache_control("rsu_a", migration="migrate"))
    event = info["cache_event"]
    restored = CacheEvent.from_dict(json.loads(json.dumps(event)))
    assert restored.base_model_hit and restored.adapter_hit and restored.full_service_ready
    assert restored.state_migration_size_mb == 20.0
    assert restored.transfer_mb_by_type == {"workflow_state": 20.0}
    result = reduce_cache_efficiency_events(
        [event],
        trace_context={"context_schema_version": "1.0.0", "initial_snapshot": initial, "final_snapshot": env.export_cache_trace_snapshot(), "episode_end_step_index": 1},
    )
    assert result.type_aware_metrics["request_count"] == 1
    assert result.type_aware_metrics["joint_base_adapter_hit_rate"] == 1.0
    assert result.type_aware_metrics["latency_saved"]["value"] is None


def _typed_manifest_and_replay() -> tuple[dict, dict]:
    catalog = _catalog()
    binding = build_typed_cache_fairness_binding(catalog)
    binding["initial_typed_cache_contents"] = [{"rsu_id": "rsu_x", "resident_object_ids": []}]
    binding["initial_typed_state_fingerprint"] = sha256_value(binding["initial_typed_cache_contents"])
    manifest = {
        "identity": {"manifest_id": "typed-test", "git_commit": "test"},
        "hashes": {"full_manifest_sha256": "full", "semantic_protocol_sha256": "semantic"},
        "cache_contract": {"capacity": {"enabled": True, "unit": "mb", "capacity_mb": 260.0, "rsu_adapter_slots": None}, "typed_model_cache": binding},
    }
    unit = {"evaluation_unit_id": "u", "benchmark_run_seed": 7, "window_id": "w", "workflow_id": "wf", "workflow_dag_sha256": "dag", "expected_workload_fingerprint": "work", "raw_frame_interval": {"start": 0, "end": 2}, "raw_time_interval": {"start": 0, "end": 2}}
    plan = catalog.resolve_typed_placement_plan(adapter_id="adapter_tracking", resident_object_ids=[])
    rows = [
        {"object_id": object_id, "object_type": catalog.get_typed_object(object_id).object_type, "resident_size_mb": catalog.get_typed_object(object_id).resident_size_mb, "transfer_size_mb": catalog.get_typed_object(object_id).transfer_size_mb}
        for object_id in plan.ordered_object_ids
    ]
    common = {"evaluation_unit_id": "u", "episode_id": "ep", "vehicle_id": "v", "workflow_id": "wf", "required_base_model": "veh_base_v1", "object_id": "adapter:tracking", "adapter_id": "adapter_tracking", "object_size_mb": 56.0, "size_source": "typed_catalog", "request_rsu_id": "rsu_x", "current_service_rsu_id": "rsu_x", "previous_rsu_id": None, "actual_next_rsu_id": "rsu_x", "predicted_next_rsu_id": None, "actual_handoff_target_rsu_id": None, "predicted_handoff_target_rsu_id": None, "eligible_service_rsu_ids": ["rsu_x"], "eligible_cache_target_rsu_ids": ["rsu_x"], "dag_provenance": {"policy_neutral": True}, "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID, "typed_model_cache_contract_version": "1.0.0", "catalog_fingerprint": catalog.canonical_fingerprint(), "requested_typed_objects": rows, "dependency_bundle": {**plan.to_dict(), "ordered_object_ids": plan.ordered_object_ids}}
    requests = [{**common, "request_id": "r1", "node_id": "n1", "step_index": 1, "time_index": 1, "request_order": 0}, {**common, "request_id": "r2", "node_id": "n2", "step_index": 2, "time_index": 2, "request_order": 1}]
    replay = build_request_replay(requests=requests, evaluation_unit=unit, source_manifest=manifest)
    return manifest, replay


def test_typed_fairness_binding_and_tiny_exact_oracle() -> None:
    catalog = _catalog()
    binding = build_typed_cache_fairness_binding(catalog)
    assert validate_typed_cache_fairness_binding(binding, capacity={"enabled": True, "unit": "mb", "capacity_mb": 320.0})["status"] == "pass"
    manifest, replay = _typed_manifest_and_replay()
    result = solve_future_horizon_cache_oracle(replay=replay, manifest=manifest, horizon=1, state_limit=10_000)
    assert result["identity"]["model_cache_profile_id"] == TYPED_MODEL_CACHE_PROFILE_ID
    assert result["identity"]["optimality_status"] == "optimal"
    assert result["action_trace"][0]["admitted_object_ids"] == ["base:veh_base_v1", "adapter:tracking"]
    assert result["action_trace"][0]["orphan_count"] == 0
    assert result["performance"]["joint_model_hit_count"] == 2
    assert result["performance"]["transfer_mb_by_type"] == {"adapter": 48.0, "base_model": 160.0}
