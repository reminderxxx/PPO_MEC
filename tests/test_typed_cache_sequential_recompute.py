from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.audit_native_typed_cache_probe import build_audit_catalog
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import (
    AdapterCatalog,
    TYPED_MODEL_CACHE_PROFILE_ID,
)
from src.envs.core.cache_eviction import (
    TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
    TYPED_EVICTION_SEMANTICS_STATIC,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction, RSUState, WorkflowGraphState, WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.cache_baseline_fairness import (
    build_typed_cache_fairness_binding,
    validate_typed_cache_fairness_binding,
)
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


def _replace_object(catalog: AdapterCatalog, object_id: str, **changes: object) -> None:
    index = next(
        index
        for index, item in enumerate(catalog.typed_cache_objects)
        if item.object_id == object_id
    )
    payload = catalog.typed_cache_objects[index].to_dict()
    payload.update(changes)
    payload["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(payload)
    catalog.typed_cache_objects[index] = type(catalog.typed_cache_objects[index])(**payload)


def _workflow() -> WorkflowGraphState:
    nodes = [
        WorkflowNode(
            node_id="n0",
            node_name="b0 request",
            required_base_model="b0",
            required_adapter="b0.a0",
            input_size=1,
            output_size=1,
            successors=["n1"],
        ),
        WorkflowNode(
            node_id="n1",
            node_name="b1 request",
            required_base_model="b1",
            required_adapter="b1.a0",
            input_size=1,
            output_size=1,
            predecessors=["n0"],
        ),
    ]
    return WorkflowGraphState(
        workflow_id="typed_cache_sequential_diagnostic",
        nodes=nodes,
        edges=[("n0", "n1")],
        execution_order=["n0", "n1"],
        current_node_id="n0",
    )


def _env(
    catalog: AdapterCatalog | None = None,
    *,
    semantics: str = TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
    capacity_mb: float = 136.0,
    workflow: WorkflowGraphState | None = None,
    max_steps: int = 4,
    policy: str = "lru",
) -> VecWorkflowCoreEnv:
    frames = [
        {
            "time_index": step,
            "vehicles": [
                {
                    "vehicle_id": "v0",
                    "position_x": 0.0,
                    "position_y": 0.0,
                    "speed": 0.0,
                    "base_model_id": "b0",
                    "active_workflow_id": "typed_cache_sequential_diagnostic",
                }
            ],
        }
        for step in range(max_steps + 2)
    ]
    return VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=workflow or _workflow(),
        adapter_catalog=catalog or build_audit_catalog(sharing_enabled=True),
        rsu_states=[
            RSUState(
                rsu_id="rsu_a",
                position_x=0.0,
                position_y=0.0,
                coverage_radius=1000.0,
            )
        ],
        max_steps=max_steps,
        cache_capacity_profile={
            "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
            "enabled": True,
            "unit": "mb",
            "capacity_mb": capacity_mb,
            "count_base_model_separately": True,
            "eviction_policy": policy,
            "typed_eviction_semantics": semantics,
        },
    )


def _control() -> ControlAction:
    return ControlAction(
        cache_action={"operation": "cache", "rsu_id": "rsu_a"},
        offload_action={"mode": "rsu", "target_rsu_id": "rsu_a"},
        migration_action={"mode": "keep"},
    )


def _request(env: VecWorkflowCoreEnv, adapter_id: str, step: int) -> dict:
    env._episode_steps = step
    return env._apply_typed_cache_action(
        control=_control(),
        primary_vehicle=None,
        current_node_id=f"request-{step}",
        required_adapter=adapter_id,
    )


def _assert_rejected_without_mutation(env: VecWorkflowCoreEnv, adapter_id: str) -> dict:
    residents_before = deepcopy(env._typed_resident_object_ids)
    policy_before = env.export_cache_eviction_policy_state()
    result = _request(env, adapter_id, 2)
    assert result["atomic_transaction_status"] == "rolled_back_no_mutation"
    assert env._typed_resident_object_ids == residents_before
    assert env.export_cache_eviction_policy_state() == policy_before
    return result


def test_default_static_semantics_remains_rejected_and_unchanged() -> None:
    env = _env(semantics=TYPED_EVICTION_SEMANTICS_STATIC)
    env.reset()
    assert _request(env, "b0.a0", 0)["atomic_transaction_status"] == "committed"
    result = _assert_rejected_without_mutation(env, "b1.a0")
    assert result["capacity_rejection_reason"] == (
        "insufficient_dependency_safe_evictable_capacity"
    )
    assert result["typed_cache_transaction_contract_version"].endswith("v1.0.0")


def test_two_request_sequential_shadow_plan_releases_adapter_then_base() -> None:
    env = _env()
    env.reset()
    first = _request(env, "b0.a0", 0)
    second = _request(env, "b1.a0", 1)
    assert first["transfer_mb_by_type"] == {"adapter": 8.0, "base_model": 96.0}
    assert second["atomic_transaction_status"] == "committed"
    assert second["evicted_object_ids"] == ["adapter:b0.a0", "base:b0"]
    assert second["evicted_size_mb_sum"] == 104.0
    assert second["transfer_mb_by_type"] == {"adapter": 8.0, "base_model": 128.0}
    assert env._typed_resident_object_ids["rsu_a"] == ["base:b1", "adapter:b1.a0"]
    trace = second["eviction_planning_trace"]
    assert [row["selected_victim_id"] for row in trace["rounds"]] == [
        "adapter:b0.a0",
        "base:b0",
    ]
    assert trace["required_free_mb"] == trace["actual_freed_mb"] == 104.0
    assert trace["real_state_mutated_during_planning"] is False


def test_multiple_adapters_are_removed_before_their_base_becomes_eligible() -> None:
    catalog = build_audit_catalog(sharing_enabled=True)
    profile = catalog.rsu_typed_cache_profiles[0]
    profile.resident_object_ids = ["base:b0", "adapter:b0.a0", "adapter:b0.a1"]
    env = _env(catalog)
    env.reset()
    result = _request(env, "b1.a0", 1)
    assert result["atomic_transaction_status"] == "committed"
    assert result["evicted_object_ids"] == [
        "adapter:b0.a0",
        "adapter:b0.a1",
        "base:b0",
    ]
    assert result["evicted_size_mb_sum"] == 112.0
    assert env._typed_resident_object_ids["rsu_a"] == ["base:b1", "adapter:b1.a0"]


@pytest.mark.parametrize(
    ("object_id", "evictability"),
    [
        ("adapter:b0.a0", "pinned"),
        ("base:b0", "pinned"),
        ("adapter:b0.a0", "non_evictable"),
    ],
)
def test_pinned_and_non_evictable_objects_block_replacement_without_mutation(
    object_id: str, evictability: str
) -> None:
    catalog = build_audit_catalog(sharing_enabled=True)
    _replace_object(catalog, object_id, evictability=evictability)
    env = _env(catalog)
    env.reset()
    assert _request(env, "b0.a0", 0)["atomic_transaction_status"] == "committed"
    result = _assert_rejected_without_mutation(env, "b1.a0")
    assert result["capacity_rejection_reason"] == (
        "insufficient_sequential_dependency_safe_evictable_capacity"
    )


def test_retained_dependent_adapter_keeps_base_protected() -> None:
    catalog = build_audit_catalog(sharing_enabled=True)
    _replace_object(catalog, "adapter:b0.a1", evictability="non_evictable")
    catalog.rsu_typed_cache_profiles[0].resident_object_ids = [
        "base:b0",
        "adapter:b0.a0",
        "adapter:b0.a1",
    ]
    env = _env(catalog)
    env.reset()
    result = _assert_rejected_without_mutation(env, "b1.a0")
    selected = [
        row["selected_victim_id"]
        for row in result["eviction_planning_trace"]["rounds"]
        if row["selected_victim_id"] is not None
    ]
    assert selected == ["adapter:b0.a0"]
    assert "base:b0" not in selected


def test_oversized_bundle_rejects_without_starting_shadow_planning() -> None:
    env = _env(capacity_mb=135.0)
    env.reset()
    result = _assert_rejected_without_mutation(env, "b1.a0")
    assert result["capacity_rejection_reason"] == "dependency_bundle_exceeds_total_capacity"
    assert result["eviction_planning_trace"] is None


def test_candidate_rejects_non_lru_policy_and_runtime_contract_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires eviction_policy=lru"):
        _env(policy="fifo")
    runtime = resolve_model_cache_runtime(
        "configs/benchmark/typed_model_cache_controlled_lru_sequential_recompute.yaml",
        root=".",
    )
    assert runtime["typed_cache_transaction_contract_version"].endswith("v1.1.0")
    assert runtime["transaction_contract"]["eviction_semantics"] == (
        TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU
    )
    assert runtime["cache_capacity_profile"]["typed_eviction_semantics"] == (
        TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU
    )
    binding = build_typed_cache_fairness_binding(
        AdapterCatalog.from_json(
            "src/data/model_catalog/typed_model_cache_controlled.json"
        ),
        runtime_contract=runtime,
    )
    assert binding["transaction_contract_version"].endswith("v1.1.0")
    assert validate_typed_cache_fairness_binding(binding)["status"] == "pass"


def test_gym_action_zero_reaches_candidate_and_completes_two_node_workflow() -> None:
    env = GymVecEnv(_env(max_steps=2))
    _, reset_info = env.reset(seed=7)
    assert reset_info["action_mask"][0] is True
    _, _, terminated_first, truncated_first, first_info = env.step(0)
    assert not terminated_first and not truncated_first
    assert first_info["control_action"]["cache_action"]["operation"] == "cache"
    assert first_info["cache_event"]["service_success"] is True
    assert first_info["semantic_state"]["workflow"]["completed_node_ids"] == ["n0"]
    _, _, terminated_second, truncated_second, second_info = env.step(0)
    assert terminated_second and not truncated_second
    assert second_info["cache_event"]["atomic_transaction_status"] == "committed"
    assert second_info["cache_event"]["typed_eviction_semantics"] == (
        TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU
    )
    assert second_info["cache_event"]["evicted_object_ids"] == [
        "adapter:b0.a0",
        "base:b0",
    ]
    assert second_info["cache_event"]["service_success"] is True
    assert second_info["semantic_state"]["workflow"]["is_completed"] is True
