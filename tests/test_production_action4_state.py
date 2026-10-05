from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.audit_native_typed_cache_probe import build_audit_catalog
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import RSUTypedCacheProfile
from src.envs.core.cache_eviction import TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.runtime.production_action4_state import export_action4_state, import_action4_state


def workflow_dict() -> dict:
    return {
        "workflow_id": "wf",
        "nodes": [{"node_id": "n0"}, {"node_id": "n1"}],
        "edges": [["n0", "n1"]],
        "execution_order": ["n0", "n1"],
    }


def test_target_model_must_be_ready_before_import() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "package"
        export_action4_state(
            package_dir=package,
            workflow=workflow_dict(),
            completed_nodes=["n0"],
            node_outputs={"n0": {"text": "prefix"}},
            identity={"base": "b", "adapter": "a", "next_node": "n1"},
            input_identity={"sha256": "input"},
            source_rsu_id="rsu_a",
            target_rsu_id="rsu_b",
        )
        blocked = import_action4_state(
            package_dir=package,
            expected_workflow=workflow_dict(),
            expected_identity={"base": "b", "adapter": "a", "next_node": "n1"},
            expected_input_identity={"sha256": "input"},
            target_model_ready=False,
            target_rsu_id="rsu_b",
        )
        assert blocked["status"] == "BLOCKED_MISSING_TARGET_MODEL"
        assert not blocked["execution_right_transferred"]
        committed = import_action4_state(
            package_dir=package,
            expected_workflow=workflow_dict(),
            expected_identity={"base": "b", "adapter": "a", "next_node": "n1"},
            expected_input_identity={"sha256": "input"},
            target_model_ready=True,
            target_rsu_id="rsu_b",
        )
        assert committed["status"] == "IMPORTED_COMMITTED"
        assert committed["completed_nodes"] == ["n0"]
        assert committed["next_node"] == "n1"


def _build_action4_env(package_root: Path) -> GymVecEnv:
    catalog = build_audit_catalog(sharing_enabled=True)
    adapter_id = "b0.a0"
    placement = catalog.resolve_typed_placement_plan(adapter_id=adapter_id, resident_object_ids=[])
    source_residents = list(placement.ordered_object_ids)
    catalog.rsu_typed_cache_profiles = [
        RSUTypedCacheProfile(rsu_id="rsu_a", resident_object_ids=source_residents),
        RSUTypedCacheProfile(rsu_id="rsu_b", resident_object_ids=[]),
    ]
    base_id = str(catalog.get_typed_adapter(adapter_id).required_base_model_id)
    nodes = [
        WorkflowNode("n0", "prefix", base_id, adapter_id, 1, 1, [], ["n1"]),
        WorkflowNode("n1", "suffix", base_id, adapter_id, 1, 1, ["n0"], []),
    ]
    workflow = WorkflowGraphState(
        workflow_id="production_action4_test",
        nodes=nodes,
        edges=[("n0", "n1")],
        execution_order=["n0", "n1"],
        current_node_id="n0",
    )
    frames = [
        {
            "time_index": index,
            "vehicles": [
                {
                    "vehicle_id": "veh",
                    "position_x": position,
                    "position_y": 0.0,
                    "speed": 40.0,
                    "base_model_id": base_id,
                    "active_workflow_id": workflow.workflow_id,
                }
            ],
        }
        for index, position in enumerate((0.0, 20.0, 80.0, 100.0))
    ]
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=workflow,
        adapter_catalog=catalog,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 55.0), RSUState("rsu_b", 100.0, 0.0, 55.0)],
        max_steps=3,
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": 320.0,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            "telemetry_enabled": True,
        },
        workflow_state_migration={
            "enabled": True,
            "package_root": str(package_root),
            "identity": {"base": base_id, "adapter": adapter_id, "next_node": "n1"},
            "input_identity": {"kind": "test", "sha256": "input"},
            "node_outputs": {"n0": {"kind": "test", "value": "prefix"}},
        },
    )
    return GymVecEnv(core)


def test_action4_normal_step_path_exports_and_imports_suffix_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = _build_action4_env(Path(tmp))
        _, reset = env.reset(seed=7)
        assert reset["action_mask"][4]
        _, _, terminated, _, first = env.step(4)
        assert not terminated
        assert first["production_action4_state_transfer"]["status"] == "EXPORTED_PENDING_IMPORT"
        assert first["semantic_state"]["workflow"]["completed_node_ids"] == ["n0"]
        _, _, terminated, _, second = env.step(0)
        assert terminated
        transfer = second["production_action4_state_transfer"]
        assert transfer["status"] == "IMPORTED_COMMITTED"
        assert transfer["execution_right_transferred"]
        assert second["cache_event"]["migration_realized"]
        assert second["semantic_state"]["workflow"]["completed_node_ids"] == ["n0", "n1"]


def test_action4_missing_target_model_does_not_advance_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = _build_action4_env(Path(tmp))
        env.reset(seed=7)
        _, _, _, _, first = env.step(4)
        assert first["semantic_state"]["workflow"]["completed_node_ids"] == ["n0"]
        env.core_env._typed_resident_object_ids["rsu_b"] = []
        _, _, terminated, _, second = env.step(3)
        assert not terminated
        transfer = second["production_action4_state_transfer"]
        assert transfer["status"] == "BLOCKED_MISSING_TARGET_MODEL"
        assert not transfer["execution_right_transferred"]
        assert not second["cache_event"]["migration_realized"]
        assert second["semantic_state"]["workflow"]["completed_node_ids"] == ["n0"]
