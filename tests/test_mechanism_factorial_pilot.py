from __future__ import annotations

import sys
from pathlib import Path

import yaml

from scripts.run_mechanism_algorithm_training import _build_manifest, _sha256_bytes
from scripts.run_mechanism_factorial_pilot import build_factorial_catalog
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


ROOT_DIR = Path(__file__).resolve().parents[1]


def _source_catalog() -> AdapterCatalog:
    return AdapterCatalog.from_json(
        ROOT_DIR / "src/data/model_catalog/typed_model_cache_controlled.json"
    )


def _capacity_profile() -> dict[str, object]:
    return {
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "enabled": True,
        "unit": "mb",
        "capacity_mb": 360.0,
        "count_base_model_separately": True,
        "eviction_policy": "lru",
    }


def test_base_sharing_off_uses_equal_size_adapter_specific_replicas() -> None:
    shared = build_factorial_catalog(
        _source_catalog(),
        base_sharing_enabled=True,
        initial_adapter_id="adapter_perception",
    )
    replicated = build_factorial_catalog(
        _source_catalog(),
        base_sharing_enabled=False,
        initial_adapter_id="adapter_perception",
    )

    shared_perception = shared.get_typed_adapter("adapter_perception")
    shared_tracking = shared.get_typed_adapter("adapter_tracking")
    replicated_perception = replicated.get_typed_adapter("adapter_perception")
    replicated_tracking = replicated.get_typed_adapter("adapter_tracking")
    assert shared_perception.required_base_model_id == shared_tracking.required_base_model_id
    assert replicated_perception.required_base_model_id != replicated_tracking.required_base_model_id

    for rsu_id in ("rsu_a", "rsu_b", "rsu_c"):
        shared_used = sum(
            shared.get_typed_object(object_id).resident_size_mb
            for object_id in shared.get_initial_typed_residents(rsu_id)
        )
        replicated_used = sum(
            replicated.get_typed_object(object_id).resident_size_mb
            for object_id in replicated.get_initial_typed_residents(rsu_id)
        )
        assert shared_used == replicated_used == 220.0


def test_migration_off_preserves_cache_and_offload_but_uses_cold_restart() -> None:
    env = VecWorkflowCoreEnv(
        adapter_catalog=_source_catalog(),
        cache_capacity_profile=_capacity_profile(),
        mechanism_profile={
            "mechanism_factorial_profile_version": "controlled_mechanism_factorial_v1",
            "profile_id": "sharing_on_migration_off",
            "base_sharing_enabled": True,
            "workflow_state_migration_enabled": False,
            "migration_disabled_fallback": "cold_restart_one_request",
        },
    )
    requested = ControlAction(
        cache_action={"operation": "cache", "rsu_id": "rsu_b"},
        offload_action={"mode": "rsu"},
        migration_action={"mode": "prepare", "expected_target_rsu_id": "rsu_b"},
        metadata={"action_id": 4},
    )

    effective = env._apply_mechanism_profile(requested)
    assert effective.cache_action == requested.cache_action
    assert effective.offload_action == requested.offload_action
    assert effective.migration_action == {
        "mode": "keep",
        "strategy": "cold_restart_one_request",
    }
    assert effective.metadata["requested_migration_mode"] == "prepare"
    assert effective.metadata["effective_migration_mode"] == "keep"
    assert effective.metadata["migration_suppressed"] is True
    assert requested.migration_action["mode"] == "prepare"


def test_no_sharing_arm_can_atomically_replace_exclusive_adapter_base_pair() -> None:
    catalog = build_factorial_catalog(
        _source_catalog(),
        base_sharing_enabled=False,
        initial_adapter_id="adapter_perception",
    )
    env = VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        cache_capacity_profile=_capacity_profile(),
        mechanism_profile={
            "profile_id": "sharing_off_migration_on",
            "base_sharing_enabled": False,
            "workflow_state_migration_enabled": True,
        },
    )
    env.reset()
    rsu = env._get_rsu_map()["rsu_a"]
    plan = env._plan_factorial_replica_eviction(
        rsu=rsu,
        required_free_capacity=96.0,
    )
    perception = catalog.get_typed_adapter("adapter_perception")
    assert plan.sufficient is True
    assert plan.ordered_victim_ids == [
        perception.object_id,
        perception.dependency_ids[0],
    ]
    assert plan.cumulative_freed_capacity == 220.0


def test_legacy_environment_semantics_are_unchanged_without_profile() -> None:
    env = VecWorkflowCoreEnv()
    requested = ControlAction(migration_action={"mode": "prepare"})
    assert env._apply_mechanism_profile(requested) is requested


def test_frozen_shared_training_runtime_has_equal_initial_rsu_state() -> None:
    runtime = resolve_model_cache_runtime(
        ROOT_DIR
        / "configs/experiment/mechanism_factorial_shared_360mb_runtime_v1.yaml",
        root=ROOT_DIR,
    )
    assert runtime["cache_capacity_profile"]["capacity_mb"] == 360.0
    assert runtime["typed_catalog_fingerprint"] == (
        "59beab40753255fe5f5a2a627a4ea07f2392618590a5239573bf290f1a51f90e"
    )
    assert {
        row["resident_mb"] for row in runtime["initial_per_rsu_typed_state"]
    } == {220.0}


def test_background_manifest_freezes_matched_algorithm_budget(tmp_path: Path) -> None:
    mobility = tmp_path / "mobility.csv"
    workflow = tmp_path / "workflow.csv"
    mobility.write_bytes(b"mobility\n")
    workflow.write_bytes(b"workflow\n")
    config = {
        "study_version": "manifest_test",
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_config": "configs/experiment/mechanism_factorial_shared_360mb_runtime_v1.yaml",
        "window_plan": "configs/experiment/mechanism_algorithm_training_windows_v1.json",
        "agents": ["sa_ghmappo", "mappo"],
        "budget": {
            "seed": 1401,
            "episodes_per_agent": 4,
            "max_steps": 6,
            "update_every": 2,
            "batch_size": 8,
            "checkpoint_every_updates": 1,
            "total_training_episodes": 8,
            "total_environment_step_cap": 48,
        },
        "data": {
            "mobility_csv_path": mobility.name,
            "workflow_csv_path": workflow.name,
            "max_mobility_rows": 10,
            "max_workflows": 2,
            "min_tasks": 5,
            "max_tasks": 20,
        },
        "expected_sources": {
            "mobility": {
                "size_bytes": mobility.stat().st_size,
                "sha256": _sha256_bytes(mobility.read_bytes()),
            },
            "workflow": {
                "size_bytes": workflow.stat().st_size,
                "sha256": _sha256_bytes(workflow.read_bytes()),
            },
        },
        "claim_boundary": "unit-test controlled supplement",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = _build_manifest(config_path, tmp_path, tmp_path / "output")

    assert manifest["agents"] == ["sa_ghmappo", "mappo"]
    assert manifest["algorithm_superiority_claim_allowed"] is False
    assert len(manifest["commands"]) == 2
    commands = [row["command"] for row in manifest["commands"]]
    for command in commands:
        assert command[command.index("--episodes") + 1] == "4"
        assert command[command.index("--random_seed") + 1] == "1401"
        assert command[command.index("--mechanism_factorial_arm") + 1] == (
            "sharing_on_migration_on"
        )
        assert "--formal-exogenous-request-execution" in command
