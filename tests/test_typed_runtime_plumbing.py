from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    FairnessManifestError,
    build_manifest,
    build_pairwise_protocol_diff,
    enforce_benchmark_args,
    full_manifest_sha256,
    observed_request_stream_fingerprint,
    semantic_protocol_sha256,
    validate_manifest,
)
from src.evaluators.main_results_support import summary_to_row
from src.metrics.cache_efficiency_metrics import (
    cache_efficiency_row_fields,
    reduce_cache_efficiency_summary,
)
from src.runtime.typed_model_cache_runtime import (
    RuntimeContractError,
    build_checkpoint_provenance,
    load_runtime_catalog,
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
TYPED_CONFIG = ROOT / "configs/benchmark/typed_model_cache_controlled_lru.yaml"
TYPED_CONFIG_384 = ROOT / "configs/benchmark/typed_model_cache_controlled_lru_384mb.yaml"
LEGACY_SLOTS_CONFIG = ROOT / "configs/benchmark/legacy_adapter_slots_lru.yaml"
LEGACY_MB_CONFIG = ROOT / "configs/benchmark/legacy_adapter_mb_lru.yaml"
TYPED_CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
LEGACY_CATALOG = ROOT / "src/data/model_catalog/sample_model_catalog.json"
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
PLAN = ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"


@pytest.fixture(scope="module")
def typed_runtime() -> dict:
    return resolve_model_cache_runtime(TYPED_CONFIG, root=ROOT)


@pytest.fixture(scope="module")
def typed_manifest() -> dict:
    return build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=PLAN,
        catalog_path=TYPED_CATALOG,
        seeds=[7, 13],
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=2500,
        primary_vehicle_selection="stable_first",
        capacity_unit="mb",
        capacity_value=320,
        output_root="artifacts/analysis/g14a_test",
        evaluation_unit_limit=1,
        controller_agents=["ppo", "mappo"],
        created_at="2026-08-19T00:00:00Z",
    )


@pytest.fixture(scope="module")
def tiny_training(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output = tmp_path_factory.mktemp("g14a_typed_training")
    command = [
        sys.executable,
        str(ROOT / "scripts/train_algo_pool_real_sample.py"),
        "--agent_name",
        "ppo",
        "--profile",
        "smoke",
        "--episodes",
        "1",
        "--update_every",
        "1",
        "--batch_size",
        "1",
        "--max_steps",
        "1",
        "--max_workflows",
        "1",
        "--max_mobility_rows",
        "500",
        "--window_plan_path",
        str(PLAN),
        "--model_cache_runtime_config",
        str(TYPED_CONFIG),
        "--reward_positive_offset",
        "0",
        "--output_root",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/g14a_pytest_pycache"
    subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    run_dir = next((output / "ppo").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episode = json.loads(
        (run_dir / "episodes/episode_0001.summary.json").read_text(encoding="utf-8")
    )
    return {
        "run_dir": run_dir,
        "summary": summary,
        "episode": episode,
        "checkpoint": run_dir / "checkpoints/latest.pt",
    }


def _typed_config() -> dict:
    import yaml

    return yaml.safe_load(TYPED_CONFIG.read_text(encoding="utf-8"))


def _reseal(manifest: dict) -> dict:
    changed = deepcopy(manifest)
    semantic = semantic_protocol_sha256(changed)
    changed["identity"]["manifest_id"] = f"cbfm-{semantic[:16]}"
    changed["hashes"]["semantic_protocol_sha256"] = semantic
    changed["hashes"]["full_manifest_sha256"] = full_manifest_sha256(changed)
    return changed


def _fairness_args(manifest: dict, runtime: dict) -> SimpleNamespace:
    return SimpleNamespace(
        agents=list(BASELINE_NAMES) + ["ppo", "mappo"],
        seeds=[7, 13],
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=2500,
        primary_vehicle_selection="stable_first",
        window_plan_path=str(PLAN),
        classical_cache_slots=3,
        reward_positive_offset=0.0,
        _fairness_root=ROOT,
        _resolved_model_cache_runtime=runtime,
        window_mode="mixed_informative",
        predictor_kind="baseline",
        prediction_horizon=3,
        prediction_noise_std=0.0,
        prediction_confidence_scale=1.0,
        prediction_delay_steps=0,
        drop_handoff_prediction_prob=0.0,
        mobility_source="ngsim",
        mobility_csv_path="",
        workflow_csv_path=str(WORKFLOW),
    )


def _checkpoint(
    tmp_path: Path,
    runtime: dict,
    *,
    agent_name: str = "ppo",
    seed: int = 7,
    window_identity: dict | None = None,
) -> tuple[Path, dict]:
    identity = window_identity or {"path": "controlled", "sha256": "1" * 64}
    provenance = build_checkpoint_provenance(
        root=ROOT,
        agent_name=agent_name,
        training_seed=seed,
        runtime_contract=runtime,
        reward_positive_offset=0.0,
        train_window_plan_identity=identity,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"training_metadata": {"typed_runtime_provenance": provenance}},
        path,
    )
    return path, identity


def test_01_legacy_missing_fields_default_compatible() -> None:
    runtime = resolve_model_cache_runtime(None, root=ROOT)
    assert runtime["model_cache_profile"] == "legacy_adapter_only_v1"
    assert runtime["cache_capacity_profile"]["enabled"] is False


def test_02_typed_mb_config_resolves_all_required_fields(typed_runtime: dict) -> None:
    assert typed_runtime["model_cache_profile"] == "typed_base_adapter_state_v1"
    assert typed_runtime["cache_capacity_profile"]["capacity_mb"] == 320.0
    assert len(typed_runtime["runtime_contract_sha256"]) == 64


def test_03_typed_slot_is_rejected() -> None:
    config = _typed_config()
    config["cache_capacity_profile"].update(unit="adapter_slots", rsu_adapter_slots=3)
    with pytest.raises(RuntimeContractError, match="MB capacity"):
        resolve_model_cache_runtime(config, root=ROOT)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_04_invalid_typed_mb_is_rejected(value: float) -> None:
    config = _typed_config()
    config["cache_capacity_profile"]["capacity_mb"] = value
    with pytest.raises(RuntimeContractError, match="finite positive"):
        resolve_model_cache_runtime(config, root=ROOT)


def test_05_catalog_fingerprint_mismatch_is_rejected() -> None:
    config = _typed_config()
    config["typed_catalog_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeContractError, match="catalog fingerprint mismatch"):
        resolve_model_cache_runtime(config, root=ROOT)


def test_06_dependency_fingerprint_mismatch_is_rejected() -> None:
    config = _typed_config()
    config["typed_dependency_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeContractError, match="dependency fingerprint mismatch"):
        resolve_model_cache_runtime(config, root=ROOT)


def test_07_initial_state_fingerprint_mismatch_is_rejected() -> None:
    config = _typed_config()
    config["typed_initial_state_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeContractError, match="initial-state fingerprint mismatch"):
        resolve_model_cache_runtime(config, root=ROOT)


def test_08_pinned_metadata_fingerprint_mismatch_is_rejected() -> None:
    config = _typed_config()
    config["typed_pinned_evictability_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeContractError, match="pinned/evictability fingerprint mismatch"):
        resolve_model_cache_runtime(config, root=ROOT)


def test_09_typed_fairness_manifest_round_trip(typed_manifest: dict) -> None:
    restored = json.loads(json.dumps(typed_manifest, allow_nan=False))
    report = validate_manifest(restored, root=ROOT, check_files=True)
    assert report["status"] == "pass"
    assert restored["identity"]["cache_baseline_fairness_manifest_version"] == "1.1.0"


def test_10_five_baselines_remain_only_policy_difference(typed_manifest: dict) -> None:
    report = build_pairwise_protocol_diff(typed_manifest)
    assert report["comparison_count"] == 10
    assert report["status"] == "pass"


def test_11_training_runner_accepts_shared_typed_config(tiny_training: dict) -> None:
    assert tiny_training["summary"]["resolved_model_cache_runtime"][
        "model_cache_profile"
    ] == "typed_base_adapter_state_v1"


def test_12_episode_environment_actually_uses_typed_profile(tiny_training: dict) -> None:
    event = tiny_training["episode"]["cache_event_trace"][0]
    assert event["model_cache_profile_id"] == "typed_base_adapter_state_v1"
    assert event["cache_capacity_unit"] == "mb"


def test_13_training_summary_contains_resolved_provenance(tiny_training: dict) -> None:
    summary = tiny_training["summary"]
    assert summary["runtime_contract_sha256"] == summary["resolved_model_cache_runtime"][
        "runtime_contract_sha256"
    ]
    assert summary["train_window_plan_identity"]["split"] == "controlled_non_hidden"


def test_14_checkpoint_contains_compatible_typed_metadata(
    tiny_training: dict, typed_runtime: dict
) -> None:
    report = validate_checkpoint_provenance(
        tiny_training["checkpoint"],
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=typed_runtime,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity=tiny_training["summary"]["train_window_plan_identity"],
    )
    assert report["status"] == "compatible"
    assert len(report["checkpoint_sha256"]) == 64


def test_15_legacy_checkpoint_is_unavailable_for_typed_eval(tmp_path: Path, typed_runtime: dict) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"training_metadata": {"agent_name": "ppo"}}, path)
    report = validate_checkpoint_provenance(
        path,
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=typed_runtime,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity={},
    )
    assert report["status"] == "unavailable_legacy_metadata"


def test_16_typed_checkpoint_catalog_mismatch(tmp_path: Path, typed_runtime: dict) -> None:
    path, identity = _checkpoint(tmp_path, typed_runtime)
    expected = deepcopy(typed_runtime)
    expected["typed_catalog_fingerprint"] = "0" * 64
    report = validate_checkpoint_provenance(
        path,
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=expected,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity=identity,
    )
    assert report["status"] == "incompatible"


def test_17_typed_checkpoint_capacity_mismatch(tmp_path: Path, typed_runtime: dict) -> None:
    path, identity = _checkpoint(tmp_path, typed_runtime)
    different = resolve_model_cache_runtime(TYPED_CONFIG_384, root=ROOT)
    report = validate_checkpoint_provenance(
        path,
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=different,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity=identity,
    )
    assert report["status"] == "incompatible"


def test_18_checkpoint_agent_identity_mismatch(tmp_path: Path, typed_runtime: dict) -> None:
    path, identity = _checkpoint(tmp_path, typed_runtime)
    report = validate_checkpoint_provenance(
        path,
        expected_agent_name="mappo",
        expected_seed=7,
        expected_runtime_contract=typed_runtime,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity=identity,
    )
    assert report["status"] == "incompatible"
    assert any("agent_identity" in error for error in report["errors"])


def test_19_legacy_slot_runtime_compatibility() -> None:
    runtime = resolve_model_cache_runtime(LEGACY_SLOTS_CONFIG, root=ROOT)
    catalog = load_runtime_catalog(runtime, root=ROOT)
    env = VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        cache_capacity_profile=runtime["cache_capacity_profile"],
        max_steps=1,
    )
    env.reset()
    assert runtime["cache_capacity_profile"]["unit"] == "adapter_slots"


def test_20_legacy_mb_runtime_compatibility() -> None:
    runtime = resolve_model_cache_runtime(LEGACY_MB_CONFIG, root=ROOT)
    catalog = load_runtime_catalog(runtime, root=ROOT)
    env = VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        cache_capacity_profile=runtime["cache_capacity_profile"],
        max_steps=1,
    )
    env.reset()
    assert runtime["cache_capacity_profile"]["unit"] == "mb"


def test_21_benchmark_typed_mb_binding_passes(typed_manifest: dict, typed_runtime: dict) -> None:
    enforce_benchmark_args(_fairness_args(typed_manifest, typed_runtime), typed_manifest)


def test_22_cli_override_of_frozen_typed_capacity_fails(
    typed_manifest: dict, typed_runtime: dict
) -> None:
    args = _fairness_args(typed_manifest, deepcopy(typed_runtime))
    args._resolved_model_cache_runtime["cache_capacity_profile"]["capacity_mb"] = 384.0
    with pytest.raises(FairnessManifestError, match="MB capacity overrides"):
        enforce_benchmark_args(args, typed_manifest)


def test_23_cache_event_13_has_typed_request_bundle(tiny_training: dict) -> None:
    episode = tiny_training["episode"]
    assert episode["cache_event_schema_version"] == "1.3.0"
    assert len(episode["cache_event_trace"]) == len(
        [step for step in episode["step_trace"] if step.get("current_node_id")]
    )
    event = episode["cache_event_trace"][0]
    assert event["dependency_bundle"] is not None
    assert {row["object_type"] for row in event["requested_typed_objects"]} == {
        "base_model",
        "adapter",
    }
    assert event["base_model_hit"] is not None and event["adapter_hit"] is not None


def test_24_metrics_11_recompute_matches_benchmark_scalars(tiny_training: dict) -> None:
    episode = tiny_training["episode"]
    reduced = reduce_cache_efficiency_summary(episode)
    scalars = cache_efficiency_row_fields(episode)
    assert reduced.cache_efficiency_metrics_version == "1.1.0"
    assert scalars["cache_joint_model_hit_rate"] == reduced.type_aware_metrics[
        "joint_base_adapter_hit_rate"
    ]


def test_25_missing_typed_metrics_are_nullable_not_zero() -> None:
    fields = cache_efficiency_row_fields({})
    assert fields == {"cache_efficiency_availability": "unavailable"}


def test_26_request_fingerprint_is_deterministic(tiny_training: dict) -> None:
    episode = tiny_training["episode"]
    assert observed_request_stream_fingerprint(episode) == observed_request_stream_fingerprint(
        json.loads(json.dumps(episode))
    )


def test_27_summary_and_row_keep_lightweight_runtime_provenance(tiny_training: dict) -> None:
    row = summary_to_row(tiny_training["episode"])
    assert row["model_cache_profile"] == "typed_base_adapter_state_v1"
    assert len(row["runtime_contract_sha256"]) == 64
    assert "cache_event_trace" not in row


def test_28_runtime_hash_is_stable_and_capacity_sensitive() -> None:
    first = resolve_model_cache_runtime(TYPED_CONFIG, root=ROOT)
    second = resolve_model_cache_runtime(TYPED_CONFIG, root=ROOT)
    larger = resolve_model_cache_runtime(TYPED_CONFIG_384, root=ROOT)
    assert first["runtime_contract_sha256"] == second["runtime_contract_sha256"]
    assert first["runtime_contract_sha256"] != larger["runtime_contract_sha256"]


def test_29_runtime_json_round_trip(typed_runtime: dict) -> None:
    restored = json.loads(json.dumps(typed_runtime, allow_nan=False))
    assert restored == typed_runtime


def test_30_fairness_dependency_initial_and_pinned_drift_fail(
    typed_manifest: dict,
) -> None:
    for field in (
        "dependency_fingerprint",
        "initial_typed_state_fingerprint",
        "pinned_evictability_fingerprint",
    ):
        changed = deepcopy(typed_manifest)
        changed["cache_contract"]["typed_model_cache"][field] = "f" * 64
        report = validate_manifest(_reseal(changed), root=ROOT, check_files=False)
        assert report["status"] == "fail"


def test_checkpoint_window_and_external_sha_mismatch(tmp_path: Path, typed_runtime: dict) -> None:
    path, _ = _checkpoint(tmp_path, typed_runtime)
    report = validate_checkpoint_provenance(
        path,
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=typed_runtime,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity={"path": "different"},
        expected_checkpoint_sha256="0" * 64,
    )
    assert report["status"] == "incompatible"
    assert any("window" in error for error in report["errors"])
    assert any("SHA-256" in error for error in report["errors"])
