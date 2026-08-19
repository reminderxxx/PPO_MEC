from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    FairnessManifestError,
    build_manifest,
    build_pairwise_protocol_diff,
    canonical_json_bytes,
    enforce_benchmark_args,
    full_manifest_sha256,
    observed_request_stream_fingerprint,
    semantic_protocol_sha256,
    stamp_summary_provenance,
    validate_manifest,
    validate_observed_fingerprint_matrix,
)
from src.evaluators.main_results_support import summary_to_row


ROOT = Path(__file__).resolve().parents[1]
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
PLAN = ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"
CATALOG = ROOT / "src/data/model_catalog/sample_model_catalog.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=PLAN,
        catalog_path=CATALOG,
        seeds=[7],
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=2500,
        primary_vehicle_selection="stable_first",
        capacity_unit="adapter_slots",
        capacity_value=3,
        output_root="artifacts/analysis/g07_test",
        evaluation_unit_limit=1,
        created_at="2026-08-18T00:00:00Z",
    )


def reseal(payload: dict) -> dict:
    payload = deepcopy(payload)
    semantic = semantic_protocol_sha256(payload)
    payload["identity"]["manifest_id"] = f"cbfm-{semantic[:16]}"
    payload["hashes"]["semantic_protocol_sha256"] = semantic
    payload["hashes"]["full_manifest_sha256"] = full_manifest_sha256(payload)
    return payload


def report(payload: dict, *, files: bool = False) -> dict:
    return validate_manifest(payload, root=ROOT, check_files=files)


def test_valid_five_baseline_manifest(manifest: dict) -> None:
    assert report(manifest, files=True)["status"] == "pass"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown"])
def test_baseline_membership_failures(manifest: dict, mode: str) -> None:
    changed = deepcopy(manifest)
    if mode == "missing":
        changed["baseline_matrix"].pop()
    elif mode == "duplicate":
        changed["baseline_matrix"][-1] = deepcopy(changed["baseline_matrix"][0])
    else:
        changed["baseline_matrix"][-1]["agent_identity"]["name"] = "reactive_unknown"
    changed = reseal(changed)
    assert report(changed)["status"] == "fail"


def test_agent_policy_mismatch(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["baseline_matrix"][0]["eviction_policy"]["name"] = "fifo"
    assert any("agent-policy mismatch" in item for item in report(reseal(changed))["errors"])


@pytest.mark.parametrize("field,value", [("enabled", False), ("unit", "bytes"), ("rsu_adapter_slots", 0)])
def test_capacity_enabled_unit_value_fail(manifest: dict, field: str, value: object) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["capacity"][field] = value
    assert report(reseal(changed))["status"] == "fail"


def test_slot_mb_mixing_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["capacity"]["capacity_mb"] = 256.0
    assert report(reseal(changed))["status"] == "fail"


def test_admission_control_drift_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["baseline_matrix"][1]["admission_control_identity"] = "other"
    assert any("admission/control drift" in item for item in report(reseal(changed))["errors"])


def test_catalog_hash_drift_fails_file_validation(manifest: dict) -> None:
    changed = deepcopy(manifest)
    catalog = next(item for item in changed["dataset_provenance"]["inputs"] if item["logical_dataset_id"] == "ppo_mec_sample_adapter_catalog")
    catalog["sha256"] = "0" * 64
    errors = report(reseal(changed), files=True)["errors"]
    assert any("dataset hash mismatch" in item for item in errors)


def test_size_fallback_drift_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["catalog_fallback_rule"] = ""
    assert report(reseal(changed))["status"] == "fail"


def test_initial_cache_drift_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["initial_per_rsu_cache_contents"][0]["cached_adapter_ids"].append("x")
    assert any("initial cache snapshot hash mismatch" in item for item in report(reseal(changed))["errors"])


@pytest.mark.parametrize("logical_id", ["ngsim_vehicle_trajectories", "alibaba_cluster_trace_2018_batch_task"])
def test_dataset_hash_drift_fails(manifest: dict, logical_id: str) -> None:
    changed = deepcopy(manifest)
    target = next(item for item in changed["dataset_provenance"]["inputs"] if item["logical_dataset_id"] == logical_id)
    target["sha256"] = "f" * 64
    assert any("dataset hash mismatch" in item for item in report(reseal(changed), files=True)["errors"])


@pytest.mark.parametrize("interval", ["raw_frame_interval", "raw_time_interval"])
def test_raw_interval_drift_fails_against_plan(manifest: dict, interval: str) -> None:
    changed = deepcopy(manifest)
    changed["window_workload_plan"]["evaluation_units"][0][interval]["end"] += 1
    errors = report(reseal(changed), files=True)["errors"]
    assert any("interval drift" in item for item in errors)


def test_request_workload_fingerprint_drift_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["window_workload_plan"]["evaluation_units"][0]["expected_workload_fingerprint"] = "1" * 64
    errors = report(reseal(changed), files=True)["errors"]
    assert any("fingerprint drift" in item for item in errors)


@pytest.mark.parametrize("field", ["environment_seed", "workload_selection_seed"])
def test_environment_and_workload_seed_drift_fails(manifest: dict, field: str) -> None:
    changed = deepcopy(manifest)
    changed["seed_plan"]["per_run"][0][field] = 99
    assert any(field in item for item in report(reseal(changed))["errors"])


def test_random_policy_seed_error_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["seed_plan"]["per_run"][0]["reactive_random_private_rng_seed"] = 99
    assert report(reseal(changed))["status"] == "fail"


def test_global_rng_contract_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["seed_plan"]["random_rng_contract"] = "global random module"
    assert report(reseal(changed))["status"] == "fail"


def test_metrics_contract_and_horizon_drift(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["metrics_aggregation"]["future_reuse_horizons_steps"] = [1, 3]
    assert report(reseal(changed))["status"] == "fail"


def test_cacheevent_incompatible_major_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["cache_event_schema_version"] = "2.0.0"
    assert report(reseal(changed))["status"] == "fail"


def test_nullable_aggregation_missing_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["metrics_aggregation"]["nullable_aggregation_contract"] = ""
    assert report(reseal(changed))["status"] == "fail"


def test_canonical_serialization_stable() -> None:
    assert canonical_json_bytes({"b": 1, "a": "中"}) == canonical_json_bytes({"a": "中", "b": 1})


def test_semantic_field_changes_hash(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["capacity"]["rsu_adapter_slots"] = 4
    assert semantic_protocol_sha256(changed) != semantic_protocol_sha256(manifest)


def test_created_at_and_output_root_do_not_change_semantic_hash(manifest: dict) -> None:
    changed = deepcopy(manifest)
    changed["identity"]["created_at"] = "2099-01-01T00:00:00Z"
    changed["artifact_plan"]["output_root"] = "elsewhere"
    assert semantic_protocol_sha256(changed) == semantic_protocol_sha256(manifest)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_fail_fast(manifest: dict, bad: float) -> None:
    changed = deepcopy(manifest)
    changed["cache_contract"]["capacity"]["rsu_adapter_slots"] = bad
    assert report(changed)["status"] == "fail"


def test_missing_required_field_fails(manifest: dict) -> None:
    changed = deepcopy(manifest)
    del changed["metrics_aggregation"]
    assert report(changed)["status"] == "fail"


def _args(manifest: dict) -> SimpleNamespace:
    return SimpleNamespace(
        agents=list(BASELINE_NAMES), seeds=[7], max_workflows=1, workflow_selector="ordered",
        min_tasks=5, max_tasks=20, max_steps=1, max_mobility_rows=2500,
        primary_vehicle_selection="stable_first", window_plan_path=str(PLAN),
        classical_cache_slots=3, reward_positive_offset=0.0, _fairness_root=ROOT,
        window_mode="mixed_informative", predictor_kind="baseline", prediction_horizon=3,
        prediction_noise_std=0.0, prediction_confidence_scale=1.0, prediction_delay_steps=0,
        drop_handoff_prediction_prob=0.0, mobility_source="ngsim", mobility_csv_path="",
        workflow_csv_path=str(WORKFLOW),
    )


def _minimal_summary() -> dict:
    return {
        "cache_event_trace": [], "cache_event_schema_version": "1.2.0", "run_info": {},
        "system_metrics": {
            "end_to_end_workflow_delay": 0.0, "workflow_continuity_rate": 0.0,
            "handoff_failure_rate": 0.0, "handoff_ready_ratio": 0.0,
            "adapter_warm_hit_ratio": 0.0, "cross_rsu_cold_start_frequency": 0.0,
            "backhaul_traffic_cost": 0.0, "adapter_state_migration_overhead": 0.0,
            "predictive_prefetch_precision": 0.0,
        },
        "handoff_summary": {"migration_during_handoff_count": 0, "handoff_ready_count": 0, "handoff_total_count": 0, "migration_prepare_count": 0},
        "prefetch_summary": {"true_predictive_prefetch_count": 0},
        "prefetch_validation_summary": {"validated_predictive_prefetch_count": 0, "prefetch_validated_hit_count": 0, "prefetch_expired_miss_count": 0, "predictive_prefetch_precision": 0.0},
        "agent_action_diagnostics": {}, "step_trace": [],
        "reward_breakdown": {"total": {"sum": 0.0}}, "episode_success": False,
    }


def test_cli_override_frozen_field_fails(manifest: dict) -> None:
    args = _args(manifest)
    args.max_steps = 2
    with pytest.raises(FairnessManifestError, match="overrides frozen manifest"):
        enforce_benchmark_args(args, manifest)


def test_observed_fingerprint_mismatch_fails(manifest: dict) -> None:
    unit = manifest["window_workload_plan"]["evaluation_units"][0]["evaluation_unit_id"]
    matrix = {unit: {name: "same" for name in BASELINE_NAMES}}
    matrix[unit]["reactive_random"] = "different"
    with pytest.raises(FairnessManifestError, match="observed request stream fingerprint mismatch"):
        validate_observed_fingerprint_matrix(matrix)


def test_summary_row_and_aggregate_provenance_fields(manifest: dict) -> None:
    unit = manifest["window_workload_plan"]["evaluation_units"][0]
    summary = _minimal_summary()
    summary["run_info"] = {"window_id": unit["window_id"], "workflow_id": unit["workflow_id"], "agent_name": "reactive_lru", "seed": 7}
    stamp_summary_provenance(summary, manifest, unit)
    row = summary_to_row(summary)
    assert row["fairness_manifest_id"] == manifest["identity"]["manifest_id"]
    assert row["fairness_manifest_hash"] == manifest["hashes"]["full_manifest_sha256"]
    assert row["observed_request_stream_fingerprint"] == observed_request_stream_fingerprint(summary)


def test_pairwise_ten_pairs_only_allowed(manifest: dict) -> None:
    diff = build_pairwise_protocol_diff(manifest)
    assert diff["comparison_count"] == 10
    assert diff["status"] == "pass"
    assert all(not item["unexpected_differences"] for item in diff["comparisons"])


def test_json_round_trip(manifest: dict) -> None:
    restored = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    assert restored == manifest
    assert report(restored)["status"] == "pass"


def test_legacy_benchmark_row_marks_manifest_unavailable() -> None:
    summary = _minimal_summary()
    summary["run_info"] = {"agent_name": "reactive_lru"}
    assert summary_to_row(summary)["fairness_manifest_status"] == "unavailable"
