"""Paper-grade fairness manifest contract for classical cache baselines."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.agents.registry import get_algo_spec
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import build_selected_workflow_states


MANIFEST_VERSION = "1.0.0"
PRODUCER_VERSION = "g07_v1"
SEED_DERIVATION_VERSION = "benchmark_seed_identity_v1"
BASELINE_NAMES = (
    "reactive_lru",
    "reactive_fifo",
    "reactive_lfu",
    "reactive_aging_lfu",
    "reactive_random",
)
POLICY_BY_BASELINE = {
    "reactive_lru": "lru",
    "reactive_fifo": "fifo",
    "reactive_lfu": "lfu",
    "reactive_aging_lfu": "aging_lfu",
    "reactive_random": "random",
}
NON_SEMANTIC_TOP_LEVEL = {"hashes", "validation"}
NON_SEMANTIC_IDENTITY_FIELDS = {"manifest_id", "created_at"}
ALLOWED_TOP_LEVEL_FIELDS = {
    "identity",
    "dataset_provenance",
    "window_workload_plan",
    "seed_plan",
    "cache_contract",
    "baseline_matrix",
    "metrics_aggregation",
    "artifact_plan",
    "claim_boundary",
    "hashes",
    "validation",
}
PAIRWISE_ALLOWED_PREFIXES = (
    "agent_identity.name",
    "agent_identity.class",
    "eviction_policy.name",
    "eviction_policy.deterministic",
    "eviction_policy.requires_seed",
    "eviction_policy.policy_seed_rule",
    "eviction_policy.policy_config",
    "config.path",
    "config.normalized_absolute_path",
    "config.sha256",
    "semantic_config.algorithm",
    "semantic_config.eviction_policy",
    "semantic_config.eviction_policy_version",
    "semantic_config.requires_seed",
    "semantic_config.seed_derivation",
    "semantic_config.aging_interval",
    "semantic_config.aging_factor",
)


class FairnessManifestError(ValueError):
    """Raised when a fairness invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FairnessManifestError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _reject_non_finite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(manifest)
    for field in NON_SEMANTIC_TOP_LEVEL:
        projected.pop(field, None)
    identity = projected.get("identity", {})
    for field in NON_SEMANTIC_IDENTITY_FIELDS:
        identity.pop(field, None)
    # Artifact locations and run labels affect storage, never experiment semantics.
    projected.pop("artifact_plan", None)
    for dataset in projected.get("dataset_provenance", {}).get("inputs", []):
        dataset.pop("normalized_absolute_path", None)
    for baseline in projected.get("baseline_matrix", []):
        baseline.get("config", {}).pop("normalized_absolute_path", None)
    return projected


def semantic_protocol_sha256(manifest: dict[str, Any]) -> str:
    return sha256_value(semantic_projection(manifest))


def full_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = deepcopy(manifest)
    payload.pop("hashes", None)
    payload.pop("validation", None)
    return sha256_value(payload)


def _path_identity(path: str | Path, root: Path, logical_id: str, provider: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"dataset/config file does not exist: {resolved}")
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
        path_kind = "repository_relative"
    except ValueError:
        relative = resolved.as_posix()
        path_kind = "external_normalized"
    return {
        "logical_dataset_id": logical_id,
        "path_kind": path_kind,
        "logical_path": relative,
        "normalized_absolute_path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "provider_parser_identity": provider,
    }


def _git_identity(root: Path) -> tuple[str, dict[str, Any]]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).splitlines()
    return commit, {
        "is_dirty": bool(status),
        "changed_path_count": len(status),
        "changed_paths": sorted(line[3:] for line in status if len(line) >= 4),
        "note": "audit-only; existing changes are not protocol inputs",
    }


def _workflow_payload(workflow: Any) -> dict[str, Any]:
    nodes = [node.to_dict() for node in workflow.nodes]
    return {
        "workflow_id": workflow.workflow_id,
        "nodes": nodes,
        "edges": [list(edge) for edge in workflow.edges],
        "execution_order": list(workflow.execution_order),
    }


def workload_fingerprint_payload(workflow: Any) -> dict[str, Any]:
    node_map = {node.node_id: node for node in workflow.nodes}
    requests = []
    for node_id in workflow.execution_order:
        node = node_map[node_id]
        requests.append(
            {
                "workflow_id": workflow.workflow_id,
                "node_id": node.node_id,
                "required_base_model": node.required_base_model,
                "required_adapter": node.required_adapter,
                "input_size": node.input_size,
                "output_size": node.output_size,
            }
        )
    return {"fingerprint_contract": "static_dag_request_plan_v1", "requests": requests}


def workload_fingerprint(workflow: Any) -> str:
    return sha256_value(workload_fingerprint_payload(workflow))


def observed_request_stream_fingerprint(summary: dict[str, Any]) -> str:
    requests = []
    for event in summary.get("cache_event_trace", []):
        if event.get("event_type") != "request":
            continue
        requests.append(
            {
                "workflow_id": event.get("workflow_id"),
                "node_id": event.get("node_id"),
                "object_id": event.get("object_id"),
                "adapter_id": event.get("adapter_id"),
                "object_type": event.get("object_type"),
                "size_mb": event.get("size_mb"),
            }
        )
    return sha256_value({"fingerprint_contract": "observed_cache_request_stream_v1", "requests": requests})


def _initial_cache_payload(catalog: AdapterCatalog) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "rsu_id": profile.rsu_id,
                "cached_adapter_ids": list(profile.cached_adapter_ids),
            }
            for profile in catalog.rsu_adapter_caches
        ),
        key=lambda item: item["rsu_id"],
    )


def _catalog_resident_sizes(catalog: AdapterCatalog, workflows: Iterable[Any]) -> list[dict[str, Any]]:
    adapter_ids = sorted(
        {
            node.required_adapter
            for workflow in workflows
            for node in workflow.nodes
        }
        | {
            adapter_id
            for profile in catalog.rsu_adapter_caches
            for adapter_id in profile.cached_adapter_ids
        }
    )
    rows = []
    for adapter_id in adapter_ids:
        resolved = catalog.resolve_adapter_resident_size_mb(adapter_id)
        rows.append(
            {
                "adapter_id": adapter_id,
                "object_id": resolved.object_id,
                "resident_size_mb": resolved.size_mb,
                "source": resolved.source,
            }
        )
    return rows


def _load_baseline_entry(root: Path, name: str) -> dict[str, Any]:
    path = root / "configs" / "algo" / f"{name}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FairnessManifestError(f"baseline config must be a mapping: {path}")
    spec = get_algo_spec(name)
    policy_name = POLICY_BY_BASELINE[name]
    deterministic = policy_name != "random"
    return {
        "agent_identity": {
            "name": name,
            "class": "".join(part.title() for part in name.split("_")) + "Agent",
            "family": spec["family"],
            "version": "1.0.0",
        },
        "eviction_policy": {
            "name": policy_name,
            "version": str(spec["eviction_policy_version"]),
            "deterministic": deterministic,
            "requires_seed": policy_name == "random",
            "policy_seed_rule": (
                "policy_seed_equals_benchmark_run_seed"
                if policy_name == "random"
                else "not_consumed"
            ),
            "policy_config": (
                {"aging_interval": 8, "aging_factor": 0.5}
                if policy_name == "aging_lfu"
                else {}
            ),
        },
        "admission_control_identity": "reactive_current_rsu_admission_v1",
        "action_control_semantics": "semantic_discrete_5/reactive_current_rsu_v1",
        "config": {
            "path": path.relative_to(root).as_posix(),
            "normalized_absolute_path": path.resolve().as_posix(),
            "sha256": sha256_file(path),
        },
        "semantic_config": payload,
        "allowed_difference_fields": list(PAIRWISE_ALLOWED_PREFIXES),
        "forbidden_difference_fields": [
            "capacity",
            "catalog",
            "workload",
            "reward",
            "admission_control_identity",
            "action_control_semantics",
        ],
        "agent_specific_overrides": {},
    }


def _recursive_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                rows.append({"path": path, "left": "<missing>", "right": right[key]})
            elif key not in right:
                rows.append({"path": path, "left": left[key], "right": "<missing>"})
            else:
                rows.extend(_recursive_diff(left[key], right[key], path))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [{"path": prefix, "left": left, "right": right}]
    if left != right:
        return [{"path": prefix, "left": left, "right": right}]
    return []


def build_pairwise_protocol_diff(manifest: dict[str, Any]) -> dict[str, Any]:
    entries = {item["agent_identity"]["name"]: item for item in manifest["baseline_matrix"]}
    comparisons = []
    for left_name, right_name in itertools.combinations(BASELINE_NAMES, 2):
        differences = _recursive_diff(entries[left_name], entries[right_name])
        allowed = [
            item
            for item in differences
            if any(item["path"] == prefix or item["path"].startswith(prefix + ".") for prefix in PAIRWISE_ALLOWED_PREFIXES)
        ]
        unexpected = [item for item in differences if item not in allowed]
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "allowed_differences": allowed,
                "unexpected_differences": unexpected,
                "status": "pass" if not unexpected else "fail",
            }
        )
    return {
        "comparison_count": len(comparisons),
        "status": "pass" if all(item["status"] == "pass" for item in comparisons) else "fail",
        "comparisons": comparisons,
    }


def build_manifest(
    *,
    root: str | Path,
    mobility_path: str | Path,
    workflow_path: str | Path,
    window_plan_path: str | Path,
    catalog_path: str | Path,
    seeds: list[int],
    max_workflows: int,
    workflow_selector: str,
    min_tasks: int,
    max_tasks: int,
    max_steps: int,
    max_mobility_rows: int,
    primary_vehicle_selection: str,
    capacity_unit: str,
    capacity_value: float,
    output_root: str,
    evaluation_unit_limit: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise FairnessManifestError("seeds must be non-empty and unique")
    if capacity_unit not in {"adapter_slots", "mb"}:
        raise FairnessManifestError("capacity_unit must be adapter_slots or mb")
    if capacity_value <= 0 or not math.isfinite(float(capacity_value)):
        raise FairnessManifestError("capacity_value must be finite and positive")
    plan_path = Path(window_plan_path).resolve()
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    windows = plan_payload.get("selected_window_plan") if isinstance(plan_payload, dict) else plan_payload
    if not isinstance(windows, list) or not windows:
        raise FairnessManifestError("window plan must contain selected_window_plan")
    if evaluation_unit_limit is not None:
        windows = windows[: int(evaluation_unit_limit)]
    required_window_fields = {
        "window_id", "frame_offset", "window_length", "time_index_start", "time_index_end",
        "window_class", "recommended_rsu_layout",
    }
    for index, window in enumerate(windows):
        missing = required_window_fields - set(window)
        if missing:
            raise FairnessManifestError(f"window {index} missing raw identity fields: {sorted(missing)}")
    catalog = AdapterCatalog.from_json(catalog_path)
    workflows_by_seed: dict[int, list[Any]] = {}
    for seed in seeds:
        workflows_by_seed[seed] = build_selected_workflow_states(
            workflow_csv_path=workflow_path,
            max_workflows=max_workflows,
            workflow_selector=workflow_selector,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
            random_seed=seed,
        )
    all_workflows = [workflow for workflows in workflows_by_seed.values() for workflow in workflows]
    initial_cache = _initial_cache_payload(catalog)
    if capacity_unit == "adapter_slots" and any(
        len(item["cached_adapter_ids"]) > int(capacity_value) for item in initial_cache
    ):
        raise FairnessManifestError(
            "G07 requires capacity to contain the identical declared initial cache without policy-specific trimming"
        )
    if capacity_unit == "mb":
        for item in initial_cache:
            used_mb = sum(
                catalog.resolve_adapter_resident_size_mb(adapter_id).size_mb
                for adapter_id in item["cached_adapter_ids"]
            )
            if used_mb > float(capacity_value) + 1e-9:
                raise FairnessManifestError(
                    "G07 requires MB capacity to contain the identical declared initial cache without policy-specific trimming"
                )
    units = []
    for seed in seeds:
        for window in windows:
            for workflow in workflows_by_seed[seed]:
                workflow_payload = _workflow_payload(workflow)
                unit_id = f"seed_{seed}/{window['window_id']}/{workflow.workflow_id}"
                units.append(
                    {
                        "evaluation_unit_id": unit_id,
                        "benchmark_run_seed": seed,
                        "window_id": window["window_id"],
                        "raw_frame_interval": {
                            "start": int(window["frame_offset"]),
                            "end": int(window["frame_offset"]) + int(window["window_length"]) - 1,
                        },
                        "raw_time_interval": {
                            "start": int(window["time_index_start"]),
                            "end": int(window["time_index_end"]),
                        },
                        "vehicle_selection": primary_vehicle_selection,
                        "workflow_id": workflow.workflow_id,
                        "workflow_selection": workflow_selector,
                        "workflow_dag_sha256": sha256_value(workflow_payload),
                        "expected_workload_fingerprint": workload_fingerprint(workflow),
                        "expected_request_fingerprint_mode": "static_dag_pre_run_plus_observed_cross_baseline",
                        "max_steps": int(max_steps),
                        "termination_rule": "workflow_completed_or_trainer_max_steps",
                        "rsu_layout": window["recommended_rsu_layout"],
                        "rsu_mapper_parameters": {
                            key: window.get(key)
                            for key in ("chosen_rsu_axis", "coverage_radius", "spacing", "dominant_axis")
                        },
                        "window_class": window["window_class"],
                    }
                )
    commit, dirty_audit = _git_identity(root)
    manifest: dict[str, Any] = {
        "identity": {
            "cache_baseline_fairness_manifest_version": MANIFEST_VERSION,
            "manifest_id": "pending",
            "created_at": created_at or utc_now(),
            "purpose": "paper-grade matched classical cache baseline fairness protocol",
            "scope": "G07 classical reactive baselines; reusable by future G08 oracle",
            "git_commit": commit,
            "dirty_worktree_audit": dirty_audit,
            "producer": {"script": "scripts/build_cache_baseline_fairness_manifest.py", "version": PRODUCER_VERSION},
            "protocol_status": "pre_run_validated_not_executed",
            "paper_claim_boundary": "validation and controlled runs are not formal or paper-ready evidence",
        },
        "dataset_provenance": {
            "mobility_source": "ngsim",
            "workflow_source": "alibaba2018",
            "inputs": [
                _path_identity(mobility_path, root, "ngsim_vehicle_trajectories", "NGSIMProvider/v1"),
                _path_identity(workflow_path, root, "alibaba_cluster_trace_2018_batch_task", "AlibabaDAGParser/legacy_batch_type"),
                _path_identity(catalog_path, root, "ppo_mec_sample_adapter_catalog", "AdapterCatalog/v1"),
                _path_identity(plan_path, root, "g07_non_hidden_window_plan", "frozen_window_plan/v1"),
            ],
            "selection_filter_parameters": {
                "max_workflows": int(max_workflows),
                "workflow_selector": workflow_selector,
                "min_tasks": int(min_tasks),
                "max_tasks": int(max_tasks),
                "max_mobility_rows": int(max_mobility_rows),
                "primary_vehicle_selection": primary_vehicle_selection,
                "window_mode": "mixed_informative",
                "predictor_kind": "baseline",
                "prediction_horizon": 3,
                "prediction_noise_std": 0.0,
                "prediction_confidence_scale": 1.0,
                "prediction_delay_steps": 0,
                "drop_handoff_prediction_prob": 0.0,
                "evaluation_unit_limit": evaluation_unit_limit,
            },
            "download_policy": "forbidden; builder validates existing local files only",
        },
        "window_workload_plan": {
            "window_plan_path": plan_path.relative_to(root).as_posix(),
            "window_plan_sha256": sha256_file(plan_path),
            "split": str(plan_payload.get("split", "unknown")),
            "hidden": False,
            "evaluation_units": units,
            "observed_request_validation": "each unit must match expected static workload fingerprint; observed event stream must match across all five baselines",
        },
        "seed_plan": {
            "benchmark_run_seeds": [int(seed) for seed in seeds],
            "seed_derivation_function": "identity(seed)",
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "per_run": [
                {
                    "benchmark_run_seed": int(seed),
                    "environment_seed": int(seed),
                    "workload_selection_seed": int(seed),
                    "policy_seed": int(seed),
                    "reactive_random_private_rng_seed": int(seed),
                    "learned_agent_seed": None,
                    "learned_checkpoint": None,
                }
                for seed in seeds
            ],
            "random_rng_contract": "private random.Random(run_seed); global RNG forbidden",
        },
        "cache_contract": {
            "capacity": {
                "enabled": True,
                "unit": capacity_unit,
                "rsu_adapter_slots": int(capacity_value) if capacity_unit == "adapter_slots" else None,
                "capacity_mb": float(capacity_value) if capacity_unit == "mb" else None,
                "comparison_stratum": f"capacity:{capacity_unit}:{capacity_value:g}",
            },
            "initial_per_rsu_cache_contents": initial_cache,
            "initial_cache_snapshot_sha256": sha256_value(initial_cache),
            "resident_sizes": _catalog_resident_sizes(catalog, all_workflows),
            "size_resolver": "AdapterCatalog.resolve_adapter_resident_size_mb/v1",
            "catalog_fallback_rule": "missing CacheObject uses explicit 64.0 MB catalog_fallback; invalid/nonpositive fails",
            "oversized_object_semantics": "reject_before_planning_without_policy_mutation",
            "multi_victim_semantics": "minimum_sufficient_ordered_prefix_atomic_commit",
            "adapter_transfer_size_source": "catalog resident size resolver",
            "state_migration_size_source": "AdapterCatalog.estimate_bundle_transfer_size_mb/v1",
            "capacity_contract_version": "cache_capacity_contract_v1",
            "cache_event_schema_version": "1.2.0",
            "cache_trace_context_version": "1.0.0",
            "cache_efficiency_metrics_contract_version": "1.0.0",
        },
        "baseline_matrix": [_load_baseline_entry(root, name) for name in BASELINE_NAMES],
        "metrics_aggregation": {
            "contract_version": "cache_efficiency_metrics_contract_v1.0.0",
            "metric_names": [
                "cache_object_hit_rate", "cache_byte_hit_rate", "cache_churn_mb",
                "cache_pollution_ratio", "cache_transfer_amplification_ratio",
                "cache_capacity_mean_occupancy", "cache_latency_saved_sum_ms",
            ],
            "nullable_aggregation_contract": "available finite values only; all unavailable => JSON null; available/unavailable counts required",
            "grouping_keys": ["cache_capacity_unit", "window_class", "agent_name"],
            "comparison_strata": [f"capacity:{capacity_unit}:{capacity_value:g}"],
            "unit_identity": ["benchmark_run_seed", "window_id", "workflow_id"],
            "benchmark_aggregation_protocol": "benchmark_main_results.aggregate_rows/paper_protocol_v1_20260409",
            "requested_hit_byte_coverage": "complete-case required",
            "pollution_availability": "requires cache_trace_context 1.0.0 and reconstructable residency",
            "future_reuse_horizons_steps": [1, 3, 6, 12],
            "latency_saved": {"availability": "unavailable", "comparable": False, "reason": "request-aligned observed and counterfactual latency absent"},
        },
        "artifact_plan": {
            "run_id": "resolved_at_runtime",
            "output_root": output_root,
            "per_episode_summary_path_template": "episodes/{window_id}/{workflow_id}/{agent_name}/seed_{seed}.summary.json",
            "raw_row_path": "benchmark_rows.csv",
            "aggregate_path": "aggregate_summary.json",
            "audit_path": "fairness_runtime_audit.json",
            "command_log_path": "resolved_command.txt",
            "resolved_manifest_path": "cache_baseline_fairness_manifest.json",
            "integrity_manifest_path": "artifact_integrity_manifest.json",
        },
        "claim_boundary": {
            "manifest_validated_is_experiment_run": False,
            "smoke_or_controlled_is_formal_evidence": False,
            "g07_produces_algorithm_ranking": False,
            "hidden_consumed": False,
            "causal_eviction_regret_requires_g08": True,
            "latency_saved_available": False,
            "paper_ready_requires": ["G14", "G15", "top_journal_review"],
        },
    }
    semantic_hash = semantic_protocol_sha256(manifest)
    manifest["identity"]["manifest_id"] = f"cbfm-{semantic_hash[:16]}"
    manifest["hashes"] = {
        "semantic_protocol_sha256": semantic_hash,
        "full_manifest_sha256": full_manifest_sha256(manifest),
        "semantic_hash_excludes": [
            "identity.manifest_id", "identity.created_at", "artifact_plan",
            "dataset normalized_absolute_path", "baseline config normalized_absolute_path",
            "hashes", "validation",
        ],
    }
    return manifest


def _require(mapping: dict[str, Any], key: str, errors: list[str], path: str) -> Any:
    if key not in mapping:
        errors.append(f"missing required field: {path}.{key}")
        return None
    return mapping[key]


def validate_manifest(manifest: dict[str, Any], *, root: str | Path, check_files: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    root = Path(root).resolve()
    try:
        _reject_non_finite(manifest)
        checked.append("json_safe_finite_values")
    except FairnessManifestError as exc:
        errors.append(str(exc))
    unknown = sorted(set(manifest) - ALLOWED_TOP_LEVEL_FIELDS)
    if unknown:
        errors.append(f"unknown critical top-level fields: {unknown}")
    for field in ALLOWED_TOP_LEVEL_FIELDS - {"validation"}:
        if field not in manifest:
            errors.append(f"missing required top-level field: {field}")
    identity = manifest.get("identity", {})
    version = _require(identity, "cache_baseline_fairness_manifest_version", errors, "identity")
    if not isinstance(version, str) or version.split(".", 1)[0] != "1":
        errors.append(f"unsupported manifest major version: {version!r}")
    dataset = manifest.get("dataset_provenance", {})
    if dataset.get("mobility_source") != "ngsim" or dataset.get("workflow_source") != "alibaba2018":
        errors.append("G07 dataset sources must be ngsim + alibaba2018")
    if dataset.get("download_policy", "").split(";", 1)[0] != "forbidden":
        errors.append("automatic dataset download must be forbidden")
    if check_files:
        resolved_inputs: dict[str, Path] = {}
        for item in dataset.get("inputs", []):
            logical_path = item.get("logical_path")
            candidate = Path(item.get("normalized_absolute_path") or "")
            if item.get("path_kind") == "repository_relative":
                candidate = root / str(logical_path)
            if not candidate.is_file():
                errors.append(f"dataset input missing: {logical_path}")
                continue
            if candidate.stat().st_size != item.get("size_bytes"):
                errors.append(f"dataset size mismatch: {logical_path}")
            if sha256_file(candidate) != item.get("sha256"):
                errors.append(f"dataset hash mismatch: {logical_path}")
            resolved_inputs[str(item.get("logical_dataset_id"))] = candidate
        checked.append("dataset_file_size_and_sha256")
    else:
        resolved_inputs = {}
    plan = manifest.get("window_workload_plan", {})
    units = plan.get("evaluation_units", [])
    if not units:
        errors.append("window_workload_plan.evaluation_units must be non-empty")
    seen_units: set[str] = set()
    expected_units: dict[tuple[int, str, str], dict[str, Any]] = {}
    for unit in units:
        unit_id = unit.get("evaluation_unit_id")
        if not unit_id or unit_id in seen_units:
            errors.append(f"duplicate/missing evaluation_unit_id: {unit_id!r}")
        seen_units.add(str(unit_id))
        for interval_name in ("raw_frame_interval", "raw_time_interval"):
            interval = unit.get(interval_name, {})
            if not isinstance(interval.get("start"), int) or not isinstance(interval.get("end"), int) or interval.get("end", -1) < interval.get("start", 0):
                errors.append(f"invalid {interval_name} for {unit_id}")
        fingerprint = unit.get("expected_workload_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            errors.append(f"invalid expected workload fingerprint for {unit_id}")
        key = (int(unit.get("benchmark_run_seed", -1)), str(unit.get("window_id")), str(unit.get("workflow_id")))
        expected_units[key] = unit
    if check_files:
        plan_file = resolved_inputs.get("g07_non_hidden_window_plan")
        if plan_file is not None:
            raw_plan = json.loads(plan_file.read_text(encoding="utf-8-sig"))
            plan_windows = raw_plan.get("selected_window_plan", raw_plan if isinstance(raw_plan, list) else [])
            by_window = {str(item.get("window_id")): item for item in plan_windows}
            for unit in units:
                source = by_window.get(str(unit.get("window_id")))
                if source is None:
                    errors.append(f"window absent from source plan: {unit.get('window_id')}")
                    continue
                expected_frame = {
                    "start": int(source["frame_offset"]),
                    "end": int(source["frame_offset"]) + int(source["window_length"]) - 1,
                }
                expected_time = {"start": int(source["time_index_start"]), "end": int(source["time_index_end"])}
                if unit.get("raw_frame_interval") != expected_frame:
                    errors.append(f"raw frame interval drift for {unit.get('evaluation_unit_id')}")
                if unit.get("raw_time_interval") != expected_time:
                    errors.append(f"raw time interval drift for {unit.get('evaluation_unit_id')}")
        selection = dataset.get("selection_filter_parameters", {})
        workflow_file = resolved_inputs.get("alibaba_cluster_trace_2018_batch_task")
        if workflow_file is not None:
            for seed in manifest.get("seed_plan", {}).get("benchmark_run_seeds", []):
                workflows = build_selected_workflow_states(
                    workflow_csv_path=workflow_file,
                    max_workflows=int(selection.get("max_workflows", 0)),
                    workflow_selector=str(selection.get("workflow_selector", "")),
                    min_tasks=int(selection.get("min_tasks", 0)),
                    max_tasks=int(selection.get("max_tasks", 0)),
                    random_seed=int(seed),
                )
                for workflow in workflows:
                    matching = [
                        unit for unit in units
                        if int(unit.get("benchmark_run_seed", -1)) == int(seed)
                        and unit.get("workflow_id") == workflow.workflow_id
                    ]
                    if not matching:
                        errors.append(f"selected workflow absent from evaluation units: seed={seed}, workflow={workflow.workflow_id}")
                    for unit in matching:
                        if unit.get("expected_workload_fingerprint") != workload_fingerprint(workflow):
                            errors.append(f"request/workload fingerprint drift for {unit.get('evaluation_unit_id')}")
    checked.append("raw_frame_time_intervals_and_workload_fingerprints")
    seed_plan = manifest.get("seed_plan", {})
    seeds = seed_plan.get("benchmark_run_seeds", [])
    if seed_plan.get("seed_derivation_version") != SEED_DERIVATION_VERSION:
        errors.append("unsupported seed derivation version")
    per_run = seed_plan.get("per_run", [])
    if [item.get("benchmark_run_seed") for item in per_run] != seeds:
        errors.append("seed plan per_run does not match benchmark_run_seeds")
    for item in per_run:
        seed = item.get("benchmark_run_seed")
        for field in ("environment_seed", "workload_selection_seed", "policy_seed", "reactive_random_private_rng_seed"):
            if item.get(field) != seed:
                errors.append(f"{field} must equal benchmark run seed {seed}")
    if "private random.Random" not in seed_plan.get("random_rng_contract", "") or "global RNG forbidden" not in seed_plan.get("random_rng_contract", ""):
        errors.append("Random must use private random.Random and forbid global RNG")
    checked.append("seed_identity_and_private_random_rng")
    cache = manifest.get("cache_contract", {})
    capacity = cache.get("capacity", {})
    if capacity.get("enabled") is not True or capacity.get("unit") not in {"adapter_slots", "mb"}:
        errors.append("cache capacity must be enabled with adapter_slots or mb")
    if capacity.get("unit") == "adapter_slots":
        if not isinstance(capacity.get("rsu_adapter_slots"), int) or capacity.get("rsu_adapter_slots", 0) <= 0 or capacity.get("capacity_mb") is not None:
            errors.append("adapter_slots stratum requires positive slots and null capacity_mb")
    if capacity.get("unit") == "mb":
        if not isinstance(capacity.get("capacity_mb"), (int, float)) or capacity.get("capacity_mb", 0) <= 0 or capacity.get("rsu_adapter_slots") is not None:
            errors.append("mb stratum requires positive capacity_mb and null slots")
    initial_cache = cache.get("initial_per_rsu_cache_contents")
    if not isinstance(initial_cache, list) or sha256_value(initial_cache) != cache.get("initial_cache_snapshot_sha256"):
        errors.append("initial cache snapshot hash mismatch")
    if cache.get("capacity_contract_version") != "cache_capacity_contract_v1":
        errors.append("capacity contract version mismatch")
    if str(cache.get("cache_event_schema_version", "")).split(".", 1)[0] != "1":
        errors.append("incompatible CacheEvent schema major version")
    if cache.get("cache_trace_context_version") != "1.0.0":
        errors.append("cache trace context version mismatch")
    if cache.get("cache_efficiency_metrics_contract_version") != "1.0.0":
        errors.append("cache efficiency metrics contract version mismatch")
    if cache.get("size_resolver") != "AdapterCatalog.resolve_adapter_resident_size_mb/v1" or not cache.get("catalog_fallback_rule"):
        errors.append("resident size/fallback contract mismatch")
    if capacity.get("unit") == "adapter_slots" and isinstance(initial_cache, list) and isinstance(capacity.get("rsu_adapter_slots"), int):
        if any(len(item.get("cached_adapter_ids", [])) > int(capacity.get("rsu_adapter_slots", 0) or 0) for item in initial_cache):
            errors.append("initial cache would require policy-specific slot trimming")
    if capacity.get("unit") == "mb" and isinstance(initial_cache, list):
        sizes = {item.get("adapter_id"): item.get("resident_size_mb") for item in cache.get("resident_sizes", [])}
        for item in initial_cache:
            if sum(float(sizes.get(adapter_id, math.inf)) for adapter_id in item.get("cached_adapter_ids", [])) > float(capacity.get("capacity_mb", 0) or 0) + 1e-9:
                errors.append("initial cache would require policy-specific MB trimming")
    checked.append("capacity_catalog_initial_cache_and_cacheevent_contracts")
    matrix = manifest.get("baseline_matrix", [])
    names = [entry.get("agent_identity", {}).get("name") for entry in matrix]
    if len(names) != len(set(names)):
        errors.append("baseline matrix contains duplicate agents")
    if set(names) != set(BASELINE_NAMES) or len(names) != 5:
        errors.append(f"baseline matrix must contain exactly {list(BASELINE_NAMES)}")
    for entry in matrix:
        name = entry.get("agent_identity", {}).get("name")
        if name not in POLICY_BY_BASELINE:
            errors.append(f"unknown baseline: {name!r}")
            continue
        policy = entry.get("eviction_policy", {})
        if policy.get("name") != POLICY_BY_BASELINE[name]:
            errors.append(f"agent-policy mismatch: {name} -> {policy.get('name')}")
        if entry.get("admission_control_identity") != "reactive_current_rsu_admission_v1":
            errors.append(f"admission/control drift for {name}")
        if entry.get("agent_specific_overrides"):
            errors.append(f"agent-specific overrides forbidden for {name}")
        if name == "reactive_random" and policy.get("policy_seed_rule") != "policy_seed_equals_benchmark_run_seed":
            errors.append("reactive_random seed rule mismatch")
        semantic = entry.get("semantic_config", {})
        if semantic.get("algorithm") != name or semantic.get("eviction_policy") != POLICY_BY_BASELINE[name]:
            errors.append(f"resolved config identity mismatch for {name}")
        if check_files:
            config_path = root / str(entry.get("config", {}).get("path", ""))
            if not config_path.is_file() or sha256_file(config_path) != entry.get("config", {}).get("sha256"):
                errors.append(f"config content hash mismatch for {name}")
    pairwise = build_pairwise_protocol_diff(manifest) if len(matrix) == 5 and set(names) == set(BASELINE_NAMES) else {"comparison_count": 0, "status": "fail", "comparisons": []}
    if pairwise.get("comparison_count") != 10 or pairwise.get("status") != "pass":
        errors.append("pairwise protocol symmetry failed")
    checked.append("five_baselines_binding_and_10_pairwise_diffs")
    metrics = manifest.get("metrics_aggregation", {})
    if metrics.get("contract_version") != "cache_efficiency_metrics_contract_v1.0.0":
        errors.append("metrics contract mismatch")
    if not metrics.get("nullable_aggregation_contract"):
        errors.append("nullable aggregation contract missing")
    if metrics.get("future_reuse_horizons_steps") != [1, 3, 6, 12]:
        errors.append("future reuse horizons mismatch")
    if metrics.get("latency_saved", {}).get("comparable") is not False:
        errors.append("latency saved must remain unavailable/non-comparable")
    checked.append("metrics_nullable_aggregation_and_latency_boundary")
    artifact_plan = manifest.get("artifact_plan", {})
    planned = [value for key, value in artifact_plan.items() if key.endswith("_path")]
    if len(planned) != len(set(planned)):
        errors.append("artifact output paths collide")
    try:
        actual_semantic = semantic_protocol_sha256(manifest)
        actual_full = full_manifest_sha256(manifest)
    except FairnessManifestError as exc:
        errors.append(str(exc))
        actual_semantic = "unavailable"
        actual_full = "unavailable"
    declared_semantic = manifest.get("hashes", {}).get("semantic_protocol_sha256")
    if actual_semantic != declared_semantic:
        errors.append("semantic protocol hash mismatch")
    expected_id = f"cbfm-{actual_semantic[:16]}" if actual_semantic != "unavailable" else "unavailable"
    if identity.get("manifest_id") != expected_id:
        errors.append("manifest ID does not match semantic protocol hash")
    if actual_full != manifest.get("hashes", {}).get("full_manifest_sha256"):
        errors.append("full manifest hash mismatch")
    checked.append("canonical_semantic_and_full_hashes")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "checked_invariants": checked,
        "manifest_id": identity.get("manifest_id"),
        "manifest_hash": actual_full,
        "semantic_protocol_hash": actual_semantic,
        "pairwise_protocol_diff": pairwise,
        "validated_at": utc_now(),
    }


def load_and_validate_manifest(path: str | Path, *, root: str | Path, check_files: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise FairnessManifestError("fairness manifest must be a JSON object")
    report = validate_manifest(manifest, root=root, check_files=check_files)
    if report["status"] != "pass":
        raise FairnessManifestError("; ".join(report["errors"]))
    return manifest, report


def expected_unit(manifest: dict[str, Any], *, seed: int, window_id: str, workflow_id: str) -> dict[str, Any]:
    matches = [
        unit
        for unit in manifest["window_workload_plan"]["evaluation_units"]
        if int(unit["benchmark_run_seed"]) == int(seed)
        and unit["window_id"] == window_id
        and unit["workflow_id"] == workflow_id
    ]
    if len(matches) != 1:
        raise FairnessManifestError(
            f"manifest evaluation unit missing/ambiguous for seed={seed}, window={window_id}, workflow={workflow_id}"
        )
    return matches[0]


def enforce_benchmark_args(args: Any, manifest: dict[str, Any]) -> None:
    expected_agents = list(BASELINE_NAMES)
    if list(args.agents) != expected_agents:
        raise FairnessManifestError(f"agents must exactly match manifest order {expected_agents}")
    seeds = manifest["seed_plan"]["benchmark_run_seeds"]
    if list(args.seeds) != seeds:
        raise FairnessManifestError(f"CLI seeds override frozen manifest: {args.seeds} != {seeds}")
    selection = manifest["dataset_provenance"]["selection_filter_parameters"]
    checks = {
        "max_workflows": selection["max_workflows"],
        "workflow_selector": selection["workflow_selector"],
        "min_tasks": selection["min_tasks"],
        "max_tasks": selection["max_tasks"],
        "max_mobility_rows": selection["max_mobility_rows"],
        "primary_vehicle_selection": selection["primary_vehicle_selection"],
        "window_mode": selection["window_mode"],
        "predictor_kind": selection["predictor_kind"],
        "prediction_horizon": selection["prediction_horizon"],
        "prediction_noise_std": selection["prediction_noise_std"],
        "prediction_confidence_scale": selection["prediction_confidence_scale"],
        "prediction_delay_steps": selection["prediction_delay_steps"],
        "drop_handoff_prediction_prob": selection["drop_handoff_prediction_prob"],
    }
    unit_steps = {int(unit["max_steps"]) for unit in manifest["window_workload_plan"]["evaluation_units"]}
    if len(unit_steps) != 1:
        raise FairnessManifestError("manifest contains mixed max_steps")
    checks["max_steps"] = next(iter(unit_steps))
    for field, expected in checks.items():
        if getattr(args, field) != expected:
            raise FairnessManifestError(f"CLI {field} overrides frozen manifest: {getattr(args, field)!r} != {expected!r}")
    if args.mobility_source != "ngsim":
        raise FairnessManifestError("CLI mobility_source overrides frozen manifest")
    inputs = {
        item["logical_dataset_id"]: item
        for item in manifest["dataset_provenance"]["inputs"]
    }
    workflow_path = Path(inputs["alibaba_cluster_trace_2018_batch_task"]["normalized_absolute_path"]).resolve()
    if Path(args.workflow_csv_path).resolve() != workflow_path or sha256_file(workflow_path) != inputs["alibaba_cluster_trace_2018_batch_task"]["sha256"]:
        raise FairnessManifestError("CLI workflow path/content overrides frozen manifest")
    if args.mobility_csv_path:
        mobility_path = Path(inputs["ngsim_vehicle_trajectories"]["normalized_absolute_path"]).resolve()
        if Path(args.mobility_csv_path).resolve() != mobility_path:
            raise FairnessManifestError("CLI mobility path overrides frozen manifest")
    plan_path = Path(args.window_plan_path).resolve()
    manifest_plan = Path(manifest["window_workload_plan"]["window_plan_path"])
    if not manifest_plan.is_absolute():
        manifest_plan = Path(args._fairness_root) / manifest_plan
    if plan_path != manifest_plan.resolve() or sha256_file(plan_path) != manifest["window_workload_plan"]["window_plan_sha256"]:
        raise FairnessManifestError("CLI window plan path/content overrides frozen manifest")
    capacity = manifest["cache_contract"]["capacity"]
    if capacity["unit"] != "adapter_slots":
        raise FairnessManifestError("benchmark_main_results currently consumes adapter_slots G07 strata only")
    if int(args.classical_cache_slots) != int(capacity["rsu_adapter_slots"]):
        raise FairnessManifestError("CLI classical_cache_slots overrides frozen manifest capacity")
    if float(args.reward_positive_offset) != 0.0:
        raise FairnessManifestError("G07 fairness manifest freezes reward_positive_offset=0.0")


def stamp_summary_provenance(summary: dict[str, Any], manifest: dict[str, Any], unit: dict[str, Any]) -> None:
    provenance = {
        "fairness_manifest_status": "validated",
        "fairness_manifest_id": manifest["identity"]["manifest_id"],
        "fairness_manifest_hash": manifest["hashes"]["full_manifest_sha256"],
        "fairness_semantic_protocol_hash": manifest["hashes"]["semantic_protocol_sha256"],
        "evaluation_unit_id": unit["evaluation_unit_id"],
        "expected_workload_fingerprint": unit["expected_workload_fingerprint"],
        "observed_request_stream_fingerprint": observed_request_stream_fingerprint(summary),
    }
    summary.setdefault("run_info", {}).update(provenance)


def validate_observed_fingerprint_matrix(matrix: dict[str, dict[str, str]]) -> None:
    for unit_id, by_agent in matrix.items():
        if set(by_agent) != set(BASELINE_NAMES):
            raise FairnessManifestError(f"unit {unit_id} did not execute all five baselines")
        if len(set(by_agent.values())) != 1:
            raise FairnessManifestError(
                f"observed request stream fingerprint mismatch across baselines for {unit_id}"
            )


def legacy_unavailable_provenance() -> dict[str, Any]:
    return {
        "fairness_manifest_status": "unavailable",
        "fairness_manifest_id": None,
        "fairness_manifest_hash": None,
        "fairness_semantic_protocol_hash": None,
        "evaluation_unit_id": None,
        "expected_workload_fingerprint": None,
        "observed_request_stream_fingerprint": None,
    }
