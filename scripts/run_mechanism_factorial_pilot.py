"""Run the controlled base-sharing x workflow-state-migration pilot.

This is a new supplemental study over real NGSIM mobility and real Alibaba DAG
structure.  Adapter labels, typed object sizes, capacity pressure, and mechanism
switches are controlled experimental factors; outputs are not a new holdout and
must not be merged into historical formal evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
)


STUDY_VERSION = "controlled_mechanism_factorial_pilot_v1"
MECHANISM_PROFILE_VERSION = "controlled_mechanism_factorial_v1"
ARMS = (
    ("sharing_on_migration_on", True, True),
    ("sharing_off_migration_on", False, True),
    ("sharing_on_migration_off", True, False),
    ("sharing_off_migration_off", False, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/mechanism_factorial_pilot_v1.yaml"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Optional root used to resolve repository-relative real-data paths.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot config must be a mapping")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _object_fingerprint(row: dict[str, Any]) -> str:
    return AdapterCatalog.compute_object_fingerprint(row)


def _adapter_slug(adapter_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in adapter_id)


def build_factorial_catalog(
    source_catalog: AdapterCatalog,
    *,
    base_sharing_enabled: bool,
    initial_adapter_id: str,
) -> AdapterCatalog:
    """Build matched shared-base or adapter-specific-base physical catalogs."""

    payload = source_catalog.to_dict()
    source_base = source_catalog.get_typed_base("veh_base_v1")
    v1_adapters = [
        item
        for item in payload["typed_cache_objects"]
        if item.get("object_type") == "adapter"
        and item.get("required_base_model_id") == "veh_base_v1"
    ]
    if initial_adapter_id not in {str(item.get("adapter_id")) for item in v1_adapters}:
        raise ValueError(f"initial adapter is not backed by veh_base_v1: {initial_adapter_id}")

    if base_sharing_enabled:
        initial_base_object_id = source_base.object_id
    else:
        base_template = source_base.to_dict()
        replica_rows: list[dict[str, Any]] = []
        replica_compatibility: dict[str, list[str]] = {}
        replica_vehicle_models: list[dict[str, Any]] = []
        base_by_adapter: dict[str, tuple[str, str]] = {}
        for adapter_row in v1_adapters:
            adapter_id = str(adapter_row["adapter_id"])
            suffix = _adapter_slug(adapter_id)
            replica_base_id = f"veh_base_v1__{suffix}"
            replica_object_id = f"base:{replica_base_id}"
            replica = deepcopy(base_template)
            replica.update(
                object_id=replica_object_id,
                base_model_id=replica_base_id,
                shareability_scope=f"adapter_specific:{adapter_id}",
                evictability="evictable",
                provenance={
                    **dict(replica.get("provenance") or {}),
                    "controlled_factor": "no_base_sharing",
                    "logical_source_base_model_id": "veh_base_v1",
                    "adapter_specific_replica_for": adapter_id,
                    "evictability_rationale": "exclusive adapter/base pair must be atomically replaceable",
                },
                stable_fingerprint="",
            )
            replica["stable_fingerprint"] = _object_fingerprint(replica)
            replica_rows.append(replica)
            replica_compatibility[replica_base_id] = [adapter_id]
            replica_vehicle_models.append(
                {
                    "base_model_id": replica_base_id,
                    "family": str(source_base.base_model_family),
                    "memory_mb": next(
                        float(item["memory_mb"])
                        for item in payload["vehicle_base_models"]
                        if item["base_model_id"] == "veh_base_v1"
                    ),
                }
            )
            base_by_adapter[adapter_id] = (replica_base_id, replica_object_id)

        for adapter_row in v1_adapters:
            adapter_id = str(adapter_row["adapter_id"])
            replica_base_id, replica_object_id = base_by_adapter[adapter_id]
            adapter_row["required_base_model_id"] = replica_base_id
            adapter_row["dependency_ids"] = [replica_object_id]
            adapter_row["provenance"] = {
                **dict(adapter_row.get("provenance") or {}),
                "controlled_factor": "no_base_sharing",
                "logical_source_base_model_id": "veh_base_v1",
            }
            adapter_row["stable_fingerprint"] = ""
            adapter_row["stable_fingerprint"] = _object_fingerprint(adapter_row)

        payload["typed_cache_objects"].extend(replica_rows)
        payload["vehicle_base_models"].extend(replica_vehicle_models)
        payload["compatibility_map"]["veh_base_v1"] = []
        payload["compatibility_map"].update(replica_compatibility)
        initial_base_object_id = base_by_adapter[initial_adapter_id][1]

    initial_adapter_object_id = str(
        next(
            item["object_id"]
            for item in payload["typed_cache_objects"]
            if item.get("object_type") == "adapter"
            and item.get("adapter_id") == initial_adapter_id
        )
    )
    payload["rsu_typed_cache_profiles"] = [
        {
            "rsu_id": rsu_id,
            "resident_object_ids": [initial_base_object_id, initial_adapter_object_id],
        }
        for rsu_id in ("rsu_a", "rsu_b", "rsu_c")
    ]
    return AdapterCatalog.from_dict(payload)


def bind_workflow_to_catalog(workflow_state: Any, catalog: AdapterCatalog) -> Any:
    bound = deepcopy(workflow_state)
    for node in bound.nodes:
        node.required_base_model = str(
            catalog.get_typed_adapter(str(node.required_adapter)).required_base_model_id
        )
    return bound


def _load_workflows(config: dict[str, Any], workflow_path: Path) -> list[Any]:
    data = dict(config["data"])
    requested_ids = [str(item) for item in data["workflow_ids"]]
    samples = WorkflowDatasetBuilder().build_alibaba_samples(
        csv_path=workflow_path,
        limit_jobs=max(64, len(requested_ids) * 24),
        min_tasks=int(data["min_tasks"]),
        max_tasks=int(data["max_tasks"]),
        adapter_assignment_profile="semantic_ai_service",
    )
    by_id = {str(sample["workflow_id"]): sample for sample in samples}
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"configured Alibaba workflows are missing: {missing}")
    builder = WorkflowDatasetBuilder()
    return [builder.sample_to_workflow_state(by_id[item]) for item in requested_ids]


def _sum_event_field(summary: dict[str, Any], field: str) -> float:
    return round(
        sum(float(event.get(field, 0.0) or 0.0) for event in summary["cache_event_trace"]),
        6,
    )


def _sum_transfer_type(summary: dict[str, Any], object_type: str) -> float:
    return round(
        sum(
            float((event.get("transfer_mb_by_type") or {}).get(object_type, 0.0) or 0.0)
            for event in summary["cache_event_trace"]
        ),
        6,
    )


def _augment_row(
    row: dict[str, Any],
    summary: dict[str, Any],
    *,
    study_version: str,
    arm_id: str,
    base_sharing_enabled: bool,
    migration_enabled: bool,
    capacity_mb: float,
) -> dict[str, Any]:
    events = list(summary["cache_event_trace"])
    steps = list(summary["step_trace"])
    adapters = sorted(
        {str(event["adapter_id"]) for event in events if event.get("adapter_id")}
    )
    bases = sorted(
        {
            str(item["object_id"])
            for event in events
            for item in list(event.get("requested_typed_objects") or [])
            if item.get("object_type") == "base_model"
        }
    )
    augmented = dict(row)
    augmented.update(
        study_version=study_version,
        evidence_scope="observed_data_controlled_supplement_not_holdout",
        arm_id=arm_id,
        controller="popularity_cache_heuristic_fixed_rule",
        base_sharing_enabled=base_sharing_enabled,
        workflow_state_migration_enabled=migration_enabled,
        migration_disabled_fallback=(
            "not_applicable" if migration_enabled else "cold_restart_one_request"
        ),
        capacity_mb=capacity_mb,
        distinct_adapter_count=len(adapters),
        distinct_base_object_count=len(bases),
        requested_adapter_ids="|".join(adapters),
        requested_base_object_ids="|".join(bases),
        base_transfer_mb=_sum_transfer_type(summary, "base_model"),
        adapter_transfer_mb=_sum_transfer_type(summary, "adapter"),
        workflow_state_transfer_mb=_sum_transfer_type(summary, "workflow_state"),
        total_transfer_mb=_sum_event_field(summary, "adapter_transfer_size_mb")
        + _sum_event_field(summary, "state_migration_size_mb")
        + _sum_transfer_type(summary, "base_model"),
        migration_requested_event_count=sum(
            int(bool(event.get("migration_requested"))) for event in events
        ),
        migration_realized_event_count=sum(
            int(bool(event.get("migration_realized"))) for event in events
        ),
        migration_suppressed_step_count=sum(
            int(bool(step.get("migration_suppressed"))) for step in steps
        ),
        capacity_rejection_count=sum(
            int(bool(event.get("capacity_rejection_reason"))) for event in events
        ),
        action_sequence="|".join(str(step.get("action_id")) for step in steps),
    )
    return augmented


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "successful_episode_rate",
        "workflow_continuity_rate",
        "full_service_ready_request_rate",
        "full_service_ready_byte_hit_rate",
        "transfer_mb_per_request",
        "backhaul_traffic_cost",
        "end_to_end_workflow_delay",
        "cache_occupancy_rate",
        "eviction_count",
        "base_transfer_mb",
        "adapter_transfer_mb",
        "workflow_state_transfer_mb",
        "migration_realized_event_count",
        "migration_suppressed_step_count",
        "capacity_rejection_count",
    )
    output: list[dict[str, Any]] = []
    for arm_id, sharing, migration in ARMS:
        group = [row for row in rows if row["arm_id"] == arm_id]
        summary: dict[str, Any] = {
            "arm_id": arm_id,
            "base_sharing_enabled": sharing,
            "workflow_state_migration_enabled": migration,
            "episode_count": len(group),
        }
        for metric in metrics:
            finite = [
                float(row[metric])
                for row in group
                if row.get(metric) not in (None, "")
            ]
            summary[f"{metric}_mean"] = round(fmean(finite), 6) if finite else None
        summary["distinct_adapter_count_max"] = max(
            (int(row["distinct_adapter_count"]) for row in group), default=0
        )
        summary["distinct_base_object_count_max"] = max(
            (int(row["distinct_base_object_count"]) for row in group), default=0
        )
        summary["handoff_total_count_sum"] = round(
            sum(float(row.get("handoff_total_count", 0.0) or 0.0) for row in group),
            6,
        )
        output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config)
    study_version = str(config.get("study_version") or STUDY_VERSION)
    data = dict(config["data"])
    output_dir = args.output_dir or Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data_root = (args.data_root or ROOT_DIR).expanduser().resolve()
    mobility_config_path = Path(data["mobility_csv_path"]).expanduser()
    workflow_config_path = Path(data["workflow_csv_path"]).expanduser()
    mobility_path = (
        mobility_config_path
        if mobility_config_path.is_absolute()
        else data_root / mobility_config_path
    ).resolve()
    workflow_path = (
        workflow_config_path
        if workflow_config_path.is_absolute()
        else data_root / workflow_config_path
    ).resolve()
    for path in (mobility_path, workflow_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_sources = dict(config.get("expected_sources") or {})
    source_rows: dict[str, Any] = {}
    for role, path in (("mobility", mobility_path), ("workflow", workflow_path)):
        observed_size = path.stat().st_size
        expected = dict(expected_sources.get(role) or {})
        if expected.get("size_bytes") is not None and observed_size != int(
            expected["size_bytes"]
        ):
            raise ValueError(f"{role} dataset size mismatch")
        observed_sha = _file_sha256(path) if config.get("verify_source_sha256", True) else None
        if expected.get("sha256") and observed_sha != expected["sha256"]:
            raise ValueError(f"{role} dataset SHA-256 mismatch")
        source_rows[role] = {
            "path": str(path),
            "size_bytes": observed_size,
            "sha256": observed_sha,
        }

    workflows = _load_workflows(config, workflow_path)
    source_catalog = AdapterCatalog.from_json(
        ROOT_DIR / "src/data/model_catalog/typed_model_cache_controlled.json"
    )
    capacity_mb = float(config["capacity_mb"])
    cache_capacity_profile = {
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "enabled": True,
        "unit": "mb",
        "capacity_mb": capacity_mb,
        "count_base_model_separately": True,
        "eviction_policy": str(config.get("eviction_policy", "lru")),
        "telemetry_enabled": True,
    }

    window_plan_value = data.get("window_plan_path")
    if window_plan_value:
        window_plan_path = Path(str(window_plan_value)).expanduser()
        if not window_plan_path.is_absolute():
            window_plan_path = ROOT_DIR / window_plan_path
        window_payload = json.loads(window_plan_path.read_text(encoding="utf-8"))
        window_payload["selected_windows"] = list(
            window_payload.get("selected_window_plan") or []
        )
    else:
        _, window_payload = resolve_window_candidates(
            root_dir=ROOT_DIR,
            mobility_source="ngsim",
            mobility_csv_path=str(mobility_path),
            lust_scenario_root="",
            max_mobility_rows=int(data["max_mobility_rows"]),
            rsu_layout=str(data["rsu_layout"]),
            frame_offset=int(data.get("frame_offset", 0)),
            window_length=int(data["window_length"]),
            window_selector="max_handoff_candidate",
            window_count=int(data["window_count"]),
            window_scan_stride=int(data["window_scan_stride"]),
            random_seed=int(config["seed"]),
            window_mode="activating_only",
            enforce_non_overlapping_selection=True,
        )
    selected_windows = list(window_payload["selected_windows"])
    if len(selected_windows) != int(data["window_count"]):
        raise RuntimeError("requested activating window count is unavailable")

    rows: list[dict[str, Any]] = []
    episode_paths: list[str] = []
    checkpoint_map = {"popularity_cache_heuristic": ""}
    for arm_id, sharing, migration in ARMS:
        catalog = build_factorial_catalog(
            source_catalog,
            base_sharing_enabled=sharing,
            initial_adapter_id=str(config["initial_adapter_id"]),
        )
        mechanism_profile = {
            "mechanism_factorial_profile_version": MECHANISM_PROFILE_VERSION,
            "profile_id": arm_id,
            "base_sharing_enabled": sharing,
            "workflow_state_migration_enabled": migration,
            "migration_disabled_fallback": "cold_restart_one_request",
        }
        runtime_contract = {
            "model_cache_profile": "typed_base_adapter_state_v1",
            "typed_catalog_fingerprint": catalog.canonical_fingerprint(),
            "runtime_contract_sha256": _canonical_sha256(
                {
                    "catalog": catalog.canonical_fingerprint(),
                    "capacity": cache_capacity_profile,
                    "mechanism": mechanism_profile,
                }
            ),
            "cache_event_schema_version": "1.3.0",
            "cache_efficiency_metrics_contract_version": "1.1.0",
        }
        for window in selected_windows:
            bundle = load_window_bundle(
                root_dir=ROOT_DIR,
                mobility_source="ngsim",
                mobility_csv_path=str(mobility_path),
                lust_scenario_root="",
                max_mobility_rows=int(data["max_mobility_rows"]),
                rsu_layout=str(window["recommended_rsu_layout"]),
                frame_offset=int(window["frame_offset"]),
                window_length=int(window["window_length"]),
                random_seed=int(config["seed"]),
            )
            bundle.rsu_metadata["window_rank"] = window["window_rank"]
            bundle.rsu_metadata["window_class"] = window["window_class"]
            for workflow in workflows:
                bound_workflow = bind_workflow_to_catalog(workflow, catalog)
                unit_id = f"{study_version}/{window['window_id']}/{workflow.workflow_id}/{arm_id}"
                exposure = build_episode_formal_request_exposure(
                    workflow_state=bound_workflow,
                    mobility_bundle=bundle,
                    adapter_catalog=catalog,
                    max_steps=int(data["max_steps"]),
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    cache_capacity_profile=cache_capacity_profile,
                    evaluation_unit={
                        "evaluation_unit_id": unit_id,
                        "workflow_id": bound_workflow.workflow_id,
                        "window_id": window["window_id"],
                    },
                    source_provenance={
                        "study_version": study_version,
                        "evidence_scope": "observed_data_controlled_supplement_not_holdout",
                        "mobility_sha256": source_rows["mobility"]["sha256"],
                        "workflow_sha256": source_rows["workflow"]["sha256"],
                    },
                )
                episode_started = time.perf_counter()
                summary = run_real_episode(
                    root_dir=ROOT_DIR,
                    agent_name="popularity_cache_heuristic",
                    checkpoint_map=checkpoint_map,
                    workflow_state=bound_workflow,
                    workflow_source_path=str(workflow_path),
                    mobility_bundle=bundle,
                    seed=int(config["seed"]),
                    max_steps=int(data["max_steps"]),
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    reward_positive_offset=0.0,
                    run_metadata={
                        "script": "scripts/run_mechanism_factorial_pilot.py",
                        "study_version": study_version,
                        "mode": "controlled_mechanism_factorial_pilot",
                        "window_mode": "joint_mechanism_opportunity",
                        "window_rank": window["window_rank"],
                        "window_class": window["window_class"],
                        "evaluation_unit_id": unit_id,
                        "evidence_scope": "observed_data_controlled_supplement_not_holdout",
                    },
                    adapter_catalog_override=catalog,
                    workflow_state_override=bound_workflow,
                    cache_capacity_profile=cache_capacity_profile,
                    model_cache_runtime_contract=runtime_contract,
                    formal_request_exposure_trace=exposure,
                    mechanism_profile=mechanism_profile,
                )
                summary["compute_audit"] = {
                    "wall_clock_sec": round(time.perf_counter() - episode_started, 6),
                    "wall_clock_sec_per_step": round(
                        (time.perf_counter() - episode_started)
                        / max(len(summary.get("step_trace", [])), 1),
                        6,
                    ),
                }
                episode_path = (
                    output_dir
                    / "episodes"
                    / arm_id
                    / str(window["window_id"])
                    / f"{bound_workflow.workflow_id}.json"
                )
                _write_json(episode_path, summary)
                episode_paths.append(str(episode_path))
                row = _augment_row(
                    summary_to_row(summary),
                    summary,
                    study_version=study_version,
                    arm_id=arm_id,
                    base_sharing_enabled=sharing,
                    migration_enabled=migration,
                    capacity_mb=capacity_mb,
                )
                row["episode_path"] = str(episode_path)
                rows.append(row)

    aggregate = _aggregate(rows)
    _write_csv(output_dir / "episode_results.csv", rows)
    _write_csv(output_dir / "four_arm_performance.csv", aggregate)
    _write_json(output_dir / "selected_windows.json", selected_windows)

    event_checks = {
        "multiple_adapters_observed": max(row["distinct_adapter_count"] for row in rows) >= 3,
        "shared_base_observed": any(
            row["base_sharing_enabled"]
            and row["distinct_adapter_count"] >= 2
            and row["distinct_base_object_count"] == 1
            for row in rows
        ),
        "adapter_specific_bases_observed": any(
            not row["base_sharing_enabled"]
            and row["distinct_base_object_count"] >= 2
            for row in rows
        ),
        "capacity_competition_observed": any(
            float(row.get("eviction_count", 0.0) or 0.0) > 0.0
            or int(row["capacity_rejection_count"]) > 0
            for row in rows
        ),
        "handoff_observed": sum(
            float(row.get("handoff_total_count", 0.0) or 0.0) for row in rows
        )
        > 0,
        "migration_realized_when_enabled": any(
            row["workflow_state_migration_enabled"]
            and int(row["migration_realized_event_count"]) > 0
            for row in rows
        ),
        "migration_suppressed_when_disabled": any(
            not row["workflow_state_migration_enabled"]
            and int(row["migration_suppressed_step_count"]) > 0
            for row in rows
        ),
        "transfer_accounting_observed": any(float(row["total_transfer_mb"]) > 0 for row in rows),
    }
    completion = {
        "study_version": study_version,
        "status": "completed" if all(event_checks.values()) else "completed_with_failed_event_check",
        "evidence_scope": "observed_data_controlled_supplement_not_holdout",
        "git_commit": _git_commit(),
        "config_path": str(args.config),
        "data_root": str(data_root),
        "config_sha256": _canonical_sha256(config),
        "source_provenance": source_rows,
        "episode_count": len(rows),
        "arm_count": len(aggregate),
        "event_checks": event_checks,
        "selected_window_ids": [row["window_id"] for row in selected_windows],
        "outputs": {
            "episode_results": str(output_dir / "episode_results.csv"),
            "four_arm_performance": str(output_dir / "four_arm_performance.csv"),
            "episodes": episode_paths,
        },
        "wall_clock_sec": round(time.time() - started, 6),
        "claim_boundary": (
            "Controlled observed-data pilot only; not independent holdout, not formal, "
            "not evidence of algorithm superiority."
        ),
    }
    _write_json(output_dir / "completion_receipt.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
