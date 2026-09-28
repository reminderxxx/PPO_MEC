"""Diagnose truncation versus true request failure for the mechanism workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_mechanism_factorial_pilot import (
    bind_workflow_to_catalog,
    build_factorial_catalog,
)
from src.agents.handoff_first_feasibility_agent import HandoffFirstFeasibilityAgent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    load_window_bundle,
    run_real_episode,
    summary_to_row,
)


DIAGNOSTIC_VERSION = "mechanism_completion_feasibility_v1"
CONTROLLERS = ("popularity_cache_heuristic", "handoff_first_feasibility")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT_DIR, text=True
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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


def _load_workflows(path: Path, workflow_ids: list[str]) -> list[Any]:
    samples = WorkflowDatasetBuilder().build_alibaba_samples(
        csv_path=path,
        limit_jobs=max(96, len(workflow_ids) * 24),
        min_tasks=5,
        max_tasks=20,
        adapter_assignment_profile="semantic_ai_service",
    )
    by_id = {str(row["workflow_id"]): row for row in samples}
    missing = sorted(set(workflow_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"configured workflows missing: {missing}")
    builder = WorkflowDatasetBuilder()
    return [builder.sample_to_workflow_state(by_id[item]) for item in workflow_ids]


def _workload_fingerprint(exposure: dict[str, Any]) -> str:
    fields = (
        "request_order",
        "step_index",
        "time_index",
        "vehicle_id",
        "workflow_id",
        "node_id",
        "required_base_model",
        "adapter_id",
        "object_id",
        "request_rsu_id",
        "current_service_rsu_id",
        "eligible_service_rsu_ids",
        "eligible_cache_target_rsu_ids",
    )
    return _canonical_sha256(
        [{key: row.get(key) for key in fields} for row in exposure["requests"]]
    )


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for horizon in sorted({int(row["max_steps"]) for row in rows}):
        for controller in CONTROLLERS:
            group = [
                row
                for row in rows
                if int(row["max_steps"]) == horizon
                and row["controller"] == controller
            ]
            result.append(
                {
                    "max_steps": horizon,
                    "controller": controller,
                    "evaluation_unit_count": len(group),
                    "completed_count": sum(
                        int(bool(row["workflow_completed_under_exogenous_execution"]))
                        for row in group
                    ),
                    "completed_rate": round(
                        fmean(
                            float(
                                bool(
                                    row[
                                        "workflow_completed_under_exogenous_execution"
                                    ]
                                )
                            )
                            for row in group
                        ),
                        6,
                    ),
                    "right_censored_count": sum(
                        int(bool(row["right_censored"])) for row in group
                    ),
                    "noncensored_failed_count": sum(
                        int(
                            not bool(row["right_censored"])
                            and not bool(
                                row["workflow_completed_under_exogenous_execution"]
                            )
                        )
                        for row in group
                    ),
                    "request_failure_count": sum(
                        int(row["request_failure_count"]) for row in group
                    ),
                    "full_service_ready_request_rate_mean": round(
                        fmean(
                            float(row["full_service_ready_request_rate"])
                            for row in group
                        ),
                        6,
                    ),
                    "workflow_continuity_rate_mean": round(
                        fmean(float(row["workflow_continuity_rate"]) for row in group),
                        6,
                    ),
                }
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("diagnostic config must be a mapping")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data_root = args.data_root.resolve()
    data = dict(config["data"])
    mobility_path = (data_root / str(data["mobility_csv_path"])).resolve()
    workflow_path = (data_root / str(data["workflow_csv_path"])).resolve()
    expected = dict(config["expected_sources"])
    for label, path in (("mobility", mobility_path), ("workflow", workflow_path)):
        if path.stat().st_size != int(expected[label]["size_bytes"]):
            raise ValueError(f"{label} size mismatch")
        if _file_sha256(path) != str(expected[label]["sha256"]):
            raise ValueError(f"{label} SHA-256 mismatch")

    window_plan_path = (ROOT_DIR / str(data["window_plan_path"])).resolve()
    window_plan = json.loads(window_plan_path.read_text(encoding="utf-8"))
    windows = list(window_plan["selected_window_plan"])
    workflows = _load_workflows(
        workflow_path, [str(item) for item in data["workflow_ids"]]
    )
    source_catalog = AdapterCatalog.from_json(
        ROOT_DIR / "src/data/model_catalog/typed_model_cache_controlled.json"
    )
    catalog = build_factorial_catalog(
        source_catalog,
        base_sharing_enabled=True,
        initial_adapter_id="adapter_perception",
    )
    capacity_profile = {
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "enabled": True,
        "unit": "mb",
        "capacity_mb": 360.0,
        "count_base_model_separately": True,
        "eviction_policy": "lru",
        "eviction_policy_seed": int(config["seed"]),
        "telemetry_enabled": True,
    }
    mechanism_profile = {
        "mechanism_factorial_profile_version": "controlled_mechanism_factorial_v1",
        "profile_id": "sharing_on_migration_on",
        "base_sharing_enabled": True,
        "workflow_state_migration_enabled": True,
        "migration_disabled_fallback": "cold_restart_one_request",
    }
    runtime_contract = {
        "model_cache_profile": "typed_base_adapter_state_v1",
        "typed_catalog_fingerprint": catalog.canonical_fingerprint(),
        "runtime_contract_sha256": _canonical_sha256(
            {
                "catalog": catalog.canonical_fingerprint(),
                "capacity": capacity_profile,
                "mechanism": mechanism_profile,
            }
        ),
        "cache_event_schema_version": "1.3.0",
        "cache_efficiency_metrics_contract_version": "1.1.0",
    }
    rows: list[dict[str, Any]] = []
    episode_paths: list[str] = []
    for max_steps in [int(item) for item in config["max_steps_values"]]:
        for controller in CONTROLLERS:
            for window in windows:
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
                    bound = bind_workflow_to_catalog(workflow, catalog)
                    unit_id = (
                        f"{DIAGNOSTIC_VERSION}/h{max_steps}/{window['window_id']}/"
                        f"{bound.workflow_id}/{controller}"
                    )
                    exposure = build_episode_formal_request_exposure(
                        workflow_state=bound,
                        mobility_bundle=bundle,
                        adapter_catalog=catalog,
                        max_steps=max_steps,
                        mobility_source="ngsim",
                        primary_vehicle_selection="handoff_pressure",
                        cache_capacity_profile=capacity_profile,
                        evaluation_unit={
                            "evaluation_unit_id": unit_id,
                            "workflow_id": bound.workflow_id,
                            "window_id": window["window_id"],
                        },
                        source_provenance={
                            "diagnostic_version": DIAGNOSTIC_VERSION,
                            "role": "completion_feasibility_not_algorithm_baseline",
                        },
                    )
                    summary = run_real_episode(
                        root_dir=ROOT_DIR,
                        agent_name=controller,
                        checkpoint_map={controller: ""},
                        workflow_state=bound,
                        workflow_source_path=str(workflow_path),
                        mobility_bundle=bundle,
                        seed=int(config["seed"]),
                        max_steps=max_steps,
                        mobility_source="ngsim",
                        primary_vehicle_selection="handoff_pressure",
                        reward_positive_offset=0.0,
                        run_metadata={
                            "script": "scripts/run_mechanism_completion_diagnostic.py",
                            "diagnostic_version": DIAGNOSTIC_VERSION,
                            "role": "completion_feasibility_not_algorithm_baseline",
                            "window_id": window["window_id"],
                            "window_class": window["window_class"],
                            "evaluation_unit_id": unit_id,
                        },
                        adapter_catalog_override=catalog,
                        workflow_state_override=bound,
                        cache_capacity_profile=capacity_profile,
                        model_cache_runtime_contract=runtime_contract,
                        formal_request_exposure_trace=exposure,
                        mechanism_profile=mechanism_profile,
                        agent_override=(
                            HandoffFirstFeasibilityAgent()
                            if controller == "handoff_first_feasibility"
                            else None
                        ),
                    )
                    episode_path = (
                        output_dir
                        / "episodes"
                        / f"h{max_steps}"
                        / controller
                        / str(window["window_id"])
                        / f"{bound.workflow_id}.json"
                    )
                    _write_json(episode_path, summary)
                    episode_paths.append(str(episode_path))
                    endpoint = dict(summary["formal_request_execution_audit"])
                    censoring = dict(exposure["exposure_censoring"])
                    failures = sum(
                        int(
                            event.get("event_type") == "request"
                            and not bool(event.get("service_success"))
                        )
                        for event in summary["cache_event_trace"]
                    )
                    row = summary_to_row(summary)
                    rows.append(
                        {
                            "max_steps": max_steps,
                            "controller": controller,
                            "diagnostic_role": "feasibility_only_not_baseline",
                            "window_id": window["window_id"],
                            "workflow_id": bound.workflow_id,
                            "planned_dag_node_count": len(bound.nodes),
                            "exposed_request_count": len(exposure["requests"]),
                            "right_censored": bool(censoring["right_censored"]),
                            "censoring_reason": censoring.get("reason"),
                            "workflow_completed_under_exogenous_execution": bool(
                                endpoint[
                                    "workflow_completed_under_exogenous_execution"
                                ]
                            ),
                            "request_failure_count": failures,
                            "full_service_ready_request_rate": endpoint[
                                "full_service_ready_request_rate"
                            ],
                            "workflow_continuity_rate": endpoint[
                                "workflow_continuity_rate"
                            ],
                            "handoff_failure_rate": row.get("handoff_failure_rate"),
                            "transfer_mb_per_request": endpoint[
                                "transfer_mb_per_request"
                            ],
                            "workload_request_fingerprint": _workload_fingerprint(
                                exposure
                            ),
                            "episode_path": str(episode_path),
                        }
                    )

    _write_csv(output_dir / "diagnostic_rows.csv", rows)
    aggregate = _aggregate(rows)
    _write_csv(output_dir / "diagnostic_aggregate.csv", aggregate)
    matched = True
    for horizon in {int(row["max_steps"]) for row in rows}:
        keys = {
            (str(row["window_id"]), str(row["workflow_id"]))
            for row in rows
            if int(row["max_steps"]) == horizon
        }
        for key in keys:
            fingerprints = {
                str(row["workload_request_fingerprint"])
                for row in rows
                if int(row["max_steps"]) == horizon
                and (str(row["window_id"]), str(row["workflow_id"])) == key
            }
            matched = matched and len(fingerprints) == 1
    receipt = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "completed",
        "evidence_scope": "observed_data_completion_feasibility_not_holdout",
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "config_path": str(args.config.resolve()),
        "config_sha256": _file_sha256(args.config.resolve()),
        "source_provenance": {
            "mobility": {
                "path": str(mobility_path),
                "size_bytes": mobility_path.stat().st_size,
                "sha256": str(expected["mobility"]["sha256"]),
            },
            "workflow": {
                "path": str(workflow_path),
                "size_bytes": workflow_path.stat().st_size,
                "sha256": str(expected["workflow"]["sha256"]),
            },
        },
        "window_plan_path": str(window_plan_path),
        "window_plan_sha256": _file_sha256(window_plan_path),
        "selected_window_ids": [str(window["window_id"]) for window in windows],
        "workflow_ids": [str(workflow.workflow_id) for workflow in workflows],
        "controller_role": "fixed_rule_feasibility_reference_not_algorithm_baseline",
        "row_count": len(rows),
        "aggregate": aggregate,
        "matched_request_exposure_within_horizon": matched,
        "max_planned_dag_nodes": max(len(workflow.nodes) for workflow in workflows),
        "mobility_window_length": min(int(window["window_length"]) for window in windows),
        "outputs": {
            "rows": str(output_dir / "diagnostic_rows.csv"),
            "aggregate": str(output_dir / "diagnostic_aggregate.csv"),
            "episodes": episode_paths,
        },
        "wall_clock_sec": round(time.time() - started, 6),
        "claim_boundary": (
            "Diagnostic only: distinguishes right censoring from true request failure. "
            "It is not an algorithm comparison, holdout, or superiority result."
        ),
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
