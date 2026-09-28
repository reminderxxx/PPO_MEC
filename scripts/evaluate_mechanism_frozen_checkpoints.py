#!/usr/bin/env python3
"""Run one fixed observed-data evaluation of frozen mechanism checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_VERSION = "mechanism_frozen_checkpoint_evaluation_v1"

from scripts.analyze_mechanism_failure_causes import classify_episode
from scripts.run_mechanism_completion_diagnostic import _load_workflows
from scripts.run_mechanism_factorial_pilot import (
    bind_workflow_to_catalog,
    build_factorial_catalog,
)
from src.agents.handoff_first_feasibility_agent import HandoffFirstFeasibilityAgent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    load_window_bundle,
    run_real_episode,
    summary_to_row,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(controller: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    delay_values = [
        float(row["end_to_end_workflow_delay"])
        for row in rows
        if row["end_to_end_workflow_delay"] is not None
    ]
    return {
        "controller": controller,
        "controller_role": rows[0]["controller_role"],
        "evaluation_unit_count": len(rows),
        "completed_count": sum(bool(row["workflow_completed"]) for row in rows),
        "completed_rate": round(
            fmean(float(bool(row["workflow_completed"])) for row in rows), 6
        ),
        "workflow_continuity_rate_mean": round(
            fmean(float(row["workflow_continuity_rate"]) for row in rows), 6
        ),
        "handoff_failure_rate_mean": round(
            fmean(float(row["handoff_failure_rate"]) for row in rows), 6
        ),
        "full_service_ready_request_rate_mean": round(
            fmean(float(row["full_service_ready_request_rate"]) for row in rows), 6
        ),
        "transfer_mb_per_request_mean": round(
            fmean(float(row["transfer_mb_per_request"]) for row in rows), 6
        ),
        "backhaul_traffic_cost_mean": round(
            fmean(float(row["backhaul_traffic_cost"]) for row in rows), 6
        ),
        "adapter_state_migration_overhead_mean": round(
            fmean(float(row["adapter_state_migration_overhead"]) for row in rows),
            6,
        ),
        "request_failure_count": sum(int(row["request_failure_count"]) for row in rows),
        "right_censored_count": sum(bool(row["right_censored"]) for row in rows),
        "delay_available_count": len(delay_values),
        "delay_coverage_rate": round(len(delay_values) / len(rows), 6),
        "conditional_end_to_end_workflow_delay_mean": (
            round(fmean(delay_values), 6) if delay_values else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = args.plan.resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("evaluation plan must be a mapping")
    if plan.get("execution_policy") != "single_execution_no_checkpoint_selection":
        raise ValueError("evaluation plan lacks single-execution policy")
    freeze_path = (ROOT_DIR / str(plan["freeze_manifest_path"])).resolve()
    if _file_sha256(freeze_path) != str(plan["freeze_manifest_sha256"]):
        raise ValueError("freeze manifest SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint_map: dict[str, str] = {}
    for row in freeze["checkpoints"]:
        path = Path(str(row["checkpoint_path"])).resolve()
        if _file_sha256(path) != str(row["checkpoint_sha256"]):
            raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
        checkpoint_map[str(row["agent_name"])] = str(path)

    data_root = args.data_root.resolve()
    data = dict(plan["data"])
    mobility_path = (data_root / str(data["mobility_csv_path"])).resolve()
    workflow_path = (data_root / str(data["workflow_csv_path"])).resolve()
    expected = dict(plan["expected_sources"])
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
        "eviction_policy_seed": int(plan["seed"]),
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
    controllers = [str(item) for item in plan["controllers"]]
    roles = dict(plan["controller_roles"])
    rows: list[dict[str, Any]] = []
    exposure_by_unit: dict[tuple[str, str], str] = {}
    for controller in controllers:
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
                random_seed=int(plan["seed"]),
            )
            bundle.rsu_metadata["window_rank"] = window["window_rank"]
            bundle.rsu_metadata["window_class"] = window["window_class"]
            for workflow in workflows:
                bound = bind_workflow_to_catalog(workflow, catalog)
                unit_key = (str(window["window_id"]), str(bound.workflow_id))
                unit_id = (
                    f"{EVALUATION_VERSION}/{window['window_id']}/"
                    f"{bound.workflow_id}/{controller}"
                )
                exposure = build_episode_formal_request_exposure(
                    workflow_state=bound,
                    mobility_bundle=bundle,
                    adapter_catalog=catalog,
                    max_steps=int(plan["max_steps"]),
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    cache_capacity_profile=capacity_profile,
                    evaluation_unit={
                        "evaluation_unit_id": unit_id,
                        "workflow_id": bound.workflow_id,
                        "window_id": window["window_id"],
                    },
                    source_provenance={
                        "evaluation_version": EVALUATION_VERSION,
                        "data_relationship": plan["data_relationship"],
                    },
                )
                fingerprint = str(exposure["request_exposure_fingerprint"])
                previous = exposure_by_unit.setdefault(unit_key, fingerprint)
                if previous != fingerprint:
                    raise RuntimeError("request exposure differs across controllers")
                agent_override = (
                    HandoffFirstFeasibilityAgent()
                    if controller == "handoff_first_feasibility"
                    else None
                )
                summary = run_real_episode(
                    root_dir=ROOT_DIR,
                    agent_name=controller,
                    checkpoint_map=checkpoint_map,
                    workflow_state=bound,
                    workflow_source_path=str(workflow_path),
                    mobility_bundle=bundle,
                    seed=int(plan["seed"]),
                    max_steps=int(plan["max_steps"]),
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    reward_positive_offset=0.0,
                    run_metadata={
                        "script": "scripts/evaluate_mechanism_frozen_checkpoints.py",
                        "evaluation_version": EVALUATION_VERSION,
                        "data_relationship": plan["data_relationship"],
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
                    agent_override=agent_override,
                )
                episode_path = (
                    output_dir
                    / "episodes"
                    / controller
                    / str(window["window_id"])
                    / f"{bound.workflow_id}.summary.json"
                )
                _write_json(episode_path, summary)
                endpoint = dict(summary["formal_request_execution_audit"])
                summary_row = summary_to_row(summary)
                failure = classify_episode(summary)
                rows.append(
                    {
                        "controller": controller,
                        "controller_role": roles[controller],
                        "window_id": window["window_id"],
                        "workflow_id": bound.workflow_id,
                        "request_exposure_fingerprint": fingerprint,
                        "workflow_completed": bool(
                            endpoint["workflow_completed_under_exogenous_execution"]
                        ),
                        "right_censored": bool(endpoint["right_censored"]),
                        "workflow_continuity_rate": float(
                            endpoint["workflow_continuity_rate"]
                        ),
                        "full_service_ready_request_rate": float(
                            endpoint["full_service_ready_request_rate"]
                        ),
                        "handoff_failure_rate": float(
                            summary_row.get("handoff_failure_rate") or 0.0
                        ),
                        "transfer_mb_per_request": float(
                            endpoint["transfer_mb_per_request"]
                        ),
                        "backhaul_traffic_cost": float(
                            summary_row.get("backhaul_traffic_cost") or 0.0
                        ),
                        "adapter_state_migration_overhead": float(
                            summary_row.get("adapter_state_migration_overhead") or 0.0
                        ),
                        "end_to_end_workflow_delay": endpoint.get(
                            "end_to_end_workflow_delay"
                        ),
                        "request_failure_count": failure["request_failure_count"],
                        "primary_failure_category": failure[
                            "primary_failure_category"
                        ],
                        "episode_path": str(episode_path),
                    }
                )
    aggregates = [
        _aggregate(
            controller,
            [row for row in rows if row["controller"] == controller],
        )
        for controller in controllers
    ]
    _write_csv(output_dir / "evaluation_rows.csv", rows)
    _write_csv(output_dir / "evaluation_aggregate.csv", aggregates)
    receipt = {
        "evaluation_version": EVALUATION_VERSION,
        "status": "completed",
        "execution_count": 1,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "plan_path": str(plan_path),
        "plan_sha256": _file_sha256(plan_path),
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_sha256": _file_sha256(freeze_path),
        "data_relationship": plan["data_relationship"],
        "matched_request_exposure_across_controllers": True,
        "evaluation_unit_count_per_controller": len(windows) * len(workflows),
        "aggregate": aggregates,
        "metrics_are_a_vector_not_a_weighted_score": True,
        "claim_boundary": plan["claim_boundary"],
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
