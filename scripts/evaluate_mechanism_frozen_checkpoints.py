#!/usr/bin/env python3
"""Run one fixed observed-data evaluation of frozen mechanism checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from torch import load as torch_load


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_VERSION = "mechanism_frozen_checkpoint_evaluation_v1"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from src.evaluators.real_eval_support import build_inference_agent


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


def evaluation_unit_id(window_id: str, workflow_id: str) -> str:
    """Return the controller-neutral identity for one matched evaluation unit."""

    return f"{EVALUATION_VERSION}/{window_id}/{workflow_id}"


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


def _aggregate(setting_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    delay_values = [
        float(row["end_to_end_workflow_delay"])
        for row in rows
        if row["end_to_end_workflow_delay"] is not None
    ]
    total_steps = sum(int(row["total_steps"]) for row in rows)
    guard_count = sum(int(row["cache_warm_start_guard_count"]) for row in rows)
    action_delta_count = sum(int(row["guard_action_delta_count"]) for row in rows)
    final_action_counts: dict[str, int] = {}
    for row in rows:
        for action_id, count in dict(row["final_action_counts"]).items():
            final_action_counts[str(action_id)] = (
                final_action_counts.get(str(action_id), 0) + int(count)
            )
    return {
        "setting_id": setting_id,
        "controller": rows[0]["controller"],
        "controller_role": rows[0]["controller_role"],
        "guard_mode": rows[0]["guard_mode"],
        "evaluation_unit_count": len(rows),
        "total_steps": total_steps,
        "cache_warm_start_guard_count": guard_count,
        "cache_warm_start_guard_rate": round(guard_count / total_steps, 6),
        "guard_action_delta_count": action_delta_count,
        "effective_policy_action_share": round(action_delta_count / total_steps, 6),
        "final_action_counts": final_action_counts,
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


def _normalise_settings(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if "settings" in plan:
        settings = [dict(item) for item in plan["settings"]]
    else:
        roles = dict(plan["controller_roles"])
        settings = [
            {
                "setting_id": str(controller),
                "agent_name": str(controller),
                "controller_role": roles[str(controller)],
                "guard_mode": "not_applicable",
                "agent_config_overrides": {},
            }
            for controller in plan["controllers"]
        ]
    setting_ids = [str(item["setting_id"]) for item in settings]
    if len(setting_ids) != len(set(setting_ids)):
        raise ValueError("evaluation setting_id values must be unique")
    for item in settings:
        item.setdefault("agent_config_overrides", {})
        item.setdefault("guard_mode", "not_applicable")
        if item["guard_mode"] not in {
            "enabled_current_only",
            "disabled_at_inference",
            "not_applicable",
        }:
            raise ValueError(f"invalid guard_mode: {item['guard_mode']}")
    return settings


def _verify_learned_setting(
    *, setting: dict[str, Any], checkpoint_path: str, seed: int
) -> dict[str, Any]:
    payload = torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = dict(payload.get("config", {}))
    checkpoint_guard = {
        "cache_warm_start_guard_enabled": bool(
            checkpoint_config.get("cache_warm_start_guard_enabled", False)
        ),
        "cache_warm_start_guard_current_only": bool(
            checkpoint_config.get("cache_warm_start_guard_current_only", False)
        ),
    }
    if checkpoint_guard != {
        "cache_warm_start_guard_enabled": True,
        "cache_warm_start_guard_current_only": True,
    }:
        raise ValueError(
            f"v3 checkpoint lacks the frozen current-only guard: {checkpoint_path}"
        )
    overrides = dict(setting["agent_config_overrides"])
    agent = build_inference_agent(
        agent_name=str(setting["agent_name"]),
        random_seed=seed,
        checkpoint_path=checkpoint_path,
        deterministic_action=True,
        agent_config_overrides=overrides,
    )
    effective_guard = {
        "cache_warm_start_guard_enabled": bool(
            getattr(agent, "_cache_warm_start_guard_enabled", False)
        ),
        "cache_warm_start_guard_current_only": bool(
            getattr(agent, "_cache_warm_start_guard_current_only", False)
        ),
    }
    expected = (
        {
            "cache_warm_start_guard_enabled": True,
            "cache_warm_start_guard_current_only": True,
        }
        if setting["guard_mode"] == "enabled_current_only"
        else {
            "cache_warm_start_guard_enabled": False,
            "cache_warm_start_guard_current_only": False,
        }
    )
    if effective_guard != expected:
        raise ValueError(
            f"effective guard mismatch for {setting['setting_id']}: "
            f"expected={expected}, observed={effective_guard}"
        )
    return {
        "setting_id": setting["setting_id"],
        "agent_name": setting["agent_name"],
        "checkpoint_guard": checkpoint_guard,
        "agent_config_overrides": overrides,
        "effective_guard": effective_guard,
        "status": "passed",
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
    checkpoint_sha256_before: dict[str, str] = {}
    for row in freeze["checkpoints"]:
        path = Path(str(row["checkpoint_path"])).resolve()
        if _file_sha256(path) != str(row["checkpoint_sha256"]):
            raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
        checkpoint_map[str(row["agent_name"])] = str(path)
        checkpoint_sha256_before[str(row["agent_name"])] = str(
            row["checkpoint_sha256"]
        )

    settings = _normalise_settings(plan)
    guard_verification: list[dict[str, Any]] = []
    for setting in settings:
        agent_name = str(setting["agent_name"])
        if agent_name in checkpoint_map:
            guard_verification.append(
                _verify_learned_setting(
                    setting=setting,
                    checkpoint_path=checkpoint_map[agent_name],
                    seed=int(plan["seed"]),
                )
            )
        elif setting["guard_mode"] != "not_applicable":
            raise ValueError(
                f"non-checkpoint setting cannot declare guard mode: {setting}"
            )

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
    rows: list[dict[str, Any]] = []
    exposure_by_unit: dict[tuple[str, str], str] = {}
    for setting in settings:
        setting_id = str(setting["setting_id"])
        controller = str(setting["agent_name"])
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
                unit_id = evaluation_unit_id(
                    str(window["window_id"]), str(bound.workflow_id)
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
                        "evaluation_setting_id": setting_id,
                        "guard_mode": setting["guard_mode"],
                        "agent_config_overrides": dict(
                            setting["agent_config_overrides"]
                        ),
                    },
                    agent_config_overrides=dict(setting["agent_config_overrides"]),
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
                    / setting_id
                    / str(window["window_id"])
                    / f"{bound.workflow_id}.summary.json"
                )
                _write_json(episode_path, summary)
                endpoint = dict(summary["formal_request_execution_audit"])
                summary_row = summary_to_row(summary)
                failure = classify_episode(summary)
                diagnostics = dict(summary.get("agent_action_diagnostics", {}))
                final_action_counts: dict[str, int] = {}
                for trace_row in summary.get("policy_trace_brief", []):
                    action_id = str(trace_row.get("action_id"))
                    final_action_counts[action_id] = (
                        final_action_counts.get(action_id, 0) + 1
                    )
                guard_count = int(
                    diagnostics.get("cache_warm_start_guard_count", 0) or 0
                )
                action_delta_count = int(
                    diagnostics.get("guard_action_delta_count", 0) or 0
                )
                if setting["guard_mode"] == "disabled_at_inference" and (
                    guard_count or action_delta_count
                ):
                    raise RuntimeError(
                        f"disabled guard emitted overrides for {setting_id}"
                    )
                rows.append(
                    {
                        "setting_id": setting_id,
                        "controller": controller,
                        "controller_role": setting["controller_role"],
                        "guard_mode": setting["guard_mode"],
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
                        "total_steps": int(summary["run_info"]["total_steps"]),
                        "cache_warm_start_guard_count": guard_count,
                        "guard_action_delta_count": action_delta_count,
                        "final_action_counts": final_action_counts,
                        "episode_path": str(episode_path),
                    }
                )
    aggregates = [
        _aggregate(
            str(setting["setting_id"]),
            [
                row
                for row in rows
                if row["setting_id"] == str(setting["setting_id"])
            ],
        )
        for setting in settings
    ]
    checkpoint_integrity = []
    for agent_name, checkpoint_path in checkpoint_map.items():
        after_sha256 = _file_sha256(Path(checkpoint_path))
        before_sha256 = checkpoint_sha256_before[agent_name]
        checkpoint_integrity.append(
            {
                "agent_name": agent_name,
                "checkpoint_path": checkpoint_path,
                "sha256_before": before_sha256,
                "sha256_after": after_sha256,
                "unchanged": before_sha256 == after_sha256,
            }
        )
        if before_sha256 != after_sha256:
            raise RuntimeError(f"checkpoint changed during evaluation: {agent_name}")
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
        "evaluation_unit_count_per_setting": len(windows) * len(workflows),
        "setting_count": len(settings),
        "guard_verification": guard_verification,
        "checkpoint_integrity": checkpoint_integrity,
        "aggregate": aggregates,
        "metrics_are_a_vector_not_a_weighted_score": True,
        "claim_boundary": plan["claim_boundary"],
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
