#!/usr/bin/env python3
"""Run the fixed 48-episode CRDCM decision-contract diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mechanism_failure_causes import classify_episode
from scripts.evaluate_mechanism_frozen_checkpoints import _load_workflows
from scripts.run_mechanism_factorial_pilot import build_factorial_catalog
from src.agents.registry import build_agent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    load_window_bundle,
    run_real_episode,
    summary_to_row,
)


STUDY_VERSION = "crdcm_decision_diagnostic_v1"
ADAPTERS = (
    "adapter_perception",
    "adapter_tracking",
    "adapter_fusion",
    "adapter_intent",
    "adapter_control",
    "adapter_batch_type_1",
)


def _sha256(path: Path) -> str:
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bind_workflow(workflow: Any, catalog: AdapterCatalog, mode: str) -> Any:
    bound = deepcopy(workflow)
    if mode == "repeated_reuse":
        sequence = ("adapter_tracking", "adapter_perception")
    elif mode == "round_robin_low_reuse":
        sequence = ADAPTERS
    else:
        raise ValueError(f"unsupported workflow binding: {mode}")
    for index, node in enumerate(bound.nodes):
        adapter_id = sequence[index % len(sequence)]
        adapter = catalog.get_typed_adapter(adapter_id)
        node.required_adapter = adapter_id
        node.required_base_model = str(adapter.required_base_model_id)
    return bound


def _metric_row(
    *,
    summary: dict[str, Any],
    controller: str,
    scenario_id: str,
    unit_id: str,
) -> dict[str, Any]:
    base = summary_to_row(summary)
    failure = classify_episode(summary)
    trace = list(summary.get("policy_decision_trace_v2", []) or [])
    override_count = sum(bool(item.get("override_applied", False)) for item in trace)
    actions = [int(item["actual_executed_action"]) for item in trace]
    endpoint = dict(summary.get("formal_request_execution_audit", {}) or {})
    return {
        "controller": controller,
        "scenario_id": scenario_id,
        "unit_id": unit_id,
        "workflow_completed": bool(
            endpoint.get(
                "workflow_completed_under_exogenous_execution",
                base.get("successful_episode", False),
            )
        ),
        "workflow_continuity_rate": float(base.get("workflow_continuity_rate", 0.0) or 0.0),
        "handoff_failure_rate": float(base.get("handoff_failure_rate", 0.0) or 0.0),
        "failure_category": str(failure.get("primary_failure_category", "unknown")),
        "request_failure_count": int(failure.get("request_failure_count", 0) or 0),
        "transfer_mb_per_request": float(base.get("transfer_mb_per_request", 0.0) or 0.0),
        "migration_cost": float(base.get("adapter_state_migration_overhead", 0.0) or 0.0),
        "end_to_end_workflow_delay": base.get("end_to_end_workflow_delay"),
        "delay_available": base.get("end_to_end_workflow_delay") is not None,
        "total_steps": int(base.get("episode_step_count", len(trace)) or len(trace)),
        "override_count": override_count,
        "override_rate": round(override_count / max(len(trace), 1), 6),
        "action_sequence": "|".join(str(item) for item in actions),
        "action_count": len(actions),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["controller"], row["scenario_id"]) for row in rows})
    for controller, scenario_id in keys:
        group = [
            row
            for row in rows
            if row["controller"] == controller and row["scenario_id"] == scenario_id
        ]
        delays = [float(row["end_to_end_workflow_delay"]) for row in group if row["delay_available"]]
        output.append(
            {
                "controller": controller,
                "scenario_id": scenario_id,
                "episode_count": len(group),
                "completed_count": sum(bool(row["workflow_completed"]) for row in group),
                "completion_rate": round(fmean(float(bool(row["workflow_completed"])) for row in group), 6),
                "continuity_rate_mean": round(fmean(float(row["workflow_continuity_rate"]) for row in group), 6),
                "handoff_failure_rate_mean": round(fmean(float(row["handoff_failure_rate"]) for row in group), 6),
                "request_failure_count": sum(int(row["request_failure_count"]) for row in group),
                "failure_categories": "|".join(
                    f"{category}:{sum(item['failure_category'] == category for item in group)}"
                    for category in sorted({str(item["failure_category"]) for item in group})
                ),
                "transfer_mb_per_request_mean": round(fmean(float(row["transfer_mb_per_request"]) for row in group), 6),
                "migration_cost_mean": round(fmean(float(row["migration_cost"]) for row in group), 6),
                "delay_available_count": len(delays),
                "delay_coverage_rate": round(len(delays) / len(group), 6),
                "conditional_delay_mean": round(fmean(delays), 6) if delays else None,
                "override_rate": round(
                    sum(int(row["override_count"]) for row in group)
                    / max(sum(int(row["action_count"]) for row in group), 1),
                    6,
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/crdcm_decision_diagnostic_v1.yaml",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("study_version") != STUDY_VERSION:
        raise ValueError("unexpected CRDCM diagnostic version")
    controllers = [str(item) for item in config["controllers"]]
    scenarios = [dict(item) for item in config["scenarios"]]
    pairs = [dict(item) for item in config["data"]["evaluation_pairs"]]
    episode_count = len(controllers) * len(scenarios) * len(pairs)
    max_steps = int(config["max_steps"])
    if episode_count > int(config["episode_cap"]):
        raise ValueError("episode cap exceeded")
    if episode_count * max_steps > int(config["environment_step_cap"]):
        raise ValueError("environment-step cap exceeded")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    output.mkdir(parents=True)
    started = time.time()

    data_root = args.data_root.resolve()
    data = dict(config["data"])
    mobility_path = (data_root / str(data["mobility_csv_path"])).resolve()
    workflow_path = (data_root / str(data["workflow_csv_path"])).resolve()
    for role, path in (("mobility", mobility_path), ("workflow", workflow_path)):
        expected = dict(config["expected_sources"][role])
        if path.stat().st_size != int(expected["size_bytes"]) or _sha256(path) != expected["sha256"]:
            raise ValueError(f"{role} source identity mismatch")

    window_plan = json.loads(
        (ROOT / str(data["window_plan_path"])).read_text(encoding="utf-8")
    )
    windows = list(window_plan["selected_window_plan"])
    workflow_ids = [str(item["workflow_id"]) for item in pairs]
    workflows = {
        str(item.workflow_id): item for item in _load_workflows(workflow_path, workflow_ids)
    }
    source_catalog = AdapterCatalog.from_json(
        ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
    )
    bundles: dict[int, Any] = {}
    for pair in pairs:
        index = int(pair["window_index"])
        window = dict(windows[index])
        bundle = load_window_bundle(
            root_dir=ROOT,
            mobility_source="ngsim",
            mobility_csv_path=str(mobility_path),
            lust_scenario_root="",
            max_mobility_rows=int(data["max_mobility_rows"]),
            rsu_layout=str(window["recommended_rsu_layout"]),
            frame_offset=int(window["frame_offset"]),
            window_length=int(window["window_length"]),
            random_seed=int(config["seed"]),
        )
        bundle.rsu_metadata.update(
            window_id=window["window_id"],
            window_rank=window["window_rank"],
            window_class=window["window_class"],
        )
        bundles[index] = bundle

    rows: list[dict[str, Any]] = []
    action_by_cell: dict[tuple[str, str, int], dict[str, int]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        catalog = build_factorial_catalog(
            source_catalog,
            base_sharing_enabled=True,
            initial_adapter_id=str(scenario["initial_adapter_id"]),
        )
        capacity_profile = {
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(scenario["capacity_mb"]),
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": int(config["seed"]),
            "telemetry_enabled": True,
        }
        mechanism_profile = {
            "mechanism_factorial_profile_version": "controlled_mechanism_factorial_v1",
            "profile_id": f"{scenario_id}_crdcm_observation_v1",
            "base_sharing_enabled": True,
            "workflow_state_migration_enabled": bool(scenario["migration_enabled"]),
            "migration_disabled_fallback": "cold_restart_one_request",
            "crdcm_observation_enabled": True,
            "crdcm_observation_contract_version": "crdcm_observation_v1",
        }
        runtime_contract = {
            "model_cache_profile": "typed_base_adapter_state_v1",
            "typed_catalog_fingerprint": catalog.canonical_fingerprint(),
            "runtime_contract_sha256": _canonical_sha256(
                {"capacity": capacity_profile, "mechanism": mechanism_profile}
            ),
            "cache_event_schema_version": "1.3.0",
            "cache_efficiency_metrics_contract_version": "1.1.0",
        }
        for pair_index, pair in enumerate(pairs):
            window_index = int(pair["window_index"])
            bundle = bundles[window_index]
            workflow = _bind_workflow(
                workflows[str(pair["workflow_id"])],
                catalog,
                str(scenario["workflow_binding"]),
            )
            unit_id = f"{scenario_id}/{bundle.rsu_metadata['window_id']}/{workflow.workflow_id}"
            exposure = build_episode_formal_request_exposure(
                workflow_state=workflow,
                mobility_bundle=bundle,
                adapter_catalog=catalog,
                max_steps=max_steps,
                mobility_source="ngsim",
                primary_vehicle_selection="handoff_pressure",
                cache_capacity_profile=capacity_profile,
                evaluation_unit={
                    "evaluation_unit_id": unit_id,
                    "workflow_id": workflow.workflow_id,
                    "window_id": bundle.rsu_metadata["window_id"],
                },
                source_provenance={
                    "study_version": STUDY_VERSION,
                    "data_relationship": config["data_relationship"],
                },
            )
            for controller in controllers:
                agent = build_agent(
                    controller,
                    deterministic_action=True,
                    random_seed=int(config["seed"]) + pair_index,
                ) if controller != "crdcm_critical_path_heuristic" else build_agent(controller)
                summary = run_real_episode(
                    root_dir=ROOT,
                    agent_name=controller,
                    checkpoint_map={},
                    workflow_state=workflow,
                    workflow_source_path=str(workflow_path),
                    mobility_bundle=bundle,
                    seed=int(config["seed"]) + pair_index,
                    max_steps=max_steps,
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    reward_positive_offset=0.0,
                    run_metadata={
                        "script": "scripts/run_crdcm_decision_diagnostic.py",
                        "study_version": STUDY_VERSION,
                        "scenario_id": scenario_id,
                        "unit_id": unit_id,
                        "training": False,
                        "claim_boundary": config["claim_boundary"],
                    },
                    adapter_catalog_override=catalog,
                    workflow_state_override=workflow,
                    cache_capacity_profile=capacity_profile,
                    model_cache_runtime_contract=runtime_contract,
                    formal_request_exposure_trace=exposure,
                    mechanism_profile=mechanism_profile,
                    agent_override=agent,
                )
                episode_path = output / "episodes" / controller / f"{scenario_id}_{pair_index}.json"
                _write_json(episode_path, summary)
                rows.append(
                    _metric_row(
                        summary=summary,
                        controller=controller,
                        scenario_id=scenario_id,
                        unit_id=unit_id,
                    )
                )
                for trace in summary.get("policy_decision_trace_v2", []):
                    key = (scenario_id, unit_id, int(trace["step_index"]))
                    action_by_cell.setdefault(key, {})[controller] = int(
                        trace["actual_executed_action"]
                    )

    aggregate = _aggregate(rows)
    divergence_rows = []
    for (scenario_id, unit_id, step_index), actions in sorted(action_by_cell.items()):
        divergence_rows.append(
            {
                "scenario_id": scenario_id,
                "unit_id": unit_id,
                "step_index": step_index,
                "controller_count": len(actions),
                "unique_action_count": len(set(actions.values())),
                "decision_disagreement": len(set(actions.values())) > 1,
                "actions": actions,
            }
        )
    pairwise_disagreement: list[dict[str, Any]] = []
    for first, second in combinations(controllers, 2):
        matched = [
            row
            for row in divergence_rows
            if first in row["actions"] and second in row["actions"]
        ]
        disagree = sum(row["actions"][first] != row["actions"][second] for row in matched)
        pairwise_disagreement.append(
            {
                "controller_a": first,
                "controller_b": second,
                "matched_decision_count": len(matched),
                "disagreement_count": disagree,
                "disagreement_rate": round(disagree / max(len(matched), 1), 6),
            }
        )
    receipt = {
        "study_version": STUDY_VERSION,
        "status": "completed",
        "git_commit_at_execution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "episode_count": len(rows),
        "actual_environment_steps": sum(int(row["total_steps"]) for row in rows),
        "episode_cap": int(config["episode_cap"]),
        "environment_step_cap": int(config["environment_step_cap"]),
        "decision_cell_count": len(divergence_rows),
        "decision_disagreement_count": sum(
            bool(row["decision_disagreement"]) for row in divergence_rows
        ),
        "decision_disagreement_rate": round(
            fmean(float(bool(row["decision_disagreement"])) for row in divergence_rows), 6
        ) if divergence_rows else 0.0,
        "pairwise_action_disagreement": pairwise_disagreement,
        "training_performed": False,
        "legacy_checkpoint_loaded": False,
        "holdout_accessed": False,
        "claim_boundary": config["claim_boundary"],
        "future_training_plan": config["future_training_plan"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_csv(output / "diagnostic_rows.csv", rows)
    _write_csv(output / "diagnostic_aggregate.csv", aggregate)
    _write_json(output / "decision_divergence.json", divergence_rows)
    _write_json(output / "pairwise_action_disagreement.json", pairwise_disagreement)
    _write_json(output / "completion_receipt.json", receipt)
    _write_json(output / "plan_snapshot.json", config)
    _write_json(
        output / "command_log.json",
        {
            "status": "completed",
            "argv": [sys.executable, *sys.argv],
            "exit_code": 0,
            "working_directory": str(ROOT),
            "source_identity": {
                "mobility": {"size_bytes": mobility_path.stat().st_size, "sha256": _sha256(mobility_path)},
                "workflow": {"size_bytes": workflow_path.stat().st_size, "sha256": _sha256(workflow_path)},
            },
        },
    )
    artifact_files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    _write_json(
        output / "artifact_integrity_manifest.json",
        {
            "status": "pass",
            "self_excluding": True,
            "file_count": len(artifact_files),
            "files": [
                {
                    "path": path.relative_to(output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifact_files
            ],
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
