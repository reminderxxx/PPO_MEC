"""Run bounded non-formal training/evaluation through the G14R2 window loader."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_main_results import main as benchmark_main
from scripts.train_algo_pool_real_sample import main as training_main
from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    build_manifest,
    validate_manifest,
)
from src.evaluators.formal_window_consumption import file_sha256
from src.runtime.typed_model_cache_runtime import (
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)


ARTIFACT_ROOT = ROOT / "artifacts/analysis/typed_model_cache_formal_window_repair_20260820_g14r2_v1"
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820"
V1_1_CONFIG = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
SPLIT_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_20260820"
CONTRACT = CONFIG_ROOT / "formal_window_consumption_contract.json"
TRAIN_PLAN = SPLIT_ROOT / "train_window_plan.json"
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
AGENTS = ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
ALL_AGENTS = [*BASELINE_NAMES, *AGENTS]
SEEDS = [7, 13]
CAPACITIES = {
    "constrained_288mb": V1_1_CONFIG / "runtime_constrained_288mb.yaml",
    "medium_576mb": V1_1_CONFIG / "runtime_medium_576mb.yaml",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def invoke(main_function: Callable[[], None], argv: list[str]) -> None:
    previous = list(sys.argv)
    try:
        sys.argv = argv
        main_function()
    finally:
        sys.argv = previous


def subset_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(TRAIN_PLAN.read_text(encoding="utf-8-sig"))
    windows = list(payload["selected_window_plan"])
    selected = [
        min(windows, key=lambda row: int(row["frame_offset"])),
        max(windows, key=lambda row: int(row["frame_offset"])),
    ]
    subset = {
        **{key: value for key, value in payload.items() if key != "selected_window_plan"},
        "split": "train",
        "sealed": False,
        "rehearsal_only": True,
        "selected_window_plan": selected,
    }
    write_json(path, subset)
    return selected


def build_fairness(
    *, capacity_mb: float, plan_path: Path, output_root: Path
) -> dict[str, Any]:
    manifest = build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=plan_path,
        catalog_path=ROOT / "src/data/model_catalog/typed_model_cache_controlled.json",
        seeds=SEEDS,
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=11_850_526,
        primary_vehicle_selection="handoff_pressure",
        capacity_unit="mb",
        capacity_value=capacity_mb,
        output_root=str(output_root),
        evaluation_unit_limit=None,
        created_at=now(),
        controller_agents=AGENTS,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise RuntimeError(report["errors"])
    return manifest


def main() -> None:
    run_id = datetime.now().strftime("g14r2_non_formal_rehearsal_%Y%m%d_%H%M%S_%f")
    root = ARTIFACT_ROOT / "rehearsal_runs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    plan_path = root / "non_formal_train_boundary_window_plan.json"
    selected_windows = subset_plan(plan_path)
    command_log: list[dict[str, Any]] = []
    training_rows = []
    manifests: dict[str, dict[str, Path]] = {}
    for capacity_label, runtime_path in CAPACITIES.items():
        runtime = resolve_model_cache_runtime(runtime_path, root=ROOT)
        seed_manifest = {agent: {} for agent in AGENTS}
        provenance_manifest = {agent: {} for agent in AGENTS}
        for agent in AGENTS:
            for seed in SEEDS:
                training_root = root / "training" / capacity_label
                cell_id = f"g14r2_non_formal_{capacity_label}_{agent}_seed{seed}"
                argv = [
                    "train_algo_pool_real_sample.py",
                    "--agent_name", agent,
                    "--profile", "smoke",
                    "--episodes", "4",
                    "--update_every", "1",
                    "--batch_size", "1",
                    "--max_steps", "1",
                    "--checkpoint_every_updates", "4",
                    "--max_workflows", "1",
                    "--workflow_selector", "ordered",
                    "--min_tasks", "5",
                    "--max_tasks", "20",
                    "--mobility_source", "ngsim",
                    "--mobility_csv_path", str(MOBILITY),
                    "--max_mobility_rows", "11850526",
                    "--window_plan_path", str(TRAIN_PLAN),
                    "--formal_window_consumption_contract_path", str(CONTRACT),
                    "--formal_window_split", "train",
                    "--window_consumption_mode", "rehearsal",
                    "--window_selector", "ordered",
                    "--window_length", "24",
                    "--rsu_layout", "auto_dominant_tight",
                    "--primary_vehicle_selection", "handoff_pressure",
                    "--model_cache_runtime_config", str(runtime_path),
                    "--agent_config_path", str(CONFIG_ROOT / "agent_training_configs.json"),
                    "--reward_positive_offset", "0",
                    "--random_seed", str(seed),
                    "--output_root", str(training_root),
                    "--run_id", cell_id,
                ]
                started_at = now()
                invoke(training_main, argv)
                completed_at = now()
                summary_path = training_root / agent / cell_id / "train_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary["saved_checkpoint_update_indices"] != [4]:
                    raise RuntimeError("tiny rehearsal checkpoint cadence mismatch")
                if summary["formal_training_contract"]["formal_protocol_version"] is not None:
                    raise RuntimeError("tiny rehearsal accidentally created a formal checkpoint")
                checkpoint_path = Path(summary["checkpoint_paths"][0])
                validation = validate_checkpoint_provenance(
                    checkpoint_path,
                    expected_agent_name=agent,
                    expected_seed=seed,
                    expected_runtime_contract=runtime,
                    expected_reward_positive_offset=0.0,
                    expected_window_plan_identity=summary["train_window_plan_identity"],
                )
                if validation["status"] != "compatible":
                    raise RuntimeError(validation)
                if (
                    agent == "sa_ghmappo"
                    and summary["resolved_agent_config"].get("auxiliary_coef") != 0.06
                ):
                    raise RuntimeError("SA tiny rehearsal auxiliary coefficient mismatch")
                seed_manifest[agent][str(seed)] = str(checkpoint_path)
                metadata = validation["metadata"]
                provenance_manifest[agent][str(seed)] = {
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "execution_git_commit": metadata["execution_git_commit"],
                    "train_window_plan_identity": metadata["train_window_plan_identity"],
                }
                training_rows.append(
                    {
                        "capacity_label": capacity_label,
                        "capacity_mb": runtime["cache_capacity_profile"]["capacity_mb"],
                        "agent": agent,
                        "seed": seed,
                        "status": "pass",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "saved_checkpoint_update_indices": [4],
                        "checkpoint_restore_and_provenance": "compatible",
                        "formal_checkpoint": False,
                        "window_contract_sha256": summary["train_window_plan_identity"][
                            "formal_window_consumption_contract_sha256"
                        ],
                    }
                )
                command_log.append(
                    {
                        "command": argv,
                        "status": "pass",
                        "formal": False,
                        "holdout": False,
                    }
                )
        input_root = root / "benchmark_inputs" / capacity_label
        seed_path = input_root / "seed_checkpoint_manifest.json"
        provenance_path = input_root / "checkpoint_provenance_manifest.json"
        write_json(seed_path, seed_manifest)
        write_json(provenance_path, provenance_manifest)
        manifests[capacity_label] = {
            "seed": seed_path,
            "provenance": provenance_path,
            "runtime": runtime_path,
        }

    evaluation_rows = []
    for capacity_label, inputs in manifests.items():
        runtime = resolve_model_cache_runtime(inputs["runtime"], root=ROOT)
        benchmark_parent = root / "evaluation" / capacity_label
        fairness_path = root / "fairness" / f"{capacity_label}.json"
        write_json(
            fairness_path,
            build_fairness(
                capacity_mb=float(runtime["cache_capacity_profile"]["capacity_mb"]),
                plan_path=plan_path,
                output_root=benchmark_parent,
            ),
        )
        argv = [
            "benchmark_main_results.py",
            "--agents", *ALL_AGENTS,
            "--seeds", *[str(seed) for seed in SEEDS],
            "--seed_checkpoint_manifest_path", str(inputs["seed"]),
            "--checkpoint_provenance_manifest_path", str(inputs["provenance"]),
            "--cache_baseline_fairness_manifest_path", str(fairness_path),
            "--model_cache_runtime_config", str(inputs["runtime"]),
            "--mobility_source", "ngsim",
            "--mobility_csv_path", str(MOBILITY),
            "--max_mobility_rows", "11850526",
            "--workflow_csv_path", str(WORKFLOW),
            "--max_workflows", "1",
            "--max_steps", "1",
            "--workflow_selector", "ordered",
            "--min_tasks", "5",
            "--max_tasks", "20",
            "--window_plan_path", str(plan_path),
            "--formal_window_consumption_contract_path", str(CONTRACT),
            "--formal_window_split", "train",
            "--window_consumption_mode", "rehearsal",
            "--window_selector", "ordered",
            "--window_length", "24",
            "--rsu_layout", "auto_dominant_tight",
            "--primary_vehicle_selection", "handoff_pressure",
            "--window_mode", "mixed_informative",
            "--reward_positive_offset", "0",
            "--output_root", str(benchmark_parent),
        ]
        before = set(benchmark_parent.iterdir()) if benchmark_parent.exists() else set()
        invoke(benchmark_main, argv)
        created = sorted(set(benchmark_parent.iterdir()) - before)
        if len(created) != 1:
            raise RuntimeError("tiny evaluation did not create exactly one run")
        aggregate_path = created[0] / "aggregate_summary.json"
        rows_path = created[0] / "benchmark_rows.csv"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        with rows_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_rows = len(ALL_AGENTS) * len(SEEDS) * len(selected_windows)
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"tiny evaluation row count mismatch: {len(rows)} != {expected_rows}"
            )
        binding = aggregate.get("formal_window_consumption_binding") or {}
        if binding.get("mode") != "rehearsal" or binding.get("split") != "train":
            raise RuntimeError("tiny evaluation window binding missing")
        evaluation_rows.append(
            {
                "capacity_label": capacity_label,
                "status": "pass",
                "non_formal_identity": True,
                "window_count": len(selected_windows),
                "seed_count": len(SEEDS),
                "agent_count": len(ALL_AGENTS),
                "benchmark_row_count": len(rows),
                "summary_path": str(aggregate_path),
                "summary_sha256": file_sha256(aggregate_path),
                "rows_path": str(rows_path),
                "rows_sha256": file_sha256(rows_path),
                "metrics_present": bool(aggregate.get("aggregate_by_agent")),
                "window_consumption_binding": binding,
                "formal": False,
                "holdout": False,
            }
        )
        command_log.append(
            {"command": argv, "status": "pass", "formal": False, "holdout": False}
        )

    summary = {
        "rehearsal_version": "2.0.0",
        "rehearsal_id": run_id,
        "status": "pass",
        "scope": "bounded non-formal frozen-train-window consumption rehearsal",
        "agents": AGENTS,
        "domain_baseline": "cache_offload_drl",
        "seeds": SEEDS,
        "capacities_mb": [288.0, 576.0],
        "training_cell_count": len(training_rows),
        "training_cells": training_rows,
        "checkpoint_frequency_updates": 4,
        "saved_checkpoint_update_indices": [4],
        "checkpoint_restore_and_provenance": "pass",
        "tiny_evaluation": evaluation_rows,
        "boundary_window_ids": [row["window_id"] for row in selected_windows],
        "window_consumption_contract_path": str(CONTRACT),
        "window_consumption_contract_sha256": file_sha256(CONTRACT),
        "formal_checkpoint_count": 0,
        "formal_episode_count": 0,
        "formal_performance_result_count": 0,
        "holdout_opened": False,
        "holdout_episode_count": 0,
        "performance_claims": [],
        "rehearsal_root": str(root),
    }
    write_json(root / "command_log.json", {"commands": command_log})
    write_json(root / "rehearsal_summary.json", summary)
    write_json(ARTIFACT_ROOT / "tiny_training_rehearsal.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
