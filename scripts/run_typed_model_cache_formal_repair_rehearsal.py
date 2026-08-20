"""Run the bounded non-formal G14R checkpoint, endpoint, support, and phase rehearsal."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import BASELINE_NAMES, build_manifest, validate_manifest
from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    CommandResult,
    PHASE_ORDER,
    reconcile_primary_endpoint_row,
    support_setting_by_id,
)
from src.metrics.cache_efficiency_metrics import cache_efficiency_row_fields
from src.oracles.cache_request_replay import build_policy_neutral_replay_from_manifest
from src.runtime.typed_model_cache_runtime import (
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)


ARTIFACT_ROOT = ROOT / "artifacts/analysis/typed_model_cache_formal_protocol_restart_20260820_g14r_v1"
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
PLAN = ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
AGENTS = ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
ALL_AGENTS = [*BASELINE_NAMES, *AGENTS]
SEEDS = [7, 13]
CAPACITIES = {
    "constrained_288mb": CONFIG_ROOT / "runtime_constrained_288mb.yaml",
    "medium_576mb": CONFIG_ROOT / "runtime_medium_576mb.yaml",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], log: list[dict[str, Any]]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/ppo_mec_g14r_rehearsal_pycache"
    started = now()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log.append(
        {
            "started_at": started,
            "completed_at": now(),
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "formal_execution": False,
            "holdout_execution": False,
        }
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def fairness(
    capacity_mb: float,
    output_root: Path,
    *,
    prediction_noise_std: float = 0.0,
    prediction_confidence_scale: float = 1.0,
    drop_handoff_prediction_prob: float = 0.0,
) -> dict[str, Any]:
    manifest = build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=PLAN,
        catalog_path=CATALOG,
        seeds=SEEDS,
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=2500,
        primary_vehicle_selection="stable_first",
        capacity_unit="mb",
        capacity_value=capacity_mb,
        output_root=str(output_root),
        evaluation_unit_limit=1,
        created_at=now(),
        controller_agents=AGENTS,
        prediction_noise_std=prediction_noise_std,
        prediction_confidence_scale=prediction_confidence_scale,
        drop_handoff_prediction_prob=drop_handoff_prediction_prob,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise RuntimeError(report["errors"])
    return manifest


def train_cells(
    root: Path, command_log: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    for capacity_label, runtime_path in CAPACITIES.items():
        runtime = resolve_model_cache_runtime(runtime_path, root=ROOT)
        seed_manifest = {agent: {} for agent in AGENTS}
        provenance_manifest = {agent: {} for agent in AGENTS}
        for agent in AGENTS:
            for seed in SEEDS:
                training_root = root / "training" / capacity_label
                run_id = f"rehearsal_{capacity_label}_{agent}_seed{seed}"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                    "--agent_name", agent,
                    "--profile", "smoke",
                    "--episodes", "4",
                    "--update_every", "1",
                    "--batch_size", "1",
                    "--max_steps", "1",
                    "--max_workflows", "1",
                    "--max_mobility_rows", "500",
                    "--window_plan_path", str(PLAN),
                    "--model_cache_runtime_config", str(runtime_path),
                    "--agent_config_path", str(CONFIG_ROOT / "agent_training_configs.json"),
                    "--checkpoint_every_updates", "4",
                    "--reward_positive_offset", "0",
                    "--random_seed", str(seed),
                    "--output_root", str(training_root),
                    "--run_id", run_id,
                ]
                run(command, command_log)
                summary_path = training_root / agent / run_id / "train_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary["saved_checkpoint_update_indices"] != [4]:
                    raise RuntimeError("rehearsal checkpoint cadence did not save only update 4")
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
                if agent == "sa_ghmappo" and summary["resolved_agent_config"].get("auxiliary_coef") != 0.06:
                    raise RuntimeError("SA rehearsal auxiliary coefficient is not 0.06")
                seed_manifest[agent][str(seed)] = str(checkpoint_path)
                metadata = validation["metadata"]
                provenance_manifest[agent][str(seed)] = {
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "execution_git_commit": metadata["execution_git_commit"],
                    "train_window_plan_identity": metadata["train_window_plan_identity"],
                }
                reports.append(
                    {
                        "capacity_label": capacity_label,
                        "agent": agent,
                        "seed": seed,
                        "saved_update_indices": [4],
                        "restore_and_provenance": "compatible",
                        "resolved_auxiliary_coef": summary["resolved_agent_config"].get("auxiliary_coef"),
                        "formal_checkpoint": False,
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
    return manifests, reports


def benchmark(
    root: Path,
    capacity_label: str,
    inputs: dict[str, Path],
    fairness_path: Path,
    output_name: str,
    command_log: list[dict[str, Any]],
) -> Path:
    output_root = root / output_name
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *ALL_AGENTS,
        "--seeds", *[str(seed) for seed in SEEDS],
        "--seed_checkpoint_manifest_path", str(inputs["seed"]),
        "--checkpoint_provenance_manifest_path", str(inputs["provenance"]),
        "--cache_baseline_fairness_manifest_path", str(fairness_path),
        "--model_cache_runtime_config", str(inputs["runtime"]),
        "--window_plan_path", str(PLAN),
        "--max_workflows", "1",
        "--max_steps", "1",
        "--max_mobility_rows", "2500",
        "--min_tasks", "5",
        "--max_tasks", "20",
        "--workflow_selector", "ordered",
        "--primary_vehicle_selection", "stable_first",
        "--window_mode", "mixed_informative",
        "--reward_positive_offset", "0",
        "--output_root", str(output_root),
    ]
    run(command, command_log)
    created = sorted(path for path in output_root.iterdir() if path.is_dir())
    if len(created) != 1:
        raise RuntimeError(f"unexpected rehearsal benchmark run count: {capacity_label}")
    return created[0]


def reconcile(benchmark_dirs: list[Path]) -> dict[str, Any]:
    count = 0
    for benchmark_dir in benchmark_dirs:
        with (benchmark_dir / "benchmark_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row_map = {
            (row["window_id"], row["workflow_id"], row["agent_name"], int(row["seed"])): row
            for row in rows
        }
        for summary_path in (benchmark_dir / "episodes").rglob("*.summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            agent = summary_path.parent.name
            workflow_id = summary_path.parent.parent.name
            window_id = summary_path.parent.parent.parent.name
            seed = int(summary_path.stem.split("_")[-1].split(".")[0])
            csv_row = row_map[(window_id, workflow_id, agent, seed)]
            row = dict(csv_row)
            for field in (
                "full_service_ready_byte_hit_rate",
                "joint_base_adapter_hit_rate",
                "full_service_ready_request_rate",
                "transfer_mb_per_request",
                "workflow_continuity_rate",
                "end_to_end_workflow_delay",
            ):
                row[field] = None if csv_row.get(field, "") == "" else float(csv_row[field])
            reconcile_primary_endpoint_row(summary, row)
            cache_efficiency_row_fields(summary)
            count += 1
    return {"status": "pass", "episode_summary_count": count, "primary_endpoint_count": 6}


def run_support(
    root: Path,
    protocol_path: Path,
    protocol: dict,
    inputs: dict[str, Path],
    command_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = [
        ("ablation", "typed_semantics", "no_prediction", {"prediction_confidence_scale": 0.0, "drop_handoff_prediction_prob": 1.0}),
        ("robustness", "prediction_condition", "noise_0.2", {"prediction_noise_std": 0.2}),
    ]
    reports: list[dict[str, Any]] = []
    for family, parameter, value, fairness_kwargs in cases:
        setting = next(
            level
            for item in protocol["ablation_and_support"]["support_setting_matrix"]["settings"]
            if item["parameter"] == parameter
            for level in item["levels"]
            if level["value"] == value
        )
        manifest = fairness(576.0, root / "support", **fairness_kwargs)
        fairness_path = root / "support_inputs" / f"{family}_fairness.json"
        write_json(fairness_path, manifest)
        output_root = root / "support" / family
        command = [
            sys.executable,
            str(ROOT / "scripts/run_typed_model_cache_formal_support.py"),
            "--protocol-path", str(protocol_path),
            "--setting-id", setting["setting_id"],
            "--model-cache-runtime-config", str(inputs["runtime"]),
            "--cache-baseline-fairness-manifest-path", str(fairness_path),
            "--seed-checkpoint-manifest-path", str(inputs["seed"]),
            "--checkpoint-provenance-manifest-path", str(inputs["provenance"]),
            "--window-plan-path", str(PLAN),
            "--agents", *ALL_AGENTS,
            "--seeds", *[str(seed) for seed in SEEDS],
            "--output-root", str(output_root),
        ]
        run(command, command_log)
        reports.append({"family": family, "setting_id": setting["setting_id"], "status": "pass", "formal": False})

    scalability = next(
        level
        for item in protocol["ablation_and_support"]["scalability_setting_matrix"]["settings"]
        if item["parameter"] == "oracle_state_limit"
        for level in item["levels"]
        if level["value"] == 1000
    )
    manifest = fairness(576.0, root / "support")
    fairness_path = root / "support_inputs/scalability_fairness.json"
    write_json(fairness_path, manifest)
    unit_id = manifest["window_workload_plan"]["evaluation_units"][0]["evaluation_unit_id"]
    replay = build_policy_neutral_replay_from_manifest(root=ROOT, manifest=manifest, evaluation_unit_id=unit_id)
    replay_path = root / "support_inputs/request_replay.json"
    write_json(replay_path, replay)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_typed_model_cache_formal_support.py"),
        "--protocol-path", str(protocol_path),
        "--setting-id", scalability["setting_id"],
        "--model-cache-runtime-config", str(inputs["runtime"]),
        "--cache-baseline-fairness-manifest-path", str(fairness_path),
        "--seed-checkpoint-manifest-path", str(inputs["seed"]),
        "--checkpoint-provenance-manifest-path", str(inputs["provenance"]),
        "--window-plan-path", str(PLAN),
        "--agents", *ALL_AGENTS,
        "--seeds", *[str(seed) for seed in SEEDS],
        "--request-replay-path", str(replay_path),
        "--output-root", str(root / "support/scalability"),
    ]
    run(command, command_log)
    reports.append({"family": "scalability", "setting_id": scalability["setting_id"], "status": "pass", "formal": False})
    return reports


def phase_simulation(root: Path, protocol: dict) -> dict[str, Any]:
    runner = AppendOnlyPhaseRunner(protocol=protocol, output_root=root / "phase_simulation")
    for phase in PHASE_ORDER:
        output = f"markers/{phase}.json"

        def execute(_command: list[str], *, marker: str = output) -> CommandResult:
            path = runner.output_root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {"phase": phase, "simulation_only": True})
            return CommandResult(0)

        runner.run_phase(
            phase,
            command=[] if phase == "complete_without_holdout" else ["rehearsal", phase],
            input_hash=f"rehearsal-{phase}",
            expected_outputs=[] if phase == "complete_without_holdout" else [output],
            executor=execute,
        )
    return {
        "status": "pass",
        "last_phase": runner.events()[-1]["phase"],
        "event_count": len(runner.events()),
        "simulation_only": True,
        "holdout_capability": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ARTIFACT_ROOT))
    args = parser.parse_args()
    artifact_root = Path(args.output_root).resolve()
    rehearsal_id = datetime.now().strftime("g14r_rehearsal_%Y%m%d_%H%M%S_%f")
    root = artifact_root / "rehearsal_runs" / rehearsal_id
    root.mkdir(parents=True, exist_ok=False)
    protocol_path = CONFIG_ROOT / "protocol_v1_1_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    command_log: list[dict[str, Any]] = []
    inputs, checkpoint_reports = train_cells(root, command_log)
    fairness_paths: dict[str, Path] = {}
    benchmark_dirs: list[Path] = []
    for label, runtime_path in CAPACITIES.items():
        runtime = resolve_model_cache_runtime(runtime_path, root=ROOT)
        path = root / "fairness" / f"{label}.json"
        write_json(path, fairness(float(runtime["cache_capacity_profile"]["capacity_mb"]), root / "benchmarks"))
        fairness_paths[label] = path
        benchmark_dirs.append(benchmark(root, label, inputs[label], path, f"benchmarks/{label}", command_log))
    endpoint_report = reconcile(benchmark_dirs)
    support_report = run_support(root, protocol_path, protocol, inputs["medium_576mb"], command_log)
    phase_report = phase_simulation(root, protocol)
    write_json(root / "command_log.json", {"commands": command_log})
    summary = {
        "rehearsal_version": "1.0.0",
        "rehearsal_id": rehearsal_id,
        "status": "pass",
        "scope": "bounded non-formal non-holdout execution-contract rehearsal",
        "agents": AGENTS,
        "domain_baseline": "cache_offload_drl",
        "seeds": SEEDS,
        "capacities_mb": [288.0, 576.0],
        "training_cell_count": len(checkpoint_reports),
        "checkpoint_frequency_updates": 4,
        "saved_checkpoint_update_indices": [4],
        "sa_auxiliary_coef": 0.06,
        "checkpoint_restore_and_provenance": "pass",
        "endpoint_reconciliation": endpoint_report,
        "support_rehearsal": support_report,
        "phase_runner": phase_report,
        "formal_checkpoint_count": 0,
        "formal_episode_count": 0,
        "holdout_opened": False,
        "hidden_data_used": False,
        "performance_claims": [],
        "rehearsal_root": str(root),
    }
    write_json(root / "rehearsal_summary.json", summary)
    write_json(artifact_root / "rehearsal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
