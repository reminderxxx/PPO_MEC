"""Run bounded non-formal G14R9 exact-unit and three-capacity rehearsals."""

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

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest
from src.evaluators.typed_model_cache_formal_protocol import sha256_file


LEARNED = [
    "sa_ghmappo",
    "ppo",
    "mappo",
    "dqn",
    "dueling_dqn",
    "qmix",
    "controller_mat",
    "dag_offload_drl",
    "cache_offload_drl",
    "dt_handoff_drl",
]
REACTIVE = [
    "reactive_lru",
    "reactive_fifo",
    "reactive_lfu",
    "reactive_aging_lfu",
    "reactive_random",
]
ALL_AGENTS = [*REACTIVE, *LEARNED]
EXACT_WINDOW_ID = "g14b_i_80_run_003_f10501_10524_t1113438615000_1113438617300"
EXACT_UNIT_ID = f"seed_7/{EXACT_WINDOW_ID}/j_8"
MOBILITY = ROOT / (
    "data/raw/mobility/ngsim/"
    "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
)
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
PROTOCOL_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831"
LEGACY_RUNTIME_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def run(command: list[str], *, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    write_json(
        log_path,
        {
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-12000:],
            "stderr_tail": completed.stderr[-12000:],
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)


def build_window_plan(path: Path, *, exact: bool) -> dict[str, Any]:
    source = read_json(
        (
            ROOT
            / "configs/experiment/typed_model_cache_formal_protocol_v1_20260820/dev_window_plan.json"
        )
        if exact
        else PROTOCOL_ROOT / "nonformal_rehearsal_window_plan.json"
    )
    if exact:
        rows = [
            row
            for row in source["selected_window_plan"]
            if row["window_id"] == EXACT_WINDOW_ID
        ]
        if len(rows) != 1:
            raise ValueError("exact G14C v9 window identity is missing or duplicated")
    else:
        rows = [source["selected_window_plan"][0]]
    plan = {
        "protocol_version": "g14r9_nonformal_rehearsal_v1",
        "split": "nonformal_rehearsal",
        "sealed": False,
        "outcome_blind_selection": True,
        "selected_window_plan": rows,
    }
    write_json(path, plan)
    return plan


def fairness(
    *, path: Path, plan_path: Path, capacity: float, max_steps: int, exact: bool
) -> dict[str, Any]:
    manifest = build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=plan_path,
        catalog_path=CATALOG,
        seeds=[7],
        max_workflows=1,
        workflow_selector="index:1" if exact else "ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=max_steps,
        max_mobility_rows=11850526 if exact else 1500,
        primary_vehicle_selection="handoff_pressure",
        capacity_unit="mb",
        capacity_value=capacity,
        output_root=str(path.parent / "benchmark"),
        evaluation_unit_limit=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        controller_agents=LEARNED,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise ValueError(report["errors"])
    write_json(path, manifest)
    return manifest


def train_checkpoints(
    work_root: Path, python: str, *, capacity_label: str, runtime_path: Path
) -> tuple[Path, Path, list[dict[str, Any]]]:
    training_root = work_root / f"tiny_training_{capacity_label}"
    seed_manifest: dict[str, dict[str, str]] = {}
    provenance_manifest: dict[str, dict[str, dict[str, Any]]] = {}
    reports = []
    for agent in LEARNED:
        run_id = f"g14r9_{capacity_label}_{agent}_seed7"
        run(
            [
                python,
                str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                "--agent_name",
                agent,
                "--profile",
                "smoke",
                "--episodes",
                "4",
                "--update_every",
                "1",
                "--batch_size",
                "1",
                "--max_steps",
                "1",
                "--checkpoint_every_updates",
                "4",
                "--max_workflows",
                "1",
                "--workflow_selector",
                "ordered",
                "--min_tasks",
                "5",
                "--max_tasks",
                "20",
                "--mobility_source",
                "ngsim",
                "--mobility_csv_path",
                str(MOBILITY),
                "--max_mobility_rows",
                "1500",
                "--window_selector",
                "ordered",
                "--window_count",
                "1",
                "--window_length",
                "24",
                "--rsu_layout",
                "auto_dominant_tight",
                "--primary_vehicle_selection",
                "handoff_pressure",
                "--model_cache_runtime_config",
                str(runtime_path),
                "--reward_positive_offset",
                "0",
                "--random_seed",
                "7",
                "--output_root",
                str(training_root),
                "--run_id",
                run_id,
            ],
            log_path=work_root / "logs" / f"train_{agent}.json",
        )
        checkpoint = training_root / agent / run_id / "checkpoints/update_0004.pt"
        payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata")
        typed = metadata["typed_runtime_provenance"]
        digest = sha256_file(checkpoint)
        seed_manifest[agent] = {"7": str(checkpoint)}
        provenance_manifest[agent] = {
            "7": {
                "checkpoint_sha256": digest,
                "execution_git_commit": typed.get("execution_git_commit"),
                "train_window_plan_identity": typed["train_window_plan_identity"],
            }
        }
        reports.append(
            {
                "agent": agent,
                "checkpoint_sha256": digest,
                "formal": False,
                "performance_evidence": False,
                "source_contains_v9": V9_RUN_ID in str(checkpoint),
            }
        )
    seed_path = work_root / f"seed_checkpoint_manifest_{capacity_label}.json"
    provenance_path = work_root / f"checkpoint_provenance_manifest_{capacity_label}.json"
    write_json(seed_path, seed_manifest)
    write_json(provenance_path, provenance_manifest)
    return seed_path, provenance_path, reports


V9_RUN_ID = "typed_model_cache_formal_20260830_113339_g14c_v9"


def benchmark_command(
    *,
    python: str,
    agents: list[str],
    runtime_path: Path,
    fairness_path: Path,
    plan_path: Path,
    output_root: Path,
    max_steps: int,
    max_rows: int,
    workflow_selector: str,
    seed_path: Path | None = None,
    provenance_path: Path | None = None,
) -> list[str]:
    command = [
        python,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents",
        *agents,
        "--seeds",
        "7",
        "--cache_baseline_fairness_manifest_path",
        str(fairness_path),
        "--model_cache_runtime_config",
        str(runtime_path),
        "--window_plan_path",
        str(plan_path),
        "--mobility_csv_path",
        str(MOBILITY),
        "--workflow_csv_path",
        str(WORKFLOW),
        "--window_selector",
        "ordered",
        "--window_length",
        "24",
        "--rsu_layout",
        "auto_dominant_tight",
        "--max_mobility_rows",
        str(max_rows),
        "--max_workflows",
        "1",
        "--max_steps",
        str(max_steps),
        "--workflow_selector",
        workflow_selector,
        "--min_tasks",
        "5",
        "--max_tasks",
        "20",
        "--primary_vehicle_selection",
        "handoff_pressure",
        "--window_mode",
        "mixed_informative",
        "--prediction_horizon",
        "3",
        "--reward_positive_offset",
        "0",
        "--formal-exogenous-request-execution",
        "--output_root",
        str(output_root),
    ]
    if seed_path and provenance_path:
        command.extend(
            [
                "--seed_checkpoint_manifest_path",
                str(seed_path),
                "--checkpoint_provenance_manifest_path",
                str(provenance_path),
                "--formal-agent-order-contract-path",
                str(PROTOCOL_ROOT / "formal_agent_order_contract.json"),
            ]
        )
    return command


def single_benchmark_run(output_root: Path) -> Path:
    runs = list(output_root.glob("main_results_*"))
    if len(runs) != 1:
        raise ValueError(f"expected one benchmark run under {output_root}: {runs}")
    return runs[0]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    global MOBILITY, WORKFLOW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--require-clean-candidate", action="store_true")
    args = parser.parse_args()
    work_root = Path(args.work_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    python = str(Path(args.python_executable).absolute())
    data_root = Path(args.data_root).resolve()
    MOBILITY = data_root / (
        "raw/mobility/ngsim/"
        "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
    )
    WORKFLOW = data_root / "raw/workflow/alibaba2018/batch_task.csv"
    candidate_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    clean_candidate_start = not bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    detached_candidate = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode != 0
    candidate_has_local_venv = (ROOT / ".venv").exists()
    if args.require_clean_candidate and (
        not clean_candidate_start or not detached_candidate or candidate_has_local_venv
    ):
        raise ValueError(
            "clean-candidate rehearsal requires clean detached HEAD without local .venv"
        )
    work_root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=True)

    exact_plan_path = artifact_root / "evidence_inputs/exact_window_plan.json"
    build_window_plan(exact_plan_path, exact=True)
    exact_fairness_path = artifact_root / "evidence_inputs/exact_fairness_288mb.json"
    exact_manifest = fairness(
        path=exact_fairness_path,
        plan_path=exact_plan_path,
        capacity=288.0,
        max_steps=22,
        exact=True,
    )
    exact_units = exact_manifest["window_workload_plan"]["evaluation_units"]
    if [row["evaluation_unit_id"] for row in exact_units] != [EXACT_UNIT_ID]:
        raise ValueError("exact failure evaluation-unit identity drift")
    seed_path, provenance_path, checkpoint_reports = train_checkpoints(
        work_root,
        python,
        capacity_label="constrained_288mb",
        runtime_path=LEGACY_RUNTIME_ROOT / "runtime_constrained_288mb.yaml",
    )
    exact_output = work_root / "exact_benchmark"
    run(
        benchmark_command(
            python=python,
            agents=ALL_AGENTS,
            runtime_path=LEGACY_RUNTIME_ROOT / "runtime_constrained_288mb.yaml",
            fairness_path=exact_fairness_path,
            plan_path=exact_plan_path,
            output_root=exact_output,
            max_steps=22,
            max_rows=11850526,
            workflow_selector="index:1",
            seed_path=seed_path,
            provenance_path=provenance_path,
        ),
        log_path=work_root / "logs/exact_benchmark.json",
    )
    exact_run = single_benchmark_run(exact_output)
    exact_rows = csv_rows(exact_run / "benchmark_rows.csv")
    fingerprints = {
        row["agent_name"]: row["request_exposure_fingerprint"] for row in exact_rows
    }
    if [row["agent_name"] for row in exact_rows] != ALL_AGENTS:
        raise ValueError("exact failure unit 15-agent order or membership drift")
    if len(set(fingerprints.values())) != 1:
        raise ValueError("exact failure unit external request fingerprint drift")
    candidate_inputs = [
        {
            "agent": row["agent_name"],
            "update_index": 4,
            "evaluation_unit_id": EXACT_UNIT_ID,
            "full_service_ready_byte_hit_rate": row[
                "full_service_ready_byte_hit_rate"
            ],
            "request_exposure_fingerprint": row["request_exposure_fingerprint"],
        }
        for row in exact_rows
        if row["agent_name"] in LEARNED
    ]
    write_json(artifact_root / "candidate_selection_inputs.json", candidate_inputs)
    exact_report = {
        "status": "pass",
        "formal": False,
        "performance_evidence": False,
        "holdout_opened": False,
        "evaluation_unit_id": EXACT_UNIT_ID,
        "capacity_mb": 288.0,
        "max_steps": 22,
        "agent_count": len(exact_rows),
        "agent_order": [row["agent_name"] for row in exact_rows],
        "request_exposure_fingerprint": next(iter(set(fingerprints.values()))),
        "fingerprint_count": len(set(fingerprints.values())),
        "observed_alignment_all_pass": all(
            row["request_alignment_status"] == "pass" for row in exact_rows
        ),
        "benchmark_rows_sha256": sha256_file(exact_run / "benchmark_rows.csv"),
        "candidate_selection_input_count": len(candidate_inputs),
        "fresh_tiny_checkpoint_count": len(checkpoint_reports),
        "checkpoint_reports": checkpoint_reports,
        "v9_checkpoint_reused": False,
        "cross_agent_validation_position_passed": True,
        "clean_candidate_start": clean_candidate_start,
        "detached_candidate": detached_candidate,
        "candidate_has_local_venv": candidate_has_local_venv,
        "candidate_commit": candidate_commit,
    }
    write_json(artifact_root / "exact_failure_unit_rehearsal.json", exact_report)

    controlled_plan_path = artifact_root / "evidence_inputs/controlled_window_plan.json"
    build_window_plan(controlled_plan_path, exact=False)
    capacity_reports = []
    for label, capacity, runtime_name in (
        ("constrained_288mb", 288.0, "runtime_constrained_288mb.yaml"),
        ("medium_576mb", 576.0, "runtime_medium_576mb.yaml"),
        ("relaxed_864mb", 864.0, "runtime_relaxed_864mb.yaml"),
    ):
        manifest_path = artifact_root / f"evidence_inputs/controlled_fairness_{label}.json"
        fairness(
            path=manifest_path,
            plan_path=controlled_plan_path,
            capacity=capacity,
            max_steps=1,
            exact=False,
        )
        output = work_root / f"controlled_benchmark_{label}"
        capacity_seed_path = seed_path
        capacity_provenance_path = provenance_path
        if label != "constrained_288mb":
            capacity_seed_path, capacity_provenance_path, _ = train_checkpoints(
                work_root,
                python,
                capacity_label=label,
                runtime_path=LEGACY_RUNTIME_ROOT / runtime_name,
            )
        run(
            benchmark_command(
                python=python,
                agents=ALL_AGENTS,
                runtime_path=LEGACY_RUNTIME_ROOT / runtime_name,
                fairness_path=manifest_path,
                plan_path=controlled_plan_path,
                output_root=output,
                max_steps=1,
                max_rows=1500,
                workflow_selector="ordered",
                seed_path=capacity_seed_path,
                provenance_path=capacity_provenance_path,
            ),
            log_path=work_root / "logs" / f"controlled_{label}.json",
        )
        run_root = single_benchmark_run(output)
        rows = csv_rows(run_root / "benchmark_rows.csv")
        exposure = {row["request_exposure_fingerprint"] for row in rows}
        capacity_reports.append(
            {
                "capacity_label": label,
                "capacity_mb": capacity,
                "runtime_contract_sha256": rows[0]["runtime_contract_sha256"],
                "agent_count": len(rows),
                "request_exposure_fingerprint_count": len(exposure),
                "request_exposure_fingerprint": next(iter(exposure)),
                "alignment_all_pass": all(
                    row["request_alignment_status"] == "pass" for row in rows
                ),
            }
        )
    write_json(
        artifact_root / "three_capacity_rehearsal.json",
        {
            "status": "pass",
            "formal": False,
            "performance_evidence": False,
            "holdout_opened": False,
            "controlled_max_steps": 1,
            "capacities": capacity_reports,
        },
    )
    print(json.dumps({"status": "pass", "exact": exact_report, "capacities": capacity_reports}, indent=2))


if __name__ == "__main__":
    main()
