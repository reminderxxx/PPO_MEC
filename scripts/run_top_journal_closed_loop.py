"""Run the top-journal experiment closed loop.

The script owns the reproducible loop: train SA-GHMAPPO, train matched
paper-grade learned baselines, build a per-seed checkpoint manifest, run benchmark
modes, and write a gate report that makes pass/fail blockers explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_CSV = ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "experiments" / "top_journal_closed_loop"
SA_CHECKPOINT_PRIORITY = [
    "best_by_reward_tiebreak_score_path",
    "best_by_continuity_path",
    "best_by_retained_mechanism_score_path",
    "best_by_mechanism_advantage_score_path",
    "best_by_mechanism_balanced_path",
    "best_by_reward_path",
    "latest_checkpoint_path",
]
SA_REWARD_FIRST_CHECKPOINT_PRIORITY = [
    "best_by_reward_path",
    "best_by_reward_tiebreak_score_path",
    "best_by_continuity_path",
    "best_by_retained_mechanism_score_path",
    "best_by_mechanism_advantage_score_path",
    "best_by_mechanism_balanced_path",
    "latest_checkpoint_path",
]
SA_REWARD_FIRST_PROFILES = {
    "top_journal_mechanism_v11_mappo_reward",
    "top_journal_mechanism_v12_learned_option",
}
SA_LATEST_FIRST_CHECKPOINT_PRIORITY = [
    "latest_checkpoint_path",
    "best_by_reward_path",
    "best_by_reward_tiebreak_score_path",
    "best_by_continuity_path",
    "best_by_retained_mechanism_score_path",
    "best_by_mechanism_advantage_score_path",
    "best_by_mechanism_balanced_path",
]
SA_LATEST_FIRST_PROFILES = {
    "top_journal_mechanism_v13_prd_option",
    "top_journal_mechanism_v14_net_utility_prd",
    "top_journal_mechanism_v15_terminal_option",
    "top_journal_mechanism_v16_conservative_terminal_option",
    "top_journal_mechanism_v17_dag_aware_option",
    "top_journal_mechanism_v18_counterfactual_option",
    "top_journal_mechanism_v19_handoff_risk_prd",
    "top_journal_mechanism_v20_idle_execution_prd",
    "top_journal_mechanism_v21_efficiency_prd",
    "top_journal_mechanism_v22_validated_utility_prd",
    "top_journal_mechanism_v23_counterfactual_constrained_prd",
    "top_journal_mechanism_v24_tail_risk_constrained_prd",
    "top_journal_mechanism_v25_opportunity_risk_prd",
    "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd",
}
LOWER_IS_BETTER = {
    "backhaul_traffic_cost",
    "handoff_failure_rate",
    "adapter_state_migration_overhead",
    "end_to_end_workflow_delay",
    "cross_rsu_cold_start_frequency",
}
FORMAL_MIN_SEED_COUNT = 3
FORMAL_REQUIRED_BENCHMARK_MODES = {"mixed_informative", "full_stratified"}
FORMAL_REQUIRED_PRIMARY_VEHICLE_SELECTION = "handoff_pressure"
FORMAL_MIN_SETTINGS = {
    "sa_episodes": 96,
    "baseline_episodes": 48,
    "sa_update_every": 4,
    "baseline_update_every": 6,
    "sa_batch_size": 32,
    "baseline_batch_size": 32,
    "max_mobility_rows": 2500,
    "max_workflows": 2,
    "window_length": 24,
    "window_count": 3,
    "train_window_count": 5,
    "window_scan_stride": 2,
    "max_steps": 16,
    "min_tasks": 5,
    "max_tasks": 20,
}
SA_PROFILE_SETTING_OVERRIDES = {
    "top_journal_mechanism_v6_strong_competition": {
        "sa_episodes": 128,
        "train_window_count": 6,
    },
    "top_journal_mechanism_v7_latency_fallback": {
        "sa_episodes": 128,
        "train_window_count": 6,
    },
    "top_journal_mechanism_v8_strict_full": {
        "sa_episodes": 96,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v9_pareto_safe": {
        "sa_episodes": 96,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v10_mappo_rl": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v11_mappo_reward": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v12_learned_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v13_prd_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v14_net_utility_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v15_terminal_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v16_conservative_terminal_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v17_dag_aware_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v18_counterfactual_option": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v19_handoff_risk_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v20_idle_execution_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v21_efficiency_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v22_validated_utility_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v23_counterfactual_constrained_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v24_tail_risk_constrained_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v25_opportunity_risk_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
    "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd": {
        "sa_episodes": 128,
        "baseline_episodes": 96,
        "sa_update_every": 8,
        "baseline_update_every": 8,
        "train_window_count": 20,
        "max_mobility_rows": 10000,
        "window_count": 20,
    },
}
CURRENT_BASELINE_PROTOCOLS = {
    "mappo": {
        "head_credit_enabled": True,
        "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
        "slow_policy_credit_floor": 0.25,
        "fast_policy_credit_floor": 0.10,
        "event_policy_credit_floor": 0.12,
        "slow_entropy_coef_scale": 1.25,
        "fast_entropy_coef_scale": 1.00,
        "event_entropy_coef_scale": 1.35,
        "slow_entropy_credit_floor": 0.20,
        "fast_entropy_credit_floor": 0.08,
        "event_entropy_credit_floor": 0.12,
        "event_advantage_blend": 0.85,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-journal closed-loop experiments")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--quick", action="store_true", help="Use a tiny smoke-sized closed loop.")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    parser.add_argument(
        "--baseline_agents",
        nargs="+",
        default=[
            "ppo",
            "mappo",
            "dqn",
            "dueling_dqn",
            "qmix",
            "controller_mat",
            "dag_offload_drl",
            "cache_offload_drl",
            "dt_handoff_drl",
        ],
    )
    parser.add_argument("--benchmark_modes", nargs="+", default=["mixed_informative", "full_stratified"])
    parser.add_argument("--workflow_csv_path", type=str, default=str(DEFAULT_WORKFLOW_CSV))
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="handoff_pressure")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_mode_for_training", type=str, default="mixed_informative")
    parser.add_argument("--train_window_plan_path", type=str, default="")
    parser.add_argument("--eval_window_plan_path", type=str, default="")
    parser.add_argument("--sa_profile", type=str, default="top_journal_mechanism_v1")
    parser.add_argument("--mappo_baseline_profile", type=str, default="mappo_strong_audit")
    parser.add_argument("--sa_episodes", type=int, default=None)
    parser.add_argument("--baseline_episodes", type=int, default=None)
    parser.add_argument("--sa_update_every", type=int, default=None)
    parser.add_argument("--baseline_update_every", type=int, default=None)
    parser.add_argument("--sa_batch_size", type=int, default=None)
    parser.add_argument("--baseline_batch_size", type=int, default=None)
    parser.add_argument("--max_mobility_rows", type=int, default=None)
    parser.add_argument("--max_workflows", type=int, default=None)
    parser.add_argument("--window_length", type=int, default=None)
    parser.add_argument("--window_count", type=int, default=None)
    parser.add_argument("--train_window_count", type=int, default=None)
    parser.add_argument("--window_scan_stride", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--min_tasks", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--continuity_tolerance", type=float, default=0.02)
    parser.add_argument("--handoff_failure_tolerance", type=float, default=0.01)
    parser.add_argument("--backhaul_tolerance", type=float, default=0.0)
    parser.add_argument("--mechanism_tolerance", type=float, default=0.02)
    parser.add_argument("--reward_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def effective_settings(args: argparse.Namespace) -> dict[str, int]:
    if args.quick:
        defaults = {
            "sa_episodes": 2,
            "baseline_episodes": 2,
            "sa_update_every": 1,
            "baseline_update_every": 1,
            "sa_batch_size": 2,
            "baseline_batch_size": 2,
            "max_mobility_rows": 500,
            "max_workflows": 1,
            "window_length": 8,
            "window_count": 1,
            "train_window_count": 1,
            "window_scan_stride": 4,
            "max_steps": 2,
            "min_tasks": 5,
            "max_tasks": 10,
        }
    else:
        defaults = {
            "sa_episodes": 96,
            "baseline_episodes": 48,
            "sa_update_every": 4,
            "baseline_update_every": 6,
            "sa_batch_size": 32,
            "baseline_batch_size": 32,
            "max_mobility_rows": 2500,
            "max_workflows": 2,
            "window_length": 24,
            "window_count": 3,
            "train_window_count": 5,
            "window_scan_stride": 2,
            "max_steps": 16,
            "min_tasks": 5,
            "max_tasks": 20,
        }
        profile_overrides = SA_PROFILE_SETTING_OVERRIDES.get(str(args.sa_profile), {})
        for key, value in profile_overrides.items():
            if getattr(args, key, None) is None:
                defaults[key] = int(value)
    return {
        key: int(getattr(args, key)) if getattr(args, key) is not None else value
        for key, value in defaults.items()
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_file(root: Path, filename: str) -> Path:
    candidates = [path for path in root.rglob(filename) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"could not find {filename} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_training_summary_for_seed(root: Path, filename: str, seed: int) -> Path | None:
    seed_suffix = f"_seed{seed}"
    candidates = [
        path
        for path in root.rglob(filename)
        if path.is_file() and path.parent.name.endswith(seed_suffix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def existing_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path if path.exists() else None


def checkpoint_protocol_blockers(agent_name: str, checkpoint_path: str, seed: int) -> list[str]:
    expected_protocol = CURRENT_BASELINE_PROTOCOLS.get(agent_name)
    if not expected_protocol:
        return []
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive audit path
        return [f"checkpoint_protocol_unreadable:{agent_name}:seed{seed}:{type(exc).__name__}"]
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if not isinstance(config, dict):
        return [f"checkpoint_protocol_missing_config:{agent_name}:seed{seed}"]
    blockers: list[str] = []
    for key, expected_value in expected_protocol.items():
        actual_value = config.get(key)
        if isinstance(expected_value, bool):
            matches = bool(actual_value) is expected_value
        elif isinstance(expected_value, float):
            try:
                matches = abs(float(actual_value) - expected_value) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            blockers.append(
                f"checkpoint_protocol_mismatch:{agent_name}:seed{seed}:{key}:{actual_value}!={expected_value}"
            )
    return blockers


def audit_baseline_checkpoint_protocols(training_records: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    for record in training_records:
        agent_name = str(record.get("agent_name", ""))
        checkpoint_path = str(record.get("selected_checkpoint_path", ""))
        seed = int(record.get("seed", 0) or 0)
        if not checkpoint_path:
            continue
        blockers.extend(checkpoint_protocol_blockers(agent_name, checkpoint_path, seed))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "expected_protocols": CURRENT_BASELINE_PROTOCOLS,
    }


def select_sa_checkpoint(train_summary: dict[str, Any]) -> tuple[Path, str]:
    config_profile = str(train_summary.get("config_profile", ""))
    if config_profile in SA_LATEST_FIRST_PROFILES:
        priority = SA_LATEST_FIRST_CHECKPOINT_PRIORITY
    elif config_profile in SA_REWARD_FIRST_PROFILES:
        priority = SA_REWARD_FIRST_CHECKPOINT_PRIORITY
    else:
        priority = SA_CHECKPOINT_PRIORITY
    for field_name in priority:
        candidate = existing_path(str(train_summary.get(field_name, "")))
        if candidate is not None:
            return candidate, field_name
    raise FileNotFoundError("SA train_summary has no usable checkpoint path")


def select_baseline_checkpoint(train_summary: dict[str, Any]) -> Path:
    candidate = existing_path(str(train_summary.get("latest_checkpoint_path", "")))
    if candidate is None:
        raise FileNotFoundError("baseline train_summary has no usable latest_checkpoint_path")
    return candidate


def existing_training_record(
    *,
    run_root: Path,
    agent_name: str,
    seed: int,
) -> dict[str, Any] | None:
    if agent_name == "sa_ghmappo":
        summary_root = run_root / "training" / "sa" / "sa_ghmappo"
    else:
        summary_root = run_root / "training" / "algo_pool" / agent_name
    summary_path = latest_training_summary_for_seed(summary_root, "train_summary.json", seed)
    if summary_path is None:
        return None
    train_summary = read_json(summary_path)
    if agent_name == "sa_ghmappo":
        checkpoint_path, selection_field = select_sa_checkpoint(train_summary)
    else:
        checkpoint_path = select_baseline_checkpoint(train_summary)
        selection_field = "latest_checkpoint_path"
    return {
        "seed": seed,
        "agent_name": agent_name,
        "train_summary_path": str(summary_path),
        "selected_checkpoint_path": str(checkpoint_path),
        "selection_field": selection_field,
        "resumed_from_existing": True,
    }


def run_command(
    *,
    label: str,
    cmd: list[str],
    cwd: Path,
    command_log: list[dict[str, Any]],
) -> None:
    started = time.time()
    print(f"[closed-loop] start {label}")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed_sec = round(time.time() - started, 3)
    command_log.append(
        {
            "label": label,
            "command": cmd,
            "returncode": completed.returncode,
            "elapsed_sec": elapsed_sec,
        }
    )
    print(f"[closed-loop] finish {label}: returncode={completed.returncode}, elapsed_sec={elapsed_sec}")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed for {label}: returncode={completed.returncode}")


def common_real_args(args: argparse.Namespace, settings: dict[str, int]) -> list[str]:
    common = [
        "--mobility_source",
        args.mobility_source,
        "--primary_vehicle_selection",
        args.primary_vehicle_selection,
        "--workflow_csv_path",
        args.workflow_csv_path,
        "--max_mobility_rows",
        str(settings["max_mobility_rows"]),
        "--max_workflows",
        str(settings["max_workflows"]),
        "--workflow_selector",
        args.workflow_selector,
        "--rsu_layout",
        args.rsu_layout,
        "--window_selector",
        args.window_selector,
        "--window_length",
        str(settings["window_length"]),
        "--window_scan_stride",
        str(settings["window_scan_stride"]),
        "--min_tasks",
        str(settings["min_tasks"]),
        "--max_tasks",
        str(settings["max_tasks"]),
    ]
    if args.mobility_csv_path:
        common.extend(["--mobility_csv_path", args.mobility_csv_path])
    return common


def train_sa_for_seed(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    seed: int,
    run_root: Path,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    training_root = run_root / "training" / "sa"
    cmd = [
        args.python_executable,
        "scripts/train_sa_ghmappo_real_sample.py",
        "--agent_name",
        "sa_ghmappo",
        "--profile",
        args.sa_profile,
        "--episodes",
        str(settings["sa_episodes"]),
        "--update_every",
        str(settings["sa_update_every"]),
        "--batch_size",
        str(settings["sa_batch_size"]),
        "--max_steps",
        str(settings["max_steps"]),
        "--train_window_count",
        str(settings["train_window_count"]),
        "--random_seed",
        str(seed),
        "--window_mode",
        args.window_mode_for_training,
        "--output_root",
        str(training_root),
        *common_real_args(args, settings),
    ]
    if args.train_window_plan_path:
        cmd.extend(["--window_plan_path", args.train_window_plan_path])
    run_command(label=f"train_sa_seed_{seed}", cmd=cmd, cwd=ROOT_DIR, command_log=command_log)
    summary_path = latest_file(training_root / "sa_ghmappo", "train_summary.json")
    train_summary = read_json(summary_path)
    checkpoint_path, selection_field = select_sa_checkpoint(train_summary)
    return {
        "seed": seed,
        "agent_name": "sa_ghmappo",
        "train_summary_path": str(summary_path),
        "selected_checkpoint_path": str(checkpoint_path),
        "selection_field": selection_field,
    }


def train_baseline_for_seed(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    agent_name: str,
    seed: int,
    run_root: Path,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    training_root = run_root / "training" / "algo_pool"
    profile = args.mappo_baseline_profile if agent_name == "mappo" else "baseline_safe"
    cmd = [
        args.python_executable,
        "scripts/train_algo_pool_real_sample.py",
        "--agent_name",
        agent_name,
        "--profile",
        profile,
        "--episodes",
        str(settings["baseline_episodes"]),
        "--update_every",
        str(settings["baseline_update_every"]),
        "--batch_size",
        str(settings["baseline_batch_size"]),
        "--max_steps",
        str(settings["max_steps"]),
        "--window_count",
        str(settings["window_count"]),
        "--window_mode",
        args.window_mode_for_training,
        "--random_seed",
        str(seed),
        "--output_root",
        str(training_root),
        *common_real_args(args, settings),
    ]
    if args.train_window_plan_path:
        cmd.extend(["--window_plan_path", args.train_window_plan_path])
    run_command(label=f"train_{agent_name}_seed_{seed}", cmd=cmd, cwd=ROOT_DIR, command_log=command_log)
    summary_path = latest_file(training_root / agent_name, "train_summary.json")
    train_summary = read_json(summary_path)
    checkpoint_path = select_baseline_checkpoint(train_summary)
    return {
        "seed": seed,
        "agent_name": agent_name,
        "train_summary_path": str(summary_path),
        "selected_checkpoint_path": str(checkpoint_path),
        "selection_field": "latest_checkpoint_path",
    }


def build_seed_manifest(training_records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for record in training_records:
        agent_name = str(record["agent_name"])
        seed = str(record["seed"])
        manifest.setdefault(agent_name, {})[seed] = str(record["selected_checkpoint_path"])
    return manifest


def run_benchmark_mode(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    mode: str,
    run_root: Path,
    manifest_path: Path,
    command_log: list[dict[str, Any]],
) -> Path:
    output_root = run_root / "benchmarks" / mode
    agents = ["sa_ghmappo", "reactive_greedy", "popularity_cache_heuristic", *args.baseline_agents]
    cmd = [
        args.python_executable,
        "scripts/benchmark_main_results.py",
        "--agents",
        *agents,
        "--seed_checkpoint_manifest_path",
        str(manifest_path),
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--max_steps",
        str(settings["max_steps"]),
        "--window_count",
        str(settings["window_count"]),
        "--window_mode",
        mode,
        "--output_root",
        str(output_root),
        *common_real_args(args, settings),
    ]
    if args.eval_window_plan_path:
        cmd.extend(["--window_plan_path", args.eval_window_plan_path])
    run_command(label=f"benchmark_{mode}", cmd=cmd, cwd=ROOT_DIR, command_log=command_log)
    return latest_file(output_root, "aggregate_summary.json")


def metric_mean(aggregate_summary: dict[str, Any], agent_name: str, metric_name: str) -> float | None:
    metric = (
        aggregate_summary.get("aggregate_by_agent", {})
        .get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
    )
    if "mean" not in metric:
        return None
    value = metric.get("mean")
    return float(value) if value is not None else None


def beats_or_ties(
    candidate: float | None,
    reference: float | None,
    *,
    metric_name: str,
    tolerance: float,
    require_win: bool = False,
) -> bool:
    if candidate is None or reference is None:
        return False
    if metric_name in LOWER_IS_BETTER:
        return candidate < reference - tolerance if require_win else candidate <= reference + tolerance
    return candidate > reference + tolerance if require_win else candidate + tolerance >= reference


def evaluate_mode_gate(
    *,
    aggregate_path: Path,
    baseline_agents: list[str],
    tolerances: dict[str, float],
    quick: bool,
) -> dict[str, Any]:
    summary = read_json(aggregate_path)
    blockers: list[str] = []
    mode = str(summary.get("window_mode", aggregate_path.parent.name))
    warnings = list(summary.get("smoke_checkpoint_warnings", []) or [])
    if warnings:
        blockers.append("smoke_checkpoint_warnings_present")
    if not quick and not bool(summary.get("canonical_paper_protocol", False)):
        blockers.append("not_canonical_paper_protocol")

    metrics: dict[str, dict[str, float | None]] = {}
    agents = ["sa_ghmappo", "popularity_cache_heuristic", *baseline_agents]
    for metric_name in [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "handoff_ready_ratio",
        "mechanism_realization_rate",
        "adapter_state_migration_overhead",
    ]:
        metrics[metric_name] = {
            agent_name: metric_mean(summary, agent_name, metric_name)
            for agent_name in agents
        }

    for agent_name in baseline_agents:
        if not beats_or_ties(
            metrics["total_reward"].get("sa_ghmappo"),
            metrics["total_reward"].get(agent_name),
            metric_name="total_reward",
            tolerance=tolerances["reward"],
            require_win=True,
        ):
            blockers.append(f"sa_total_reward_not_above_{agent_name}")

    if not beats_or_ties(
        metrics["total_reward"].get("sa_ghmappo"),
        metrics["total_reward"].get("popularity_cache_heuristic"),
        metric_name="total_reward",
        tolerance=tolerances["reward"],
        require_win=True,
    ):
        blockers.append("sa_total_reward_not_above_popularity")
    if not beats_or_ties(
        metrics["workflow_continuity_rate"].get("sa_ghmappo"),
        metrics["workflow_continuity_rate"].get("popularity_cache_heuristic"),
        metric_name="workflow_continuity_rate",
        tolerance=tolerances["continuity"],
    ):
        blockers.append("continuity_below_popularity_tolerance")
    if not beats_or_ties(
        metrics["handoff_failure_rate"].get("sa_ghmappo"),
        metrics["handoff_failure_rate"].get("popularity_cache_heuristic"),
        metric_name="handoff_failure_rate",
        tolerance=tolerances["handoff_failure"],
    ):
        blockers.append("handoff_failure_above_popularity_tolerance")
    if not beats_or_ties(
        metrics["backhaul_traffic_cost"].get("sa_ghmappo"),
        metrics["backhaul_traffic_cost"].get("popularity_cache_heuristic"),
        metric_name="backhaul_traffic_cost",
        tolerance=tolerances["backhaul"],
    ):
        blockers.append("backhaul_above_popularity_tolerance")

    mechanism_ok = beats_or_ties(
        metrics["mechanism_realization_rate"].get("sa_ghmappo"),
        metrics["mechanism_realization_rate"].get("popularity_cache_heuristic"),
        metric_name="mechanism_realization_rate",
        tolerance=tolerances["mechanism"],
    )
    ready_ok = beats_or_ties(
        metrics["handoff_ready_ratio"].get("sa_ghmappo"),
        metrics["handoff_ready_ratio"].get("popularity_cache_heuristic"),
        metric_name="handoff_ready_ratio",
        tolerance=tolerances["mechanism"],
    )
    if not (mechanism_ok or ready_ok):
        blockers.append("mechanism_realization_and_ready_below_popularity_tolerance")

    diagnosis = summary.get("sa_advantage_diagnosis", {})
    if diagnosis and not bool(diagnosis.get("minimum_success_reached", False)):
        blockers.append("benchmark_minimum_success_not_reached")

    return {
        "mode": mode,
        "aggregate_summary_path": str(aggregate_path),
        "passed": not blockers,
        "blockers": blockers,
        "smoke_checkpoint_warnings": warnings,
        "sa_advantage_diagnosis": diagnosis,
        "metrics": metrics,
        "episode_count": summary.get("episode_count", 0),
        "canonical_paper_protocol": bool(summary.get("canonical_paper_protocol", False)),
        "config_profile": summary.get("config_profile", {}),
    }


def write_gate_csv(path: Path, mode_reports: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for report in mode_reports:
        metrics = report.get("metrics", {})
        rows.append(
            {
                "mode": report.get("mode", ""),
                "passed": report.get("passed", False),
                "blockers": ";".join(report.get("blockers", [])),
                "episode_count": report.get("episode_count", 0),
                "sa_total_reward": metrics.get("total_reward", {}).get("sa_ghmappo"),
                "popularity_total_reward": metrics.get("total_reward", {}).get("popularity_cache_heuristic"),
                "sa_continuity": metrics.get("workflow_continuity_rate", {}).get("sa_ghmappo"),
                "popularity_continuity": metrics.get("workflow_continuity_rate", {}).get("popularity_cache_heuristic"),
                "sa_handoff_failure": metrics.get("handoff_failure_rate", {}).get("sa_ghmappo"),
                "popularity_handoff_failure": metrics.get("handoff_failure_rate", {}).get("popularity_cache_heuristic"),
                "sa_backhaul": metrics.get("backhaul_traffic_cost", {}).get("sa_ghmappo"),
                "popularity_backhaul": metrics.get("backhaul_traffic_cost", {}).get("popularity_cache_heuristic"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_formal_contract(args: argparse.Namespace, settings: dict[str, int]) -> dict[str, Any]:
    blockers: list[str] = []
    if args.quick:
        blockers.append("quick_mode")
    if args.primary_vehicle_selection != FORMAL_REQUIRED_PRIMARY_VEHICLE_SELECTION:
        blockers.append(
            "primary_vehicle_selection_not_handoff_pressure:"
            f"{args.primary_vehicle_selection}"
        )
    unique_seed_count = len(set(int(seed) for seed in args.seeds))
    if unique_seed_count < FORMAL_MIN_SEED_COUNT:
        blockers.append(f"fewer_than_{FORMAL_MIN_SEED_COUNT}_seeds:{unique_seed_count}")
    missing_modes = sorted(FORMAL_REQUIRED_BENCHMARK_MODES - set(args.benchmark_modes))
    if missing_modes:
        blockers.append("missing_required_benchmark_modes:" + ",".join(missing_modes))
    for key, minimum in FORMAL_MIN_SETTINGS.items():
        actual = int(settings.get(key, 0))
        if actual < minimum:
            blockers.append(f"{key}_below_formal_min:{actual}<{minimum}")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "required_primary_vehicle_selection": FORMAL_REQUIRED_PRIMARY_VEHICLE_SELECTION,
        "min_seed_count": FORMAL_MIN_SEED_COUNT,
        "required_benchmark_modes": sorted(FORMAL_REQUIRED_BENCHMARK_MODES),
        "min_settings": FORMAL_MIN_SETTINGS,
    }


def build_gate_report(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    run_root: Path,
    manifest_path: Path,
    training_records: list[dict[str, Any]],
    benchmark_paths: list[Path],
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    tolerances = {
        "continuity": args.continuity_tolerance,
        "handoff_failure": args.handoff_failure_tolerance,
        "backhaul": args.backhaul_tolerance,
        "mechanism": args.mechanism_tolerance,
        "reward": args.reward_tolerance,
    }
    mode_reports = [
        evaluate_mode_gate(
            aggregate_path=path,
            baseline_agents=list(args.baseline_agents),
            tolerances=tolerances,
            quick=args.quick,
        )
        for path in benchmark_paths
    ]
    protocol_audit = audit_baseline_checkpoint_protocols(training_records)
    passed = (
        bool(mode_reports)
        and all(report["passed"] for report in mode_reports)
        and bool(protocol_audit["passed"])
    )
    formal_contract = audit_formal_contract(args, settings)
    if protocol_audit["blockers"]:
        formal_contract["blockers"] = sorted(set([*formal_contract["blockers"], *protocol_audit["blockers"]]))
        formal_contract["ready"] = False
    report = {
        "run_id": run_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "quick_mode": bool(args.quick),
        "formal_contract": formal_contract,
        "paper_claim_ready": bool(passed and formal_contract["ready"]),
        "passed": passed,
        "settings": settings,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "seeds": args.seeds,
        "baseline_agents": args.baseline_agents,
        "mappo_baseline_profile": args.mappo_baseline_profile,
        "benchmark_modes": args.benchmark_modes,
        "seed_checkpoint_manifest_path": str(manifest_path),
        "training_records": training_records,
        "baseline_protocol_audit": protocol_audit,
        "baseline_protocol_versions": CURRENT_BASELINE_PROTOCOLS,
        "mode_reports": mode_reports,
        "command_log": command_log,
        "next_optimization_hints": optimization_hints(mode_reports),
    }
    return report


def optimization_hints(mode_reports: list[dict[str, Any]]) -> list[str]:
    blockers = {
        blocker
        for report in mode_reports
        for blocker in report.get("blockers", [])
    }
    hints: list[str] = []
    if any("total_reward_not_above" in blocker for blocker in blockers):
        hints.append("raise SA reward advantage: expand train_window_count/seeds, inspect event/value losses, and sweep event_logit_temperature_final.")
    if "continuity_below_popularity_tolerance" in blockers or "handoff_failure_above_popularity_tolerance" in blockers:
        hints.append("tighten continuity guard: audit target_mismatch_guard_count and prefer continuity/retained-mechanism checkpoint selection.")
    if "backhaul_above_popularity_tolerance" in blockers:
        hints.append("reduce backhaul cost: lower prepare_action_prior_weight or add backhaul-aware auxiliary penalty in mechanism windows.")
    if "mechanism_realization_and_ready_below_popularity_tolerance" in blockers:
        hints.append("increase mechanism realization: keep mechanism_aux_coef active longer and oversample imminent-handoff windows.")
    if "smoke_checkpoint_warnings_present" in blockers:
        hints.append("retrain every learned baseline under the current non-smoke contract before making paper claims.")
    if not hints and mode_reports:
        hints.append("gate passed; freeze manifest, export tables, and run ablations/robustness/scalability.")
    return hints


def main() -> None:
    args = parse_args()
    settings = effective_settings(args)
    run_id = args.run_id or datetime.now().strftime("top_journal_closed_loop_%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    write_json(
        run_root / "run_config.json",
        {
            "run_id": run_id,
            "quick": bool(args.quick),
            "args": vars(args),
            "settings": settings,
        },
    )

    training_records: list[dict[str, Any]] = []
    manifest_path = run_root / "seed_checkpoint_manifest.json"
    if not args.skip_training:
        for seed in args.seeds:
            sa_record = (
                existing_training_record(run_root=run_root, agent_name="sa_ghmappo", seed=seed)
                if args.resume_training
                else None
            )
            if sa_record is None:
                sa_record = train_sa_for_seed(
                    args=args,
                    settings=settings,
                    seed=seed,
                    run_root=run_root,
                    command_log=command_log,
                )
            else:
                command_log.append(
                    {
                        "label": f"reuse_train_sa_seed_{seed}",
                        "command": [],
                        "returncode": 0,
                        "elapsed_sec": 0.0,
                        "reused_train_summary_path": sa_record["train_summary_path"],
                    }
                )
            training_records.append(sa_record)
            for baseline_agent in args.baseline_agents:
                baseline_record = (
                    existing_training_record(run_root=run_root, agent_name=baseline_agent, seed=seed)
                    if args.resume_training
                    else None
                )
                if baseline_record is None:
                    baseline_record = train_baseline_for_seed(
                        args=args,
                        settings=settings,
                        agent_name=baseline_agent,
                        seed=seed,
                        run_root=run_root,
                        command_log=command_log,
                    )
                else:
                    command_log.append(
                        {
                            "label": f"reuse_train_{baseline_agent}_seed_{seed}",
                            "command": [],
                            "returncode": 0,
                            "elapsed_sec": 0.0,
                            "reused_train_summary_path": baseline_record["train_summary_path"],
                        }
                    )
                training_records.append(baseline_record)
        write_json(manifest_path, build_seed_manifest(training_records))
    elif not manifest_path.exists():
        raise FileNotFoundError(f"--skip_training requires existing manifest: {manifest_path}")

    benchmark_paths: list[Path] = []
    if not args.skip_benchmark:
        for mode in args.benchmark_modes:
            benchmark_paths.append(
                run_benchmark_mode(
                    args=args,
                    settings=settings,
                    mode=mode,
                    run_root=run_root,
                    manifest_path=manifest_path,
                    command_log=command_log,
                )
            )
    else:
        for mode in args.benchmark_modes:
            mode_root = run_root / "benchmarks" / mode
            if mode_root.exists():
                benchmark_paths.append(latest_file(mode_root, "aggregate_summary.json"))

    gate_report = build_gate_report(
        args=args,
        settings=settings,
        run_root=run_root,
        manifest_path=manifest_path,
        training_records=training_records,
        benchmark_paths=benchmark_paths,
        command_log=command_log,
    )
    write_json(run_root / "gate_report.json", gate_report)
    write_gate_csv(run_root / "gate_summary.csv", list(gate_report["mode_reports"]))
    write_json(run_root / "command_log.json", {"commands": command_log})

    print("top-journal closed loop complete")
    print(f"run_root: {run_root}")
    print(f"seed_checkpoint_manifest_path: {manifest_path}")
    print(f"gate_report_path: {run_root / 'gate_report.json'}")
    print(f"paper_claim_ready: {gate_report['paper_claim_ready']}")
    print(f"formal_contract_ready: {gate_report['formal_contract']['ready']}")
    print(f"passed: {gate_report['passed']}")


if __name__ == "__main__":
    main()
