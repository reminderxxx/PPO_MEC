"""Run the top-journal learned-baseline suite.

This suite reuses the frozen formal_v2 SA/PPO manifest, trains any
missing learned baselines under the same real-sample contract, reruns the main
benchmark modes, and gates only against learned baselines. Hand-written
heuristics remain in the benchmark as supplementary reference lines.
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
DEFAULT_BASE_RUN = ROOT_DIR / "artifacts" / "experiments" / "top_journal_closed_loop" / "top_journal_closed_loop_formal_20260505_v2"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "experiments" / "top_journal_learned_baseline_suite"
DEFAULT_WORKFLOW_CSV = ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"
PAPER_GRADE_DEFAULT_LEARNED_AGENTS = [
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
CONTRACT_BLOCKED_LEARNED_BASELINES = {
    "ippo": "single_wrapper_decision_stream_has_no_independent_per_agent_action_surface",
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
FORMAL_MIN_SETTINGS = {
    "baseline_episodes": 48,
    "baseline_update_every": 6,
    "baseline_batch_size": 32,
    "max_mobility_rows": 2500,
    "max_workflows": 2,
    "window_length": 24,
    "window_count": 3,
    "window_scan_stride": 2,
    "max_steps": 16,
    "min_tasks": 5,
    "max_tasks": 20,
}
TRACE_IDENTITY_KEY_FIELDS = ["source_rows_path", "window_id", "scenario_id", "mode", "workflow_id", "seed"]
TRACE_IDENTITY_IGNORE_FIELDS = {
    "agent_name",
    "policy_name",
    "checkpoint_run_id",
    "checkpoint_profile",
    "checkpoint_run_update_count",
    "checkpoint_source_update_index",
    "checkpoint_episode_count",
    "checkpoint_is_smoke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-journal learned-baseline suite")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--base_manifest_path", type=str, default=str(DEFAULT_BASE_RUN / "seed_checkpoint_manifest.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    parser.add_argument(
        "--learned_baseline_agents",
        nargs="+",
        default=PAPER_GRADE_DEFAULT_LEARNED_AGENTS,
    )
    parser.add_argument("--heuristic_reference_agents", nargs="+", default=["reactive_greedy", "popularity_cache_heuristic"])
    parser.add_argument("--benchmark_modes", nargs="+", default=["mixed_informative", "full_stratified"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--resume_benchmark", action="store_true")
    parser.add_argument("--command_retries", type=int, default=0)
    parser.add_argument("--baseline_profile", type=str, default="baseline_safe")
    parser.add_argument("--mappo_baseline_profile", type=str, default="mappo_strong_audit")
    parser.add_argument(
        "--force_retrain_agents",
        nargs="*",
        default=[],
        help="Retrain these learned baselines even if base_manifest_path already contains checkpoints.",
    )
    parser.add_argument(
        "--force_retrain_all_learned",
        action="store_true",
        help="Retrain every learned baseline in learned_baseline_agents.",
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(DEFAULT_WORKFLOW_CSV))
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="handoff_pressure")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_mode_for_training", type=str, default="mixed_informative")
    parser.add_argument("--window_rank_offset", type=int, default=0)
    parser.add_argument("--baseline_episodes", type=int, default=48)
    parser.add_argument("--baseline_update_every", type=int, default=6)
    parser.add_argument("--baseline_batch_size", type=int, default=32)
    parser.add_argument("--max_mobility_rows", type=int, default=2500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_count", type=int, default=3)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--reward_tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--minimum_reward_delta",
        type=float,
        default=0.0,
        help="Optional required SA reward margin over every learned baseline in each benchmark mode.",
    )
    parser.add_argument(
        "--statistics_cluster_keys",
        nargs="*",
        default=[],
        help="Legacy one-level cluster bootstrap keys passed to analyze_top_journal_statistics.py.",
    )
    parser.add_argument(
        "--statistics_outer_cluster_keys",
        nargs="*",
        default=["window_id"],
        help="Outer hierarchical bootstrap keys; strict mobility claims default to window_id.",
    )
    parser.add_argument(
        "--statistics_inner_cluster_keys",
        nargs="*",
        default=["seed", "workflow_id"],
        help="Inner hierarchical bootstrap keys nested within each outer mobility window.",
    )
    parser.add_argument(
        "--allow_contract_blocked_baselines",
        action="store_true",
        help=(
            "Allow diagnostic contract-blocked baselines such as IPPO to run. "
            "They remain hard blockers for paper_claim_ready."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def existing_manifest_path(manifest: dict[str, dict[str, str]], agent_name: str, seed: int) -> str:
    path_text = str(manifest.get(agent_name, {}).get(str(seed), ""))
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path) if path.exists() else ""


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


def existing_training_record(run_root: Path, agent_name: str, seed: int) -> dict[str, Any] | None:
    summary_root = run_root / "training" / "algo_pool" / agent_name
    summary_path = latest_training_summary_for_seed(summary_root, "train_summary.json", seed)
    if summary_path is None:
        return None
    summary = read_json(summary_path)
    checkpoint_path = str(summary.get("latest_checkpoint_path", ""))
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    return {
        "agent_name": agent_name,
        "seed": seed,
        "train_summary_path": str(summary_path),
        "selected_checkpoint_path": checkpoint_path,
        "selection_field": "latest_checkpoint_path",
        "trained_by_suite": True,
        "resumed_from_existing": True,
    }


def run_command(
    *,
    label: str,
    cmd: list[str],
    command_log: list[dict[str, Any]],
    retries: int = 0,
) -> None:
    max_attempts = max(1, retries + 1)
    last_returncode = 0
    for attempt_index in range(1, max_attempts + 1):
        started = time.time()
        attempt_label = label if max_attempts == 1 else f"{label}_attempt_{attempt_index}"
        print(f"[learned-suite] start {attempt_label}")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
        last_returncode = completed.returncode
        elapsed_sec = round(time.time() - started, 3)
        command_log.append(
            {
                "label": label,
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "command": cmd,
                "returncode": completed.returncode,
                "elapsed_sec": elapsed_sec,
            }
        )
        print(f"[learned-suite] finish {attempt_label}: returncode={completed.returncode}, elapsed_sec={elapsed_sec}")
        if completed.returncode == 0:
            return
    raise RuntimeError(f"command failed for {label}: returncode={last_returncode}")


def common_real_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--mobility_source",
        args.mobility_source,
        "--primary_vehicle_selection",
        args.primary_vehicle_selection,
        "--workflow_csv_path",
        args.workflow_csv_path,
        "--max_mobility_rows",
        str(args.max_mobility_rows),
        "--max_workflows",
        str(args.max_workflows),
        "--workflow_selector",
        args.workflow_selector,
        "--rsu_layout",
        args.rsu_layout,
        "--window_selector",
        args.window_selector,
        "--window_length",
        str(args.window_length),
        "--window_scan_stride",
        str(args.window_scan_stride),
        "--min_tasks",
        str(args.min_tasks),
        "--max_tasks",
        str(args.max_tasks),
    ]
    if args.mobility_csv_path:
        common.extend(["--mobility_csv_path", args.mobility_csv_path])
    return common


def train_missing_baseline(
    *,
    args: argparse.Namespace,
    run_root: Path,
    agent_name: str,
    seed: int,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    training_root = run_root / "training" / "algo_pool"
    profile = args.mappo_baseline_profile if agent_name == "mappo" else args.baseline_profile
    cmd = [
        args.python_executable,
        "scripts/train_algo_pool_real_sample.py",
        "--agent_name",
        agent_name,
        "--profile",
        profile,
        "--episodes",
        str(args.baseline_episodes),
        "--update_every",
        str(args.baseline_update_every),
        "--batch_size",
        str(args.baseline_batch_size),
        "--max_steps",
        str(args.max_steps),
        "--window_count",
        str(args.window_count),
        "--window_mode",
        args.window_mode_for_training,
        "--random_seed",
        str(seed),
        "--output_root",
        str(training_root),
        *common_real_args(args),
    ]
    run_command(label=f"train_{agent_name}_seed_{seed}", cmd=cmd, command_log=command_log, retries=args.command_retries)
    summary_path = latest_training_summary_for_seed(training_root / agent_name, "train_summary.json", seed)
    if summary_path is None:
        raise FileNotFoundError(f"missing train_summary.json for {agent_name} seed {seed}")
    summary = read_json(summary_path)
    checkpoint_path = str(summary.get("latest_checkpoint_path", ""))
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"missing latest checkpoint for {agent_name} seed {seed}: {checkpoint_path}")
    return {
        "agent_name": agent_name,
        "seed": seed,
        "train_summary_path": str(summary_path),
        "selected_checkpoint_path": checkpoint_path,
        "selection_field": "latest_checkpoint_path",
        "trained_by_suite": True,
    }


def unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def audit_learned_baseline_contract(
    learned_agents: list[str],
    *,
    diagnostic_allowed: bool,
) -> dict[str, Any]:
    blocked = [
        {
            "agent_name": agent_name,
            "reason": CONTRACT_BLOCKED_LEARNED_BASELINES[agent_name],
        }
        for agent_name in learned_agents
        if agent_name in CONTRACT_BLOCKED_LEARNED_BASELINES
    ]
    blockers = [
        f"contract_blocked_baseline:{item['agent_name']}:{item['reason']}"
        for item in blocked
    ]
    return {
        "passed": not blockers,
        "blockers": blockers,
        "diagnostic_allowed": bool(diagnostic_allowed),
        "paper_grade_default_learned_agents": PAPER_GRADE_DEFAULT_LEARNED_AGENTS,
        "contract_blocked_agents": blocked,
        "policy": (
            "Contract-blocked learned baselines may be used only for diagnostics; "
            "they cannot contribute to paper_claim_ready under the current wrapper."
        ),
    }


def run_benchmark(
    *,
    args: argparse.Namespace,
    run_root: Path,
    manifest_path: Path,
    mode: str,
    command_log: list[dict[str, Any]],
) -> Path:
    agents = unique_ordered(["sa_ghmappo", *args.learned_baseline_agents, *args.heuristic_reference_agents])
    output_root = run_root / "benchmarks" / mode
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
        str(args.max_steps),
        "--window_count",
        str(args.window_count),
        "--window_mode",
        mode,
        "--window_rank_offset",
        str(args.window_rank_offset),
        "--output_root",
        str(output_root),
        *common_real_args(args),
    ]
    run_command(label=f"benchmark_{mode}", cmd=cmd, command_log=command_log, retries=args.command_retries)
    return latest_file(output_root, "aggregate_summary.json")


def run_statistics(
    *,
    args: argparse.Namespace,
    run_root: Path,
    benchmark_paths: list[Path],
    command_log: list[dict[str, Any]],
) -> Path:
    rows_paths = [path.parent / "benchmark_rows.csv" for path in benchmark_paths]
    output_root = run_root / "statistics" / "learned_main_results"
    cmd = [
        args.python_executable,
        "scripts/analyze_top_journal_statistics.py",
        *(item for path in rows_paths for item in ["--rows_path", str(path)]),
        "--candidate_agent",
        "sa_ghmappo",
        "--baseline_agents",
        *args.learned_baseline_agents,
        "--output_root",
        str(output_root),
    ]
    if args.statistics_cluster_keys:
        cmd.extend(["--cluster_keys", *args.statistics_cluster_keys])
    elif args.statistics_outer_cluster_keys:
        cmd.extend(["--outer_cluster_keys", *args.statistics_outer_cluster_keys])
        if args.statistics_inner_cluster_keys:
            cmd.extend(["--inner_cluster_keys", *args.statistics_inner_cluster_keys])
    run_command(label="learned_paired_statistics", cmd=cmd, command_log=command_log, retries=args.command_retries)
    return output_root / "paired_statistics.csv"


def metric_mean(summary: dict[str, Any], agent_name: str, metric_name: str) -> float | None:
    metric = (
        summary.get("aggregate_by_agent", {})
        .get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
    )
    if not isinstance(metric, dict) or "mean" not in metric:
        return None
    return float(metric["mean"])


def row_identity_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field_name, "")) for field_name in TRACE_IDENTITY_KEY_FIELDS)


def row_trace_signature(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (field_name, str(value))
            for field_name, value in row.items()
            if field_name not in TRACE_IDENTITY_IGNORE_FIELDS
        )
    )


def audit_duplicate_baseline_traces(
    *,
    benchmark_paths: list[Path],
    learned_agents: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for aggregate_path in benchmark_paths:
        rows_path = aggregate_path.parent / "benchmark_rows.csv"
        if not rows_path.exists():
            continue
        with rows_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["source_rows_path"] = str(rows_path)
                rows.append(row)
    by_key: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        agent_name = str(row.get("agent_name", ""))
        if agent_name not in learned_agents:
            continue
        by_key.setdefault(row_identity_key(row), {})[agent_name] = row

    duplicate_pairs: list[dict[str, Any]] = []
    for left_index, left_agent in enumerate(learned_agents):
        for right_agent in learned_agents[left_index + 1 :]:
            comparable_count = 0
            identical_count = 0
            for agents_by_key in by_key.values():
                left_row = agents_by_key.get(left_agent)
                right_row = agents_by_key.get(right_agent)
                if left_row is None or right_row is None:
                    continue
                comparable_count += 1
                if row_trace_signature(left_row) == row_trace_signature(right_row):
                    identical_count += 1
            if comparable_count > 0 and identical_count == comparable_count:
                duplicate_pairs.append(
                    {
                        "left_agent": left_agent,
                        "right_agent": right_agent,
                        "comparable_count": comparable_count,
                        "identical_count": identical_count,
                    }
                )
    blockers = [
        f"duplicate_benchmark_trace:{item['left_agent']}:{item['right_agent']}:n={item['identical_count']}"
        for item in duplicate_pairs
    ]
    return {
        "passed": not blockers,
        "blockers": blockers,
        "duplicate_pairs": duplicate_pairs,
        "identity_key_fields": TRACE_IDENTITY_KEY_FIELDS,
        "ignored_fields": sorted(TRACE_IDENTITY_IGNORE_FIELDS),
    }


def audit_formal_contract(args: argparse.Namespace, manifest: dict[str, dict[str, str]]) -> dict[str, Any]:
    blockers: list[str] = []
    if args.primary_vehicle_selection != "handoff_pressure":
        blockers.append(f"primary_vehicle_selection_not_handoff_pressure:{args.primary_vehicle_selection}")
    if len(set(int(seed) for seed in args.seeds)) < 3:
        blockers.append("fewer_than_3_seeds")
    for key, minimum in FORMAL_MIN_SETTINGS.items():
        actual = int(getattr(args, key))
        if actual < minimum:
            blockers.append(f"{key}_below_formal_min:{actual}<{minimum}")
    for agent_name in ["sa_ghmappo", *args.learned_baseline_agents]:
        for seed in args.seeds:
            checkpoint_path = existing_manifest_path(manifest, agent_name, seed)
            if not checkpoint_path:
                blockers.append(f"missing_checkpoint:{agent_name}:seed{seed}")
                continue
            blockers.extend(checkpoint_protocol_blockers(agent_name, checkpoint_path, seed))
    return {
        "ready": not blockers,
        "blockers": blockers,
        "required_primary_vehicle_selection": "handoff_pressure",
        "min_settings": FORMAL_MIN_SETTINGS,
        "baseline_protocol_versions": CURRENT_BASELINE_PROTOCOLS,
    }


def build_gate_report(
    *,
    args: argparse.Namespace,
    run_root: Path,
    manifest_path: Path,
    manifest: dict[str, dict[str, str]],
    training_records: list[dict[str, Any]],
    benchmark_paths: list[Path],
    statistics_path: Path | None,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    formal_contract = audit_formal_contract(args, manifest)
    mode_reports: list[dict[str, Any]] = []
    all_blockers: list[str] = []
    baseline_contract = audit_learned_baseline_contract(
        args.learned_baseline_agents,
        diagnostic_allowed=args.allow_contract_blocked_baselines,
    )
    all_blockers.extend(str(item) for item in baseline_contract["blockers"])
    baseline_independence = audit_duplicate_baseline_traces(
        benchmark_paths=benchmark_paths,
        learned_agents=args.learned_baseline_agents,
    )
    all_blockers.extend(str(item) for item in baseline_independence["blockers"])
    for aggregate_path in benchmark_paths:
        summary = read_json(aggregate_path)
        mode = str(summary.get("window_mode", aggregate_path.parent.name))
        metrics = {
            "total_reward": {
                agent_name: metric_mean(summary, agent_name, "total_reward")
                for agent_name in unique_ordered(["sa_ghmappo", *args.learned_baseline_agents, *args.heuristic_reference_agents])
            },
            "workflow_continuity_rate": {
                agent_name: metric_mean(summary, agent_name, "workflow_continuity_rate")
                for agent_name in unique_ordered(["sa_ghmappo", *args.learned_baseline_agents, *args.heuristic_reference_agents])
            },
            "handoff_failure_rate": {
                agent_name: metric_mean(summary, agent_name, "handoff_failure_rate")
                for agent_name in unique_ordered(["sa_ghmappo", *args.learned_baseline_agents, *args.heuristic_reference_agents])
            },
        }
        blockers = list(summary.get("smoke_checkpoint_warnings", []) or [])
        sa_reward = metrics["total_reward"].get("sa_ghmappo")
        strongest_agent = ""
        strongest_reward: float | None = None
        for baseline_agent in args.learned_baseline_agents:
            baseline_reward = metrics["total_reward"].get(baseline_agent)
            if baseline_reward is not None and (
                strongest_reward is None or baseline_reward > strongest_reward
            ):
                strongest_reward = baseline_reward
                strongest_agent = baseline_agent
            if sa_reward is None or baseline_reward is None:
                blockers.append(f"missing_reward:{baseline_agent}")
            elif sa_reward <= baseline_reward + max(args.reward_tolerance, args.minimum_reward_delta):
                blockers.append(f"sa_total_reward_margin_below_{baseline_agent}")
        mode_report = {
            "mode": mode,
            "aggregate_summary_path": str(aggregate_path),
            "passed": not blockers,
            "blockers": blockers,
            "metrics": metrics,
            "strongest_learned_baseline": strongest_agent,
            "strongest_learned_baseline_reward": strongest_reward,
            "sa_total_reward": sa_reward,
            "sa_minus_strongest_learned_reward": (
                round(float(sa_reward - strongest_reward), 6)
                if sa_reward is not None and strongest_reward is not None
                else None
            ),
            "episode_count": summary.get("episode_count", 0),
            "canonical_paper_protocol": bool(summary.get("canonical_paper_protocol", False)),
            "window_rank_offset": summary.get("window_rank_offset", args.window_rank_offset),
        }
        if not mode_report["canonical_paper_protocol"]:
            mode_report["blockers"].append("not_canonical_paper_protocol")
            mode_report["passed"] = False
        all_blockers.extend(str(item) for item in mode_report["blockers"])
        mode_reports.append(mode_report)

    suite_passed = (
        bool(mode_reports)
        and all(report["passed"] for report in mode_reports)
        and bool(baseline_contract["passed"])
        and bool(baseline_independence["passed"])
    )
    return {
        "run_id": run_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gate_policy": "learned_baseline_strict",
        "formal_contract": formal_contract,
        "paper_claim_ready": bool(suite_passed and formal_contract["ready"]),
        "passed": suite_passed,
        "support_suite_complete": bool(suite_passed and statistics_path and statistics_path.exists()),
        "seeds": args.seeds,
        "learned_baseline_agents": args.learned_baseline_agents,
        "baseline_contract": baseline_contract,
        "heuristic_reference_agents": args.heuristic_reference_agents,
        "benchmark_modes": args.benchmark_modes,
        "window_rank_offset": args.window_rank_offset,
        "baseline_profile": args.baseline_profile,
        "mappo_baseline_profile": args.mappo_baseline_profile,
        "baseline_episodes": args.baseline_episodes,
        "baseline_protocol_versions": CURRENT_BASELINE_PROTOCOLS,
        "minimum_reward_delta": args.minimum_reward_delta,
        "budget_protocol": {
            "policy": "matched_environment_interaction_budget",
            "baseline_update_rule": (
                "Every learned baseline uses the same seeds, NGSIM+Alibaba windows, workflow selector, "
                "episode count, max steps, and benchmark/support suites. Architecture-specific update "
                "rules are allowed, but no baseline receives SA-GHMAPPO-only graph/surrogate/guard mechanisms."
            ),
            "reference_basis": [
                "PPO-style on-policy baselines follow matched rollout budgets.",
                "Controller-level MAPPO must use aggregation-reason controller head-credit v3 under the current multi-controller action contract.",
                "Replay/value-decomposition baselines follow matched environment interactions rather than copied Atari frame counts.",
                "Transformer/MARL baselines are adapted at the controller-agent level under the same action contract.",
                "Domain VEC baselines for DAG offloading, model cache/offloading, and digital-twin handoff use the same NGSIM+Alibaba interaction budget.",
            ],
        },
        "statistics_cluster_keys": args.statistics_cluster_keys,
        "statistics_outer_cluster_keys": args.statistics_outer_cluster_keys,
        "statistics_inner_cluster_keys": args.statistics_inner_cluster_keys,
        "seed_checkpoint_manifest_path": str(manifest_path),
        "training_records": training_records,
        "mode_reports": mode_reports,
        "baseline_independence": baseline_independence,
        "statistics_path": str(statistics_path) if statistics_path is not None else "",
        "blockers": sorted(set(all_blockers + formal_contract["blockers"])),
        "claim_boundary": [
            "Main learned-baseline claim is gated against learned baselines only.",
            "Heuristic agents are supplementary reference lines, not the top-journal primary comparator.",
            "Continuous-control baselines remain blocked by the current semantic_discrete_5 action contract.",
            "MAPPO claims require the current controller head-credit v3 checkpoint protocol; pre-v3/pre-head-credit MAPPO checkpoints are archived only.",
            "When force_retrain_* is used, learned baselines in the manifest are refreshed under the current suite budget.",
        ],
        "command_log": command_log,
    }


def write_gate_csv(path: Path, mode_reports: list[dict[str, Any]]) -> None:
    rows = [
        {
            "mode": report.get("mode", ""),
            "passed": report.get("passed", False),
            "blockers": ";".join(report.get("blockers", [])),
            "episode_count": report.get("episode_count", 0),
            "sa_total_reward": report.get("sa_total_reward"),
            "strongest_learned_baseline": report.get("strongest_learned_baseline"),
            "strongest_learned_baseline_reward": report.get("strongest_learned_baseline_reward"),
            "sa_minus_strongest_learned_reward": report.get("sa_minus_strongest_learned_reward"),
        }
        for report in mode_reports
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_contract = audit_learned_baseline_contract(
        args.learned_baseline_agents,
        diagnostic_allowed=args.allow_contract_blocked_baselines,
    )
    if baseline_contract["blockers"] and not args.allow_contract_blocked_baselines:
        blockers = "; ".join(str(item) for item in baseline_contract["blockers"])
        raise SystemExit(
            "contract-blocked learned baselines cannot be used in the paper-grade gate. "
            f"Use {PAPER_GRADE_DEFAULT_LEARNED_AGENTS} for the current wrapper, or pass "
            f"--allow_contract_blocked_baselines for diagnostic-only reruns. blockers={blockers}"
        )
    run_id = args.run_id or datetime.now().strftime("top_journal_learned_baseline_%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    base_manifest = read_json(Path(args.base_manifest_path))
    augmented_manifest: dict[str, dict[str, str]] = {
        str(agent_name): {str(seed): str(path) for seed, path in seed_map.items()}
        for agent_name, seed_map in base_manifest.items()
    }
    training_records: list[dict[str, Any]] = []

    if not args.skip_training:
        force_retrain_agents = set(args.force_retrain_agents)
        if args.force_retrain_all_learned:
            force_retrain_agents.update(args.learned_baseline_agents)
        for agent_name in args.learned_baseline_agents:
            for seed in args.seeds:
                existing = "" if agent_name in force_retrain_agents else existing_manifest_path(augmented_manifest, agent_name, seed)
                if existing:
                    training_records.append(
                        {
                            "agent_name": agent_name,
                            "seed": seed,
                            "selected_checkpoint_path": existing,
                            "trained_by_suite": False,
                            "source": "base_manifest",
                        }
                    )
                    continue
                if args.resume_training:
                    resumed = existing_training_record(run_root, agent_name, seed)
                    if resumed is not None:
                        augmented_manifest.setdefault(agent_name, {})[str(seed)] = resumed["selected_checkpoint_path"]
                        training_records.append(resumed)
                        command_log.append(
                            {
                                "label": f"reuse_train_{agent_name}_seed_{seed}",
                                "command": [],
                                "returncode": 0,
                                "elapsed_sec": 0.0,
                                "reused_train_summary_path": resumed["train_summary_path"],
                            }
                        )
                        continue
                if agent_name in CONTRACT_BLOCKED_LEARNED_BASELINES:
                    raise RuntimeError(
                        f"cannot train diagnostic contract-blocked baseline {agent_name!r} under the current wrapper: "
                        f"{CONTRACT_BLOCKED_LEARNED_BASELINES[agent_name]}. Provide an existing checkpoint and "
                        "use --allow_contract_blocked_baselines for diagnostic-only benchmark reruns, or remove it "
                        "from --learned_baseline_agents for paper-grade runs."
                    )
                record = train_missing_baseline(
                    args=args,
                    run_root=run_root,
                    agent_name=agent_name,
                    seed=seed,
                    command_log=command_log,
                )
                augmented_manifest.setdefault(agent_name, {})[str(seed)] = record["selected_checkpoint_path"]
                training_records.append(record)

    manifest_path = run_root / "seed_checkpoint_manifest_learned_baselines.json"
    write_json(manifest_path, augmented_manifest)

    benchmark_paths: list[Path] = []
    if not args.skip_benchmark:
        for mode in args.benchmark_modes:
            if args.resume_benchmark:
                try:
                    benchmark_paths.append(latest_file(run_root / "benchmarks" / mode, "aggregate_summary.json"))
                    command_log.append(
                        {
                            "label": f"reuse_benchmark_{mode}",
                            "command": [],
                            "returncode": 0,
                            "elapsed_sec": 0.0,
                        }
                    )
                    continue
                except FileNotFoundError:
                    pass
            benchmark_paths.append(
                run_benchmark(
                    args=args,
                    run_root=run_root,
                    manifest_path=manifest_path,
                    mode=mode,
                    command_log=command_log,
                )
            )
    else:
        for mode in args.benchmark_modes:
            benchmark_paths.append(latest_file(run_root / "benchmarks" / mode, "aggregate_summary.json"))

    statistics_path: Path | None = None
    if benchmark_paths:
        statistics_path = run_statistics(
            args=args,
            run_root=run_root,
            benchmark_paths=benchmark_paths,
            command_log=command_log,
        )

    gate_report = build_gate_report(
        args=args,
        run_root=run_root,
        manifest_path=manifest_path,
        manifest=augmented_manifest,
        training_records=training_records,
        benchmark_paths=benchmark_paths,
        statistics_path=statistics_path,
        command_log=command_log,
    )
    write_json(run_root / "learned_baseline_gate_report.json", gate_report)
    write_gate_csv(run_root / "learned_baseline_gate_summary.csv", list(gate_report["mode_reports"]))
    write_json(run_root / "command_log.json", {"commands": command_log})
    write_json(run_root / "run_config.json", {"args": vars(args), "run_id": run_id})

    print("top-journal learned-baseline suite complete")
    print(f"run_root: {run_root}")
    print(f"seed_checkpoint_manifest_path: {manifest_path}")
    print(f"learned_baseline_gate_report_path: {run_root / 'learned_baseline_gate_report.json'}")
    print(f"paper_claim_ready: {gate_report['paper_claim_ready']}")
    print(f"formal_contract_ready: {gate_report['formal_contract']['ready']}")
    print(f"passed: {gate_report['passed']}")


if __name__ == "__main__":
    main()
