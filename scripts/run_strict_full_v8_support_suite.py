"""Run the strict-full v8 support suite under a frozen window plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT_DIR
    / "artifacts"
    / "experiments"
    / "top_journal_closed_loop"
    / "strict_full_v8_dev_screen_20260621_v2"
    / "seed_checkpoint_manifest.json"
)
DEFAULT_FORMAL_PLAN = (
    ROOT_DIR / "configs" / "experiment" / "top_journal_v8_strict_split_20260621" / "formal_window_plan.json"
)
DEFAULT_GUARD_MANIFEST = ROOT_DIR / "configs" / "ablation_checkpoint_manifest_v8_guard_attribution.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict-full v8 support suite without reopening hidden.")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--seed_checkpoint_manifest_path", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--window_plan_path", type=str, default=str(DEFAULT_FORMAL_PLAN))
    parser.add_argument("--guard_manifest_path", type=str, default=str(DEFAULT_GUARD_MANIFEST))
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "experiments" / "top_journal_support_suite"))
    parser.add_argument("--agents", nargs="+", default=["sa_ghmappo", "ppo", "mappo", "dqn", "dueling_dqn", "qmix", "controller_mat", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl", "popularity_cache_heuristic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29, 41, 53])
    parser.add_argument("--max_mobility_rows", type=int, default=10000)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--window_mode", type=str, default="full_stratified")
    parser.add_argument("--primary_vehicle_selection", type=str, default="handoff_pressure")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def reject_hidden_plan(path_text: str) -> None:
    path = Path(path_text)
    lowered = str(path).lower()
    if "hidden" in lowered:
        raise ValueError(f"support suite cannot use consumed hidden holdout plan: {path}")
    payload = _load_json(path)
    split_name = ""
    if isinstance(payload, dict):
        split_name = str(payload.get("split_name", payload.get("split", ""))).lower()
    if "hidden" in split_name:
        raise ValueError(f"support suite cannot use consumed hidden holdout split: {path}")


def command_common(args: argparse.Namespace, output_root: Path) -> list[str]:
    return [
        "--seed_checkpoint_manifest_path",
        args.seed_checkpoint_manifest_path,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--max_mobility_rows",
        str(args.max_mobility_rows),
        "--max_workflows",
        str(args.max_workflows),
        "--max_steps",
        str(args.max_steps),
        "--window_mode",
        args.window_mode,
        "--window_plan_path",
        args.window_plan_path,
        "--primary_vehicle_selection",
        args.primary_vehicle_selection,
        "--output_root",
        str(output_root),
    ]


def latest_child(path: Path) -> Path:
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        raise FileNotFoundError(f"no run directory created under {path}")
    return max(children, key=lambda child: child.stat().st_mtime)


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "returncode": None, "skipped": True}
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    return {"command": command, "returncode": completed.returncode, "skipped": False}


def statistics_command(
    rows_path: Path,
    output_root: Path,
    candidate_agent: str,
    baseline_agents: list[str],
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/analyze_top_journal_statistics.py",
        "--rows_path",
        str(rows_path),
        "--candidate_agent",
        candidate_agent,
        "--baseline_agents",
        *baseline_agents,
        "--outer_cluster_keys",
        "window_id",
        "--inner_cluster_keys",
        "seed",
        "workflow_id",
        "--ci_method",
        "bca",
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--random_seed",
        "7",
        "--output_root",
        str(output_root),
    ]


def main() -> None:
    args = parse_args()
    reject_hidden_plan(args.window_plan_path)
    run_id = args.run_id or datetime.now().strftime("strict_full_v8_support_%Y%m%d_v1")
    support_root = Path(args.output_root) / run_id
    support_root.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str], Path, str, list[str]]] = []
    common_agents = ["ppo", "mappo", "dqn", "dueling_dqn", "qmix", "controller_mat", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl", "popularity_cache_heuristic"]
    prediction_root = support_root / "prediction_robustness"
    commands.append(
        (
            "prediction_robustness",
            [
                sys.executable,
                "scripts/benchmark_prediction_robustness.py",
                "--agents",
                *args.agents,
                *command_common(args, prediction_root),
            ],
            prediction_root,
            "sa_ghmappo",
            common_agents,
        )
    )
    robustness_root = support_root / "system_robustness"
    commands.append(
        (
            "system_robustness",
            [
                sys.executable,
                "scripts/benchmark_robustness.py",
                "--agents",
                *args.agents,
                *command_common(args, robustness_root),
            ],
            robustness_root,
            "sa_ghmappo",
            common_agents,
        )
    )
    scalability_root = support_root / "scalability"
    commands.append(
        (
            "scalability",
            [
                sys.executable,
                "scripts/benchmark_scalability.py",
                "--agents",
                *args.agents,
                *command_common(args, scalability_root),
            ],
            scalability_root,
            "sa_ghmappo",
            common_agents,
        )
    )
    guard_root = support_root / "guard_attribution"
    commands.append(
        (
            "guard_attribution",
            [
                sys.executable,
                "scripts/benchmark_ablation.py",
                "--manifest_path",
                args.guard_manifest_path,
                "--ablation_labels",
                "learned_core_only",
                "no_guard",
                "cache_warm_guard_only",
                "prefetch_admission_guard_only",
                "backhaul_guard_only",
                "all_guards",
                "--seed_checkpoint_manifest_path",
                args.seed_checkpoint_manifest_path,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--max_mobility_rows",
                str(args.max_mobility_rows),
                "--max_workflows",
                str(args.max_workflows),
                "--max_steps",
                str(args.max_steps),
                "--window_mode",
                args.window_mode,
                "--window_plan_path",
                args.window_plan_path,
                "--primary_vehicle_selection",
                args.primary_vehicle_selection,
                "--output_root",
                str(guard_root),
            ],
            guard_root,
            "all_guards",
            ["learned_core_only", "no_guard", "cache_warm_guard_only", "prefetch_admission_guard_only", "backhaul_guard_only"],
        )
    )

    execution_log: list[dict[str, Any]] = []
    suite_outputs: dict[str, Any] = {}
    for suite_name, command, suite_root, candidate_agent, baselines in commands:
        suite_root.mkdir(parents=True, exist_ok=True)
        result = run_command(command, dry_run=args.dry_run)
        execution_log.append({"suite": suite_name, **result})
        if args.dry_run:
            continue
        if result["returncode"] != 0:
            raise RuntimeError(f"{suite_name} failed with returncode={result['returncode']}")
        run_dir = latest_child(suite_root)
        rows_path = run_dir / "benchmark_rows.csv"
        if not rows_path.exists():
            raise FileNotFoundError(f"{suite_name} did not produce benchmark_rows.csv: {rows_path}")
        stats_root = support_root / "statistics" / suite_name
        stats_result = run_command(
            statistics_command(rows_path, stats_root, candidate_agent, baselines, args),
            dry_run=False,
        )
        execution_log.append({"suite": f"{suite_name}_statistics", **stats_result})
        if stats_result["returncode"] != 0:
            raise RuntimeError(f"{suite_name} statistics failed with returncode={stats_result['returncode']}")
        suite_outputs[suite_name] = {
            "run_dir": str(run_dir),
            "rows_path": str(rows_path),
            "aggregate_summary_path": str(run_dir / "aggregate_summary.json"),
            "statistics_root": str(stats_root),
            "paired_statistics_path": str(stats_root / "paired_statistics.csv"),
        }

    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "support_root": str(support_root),
        "dry_run": bool(args.dry_run),
        "hidden_holdout_reopened": False,
        "window_plan_path": args.window_plan_path,
        "seed_checkpoint_manifest_path": args.seed_checkpoint_manifest_path,
        "guard_manifest_path": args.guard_manifest_path,
        "statistics_protocol": {
            "outer_cluster_keys": ["window_id"],
            "inner_cluster_keys": ["seed", "workflow_id"],
            "ci_method": "bca",
            "bootstrap_samples": args.bootstrap_samples,
            "multiplicity_adjustment": "holm",
        },
        "suite_outputs": suite_outputs,
        "execution_log": execution_log,
        "claim_boundary": [
            "This support suite does not reopen hidden holdout.",
            "Guard attribution uses one loaded checkpoint with agent_config_overrides.",
            "Support-suite completion does not by itself promote v8/v9 to TMC-ready.",
        ],
    }
    (support_root / "support_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"support_gate_report_path: {support_root / 'support_gate_report.json'}")


if __name__ == "__main__":
    main()
