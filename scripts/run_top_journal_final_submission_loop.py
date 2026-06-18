"""Run the final-submission experiment loop.

This script treats learned baselines as the primary paper gate and keeps
hand-written heuristics as supplementary reference lines. It composes existing
training, benchmark, statistics, robustness, and scalability entrypoints into a
single auditable loop.
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


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MANIFEST = (
    ROOT_DIR
    / "artifacts"
    / "experiments"
    / "top_journal_closed_loop"
    / "top_journal_closed_loop_formal_20260505_v2"
    / "seed_checkpoint_manifest.json"
)
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "experiments" / "top_journal_final_submission"
DEFAULT_WORKFLOW_CSV = ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"
DEFAULT_LEARNED_AGENTS = [
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
DEFAULT_HEURISTIC_REFERENCES = ["reactive_greedy", "popularity_cache_heuristic"]
DEFAULT_SUPPORT_AGENTS = ["sa_ghmappo", *DEFAULT_LEARNED_AGENTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final-submission learned-baseline experiment loop.")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--base_manifest_path", type=str, default=str(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--quick", action="store_true", help="Small chain validation; never paper-ready.")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--resume_benchmark", action="store_true")
    parser.add_argument("--force_retrain_learned", action="store_true")
    parser.add_argument("--skip_support", action="store_true")
    parser.add_argument("--resume_support", action="store_true")
    parser.add_argument("--fail_on_blocker", action="store_true")
    parser.add_argument("--command_retries", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--holdout_offsets", nargs="*", type=int, default=[3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    parser.add_argument("--learned_baseline_agents", nargs="+", default=DEFAULT_LEARNED_AGENTS)
    parser.add_argument("--heuristic_reference_agents", nargs="+", default=DEFAULT_HEURISTIC_REFERENCES)
    parser.add_argument("--support_agents", nargs="+", default=DEFAULT_SUPPORT_AGENTS)
    parser.add_argument(
        "--allow_contract_blocked_baselines",
        action="store_true",
        help="Allow diagnostic-only contract-blocked reruns such as IPPO; final paper_claim_ready remains blocked by the learned suite.",
    )
    parser.add_argument("--prediction_required_settings", nargs="*", default=["learned_prediction", "noisy_prediction"])
    parser.add_argument("--benchmark_modes", nargs="+", default=["mixed_informative", "full_stratified"])
    parser.add_argument("--minimum_reward_delta", type=float, default=0.5)
    parser.add_argument("--cluster_keys", nargs="*", default=["seed", "window_id", "workflow_id"])
    parser.add_argument("--workflow_csv_path", type=str, default=str(DEFAULT_WORKFLOW_CSV))
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="handoff_pressure")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_mode_for_training", type=str, default="mixed_informative")
    parser.add_argument("--baseline_profile", type=str, default="baseline_safe")
    parser.add_argument("--mappo_baseline_profile", type=str, default="mappo_strong_audit")
    parser.add_argument("--baseline_episodes", type=int, default=96)
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
    return parser.parse_args()


def effective_settings(args: argparse.Namespace) -> dict[str, int]:
    if not args.quick:
        return {
            "baseline_episodes": args.baseline_episodes,
            "baseline_update_every": args.baseline_update_every,
            "baseline_batch_size": args.baseline_batch_size,
            "max_mobility_rows": args.max_mobility_rows,
            "max_workflows": args.max_workflows,
            "window_length": args.window_length,
            "window_count": args.window_count,
            "window_scan_stride": args.window_scan_stride,
            "max_steps": args.max_steps,
            "min_tasks": args.min_tasks,
            "max_tasks": args.max_tasks,
        }
    return {
        "baseline_episodes": 2,
        "baseline_update_every": 1,
        "baseline_batch_size": 2,
        "max_mobility_rows": 500,
        "max_workflows": 1,
        "window_length": 8,
        "window_count": 1,
        "window_scan_stride": 4,
        "max_steps": 2,
        "min_tasks": 5,
        "max_tasks": 10,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(label: str, cmd: list[str], command_log: list[dict[str, Any]], retries: int = 0) -> None:
    max_attempts = max(1, retries + 1)
    last_returncode = 0
    for attempt_index in range(1, max_attempts + 1):
        started = time.time()
        attempt_label = label if max_attempts == 1 else f"{label}_attempt_{attempt_index}"
        print(f"[final-loop] start {attempt_label}")
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
        print(f"[final-loop] finish {attempt_label}: returncode={completed.returncode}, elapsed_sec={elapsed_sec}")
        if completed.returncode == 0:
            return
    raise RuntimeError(f"command failed for {label}: returncode={last_returncode}")


def common_real_args(
    args: argparse.Namespace,
    settings: dict[str, int],
    *,
    include_window_selector: bool = True,
) -> list[str]:
    payload = [
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
        "--window_length",
        str(settings["window_length"]),
        "--window_scan_stride",
        str(settings["window_scan_stride"]),
        "--min_tasks",
        str(settings["min_tasks"]),
        "--max_tasks",
        str(settings["max_tasks"]),
    ]
    if include_window_selector:
        insert_at = payload.index("--window_length")
        payload[insert_at:insert_at] = ["--window_selector", args.window_selector]
    if args.mobility_csv_path:
        payload.extend(["--mobility_csv_path", args.mobility_csv_path])
    return payload


def latest_child_report(run_root: Path, filename: str) -> Path:
    candidates = [path for path in run_root.rglob(filename) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"could not find {filename} under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_learned_suite(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    run_root: Path,
    suite_run_id: str,
    base_manifest_path: Path,
    window_rank_offset: int,
    skip_training: bool,
    force_retrain: bool,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    cmd = [
        args.python_executable,
        "scripts/run_top_journal_learned_baseline_suite.py",
        "--run_id",
        suite_run_id,
        "--base_manifest_path",
        str(base_manifest_path),
        "--output_root",
        str(run_root / "learned_suites"),
        "--learned_baseline_agents",
        *args.learned_baseline_agents,
        "--heuristic_reference_agents",
        *args.heuristic_reference_agents,
        "--benchmark_modes",
        *args.benchmark_modes,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--baseline_profile",
        args.baseline_profile,
        "--mappo_baseline_profile",
        args.mappo_baseline_profile,
        "--baseline_episodes",
        str(settings["baseline_episodes"]),
        "--baseline_update_every",
        str(settings["baseline_update_every"]),
        "--baseline_batch_size",
        str(settings["baseline_batch_size"]),
        "--max_steps",
        str(settings["max_steps"]),
        "--window_count",
        str(settings["window_count"]),
        "--window_mode_for_training",
        args.window_mode_for_training,
        "--window_rank_offset",
        str(window_rank_offset),
        "--minimum_reward_delta",
        str(args.minimum_reward_delta),
        "--statistics_cluster_keys",
        *args.cluster_keys,
        "--command_retries",
        str(args.command_retries),
        *common_real_args(args, settings),
    ]
    if skip_training:
        cmd.append("--skip_training")
    if args.resume_training and not skip_training:
        cmd.append("--resume_training")
    if args.resume_benchmark:
        cmd.append("--resume_benchmark")
    if force_retrain:
        cmd.append("--force_retrain_all_learned")
    if args.allow_contract_blocked_baselines:
        cmd.append("--allow_contract_blocked_baselines")
    run_command(f"learned_suite_offset_{window_rank_offset}", cmd, command_log, retries=args.command_retries)
    suite_root = run_root / "learned_suites" / suite_run_id
    report_path = suite_root / "learned_baseline_gate_report.json"
    report = read_json(report_path)
    report["report_path"] = str(report_path)
    report["suite_root"] = str(suite_root)
    return report


def run_support_benchmark(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    run_root: Path,
    manifest_path: Path,
    kind: str,
    command_log: list[dict[str, Any]],
) -> Path:
    script_by_kind = {
        "prediction": "scripts/benchmark_prediction_robustness.py",
        "robustness": "scripts/benchmark_robustness.py",
        "scalability": "scripts/benchmark_scalability.py",
    }
    output_root = run_root / "support" / kind
    cmd = [
        args.python_executable,
        script_by_kind[kind],
        "--agents",
        *args.support_agents,
        "--seed_checkpoint_manifest_path",
        str(manifest_path),
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--max_steps",
        str(settings["max_steps"]),
        "--window_count",
        str(settings["window_count"]),
        "--window_mode",
        "mixed_informative",
        "--output_root",
        str(output_root),
        *common_real_args(args, settings, include_window_selector=kind == "prediction"),
    ]
    run_command(f"support_{kind}", cmd, command_log, retries=args.command_retries)
    filename = "prediction_robustness_summary.json" if kind == "prediction" else "aggregate_summary.json"
    return latest_child_report(output_root, filename)


def metric_mean(summary: dict[str, Any], agent_name: str, metric_name: str) -> float | None:
    value = (
        summary.get("aggregate_by_agent", {})
        .get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
        .get("mean")
    )
    return float(value) if value is not None else None


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_statistics_ci(report: dict[str, Any], learned_agents: list[str]) -> list[str]:
    blockers: list[str] = []
    statistics_path = report.get("statistics_path", "")
    if not statistics_path or not Path(statistics_path).exists():
        return ["missing_learned_statistics"]
    rows = load_csv_rows(statistics_path)
    seen: set[str] = set()
    for row in rows:
        if row.get("metric") != "total_reward":
            continue
        baseline = str(row.get("baseline_agent", ""))
        if baseline not in learned_agents:
            continue
        seen.add(baseline)
        ci_low = float(row.get("ci95_low", 0.0) or 0.0)
        if ci_low <= 0.0:
            blockers.append(f"cluster_ci_not_positive:{baseline}:ci95_low={ci_low}")
    missing = sorted(set(learned_agents) - seen)
    blockers.extend(f"missing_total_reward_stat:{agent}" for agent in missing)
    return blockers


def audit_formal_training_provenance(
    *,
    formal_report: dict[str, Any],
    learned_agents: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Require formal learned checkpoints to be trained by this final suite."""

    records = list(formal_report.get("training_records", []) or [])
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        agent = str(record.get("agent_name", ""))
        seed = str(record.get("seed", ""))
        if agent and seed:
            by_pair[(agent, seed)] = record

    missing: list[str] = []
    reused_external: list[str] = []
    for agent in learned_agents:
        for seed in seeds:
            key = (agent, str(seed))
            record = by_pair.get(key)
            if record is None:
                missing.append(f"{agent}:seed{seed}")
                continue
            if not bool(record.get("trained_by_suite", False)):
                reused_external.append(f"{agent}:seed{seed}:{record.get('source', 'unknown')}")

    blockers = [
        *(f"missing_formal_training_record:{item}" for item in missing),
        *(f"external_checkpoint_reused_in_formal:{item}" for item in reused_external),
    ]
    return {
        "passed": not blockers,
        "blockers": blockers,
        "required_agents": learned_agents,
        "required_seeds": seeds,
        "record_count": len(records),
        "policy": "Final paper-ready gate requires formal learned checkpoints to be trained by this final suite or resumed from the same run.",
    }


def audit_support_summary(
    *,
    summary_path: Path,
    kind: str,
    support_agents: list[str],
    prediction_required_settings: list[str],
) -> list[str]:
    summary = read_json(summary_path)
    blockers: list[str] = []
    baselines = [agent for agent in support_agents if agent != "sa_ghmappo"]
    sa_reward = metric_mean(summary, "sa_ghmappo", "total_reward")
    for baseline in baselines:
        baseline_reward = metric_mean(summary, baseline, "total_reward")
        if sa_reward is None or baseline_reward is None or sa_reward <= baseline_reward:
            blockers.append(f"{kind}_sa_not_above_{baseline}")
    if kind == "prediction":
        by_setting = summary.get("aggregate_by_setting_and_agent", {})
        setting_agent: dict[str, dict[str, float]] = {}
        for payload in by_setting.values():
            group = payload.get("group", {})
            setting = str(group.get("prediction_setting_id", ""))
            agent = str(group.get("agent_name", ""))
            reward = payload.get("metrics", {}).get("total_reward", {}).get("mean")
            if setting and agent and reward is not None:
                setting_agent.setdefault(setting, {})[agent] = float(reward)
        for setting, rewards in sorted(setting_agent.items()):
            if prediction_required_settings and setting not in prediction_required_settings:
                continue
            sa_setting_reward = rewards.get("sa_ghmappo")
            for baseline in baselines:
                baseline_reward = rewards.get(baseline)
                if (
                    sa_setting_reward is None
                    or baseline_reward is None
                    or sa_setting_reward <= baseline_reward
                ):
                    blockers.append(f"prediction_setting_sa_not_above_{baseline}:{setting}")
    return blockers


def build_final_gate_report(
    *,
    args: argparse.Namespace,
    settings: dict[str, int],
    run_root: Path,
    iteration: int,
    formal_report: dict[str, Any],
    holdout_reports: list[dict[str, Any]],
    support_paths: dict[str, str],
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    reports = [formal_report, *holdout_reports]
    formal_training_provenance = audit_formal_training_provenance(
        formal_report=formal_report,
        learned_agents=args.learned_baseline_agents,
        seeds=args.seeds,
    )
    blockers.extend(str(item) for item in formal_training_provenance["blockers"])
    for report in reports:
        label = f"offset_{report.get('window_rank_offset', 0)}"
        if not bool(report.get("passed")):
            blockers.append(f"learned_gate_failed:{label}")
        if not bool(report.get("formal_contract", {}).get("ready")):
            blockers.append(f"formal_contract_not_ready:{label}")
        blockers.extend(f"{label}:{item}" for item in audit_statistics_ci(report, args.learned_baseline_agents))
    if args.skip_support:
        blockers.append("support_suite_not_run")
    else:
        for kind, path_text in support_paths.items():
            blockers.extend(
                audit_support_summary(
                    summary_path=Path(path_text),
                    kind=kind,
                    support_agents=args.support_agents,
                    prediction_required_settings=args.prediction_required_settings,
                )
            )
    if args.quick:
        blockers.append("quick_mode_not_paper_ready")

    target_reached = not blockers
    return {
        "run_id": run_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "target_reached": target_reached,
        "paper_claim_ready": target_reached,
        "gate_policy": "final_submission_learned_primary",
        "settings": settings,
        "base_manifest_path": args.base_manifest_path,
        "learned_baseline_agents": args.learned_baseline_agents,
        "heuristic_reference_agents": args.heuristic_reference_agents,
        "baseline_profile": args.baseline_profile,
        "mappo_baseline_profile": args.mappo_baseline_profile,
        "heuristic_policy": "supplementary_reference_only",
        "support_agents": args.support_agents,
        "formal_training_provenance": formal_training_provenance,
        "baseline_protocol_versions": formal_report.get("baseline_protocol_versions", {}),
        "budget_protocol": {
            "policy": "matched_environment_interaction_budget",
            "baseline_episodes": settings["baseline_episodes"],
            "max_steps": settings["max_steps"],
            "seeds": args.seeds,
            "cluster_keys": args.cluster_keys,
            "rationale": (
                "Top conference RL papers usually equalize interaction/sample budgets inside a benchmark family. "
                "This final loop follows that standard for VEC: all learned baselines share the same real-data "
                "windows, workflow source, seeds, episode count, and support-suite gates; algorithm-specific "
                "optimization internals are preserved. MAPPO uses the current controller head-credit v3 protocol "
                "under the multi-controller action contract. Domain baselines for DAG offloading, model cache/offloading, "
                "and digital-twin handoff receive no SA-GHMAPPO graph/surrogate/guard-only mechanisms."
            ),
        },
        "prediction_required_settings": args.prediction_required_settings,
        "cluster_keys": args.cluster_keys,
        "minimum_reward_delta": args.minimum_reward_delta,
        "formal_report_path": formal_report.get("report_path", ""),
        "holdout_report_paths": [report.get("report_path", "") for report in holdout_reports],
        "support_paths": support_paths,
        "blockers": sorted(set(blockers)),
        "claim_boundary": [
            "Primary acceptance gate compares SA-GHMAPPO against learned baselines only.",
            "popularity_cache_heuristic is retained as a supplementary reference and does not block the main gate.",
            "Final claims require formal and holdout learned gates plus cluster-bootstrap positive reward CI.",
            "Prediction, robustness, and scalability support are checked against learned support agents.",
            "Formal learned checkpoints must be trained in the final suite or resumed from the same run.",
            "MAPPO claims require the current controller head-credit v3 checkpoint protocol; pre-v3/pre-head-credit MAPPO results are archived only.",
            "Prediction setting-level dominance is required only for claim-relevant learned/noisy predictor settings; no-prediction and oracle settings remain diagnostic.",
        ],
        "command_log": command_log,
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("top_journal_final_submission_%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    settings = effective_settings(args)
    base_manifest = Path(args.base_manifest_path)
    final_report: dict[str, Any] | None = None

    for iteration in range(1, max(1, args.max_iterations) + 1):
        iter_label = f"iter{iteration}"
        formal_report = run_learned_suite(
            args=args,
            settings=settings,
            run_root=run_root,
            suite_run_id=f"{run_id}_{iter_label}_formal",
            base_manifest_path=base_manifest,
            window_rank_offset=0,
            skip_training=args.skip_training,
            force_retrain=args.force_retrain_learned and not args.skip_training,
            command_log=command_log,
        )
        formal_manifest = Path(formal_report["seed_checkpoint_manifest_path"])
        holdout_reports = [
            run_learned_suite(
                args=args,
                settings=settings,
                run_root=run_root,
                suite_run_id=f"{run_id}_{iter_label}_holdout_offset{offset}",
                base_manifest_path=formal_manifest,
                window_rank_offset=offset,
                skip_training=True,
                force_retrain=False,
                command_log=command_log,
            )
            for offset in args.holdout_offsets
        ]
        support_paths: dict[str, str] = {}
        if not args.skip_support:
            for kind in ["prediction", "robustness", "scalability"]:
                if args.resume_support:
                    filename = "prediction_robustness_summary.json" if kind == "prediction" else "aggregate_summary.json"
                    try:
                        reused_path = latest_child_report(run_root / "support" / kind, filename)
                        support_paths[kind] = str(reused_path)
                        command_log.append(
                            {
                                "label": f"reuse_support_{kind}",
                                "command": [],
                                "returncode": 0,
                                "elapsed_sec": 0.0,
                            }
                        )
                        continue
                    except FileNotFoundError:
                        pass
                support_paths[kind] = str(
                    run_support_benchmark(
                        args=args,
                        settings=settings,
                        run_root=run_root,
                        manifest_path=formal_manifest,
                        kind=kind,
                        command_log=command_log,
                    )
                )
        final_report = build_final_gate_report(
            args=args,
            settings=settings,
            run_root=run_root,
            iteration=iteration,
            formal_report=formal_report,
            holdout_reports=holdout_reports,
            support_paths=support_paths,
            command_log=command_log,
        )
        write_json(run_root / "final_submission_gate_report.json", final_report)
        write_json(run_root / "command_log.json", {"commands": command_log})
        if final_report["target_reached"]:
            break

    assert final_report is not None
    print("top-journal final submission loop complete")
    print(f"run_root: {run_root}")
    print(f"final_submission_gate_report_path: {run_root / 'final_submission_gate_report.json'}")
    print(f"target_reached: {final_report['target_reached']}")
    if final_report["blockers"]:
        print("blockers:")
        for blocker in final_report["blockers"]:
            print(f"- {blocker}")
    if args.fail_on_blocker and final_report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
