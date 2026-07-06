"""Run main-method and baseline benchmark across seeds, windows, and workflows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import list_evaluable_agents
from src.evaluators.main_results_support import (
    aggregate_rows,
    audit_checkpoint_map,
    build_pairwise_comparison,
    build_mechanism_diagnosis,
    build_selected_workflow_states,
    build_win_tie_loss_summary,
    infer_benchmark_config_profile,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
    write_rows_csv,
    MAIN_RESULT_METRICS,
    PAPER_PROTOCOL_VERSION,
    PAPER_PROTOCOL_FROZEN,
)

BENCHMARK_AGENT_CHOICES = list_evaluable_agents()
SA_ADVANTAGE_FOCUS_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "handoff_ready_ratio",
    "mechanism_realization_rate",
    "adapter_state_migration_overhead",
]
LOWER_IS_BETTER_FOR_ADVANTAGE = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行主方法与方向匹配对照算法 benchmark")
    parser.add_argument("--agents", nargs="+", default=["sa_ghmappo"], choices=BENCHMARK_AGENT_CHOICES)
    parser.add_argument("--sa_ghmappo_checkpoint_path", type=str, default="")
    parser.add_argument("--flat_ppo_checkpoint_path", type=str, default="")
    parser.add_argument("--flat_mappo_checkpoint_path", type=str, default="")
    parser.add_argument("--seed_checkpoint_manifest_path", type=str, default="")
    parser.add_argument("--mobility_source", type=str, default="ngsim", choices=["ngsim", "lust"])
    parser.add_argument("--primary_vehicle_selection", type=str, default="stable_first", choices=["stable_first", "handoff_pressure"])
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    parser.add_argument("--max_mobility_rows", type=int, default=2500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_count", type=int, default=3)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate", choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"])
    parser.add_argument("--window_mode", type=str, default="mixed_informative", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--window_rank_offset", type=int, default=0)
    parser.add_argument("--exclude_window_plan_path", action="append", default=[])
    parser.add_argument("--predictor_kind", type=str, default="baseline", choices=["baseline", "oracle", "learned_or_calibrated", "supervised"])
    parser.add_argument("--predictor_checkpoint_path", type=str, default="")
    parser.add_argument("--prediction_noise_std", type=float, default=0.0)
    parser.add_argument("--prediction_confidence_scale", type=float, default=1.0)
    parser.add_argument("--prediction_delay_steps", type=int, default=0)
    parser.add_argument("--drop_handoff_prediction_prob", type=float, default=0.0)
    parser.add_argument("--holdout_min_gap_frames", type=int, default=0)
    parser.add_argument("--enforce_non_overlapping_selection", action="store_true")
    parser.add_argument("--activating_handoff_threshold", type=int, default=2)
    parser.add_argument("--activating_vehicle_threshold", type=float, default=2.0)
    parser.add_argument("--activating_predicted_next_ratio_threshold", type=float, default=0.3)
    parser.add_argument("--activating_handoff_prediction_ratio_threshold", type=float, default=0.15)
    parser.add_argument("--non_mechanism_handoff_max", type=int, default=0)
    parser.add_argument("--non_mechanism_prediction_ratio_max", type=float, default=0.05)
    parser.add_argument("--active_non_mechanism_vehicle_threshold", type=float, default=2.0)
    parser.add_argument("--active_non_mechanism_association_change_min", type=int, default=1)
    parser.add_argument("--active_non_mechanism_handoff_max", type=int, default=1)
    parser.add_argument("--active_non_mechanism_predicted_next_ratio_max", type=float, default=0.2)
    parser.add_argument("--active_non_mechanism_handoff_prediction_ratio_max", type=float, default=0.1)
    parser.add_argument("--idle_or_sparse_vehicle_max", type=float, default=1.5)
    parser.add_argument("--idle_or_sparse_association_change_max", type=int, default=0)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "benchmarks" / "main_results"))
    return parser.parse_args()


def load_excluded_window_intervals(paths: list[str]) -> list[tuple[int, int]]:
    intervals: set[tuple[int, int]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            plan = payload
        else:
            plan = payload.get("selected_window_plan", payload.get("selected_windows", []))
        for item in plan:
            start = int(item["frame_offset"])
            end = start + int(item["window_length"]) - 1
            intervals.add((start, end))
    return sorted(intervals)


def build_checkpoint_map(args: argparse.Namespace) -> dict[str, str]:
    return {
        "sa_ghmappo": args.sa_ghmappo_checkpoint_path,
        "ppo": args.flat_ppo_checkpoint_path,
        "mappo": args.flat_mappo_checkpoint_path,
        "flat_ppo": args.flat_ppo_checkpoint_path,
        "flat_mappo": args.flat_mappo_checkpoint_path,
        "reactive_greedy": "",
        "popularity_cache_heuristic": "",
    }


def expand_checkpoint_aliases(checkpoint_map: dict[str, str]) -> dict[str, str]:
    alias_pairs = [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]
    expanded = dict(checkpoint_map)
    for primary_name, legacy_name in alias_pairs:
        primary_path = expanded.get(primary_name, "")
        legacy_path = expanded.get(legacy_name, "")
        if primary_path and not legacy_path:
            expanded[legacy_name] = primary_path
        elif legacy_path and not primary_path:
            expanded[primary_name] = legacy_path
    return expanded


def load_seed_checkpoint_manifest(path: str) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"seed checkpoint manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"seed checkpoint manifest must be a mapping: {manifest_path}")
    normalized: dict[str, dict[str, str]] = {}
    for agent_name, seed_map in payload.items():
        if isinstance(seed_map, dict):
            normalized[str(agent_name)] = {
                str(seed): str(checkpoint_path)
                for seed, checkpoint_path in seed_map.items()
                if str(checkpoint_path)
            }
    for primary_name, legacy_name in [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]:
        if primary_name in normalized and legacy_name not in normalized:
            normalized[legacy_name] = dict(normalized[primary_name])
        elif legacy_name in normalized and primary_name not in normalized:
            normalized[primary_name] = dict(normalized[legacy_name])
    return normalized


def checkpoint_map_for_seed(
    base_checkpoint_map: dict[str, str],
    seed_checkpoint_manifest: dict[str, dict[str, str]],
    seed: int,
) -> dict[str, str]:
    checkpoint_map = dict(base_checkpoint_map)
    seed_key = str(seed)
    for agent_name, seed_map in seed_checkpoint_manifest.items():
        if seed_key in seed_map:
            checkpoint_map[agent_name] = seed_map[seed_key]
    return expand_checkpoint_aliases(checkpoint_map)


def representative_checkpoint_map(
    base_checkpoint_map: dict[str, str],
    seed_checkpoint_manifest: dict[str, dict[str, str]],
    seeds: list[int],
) -> dict[str, str]:
    checkpoint_map = dict(base_checkpoint_map)
    for agent_name, seed_map in seed_checkpoint_manifest.items():
        for seed in seeds:
            checkpoint_path = seed_map.get(str(seed), "")
            if checkpoint_path:
                checkpoint_map[agent_name] = checkpoint_path
                break
    return expand_checkpoint_aliases(checkpoint_map)


def metric_mean(aggregate_by_agent: dict[str, Any], agent_name: str, metric_name: str) -> float:
    return float(
        aggregate_by_agent.get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
        .get("mean", 0.0)
        or 0.0
    )


def build_comparison_against_popularity(
    aggregate_by_agent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if "sa_ghmappo" not in aggregate_by_agent or "popularity_cache_heuristic" not in aggregate_by_agent:
        return {
            "available": False,
            "reason": "sa_ghmappo_or_popularity_cache_heuristic_missing",
        }
    metric_rows: dict[str, Any] = {}
    wins = 0
    losses = 0
    ties = 0
    for metric_name in SA_ADVANTAGE_FOCUS_METRICS:
        sa_value = metric_mean(aggregate_by_agent, "sa_ghmappo", metric_name)
        popularity_value = metric_mean(aggregate_by_agent, "popularity_cache_heuristic", metric_name)
        delta = round(sa_value - popularity_value, 6)
        effective_delta = -delta if metric_name in LOWER_IS_BETTER_FOR_ADVANTAGE else delta
        if effective_delta > 1e-6:
            result = "win"
            wins += 1
        elif effective_delta < -1e-6:
            result = "loss"
            losses += 1
        else:
            result = "tie"
            ties += 1
        metric_rows[metric_name] = {
            "sa_ghmappo": sa_value,
            "popularity_cache_heuristic": popularity_value,
            "delta_sa_minus_popularity": delta,
            "higher_is_better": metric_name not in LOWER_IS_BETTER_FOR_ADVANTAGE,
            "result": result,
        }
    guard_trigger_count = sum(
        int(row.get("continuity_guard_trigger_count", 0) or 0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    target_mismatch_guard_count = sum(
        int(row.get("target_mismatch_guard_count", 0) or 0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    action_projection_count = sum(
        int(row.get("action_projection_count", 0) or 0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    invalid_action_attempt_count = sum(
        int(row.get("invalid_action_attempt_count", 0) or 0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    guard_action_delta_count = sum(
        int(row.get("guard_action_delta_count", 0) or 0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    mechanism_attempt_count = sum(
        float(row.get("mechanism_attempt_count", 0.0) or 0.0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    mechanism_validated_success_count = sum(
        float(row.get("mechanism_validated_success_count", 0.0) or 0.0)
        for row in rows
        if row.get("agent_name") == "sa_ghmappo"
    )
    return {
        "available": True,
        "baseline_agent": "popularity_cache_heuristic",
        "candidate_agent": "sa_ghmappo",
        "metrics": metric_rows,
        "win_loss_tie": {"win": wins, "loss": losses, "tie": ties},
        "guard_summary": {
            "continuity_guard_trigger_count": guard_trigger_count,
            "target_mismatch_guard_count": target_mismatch_guard_count,
            "action_projection_count": action_projection_count,
            "invalid_action_attempt_count": invalid_action_attempt_count,
            "guard_action_delta_count": guard_action_delta_count,
            "mechanism_attempt_count": round(mechanism_attempt_count, 6),
            "mechanism_validated_success_count": round(mechanism_validated_success_count, 6),
            "mechanism_success_rate": round(
                mechanism_validated_success_count / float(max(mechanism_attempt_count, 1.0)),
                6,
            ),
        },
    }


def build_sa_advantage_diagnosis(
    aggregate_by_agent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison = build_comparison_against_popularity(aggregate_by_agent, rows)
    if not comparison.get("available", False):
        return comparison
    metrics = dict(comparison.get("metrics", {}))
    reached_minimum = bool(
        metrics.get("total_reward", {}).get("result") == "win"
        and metrics.get("backhaul_traffic_cost", {}).get("result") in {"win", "tie"}
        and (
            metrics.get("workflow_continuity_rate", {}).get("result") in {"win", "tie"}
            or metrics.get("handoff_failure_rate", {}).get("result") in {"win", "tie"}
        )
    )
    blockers: list[str] = []
    if metrics.get("total_reward", {}).get("result") != "win":
        blockers.append("total_reward_not_above_popularity")
    if metrics.get("workflow_continuity_rate", {}).get("result") == "loss":
        blockers.append("continuity_below_popularity")
    if metrics.get("handoff_failure_rate", {}).get("result") == "loss":
        blockers.append("handoff_failure_above_popularity")
    if metrics.get("backhaul_traffic_cost", {}).get("result") == "loss":
        blockers.append("backhaul_cost_above_popularity")
    if (
        metrics.get("mechanism_realization_rate", {}).get("result") == "loss"
        and metrics.get("handoff_ready_ratio", {}).get("result") == "loss"
    ):
        blockers.append("mechanism_realization_and_ready_below_popularity")
    sa_guard_rows = [row for row in rows if row.get("agent_name") == "sa_ghmappo"]
    mechanism_attempt_count = sum(float(row.get("mechanism_attempt_count", 0.0) or 0.0) for row in sa_guard_rows)
    mechanism_validated_success_count = sum(
        float(row.get("mechanism_validated_success_count", 0.0) or 0.0)
        for row in sa_guard_rows
    )
    if mechanism_attempt_count > 0.0 and mechanism_validated_success_count <= 0.0:
        blockers.append("mechanism_attempts_without_validated_success")
    return {
        **comparison,
        "minimum_success_reached": reached_minimum,
        "blockers": blockers,
        "sa_episode_count": len(sa_guard_rows),
        "mechanism_success_gate": {
            "attempt_count": round(mechanism_attempt_count, 6),
            "validated_success_count": round(mechanism_validated_success_count, 6),
            "success_rate": round(
                mechanism_validated_success_count / float(max(mechanism_attempt_count, 1.0)),
                6,
            ),
        },
        "diagnosis_note": (
            "guard/imitation advantage should be read together with checkpoint metadata; "
            "benchmark protocol and baseline behavior are unchanged."
        ),
    }


def main() -> None:
    args = parse_args()
    excluded_window_intervals = load_excluded_window_intervals(args.exclude_window_plan_path)
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    base_checkpoint_map = expand_checkpoint_aliases(build_checkpoint_map(args))
    seed_checkpoint_manifest = load_seed_checkpoint_manifest(args.seed_checkpoint_manifest_path)
    audit_checkpoint_source_map = representative_checkpoint_map(
        base_checkpoint_map=base_checkpoint_map,
        seed_checkpoint_manifest=seed_checkpoint_manifest,
        seeds=args.seeds,
    )
    mobility_source_path, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector=args.window_selector,
        window_count=args.window_count,
        window_scan_stride=args.window_scan_stride,
        random_seed=args.seeds[0] if args.seeds else 7,
        window_mode=args.window_mode,
        window_rank_offset=args.window_rank_offset,
        excluded_window_intervals=excluded_window_intervals,
        holdout_min_gap_frames=args.holdout_min_gap_frames,
        enforce_non_overlapping_selection=args.enforce_non_overlapping_selection,
        activating_handoff_threshold=args.activating_handoff_threshold,
        activating_vehicle_threshold=args.activating_vehicle_threshold,
        activating_predicted_next_ratio_threshold=args.activating_predicted_next_ratio_threshold,
        activating_handoff_prediction_ratio_threshold=args.activating_handoff_prediction_ratio_threshold,
        non_mechanism_handoff_max=args.non_mechanism_handoff_max,
        non_mechanism_prediction_ratio_max=args.non_mechanism_prediction_ratio_max,
        active_non_mechanism_vehicle_threshold=args.active_non_mechanism_vehicle_threshold,
        active_non_mechanism_association_change_min=args.active_non_mechanism_association_change_min,
        active_non_mechanism_handoff_max=args.active_non_mechanism_handoff_max,
        active_non_mechanism_predicted_next_ratio_max=args.active_non_mechanism_predicted_next_ratio_max,
        active_non_mechanism_handoff_prediction_ratio_max=args.active_non_mechanism_handoff_prediction_ratio_max,
        idle_or_sparse_vehicle_max=args.idle_or_sparse_vehicle_max,
        idle_or_sparse_association_change_max=args.idle_or_sparse_association_change_max,
    )
    checkpoint_audit_bundle = audit_checkpoint_map(checkpoint_map=audit_checkpoint_source_map, agents=args.agents)
    checkpoint_audit = checkpoint_audit_bundle["checkpoint_audit"]
    smoke_warnings = checkpoint_audit_bundle["warnings"]

    benchmark_run_id = datetime.now().strftime(f"main_results_{args.window_mode}_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / benchmark_run_id
    episode_root = output_root / "episodes"
    rows: list[dict[str, Any]] = []
    selected_workflow_ids_by_seed: dict[str, list[str]] = {}
    selected_window_plan = list(window_payload["selected_windows"])

    for seed in args.seeds:
        workflow_states = build_selected_workflow_states(
            workflow_csv_path=args.workflow_csv_path,
            max_workflows=args.max_workflows,
            workflow_selector=args.workflow_selector,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            random_seed=seed,
        )
        selected_workflow_ids_by_seed[str(seed)] = [workflow_state.workflow_id for workflow_state in workflow_states]
        seed_checkpoint_map = checkpoint_map_for_seed(
            base_checkpoint_map=base_checkpoint_map,
            seed_checkpoint_manifest=seed_checkpoint_manifest,
            seed=seed,
        )
        for window_candidate in selected_window_plan:
            mobility_bundle = load_window_bundle(
                root_dir=ROOT_DIR,
                mobility_source=args.mobility_source,
                mobility_csv_path=args.mobility_csv_path,
                lust_scenario_root=args.lust_scenario_root,
                max_mobility_rows=args.max_mobility_rows,
                rsu_layout=str(window_candidate.get("recommended_rsu_layout", args.rsu_layout)),
                frame_offset=int(window_candidate["frame_offset"]),
                window_length=int(window_candidate["window_length"]),
                random_seed=seed,
            )
            mobility_bundle.rsu_metadata["window_rank"] = window_candidate["window_rank"]
            mobility_bundle.rsu_metadata["window_class"] = window_candidate["window_class"]
            for workflow_state in workflow_states:
                for agent_name in args.agents:
                    summary = run_real_episode(
                        root_dir=ROOT_DIR,
                        agent_name=agent_name,
                        checkpoint_map=seed_checkpoint_map,
                        workflow_state=workflow_state,
                        workflow_source_path=args.workflow_csv_path,
                        mobility_bundle=mobility_bundle,
                        seed=seed,
                        max_steps=args.max_steps,
                        mobility_source=args.mobility_source,
                        primary_vehicle_selection=args.primary_vehicle_selection,
                        run_metadata={
                            "script": "scripts/benchmark_main_results.py",
                            "benchmark_run_id": benchmark_run_id,
                            "mainline": mainline_label,
                            "mobility_source_path": mobility_source_path,
                            "window_rank": window_candidate["window_rank"],
                            "window_class": window_candidate["window_class"],
                            "window_mode": args.window_mode,
                            "window_rank_offset": args.window_rank_offset,
                        },
                        predictor_kwargs={
                            "predictor_kind": args.predictor_kind,
                            "predictor_checkpoint_path": args.predictor_checkpoint_path,
                            "prediction_noise_std": args.prediction_noise_std,
                            "prediction_confidence_scale": args.prediction_confidence_scale,
                            "prediction_delay_steps": args.prediction_delay_steps,
                            "drop_handoff_prediction_prob": args.drop_handoff_prediction_prob,
                        },
                    )
                    summary_path = episode_root / str(mobility_bundle.rsu_metadata.get("window_id")) / workflow_state.workflow_id / agent_name / f"seed_{seed}.summary.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary["run_info"]["summary_path"] = str(summary_path)
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    rows.append(summary_to_row(summary))

    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_seed_and_agent = aggregate_rows(rows, group_keys=["seed", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_workflow_and_agent = aggregate_rows(rows, group_keys=["workflow_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_window_and_agent = aggregate_rows(rows, group_keys=["window_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_seed_window_agent = aggregate_rows(rows, group_keys=["seed", "window_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    mechanism_window_rows = [row for row in rows if str(row.get("window_class", "")) == "mechanism_activating"]
    active_non_mechanism_window_rows = [row for row in rows if str(row.get("window_class", "")) == "active_non_mechanism"]
    idle_or_sparse_window_rows = [row for row in rows if str(row.get("window_class", "")) == "idle_or_sparse"]
    aggregate_mechanism_windows_by_agent = aggregate_rows(mechanism_window_rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_active_non_mechanism_windows_by_agent = aggregate_rows(active_non_mechanism_window_rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_idle_or_sparse_windows_by_agent = aggregate_rows(idle_or_sparse_window_rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    comparison_against_popularity = build_comparison_against_popularity(aggregate_by_agent, rows)
    sa_advantage_diagnosis = build_sa_advantage_diagnosis(aggregate_by_agent, rows)

    pairwise_comparison: dict[str, Any] = {}
    if "sa_ghmappo" in aggregate_by_agent:
        pairwise_comparison = build_pairwise_comparison(
            aggregate_by_agent=aggregate_by_agent,
            baseline_agent="sa_ghmappo",
            metrics=MAIN_RESULT_METRICS,
        )
    win_tie_loss_summary = build_win_tie_loss_summary(
        aggregate_by_window_and_agent=aggregate_by_window_and_agent,
        aggregate_by_workflow_and_agent=aggregate_by_workflow_and_agent,
        metrics=MAIN_RESULT_METRICS,
    )
    aggregate_summary = {
        "run_id": benchmark_run_id,
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
        "canonical_paper_protocol": bool(args.window_mode in {"activating_only", "mixed_informative", "full_stratified"}),
        "protocol_note": "使用 frozen paper protocol 的窗口分层，用于主方法与当前 baseline 的公平对照。",
        "config_profile": infer_benchmark_config_profile(checkpoint_audit, args.agents),
        "window_mode": args.window_mode,
        "window_rank_offset": args.window_rank_offset,
        "exclude_window_plan_paths": list(args.exclude_window_plan_path),
        "excluded_window_intervals": [list(interval) for interval in excluded_window_intervals],
        "holdout_min_gap_frames": args.holdout_min_gap_frames,
        "enforce_non_overlapping_selection": args.enforce_non_overlapping_selection,
        "excluded_window_count": window_payload.get("excluded_window_count", 0),
        "excluded_window_ids": window_payload.get("excluded_window_ids", []),
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "mainline": mainline_label,
        "predictor_runtime_config": {
            "predictor_kind": args.predictor_kind,
            "predictor_checkpoint_path": args.predictor_checkpoint_path,
            "prediction_noise_std": args.prediction_noise_std,
            "prediction_confidence_scale": args.prediction_confidence_scale,
            "prediction_delay_steps": args.prediction_delay_steps,
            "drop_handoff_prediction_prob": args.drop_handoff_prediction_prob,
        },
        "agents": args.agents,
        "seeds": args.seeds,
        "workflow_selector": args.workflow_selector,
        "selected_workflow_ids_by_seed": selected_workflow_ids_by_seed,
        "selected_window_plan": selected_window_plan,
        "selected_window_plan_by_strata": window_payload.get("selected_window_plan_by_strata", {}),
        "mechanism_activating_windows": window_payload["mechanism_activating_windows"],
        "active_non_mechanism_windows": window_payload.get("active_non_mechanism_windows", []),
        "idle_or_sparse_windows": window_payload.get("idle_or_sparse_windows", []),
        "non_mechanism_windows": window_payload["non_mechanism_windows"],
        "checkpoint_map": audit_checkpoint_source_map,
        "seed_checkpoint_manifest_path": args.seed_checkpoint_manifest_path,
        "seed_checkpoint_manifest": seed_checkpoint_manifest,
        "checkpoint_audit": checkpoint_audit,
        "smoke_checkpoint_warnings": smoke_warnings,
        "episode_count": len(rows),
        "mobility_source_path": mobility_source_path,
        "workflow_source_path": args.workflow_csv_path,
        "window_strata_summary": {
            "mechanism_activating": len(window_payload.get("mechanism_activating_windows", [])),
            "active_non_mechanism": len(window_payload.get("active_non_mechanism_windows", [])),
            "idle_or_sparse": len(window_payload.get("idle_or_sparse_windows", [])),
            "selected_mechanism_activating": len(window_payload.get("selected_window_plan_by_strata", {}).get("mechanism_activating", [])),
            "selected_active_non_mechanism": len(window_payload.get("selected_window_plan_by_strata", {}).get("active_non_mechanism", [])),
            "selected_idle_or_sparse": len(window_payload.get("selected_window_plan_by_strata", {}).get("idle_or_sparse", [])),
        },
        "aggregate_by_agent": aggregate_by_agent,
        "aggregate_by_seed_and_agent": aggregate_by_seed_and_agent,
        "aggregate_by_window_and_agent": aggregate_by_window_and_agent,
        "aggregate_by_workflow_and_agent": aggregate_by_workflow_and_agent,
        "aggregate_by_seed_window_agent": aggregate_by_seed_window_agent,
        "aggregate_mechanism_windows_by_agent": aggregate_mechanism_windows_by_agent,
        "aggregate_active_non_mechanism_windows_by_agent": aggregate_active_non_mechanism_windows_by_agent,
        "aggregate_idle_or_sparse_windows_by_agent": aggregate_idle_or_sparse_windows_by_agent,
        "aggregate_non_mechanism_windows_by_agent": aggregate_active_non_mechanism_windows_by_agent,
        "pairwise_comparison": pairwise_comparison,
        "mechanism_diagnosis": build_mechanism_diagnosis(rows),
        "win_tie_loss_summary": win_tie_loss_summary,
        "comparison_against_popularity": comparison_against_popularity,
        "sa_advantage_diagnosis": sa_advantage_diagnosis,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_root / "aggregate_summary.json"
    rows_path = output_root / "benchmark_rows.csv"
    comparison_path = output_root / "comparison_against_popularity.json"
    diagnosis_path = output_root / "sa_advantage_diagnosis.json"
    aggregate_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison_against_popularity, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnosis_path.write_text(json.dumps(sa_advantage_diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(rows_path, rows)

    print("main results benchmark complete")
    print(f"run_id: {benchmark_run_id}")
    print(f"aggregate_summary_path: {aggregate_path}")
    print(f"benchmark_rows_path: {rows_path}")
    print(f"comparison_against_popularity_path: {comparison_path}")
    print(f"sa_advantage_diagnosis_path: {diagnosis_path}")
    if smoke_warnings:
        print("checkpoint warnings:")
        for warning in smoke_warnings:
            print(f"  {warning}")
    for agent_name, payload in aggregate_by_agent.items():
        metrics = payload["metrics"]
        print(f"[{agent_name}] total_reward_mean={metrics['total_reward']['mean']:.3f}")
        print(f"[{agent_name}] continuity_mean={metrics['workflow_continuity_rate']['mean']:.3f}")
        print(f"[{agent_name}] ready_mean={metrics['handoff_ready_ratio']['mean']:.3f}")
        print(f"[{agent_name}] mechanism_realization_rate={metrics['mechanism_realization_rate']['mean']:.3f}")


if __name__ == "__main__":
    main()
