"""Analyze action/cache/service differences for mechanism-window benchmark gaps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


SA_AGENT = "sa_ghmappo"
POPULARITY_AGENT = "popularity_cache_heuristic"

PAIR_KEYS = ["mode", "window_id", "window_class", "workflow_id", "seed"]

CORE_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "mechanism_realization_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
]

SERVICE_CACHE_ACTION_METRICS = [
    "service_success_count",
    "service_delay_sum",
    "service_wait_sum",
    "service_restart_count",
    "workflow_completed_count",
    "workflow_unfinished_count",
    "adapter_hit_count",
    "adapter_miss_count",
    "adapter_warm_hit_count",
    "adapter_cold_start_count",
    "cache_eviction_count",
    "cache_admission_count",
    "cache_noop_count",
    "local_exec_count",
    "current_rsu_exec_count",
    "next_rsu_exec_count",
    "neighbor_rsu_exec_count",
    "cloud_exec_count",
    "prefetch_action_count",
    "migration_action_count",
    "no_op_action_count",
    "prefetch_attempt_count",
    "prefetch_success_count",
    "prefetch_failed_count",
    "migration_attempt_count",
    "migration_success_count",
    "migration_failed_count",
    "migration_overhead_sum",
]

REWARD_PROXY_METRICS = [
    "delay_reward_component",
    "cache_reward_component",
    "handoff_reward_component",
    "backhaul_reward_component",
    "failure_reward_component",
    "continuity_reward_component",
    "service_reward_component",
    "mechanism_exploration_reward_component",
    "constraint_penalty_sum",
    "migration_cost_sum",
    "cache_miss_penalty_sum",
    "delay_penalty_sum",
]

DIFF_METRICS = CORE_METRICS + SERVICE_CACHE_ACTION_METRICS + REWARD_PROXY_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed_benchmark_dir", type=Path, required=True)
    parser.add_argument("--full_benchmark_dir", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5"),
    )
    return parser.parse_args()


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_window_mode(benchmark_dir: Path, fallback: str) -> str:
    aggregate_path = benchmark_dir / "aggregate_summary.json"
    if not aggregate_path.exists():
        return fallback
    try:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return str(payload.get("window_mode") or fallback)


def read_rows(benchmark_dir: Path, fallback_mode: str) -> list[dict[str, Any]]:
    rows_path = benchmark_dir / "benchmark_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing benchmark_rows.csv: {rows_path}")
    mode = _read_window_mode(benchmark_dir, fallback=fallback_mode)
    with rows_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row.setdefault("mode", mode)
        if not row.get("mode") or row["mode"] == "unknown":
            row["mode"] = mode
        row.setdefault("window_tag", row.get("window_class", "unknown"))
        row.setdefault("policy_name", row.get("agent_name", "unknown"))
        row.setdefault("scenario_id", row.get("window_id", "unknown"))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        agent_name = str(row.get("agent_name") or row.get("policy_name") or "")
        if agent_name not in {SA_AGENT, POPULARITY_AGENT}:
            continue
        key = tuple(str(row.get(group_key, "")) for group_key in PAIR_KEYS)
        grouped[key][agent_name] = row

    pair_rows: list[dict[str, Any]] = []
    for key, agent_rows in grouped.items():
        if SA_AGENT not in agent_rows or POPULARITY_AGENT not in agent_rows:
            continue
        sa_row = agent_rows[SA_AGENT]
        pop_row = agent_rows[POPULARITY_AGENT]
        pair_row: dict[str, Any] = {
            "mode": key[0],
            "window_id": key[1],
            "window_class": key[2],
            "window_tag": key[2],
            "workflow_id": key[3],
            "seed": key[4],
            "scenario_id": sa_row.get("scenario_id") or key[1],
        }
        for metric_name in DIFF_METRICS:
            sa_value = _float_value(sa_row.get(metric_name), 0.0)
            pop_value = _float_value(pop_row.get(metric_name), 0.0)
            pair_row[f"sa_{metric_name}"] = round(sa_value, 6)
            pair_row[f"popularity_{metric_name}"] = round(pop_value, 6)
            pair_row[f"delta_{metric_name}"] = round(sa_value - pop_value, 6)
        pair_rows.append(pair_row)
    pair_rows.sort(key=lambda item: (str(item["mode"]), str(item["window_class"]), _float_value(item["delta_total_reward"])))
    return pair_rows


def _mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def build_gap_summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row.get("mode")), str(row.get("window_class")))].append(row)

    output_rows: list[dict[str, Any]] = []
    summary_metrics = [
        "total_reward",
        "adapter_miss_count",
        "adapter_warm_hit_count",
        "adapter_cold_start_count",
        "service_wait_sum",
        "service_delay_sum",
        "current_rsu_exec_count",
        "next_rsu_exec_count",
        "prefetch_attempt_count",
        "prefetch_success_count",
        "migration_attempt_count",
        "migration_success_count",
        "backhaul_traffic_cost",
        "delay_reward_component",
        "cache_reward_component",
        "service_reward_component",
        "continuity_reward_component",
    ]
    for (mode, window_class), rows in sorted(grouped.items()):
        out = {
            "mode": mode,
            "window_tag": window_class,
            "paired_episode_count": len(rows),
            "sa_losing_episode_count": sum(1 for row in rows if _float_value(row.get("delta_total_reward")) < -1e-6),
            "sa_winning_episode_count": sum(1 for row in rows if _float_value(row.get("delta_total_reward")) > 1e-6),
        }
        for metric_name in summary_metrics:
            out[f"mean_delta_{metric_name}"] = _mean([_float_value(row.get(f"delta_{metric_name}")) for row in rows])
            out[f"mean_sa_{metric_name}"] = _mean([_float_value(row.get(f"sa_{metric_name}")) for row in rows])
            out[f"mean_popularity_{metric_name}"] = _mean([_float_value(row.get(f"popularity_{metric_name}")) for row in rows])
        output_rows.append(out)
    return output_rows


def build_policy_breakdown(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agent_name = str(row.get("agent_name") or row.get("policy_name") or "unknown")
        if agent_name not in {SA_AGENT, POPULARITY_AGENT}:
            continue
        grouped[(str(row.get("mode")), str(row.get("window_class")), agent_name)].append(row)

    output_rows: list[dict[str, Any]] = []
    for (mode, window_class, agent_name), group_rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "mode": mode,
            "window_tag": window_class,
            "policy_name": agent_name,
            "episode_count": len(group_rows),
        }
        for metric_name in metrics:
            values = [_float_value(row.get(metric_name), 0.0) for row in group_rows]
            out[f"{metric_name}_mean"] = _mean(values)
            out[f"{metric_name}_sum"] = round(sum(values), 6)
        output_rows.append(out)
    return output_rows


def infer_likely_sources(mechanism_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    losing_rows = [row for row in mechanism_pairs if _float_value(row.get("delta_total_reward")) < -1e-6]
    if not losing_rows:
        return {
            "status": "no_sa_loss_in_mechanism_pairs",
            "likely_sources": [],
            "notes": ["SA did not lose reward on paired mechanism_activating rows in the diagnostic export."],
        }

    means = {
        metric_name: _mean([_float_value(row.get(f"delta_{metric_name}")) for row in losing_rows])
        for metric_name in DIFF_METRICS
    }
    likely_sources: list[str] = []
    notes: list[str] = []
    if means.get("adapter_miss_count", 0.0) > 0.1 or means.get("adapter_warm_hit_count", 0.0) < -0.1:
        likely_sources.append("cache_miss_or_warm_hit_gap")
    if means.get("adapter_cold_start_count", 0.0) > 0.05:
        likely_sources.append("cold_start_gap")
    if means.get("service_wait_sum", 0.0) > 0.05 or means.get("delay_reward_component", 0.0) < -0.05:
        likely_sources.append("service_wait_or_delay_gap")
    if abs(means.get("current_rsu_exec_count", 0.0)) > 0.2 or abs(means.get("next_rsu_exec_count", 0.0)) > 0.2:
        likely_sources.append("placement_action_mix_gap")
    if means.get("prefetch_success_count", 0.0) < -0.05 or means.get("migration_success_count", 0.0) < -0.05:
        likely_sources.append("prefetch_migration_realization_gap")
    if means.get("continuity_reward_component", 0.0) < -0.05:
        likely_sources.append("continuity_bonus_tiebreak_gap")
    if means.get("prefetch_attempt_count", 0.0) < -0.2 and means.get("migration_attempt_count", 0.0) > 0.2:
        likely_sources.append("prefetch_to_prepare_action_mix_shift")
    if means.get("backhaul_traffic_cost", 0.0) < -1.0 and abs(means.get("backhaul_reward_component", 0.0)) <= 1e-6:
        likely_sources.append("backhaul_metric_advantage_not_reflected_in_reward_component")

    explained_proxy_delta = sum(
        means.get(metric_name, 0.0)
        for metric_name in [
            "service_reward_component",
            "delay_reward_component",
            "cache_reward_component",
            "backhaul_reward_component",
            "failure_reward_component",
            "continuity_reward_component",
            "handoff_reward_component",
        ]
    )
    reward_delta = means.get("total_reward", 0.0)
    unexplained = round(reward_delta - explained_proxy_delta, 6)
    if abs(unexplained) > 0.05:
        likely_sources.append("hidden_or_unmatched_reward_component")
        notes.append(
            "The summed exported reward proxy components do not fully match total_reward delta; inspect reward_dict coverage."
        )
    if not likely_sources:
        likely_sources.append("aggregation_noise_or_tie_break_scale")
        notes.append("Observable cache/service/action proxies are close; remaining gap is near the export precision/noise scale.")

    return {
        "status": "diagnosed_from_observable_proxies",
        "losing_pair_count": len(losing_rows),
        "mean_delta_on_losing_pairs": means,
        "reward_delta_minus_exported_proxy_delta": unexplained,
        "likely_sources": likely_sources,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.mixed_benchmark_dir, fallback_mode="mixed_informative")
    loaded_inputs = {"mixed_benchmark_dir": str(args.mixed_benchmark_dir)}
    if args.full_benchmark_dir is not None:
        rows.extend(read_rows(args.full_benchmark_dir, fallback_mode="full_stratified"))
        loaded_inputs["full_benchmark_dir"] = str(args.full_benchmark_dir)

    pair_rows = build_pairs(rows)
    mechanism_pair_rows = [row for row in pair_rows if str(row.get("window_class")) == "mechanism_activating"]
    actionmix_gap_summary = build_gap_summary(pair_rows)
    cache_service_breakdown = build_policy_breakdown(rows, SERVICE_CACHE_ACTION_METRICS + CORE_METRICS)
    reward_proxy_breakdown = build_policy_breakdown(rows, REWARD_PROXY_METRICS + ["total_reward"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    actionmix_path = output_dir / "actionmix_gap_summary.csv"
    mechanism_diff_path = output_dir / "mechanism_activating_policy_diff.csv"
    cache_service_path = output_dir / "cache_service_breakdown.csv"
    reward_proxy_path = output_dir / "reward_proxy_breakdown.csv"
    summary_path = output_dir / "diagnosis_summary.json"

    write_csv(actionmix_path, actionmix_gap_summary)
    write_csv(mechanism_diff_path, mechanism_pair_rows)
    write_csv(cache_service_path, cache_service_breakdown)
    write_csv(reward_proxy_path, reward_proxy_breakdown)

    top_losses = sorted(mechanism_pair_rows, key=lambda row: _float_value(row.get("delta_total_reward")))[:10]
    mechanism_pairs_by_mode = {
        mode: [row for row in mechanism_pair_rows if str(row.get("mode")) == mode]
        for mode in sorted({str(row.get("mode")) for row in mechanism_pair_rows})
    }
    likely_gap_sources_by_mode = {
        mode: infer_likely_sources(mode_rows)
        for mode, mode_rows in mechanism_pairs_by_mode.items()
    }
    summary_payload = {
        "task": "sa_mechanism_actionmix_diagnosis_round5",
        "policy_or_reward_modified": False,
        "environment_semantics_modified": False,
        "inputs": loaded_inputs,
        "row_count": len(rows),
        "paired_episode_count": len(pair_rows),
        "mechanism_activating_pair_count": len(mechanism_pair_rows),
        "outputs": {
            "actionmix_gap_summary": str(actionmix_path),
            "mechanism_activating_policy_diff": str(mechanism_diff_path),
            "cache_service_breakdown": str(cache_service_path),
            "reward_proxy_breakdown": str(reward_proxy_path),
        },
        "gap_summary": actionmix_gap_summary,
        "top_mechanism_activating_sa_losses": top_losses,
        "likely_gap_sources": likely_gap_sources_by_mode.get("mixed_informative", infer_likely_sources(mechanism_pair_rows)),
        "likely_gap_sources_by_mode": likely_gap_sources_by_mode,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("mechanism action-mix diagnosis complete")
    print(f"actionmix_gap_summary: {actionmix_path}")
    print(f"mechanism_activating_policy_diff: {mechanism_diff_path}")
    print(f"cache_service_breakdown: {cache_service_path}")
    print(f"reward_proxy_breakdown: {reward_proxy_path}")
    print(f"diagnosis_summary: {summary_path}")


if __name__ == "__main__":
    main()
