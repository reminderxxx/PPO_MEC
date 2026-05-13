"""Paired statistics for top-journal benchmark rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


DEFAULT_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "mechanism_realization_rate",
]
LOWER_IS_BETTER = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
}
DEFAULT_PAIR_KEYS = [
    "seed",
    "window_id",
    "workflow_id",
    "prediction_setting_id",
    "robustness_setting_id",
    "scalability_setting_id",
    "ablation_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap CI and sign-test summaries.")
    parser.add_argument("--rows_path", action="append", required=True, help="benchmark_rows.csv or equivalent; repeatable")
    parser.add_argument("--candidate_agent", type=str, default="sa_ghmappo")
    parser.add_argument("--baseline_agents", nargs="+", default=["popularity_cache_heuristic", "ppo", "mappo", "reactive_greedy"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--pair_keys", nargs="+", default=DEFAULT_PAIR_KEYS)
    parser.add_argument(
        "--cluster_keys",
        nargs="*",
        default=[],
        help=(
            "Optional cluster bootstrap keys, e.g. seed window_id workflow_id. "
            "When omitted, bootstrap resamples paired rows directly."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--output_root", type=str, required=True)
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["source_rows_path"] = str(path)
                rows.append(row)
    return rows


def float_value(row: dict[str, str], field_name: str) -> float | None:
    value = row.get(field_name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pair_key(row: dict[str, str], key_fields: list[str]) -> tuple[str, ...]:
    values = [row.get("source_rows_path", "")]
    for field_name in key_fields:
        if field_name in row and row.get(field_name, "") != "":
            values.append(f"{field_name}={row.get(field_name, '')}")
    return tuple(values)


def cluster_key(row: dict[str, str], key_fields: list[str], fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not key_fields:
        return fallback
    values = []
    for field_name in key_fields:
        if field_name in row and row.get(field_name, "") != "":
            values.append(f"{field_name}={row.get(field_name, '')}")
    return tuple(values) if values else fallback


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_sign_test_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials <= 0:
        return 1.0
    tail = min(wins, losses)
    cumulative = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * cumulative)


def bootstrap_pair_means(deltas: list[float], bootstrap_samples: int, rng: random.Random) -> list[float]:
    if not deltas:
        return []
    del rng
    state = (len(deltas) * 1_000_003 + bootstrap_samples * 97 + 17) & 0x7FFFFFFF
    bootstrap_means = []
    for _ in range(max(1, bootstrap_samples)):
        sample = []
        for _ in deltas:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.append(deltas[state % len(deltas)])
        bootstrap_means.append(fmean(sample))
    return bootstrap_means


def bootstrap_cluster_means(
    deltas: list[float],
    clusters: list[tuple[str, ...]],
    bootstrap_samples: int,
    rng: random.Random,
) -> list[float]:
    by_cluster: dict[tuple[str, ...], list[float]] = {}
    for delta, cluster in zip(deltas, clusters):
        by_cluster.setdefault(cluster, []).append(delta)
    cluster_values = list(by_cluster.values())
    if not cluster_values:
        return []
    del rng
    state = (len(deltas) * 1_000_003 + len(cluster_values) * 9_176 + bootstrap_samples * 97 + 31) & 0x7FFFFFFF
    bootstrap_means = []
    for _ in range(max(1, bootstrap_samples)):
        sample: list[float] = []
        for _ in cluster_values:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.extend(cluster_values[state % len(cluster_values)])
        bootstrap_means.append(fmean(sample) if sample else 0.0)
    return bootstrap_means


def summarize_deltas(
    deltas: list[float],
    bootstrap_samples: int,
    rng: random.Random,
    clusters: list[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    if not deltas:
        return {
            "paired_count": 0,
            "bootstrap_unit": "cluster" if clusters else "pair",
            "cluster_count": 0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "std_delta": 0.0,
            "cohen_dz": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_pvalue": 1.0,
        }
    mean_delta = fmean(deltas)
    std_delta = stdev(deltas) if len(deltas) > 1 else 0.0
    if clusters is not None and len(clusters) == len(deltas):
        bootstrap_means = bootstrap_cluster_means(deltas, clusters, bootstrap_samples, rng)
        bootstrap_unit = "cluster"
        cluster_count = len(set(clusters))
    else:
        bootstrap_means = bootstrap_pair_means(deltas, bootstrap_samples, rng)
        bootstrap_unit = "pair"
        cluster_count = len(deltas)
    wins = sum(1 for value in deltas if value > 1e-9)
    losses = sum(1 for value in deltas if value < -1e-9)
    ties = len(deltas) - wins - losses
    return {
        "paired_count": len(deltas),
        "bootstrap_unit": bootstrap_unit,
        "cluster_count": cluster_count,
        "mean_delta": round(mean_delta, 6),
        "ci95_low": round(percentile(bootstrap_means, 0.025), 6),
        "ci95_high": round(percentile(bootstrap_means, 0.975), 6),
        "std_delta": round(std_delta, 6),
        "cohen_dz": round(mean_delta / std_delta, 6) if std_delta > 1e-12 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_pvalue": round(exact_sign_test_pvalue(wins, losses), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    rows = load_rows(args.rows_path)
    by_key_and_agent: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = pair_key(row, args.pair_keys)
        by_key_and_agent.setdefault(key, {})[row.get("agent_name", "")] = row

    output_rows: list[dict[str, Any]] = []
    for baseline_agent in args.baseline_agents:
        for metric in args.metrics:
            signed_deltas: list[float] = []
            raw_deltas: list[float] = []
            delta_clusters: list[tuple[str, ...]] = []
            for key, agents_by_key in by_key_and_agent.items():
                candidate_row = agents_by_key.get(args.candidate_agent)
                baseline_row = agents_by_key.get(baseline_agent)
                if candidate_row is None or baseline_row is None:
                    continue
                candidate_value = float_value(candidate_row, metric)
                baseline_value = float_value(baseline_row, metric)
                if candidate_value is None or baseline_value is None:
                    continue
                raw_delta = candidate_value - baseline_value
                raw_deltas.append(raw_delta)
                signed_deltas.append(-raw_delta if metric in LOWER_IS_BETTER else raw_delta)
                delta_clusters.append(cluster_key(candidate_row, args.cluster_keys, key))
            cluster_payload = delta_clusters if args.cluster_keys else None
            summary = summarize_deltas(signed_deltas, args.bootstrap_samples, rng, cluster_payload)
            raw_summary = summarize_deltas(raw_deltas, args.bootstrap_samples, rng, cluster_payload)
            output_rows.append(
                {
                    "candidate_agent": args.candidate_agent,
                    "baseline_agent": baseline_agent,
                    "metric": metric,
                    "higher_is_better": metric not in LOWER_IS_BETTER,
                    "signed_positive_favors_candidate": True,
                    **summary,
                    "raw_mean_delta_candidate_minus_baseline": raw_summary["mean_delta"],
                    "raw_ci95_low": raw_summary["ci95_low"],
                    "raw_ci95_high": raw_summary["ci95_high"],
                }
            )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "paired_statistics.csv"
    json_path = output_root / "paired_statistics.json"
    write_csv(csv_path, output_rows)
    json_path.write_text(json.dumps({"rows": output_rows, "source_rows_path": args.rows_path}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("top journal statistics complete")
    print(f"paired_statistics_csv: {csv_path}")
    print(f"paired_statistics_json: {json_path}")


if __name__ == "__main__":
    main()
