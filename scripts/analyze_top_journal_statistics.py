"""Paired statistics for top-journal benchmark rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import NormalDist, fmean, stdev
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
STATISTICS_PROTOCOL_VERSION = "hierarchical_window_bootstrap_v1_20260621"


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
    parser.add_argument(
        "--outer_cluster_keys",
        nargs="*",
        default=[],
        help="Outer hierarchical bootstrap keys. For strict mobility claims use window_id.",
    )
    parser.add_argument(
        "--inner_cluster_keys",
        nargs="*",
        default=[],
        help="Inner hierarchical bootstrap keys nested within outer clusters, e.g. seed workflow_id.",
    )
    parser.add_argument(
        "--ci_method",
        choices=["percentile", "bca"],
        default="bca",
        help="Primary 95%% confidence interval method; both percentile and BCa bounds are always recorded.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()
    if args.cluster_keys and args.outer_cluster_keys:
        parser.error("--cluster_keys cannot be combined with --outer_cluster_keys")
    if args.inner_cluster_keys and not args.outer_cluster_keys:
        parser.error("--inner_cluster_keys requires --outer_cluster_keys")
    return args


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
    bootstrap_means = []
    for _ in range(max(1, bootstrap_samples)):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
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
    bootstrap_means = []
    for _ in range(max(1, bootstrap_samples)):
        sample: list[float] = []
        for _ in cluster_values:
            sample.extend(cluster_values[rng.randrange(len(cluster_values))])
        bootstrap_means.append(fmean(sample) if sample else 0.0)
    return bootstrap_means


def nested_cluster_values(
    deltas: list[float],
    outer_clusters: list[tuple[str, ...]],
    inner_clusters: list[tuple[str, ...]] | None,
) -> dict[tuple[str, ...], dict[tuple[str, ...], list[float]]]:
    nested: dict[tuple[str, ...], dict[tuple[str, ...], list[float]]] = {}
    for index, (delta, outer_cluster) in enumerate(zip(deltas, outer_clusters)):
        inner_cluster = inner_clusters[index] if inner_clusters is not None else ("all_inner_rows",)
        nested.setdefault(outer_cluster, {}).setdefault(inner_cluster, []).append(delta)
    return nested


def hierarchical_outer_means(
    deltas: list[float],
    outer_clusters: list[tuple[str, ...]],
    inner_clusters: list[tuple[str, ...]] | None,
) -> list[float]:
    nested = nested_cluster_values(deltas, outer_clusters, inner_clusters)
    outer_means: list[float] = []
    for inner_groups in nested.values():
        inner_means = [fmean(values) for values in inner_groups.values()]
        outer_means.append(fmean(inner_means))
    return outer_means


def bootstrap_hierarchical_means(
    deltas: list[float],
    outer_clusters: list[tuple[str, ...]],
    inner_clusters: list[tuple[str, ...]] | None,
    bootstrap_samples: int,
    rng: random.Random,
) -> list[float]:
    nested = nested_cluster_values(deltas, outer_clusters, inner_clusters)
    outer_values = list(nested.values())
    if not outer_values:
        return []
    bootstrap_means: list[float] = []
    for _ in range(max(1, bootstrap_samples)):
        sampled_outer_means: list[float] = []
        for _ in outer_values:
            selected_outer = outer_values[rng.randrange(len(outer_values))]
            inner_values = list(selected_outer.values())
            sampled_inner_means: list[float] = []
            for _ in inner_values:
                selected_inner = inner_values[rng.randrange(len(inner_values))]
                sampled_inner_means.append(fmean(selected_inner))
            sampled_outer_means.append(fmean(sampled_inner_means))
        bootstrap_means.append(fmean(sampled_outer_means))
    return bootstrap_means


def jackknife_pair_means(deltas: list[float]) -> list[float]:
    if len(deltas) < 2:
        return []
    return [fmean(deltas[:index] + deltas[index + 1 :]) for index in range(len(deltas))]


def jackknife_cluster_means(deltas: list[float], clusters: list[tuple[str, ...]]) -> list[float]:
    unique_clusters = list(dict.fromkeys(clusters))
    if len(unique_clusters) < 2:
        return []
    estimates: list[float] = []
    for omitted in unique_clusters:
        retained = [delta for delta, cluster in zip(deltas, clusters) if cluster != omitted]
        if retained:
            estimates.append(fmean(retained))
    return estimates


def jackknife_hierarchical_means(
    deltas: list[float],
    outer_clusters: list[tuple[str, ...]],
    inner_clusters: list[tuple[str, ...]] | None,
) -> list[float]:
    unique_outer = list(dict.fromkeys(outer_clusters))
    if len(unique_outer) < 2:
        return []
    estimates: list[float] = []
    for omitted in unique_outer:
        retained_indices = [index for index, cluster in enumerate(outer_clusters) if cluster != omitted]
        retained_deltas = [deltas[index] for index in retained_indices]
        retained_outer = [outer_clusters[index] for index in retained_indices]
        retained_inner = [inner_clusters[index] for index in retained_indices] if inner_clusters is not None else None
        outer_means = hierarchical_outer_means(retained_deltas, retained_outer, retained_inner)
        if outer_means:
            estimates.append(fmean(outer_means))
    return estimates


def bca_interval(
    bootstrap_means: list[float],
    observed: float,
    jackknife_estimates: list[float],
    lower_q: float = 0.025,
    upper_q: float = 0.975,
) -> tuple[float, float, bool]:
    if len(bootstrap_means) < 2 or len(jackknife_estimates) < 3:
        return percentile(bootstrap_means, lower_q), percentile(bootstrap_means, upper_q), False
    distribution = NormalDist()
    below = sum(1 for value in bootstrap_means if value < observed)
    probability = min(
        1.0 - 0.5 / len(bootstrap_means),
        max(0.5 / len(bootstrap_means), below / len(bootstrap_means)),
    )
    bias_correction = distribution.inv_cdf(probability)
    jackknife_mean = fmean(jackknife_estimates)
    centered = [jackknife_mean - value for value in jackknife_estimates]
    numerator = sum(value**3 for value in centered)
    denominator_base = sum(value**2 for value in centered)
    acceleration = numerator / (6.0 * denominator_base**1.5) if denominator_base > 1e-18 else 0.0

    def adjusted_quantile(quantile: float) -> float:
        normal_quantile = distribution.inv_cdf(quantile)
        denominator = 1.0 - acceleration * (bias_correction + normal_quantile)
        if abs(denominator) < 1e-12:
            return quantile
        adjusted = distribution.cdf(
            bias_correction + (bias_correction + normal_quantile) / denominator
        )
        return min(1.0, max(0.0, adjusted))

    return (
        percentile(bootstrap_means, adjusted_quantile(lower_q)),
        percentile(bootstrap_means, adjusted_quantile(upper_q)),
        True,
    )


def summarize_deltas(
    deltas: list[float],
    bootstrap_samples: int,
    rng: random.Random,
    clusters: list[tuple[str, ...]] | None = None,
    outer_clusters: list[tuple[str, ...]] | None = None,
    inner_clusters: list[tuple[str, ...]] | None = None,
    ci_method: str = "bca",
) -> dict[str, Any]:
    if not deltas:
        return {
            "paired_count": 0,
            "bootstrap_unit": "hierarchical" if outer_clusters else ("cluster" if clusters else "pair"),
            "cluster_count": 0,
            "outer_cluster_count": 0,
            "inner_cluster_count": 0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_method": ci_method,
            "percentile_ci95_low": 0.0,
            "percentile_ci95_high": 0.0,
            "bca_ci95_low": 0.0,
            "bca_ci95_high": 0.0,
            "bca_available": False,
            "std_delta": 0.0,
            "cohen_dz": 0.0,
            "outer_cluster_std_delta": 0.0,
            "outer_cluster_cohen_d": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_pvalue": 1.0,
        }
    std_delta = stdev(deltas) if len(deltas) > 1 else 0.0
    outer_cluster_means: list[float] = []
    inner_cluster_count = 0
    if outer_clusters is not None and len(outer_clusters) == len(deltas):
        inner_payload = inner_clusters if inner_clusters is not None and len(inner_clusters) == len(deltas) else None
        outer_cluster_means = hierarchical_outer_means(deltas, outer_clusters, inner_payload)
        mean_delta = fmean(outer_cluster_means)
        bootstrap_means = bootstrap_hierarchical_means(
            deltas,
            outer_clusters,
            inner_payload,
            bootstrap_samples,
            rng,
        )
        jackknife_estimates = jackknife_hierarchical_means(deltas, outer_clusters, inner_payload)
        bootstrap_unit = "hierarchical"
        cluster_count = len(set(outer_clusters))
        outer_cluster_count = cluster_count
        inner_cluster_count = len(set(zip(outer_clusters, inner_payload))) if inner_payload is not None else len(deltas)
    elif clusters is not None and len(clusters) == len(deltas):
        mean_delta = fmean(deltas)
        bootstrap_means = bootstrap_cluster_means(deltas, clusters, bootstrap_samples, rng)
        jackknife_estimates = jackknife_cluster_means(deltas, clusters)
        bootstrap_unit = "cluster"
        cluster_count = len(set(clusters))
        outer_cluster_count = cluster_count
    else:
        mean_delta = fmean(deltas)
        bootstrap_means = bootstrap_pair_means(deltas, bootstrap_samples, rng)
        jackknife_estimates = jackknife_pair_means(deltas)
        bootstrap_unit = "pair"
        cluster_count = len(deltas)
        outer_cluster_count = 0
    percentile_low = percentile(bootstrap_means, 0.025)
    percentile_high = percentile(bootstrap_means, 0.975)
    bca_low, bca_high, bca_available = bca_interval(
        bootstrap_means,
        mean_delta,
        jackknife_estimates,
    )
    if ci_method == "bca" and bca_available:
        primary_low, primary_high, primary_method = bca_low, bca_high, "bca"
    else:
        primary_low, primary_high, primary_method = percentile_low, percentile_high, "percentile"
    outer_cluster_std = stdev(outer_cluster_means) if len(outer_cluster_means) > 1 else 0.0
    wins = sum(1 for value in deltas if value > 1e-9)
    losses = sum(1 for value in deltas if value < -1e-9)
    ties = len(deltas) - wins - losses
    return {
        "paired_count": len(deltas),
        "bootstrap_unit": bootstrap_unit,
        "cluster_count": cluster_count,
        "outer_cluster_count": outer_cluster_count,
        "inner_cluster_count": inner_cluster_count,
        "mean_delta": round(mean_delta, 6),
        "ci95_low": round(primary_low, 6),
        "ci95_high": round(primary_high, 6),
        "ci95_method": primary_method,
        "percentile_ci95_low": round(percentile_low, 6),
        "percentile_ci95_high": round(percentile_high, 6),
        "bca_ci95_low": round(bca_low, 6),
        "bca_ci95_high": round(bca_high, 6),
        "bca_available": bca_available,
        "std_delta": round(std_delta, 6),
        "cohen_dz": round(mean_delta / std_delta, 6) if std_delta > 1e-12 else 0.0,
        "outer_cluster_std_delta": round(outer_cluster_std, 6),
        "outer_cluster_cohen_d": round(mean_delta / outer_cluster_std, 6) if outer_cluster_std > 1e-12 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_pvalue": round(exact_sign_test_pvalue(wins, losses), 6),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * len(pvalues)
    running_max = 0.0
    family_size = len(pvalues)
    for rank, (original_index, pvalue) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * pvalue)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


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
            delta_outer_clusters: list[tuple[str, ...]] = []
            delta_inner_clusters: list[tuple[str, ...]] = []
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
                delta_outer_clusters.append(cluster_key(candidate_row, args.outer_cluster_keys, key))
                delta_inner_clusters.append(cluster_key(candidate_row, args.inner_cluster_keys, key))
            cluster_payload = delta_clusters if args.cluster_keys else None
            outer_payload = delta_outer_clusters if args.outer_cluster_keys else None
            inner_payload = delta_inner_clusters if args.inner_cluster_keys else None
            summary = summarize_deltas(
                signed_deltas,
                args.bootstrap_samples,
                rng,
                cluster_payload,
                outer_payload,
                inner_payload,
                args.ci_method,
            )
            raw_summary = summarize_deltas(
                raw_deltas,
                args.bootstrap_samples,
                rng,
                cluster_payload,
                outer_payload,
                inner_payload,
                args.ci_method,
            )
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

    adjusted_sign_tests = holm_adjust([float(row["sign_test_pvalue"]) for row in output_rows])
    for row, adjusted_pvalue in zip(output_rows, adjusted_sign_tests):
        row["holm_sign_test_pvalue"] = round(adjusted_pvalue, 6)
        row["holm_family"] = "all_baseline_metric_comparisons"

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "paired_statistics.csv"
    json_path = output_root / "paired_statistics.json"
    write_csv(csv_path, output_rows)
    json_path.write_text(
        json.dumps(
            {
                "statistics_protocol_version": STATISTICS_PROTOCOL_VERSION,
                "pair_keys": args.pair_keys,
                "legacy_cluster_keys": args.cluster_keys,
                "outer_cluster_keys": args.outer_cluster_keys,
                "inner_cluster_keys": args.inner_cluster_keys,
                "requested_ci_method": args.ci_method,
                "bootstrap_samples": args.bootstrap_samples,
                "rows": output_rows,
                "source_rows_path": args.rows_path,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("top journal statistics complete")
    print(f"paired_statistics_csv: {csv_path}")
    print(f"paired_statistics_json: {json_path}")


if __name__ == "__main__":
    main()
