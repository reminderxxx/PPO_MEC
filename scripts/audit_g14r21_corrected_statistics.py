#!/usr/bin/env python3
"""Independent raw-row audit of G14R21 window sign tests and Holm values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

LOWER_IS_BETTER = {"transfer_mb_per_request", "end_to_end_workflow_delay"}
TOLERANCE = 1e-9


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def number(value: str) -> float | None:
    return None if value == "" else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corrected = json.loads(args.corrected_statistics.read_text(encoding="utf-8"))
    pair_fields = corrected["pair_keys"]
    candidate = corrected["pairwise_comparison_identity"]["candidate_agent"]
    baselines = corrected["pairwise_comparison_identity"]["baseline_agent_order"]
    metrics = list(dict.fromkeys(row["metric"] for row in corrected["rows"]))
    paired: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for path_text in corrected["source_rows_path"]:
        path = Path(path_text)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(path), *(f"{field}={row.get(field, '')}" for field in pair_fields if row.get(field, "") != ""))
                agents = paired.setdefault(key, {})
                if row["agent_name"] in agents:
                    raise ValueError("duplicate independent-audit pair row")
                agents[row["agent_name"]] = row
    recomputed = []
    for baseline in baselines:
        for metric in metrics:
            nested: dict[tuple[str, str], dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
            raw_values: list[float] = []
            total = available = candidate_only = baseline_only = both_missing = 0
            for agents in paired.values():
                if candidate not in agents or baseline not in agents:
                    continue
                total += 1
                candidate_row, baseline_row = agents[candidate], agents[baseline]
                left, right = number(candidate_row[metric]), number(baseline_row[metric])
                if left is None and right is None:
                    both_missing += 1
                    continue
                if left is None:
                    baseline_only += 1
                    continue
                if right is None:
                    candidate_only += 1
                    continue
                available += 1
                raw = left - right
                signed = -raw if metric in LOWER_IS_BETTER else raw
                raw_values.append(raw)
                outer = tuple(
                    candidate_row[field]
                    for field in ("source_segment_run_id", "window_id")
                    if candidate_row.get(field, "") != ""
                )
                inner = (candidate_row["seed"], candidate_row["workflow_id"])
                nested[outer][inner].append(signed)
            window_values = [fmean([fmean(values) for values in inner.values()]) for inner in nested.values()]
            wins = sum(value > TOLERANCE for value in window_values)
            losses = sum(value < -TOLERANCE for value in window_values)
            ties = len(window_values) - wins - losses
            recomputed.append(
                {
                    "baseline_agent": baseline,
                    "metric": metric,
                    "total_pair_count": total,
                    "available_paired_count": available,
                    "candidate_only_available_drop_count": candidate_only,
                    "baseline_only_available_drop_count": baseline_only,
                    "both_unavailable_drop_count": both_missing,
                    "raw_mean_delta_candidate_minus_baseline": fmean(raw_values) if raw_values else None,
                    "signed_outer_window_mean": fmean(window_values) if window_values else None,
                    "outer_window_count": len(window_values),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "sign_test_denominator": wins + losses,
                    "sign_test_pvalue_exact": exact_sign_p(wins, losses) if window_values else None,
                }
            )
    adjusted = holm([row["sign_test_pvalue_exact"] for row in recomputed if row["sign_test_pvalue_exact"] is not None])
    cursor = 0
    for row in recomputed:
        if row["sign_test_pvalue_exact"] is None:
            row["holm_sign_test_pvalue"] = None
        else:
            row["holm_sign_test_pvalue"] = adjusted[cursor]
            cursor += 1
    expected = {(row["baseline_agent"], row["metric"]): row for row in corrected["rows"]}
    mismatches = []
    for row in recomputed:
        observed = expected[(row["baseline_agent"], row["metric"])]
        for field in (
            "total_pair_count", "available_paired_count", "candidate_only_available_drop_count",
            "baseline_only_available_drop_count", "both_unavailable_drop_count", "wins", "ties", "losses",
            "sign_test_denominator", "sign_test_pvalue_exact", "holm_sign_test_pvalue",
        ):
            if observed[field] != row[field]:
                mismatches.append({"baseline_agent": row["baseline_agent"], "metric": row["metric"], "field": field, "corrected": observed[field], "independent": row[field]})
        if abs(observed["raw_mean_delta_candidate_minus_baseline"] - row["raw_mean_delta_candidate_minus_baseline"]) > 5e-7:
            mismatches.append({"baseline_agent": row["baseline_agent"], "metric": row["metric"], "field": "raw_mean_delta_candidate_minus_baseline"})
        if abs(observed["mean_delta"] - row["signed_outer_window_mean"]) > 5e-7:
            mismatches.append({"baseline_agent": row["baseline_agent"], "metric": row["metric"], "field": "mean_delta"})
    claim_counts: dict[str, int] = defaultdict(int)
    for row in corrected["rows"]:
        if row["ci95_low"] > 0:
            status = "supported"
        elif row["ci95_high"] < 0:
            status = "contradicted"
        else:
            status = "mixed"
        claim_counts[status] += 1
    report = {
        "independent_recalculation_version": "g14r21_raw_rows_v1",
        "implementation_independence": "does not import the production statistics or claim helper",
        "source_rows_path": corrected["source_rows_path"],
        "family_size": len(recomputed),
        "claim_counts": dict(sorted(claim_counts.items())),
        "holm_significant_count_alpha_0_05": sum((row["holm_sign_test_pvalue"] or 1.0) < 0.05 for row in recomputed),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches and len(recomputed) == 84,
        "rows": recomputed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "family_size": len(recomputed), "claim_counts": report["claim_counts"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
