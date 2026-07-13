#!/usr/bin/env python3
"""Diagnose strict-full candidate/reference gaps without touching hidden holdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any


PROTOCOL_VERSION = "strict_full_failure_diagnosis_v1_20260621"
PAIR_KEYS = ("seed", "window_id", "workflow_id")
DEFAULT_METRICS = (
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "mechanism_realization_rate",
    "service_success_count",
    "service_wait_sum",
    "service_restart_count",
    "adapter_hit_count",
    "adapter_miss_count",
    "local_exec_count",
    "current_rsu_exec_count",
    "next_rsu_exec_count",
    "neighbor_rsu_exec_count",
    "cloud_exec_count",
    "prefetch_action_count",
    "migration_action_count",
    "prefetch_success_count",
    "migration_success_count",
    "mechanism_attempt_count",
    "mechanism_validated_success_count",
    "continuity_guard_trigger_count",
    "guard_prefetch_to_prepare_count",
    "guard_hard_override_count",
    "action_projection_count",
    "delay_reward_component",
    "cache_reward_component",
    "handoff_reward_component",
    "backhaul_reward_component",
    "failure_reward_component",
    "continuity_reward_component",
    "service_reward_component",
    "mechanism_exploration_reward_component",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        action="append",
        required=True,
        help="LABEL=/path/to/benchmark_rows.csv; do not pass sealed hidden results",
    )
    parser.add_argument("--candidate_agent", default="sa_ghmappo")
    parser.add_argument("--reference_agent", default="dt_handoff_drl")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if "hidden" in label.lower():
        raise ValueError("sealed hidden holdout must not be used for failure diagnosis")
    return label.strip(), Path(raw_path)


def float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pair_rows(
    rows: list[dict[str, str]],
    candidate_agent: str,
    reference_agent: str,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        agent = str(row.get("agent_name", ""))
        if agent not in {candidate_agent, reference_agent}:
            continue
        key = tuple(str(row.get(field, "")) for field in PAIR_KEYS)
        indexed[key][agent] = row
    pairs = [
        (agents[candidate_agent], agents[reference_agent])
        for agents in indexed.values()
        if candidate_agent in agents and reference_agent in agents
    ]
    if not pairs:
        raise ValueError(f"no paired {candidate_agent}/{reference_agent} rows")
    return pairs


def build_delta_rows(
    split_label: str,
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate, reference in pairs:
        row: dict[str, Any] = {
            "split": split_label,
            **{key: candidate.get(key, "") for key in PAIR_KEYS},
            "window_class": candidate.get("window_class", "unknown"),
        }
        for metric in metrics:
            candidate_value = float_or_none(candidate.get(metric))
            reference_value = float_or_none(reference.get(metric))
            if candidate_value is None or reference_value is None:
                continue
            row[f"candidate_{metric}"] = candidate_value
            row[f"reference_{metric}"] = reference_value
            row[f"delta_{metric}"] = candidate_value - reference_value
        output.append(row)
    return output


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(fmean(values), 6),
        "median": round(median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return round(numerator / denominator, 6) if denominator else None


def aggregate_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["window_class"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split_label, window_class), group_rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "split": split_label,
            "window_class": window_class,
            "paired_count": len(group_rows),
        }
        for metric in DEFAULT_METRICS:
            values = [
                float(row[f"delta_{metric}"])
                for row in group_rows
                if f"delta_{metric}" in row
            ]
            if values:
                record[f"mean_delta_{metric}"] = round(fmean(values), 6)
        output.append(record)
    return output


def rank_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["window_id"]), str(row["window_class"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split_label, window_id, window_class), group_rows in groups.items():
        record: dict[str, Any] = {
            "split": split_label,
            "window_id": window_id,
            "window_class": window_class,
            "paired_count": len(group_rows),
        }
        for metric in (
            "total_reward",
            "workflow_continuity_rate",
            "handoff_failure_rate",
            "handoff_ready_ratio",
            "local_exec_count",
            "current_rsu_exec_count",
            "next_rsu_exec_count",
            "prefetch_action_count",
            "migration_action_count",
        ):
            key = f"delta_{metric}"
            values = [float(row[key]) for row in group_rows if key in row]
            if values:
                record[f"mean_{key}"] = round(fmean(values), 6)
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            float(row.get("mean_delta_workflow_continuity_rate", 0.0)),
            -float(row.get("mean_delta_handoff_failure_rate", 0.0)),
        ),
    )


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_summary = {}
    for metric in DEFAULT_METRICS:
        values = [float(row[f"delta_{metric}"]) for row in rows if f"delta_{metric}" in row]
        if values:
            metric_summary[metric] = summarize(values)
    continuity_worse = [row for row in rows if float(row.get("delta_workflow_continuity_rate", 0.0)) < 0.0]
    failure_worse = [row for row in rows if float(row.get("delta_handoff_failure_rate", 0.0)) > 0.0]
    both_worse = [
        row
        for row in rows
        if float(row.get("delta_workflow_continuity_rate", 0.0)) < 0.0
        and float(row.get("delta_handoff_failure_rate", 0.0)) > 0.0
    ]
    target_metrics = (
        "local_exec_count",
        "current_rsu_exec_count",
        "next_rsu_exec_count",
        "neighbor_rsu_exec_count",
        "cloud_exec_count",
        "prefetch_action_count",
        "migration_action_count",
        "continuity_guard_trigger_count",
        "guard_prefetch_to_prepare_count",
        "action_projection_count",
        "service_wait_sum",
        "adapter_miss_count",
    )
    correlations: dict[str, dict[str, float | None]] = {}
    for metric in target_metrics:
        filtered = [
            row
            for row in rows
            if f"delta_{metric}" in row and "delta_total_reward" in row
        ]
        metric_values = [float(row[f"delta_{metric}"]) for row in filtered]
        correlations[metric] = {
            "with_reward_delta": pearson(metric_values, [float(row["delta_total_reward"]) for row in filtered]),
            "with_continuity_delta": pearson(
                metric_values,
                [float(row.get("delta_workflow_continuity_rate", 0.0)) for row in filtered],
            ),
            "with_failure_delta": pearson(
                metric_values,
                [float(row.get("delta_handoff_failure_rate", 0.0)) for row in filtered],
            ),
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "previously opened formal/diagnostic holdout only; sealed hidden holdout prohibited",
        "paired_count": len(rows),
        "metric_summary": metric_summary,
        "failure_counts": {
            "continuity_worse": len(continuity_worse),
            "handoff_failure_worse": len(failure_worse),
            "both_worse": len(both_worse),
        },
        "driver_correlations": correlations,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    delta_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for raw_spec in args.rows:
        split_label, rows_path = parse_labeled_path(raw_spec)
        pairs = pair_rows(load_rows(rows_path), args.candidate_agent, args.reference_agent)
        delta_rows.extend(build_delta_rows(split_label, pairs))
        sources[split_label] = str(rows_path.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = aggregate_delta_rows(delta_rows)
    window_rows = rank_windows(delta_rows)
    report = build_report(delta_rows)
    report.update(
        {
            "candidate_agent": args.candidate_agent,
            "reference_agent": args.reference_agent,
            "sources": sources,
            "worst_continuity_windows": window_rows[:10],
        }
    )
    write_csv(args.output_dir / "paired_failure_deltas.csv", delta_rows)
    write_csv(args.output_dir / "aggregate_failure_deltas.csv", aggregate_rows)
    write_csv(args.output_dir / "window_failure_ranking.csv", window_rows)
    (args.output_dir / "failure_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
