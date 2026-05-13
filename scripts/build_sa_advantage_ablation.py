"""Build SA-GHMAPPO advantage ablation summaries from benchmark aggregates."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


FOCUS_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "handoff_ready_ratio",
    "mechanism_realization_rate",
    "adapter_state_migration_overhead",
]
DIAGNOSTIC_METRICS = [
    "continuity_guard_trigger_count",
    "target_mismatch_guard_count",
    "guard_prefetch_to_prepare_count",
    "guard_hard_override_count",
]
LOWER_IS_BETTER = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SA-GHMAPPO guard/imitation ablations from aggregate_summary.json files."
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant mapping in name=path form, e.g. guard=artifacts/.../aggregate_summary.json",
    )
    parser.add_argument("--candidate_agent", type=str, default="sa_ghmappo")
    parser.add_argument("--baseline_agent", type=str, default="popularity_cache_heuristic")
    parser.add_argument("--output_root", type=str, default="artifacts/benchmarks/sa_advantage_round1_ablation")
    parser.add_argument("--run_id", type=str, default="")
    return parser.parse_args()


def parse_variant_specs(raw_specs: list[str]) -> dict[str, Path]:
    variant_paths: dict[str, Path] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Variant spec must be name=path: {raw_spec}")
        name, raw_path = raw_spec.split("=", 1)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError(f"Variant name is empty: {raw_spec}")
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(path)
        variant_paths[clean_name] = path
    return variant_paths


def metric_mean(aggregate_by_agent: dict[str, Any], agent_name: str, metric_name: str) -> float:
    return float(
        aggregate_by_agent.get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
        .get("mean", 0.0)
        or 0.0
    )


def compare_metric(candidate_value: float, baseline_value: float, metric_name: str) -> str:
    delta = candidate_value - baseline_value
    effective_delta = -delta if metric_name in LOWER_IS_BETTER else delta
    if effective_delta > 1e-6:
        return "win"
    if effective_delta < -1e-6:
        return "loss"
    return "tie"


def summarize_variant(
    *,
    variant_name: str,
    aggregate_path: Path,
    candidate_agent: str,
    baseline_agent: str,
) -> dict[str, Any]:
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_by_agent = dict(payload.get("aggregate_by_agent", {}))
    missing_agents = [
        agent_name
        for agent_name in [candidate_agent, baseline_agent]
        if agent_name not in aggregate_by_agent
    ]
    if missing_agents:
        raise ValueError(
            f"{aggregate_path} missing required agents for ablation comparison: {', '.join(missing_agents)}"
        )
    row: dict[str, Any] = {
        "variant": variant_name,
        "aggregate_summary_path": str(aggregate_path),
        "run_id": payload.get("run_id", ""),
        "window_mode": payload.get("window_mode", ""),
        "config_profile": payload.get("config_profile", ""),
        "candidate_agent": candidate_agent,
        "baseline_agent": baseline_agent,
    }
    metric_details: dict[str, Any] = {}
    wins = 0
    losses = 0
    ties = 0
    for metric_name in FOCUS_METRICS:
        candidate_value = metric_mean(aggregate_by_agent, candidate_agent, metric_name)
        baseline_value = metric_mean(aggregate_by_agent, baseline_agent, metric_name)
        delta = candidate_value - baseline_value
        result = compare_metric(candidate_value, baseline_value, metric_name)
        wins += 1 if result == "win" else 0
        losses += 1 if result == "loss" else 0
        ties += 1 if result == "tie" else 0
        row[f"{metric_name}_candidate"] = round(candidate_value, 6)
        row[f"{metric_name}_baseline"] = round(baseline_value, 6)
        row[f"{metric_name}_delta"] = round(delta, 6)
        row[f"{metric_name}_result"] = result
        metric_details[metric_name] = {
            "candidate": round(candidate_value, 6),
            "baseline": round(baseline_value, 6),
            "delta_candidate_minus_baseline": round(delta, 6),
            "higher_is_better": metric_name not in LOWER_IS_BETTER,
            "result": result,
        }
    diagnostic_details: dict[str, Any] = {}
    for metric_name in DIAGNOSTIC_METRICS:
        candidate_value = metric_mean(aggregate_by_agent, candidate_agent, metric_name)
        baseline_value = metric_mean(aggregate_by_agent, baseline_agent, metric_name)
        delta = candidate_value - baseline_value
        row[f"{metric_name}_candidate"] = round(candidate_value, 6)
        row[f"{metric_name}_baseline"] = round(baseline_value, 6)
        row[f"{metric_name}_delta"] = round(delta, 6)
        diagnostic_details[metric_name] = {
            "candidate": round(candidate_value, 6),
            "baseline": round(baseline_value, 6),
            "delta_candidate_minus_baseline": round(delta, 6),
            "interpretation": "diagnostic_only",
        }
    minimum_success = bool(
        metric_details["total_reward"]["result"] == "win"
        and metric_details["backhaul_traffic_cost"]["result"] in {"win", "tie"}
        and (
            metric_details["workflow_continuity_rate"]["result"] in {"win", "tie"}
            or metric_details["handoff_failure_rate"]["result"] in {"win", "tie"}
        )
    )
    row["minimum_success_reached"] = minimum_success
    row["metric_wins"] = wins
    row["metric_losses"] = losses
    row["metric_ties"] = ties
    return {
        "row": row,
        "detail": {
            "variant": variant_name,
            "aggregate_summary_path": str(aggregate_path),
            "minimum_success_reached": minimum_success,
            "metrics": metric_details,
            "diagnostics": diagnostic_details,
            "win_loss_tie": {"win": wins, "loss": losses, "tie": ties},
            "source_summary_comparison": payload.get("comparison_against_popularity", {}),
            "source_summary_diagnosis": payload.get("sa_advantage_diagnosis", {}),
        },
    }


def main() -> None:
    args = parse_args()
    variant_paths = parse_variant_specs(args.variant)
    run_id = args.run_id or datetime.now().strftime("sa_advantage_ablation_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_variant(
            variant_name=variant_name,
            aggregate_path=aggregate_path,
            candidate_agent=args.candidate_agent,
            baseline_agent=args.baseline_agent,
        )
        for variant_name, aggregate_path in variant_paths.items()
    ]
    rows = [item["row"] for item in summaries]
    details = [item["detail"] for item in summaries]
    csv_path = output_root / "ablation_summary.csv"
    json_path = output_root / "ablation_summary.json"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "candidate_agent": args.candidate_agent,
                "baseline_agent": args.baseline_agent,
                "focus_metrics": FOCUS_METRICS,
                "diagnostic_metrics": DIAGNOSTIC_METRICS,
                "variants": details,
                "csv_path": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ablation_summary_csv: {csv_path}")
    print(f"ablation_summary_json: {json_path}")


if __name__ == "__main__":
    main()
