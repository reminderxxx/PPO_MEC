"""Export paper-facing tables from canonical benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.main_results_support import PAPER_PROTOCOL_FROZEN, PAPER_PROTOCOL_VERSION


DEFAULT_MIXED_PATH = (
    ROOT_DIR
    / "artifacts"
    / "benchmarks"
    / "main_results"
    / "main_results_mixed_informative_20260409_131356_413593"
    / "aggregate_summary.json"
)
DEFAULT_FULL_PATH = (
    ROOT_DIR
    / "artifacts"
    / "benchmarks"
    / "main_results"
    / "main_results_full_stratified_20260409_131356_432799"
    / "aggregate_summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "paper" / PAPER_PROTOCOL_VERSION
CORE_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
    "validated_predictive_prefetch_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "mechanism_realization_rate",
]
LOWER_IS_BETTER = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper artifacts from benchmark aggregate summaries.")
    parser.add_argument("--mixed_summary_path", type=str, default=str(DEFAULT_MIXED_PATH))
    parser.add_argument("--full_summary_path", type=str, default=str(DEFAULT_FULL_PATH))
    parser.add_argument("--gate_report_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_mean(agent_payload: dict[str, Any], metric: str) -> float:
    return float(agent_payload.get("metrics", {}).get(metric, {}).get("mean", 0.0) or 0.0)


def flatten_table_rows(summary: dict[str, Any], scope_name: str, strata_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strata_name, aggregate in strata_map.items():
        for agent_name, payload in sorted(aggregate.items()):
            metrics = payload.get("metrics", {})
            row: dict[str, Any] = {
                "source_scope": scope_name,
                "run_id": summary.get("run_id", ""),
                "window_mode": summary.get("window_mode", ""),
                "primary_vehicle_selection": summary.get("primary_vehicle_selection", ""),
                "strata": strata_name,
                "agent_name": agent_name,
                "episode_count": payload.get("episode_count", 0),
            }
            for metric in CORE_METRICS:
                stats = metrics.get(metric, {})
                row[f"{metric}_mean"] = stats.get("mean", 0.0)
                row[f"{metric}_std"] = stats.get("std", 0.0)
            rows.append(row)
    return rows


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


def build_delta_rows(summary: dict[str, Any], scope_name: str) -> list[dict[str, Any]]:
    aggregate = summary.get("aggregate_by_agent", {})
    sa_payload = aggregate.get("sa_ghmappo")
    if not isinstance(sa_payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for baseline_name, baseline_payload in sorted(aggregate.items()):
        if baseline_name == "sa_ghmappo":
            continue
        for metric in CORE_METRICS:
            sa_value = metric_mean(sa_payload, metric)
            baseline_value = metric_mean(baseline_payload, metric)
            delta = sa_value - baseline_value
            effective_delta = -delta if metric in LOWER_IS_BETTER else delta
            rows.append(
                {
                    "source_scope": scope_name,
                    "window_mode": summary.get("window_mode", ""),
                    "baseline_agent": baseline_name,
                    "metric": metric,
                    "sa_ghmappo": round(sa_value, 6),
                    "baseline": round(baseline_value, 6),
                    "delta_sa_minus_baseline": round(delta, 6),
                    "higher_is_better": metric not in LOWER_IS_BETTER,
                    "result": "win" if effective_delta > 1e-9 else "loss" if effective_delta < -1e-9 else "tie",
                }
            )
    return rows


def build_claim_summary(
    mixed_summary: dict[str, Any],
    full_summary: dict[str, Any],
    mixed_path: Path,
    full_path: Path,
    gate_report: dict[str, Any] | None,
) -> dict[str, Any]:
    def key_numbers(summary: dict[str, Any]) -> dict[str, Any]:
        aggregate = summary.get("aggregate_by_agent", {})
        return {
            agent_name: {
                metric: metric_mean(payload, metric)
                for metric in [
                    "total_reward",
                    "workflow_continuity_rate",
                    "handoff_failure_rate",
                    "backhaul_traffic_cost",
                    "mechanism_realization_rate",
                    "adapter_state_migration_overhead",
                ]
            }
            for agent_name, payload in sorted(aggregate.items())
        }

    return {
        "canonical_protocol_version": PAPER_PROTOCOL_VERSION,
        "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
        "main_table_source": str(mixed_path),
        "supplementary_table_source": str(full_path),
        "recommended_main_table": "mixed_informative",
        "recommended_supplementary_table": "full_stratified",
        "gate_report_source": gate_report.get("run_id") if gate_report else "",
        "gate_passed": bool(gate_report.get("passed")) if gate_report else None,
        "formal_contract_ready": bool(gate_report.get("formal_contract", {}).get("ready")) if gate_report else None,
        "paper_claim_ready": bool(gate_report.get("paper_claim_ready")) if gate_report else None,
        "mixed_informative_key_numbers": key_numbers(mixed_summary),
        "full_stratified_key_numbers": key_numbers(full_summary),
        "claim_boundary": [
            "Formal_v2 main gate is paper-claim ready only together with its frozen manifest and benchmark summaries.",
            "Ablation, robustness, scalability, and paired statistics should be cited from the matching support-suite outputs.",
        ],
    }


def main() -> None:
    args = parse_args()
    mixed_path = Path(args.mixed_summary_path)
    full_path = Path(args.full_summary_path)
    output_root = Path(args.output_root)
    gate_report = load_json(Path(args.gate_report_path)) if args.gate_report_path else None

    mixed_summary = load_json(mixed_path)
    full_summary = load_json(full_path)
    output_root.mkdir(parents=True, exist_ok=True)

    mixed_rows = flatten_table_rows(
        mixed_summary,
        scope_name="paper_main_table",
        strata_map={
            "overall": mixed_summary.get("aggregate_by_agent", {}),
            "mechanism_activating": mixed_summary.get("aggregate_mechanism_windows_by_agent", {}),
            "active_non_mechanism": mixed_summary.get("aggregate_active_non_mechanism_windows_by_agent", {}),
        },
    )
    full_rows = flatten_table_rows(
        full_summary,
        scope_name="paper_supplementary_table",
        strata_map={
            "overall": full_summary.get("aggregate_by_agent", {}),
            "mechanism_activating": full_summary.get("aggregate_mechanism_windows_by_agent", {}),
            "active_non_mechanism": full_summary.get("aggregate_active_non_mechanism_windows_by_agent", {}),
            "idle_or_sparse": full_summary.get("aggregate_idle_or_sparse_windows_by_agent", {}),
        },
    )
    delta_rows = build_delta_rows(mixed_summary, "paper_main_table") + build_delta_rows(full_summary, "paper_supplementary_table")

    write_csv(output_root / "paper_main_table.csv", mixed_rows)
    write_csv(output_root / "paper_supplementary_table.csv", full_rows)
    write_csv(output_root / "paper_baseline_deltas.csv", delta_rows)

    paper_main_json = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
        "main_table_source": str(mixed_path),
        "supplementary_table_source": str(full_path),
        "gate_report_path": args.gate_report_path,
        "paper_main_table_rows": mixed_rows,
        "paper_supplementary_table_rows": full_rows,
        "paper_baseline_delta_rows": delta_rows,
    }
    (output_root / "paper_main_table.json").write_text(json.dumps(paper_main_json, ensure_ascii=False, indent=2), encoding="utf-8")

    claim_summary = build_claim_summary(
        mixed_summary,
        full_summary,
        mixed_path=mixed_path,
        full_path=full_path,
        gate_report=gate_report,
    )
    (output_root / "paper_claim_summary.json").write_text(json.dumps(claim_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("paper artifacts exported")
    print(f"output_root: {output_root}")
    print(f"paper_main_table_csv: {output_root / 'paper_main_table.csv'}")
    print(f"paper_supplementary_table_csv: {output_root / 'paper_supplementary_table.csv'}")
    print(f"paper_baseline_deltas_csv: {output_root / 'paper_baseline_deltas.csv'}")
    print(f"paper_claim_summary_json: {output_root / 'paper_claim_summary.json'}")


if __name__ == "__main__":
    main()
