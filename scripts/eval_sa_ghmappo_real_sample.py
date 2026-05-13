"""评估 SA-GHMAPPO 主方法 checkpoint。"""

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

from src.evaluators.main_results_support import (
    aggregate_rows,
    build_selected_workflow_states,
    load_checkpoint_metadata,
    load_window_bundle,
    run_real_episode,
    summary_to_row,
    write_rows_csv,
)


COMPARISON_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
    "predictive_prefetch_request_count",
    "validated_predictive_prefetch_count",
    "migration_prepare_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "handoff_total_count",
    "successful_episode_rate",
    "mechanism_realization_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SA-GHMAPPO checkpoint")
    parser.add_argument("--agent_name", type=str, default="sa_ghmappo", choices=["sa_ghmappo"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--mobility_source", type=str, default="ngsim", choices=["ngsim", "lust"])
    parser.add_argument("--primary_vehicle_selection", type=str, default="stable_first", choices=["stable_first", "handoff_pressure"])
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--long_max_steps", type=int, default=0)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "eval" / "main_agents"))
    return parser.parse_args()


def build_checkpoint_map(args: argparse.Namespace) -> dict[str, str]:
    return {args.agent_name: args.checkpoint_path}


def run_horizon(args: argparse.Namespace, max_steps: int, checkpoint_map: dict[str, str], compare_agents: list[str]) -> dict[str, Any]:
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    horizon_name = "long_horizon" if max_steps != args.max_steps else "short_horizon"
    workflow_states = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
    mobility_bundle = load_window_bundle(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        random_seed=args.random_seed,
    )
    rows: list[dict[str, Any]] = []
    for workflow_state in workflow_states:
        for agent_name in compare_agents:
            summary = run_real_episode(
                root_dir=ROOT_DIR,
                agent_name=agent_name,
                checkpoint_map=checkpoint_map,
                workflow_state=workflow_state,
                workflow_source_path=args.workflow_csv_path,
                mobility_bundle=mobility_bundle,
                seed=args.random_seed,
                max_steps=max_steps,
                mobility_source=args.mobility_source,
                primary_vehicle_selection=args.primary_vehicle_selection,
                run_metadata={
                    "script": "scripts/eval_sa_ghmappo_real_sample.py",
                    "mainline": mainline_label,
                    "evaluation_agent": args.agent_name,
                    "horizon_name": horizon_name,
                    "primary_vehicle_selection": args.primary_vehicle_selection,
                },
            )
            row = summary_to_row(summary)
            row["horizon_name"] = horizon_name
            rows.append(row)
    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=COMPARISON_METRICS)
    return {
        "rows": rows,
        "aggregate_by_agent": aggregate_by_agent,
        "comparison": {},
    }


def main() -> None:
    args = parse_args()
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    checkpoint_map = build_checkpoint_map(args)
    compare_agents = [args.agent_name]

    checkpoint_metadata = load_checkpoint_metadata(args.checkpoint_path)
    horizon_results = {"short_horizon": run_horizon(args, args.max_steps, checkpoint_map, compare_agents)}
    if args.long_max_steps > 0 and args.long_max_steps != args.max_steps:
        horizon_results["long_horizon"] = run_horizon(args, args.long_max_steps, checkpoint_map, compare_agents)

    run_id = datetime.now().strftime(f"{args.agent_name}_eval_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / args.agent_name / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "mainline": mainline_label,
        "agent_name": args.agent_name,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_smoke_warning": checkpoint_metadata.get("is_smoke_checkpoint", False),
        "compare_agents": compare_agents,
        "horizon_results": horizon_results,
    }
    comparison: dict[str, Any] = {}
    eval_rows: list[dict[str, Any]] = []
    for horizon_name, payload in horizon_results.items():
        comparison[horizon_name] = payload["comparison"]
        eval_rows.extend(dict(row) for row in payload["rows"])
    eval_csv_path = output_root / "eval.csv"
    summary_path = output_root / "summary.json"
    agent_comparison_path = output_root / "agent_comparison.json"
    summary["eval_csv_path"] = str(eval_csv_path)
    write_rows_csv(eval_csv_path, eval_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    agent_comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print("sa_ghmappo evaluation complete")
    print(f"run_id: {run_id}")
    print(f"eval_csv_path: {eval_csv_path}")
    print(f"summary_path: {summary_path}")
    print(f"agent_comparison_path: {agent_comparison_path}")
    if checkpoint_metadata.get("is_smoke_checkpoint", False):
        print("warning: checkpoint comes from a smoke run and is not paper-claim ready")


if __name__ == "__main__":
    main()
