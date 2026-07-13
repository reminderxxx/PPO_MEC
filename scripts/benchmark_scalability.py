"""?????????????"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import list_evaluable_agents
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    aggregate_rows,
    apply_frozen_window_plan,
    apply_adapter_capacity_scale,
    apply_adapter_type_proxy,
    audit_checkpoint_map,
    build_mechanism_diagnosis,
    build_rsu_layout_proxy,
    build_selected_workflow_states,
    checkpoint_map_for_seed,
    expand_checkpoint_aliases,
    filter_workflow_states_by_buckets,
    infer_benchmark_config_profile,
    load_seed_checkpoint_manifest,
    load_window_bundle,
    representative_checkpoint_map,
    resolve_window_candidates,
    run_real_episode,
    subsample_mobility_bundle_by_vehicle_count,
    summary_to_row,
    write_rows_csv,
    MAIN_RESULT_METRICS,
)

BENCHMARK_AGENT_CHOICES = list_evaluable_agents()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?????????? sweep")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_mode", type=str, default="activating_only", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--window_count", type=int, default=1)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_plan_path", type=str, default="")
    parser.add_argument("--rsu_count_list", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--rsu_coverage_radius_list", nargs="+", type=float, default=[10.0, 18.0])
    parser.add_argument("--adapter_type_count_list", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--workflow_node_count_buckets", nargs="+", default=["all", "medium"])
    parser.add_argument("--workflow_width_buckets", nargs="+", default=["all", "medium"])
    parser.add_argument("--vehicle_sample_count_list", nargs="+", type=int, default=[0, 8])
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "benchmarks" / "scalability"))
    return parser.parse_args()


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


def build_scalability_settings(args: argparse.Namespace) -> list[dict[str, object]]:
    settings = []
    for rsu_count in args.rsu_count_list:
        for coverage in args.rsu_coverage_radius_list:
            for adapter_type_count in args.adapter_type_count_list:
                for node_bucket in args.workflow_node_count_buckets:
                    for width_bucket in args.workflow_width_buckets:
                        for vehicle_count in args.vehicle_sample_count_list:
                            settings.append(
                                {
                                    "setting_id": f"rsu={rsu_count}|cov={coverage}|adapter={adapter_type_count}|node={node_bucket}|width={width_bucket}|veh={vehicle_count}",
                                    "rsu_count": rsu_count,
                                    "rsu_coverage_radius": coverage,
                                    "adapter_type_count": adapter_type_count,
                                    "workflow_node_count_bucket": node_bucket,
                                    "workflow_width_bucket": width_bucket,
                                    "vehicle_sample_count": vehicle_count,
                                }
                            )
    return settings


def main() -> None:
    args = parse_args()
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    base_checkpoint_map = expand_checkpoint_aliases(build_checkpoint_map(args))
    seed_checkpoint_manifest = load_seed_checkpoint_manifest(args.seed_checkpoint_manifest_path)
    audit_checkpoint_source_map = representative_checkpoint_map(
        base_checkpoint_map=base_checkpoint_map,
        seed_checkpoint_manifest=seed_checkpoint_manifest,
        seeds=args.seeds,
    )
    audit_bundle = audit_checkpoint_map(audit_checkpoint_source_map, args.agents)
    mobility_source_path, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector="max_handoff_candidate",
        window_count=args.window_count,
        window_scan_stride=args.window_scan_stride,
        random_seed=args.seeds[0] if args.seeds else 7,
        window_mode=args.window_mode,
    )
    selected_windows = [dict(window_candidate) for window_candidate in window_payload["selected_windows"]]
    if args.window_plan_path:
        window_payload = apply_frozen_window_plan(window_payload, args.window_plan_path)
        selected_windows = [dict(window_candidate) for window_candidate in window_payload["selected_windows"]]
    adapter_catalog = AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json")
    run_id = datetime.now().strftime("scalability_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_id
    rows = []
    settings = build_scalability_settings(args)
    selected_workflow_ids_by_seed: dict[str, list[str]] = {}

    for seed in args.seeds:
        seed_checkpoint_map = checkpoint_map_for_seed(
            base_checkpoint_map=base_checkpoint_map,
            seed_checkpoint_manifest=seed_checkpoint_manifest,
            seed=seed,
        )
        workflow_states = build_selected_workflow_states(
            workflow_csv_path=args.workflow_csv_path,
            max_workflows=args.max_workflows,
            workflow_selector=args.workflow_selector,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            random_seed=seed,
        )
        selected_workflow_ids_by_seed[str(seed)] = [workflow_state.workflow_id for workflow_state in workflow_states]
        for selected_window in selected_windows:
            base_bundle = load_window_bundle(
                root_dir=ROOT_DIR,
                mobility_source=args.mobility_source,
                mobility_csv_path=args.mobility_csv_path,
                lust_scenario_root=args.lust_scenario_root,
                max_mobility_rows=args.max_mobility_rows,
                rsu_layout=str(selected_window.get("recommended_rsu_layout", args.rsu_layout)),
                frame_offset=int(selected_window["frame_offset"]),
                window_length=int(selected_window["window_length"]),
                random_seed=seed,
            )
            base_bundle.rsu_metadata["window_class"] = selected_window.get("window_class", "unknown")
            for setting in settings:
                filtered_workflows = filter_workflow_states_by_buckets(
                    workflow_states,
                    workflow_node_count_bucket=str(setting["workflow_node_count_bucket"]),
                    critical_path_bucket="all",
                    workflow_width_bucket=str(setting["workflow_width_bucket"]),
                )
                if not filtered_workflows:
                    continue
                layout_string, layout_meta = build_rsu_layout_proxy(base_bundle, rsu_count=int(setting["rsu_count"]), rsu_coverage_radius=float(setting["rsu_coverage_radius"]))
                scaled_bundle = load_window_bundle(
                    root_dir=ROOT_DIR,
                    mobility_source=args.mobility_source,
                    mobility_csv_path=args.mobility_csv_path,
                    lust_scenario_root=args.lust_scenario_root,
                    max_mobility_rows=args.max_mobility_rows,
                    rsu_layout=layout_string,
                    frame_offset=int(selected_window["frame_offset"]),
                    window_length=int(selected_window["window_length"]),
                    random_seed=seed,
                )
                scaled_bundle.rsu_metadata["window_class"] = selected_window.get("window_class", "unknown")
                sampled_bundle, vehicle_meta = subsample_mobility_bundle_by_vehicle_count(scaled_bundle, vehicle_sample_count=int(setting["vehicle_sample_count"]))
                scaled_catalog, scaled_rsus, capacity_meta = apply_adapter_capacity_scale(adapter_catalog, sampled_bundle.rsu_states, adapter_capacity_scale=1.0)
                for workflow_state in filtered_workflows:
                    workflow_proxy, adapter_proxy_catalog, adapter_type_meta = apply_adapter_type_proxy(
                        workflow_state,
                        scaled_catalog,
                        adapter_type_count=int(setting["adapter_type_count"]),
                    )
                    for agent_name in args.agents:
                        summary = run_real_episode(
                            root_dir=ROOT_DIR,
                            agent_name=agent_name,
                            checkpoint_map=seed_checkpoint_map,
                            workflow_state=workflow_state,
                            workflow_source_path=args.workflow_csv_path,
                            mobility_bundle=sampled_bundle,
                            seed=seed,
                            max_steps=args.max_steps,
                            mobility_source=args.mobility_source,
                            primary_vehicle_selection=args.primary_vehicle_selection,
                            run_metadata={
                                "script": "scripts/benchmark_scalability.py",
                                "scalability_setting_id": setting["setting_id"],
                                "window_mode": args.window_mode,
                                "mainline": mainline_label,
                            },
                            adapter_catalog_override=adapter_proxy_catalog,
                            workflow_state_override=workflow_proxy,
                            rsu_states_override=scaled_rsus,
                        )
                        row = summary_to_row(summary)
                        row["scalability_setting_id"] = setting["setting_id"]
                        row.update(setting)
                        row.update(layout_meta)
                        row.update(vehicle_meta)
                        row.update(capacity_meta)
                        row.update(adapter_type_meta)
                        rows.append(row)

    aggregate_by_setting_and_agent = aggregate_rows(rows, group_keys=["scalability_setting_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_summary = {
        "run_id": run_id,
        "config_profile": infer_benchmark_config_profile(audit_bundle["checkpoint_audit"], args.agents),
        "mainline": mainline_label,
        "mobility_source": args.mobility_source,
        "mobility_source_path": mobility_source_path,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "window_mode": args.window_mode,
        "selected_window_plan": selected_windows,
        "selected_window_plan_by_strata": window_payload.get("selected_window_plan_by_strata", {}),
        "selected_workflow_ids_by_seed": selected_workflow_ids_by_seed,
        "checkpoint_map": audit_checkpoint_source_map,
        "seed_checkpoint_manifest_path": args.seed_checkpoint_manifest_path,
        "seed_checkpoint_manifest": seed_checkpoint_manifest,
        "checkpoint_audit": audit_bundle["checkpoint_audit"],
        "episode_count": len(rows),
        "aggregate_by_agent": aggregate_by_agent,
        "pairwise_comparison": {},
        "mechanism_diagnosis": build_mechanism_diagnosis(rows),
        "win_tie_loss_summary": {},
        "aggregate_by_setting_and_agent": aggregate_by_setting_and_agent,
        "proxy_scaling": True,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(output_root / "benchmark_rows.csv", rows)
    print("scalability benchmark ??")
    print(f"aggregate_summary_path: {output_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
