"""????????????"""

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
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    aggregate_rows,
    apply_frozen_window_plan,
    apply_adapter_capacity_scale,
    apply_dependency_simplification,
    audit_checkpoint_map,
    build_mechanism_diagnosis,
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
    summary_to_row,
    write_rows_csv,
    MAIN_RESULT_METRICS,
)

BENCHMARK_AGENT_CHOICES = list_evaluable_agents()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="????????? sweep")
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
    parser.add_argument("--prediction_noise_std_list", nargs="+", type=float, default=[0.0, 0.2])
    parser.add_argument("--prediction_confidence_scale_list", nargs="+", type=float, default=[1.0, 0.7])
    parser.add_argument("--prediction_delay_steps_list", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--drop_handoff_prediction_prob_list", nargs="+", type=float, default=[0.0, 0.3])
    parser.add_argument("--adapter_capacity_scale_list", nargs="+", type=float, default=[1.0, 0.5])
    parser.add_argument("--dag_dependency_drop_rate_list", nargs="+", type=float, default=[0.0, 0.3])
    parser.add_argument("--workflow_size_buckets", nargs="+", default=["all", "medium"])
    parser.add_argument("--critical_path_buckets", nargs="+", default=["all", "medium"])
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "benchmarks" / "robustness"))
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


def build_robustness_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = [{
        "setting_id": "clean",
        "prediction_noise_std": 0.0,
        "prediction_confidence_scale": 1.0,
        "prediction_delay_steps": 0,
        "drop_handoff_prediction_prob": 0.0,
        "adapter_capacity_scale": 1.0,
        "dag_dependency_drop_rate": 0.0,
        "workflow_size_bucket": "all",
        "critical_path_bucket": "all",
    }]
    for value in args.prediction_noise_std_list:
        if value != 0.0:
            settings.append({**settings[0], "setting_id": f"prediction_noise_std={value}", "prediction_noise_std": value})
    for value in args.prediction_confidence_scale_list:
        if value != 1.0:
            settings.append({**settings[0], "setting_id": f"prediction_confidence_scale={value}", "prediction_confidence_scale": value})
    for value in args.prediction_delay_steps_list:
        if value != 0:
            settings.append({**settings[0], "setting_id": f"prediction_delay_steps={value}", "prediction_delay_steps": value})
    for value in args.drop_handoff_prediction_prob_list:
        if value != 0.0:
            settings.append({**settings[0], "setting_id": f"drop_handoff_prediction_prob={value}", "drop_handoff_prediction_prob": value})
    for value in args.adapter_capacity_scale_list:
        if value != 1.0:
            settings.append({**settings[0], "setting_id": f"adapter_capacity_scale={value}", "adapter_capacity_scale": value})
    for value in args.dag_dependency_drop_rate_list:
        if value != 0.0:
            settings.append({**settings[0], "setting_id": f"dag_dependency_drop_rate={value}", "dag_dependency_drop_rate": value})
    for value in args.workflow_size_buckets:
        if value != "all":
            settings.append({**settings[0], "setting_id": f"workflow_size_bucket={value}", "workflow_size_bucket": value})
    for value in args.critical_path_buckets:
        if value != "all":
            settings.append({**settings[0], "setting_id": f"critical_path_bucket={value}", "critical_path_bucket": value})
    unique = []
    seen = set()
    for item in settings:
        if item["setting_id"] in seen:
            continue
        unique.append(item)
        seen.add(item["setting_id"])
    return unique


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
    run_id = datetime.now().strftime("robustness_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_id
    rows: list[dict[str, Any]] = []
    settings = build_robustness_settings(args)
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
                    workflow_node_count_bucket=setting["workflow_size_bucket"],
                    critical_path_bucket=setting["critical_path_bucket"],
                    workflow_width_bucket="all",
                )
                if not filtered_workflows:
                    continue
                scaled_catalog, scaled_rsus, capacity_meta = apply_adapter_capacity_scale(
                    adapter_catalog,
                    base_bundle.rsu_states,
                    adapter_capacity_scale=float(setting["adapter_capacity_scale"]),
                )
                predictor_kwargs = {
                    "prediction_noise_std": float(setting["prediction_noise_std"]),
                    "prediction_confidence_scale": float(setting["prediction_confidence_scale"]),
                    "prediction_delay_steps": int(setting["prediction_delay_steps"]),
                    "drop_handoff_prediction_prob": float(setting["drop_handoff_prediction_prob"]),
                    "random_seed": seed,
                }
                for workflow_state in filtered_workflows:
                    simplified_workflow = apply_dependency_simplification(
                        workflow_state,
                        drop_rate=float(setting["dag_dependency_drop_rate"]),
                        random_seed=seed,
                    )
                    for agent_name in args.agents:
                        summary = run_real_episode(
                            root_dir=ROOT_DIR,
                            agent_name=agent_name,
                            checkpoint_map=seed_checkpoint_map,
                            workflow_state=workflow_state,
                            workflow_source_path=args.workflow_csv_path,
                            mobility_bundle=base_bundle,
                            seed=seed,
                            max_steps=args.max_steps,
                            mobility_source=args.mobility_source,
                            primary_vehicle_selection=args.primary_vehicle_selection,
                            run_metadata={
                                "script": "scripts/benchmark_robustness.py",
                                "robustness_setting_id": setting["setting_id"],
                                "window_mode": args.window_mode,
                                "mainline": mainline_label,
                            },
                            predictor_kwargs=predictor_kwargs,
                            adapter_catalog_override=scaled_catalog,
                            workflow_state_override=simplified_workflow,
                            rsu_states_override=scaled_rsus,
                        )
                        row = summary_to_row(summary)
                        row["robustness_setting_id"] = setting["setting_id"]
                        row.update(setting)
                        row.update(capacity_meta)
                        rows.append(row)

    aggregate_by_setting_and_agent = aggregate_rows(rows, group_keys=["robustness_setting_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    clean_reference = aggregate_by_setting_and_agent.get("clean|sa_ghmappo", None)
    degradation_summary: dict[str, Any] = {}
    if clean_reference is not None:
        for key, payload in aggregate_by_setting_and_agent.items():
            if key.startswith("clean|"):
                continue
            agent_name = payload["group"]["agent_name"]
            clean_key = f"clean|{agent_name}"
            if clean_key not in aggregate_by_setting_and_agent:
                continue
            degradation_summary[key] = {
                metric: round(float(payload["metrics"][metric]["mean"]) - float(aggregate_by_setting_and_agent[clean_key]["metrics"][metric]["mean"]), 6)
                for metric in MAIN_RESULT_METRICS
            }
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
        "robustness_settings": settings,
        "degradation_summary": degradation_summary,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(output_root / "benchmark_rows.csv", rows)
    print("robustness benchmark ??")
    print(f"aggregate_summary_path: {output_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
