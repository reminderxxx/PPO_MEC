"""???????? predictor robustness benchmark?"""

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
from src.evaluators.main_results_support import (
    MAIN_RESULT_METRICS,
    PAPER_PROTOCOL_FROZEN,
    PAPER_PROTOCOL_VERSION,
    aggregate_rows,
    build_mechanism_diagnosis,
    build_selected_workflow_states,
    checkpoint_map_for_seed,
    expand_checkpoint_aliases,
    infer_benchmark_config_profile,
    load_checkpoint_metadata,
    load_seed_checkpoint_manifest,
    load_window_bundle,
    representative_checkpoint_map,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
    write_rows_csv,
)

BENCHMARK_AGENT_CHOICES = list_evaluable_agents()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?? frozen paper protocol ?? predictor robustness benchmark")
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
    parser.add_argument("--max_steps", type=int, default=14)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_mode", type=str, default="mixed_informative", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--window_count", type=int, default=3)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--predictor_kind", type=str, default="baseline", choices=["baseline", "supervised"])
    parser.add_argument("--predictor_checkpoint_path", type=str, default="")
    parser.add_argument("--include_noise_sweep", action="store_true")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "benchmarks" / "prediction_robustness"))
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


def build_settings(include_noise_sweep: bool, predictor_kind: str, predictor_checkpoint_path: str) -> list[dict[str, Any]]:
    settings = [
        {
            "setting_id": "baseline_prediction",
            "predictor_kwargs": {"predictor_kind": "baseline"},
            "setting_note": "baseline short-horizon predictor",
        },
        {
            "setting_id": "no_prediction",
            "predictor_kwargs": {"disable_prediction_output": True},
            "setting_note": "prediction output disabled diagnostic lower bound",
        },
        {
            "setting_id": "oracle_prediction",
            "predictor_kwargs": {"oracle_prediction_enabled": True},
            "setting_note": "oracle next-rsu / handoff-target diagnostic upper-bound setting",
        },
    ]
    if predictor_kind == "supervised":
        if not predictor_checkpoint_path:
            raise ValueError("--predictor_checkpoint_path is required for --predictor_kind supervised")
        supervised_kwargs = {
            "predictor_kind": "supervised",
            "predictor_checkpoint_path": predictor_checkpoint_path,
        }
        settings.insert(
            1,
            {
                "setting_id": "supervised_prediction",
                "predictor_kwargs": dict(supervised_kwargs),
                "setting_note": "frozen supervised short-horizon handoff predictor",
            },
        )
        settings.insert(
            2,
            {
                "setting_id": "noisy_supervised_prediction",
                "predictor_kwargs": {
                    **supervised_kwargs,
                    "prediction_noise_std": 0.2,
                    "prediction_confidence_scale": 0.7,
                    "prediction_delay_steps": 1,
                    "drop_handoff_prediction_prob": 0.2,
                },
                "setting_note": "frozen supervised predictor under delay/drop/noise stress",
            },
        )
    else:
        settings.insert(
            1,
            {
                "setting_id": "noisy_prediction",
                "predictor_kwargs": {
                    "predictor_kind": "baseline",
                    "prediction_noise_std": 0.2,
                    "prediction_confidence_scale": 0.7,
                    "prediction_delay_steps": 1,
                    "drop_handoff_prediction_prob": 0.2,
                },
                "setting_note": "baseline predictor under delay/drop/noise stress",
            },
        )
    if include_noise_sweep:
        for level in [0.0, 0.1, 0.2, 0.3]:
            sweep_kwargs: dict[str, Any] = {"prediction_noise_std": level}
            if predictor_kind == "supervised":
                sweep_kwargs.update(
                    {
                        "predictor_kind": "supervised",
                        "predictor_checkpoint_path": predictor_checkpoint_path,
                    }
                )
            settings.append({
                "setting_id": f"noise_sweep_{level:.1f}",
                "predictor_kwargs": sweep_kwargs,
                "setting_note": f"???? sweep?noise_std={level:.1f}",
            })
    return settings


def build_checkpoint_audit(checkpoint_map: dict[str, str], agents: list[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for agent_name in agents:
        checkpoint_path = checkpoint_map.get(agent_name, "")
        if checkpoint_path:
            audit[agent_name] = load_checkpoint_metadata(checkpoint_path)
        else:
            audit[agent_name] = {
                "checkpoint_path": "",
                "run_id": agent_name,
                "config_profile": "non_checkpoint_agent",
                "run_update_count": 0,
                "checkpoint_source_update_index": 0,
                "is_smoke_checkpoint": False,
                "requires_checkpoint": False,
            }
    return audit


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
    checkpoint_audit = build_checkpoint_audit(audit_checkpoint_source_map, args.agents)
    mobility_source_path, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector=args.window_selector,
        window_count=args.window_count,
        window_scan_stride=args.window_scan_stride,
        random_seed=args.seeds[0] if args.seeds else 7,
        window_mode=args.window_mode,
    )
    selected_windows = list(window_payload["selected_windows"])
    settings = build_settings(
        include_noise_sweep=args.include_noise_sweep,
        predictor_kind=args.predictor_kind,
        predictor_checkpoint_path=args.predictor_checkpoint_path,
    )
    run_id = datetime.now().strftime(f"prediction_robustness_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / run_id
    rows: list[dict[str, Any]] = []

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
        for window_candidate in selected_windows:
            mobility_bundle = load_window_bundle(
                root_dir=ROOT_DIR,
                mobility_source=args.mobility_source,
                mobility_csv_path=args.mobility_csv_path,
                lust_scenario_root=args.lust_scenario_root,
                max_mobility_rows=args.max_mobility_rows,
                rsu_layout=str(window_candidate.get("recommended_rsu_layout", args.rsu_layout)),
                frame_offset=int(window_candidate["frame_offset"]),
                window_length=int(window_candidate["window_length"]),
                random_seed=seed,
            )
            mobility_bundle.rsu_metadata["window_class"] = window_candidate["window_class"]
            mobility_bundle.rsu_metadata["window_rank"] = window_candidate["window_rank"]
            for workflow_state in workflow_states:
                for setting in settings:
                    compare_agents = list(args.agents)
                    if setting["setting_id"].startswith("noise_sweep_"):
                        compare_agents = [agent_name for agent_name in compare_agents if agent_name == "sa_ghmappo"]
                    for agent_name in compare_agents:
                        summary = run_real_episode(
                            root_dir=ROOT_DIR,
                            agent_name=agent_name,
                            checkpoint_map=seed_checkpoint_map,
                            workflow_state=workflow_state,
                            workflow_source_path=args.workflow_csv_path,
                            mobility_bundle=mobility_bundle,
                            seed=seed,
                            max_steps=args.max_steps,
                            mobility_source=args.mobility_source,
                            primary_vehicle_selection=args.primary_vehicle_selection,
                            run_metadata={
                            "script": "scripts/benchmark_prediction_robustness.py",
                            "robustness_setting_id": setting["setting_id"],
                            "window_mode": args.window_mode,
                            "mainline": mainline_label,
                            "protocol_version": PAPER_PROTOCOL_VERSION,
                            "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
                            },
                            predictor_kwargs=dict(setting["predictor_kwargs"]),
                        )
                        row = summary_to_row(summary)
                        row["prediction_setting_id"] = setting["setting_id"]
                        row["prediction_setting_note"] = setting["setting_note"]
                        rows.append(row)

    aggregate_by_setting_and_agent = aggregate_rows(rows, group_keys=["prediction_setting_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_setting = aggregate_rows(rows, group_keys=["prediction_setting_id"], metrics=MAIN_RESULT_METRICS)

    claim_summary = {}
    for agent_name in [agent for agent in args.agents if agent == "sa_ghmappo"]:
        supervised = aggregate_by_setting_and_agent.get(f"supervised_prediction|{agent_name}", {"metrics": {}})
        baseline = aggregate_by_setting_and_agent.get(f"baseline_prediction|{agent_name}", {"metrics": {}})
        oracle = aggregate_by_setting_and_agent.get(f"oracle_prediction|{agent_name}", {"metrics": {}})
        noisy = aggregate_by_setting_and_agent.get(f"noisy_supervised_prediction|{agent_name}", {"metrics": {}})
        if not noisy.get("metrics"):
            noisy = aggregate_by_setting_and_agent.get(f"noisy_prediction|{agent_name}", {"metrics": {}})
        none = aggregate_by_setting_and_agent.get(f"no_prediction|{agent_name}", {"metrics": {}})
        reference = supervised if supervised.get("metrics") else baseline
        if not reference.get("metrics"):
            continue
        claim_summary[agent_name] = {
            "reference_prediction_setting": "supervised_prediction" if supervised.get("metrics") else "baseline_prediction",
            "oracle_minus_reference_total_reward": round(float(oracle.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)) - float(reference.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)), 6),
            "reference_minus_noisy_total_reward": round(float(reference.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)) - float(noisy.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)), 6),
            "reference_minus_no_prediction_total_reward": round(float(reference.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)) - float(none.get("metrics", {}).get("total_reward", {}).get("mean", 0.0)), 6),
        }

    aggregate_summary = {
        "run_id": run_id,
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
        "canonical_paper_protocol": True,
        "config_profile": infer_benchmark_config_profile(checkpoint_audit, args.agents),
        "mainline": mainline_label,
        "mobility_source": args.mobility_source,
        "mobility_source_path": mobility_source_path,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "predictor_kind": args.predictor_kind,
        "predictor_checkpoint_path": args.predictor_checkpoint_path,
        "window_mode": args.window_mode,
        "selected_window_plan": selected_windows,
        "selected_window_plan_by_strata": window_payload.get("selected_window_plan_by_strata", {}),
        "seed_checkpoint_manifest_path": args.seed_checkpoint_manifest_path,
        "seed_checkpoint_manifest": seed_checkpoint_manifest,
        "checkpoint_audit": checkpoint_audit,
        "episode_count": len(rows),
        "aggregate_by_agent": aggregate_by_agent,
        "aggregate_by_setting": aggregate_by_setting,
        "aggregate_by_setting_and_agent": aggregate_by_setting_and_agent,
        "pairwise_comparison": {},
        "mechanism_diagnosis": build_mechanism_diagnosis(rows),
        "claim_summary": claim_summary,
        "rows": rows,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "prediction_robustness_summary.json"
    rows_path = output_root / "prediction_robustness_rows.csv"
    summary_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(rows_path, rows)
    print("prediction robustness ????")
    print(f"run_id: {run_id}")
    print(f"prediction_robustness_summary_path: {summary_path}")
    print(f"prediction_robustness_rows_path: {rows_path}")


if __name__ == "__main__":
    main()
