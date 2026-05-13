"""SA-GHMAPPO ???? benchmark?"""

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
    ABLATION_CONTRIBUTION_MAP,
    MAIN_RESULT_METRICS,
    PAPER_PROTOCOL_FROZEN,
    PAPER_PROTOCOL_VERSION,
    aggregate_rows,
    build_mechanism_diagnosis,
    build_pairwise_comparison,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?? frozen paper protocol ?? SA-GHMAPPO ?? benchmark")
    parser.add_argument("--ablation_labels", nargs="+", default=["sa_ghmappo_full", "w/o_dt_predictor", "w/o_adapter_prefetch_or_cache", "w/o_handoff_migration", "flat_or_no_hierarchy"])
    parser.add_argument("--manifest_path", type=str, default=str(ROOT_DIR / "configs" / "ablation_checkpoint_manifest_paper.json"))
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
    parser.add_argument("--window_rank_offset", type=int, default=0)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "benchmarks" / "ablation"))
    return parser.parse_args()


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"ablation manifest ???: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_path_for_label_seed(payload: dict[str, Any], seed: int) -> str:
    checkpoint_by_seed = payload.get("checkpoint_by_seed", {})
    if isinstance(checkpoint_by_seed, dict):
        checkpoint_path = str(checkpoint_by_seed.get(str(seed), "") or checkpoint_by_seed.get(seed, "") or "")
        if checkpoint_path:
            return checkpoint_path
    return str(payload.get("checkpoint_path", "") or "")


def build_checkpoint_audit_by_label(selected_manifest: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for label, payload in selected_manifest.items():
        checkpoint_path = ""
        for seed in seeds:
            checkpoint_path = _checkpoint_path_for_label_seed(payload, seed)
            if checkpoint_path:
                break
        if not checkpoint_path:
            audit[label] = {
                "exists": False,
                "checkpoint_path": "",
                "run_id": payload.get("run_id", "none"),
                "config_profile": payload.get("config_profile", "none"),
                "run_update_count": 0,
                "checkpoint_source_update_index": 0,
                "requires_checkpoint": False,
                "is_smoke_checkpoint": False,
            }
            continue
        metadata = load_checkpoint_metadata(checkpoint_path)
        audit[label] = metadata
    return audit


def main() -> None:
    args = parse_args()
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    manifest = load_manifest(args.manifest_path)
    seed_checkpoint_manifest = load_seed_checkpoint_manifest(args.seed_checkpoint_manifest_path)
    selected_manifest = {label: manifest[label] for label in args.ablation_labels if label in manifest}
    if not selected_manifest:
        raise RuntimeError("ablation_labels ??? manifest ??????")

    source_path, window_payload = resolve_window_candidates(
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
        window_rank_offset=args.window_rank_offset,
    )
    selected_windows = list(window_payload["selected_windows"])
    if not selected_windows:
        raise RuntimeError("frozen protocol ????????")

    run_id = datetime.now().strftime(f"ablation_{args.window_mode}_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / run_id
    rows: list[dict[str, Any]] = []
    selected_workflow_ids_by_seed: dict[str, list[str]] = {}

    for seed in args.seeds:
        workflow_states = build_selected_workflow_states(
            workflow_csv_path=args.workflow_csv_path,
            max_workflows=args.max_workflows,
            workflow_selector=args.workflow_selector,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            random_seed=seed,
        )
        selected_workflow_ids_by_seed[str(seed)] = [workflow_state.workflow_id for workflow_state in workflow_states]
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
            mobility_bundle.rsu_metadata["window_rank"] = window_candidate["window_rank"]
            mobility_bundle.rsu_metadata["window_class"] = window_candidate["window_class"]
            for workflow_state in workflow_states:
                for label, payload in selected_manifest.items():
                    base_checkpoint_map: dict[str, str] = {}
                    checkpoint_path = _checkpoint_path_for_label_seed(payload, seed)
                    if checkpoint_path:
                        base_checkpoint_map[str(payload["agent_name"])] = checkpoint_path
                        checkpoint_map = expand_checkpoint_aliases(base_checkpoint_map)
                    else:
                        checkpoint_map = checkpoint_map_for_seed(
                            base_checkpoint_map=expand_checkpoint_aliases(base_checkpoint_map),
                            seed_checkpoint_manifest=seed_checkpoint_manifest,
                            seed=seed,
                        )
                    summary = run_real_episode(
                        root_dir=ROOT_DIR,
                        agent_name=str(payload["agent_name"]),
                        checkpoint_map=checkpoint_map,
                        workflow_state=workflow_state,
                        workflow_source_path=args.workflow_csv_path,
                        mobility_bundle=mobility_bundle,
                        seed=seed,
                        max_steps=args.max_steps,
                        mobility_source=args.mobility_source,
                        primary_vehicle_selection=args.primary_vehicle_selection,
                        predictor_kwargs=dict(payload.get("predictor_kwargs", {}) or {}),
                        run_metadata={
                            "script": "scripts/benchmark_ablation.py",
                            "ablation_label": label,
                            "mainline": mainline_label,
                            "window_mode": args.window_mode,
                            "window_rank_offset": args.window_rank_offset,
                            "window_class": window_candidate.get("window_class", "unknown"),
                            "protocol_version": PAPER_PROTOCOL_VERSION,
                            "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
                        },
                    )
                    row = summary_to_row(summary)
                    row["agent_name"] = label
                    row["removed_module"] = payload.get("removed_module", ABLATION_CONTRIBUTION_MAP.get(label, {}).get("removed_module", "unknown"))
                    row["paper_contribution"] = payload.get("paper_contribution", ABLATION_CONTRIBUTION_MAP.get(label, {}).get("paper_contribution", "unknown"))
                    rows.append(row)

    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_window_and_agent = aggregate_rows(rows, group_keys=["window_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    checkpoint_audit = build_checkpoint_audit_by_label(selected_manifest, args.seeds)
    audit_checkpoint_source_map = representative_checkpoint_map(
        base_checkpoint_map={},
        seed_checkpoint_manifest=seed_checkpoint_manifest,
        seeds=args.seeds,
    )
    aggregate_summary = {
        "run_id": run_id,
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "paper_protocol_frozen": PAPER_PROTOCOL_FROZEN,
        "canonical_paper_protocol": True,
        "protocol_note": "?????? frozen paper protocol strata ?????????? ablation ???",
        "config_profile": infer_benchmark_config_profile(checkpoint_audit, list(selected_manifest.keys())),
        "mainline": mainline_label,
        "mobility_source": args.mobility_source,
        "mobility_source_path": source_path,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "window_mode": args.window_mode,
        "window_rank_offset": args.window_rank_offset,
        "selected_window_plan": selected_windows,
        "selected_window_plan_by_strata": window_payload.get("selected_window_plan_by_strata", {}),
        "selected_workflow_ids_by_seed": selected_workflow_ids_by_seed,
        "seed_checkpoint_manifest_path": args.seed_checkpoint_manifest_path,
        "seed_checkpoint_manifest": seed_checkpoint_manifest,
        "checkpoint_map": audit_checkpoint_source_map,
        "checkpoint_audit": checkpoint_audit,
        "ablation_manifest": selected_manifest,
        "ablation_explanations": {
            label: {
                "removed_module": payload.get("removed_module", ABLATION_CONTRIBUTION_MAP.get(label, {}).get("removed_module", "unknown")),
                "paper_contribution": payload.get("paper_contribution", ABLATION_CONTRIBUTION_MAP.get(label, {}).get("paper_contribution", "unknown")),
            }
            for label, payload in selected_manifest.items()
        },
        "episode_count": len(rows),
        "aggregate_by_agent": aggregate_by_agent,
        "aggregate_by_window_and_agent": aggregate_by_window_and_agent,
        "pairwise_comparison": {
            "vs_sa_ghmappo_full": build_pairwise_comparison(aggregate_by_agent, baseline_agent="sa_ghmappo_full", metrics=MAIN_RESULT_METRICS),
        },
        "mechanism_diagnosis": build_mechanism_diagnosis(rows),
        "rows": rows,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_root / "aggregate_summary.json"
    rows_path = output_root / "benchmark_rows.csv"
    aggregate_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(rows_path, rows)
    (output_root / "ablation_summary.json").write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(output_root / "ablation_rows.csv", rows)

    print("ablation benchmark ????")
    print(f"run_id: {run_id}")
    print(f"ablation_summary_path: {output_root / 'ablation_summary.json'}")
    print(f"ablation_rows_path: {output_root / 'ablation_rows.csv'}")


if __name__ == "__main__":
    main()
