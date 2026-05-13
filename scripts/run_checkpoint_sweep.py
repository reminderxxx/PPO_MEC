from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.composite_checkpoint_selection import select_best_composite_candidate
from src.evaluators.main_results_support import (
    MAIN_RESULT_METRICS,
    aggregate_rows,
    build_selected_workflow_states,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SA-GHMAPPO checkpoints with a composite mechanism-oriented score.")
    parser.add_argument("--training_run_dir", type=str, required=True)
    parser.add_argument("--mobility_source", type=str, default="lust", choices=["ngsim", "lust"])
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument(
        "--workflow_csv_path",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max_mobility_rows", type=int, default=80000)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="lust_micro")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_mode", type=str, default="activating_only")
    parser.add_argument("--window_count", type=int, default=3)
    parser.add_argument("--window_scan_stride", type=int, default=4)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_mechanism_realization_rate", type=float, default=0.3)
    parser.add_argument("--min_handoff_ready_ratio", type=float, default=1e-9)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "artifacts" / "eval" / "checkpoint_sweep"),
    )
    return parser.parse_args()


def resolve_checkpoint_paths(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    candidates = [(path.stem, path) for path in sorted(checkpoint_dir.glob("update_*.pt"))]
    if not candidates:
        raise RuntimeError(f"No update checkpoints found in {checkpoint_dir}.")
    return candidates


def write_candidate_rows_csv(output_path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "checkpoint_label",
        "checkpoint_path",
        "update_index",
        "selection_metrics_source",
        "overall_total_reward",
        "overall_workflow_continuity_rate",
        "overall_handoff_ready_ratio",
        "overall_mechanism_realization_rate",
        "selection_total_reward",
        "selection_workflow_continuity_rate",
        "selection_handoff_ready_ratio",
        "selection_mechanism_realization_rate",
        "normalized_reward",
        "composite_score",
        "eligible",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            overall = dict(candidate.get("overall_metrics", {}))
            selection = dict(candidate.get("selection_metrics", {}))
            composite = dict(candidate.get("composite_score", {}))
            writer.writerow(
                {
                    "checkpoint_label": candidate.get("checkpoint_label", ""),
                    "checkpoint_path": candidate.get("checkpoint_path", ""),
                    "update_index": candidate.get("update_index", 0),
                    "selection_metrics_source": candidate.get("selection_metrics_source", ""),
                    "overall_total_reward": overall.get("total_reward", 0.0),
                    "overall_workflow_continuity_rate": overall.get("workflow_continuity_rate", 0.0),
                    "overall_handoff_ready_ratio": overall.get("handoff_ready_ratio", 0.0),
                    "overall_mechanism_realization_rate": overall.get("mechanism_realization_rate", 0.0),
                    "selection_total_reward": selection.get("total_reward", 0.0),
                    "selection_workflow_continuity_rate": selection.get("workflow_continuity_rate", 0.0),
                    "selection_handoff_ready_ratio": selection.get("handoff_ready_ratio", 0.0),
                    "selection_mechanism_realization_rate": selection.get("mechanism_realization_rate", 0.0),
                    "normalized_reward": composite.get("normalized_reward", 0.0),
                    "composite_score": composite.get("score", 0.0),
                    "eligible": candidate.get("eligible_for_mechanism_target", False),
                }
            )


def extract_mean_metrics(aggregate_entry: dict[str, Any]) -> dict[str, float]:
    return {
        metric_name: float(metric_payload["mean"])
        for metric_name, metric_payload in dict(aggregate_entry.get("metrics", {})).items()
    }


def evaluate_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_label: str,
    checkpoint_path: Path,
    selected_windows: list[dict[str, Any]],
) -> dict[str, Any]:
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
                summary = run_real_episode(
                    root_dir=ROOT_DIR,
                    agent_name="sa_ghmappo",
                    checkpoint_map={"sa_ghmappo": str(checkpoint_path)},
                    workflow_state=workflow_state,
                    workflow_source_path=args.workflow_csv_path,
                    mobility_bundle=mobility_bundle,
                    seed=seed,
                    max_steps=args.max_steps,
                    mobility_source=args.mobility_source,
                    run_metadata={
                        "script": "scripts/run_checkpoint_sweep.py",
                        "checkpoint_label": checkpoint_label,
                        "window_mode": args.window_mode,
                        "mainline": "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba",
                    },
                )
                row = summary_to_row(summary)
                row["checkpoint_label"] = checkpoint_label
                row["checkpoint_path"] = str(checkpoint_path)
                rows.append(row)

    overall_aggregate = aggregate_rows(rows, group_keys=["checkpoint_label"], metrics=MAIN_RESULT_METRICS)
    overall_metrics = extract_mean_metrics(overall_aggregate[checkpoint_label])
    mechanism_rows = [row for row in rows if str(row.get("window_class", "")) == "mechanism_activating"]
    mechanism_metrics = dict(overall_metrics)
    selection_metrics_source = "overall"
    if mechanism_rows:
        mechanism_aggregate = aggregate_rows(
            mechanism_rows,
            group_keys=["checkpoint_label"],
            metrics=MAIN_RESULT_METRICS,
        )
        mechanism_metrics = extract_mean_metrics(mechanism_aggregate[checkpoint_label])
        selection_metrics_source = "mechanism_activating"

    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "update_index": int(checkpoint_label.split("_")[-1]) if checkpoint_label.startswith("update_") else 0,
        "episode_count": len(rows),
        "selected_workflow_ids_by_seed": selected_workflow_ids_by_seed,
        "overall_metrics": overall_metrics,
        "mechanism_metrics": mechanism_metrics,
        "selection_metrics_source": selection_metrics_source,
        "selection_metrics": mechanism_metrics if selection_metrics_source == "mechanism_activating" else overall_metrics,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    training_run_dir = Path(args.training_run_dir)
    checkpoint_dir = training_run_dir / "checkpoints"
    checkpoint_paths = resolve_checkpoint_paths(checkpoint_dir)

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
    if not selected_windows:
        raise RuntimeError("No selected windows available for checkpoint sweep.")

    run_id = datetime.now().strftime("checkpoint_sweep_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for checkpoint_label, checkpoint_path in checkpoint_paths:
        candidate = evaluate_checkpoint(
            args=args,
            checkpoint_label=checkpoint_label,
            checkpoint_path=checkpoint_path,
            selected_windows=selected_windows,
        )
        candidates.append(candidate)
        selection_metrics = dict(candidate.get("selection_metrics", {}))
        print(
            f"checkpoint={checkpoint_label} "
            f"reward={selection_metrics.get('total_reward', 0.0):.3f} "
            f"continuity={selection_metrics.get('workflow_continuity_rate', 0.0):.3f} "
            f"ready={selection_metrics.get('handoff_ready_ratio', 0.0):.3f} "
            f"mechanism={selection_metrics.get('mechanism_realization_rate', 0.0):.3f}"
        )

    selection_payload = select_best_composite_candidate(
        candidates,
        metrics_key="selection_metrics",
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        min_mechanism_realization_rate=args.min_mechanism_realization_rate,
        min_handoff_ready_ratio=args.min_handoff_ready_ratio,
    )
    annotated_candidates = list(selection_payload["annotated_candidates"])
    eligible_paths = {
        str(candidate.get("checkpoint_path", ""))
        for candidate in selection_payload.get("eligible_candidates", [])
    }
    for candidate in annotated_candidates:
        candidate["eligible_for_mechanism_target"] = str(candidate.get("checkpoint_path", "")) in eligible_paths

    best_candidate = dict(selection_payload["best_candidate"])
    best_source_path = Path(str(best_candidate["checkpoint_path"]))
    best_target_path = checkpoint_dir / "best_mechanism_checkpoint.pt"
    shutil.copy2(best_source_path, best_target_path)

    best_record = {
        "run_id": run_id,
        "training_run_dir": str(training_run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "best_mechanism_checkpoint_path": str(best_target_path),
        "source_checkpoint_path": str(best_source_path),
        "checkpoint_label": best_candidate.get("checkpoint_label", ""),
        "update_index": int(best_candidate.get("update_index", 0) or 0),
        "selection_metrics_source": best_candidate.get("selection_metrics_source", "overall"),
        "overall_metrics": dict(best_candidate.get("overall_metrics", {})),
        "mechanism_metrics": dict(best_candidate.get("mechanism_metrics", {})),
        "selection_metrics": dict(best_candidate.get("selection_metrics", {})),
        "composite_score": dict(best_candidate.get("composite_score", {})),
        "selected_from_eligible_pool": bool(selection_payload.get("selected_from_eligible_pool", False)),
        "selection_formula": selection_payload.get("selection_formula", ""),
        "selection_weights": dict(selection_payload.get("selection_weights", {})),
        "eligibility_thresholds": dict(selection_payload.get("eligibility_thresholds", {})),
        "window_mode": args.window_mode,
        "selected_windows": selected_windows,
        "mobility_source_path": mobility_source_path,
    }

    summary_payload = {
        "run_id": run_id,
        "training_run_dir": str(training_run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "mobility_source": args.mobility_source,
        "mobility_source_path": mobility_source_path,
        "rsu_layout": args.rsu_layout,
        "window_mode": args.window_mode,
        "selected_windows": selected_windows,
        "selection_payload": {
            "selection_formula": selection_payload.get("selection_formula", ""),
            "selection_weights": dict(selection_payload.get("selection_weights", {})),
            "eligibility_thresholds": dict(selection_payload.get("eligibility_thresholds", {})),
            "selected_from_eligible_pool": bool(selection_payload.get("selected_from_eligible_pool", False)),
            "eligible_candidate_count": len(selection_payload.get("eligible_candidates", [])),
        },
        "candidates": annotated_candidates,
        "best_candidate": best_record,
    }

    summary_path = output_root / "checkpoint_sweep_summary.json"
    rows_csv_path = output_root / "checkpoint_sweep_rows.csv"
    best_record_path = training_run_dir / "best_mechanism_checkpoint_record.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    best_record_path.write_text(json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidate_rows_csv(rows_csv_path, annotated_candidates)

    print("checkpoint sweep complete")
    print(f"summary_path: {summary_path}")
    print(f"best_mechanism_checkpoint_path: {best_target_path}")
    print(f"best_mechanism_record_path: {best_record_path}")
    print(
        "best_selection "
        f"checkpoint={best_candidate.get('checkpoint_label', '')} "
        f"score={best_candidate.get('composite_score', {}).get('score', 0.0):.3f} "
        f"reward={best_candidate.get('selection_metrics', {}).get('total_reward', 0.0):.3f} "
        f"continuity={best_candidate.get('selection_metrics', {}).get('workflow_continuity_rate', 0.0):.3f} "
        f"ready={best_candidate.get('selection_metrics', {}).get('handoff_ready_ratio', 0.0):.3f} "
        f"mechanism={best_candidate.get('selection_metrics', {}).get('mechanism_realization_rate', 0.0):.3f}"
    )


if __name__ == "__main__":
    main()
