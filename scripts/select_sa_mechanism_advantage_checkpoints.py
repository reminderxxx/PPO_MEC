"""Select SA-GHMAPPO mechanism-advantage checkpoints from completed training runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_sa_ghmappo_real_sample import (  # noqa: E402
    annotate_checkpoint_metadata,
    build_mechanism_advantage_selection_reason,
    compute_mechanism_advantage_checkpoint_score,
    mechanism_advantage_score_priority_tuple,
)
from src.evaluators.main_results_support import load_checkpoint_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc mechanism-aware checkpoint selection for completed SA-GHMAPPO runs.",
    )
    parser.add_argument("--training_root", type=str, default="artifacts/training/sa_advantage_round1/sa_ghmappo")
    parser.add_argument("--profile", type=str, default="sa_advantage_round1")
    parser.add_argument("--agent_name", type=str, default="sa_ghmappo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--checkpoint_label", type=str, default="best_by_mechanism_advantage_score")
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_mechanism_advantage_score.json",
    )
    parser.add_argument(
        "--selection_summary_path",
        type=str,
        default="artifacts/training/sa_advantage_round1/mechanism_advantage_selection_summary.json",
    )
    return parser.parse_args()


def discover_seed_run(training_root: Path, seed: int) -> Path:
    candidates = sorted(training_root.glob(f"*seed{seed}"))
    if not candidates:
        raise FileNotFoundError(f"No training run found for seed {seed} under {training_root}")
    return candidates[-1]


def read_update_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "update_eval_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing update_eval_history.json: {history_path}")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"update_eval_history must be a list: {history_path}")
    return payload


def summarize_candidate(item: dict[str, Any], agent_name: str) -> dict[str, Any]:
    metrics = dict(item.get("aggregate_by_agent", {}).get(agent_name, {}))
    rows = list(item.get("rows", []))
    priority_tuple = mechanism_advantage_score_priority_tuple(
        metrics,
        rows=rows,
        current_agent_name=agent_name,
    )
    score_breakdown = compute_mechanism_advantage_checkpoint_score(
        metrics,
        rows=rows,
        current_agent_name=agent_name,
    )
    return {
        "update_index": int(item.get("update_index", 0) or 0),
        "episode_index": int(item.get("episode_index", 0) or 0),
        "priority_tuple": list(priority_tuple),
        "score_breakdown": score_breakdown,
        "metrics": metrics,
        "selection_protocol": {
            "protocol_name": item.get("protocol_name", "update_eval"),
            "deterministic_eval": bool(item.get("deterministic_eval", True)),
            "eval_window_ids": list(item.get("eval_window_ids", [])),
            "workflow_ids": list(item.get("workflow_ids", [])),
        },
    }


def select_candidate(history: list[dict[str, Any]], agent_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [summarize_candidate(item, agent_name) for item in history]
    if not candidates:
        raise ValueError("No candidate checkpoints found in update_eval_history")
    selected = max(
        candidates,
        key=lambda item: (
            tuple(item["priority_tuple"]),
            int(item.get("update_index", 0) or 0),
        ),
    )
    return selected, candidates


def copy_selected_checkpoint(
    *,
    run_dir: Path,
    selected: dict[str, Any],
    checkpoint_label: str,
    profile: str,
    agent_name: str,
    selection_record_path: Path,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    update_index = int(selected.get("update_index", 0) or 0)
    source_checkpoint = checkpoint_root / f"update_{update_index:04d}.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Selected checkpoint source does not exist: {source_checkpoint}")
    target_checkpoint = checkpoint_root / f"{checkpoint_label}.pt"
    shutil.copy2(source_checkpoint, target_checkpoint)
    metadata = dict(load_checkpoint_metadata(str(target_checkpoint)))
    metadata.update(
        {
            "agent_name": agent_name,
            "config_profile": profile,
            "checkpoint_selection_label": checkpoint_label,
            "selection_score_name": "mechanism_advantage_score_v2",
            "checkpoint_source_update_index": update_index,
            "selection_record_path": str(selection_record_path),
        }
    )
    annotate_checkpoint_metadata(target_checkpoint, metadata)
    return target_checkpoint


def main() -> None:
    args = parse_args()
    training_root = Path(args.training_root)
    output_manifest = Path(args.output_manifest)
    selection_summary_path = Path(args.selection_summary_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    selection_summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {args.agent_name: {}}
    summary: dict[str, Any] = {
        "profile": args.profile,
        "agent_name": args.agent_name,
        "checkpoint_label": args.checkpoint_label,
        "selection_rule": "best_by_mechanism_advantage_score",
        "selection_score_version": "mechanism_advantage_score_v2",
        "seed_records": {},
    }
    for seed in args.seeds:
        run_dir = discover_seed_run(training_root, seed)
        history = read_update_history(run_dir)
        selected, candidates = select_candidate(history, args.agent_name)
        run_record_path = run_dir / "mechanism_advantage_selection_record.json"
        target_checkpoint = copy_selected_checkpoint(
            run_dir=run_dir,
            selected=selected,
            checkpoint_label=args.checkpoint_label,
            profile=args.profile,
            agent_name=args.agent_name,
            selection_record_path=run_record_path,
        )
        selection_reason = build_mechanism_advantage_selection_reason(
            candidate_metrics=dict(selected.get("metrics", {})),
            popularity_metrics={},
            rows=[],
            selected=True,
            current_agent_name=args.agent_name,
        )
        selection_reason["score_breakdown"] = dict(selected.get("score_breakdown", {}))
        selection_reason["graceful_fallback"] = {
            "reference_metrics_available": False,
            "missing_metrics": selected.get("score_breakdown", {}).get("missing_metrics", []),
            "reference_missing_metrics": selected.get("score_breakdown", {}).get("reference_missing_metrics", []),
        }
        run_record = {
            "seed": seed,
            "run_dir": str(run_dir),
            "target_checkpoint_path": str(target_checkpoint),
            "selected": selected,
            "selection_reason": selection_reason,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        run_record_path.write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest[args.agent_name][str(seed)] = str(target_checkpoint)
        summary["seed_records"][str(seed)] = {
            "run_dir": str(run_dir),
            "target_checkpoint_path": str(target_checkpoint),
            "source_update_index": selected.get("update_index"),
            "source_episode_index": selected.get("episode_index"),
            "score": selected.get("score_breakdown", {}).get("score"),
            "stability_summary": selected.get("score_breakdown", {}).get("stability_summary", {}),
            "record_path": str(run_record_path),
        }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary["manifest_path"] = str(output_manifest)
    selection_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest_path: {output_manifest}")
    print(f"selection_summary_path: {selection_summary_path}")
    for seed, record in summary["seed_records"].items():
        print(
            f"seed={seed} checkpoint={record['target_checkpoint_path']} "
            f"update={record['source_update_index']} score={record['score']}"
        )


if __name__ == "__main__":
    main()
