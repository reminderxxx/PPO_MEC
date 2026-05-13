"""Select SA-GHMAPPO reward-tiebreak checkpoints from completed round4 runs."""

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
    build_reward_tiebreak_selection_reason,
    compute_reward_tiebreak_checkpoint_score,
    reward_tiebreak_score_priority_tuple,
)
from src.evaluators.main_results_support import load_checkpoint_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc reward-tiebreak checkpoint selection for completed SA-GHMAPPO round4 runs.",
    )
    parser.add_argument("--training_root", type=str, default="artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo")
    parser.add_argument("--profile", type=str, default="sa_reward_tiebreak_round4")
    parser.add_argument("--agent_name", type=str, default="sa_ghmappo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--checkpoint_label", type=str, default="best_by_reward_tiebreak_score")
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="artifacts/training/sa_reward_tiebreak_round4/seed_checkpoint_manifest_sa_reward_tiebreak_round4_best_by_reward_tiebreak_score.json",
    )
    parser.add_argument(
        "--selection_summary_path",
        type=str,
        default="artifacts/training/sa_reward_tiebreak_round4/reward_tiebreak_selection_summary.json",
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
    popularity_metrics = dict(item.get("aggregate_by_agent", {}).get("popularity_cache_heuristic", {}))
    rows = list(item.get("rows", []))
    priority_tuple = reward_tiebreak_score_priority_tuple(
        metrics,
        reference_metrics=popularity_metrics,
        rows=rows,
        current_agent_name=agent_name,
    )
    score_breakdown = compute_reward_tiebreak_checkpoint_score(
        metrics,
        reference_metrics=popularity_metrics,
        rows=rows,
        current_agent_name=agent_name,
    )
    return {
        "update_index": int(item.get("update_index", 0) or 0),
        "episode_index": int(item.get("episode_index", 0) or 0),
        "warm_start_eval": bool(item.get("warm_start_eval", False)),
        "priority_tuple": list(priority_tuple),
        "score_breakdown": score_breakdown,
        "metrics": metrics,
        "popularity_metrics_available": bool(popularity_metrics),
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


def selected_source_checkpoint(run_dir: Path, selected: dict[str, Any]) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    if bool(selected.get("warm_start_eval", False)):
        return checkpoint_root / "warm_start.pt"
    update_index = int(selected.get("update_index", 0) or 0)
    return checkpoint_root / f"update_{update_index:04d}.pt"


def copy_selected_checkpoint(
    *,
    run_dir: Path,
    selected: dict[str, Any],
    checkpoint_label: str,
    profile: str,
    agent_name: str,
    selection_record_path: Path,
) -> Path:
    source_checkpoint = selected_source_checkpoint(run_dir, selected)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Selected checkpoint source does not exist: {source_checkpoint}")
    target_checkpoint = run_dir / "checkpoints" / f"{checkpoint_label}.pt"
    shutil.copy2(source_checkpoint, target_checkpoint)
    metadata = dict(load_checkpoint_metadata(str(target_checkpoint)))
    metadata.update(
        {
            "agent_name": agent_name,
            "config_profile": profile,
            "checkpoint_selection_label": checkpoint_label,
            "selection_score_name": "reward_tiebreak_score_round4",
            "checkpoint_source_update_index": int(selected.get("update_index", 0) or 0),
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
        "selection_rule": "best_by_reward_tiebreak_score",
        "selection_score_version": "reward_tiebreak_score_round4",
        "seed_records": {},
    }
    for seed in args.seeds:
        run_dir = discover_seed_run(training_root, seed)
        history = read_update_history(run_dir)
        selected, candidates = select_candidate(history, args.agent_name)
        run_record_path = run_dir / "reward_tiebreak_selection_record.json"
        target_checkpoint = copy_selected_checkpoint(
            run_dir=run_dir,
            selected=selected,
            checkpoint_label=args.checkpoint_label,
            profile=args.profile,
            agent_name=args.agent_name,
            selection_record_path=run_record_path,
        )
        selection_reason = build_reward_tiebreak_selection_reason(
            candidate_metrics=dict(selected.get("metrics", {})),
            popularity_metrics={},
            rows=[],
            selected=True,
            current_agent_name=args.agent_name,
        )
        selection_reason["score_breakdown"] = dict(selected.get("score_breakdown", {}))
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
            "warm_start_eval": selected.get("warm_start_eval", False),
            "score": selected.get("score_breakdown", {}).get("score"),
            "mixed_reward_gap_or_fallback": selected.get("score_breakdown", {}).get("mixed_reward_gap"),
            "safety_penalty": selected.get("score_breakdown", {}).get("safety_penalty"),
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
