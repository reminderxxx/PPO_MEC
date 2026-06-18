"""Train current-contract SA-GHMAPPO ablation checkpoints and build a seed manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]


VARIANTS: dict[str, dict[str, Any]] = {
    "no_latency_fallback": {
        "disable_flags": ["--no-latency_fallback_bias_enabled"],
        "removed_module": "inference-calibrated latency fallback",
        "paper_contribution": "removes latency-fallback action calibration while retaining the v7 training contract",
    },
    "no_prediction": {
        "disable_flags": ["--disable_prediction"],
        "removed_module": "surrogate-assisted prediction",
        "paper_contribution": "removes surrogate prediction features from SA-GHMAPPO",
        "predictor_kwargs": {"disable_prediction_output": True},
    },
    "no_graph_encoder": {
        "disable_flags": ["--disable_graph_encoder"],
        "removed_module": "DAG-aware graph encoder",
        "paper_contribution": "replaces graph encoding with the flat semantic encoder",
    },
    "no_hierarchy": {
        "disable_flags": ["--disable_hierarchy"],
        "removed_module": "multi-timescale hierarchy",
        "paper_contribution": "removes slow/fast/event hierarchical conditioning",
    },
    "no_event_agent": {
        "disable_flags": ["--disable_event_agent"],
        "removed_module": "handoff-aware event controller",
        "paper_contribution": "removes event-head handoff migration preparation",
    },
    "no_adapter_prefetch": {
        "disable_flags": ["--disable_adapter_prefetch"],
        "removed_module": "proactive adapter prefetch",
        "paper_contribution": "removes proactive adapter prefetch behavior",
    },
    "no_dag_dependency_aware": {
        "disable_flags": ["--disable_dag_dependency_aware"],
        "removed_module": "dependency-aware DAG modeling",
        "paper_contribution": "removes dependency-aware workflow modeling signals",
    },
    "no_uncertainty_signal": {
        "disable_flags": ["--disable_uncertainty_signal"],
        "removed_module": "prediction confidence / uncertainty gating",
        "paper_contribution": "removes prediction uncertainty and confidence gating",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current-contract ablation training.")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "experiments" / "top_journal_support_suite"))
    parser.add_argument("--full_seed_manifest_path", type=str, required=True)
    parser.add_argument("--profile", type=str, default="top_journal_mechanism_v1")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--update_every", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--train_window_count", type=int, default=5)
    parser.add_argument("--max_mobility_rows", type=int, default=2500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_mode", type=str, default="mixed_informative")
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_latest_summary(variant_root: Path, seed: int) -> Path | None:
    candidates = list(variant_root.glob(f"sa_ghmappo/*_seed{seed}/train_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def selected_checkpoint_from_summary(summary_path: Path) -> str:
    payload = load_json(summary_path)
    for field_name in [
        "best_by_reward_tiebreak_score_path",
        "best_by_mechanism_advantage_score_path",
        "best_by_continuity_path",
        "best_by_reward_path",
        "latest_checkpoint_path",
    ]:
        checkpoint_path = str(payload.get(field_name, "") or "")
        if checkpoint_path and Path(checkpoint_path).exists():
            return checkpoint_path
    raise RuntimeError(f"no usable checkpoint found in {summary_path}")


def build_train_command(args: argparse.Namespace, variant: str, seed: int, variant_root: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_sa_ghmappo_real_sample.py",
        "--agent_name",
        "sa_ghmappo",
        "--profile",
        args.profile,
        "--episodes",
        str(args.episodes),
        "--update_every",
        str(args.update_every),
        "--batch_size",
        str(args.batch_size),
        "--max_steps",
        str(args.max_steps),
        "--train_window_count",
        str(args.train_window_count),
        "--random_seed",
        str(seed),
        "--window_mode",
        args.window_mode,
        "--output_root",
        str(variant_root),
        "--mobility_source",
        "ngsim",
        "--primary_vehicle_selection",
        "handoff_pressure",
        "--workflow_csv_path",
        args.workflow_csv_path,
        "--max_mobility_rows",
        str(args.max_mobility_rows),
        "--max_workflows",
        str(args.max_workflows),
        "--workflow_selector",
        args.workflow_selector,
        "--rsu_layout",
        args.rsu_layout,
        "--window_selector",
        args.window_selector,
        "--window_length",
        str(args.window_length),
        "--window_scan_stride",
        str(args.window_scan_stride),
        "--min_tasks",
        str(args.min_tasks),
        "--max_tasks",
        str(args.max_tasks),
    ]
    command.extend(VARIANTS[variant]["disable_flags"])
    return command


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("top_journal_ablation_training_%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    full_seed_manifest = load_json(Path(args.full_seed_manifest_path))
    manifest_path = run_root / "ablation_checkpoint_manifest.json"
    command_log_path = run_root / "command_log.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {}
    manifest["sa_ghmappo_full"] = {
        "agent_name": "sa_ghmappo",
        "checkpoint_by_seed": full_seed_manifest.get("sa_ghmappo", {}),
        "removed_module": "none",
        "paper_contribution": "full current-contract SA-GHMAPPO",
        "profile": args.profile,
    }
    command_log: list[dict[str, Any]] = []
    if command_log_path.exists():
        existing_log = load_json(command_log_path)
        if isinstance(existing_log, list):
            command_log.extend(existing_log)

    for variant in args.variants:
        variant_root = run_root / "training" / variant
        variant_manifest = {
            "agent_name": "sa_ghmappo",
            "checkpoint_by_seed": {},
            "removed_module": VARIANTS[variant]["removed_module"],
            "paper_contribution": VARIANTS[variant]["paper_contribution"],
            "predictor_kwargs": dict(VARIANTS[variant].get("predictor_kwargs", {})),
            "profile": args.profile,
        }
        for seed in args.seeds:
            existing_summary = find_latest_summary(variant_root, seed) if args.resume else None
            if existing_summary is not None:
                checkpoint_path = selected_checkpoint_from_summary(existing_summary)
                variant_manifest["checkpoint_by_seed"][str(seed)] = checkpoint_path
                command_log.append(
                    {
                        "label": f"train_{variant}_seed_{seed}",
                        "skipped": True,
                        "summary_path": str(existing_summary),
                        "selected_checkpoint_path": checkpoint_path,
                    }
                )
                continue
            command = build_train_command(args, variant, seed, variant_root)
            started = time.time()
            result = subprocess.run(command, cwd=ROOT_DIR, text=True, capture_output=True)
            elapsed_sec = round(time.time() - started, 3)
            summary_path = find_latest_summary(variant_root, seed)
            checkpoint_path = selected_checkpoint_from_summary(summary_path) if summary_path else ""
            command_log.append(
                {
                    "label": f"train_{variant}_seed_{seed}",
                    "command": command,
                    "returncode": result.returncode,
                    "elapsed_sec": elapsed_sec,
                    "summary_path": str(summary_path) if summary_path else "",
                    "selected_checkpoint_path": checkpoint_path,
                    "stdout_tail": result.stdout[-4000:],
                    "stderr_tail": result.stderr[-4000:],
                }
            )
            if result.returncode != 0:
                (run_root / "command_log.json").write_text(json.dumps(command_log, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(f"training failed for variant={variant} seed={seed}; see {run_root / 'command_log.json'}")
            variant_manifest["checkpoint_by_seed"][str(seed)] = checkpoint_path
        manifest[variant] = variant_manifest

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    command_log_path.write_text(json.dumps(command_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("top journal ablation training complete")
    print(f"run_root: {run_root}")
    print(f"ablation_manifest_path: {manifest_path}")
    print(f"command_log_path: {command_log_path}")


if __name__ == "__main__":
    main()
