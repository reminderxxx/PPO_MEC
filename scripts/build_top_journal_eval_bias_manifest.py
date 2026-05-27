from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manifest with SA-GHMAPPO inference-calibration fields enabled."
    )
    parser.add_argument(
        "--base_manifest_path",
        type=str,
        default=str(
            ROOT_DIR
            / "artifacts"
            / "experiments"
            / "top_journal_learned_baseline_suite"
            / "top_journal_learned_baseline_formal_20260505_v1"
            / "seed_checkpoint_manifest_learned_baselines.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(
            ROOT_DIR
            / "artifacts"
            / "experiments"
            / "top_journal_sa_iteration"
            / "top_journal_mechanism_v3_eval_bias"
        ),
    )
    parser.add_argument("--label", type=str, default="v3_eval_bias")
    parser.add_argument("--latency_fallback_bias_strength", type=float, default=1.20)
    parser.add_argument("--latency_fallback_confidence_floor", type=float, default=0.62)
    parser.add_argument("--latency_fallback_slow_suppression_strength", type=float, default=1.20)
    parser.add_argument("--predictive_prepare_hard_override_enabled", action="store_true")
    parser.add_argument("--predictive_prepare_hard_override_score_threshold", type=float, default=0.55)
    parser.add_argument("--predictive_prepare_hard_override_confidence_threshold", type=float, default=0.70)
    parser.add_argument("--cache_warm_start_guard_min_countdown", type=float, default=None)
    parser.add_argument("--cache_warm_start_guard_max_prefetch_countdown", type=float, default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    args = parse_args()
    base_manifest_path = Path(args.base_manifest_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "formal_v2_weight_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(base_manifest_path)
    sa_entries = dict(manifest.get("sa_ghmappo", {}))
    if not sa_entries:
        raise RuntimeError(f"No sa_ghmappo entries found in {base_manifest_path}")

    for seed, checkpoint_path in sa_entries.items():
        source_path = Path(str(checkpoint_path))
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        config = dict(payload.get("config", {}))
        config["latency_fallback_bias_enabled"] = True
        config["latency_fallback_bias_strength"] = float(args.latency_fallback_bias_strength)
        config["latency_fallback_confidence_floor"] = float(args.latency_fallback_confidence_floor)
        config["latency_fallback_slow_suppression_strength"] = float(
            args.latency_fallback_slow_suppression_strength
        )
        config["predictive_prepare_hard_override_enabled"] = bool(
            args.predictive_prepare_hard_override_enabled
        )
        config["predictive_prepare_hard_override_score_threshold"] = float(
            args.predictive_prepare_hard_override_score_threshold
        )
        config["predictive_prepare_hard_override_confidence_threshold"] = float(
            args.predictive_prepare_hard_override_confidence_threshold
        )
        if args.cache_warm_start_guard_min_countdown is not None:
            config["cache_warm_start_guard_min_countdown"] = float(
                args.cache_warm_start_guard_min_countdown
            )
        if args.cache_warm_start_guard_max_prefetch_countdown is not None:
            config["cache_warm_start_guard_max_prefetch_countdown"] = float(
                args.cache_warm_start_guard_max_prefetch_countdown
            )
        payload["config"] = config
        payload["derived_checkpoint"] = {
            "source_checkpoint_path": str(source_path),
            "derivation": args.label,
            "latency_fallback_bias_strength": float(args.latency_fallback_bias_strength),
            "latency_fallback_confidence_floor": float(args.latency_fallback_confidence_floor),
            "latency_fallback_slow_suppression_strength": float(
                args.latency_fallback_slow_suppression_strength
            ),
            "predictive_prepare_hard_override_enabled": bool(
                args.predictive_prepare_hard_override_enabled
            ),
            "predictive_prepare_hard_override_score_threshold": float(
                args.predictive_prepare_hard_override_score_threshold
            ),
            "predictive_prepare_hard_override_confidence_threshold": float(
                args.predictive_prepare_hard_override_confidence_threshold
            ),
            "cache_warm_start_guard_min_countdown": (
                None
                if args.cache_warm_start_guard_min_countdown is None
                else float(args.cache_warm_start_guard_min_countdown)
            ),
            "cache_warm_start_guard_max_prefetch_countdown": (
                None
                if args.cache_warm_start_guard_max_prefetch_countdown is None
                else float(args.cache_warm_start_guard_max_prefetch_countdown)
            ),
        }
        target_path = checkpoint_dir / f"sa_ghmappo_seed{seed}_{args.label}.pt"
        torch.save(payload, target_path)
        manifest["sa_ghmappo"][str(seed)] = str(target_path.resolve())

    manifest_path = output_root / f"seed_checkpoint_manifest_{args.label}_learned_baselines.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("top-journal eval-bias manifest complete")
    print(f"manifest_path: {manifest_path}")
    print(f"checkpoint_dir: {checkpoint_dir}")


if __name__ == "__main__":
    main()
