"""Build an ablation manifest for the inference-calibrated v3 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 eval-bias latency-fallback ablation manifest.")
    parser.add_argument(
        "--calibrated_manifest_path",
        type=str,
        default=str(
            ROOT_DIR
            / "artifacts"
            / "experiments"
            / "top_journal_sa_iteration"
            / "top_journal_mechanism_v3_eval_bias"
            / "seed_checkpoint_manifest_v3_eval_bias_learned_baselines.json"
        ),
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
            / "top_journal_mechanism_v3_eval_bias_support"
            / "ablation_manifest"
        ),
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main() -> None:
    args = parse_args()
    calibrated_manifest = load_manifest(args.calibrated_manifest_path)
    base_manifest = load_manifest(args.base_manifest_path)
    calibrated_sa = dict(calibrated_manifest.get("sa_ghmappo", {}))
    base_sa = dict(base_manifest.get("sa_ghmappo", {}))
    if not calibrated_sa:
        raise RuntimeError(f"missing calibrated sa_ghmappo checkpoints: {args.calibrated_manifest_path}")
    if not base_sa:
        raise RuntimeError(f"missing base sa_ghmappo checkpoints: {args.base_manifest_path}")
    manifest = {
        "sa_ghmappo_full": {
            "agent_name": "sa_ghmappo",
            "checkpoint_by_seed": calibrated_sa,
            "removed_module": "none",
            "paper_contribution": "full method with inference-calibrated latency fallback",
        },
        "no_latency_fallback": {
            "agent_name": "sa_ghmappo",
            "checkpoint_by_seed": base_sa,
            "removed_module": "inference_calibrated_latency_fallback",
            "paper_contribution": "test-time low-risk latency fallback calibration",
        },
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "eval_bias_latency_fallback_ablation_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("eval-bias ablation manifest complete")
    print(f"manifest_path: {output_path}")


if __name__ == "__main__":
    main()
