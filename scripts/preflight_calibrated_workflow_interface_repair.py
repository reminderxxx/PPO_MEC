"""Create-only preflight for the bounded calibrated-workflow repair run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import (
    _build_agent,
    _collect_training_episode,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["source_segment_id"] != second["source_segment_id"]:
        return False
    return not (
        int(first["time_index_end"]) < int(second["time_index_start"])
        or int(second["time_index_end"]) < int(first["time_index_start"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json",
    )
    parser.add_argument(
        "--manifest",
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json",
    )
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config_path = ROOT_DIR / args.config
    manifest_path = ROOT_DIR / args.manifest
    config, _ = _load_experiment_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    checks: dict[str, Any] = {
        "config_hash_match": manifest["config_sha256"] == _sha256(config_path),
        "resolved_config_hash_match": manifest["resolved_config_sha256"]
        == _canonical_sha256(config),
        "interface_profile": config.get("interface_profile"),
        "mobility_progression": config.get("mobility_progression"),
        "episodes_within_bound": int(config["training"]["episodes"]) <= 192,
        "seed_count": len(config["training"]["seeds"]),
        "no_sa_ablation": config["learning_interface"].get("sa_event_ablation") is None,
        "reward_changed": bool(config["learning_interface"].get("reward_changed")),
        "formal_access": bool(config["claim_boundary"].get("formal")),
        "holdout_access": bool(config["claim_boundary"].get("holdout")),
    }
    splits = {
        split: [row for row in manifest["instances"] if row["split"] == split]
        for split in ("train", "dev", "regression", "frozen_check")
    }
    overlap_counts: dict[str, int] = {}
    split_names = list(splits)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap_counts[f"{left}__{right}"] = sum(
                _intervals_overlap(a["source_interval"], b["source_interval"])
                for a in splits[left]
                for b in splits[right]
            )
    checks["raw_interval_overlap_counts"] = overlap_counts

    smoke: dict[str, Any] = {}
    for method in config["training"]["agents"]:
        agent = _build_agent(method, int(config["training"]["seeds"][0]), config)
        rollout, summary = _collect_training_episode(
            agent,
            config,
            splits["train"][0],
            int(config["training"]["budget"]["episode_max_steps"]),
        )
        update = agent.learn(rollout)
        smoke[method] = {
            "steps": len(rollout),
            "workflow_completed": int(summary["workflow_completed"]),
            "update_skipped": bool(update.get("policy_update_skipped")),
            "executed_action_ppo_only": update.get("executed_action_ppo_only"),
            "hierarchical_action_contract": update.get(
                "hierarchical_action_contract", "flat_action_contract"
            ),
        }
    checks["learning_smoke"] = smoke
    passed = bool(
        checks["config_hash_match"]
        and checks["resolved_config_hash_match"]
        and checks["interface_profile"] == "calibrated_workflow_interface_v2"
        and checks["mobility_progression"] == "decision_step_index"
        and checks["episodes_within_bound"]
        and checks["seed_count"] == 3
        and checks["no_sa_ablation"]
        and not checks["reward_changed"]
        and not checks["formal_access"]
        and not checks["holdout_access"]
        and all(value == 0 for value in overlap_counts.values())
        and all(not value["update_skipped"] for value in smoke.values())
    )
    _write_json(output_root / "preflight.json", {"status": "pass" if passed else "fail", **checks})
    _write_json(
        output_root / "completion_receipt.json",
        {"status": "complete", "preflight_status": "pass" if passed else "fail"},
    )
    files = [output_root / "preflight.json", output_root / "completion_receipt.json"]
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )
    if not passed:
        raise RuntimeError("interface-repair preflight failed")
    print(json.dumps({"status": "pass", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
