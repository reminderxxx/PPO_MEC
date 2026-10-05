"""Read-only replay that reconciles raw actor heads with preserved executed actions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_interface_repair import (
    _rebuild_integrity,
    _verify_integrity,
    _write_json,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    integrity_before = _verify_integrity(run_root)
    if integrity_before["failure_count"]:
        raise RuntimeError(f"input integrity failed: {integrity_before}")
    manifest_info = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    config_path = ROOT_DIR / manifest_info["config"]["path"]
    workload_path = ROOT_DIR / manifest_info["workload_manifest"]["path"]
    config, _ = _load_experiment_config(config_path)
    manifest = json.loads(workload_path.read_text(encoding="utf-8"))
    summaries = json.loads((run_root / "training_summary.json").read_text(encoding="utf-8"))
    expected = _read_csv(run_root / "action_ledger.csv")
    expected_by_key = {
        (
            row["split"],
            row["method"],
            row["seed"],
            row["design_id"],
            int(row["step_index"]),
        ): row
        for row in expected
        if row["seed"] != "rule"
    }
    step_cap = int(config["training"]["budget"]["episode_max_steps"])
    output_rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for summary in summaries:
        method = str(summary["method"])
        seed = int(summary["seed"])
        agent = _build_agent(method, seed, config)
        agent.load(str(run_root / summary["selected_checkpoint"]))
        agent._deterministic_action = True
        for split in ("regression", "frozen_check"):
            instances = [row for row in manifest["instances"] if row["split"] == split]
            for instance in instances:
                env = CalibratedContinuousWorkflowEnv(config, instance)
                observation, info = env.reset()
                while not env.terminated and env.step_index < step_cap:
                    semantic = agent._extract_semantic_state(info)
                    run_metadata = dict(info.get("run_metadata", {}) or {})
                    with torch.no_grad():
                        policy_output = agent._forward_policy(
                            semantic, run_metadata=run_metadata
                        )
                    action, action_info = agent.act(observation, info)
                    step_index = env.step_index
                    key = (split, method, str(seed), instance["design_id"], step_index)
                    expected_row = expected_by_key.get(key)
                    matches = bool(
                        expected_row is not None
                        and int(expected_row["executed_action"]) == int(action)
                    )
                    mismatch_count += int(not matches)
                    if method == "ppo":
                        raw_logits = {
                            "flat": [float(item) for item in policy_output["flat_logits"].tolist()]
                        }
                    else:
                        raw_logits = {
                            head: [
                                float(item)
                                for item in policy_output[f"{head}_logits"].tolist()
                            ]
                            for head in ("slow", "fast", "event")
                        }
                    output_rows.append(
                        {
                            "split": split,
                            "method": method,
                            "seed": seed,
                            "design_id": instance["design_id"],
                            "step_index": step_index,
                            "raw_actor_logits": json.dumps(raw_logits, sort_keys=True),
                            "raw_actor_probs": json.dumps(
                                action_info.get("action_probs", {}), sort_keys=True
                            ),
                            "raw_head_actions": json.dumps(
                                action_info.get("raw_head_actions", {}), sort_keys=True
                            ),
                            "raw_head_actions_source": action_info.get(
                                "raw_head_actions_source", "unspecified"
                            ),
                            "raw_env_action": int(
                                action_info.get("raw_env_action", action)
                            ),
                            "projected_env_action": int(
                                action_info.get("projected_env_action", action)
                            ),
                            "executed_action": int(action),
                            "aggregation_reason": action_info.get(
                                "aggregation_reason", ""
                            ),
                            "action_mask": json.dumps(info.get("action_mask")),
                            "projection_applied": bool(
                                action_info.get("action_projection_applied", False)
                            ),
                            "executed_action_probs": json.dumps(
                                action_info.get("env_action_probs", [])
                            ),
                            "policy_log_prob": action_info.get("log_prob"),
                            "executed_action_log_prob": action_info.get(
                                "env_action_log_prob"
                            ),
                            "preserved_action_match": matches,
                        }
                    )
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
    output_path = run_root / "head_action_reconciliation.csv"
    _write_csv(output_path, output_rows)
    _write_json(
        run_root / "head_action_reconciliation_receipt.json",
        {
            "status": "complete" if mismatch_count == 0 else "failed",
            "mode": "read_only_deterministic_replay_no_selection_no_training",
            "row_count": len(output_rows),
            "preserved_action_mismatch_count": mismatch_count,
            "final_check_used_for_tuning": False,
            "raw_logits_recorded": True,
            "raw_probabilities_recorded": True,
        },
    )
    _rebuild_integrity(run_root)
    integrity_after = _verify_integrity(run_root)
    if mismatch_count or integrity_after["failure_count"]:
        raise RuntimeError(
            f"reconciliation failed: mismatches={mismatch_count}, integrity={integrity_after}"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "row_count": len(output_rows),
                "preserved_action_mismatch_count": mismatch_count,
                "integrity_file_count": integrity_after["verified_file_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
