"""Bounded read-only SA auxiliary-label probe at frozen first-failure states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_prepared_state_event_chain import (
    OUTPUT, _csv, _preflight, _public_state_hash, _write_csv,
)
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv


def main() -> None:
    config, instances, _, episodes, _ = _preflight()
    manifest_path = OUTPUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["episodes_replayed"] != 800 or "auxiliary_probe_rows.csv" in manifest["output_files"]:
        raise RuntimeError("event replay absent or auxiliary probe already completed")
    selected = [row for row in _csv(OUTPUT / "episode_diagnosis_rows.csv")
                if row["method"] == "sa_ghmappo" and row["first_failure_cause"]]
    selected.sort(key=lambda row: (row["split"], row["design_id"], int(row["seed"]),
                                   row["checkpoint_view"], int(row["first_failure_step"])))
    selected = selected[:24]
    agent = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    event_hashes = {(row["checkpoint_view"], row["split"], row["method"], row["seed"],
                     row["design_id"], int(row["step_index"])): row["public_state_sha256"]
                    for row in _csv(OUTPUT / "event_chain_rows.csv")}
    output = []
    for row in selected:
        key = (row["checkpoint_view"], row["split"], row["method"], row["seed"], row["design_id"])
        failure_step = int(row["first_failure_step"])
        records = sorted(episodes[key], key=lambda value: int(value["step_index"]))
        env = CalibratedContinuousWorkflowEnv(config, instances[row["design_id"]])
        observation, info = env.reset()
        for item in records:
            if int(item["step_index"]) == failure_step:
                if _public_state_hash(observation, info) != event_hashes[(*key, failure_step)]:
                    raise RuntimeError(f"public state hash mismatch: {key} {failure_step}")
                targets = agent._build_mechanism_targets(info["semantic_state"])
                probs = json.loads(item["env_action_probs"])
                output.append({"checkpoint_view": row["checkpoint_view"], "split": row["split"],
                               "method": row["method"], "seed": row["seed"], "design_id": row["design_id"],
                               "first_failure_step": failure_step, "public_state_sha256": event_hashes[(*key, failure_step)],
                               "executed_action": int(item["executed_action"]),
                               "current_bundle_ready": item["current_bundle_ready"],
                               "target_bundle_ready": item["target_bundle_ready"],
                               "target_prepared_valid": item["target_prepared_valid"],
                               "pseudo_slow_target": int(targets["slow_target"]),
                               "pseudo_fast_target": int(targets["fast_target"]),
                               "pseudo_event_target": int(targets["event_target"]),
                               "pseudo_event_soft_target": float(targets["event_soft_target"]),
                               "pseudo_confidence_weight": float(targets["confidence_weight"]),
                               "pseudo_prepare_window_score": float(targets["prepare_window_score"]),
                               "raw_head_actions": item["raw_head_actions"],
                               "raw_env_action": int(item["raw_env_action"]),
                               "action4_probability": float(probs[4]), "action0_probability": float(probs[0]),
                               "action2_probability": float(probs[2]),
                               "probe_kind": "pseudo_label_from_public_state_no_policy_forward_no_branch"})
                break
            observation, _, _, _, info = env.step(int(item["executed_action"]))
        else:
            raise RuntimeError(f"first-failure record absent: {key}")
    if len(output) != 24:
        raise RuntimeError("bounded probe count drift")
    path = OUTPUT / "auxiliary_probe_rows.csv"
    _write_csv(path, output)
    manifest["auxiliary_probe"] = {"first_failure_states": 24, "new_policy_forwards": 0,
                                    "action_branches": 0, "suffix_steps": 0}
    manifest["output_files"][path.name] = _sha256(path)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest["auxiliary_probe"], ensure_ascii=False))


if __name__ == "__main__":
    main()
