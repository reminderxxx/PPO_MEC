"""Create an immutable paired and auxiliary-consumer supplement to the bounded audit."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
B_ROOT = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC")
for entry in (str(ROOT), str(B_ROOT)):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

from scripts.diagnose_cscwd_abstention_cost_causal import (
    CANDIDATE, CONTROL, OUTPUT as SOURCE, _prepare, _rebuild_prefix,
)
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from scripts.run_calibrated_workflow_value_normalization_ab import _training_signal_row
from src.encoders.calibrated_workflow_features import bundle_ready, rsu_by_id
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv

OUTPUT = ROOT / "artifacts/analysis/cscwd_abstention_cost_causal_supplement_20261010_v2"
METRICS = ("workflow_completion_rate", "on_time_workflow_completion_rate", "service_failures",
           "modeled_completion_seconds", "model_prepare_mb", "state_transfer_mb",
           "input_transfer_mb", "recompute_seconds", "action_0", "action_2", "action_4")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("empty supplement output")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("create-only supplement already exists")
    source_manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete" or source_manifest.get("selected_sources") != 11 or source_manifest.get("branch_count") != 26:
        raise RuntimeError("bounded source audit incomplete")
    for name, digest in source_manifest["files"].items():
        if _sha256(SOURCE / name) != digest:
            raise RuntimeError(f"bounded source artifact drift: {name}")
    config, instances, _, _, evaluations, episodes, selected = _prepare()
    paired = []
    for view in ("selected", "update96"):
        for seed in (7, 17, 29, 43, 61):
            for split in ("regression", "frozen_check", "all"):
                keys = [key for key in evaluations["candidate"] if key[0] == view and key[2] == seed and (split == "all" or key[1] == split)]
                keys.sort()
                if not keys:
                    raise RuntimeError("empty pair stratum")
                for metric in METRICS:
                    old = statistics.mean(float(evaluations["control"][key][metric]) for key in keys)
                    new = statistics.mean(float(evaluations["candidate"][key][metric]) for key in keys)
                    paired.append({"view": view, "seed": seed, "split": split, "episodes": len(keys),
                                   "metric": metric, "control_mean": old, "candidate_mean": new, "candidate_minus_control": new - old})
    target_rows = []
    candidate_agent = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    for row in selected:
        key = (row["view"], row["split"], row["seed"], row["design_id"])
        env, _, info = _rebuild_prefix(config, instances[key[3]], episodes["candidate"][key], episodes["control"][key], row["first_divergence_step"])
        targets = candidate_agent.agent._build_mechanism_targets(info["semantic_state"])
        target_rows.append({"stratum": row["stratum"], "view": key[0], "split": key[1], "seed": key[2], "design_id": key[3],
                            "step_index": row["first_divergence_step"], "current_bundle_ready": bool(env._bundle_ready(env._current_rsu_id(), env._current_node()["required_adapter"])),
                            "control_action": row["control_action"], "candidate_action": row["candidate_action"],
                            "slow_target": int(targets["slow_target"]), "fast_target": int(targets["fast_target"]),
                            "event_target": int(targets["event_target"]), "confidence_weight": float(targets["confidence_weight"]),
                            "candidate_raw_head_actions": episodes["candidate"][key][row["first_divergence_step"]]["raw_head_actions"]})
    target_counts = {}
    for arm in ("candidate", "control"):
        arm_config = config if arm == "candidate" else {**config, "mechanism_aux_missing_current_event_abstention_enabled": False}
        target_agent = _build_learned("sa_ghmappo", 7, arm_config, popart_enabled=False)
        for view in ("selected", "update96"):
            counts = Counter()
            for key in sorted(evaluations[arm]):
                if key[0] != view:
                    continue
                env = CalibratedContinuousWorkflowEnv(arm_config, instances[key[3]])
                observation, info = env.reset()
                for step in episodes[arm][key]:
                    targets = target_agent.agent._build_mechanism_targets(info["semantic_state"])
                    action = int(step["executed_action"])
                    counts["steps"] += 1
                    counts[f"slow_target_{int(targets['slow_target'])}"] += 1
                    counts[f"fast_target_{int(targets['fast_target'])}"] += 1
                    counts["confidence_eligible"] += int(float(targets["confidence_weight"]) > 1e-6)
                    counts["action_0"] += int(action == 0)
                    counts["action_0_with_slow_target_1"] += int(action == 0 and int(targets["slow_target"]) == 1)
                    observation, _, _, _, info = env.step(action)
            target_counts[f"{arm}|{view}"] = dict(counts)
    training = {}
    for arm, root in (("candidate", CANDIDATE), ("control", CONTROL)):
        signals = [row for row in _rows(root / "training_signal_rows.csv") if row["method"] == "sa_ghmappo"]
        optimizer = [row for row in _rows(root / "optimizer_step_records.csv") if row["method"] == "sa_ghmappo"]
        if len(signals) != 28800 or len(optimizer) != 3840:
            raise RuntimeError("training signal/optimizer denominator drift")
        action_stats = {}
        for action in (0, 1, 2, 3, 4):
            group = [row for row in signals if int(row["action"]) == action]
            action_stats[str(action)] = {"count": len(group),
                                         "mean_raw_advantage": statistics.mean(float(row["advantage_raw"]) for row in group),
                                         "positive_raw_advantage_fraction": sum(float(row["advantage_raw"]) > 0 for row in group) / len(group),
                                         "mean_immediate_reward": statistics.mean(float(row["reward"]) for row in group)}
        training[arm] = {"signal_rows": len(signals), "optimizer_step_rows": len(optimizer),
                         "logged_current_bundle_ready_true": sum(row["current_bundle_ready"] == "True" for row in signals),
                         "nonzero_weighted_auxiliary_grad_records": sum(float(row["weighted_auxiliary_grad_norm"]) > 1e-8 for row in optimizer),
                         "action_stats": action_stats}
        if arm == "candidate":
            training[arm]["optimizer_event_supervision_exposures"] = {
                field: sum(int(row[f"event_aux_supervision_{field}_count"]) for row in optimizer)
                for field in ("eligible", "supervised", "abstained")}
    if training["candidate"]["optimizer_event_supervision_exposures"] != {"eligible": 110072, "supervised": 79796, "abstained": 30276}:
        raise RuntimeError("optimizer supervision exposure drift")
    witness = None
    for instance in (instances[key] for key in sorted(instances)):
        env = CalibratedContinuousWorkflowEnv(config, instance)
        _, info = env.reset()
        semantic = info["semantic_state"]
        rsu_id = semantic["vehicles"][0]["associated_rsu_id"]
        actual = bool(bundle_ready(semantic, rsu_by_id(semantic, rsu_id), semantic["current_workflow_node"]))
        if not actual:
            continue
        synthetic_row = {"action_info": {}, "decision_info": info, "training_episode_index": 1,
                         "rollout_segment_index": 1, "action": 0, "reward": 0.0, "terminated": False,
                         "truncated": False, "value": 0.0, "return": 0.0, "advantage": 0.0, "log_prob": 0.0}
        logged = _training_signal_row(arm="probe", method="sa_ghmappo", seed=7, update_index=1, global_step=1, row=synthetic_row)["current_bundle_ready"]
        witness = {"design_id": instance["design_id"], "public_vehicle_rsu": rsu_id,
                   "semantic_top_level_current_rsu_id": semantic.get("current_rsu_id"),
                   "actual_current_bundle_ready": actual, "logged_current_bundle_ready": logged}
        break
    if witness is None or witness["logged_current_bundle_ready"] is not False:
        raise RuntimeError("training readiness log defect witness not reproduced")
    OUTPUT.mkdir(parents=True)
    _write_csv(OUTPUT / "paired_metric_rows.csv", paired)
    _write_csv(OUTPUT / "selected_target_rows.csv", target_rows)
    _write_json(OUTPUT / "evaluation_target_counts.json", target_counts)
    _write_json(OUTPUT / "training_signal_quality.json", {"schema_version": "cscwd_abstention_training_signal_quality_v1", "training": training,
                                                      "witness": witness, "inference": "logging_consumer_key_mismatch_only; policy_loss_path_unaffected_by_this_field"})
    names = ("paired_metric_rows.csv", "selected_target_rows.csv", "evaluation_target_counts.json", "training_signal_quality.json")
    _write_json(OUTPUT / "analysis_manifest.json", {"schema_version": "cscwd_abstention_cost_causal_supplement_manifest_v1",
                                                "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
                                                "source_bounded_manifest_sha256": _sha256(SOURCE / "analysis_manifest.json"),
                                                "selected_target_rows": len(target_rows), "paired_metric_rows": len(paired),
                                                "additional_counterfactual_env_steps": 0, "training_steps": 0,
                                                "files": {name: _sha256(OUTPUT / name) for name in names}})
    print(json.dumps({"status": "complete", "target_rows": len(target_rows), "paired_rows": len(paired),
                      "training_log_witness": witness}))


if __name__ == "__main__":
    main()
