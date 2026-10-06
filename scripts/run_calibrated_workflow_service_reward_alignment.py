"""Run the frozen two-reward, three-method service-objective experiment once."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _build_agent,
    _collect_training_episode,
    _evaluate_agent,
    _evaluate_rule,
    _git_commit,
    _integrity,
    _json_safe,
    _load_json,
    _sha256,
    _write_csv,
    _write_curve_svg,
    _write_json,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    ORIGINAL_REWARD_PROFILE,
    SERVICE_ALIGNED_REWARD_PROFILE,
)


def _selection_score(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    mean = lambda key: float(np.mean([float(row[key]) for row in rows]))
    completed_elapsed = [
        float(row["completed_sample_elapsed_seconds"])
        for row in rows
        if row.get("completed_sample_elapsed_seconds", "") != ""
    ]
    elapsed = float(np.mean(completed_elapsed)) if completed_elapsed else 1e12
    return (
        mean("on_time_workflow_completion_rate"),
        mean("workflow_completion_rate"),
        -mean("unfinished_after_deadline_rate"),
        -mean("service_failure_rate"),
        -mean("handoff_failure_rate"),
        -elapsed,
        -mean("total_transfer_mb"),
        -mean("recompute_seconds"),
        -mean("invalid_prepare_attempts"),
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "on_time_workflow_completion_rate",
        "workflow_completion_rate",
        "late_workflow_completion_rate",
        "unfinished_after_deadline_rate",
        "node_coverage_rate",
        "service_failure_rate",
        "failed_service_attempt_seconds_proxy",
        "handoff_failure_rate",
        "total_transfer_mb",
        "model_prepare_mb",
        "state_transfer_mb",
        "recompute_seconds",
        "action4_attempts",
        "action4_successes",
        "current_missing_action_4",
        "invalid_prepare_attempts",
        "max_consecutive_no_progress_steps",
        "reward",
        "rescored_return_original_reward_v1",
        "rescored_return_service_aligned_v1",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["reward_arm"]), str(row["method"]))].append(row)
    result: dict[str, Any] = {}
    for (split, reward_arm, method), group in sorted(grouped.items()):
        by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_window[str(row["window_id"])].append(row)
        values: dict[str, Any] = {}
        for metric in metrics:
            if not all(metric in row and row[metric] != "" for row in group):
                continue
            outer = [
                float(np.mean([float(item[metric]) for item in window_rows]))
                for window_rows in by_window.values()
            ]
            values[metric] = {
                "mean": float(np.mean(outer)),
                "min": float(np.min(outer)),
                "max": float(np.max(outer)),
            }
        completed_elapsed = [
            float(row["completed_sample_elapsed_seconds"])
            for row in group
            if row.get("completed_sample_elapsed_seconds", "") != ""
        ]
        values["completed_sample_elapsed_seconds"] = {
            "mean": float(np.mean(completed_elapsed)) if completed_elapsed else None,
            "sample_n": len(completed_elapsed),
            "episode_n": len(group),
        }
        result.setdefault(split, {}).setdefault(reward_arm, {})[method] = {
            "outer_unit": "source_window",
            "outer_n": len(by_window),
            "row_n": len(group),
            "metrics": values,
        }
    return result


def _stratified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in ("handoff_pressure", "link_error_class", "capacity", "sharing", "topology_class"):
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["reward_arm"]), str(row["method"]), field, str(row[field]))].append(row)
        for (arm, method, name, value), group in sorted(grouped.items()):
            output.append(
                {
                    "reward_arm": arm,
                    "method": method,
                    "stratum": name,
                    "value": value,
                    "row_n": len(group),
                    "source_window_n": len({str(row["window_id"]) for row in group}),
                    "on_time_workflow_completion_rate": float(np.mean([float(row["on_time_workflow_completion_rate"]) for row in group])),
                    "workflow_completion_rate": float(np.mean([float(row["workflow_completion_rate"]) for row in group])),
                    "unfinished_after_deadline_rate": float(np.mean([float(row["unfinished_after_deadline_rate"]) for row in group])),
                    "invalid_prepare_attempts": float(np.mean([float(row["invalid_prepare_attempts"]) for row in group])),
                    "service_failure_rate": float(np.mean([float(row["service_failure_rate"]) for row in group])),
                }
            )
    return output


def _resolved_arm_config(base: dict[str, Any], experiment: dict[str, Any], profile: str) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["reward_profile"] = profile
    config["objective"].setdefault("reward_profiles", {})[
        SERVICE_ALIGNED_REWARD_PROFILE
    ] = dict(experiment["service_aligned_reward"])
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/calibrated_workflow_service_reward_alignment_v1.json",
    )
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    experiment_path = (ROOT_DIR / args.config).resolve()
    experiment = _load_json(experiment_path)
    base_path = (ROOT_DIR / experiment["base_config"]).resolve()
    manifest_path = (ROOT_DIR / experiment["workload_manifest"]).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "checkpoints").mkdir()
    _write_json(output_root / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    started_at = time.monotonic()

    base, _ = _load_experiment_config(base_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(base_path):
        raise RuntimeError("frozen base-config hash mismatch")
    if manifest["resolved_config_sha256"] != _canonical_sha256(base):
        raise RuntimeError("frozen resolved-config hash mismatch")
    if base["interface_profile"] != experiment["interface_profile"]:
        raise RuntimeError("interface profile drift")
    if base["learning_interface"]["hierarchical_action_contract"] != experiment["hierarchical_action_contract"]:
        raise RuntimeError("hierarchical action contract drift")
    if float(_build_agent("sa_ghmappo", 7, base)._auxiliary_coef) != 0.1:
        raise RuntimeError("SA auxiliary_coef must remain 0.1")

    training = experiment["training"]
    methods = list(training["agents"])
    seeds = [int(seed) for seed in training["seeds"]]
    episodes = int(training["episodes"])
    step_cap = int(training["episode_max_steps"])
    candidates = {int(value) for value in training["checkpoint_episode_candidates"]}
    arms = list(experiment["reward_arms"])
    ceiling = len(arms) * len(methods) * len(seeds) * episodes * step_cap
    if methods != ["sa_ghmappo", "mappo", "ppo"] or seeds != [7, 17, 29]:
        raise RuntimeError("method/seed identity drift")
    if episodes != 192 or step_cap != 24 or candidates != {48, 96, 144, 192}:
        raise RuntimeError("frozen episode/checkpoint identity drift")
    if ceiling != 82944 or ceiling != int(training["theoretical_total_step_cap"]):
        raise RuntimeError("frozen total training bound drift")
    if bool(training["automatic_retry"]):
        raise RuntimeError("automatic retry is forbidden")

    splits = {
        split: [row for row in manifest["instances"] if row["split"] == split]
        for split in ("train", "dev", "regression", "frozen_check")
    }
    if {key: len(value) for key, value in splits.items()} != {"train": 12, "dev": 4, "regression": 12, "frozen_check": 8}:
        raise RuntimeError("workload split identity drift")

    training_curves: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    checkpoint_selection: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    behavior_ledger: list[dict[str, Any]] = []
    total_steps = total_updates = 0
    wall_clock_cap = float(training["wall_clock_cap_seconds"])

    for arm in arms:
        reward_arm = str(arm["arm"])
        profile = str(arm["reward_profile"])
        if profile not in {ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE}:
            raise RuntimeError("unexpected reward profile")
        config = _resolved_arm_config(base, experiment, profile)
        for method in methods:
            for seed in seeds:
                if time.monotonic() - started_at > wall_clock_cap:
                    raise TimeoutError("frozen wall-clock cap exceeded")
                seed_started = time.monotonic()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                agent = _build_agent(method, seed, config)
                if method == "sa_ghmappo" and float(agent._auxiliary_coef) != 0.1:
                    raise RuntimeError("SA auxiliary target identity drift")
                rng = random.Random(seed)
                order = list(range(len(splits["train"])))
                pending: list[dict[str, Any]] = []
                update_records: list[dict[str, Any]] = []
                candidate_records: list[dict[str, Any]] = []
                seed_steps = 0
                for episode_index in range(episodes):
                    if time.monotonic() - started_at > wall_clock_cap:
                        raise TimeoutError("frozen wall-clock cap exceeded during training")
                    if episode_index % len(order) == 0:
                        rng.shuffle(order)
                    instance = splits["train"][order[episode_index % len(order)]]
                    rollout, summary = _collect_training_episode(agent, config, instance, step_cap)
                    pending.extend(rollout)
                    seed_steps += len(rollout)
                    total_steps += len(rollout)
                    training_curves.append(
                        {
                            "reward_arm": reward_arm,
                            "method": method,
                            "seed": seed,
                            "episode": episode_index + 1,
                            "design_id": summary["design_id"],
                            "steps": summary["steps"],
                            "workflow_completed": summary["workflow_completed"],
                            "on_time_workflow_completed": summary["on_time_workflow_completed"],
                            "deadline_missed": summary["deadline_missed"],
                            "service_failures": summary["service_failures"],
                            "node_coverage_rate": float(summary["completed_nodes"]) / max(int(summary["node_count"]), 1),
                            "reward": summary["reward"],
                        }
                    )
                    if (episode_index + 1) % int(training["update_every_episodes"]) == 0:
                        update_records.append(_json_safe(agent.learn(pending)))
                        total_updates += 1
                        pending = []
                    if episode_index + 1 in candidates:
                        checkpoint = output_root / "checkpoints" / f"{reward_arm}_{method}_seed{seed}_episode{episode_index + 1}.pt"
                        agent.save(str(checkpoint))
                        dev_rows, _ = _evaluate_agent(agent, method, seed, config, splits["dev"], step_cap)
                        score = list(_selection_score(dev_rows))
                        candidate_records.append(
                            {
                                "reward_arm": reward_arm,
                                "method": method,
                                "seed": seed,
                                "episode": episode_index + 1,
                                "score": score,
                                "score_fields": list(training["checkpoint_selection"]["order"]),
                                "reward_value_used": False,
                                "checkpoint": str(checkpoint.relative_to(output_root)),
                                "checkpoint_sha256": _sha256(checkpoint),
                                "dev_rows": dev_rows,
                            }
                        )
                if pending:
                    update_records.append(_json_safe(agent.learn(pending)))
                    total_updates += 1
                best = max(candidate_records, key=lambda row: (tuple(row["score"]), -int(row["episode"])))
                agent.load(str(output_root / best["checkpoint"]))
                selected = output_root / "checkpoints" / f"{reward_arm}_{method}_seed{seed}_selected.pt"
                agent.save(str(selected))
                for split in experiment["evaluation"]["splits"]:
                    rows, ledger = _evaluate_agent(agent, method, seed, config, splits[split], step_cap)
                    for row in rows:
                        row["reward_arm"] = reward_arm
                    for row in ledger:
                        row["reward_arm"] = reward_arm
                    evaluation_rows.extend(rows)
                    behavior_ledger.extend(ledger)
                checkpoint_selection.extend(candidate_records)
                training_summary.append(
                    {
                        "reward_arm": reward_arm,
                        "method": method,
                        "seed": seed,
                        "episodes": episodes,
                        "actual_steps": seed_steps,
                        "updates": len(update_records),
                        "elapsed_seconds": time.monotonic() - seed_started,
                        "selected_episode": best["episode"],
                        "selected_checkpoint": str(selected.relative_to(output_root)),
                        "selected_checkpoint_sha256": _sha256(selected),
                        "auxiliary_coef": float(agent._auxiliary_coef) if method == "sa_ghmappo" else None,
                        "last_update": update_records[-1] if update_records else None,
                    }
                )
                print(json.dumps({"reward_arm": reward_arm, "completed": method, "seed": seed, "actual_steps": seed_steps, "selected_episode": best["episode"]}), flush=True)

        rule_rows_by_split: list[dict[str, Any]] = []
        for split in experiment["evaluation"]["splits"]:
            rows, ledger = _evaluate_rule(config, splits[split], step_cap)
            for row in rows:
                row["reward_arm"] = reward_arm
            for row in ledger:
                row["reward_arm"] = reward_arm
            rule_rows_by_split.extend(rows)
            behavior_ledger.extend(ledger)
        evaluation_rows.extend(rule_rows_by_split)

    if total_steps > ceiling:
        raise RuntimeError("observed training steps exceeded frozen cap")
    for arm in (ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE):
        arm_rows = [row for row in training_curves if row["reward_arm"] == arm]
        _write_curve_svg(output_root / f"training_completion_curve_{arm}.svg", arm_rows)
    _write_json(output_root / "training_curves.json", training_curves)
    _write_csv(output_root / "training_curves.csv", training_curves)
    _write_json(output_root / "training_summary.json", training_summary)
    _write_json(output_root / "checkpoint_selection.json", checkpoint_selection)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output_root / "behavior_ledger.csv", behavior_ledger)
    _write_json(output_root / "aggregate.json", _aggregate(evaluation_rows))
    _write_csv(output_root / "stratified_results.csv", _stratified(evaluation_rows))

    elapsed = time.monotonic() - started_at
    run_manifest = {
        "schema_version": "calibrated_workflow_service_reward_alignment_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "experiment_config": {"path": args.config, "sha256": _sha256(experiment_path)},
        "base_config": {"path": experiment["base_config"], "sha256": _sha256(base_path)},
        "workload_manifest": {"path": experiment["workload_manifest"], "sha256": _sha256(manifest_path)},
        "interface_profile": experiment["interface_profile"],
        "hierarchical_action_contract": experiment["hierarchical_action_contract"],
        "reward_arms": arms,
        "methods": methods,
        "rules": training["rules"],
        "seeds": seeds,
        "episodes_per_method_seed": episodes,
        "episode_max_steps": step_cap,
        "theoretical_training_step_cap": ceiling,
        "actual_training_steps": total_steps,
        "actual_updates": total_updates,
        "elapsed_seconds": elapsed,
        "checkpoint_selection": training["checkpoint_selection"],
        "claim_boundary": experiment["claim_boundary"],
        "original_reward_reused": False,
        "automatic_retry": False,
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(output_root / "completion_receipt.json", {"status": "complete", "actual_training_steps": total_steps, "actual_updates": total_updates, "evaluation_rows": len(evaluation_rows), "behavior_ledger_rows": len(behavior_ledger), "elapsed_seconds": elapsed})
    _write_json(output_root / "run_status.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "completion_receipt": "completion_receipt.json"})
    _integrity(output_root)
    print(json.dumps({"output_root": str(output_root), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
