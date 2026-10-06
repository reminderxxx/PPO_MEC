"""Run the frozen one-factor SA prepare-balance ablation once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
from scripts.run_calibrated_continuous_workflow_pilot_v2 import (  # noqa: E402
    _json_safe,
    _selection_score,
)
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _base_evaluation_row,
    _build_agent,
    _collect_training_episode,
    _load_json,
)
from src.encoders.calibrated_workflow_features import (  # noqa: E402
    bundle_ready,
    predicted_target_rsu_id,
    rsu_by_id,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)


CANDIDATE_METHOD = "sa_ghmappo_no_auxiliary"
ORIGINAL_METHOD = "sa_ghmappo_original"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_feasibility(
    env: CalibratedContinuousWorkflowEnv,
    action_mask: list[bool],
) -> tuple[bool, str]:
    if len(action_mask) <= 4 or not action_mask[4]:
        return False, "action_4_masked_missing_distinct_target"
    clone = env.clone_for_decision_model()
    _, _, _, _, next_info = clone.step(4)
    transition = dict(next_info.get("transition", {}) or {})
    events = list(transition.get("cache_events", []) or [])
    if not events:
        return False, "action_4_no_cache_event"
    reason = str(events[0].get("reason", "unknown"))
    return reason != "contact_budget_exceeded", reason


def _run_extended_evaluation_episode(
    *,
    method: str,
    seed: int,
    config: dict[str, Any],
    instance: dict[str, Any],
    step_cap: int,
    agent: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    counts: Counter[int] = Counter()
    missing_counts: Counter[int] = Counter()
    action4_attempts = 0
    action4_successes = 0
    safe_prepare_attempts = 0
    invalid_prepare_attempts = 0
    feasible_target_prepare_attempts = 0
    infeasible_target_prepare_attempts = 0
    no_progress_streak = 0
    max_no_progress_streak = 0
    decision_ns = 0
    ledger: list[dict[str, Any]] = []
    while not env.terminated and env.step_index < step_cap:
        semantic = info["semantic_state"]
        action_mask = list(info["action_mask"])
        node = semantic.get("current_workflow_node") or {}
        vehicle = (semantic.get("vehicles") or [{}])[0]
        current_rsu_id = vehicle.get("associated_rsu_id")
        target_rsu_id = predicted_target_rsu_id(semantic)
        current_ready = bool(
            bundle_ready(semantic, rsu_by_id(semantic, current_rsu_id), node)
        )
        target_ready = bool(
            bundle_ready(semantic, rsu_by_id(semantic, target_rsu_id), node)
        )
        target_prepare_feasible, feasibility_reason = _prepare_feasibility(
            env, action_mask
        )
        before_completed = len(env.completed)
        started = time.perf_counter_ns()
        action, action_info = agent.act(observation, info)
        decision_ns += time.perf_counter_ns() - started
        action = int(action)
        counts[action] += 1
        if not current_ready:
            missing_counts[action] += 1
        if action == 4:
            action4_attempts += 1
            feasible_target_prepare_attempts += int(target_prepare_feasible)
            infeasible_target_prepare_attempts += int(not target_prepare_feasible)
            service_safe = bool(current_ready and target_prepare_feasible)
            safe_prepare_attempts += int(service_safe)
            invalid_prepare_attempts += int(not service_safe)
        observation, _, terminated, truncated, next_info = env.step(action)
        transition = dict(next_info.get("transition", {}) or {})
        realized_prepare = bool(
            action == 4
            and transition.get("service_completed")
            and transition.get("migration_success")
        )
        action4_successes += int(realized_prepare)
        progressed = len(env.completed) > before_completed
        no_progress_streak = 0 if progressed else no_progress_streak + 1
        max_no_progress_streak = max(max_no_progress_streak, no_progress_streak)
        ledger.append(
            {
                "split": instance["split"],
                "method": method,
                "seed": seed,
                "design_id": instance["design_id"],
                "step_index": int(transition.get("step_index", env.step_index - 1)),
                "node_id": transition.get("node_id"),
                "current_rsu_id": current_rsu_id,
                "target_rsu_id": target_rsu_id,
                "current_bundle_ready": current_ready,
                "target_bundle_ready": target_ready,
                "target_prepare_feasible": target_prepare_feasible,
                "prepare_feasibility_reason": feasibility_reason,
                "service_safe_prepare_state": bool(
                    current_ready and target_prepare_feasible
                ),
                "executed_action": action,
                "event_prepare_prob": action_info.get("event_prepare_prob", ""),
                "event_margin": action_info.get("event_margin", ""),
                "raw_policy_evaluation": action_info.get(
                    "raw_policy_evaluation", ""
                ),
                "env_action_probs": json.dumps(
                    action_info.get("env_action_probs", []), sort_keys=True
                ),
                "policy_log_prob": action_info.get("log_prob", ""),
                "executed_action_log_prob": action_info.get(
                    "env_action_log_prob", ""
                ),
                "realized_prepare": realized_prepare,
                "service_completed": bool(transition.get("service_completed")),
                "migration_success": bool(transition.get("migration_success")),
                "progressed": progressed,
                "no_progress_streak": no_progress_streak,
                "model_transfer_bytes": int(
                    transition.get("model_transfer_bytes", 0) or 0
                ),
                "state_transfer_bytes": int(
                    transition.get("state_transfer_bytes", 0) or 0
                ),
                "recompute_seconds": float(
                    transition.get("recompute_seconds", 0.0) or 0.0
                ),
            }
        )
        info = next_info
        if terminated or truncated:
            break

    row = _base_evaluation_row(
        split=instance["split"],
        method=method,
        seed=seed,
        env=env,
        counts=counts,
        decision_ns=decision_ns,
        action4_attempts=action4_attempts,
        action4_successes=action4_successes,
        missing_current_counts=missing_counts,
        max_no_progress_streak=max_no_progress_streak,
    )
    row.update(
        {
            "on_time_completion_rate": float(
                bool(row["workflow_completion_rate"])
                and float(row["deadline_violation_rate"]) <= 0.0
            ),
            "safe_prepare_attempts": safe_prepare_attempts,
            "invalid_prepare_attempts": invalid_prepare_attempts,
            "feasible_target_prepare_attempts": feasible_target_prepare_attempts,
            "infeasible_target_prepare_attempts": infeasible_target_prepare_attempts,
            "realized_prepare_rate": float(action4_successes)
            / max(action4_attempts, 1),
            "current_missing_action4_rate": float(missing_counts[4])
            / max(sum(missing_counts.values()), 1),
            "result_provenance": "replayed_selected_checkpoint_extended_behavior",
        }
    )
    return row, ledger


def _evaluate_extended(
    *,
    method: str,
    seed: int,
    agent: Any,
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    step_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_deterministic = bool(agent._deterministic_action)
    agent._deterministic_action = True
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for instance in instances:
        row, episode_ledger = _run_extended_evaluation_episode(
            method=method,
            seed=seed,
            config=config,
            instance=instance,
            step_cap=step_cap,
            agent=agent,
        )
        rows.append(row)
        ledger.extend(episode_ledger)
    agent._deterministic_action = old_deterministic
    return rows, ledger


def _aggregate_extended(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "workflow_completion_rate",
        "node_coverage_rate",
        "deadline_violation_rate",
        "service_failure_rate",
        "handoff_failure_rate",
        "total_transfer_mb",
        "model_prepare_mb",
        "state_transfer_mb",
        "recompute_seconds",
        "action4_attempts",
        "action4_successes",
        "safe_prepare_attempts",
        "invalid_prepare_attempts",
        "feasible_target_prepare_attempts",
        "infeasible_target_prepare_attempts",
        "current_missing_action4_rate",
        "max_consecutive_no_progress_steps",
        "reward",
    )
    result: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["method"]))].append(row)
    for (split, method), group in sorted(grouped.items()):
        by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_window[str(row["window_id"])].append(row)
        summary: dict[str, Any] = {}
        for metric in metrics:
            if not all(metric in row and row[metric] != "" for row in group):
                continue
            window_values = [
                float(np.mean([float(row[metric]) for row in window_rows]))
                for window_rows in by_window.values()
            ]
            summary[metric] = {
                "mean": float(np.mean(window_values)),
                "min": float(np.min(window_values)),
                "max": float(np.max(window_values)),
            }
        completed_elapsed = [
            float(row["completed_sample_elapsed_seconds"])
            for row in group
            if row.get("completed_sample_elapsed_seconds", "") != ""
        ]
        summary["completed_sample_elapsed_seconds"] = {
            "mean": float(np.mean(completed_elapsed)) if completed_elapsed else None,
            "sample_n": len(completed_elapsed),
            "episode_n": len(group),
        }
        result.setdefault(split, {})[method] = {
            "outer_unit": "source_window",
            "outer_n": len(by_window),
            "row_n": len(group),
            "metrics": summary,
        }
    return result


def _integrity(output_root: Path) -> None:
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(output_root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/calibrated_workflow_prepare_balance_ablation_v1.json",
    )
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    experiment_path = (ROOT_DIR / args.config).resolve()
    experiment = _load_json(experiment_path)
    base_config_path = (ROOT_DIR / experiment["base_config"]).resolve()
    manifest_path = (ROOT_DIR / experiment["workload_manifest"]).resolve()
    baseline_root = (ROOT_DIR / experiment["historical_baseline_root"]).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "checkpoints").mkdir()
    _write_json(
        output_root / "run_status.json",
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )
    started_at = time.monotonic()

    config, _ = _load_experiment_config(base_config_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(base_config_path):
        raise RuntimeError("base config hash does not match workload manifest")
    if manifest["resolved_config_sha256"] != _canonical_sha256(config):
        raise RuntimeError("resolved base config hash does not match workload manifest")
    if config["interface_profile"] != experiment["interface_profile"]:
        raise RuntimeError("interface profile drift")
    if (
        config["learning_interface"]["hierarchical_action_contract"]
        != experiment["hierarchical_action_contract"]
    ):
        raise RuntimeError("hierarchical action contract drift")
    if bool(config["learning_interface"].get("reward_changed", True)):
        raise RuntimeError("reward-change flag drift")
    factor = experiment["frozen_factor"]
    if set(factor) != {
        "name",
        "implementation_parameter",
        "baseline_value",
        "candidate_value",
        "other_algorithm_parameters_changed",
    }:
        raise RuntimeError("unexpected factor specification")
    if factor["implementation_parameter"] != "auxiliary_coef":
        raise RuntimeError("only the frozen auxiliary coefficient ablation is allowed")
    if float(factor["baseline_value"]) != 0.1 or float(factor["candidate_value"]) != 0.0:
        raise RuntimeError("frozen auxiliary coefficient values drifted")
    if bool(factor["other_algorithm_parameters_changed"]):
        raise RuntimeError("multiple algorithm factors are forbidden")

    frozen_training = experiment["training"]
    seeds = [int(seed) for seed in frozen_training["seeds"]]
    episodes = int(frozen_training["episodes"])
    step_cap = int(frozen_training["episode_max_steps"])
    candidates = {int(value) for value in frozen_training["checkpoint_episode_candidates"]}
    theoretical_cap = len(seeds) * episodes * step_cap
    if seeds != [7, 17, 29] or episodes != 192 or step_cap != 24:
        raise RuntimeError("frozen training identity drift")
    if candidates != {48, 96, 144, 192}:
        raise RuntimeError("checkpoint candidate drift")
    if theoretical_cap != int(frozen_training["new_training_step_cap"]):
        raise RuntimeError("training step cap mismatch")
    if bool(frozen_training["automatic_retry"]):
        raise RuntimeError("automatic retry is forbidden")
    wall_clock_cap = float(frozen_training["wall_clock_cap_seconds"])

    splits = {
        split: [row for row in manifest["instances"] if row["split"] == split]
        for split in ("train", "dev", "regression", "frozen_check")
    }
    if {key: len(value) for key, value in splits.items()} != {
        "train": 12,
        "dev": 4,
        "regression": 12,
        "frozen_check": 8,
    }:
        raise RuntimeError("workload split identity drift")

    baseline_manifest = _load_json(baseline_root / "run_manifest.json")
    if baseline_manifest["config"]["sha256"] != _sha256(base_config_path):
        raise RuntimeError("historical baseline config mismatch")
    if baseline_manifest["workload_manifest"]["sha256"] != _sha256(manifest_path):
        raise RuntimeError("historical baseline workload mismatch")
    if baseline_manifest["seeds"] != seeds or int(
        baseline_manifest["episodes_per_method_seed"]
    ) != episodes:
        raise RuntimeError("historical baseline training identity mismatch")
    if int(baseline_manifest["episode_max_steps"]) != step_cap:
        raise RuntimeError("historical baseline step cap mismatch")

    candidate_curves: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    checkpoint_selection: list[dict[str, Any]] = []
    candidate_agents: dict[int, Any] = {}
    total_steps = 0
    total_updates = 0
    for seed in seeds:
        if time.monotonic() - started_at > wall_clock_cap:
            raise TimeoutError("wall-clock cap exceeded before next seed")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = _build_agent(
            "sa_ghmappo",
            seed,
            config,
            agent_overrides={"auxiliary_coef": 0.0},
        )
        if float(agent._auxiliary_coef) != 0.0:
            raise RuntimeError("candidate auxiliary coefficient not applied")
        seed_started = time.monotonic()
        rng = random.Random(seed)
        order = list(range(len(splits["train"])))
        pending: list[dict[str, Any]] = []
        update_records: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        seed_steps = 0
        for episode_index in range(episodes):
            if time.monotonic() - started_at > wall_clock_cap:
                raise TimeoutError("wall-clock cap exceeded during training")
            if episode_index % len(order) == 0:
                rng.shuffle(order)
            instance = splits["train"][order[episode_index % len(order)]]
            rollout, summary = _collect_training_episode(
                agent, config, instance, step_cap
            )
            pending.extend(rollout)
            seed_steps += len(rollout)
            total_steps += len(rollout)
            candidate_curves.append(
                {
                    "method": CANDIDATE_METHOD,
                    "seed": seed,
                    "episode": episode_index + 1,
                    "design_id": summary["design_id"],
                    "steps": summary["steps"],
                    "workflow_completed": summary["workflow_completed"],
                    "node_coverage_rate": float(summary["completed_nodes"])
                    / max(int(summary["node_count"]), 1),
                    "deadline_violation": summary["deadline_violations"],
                    "modeled_completion_seconds": summary[
                        "modeled_completion_seconds"
                    ],
                    "reward": summary["reward"],
                    "result_provenance": "new_bounded_training",
                }
            )
            if (episode_index + 1) % int(
                frozen_training["update_every_episodes"]
            ) == 0:
                update_records.append(_json_safe(agent.learn(pending)))
                pending = []
                total_updates += 1
            if episode_index + 1 in candidates:
                checkpoint = (
                    output_root
                    / "checkpoints"
                    / f"{CANDIDATE_METHOD}_seed{seed}_episode{episode_index + 1}.pt"
                )
                agent.save(str(checkpoint))
                old_deterministic = bool(agent._deterministic_action)
                agent._deterministic_action = True
                dev_rows: list[dict[str, Any]] = []
                for instance_row in splits["dev"]:
                    row, _ = _run_extended_evaluation_episode(
                        method=CANDIDATE_METHOD,
                        seed=seed,
                        config=config,
                        instance=instance_row,
                        step_cap=step_cap,
                        agent=agent,
                    )
                    dev_rows.append(row)
                agent._deterministic_action = old_deterministic
                candidate_records.append(
                    {
                        "method": CANDIDATE_METHOD,
                        "seed": seed,
                        "episode": episode_index + 1,
                        "score": list(_selection_score(dev_rows)),
                        "checkpoint": str(checkpoint.relative_to(output_root)),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "dev_rows": dev_rows,
                    }
                )
        if pending:
            update_records.append(_json_safe(agent.learn(pending)))
            total_updates += 1
        best = max(
            candidate_records,
            key=lambda row: (tuple(row["score"]), -int(row["episode"])),
        )
        agent.load(str(output_root / best["checkpoint"]))
        selected = (
            output_root / "checkpoints" / f"{CANDIDATE_METHOD}_seed{seed}_selected.pt"
        )
        agent.save(str(selected))
        candidate_agents[seed] = agent
        checkpoint_selection.extend(candidate_records)
        training_summary.append(
            {
                "method": CANDIDATE_METHOD,
                "seed": seed,
                "episodes": episodes,
                "actual_steps": seed_steps,
                "updates": len(update_records),
                "elapsed_seconds": time.monotonic() - seed_started,
                "selected_episode": best["episode"],
                "selected_checkpoint": str(selected.relative_to(output_root)),
                "selected_checkpoint_sha256": _sha256(selected),
                "auxiliary_coef": float(agent._auxiliary_coef),
                "last_update": update_records[-1],
            }
        )
        print(
            json.dumps(
                {
                    "completed": CANDIDATE_METHOD,
                    "seed": seed,
                    "actual_steps": seed_steps,
                    "selected_episode": best["episode"],
                }
            ),
            flush=True,
        )
    if total_steps > theoretical_cap:
        raise RuntimeError("candidate exceeded frozen training step cap")

    evaluation_rows: list[dict[str, Any]] = []
    behavior_ledger: list[dict[str, Any]] = []
    baseline_checkpoint_hashes: dict[str, str] = {}
    for seed in seeds:
        baseline_checkpoint = (
            baseline_root / "checkpoints" / f"sa_ghmappo_seed{seed}_selected.pt"
        )
        baseline_checkpoint_hashes[str(seed)] = _sha256(baseline_checkpoint)
        baseline_agent = _build_agent("sa_ghmappo", seed, config)
        baseline_agent.load(str(baseline_checkpoint))
        if float(baseline_agent._auxiliary_coef) != 0.1:
            raise RuntimeError("historical baseline auxiliary coefficient drift")
        for split in experiment["evaluation"]["splits"]:
            rows, ledger = _evaluate_extended(
                method=ORIGINAL_METHOD,
                seed=seed,
                agent=baseline_agent,
                config=config,
                instances=splits[split],
                step_cap=step_cap,
            )
            evaluation_rows.extend(rows)
            behavior_ledger.extend(ledger)
            rows, ledger = _evaluate_extended(
                method=CANDIDATE_METHOD,
                seed=seed,
                agent=candidate_agents[seed],
                config=config,
                instances=splits[split],
                step_cap=step_cap,
            )
            evaluation_rows.extend(rows)
            behavior_ledger.extend(ledger)

    historical_rows = _load_json(baseline_root / "evaluation_rows.json")
    for row in historical_rows:
        if row["method"] not in set(experiment["evaluation"]["reuse_matching_controls"]):
            continue
        copied = dict(row)
        copied["on_time_completion_rate"] = float(
            bool(copied["workflow_completion_rate"])
            and float(copied["deadline_violation_rate"]) <= 0.0
        )
        copied["result_provenance"] = "historical_exact_match_reuse"
        evaluation_rows.append(copied)

    historical_curves = _load_json(baseline_root / "training_curves.json")
    comparison_curves = []
    for row in historical_curves:
        if row["method"] != "sa_ghmappo":
            continue
        copied = dict(row)
        copied["method"] = ORIGINAL_METHOD
        copied["result_provenance"] = "historical_exact_match_reuse"
        comparison_curves.append(copied)
    comparison_curves.extend(candidate_curves)

    _write_json(output_root / "training_curves.json", comparison_curves)
    _write_csv(output_root / "training_curves.csv", comparison_curves)
    _write_json(output_root / "candidate_training_summary.json", training_summary)
    _write_json(output_root / "candidate_checkpoint_selection.json", checkpoint_selection)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output_root / "behavior_ledger.csv", behavior_ledger)
    _write_json(output_root / "aggregate.json", _aggregate_extended(evaluation_rows))

    elapsed_seconds = time.monotonic() - started_at
    run_manifest = {
        "schema_version": "calibrated_workflow_prepare_balance_ablation_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "experiment_config": {"path": args.config, "sha256": _sha256(experiment_path)},
        "base_config": {
            "path": experiment["base_config"],
            "sha256": _sha256(base_config_path),
        },
        "workload_manifest": {
            "path": experiment["workload_manifest"],
            "sha256": _sha256(manifest_path),
        },
        "historical_baseline": {
            "root": experiment["historical_baseline_root"],
            "run_manifest_sha256": _sha256(baseline_root / "run_manifest.json"),
            "run_git_commit": baseline_manifest["git_commit"],
            "checkpoint_hashes": baseline_checkpoint_hashes,
            "training_and_primary_results_reused": True,
            "selected_checkpoints_replayed_for_extended_behavior_fields": True,
        },
        "frozen_factor": factor,
        "seeds": seeds,
        "episodes_per_candidate_seed": episodes,
        "episode_max_steps": step_cap,
        "new_training_step_ceiling": theoretical_cap,
        "actual_new_training_steps": total_steps,
        "actual_new_updates": total_updates,
        "elapsed_seconds": elapsed_seconds,
        "wall_clock_cap_seconds": wall_clock_cap,
        "automatic_retry": False,
        "evaluation_splits": experiment["evaluation"]["splits"],
        "claim_boundary": experiment["claim_boundary"],
        "real_model_generate_calls": 0,
        "downloads": 0,
        "old_holdout_calls": 0,
        "reward_changed": False,
        "environment_changed": False,
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "candidate": CANDIDATE_METHOD,
            "historical_baseline": ORIGINAL_METHOD,
            "candidate_seeds": seeds,
            "actual_new_training_steps": total_steps,
            "actual_new_updates": total_updates,
            "evaluation_rows": len(evaluation_rows),
            "behavior_ledger_rows": len(behavior_ledger),
            "elapsed_seconds": elapsed_seconds,
        },
    )
    _write_json(
        output_root / "run_status.json",
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "completion_receipt": "completion_receipt.json",
        },
    )
    _integrity(output_root)
    print(json.dumps({"output_root": str(output_root), "status": "complete"}))


if __name__ == "__main__":
    main()
