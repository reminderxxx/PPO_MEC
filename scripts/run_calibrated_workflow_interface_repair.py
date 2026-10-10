"""Run the bounded calibrated-workflow interface-repair validation once."""

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

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_continuous_workflow_pilot_v2 import (
    PRIMARY_METRICS,
    _json_safe,
    _selection_score,
)
from src.agents.registry import build_agent
from src.encoders.calibrated_workflow_features import (
    bundle_ready,
    predicted_target_rsu_id,
    rsu_by_id,
)
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    TwoStepCostRule,
)
from src.trainers.ppo_buffer import PPORolloutBuffer


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def _build_agent(
    method: str,
    seed: int,
    config: dict[str, Any],
    agent_overrides: dict[str, Any] | None = None,
) -> Any:
    training = config["training"]
    kwargs: dict[str, Any] = {
        "random_seed": seed,
        "learning_rate": float(training["learning_rate"]),
        "clip_ratio": float(training["clip_ratio"]),
        "entropy_coef": float(training["entropy_coef"]),
        "value_coef": float(training["value_coef"]),
        "batch_size": 32,
        "train_epochs": 4,
        "deterministic_action": False,
        "prepared_state_features_enabled": config.get("interface_profile") == "calibrated_workflow_interface_v4_prepared_state_prefix",
    }
    if method == "sa_ghmappo":
        kwargs.update(
            {
                "mechanism_aux_current_service_feasibility_gate_enabled": bool(
                    config.get(
                        "mechanism_aux_current_service_feasibility_gate_enabled",
                        False,
                    )
                ),
                "mechanism_aux_missing_current_event_abstention_enabled": bool(
                    config.get(
                        "mechanism_aux_missing_current_event_abstention_enabled",
                        False,
                    )
                ),
            }
        )
    if method in {"sa_ghmappo", "mappo", "dt_handoff_drl"}:
        interface = config["learning_interface"]
        kwargs.update(
            {
                "hierarchical_action_contract": interface[
                    "hierarchical_action_contract"
                ],
                "executed_action_ppo_only": bool(
                    interface["executed_action_ppo_only"]
                ),
                "env_action_ppo_enabled": bool(interface["env_action_ppo_enabled"]),
                "env_action_ppo_coef": float(interface["env_action_ppo_coef"]),
            }
        )
    kwargs.update(dict(agent_overrides or {}))
    return build_agent(method, **kwargs)


def _collect_training_episode(
    agent: Any,
    config: dict[str, Any],
    instance: dict[str, Any],
    step_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    buffer = PPORolloutBuffer()
    terminated = truncated = False
    while not terminated and not truncated and env.step_index < step_cap:
        decision_info = dict(info)
        action, action_info = agent.act(observation, decision_info)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        value = float(
            action_info.get("value", agent.evaluate_value(observation, decision_info))
        )
        buffer.add_step(
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            log_prob=float(action_info.get("log_prob", 0.0)),
            value=value,
            next_observation=next_observation,
            action_info=action_info,
            decision_info=decision_info,
            env_info=next_info,
        )
        observation, info = next_observation, next_info
    last_value = 0.0 if terminated else float(agent.evaluate_value(observation, info))
    training = config["training"]
    buffer.finalize(
        last_value=last_value,
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
    )
    return buffer.to_training_rows(), env.summary()


def _base_evaluation_row(
    *,
    split: str,
    method: str,
    seed: int | None,
    env: CalibratedContinuousWorkflowEnv,
    counts: Counter[int],
    decision_ns: int,
    action4_attempts: int,
    action4_successes: int,
    missing_current_counts: Counter[int],
    max_no_progress_streak: int,
) -> dict[str, Any]:
    summary = env.summary()
    nodes = int(summary["node_count"])
    completed = int(summary["completed_nodes"])
    total_bytes = int(summary["total_transfer_bytes"])
    factors = env.instance["factors"]
    completed_workflow = bool(summary["workflow_completed"])
    row = {
        "split": split,
        "method": method,
        "seed": "rule" if seed is None else seed,
        "design_id": summary["design_id"],
        "window_id": summary["window_id"],
        "workflow_id": summary["workflow_id"],
        "source_segment_id": env.instance["source_interval"]["source_segment_id"],
        "sharing": factors["sharing"],
        "capacity": factors["capacity"],
        "state_scale": factors["state_scale"],
        "initial_target": factors["initial_target"],
        "link_error_class": env.instance.get("link_profile", {}).get(
            "error_class", "matched"
        ),
        "handoff_pressure": env.instance["trace_features"]["handoff_pressure"],
        "topology_class": env.instance["workflow_features"]["topology_class"],
        "node_count": nodes,
        "completed_nodes": completed,
        "workflow_completion_rate": float(completed_workflow),
        "node_coverage_rate": float(completed) / max(nodes, 1),
        "unfinished_rate": float(not completed_workflow),
        "truncated_rate": float(not bool(summary["terminated"])),
        "deadline_violation_rate": float(summary["deadline_violations"]),
        "deadline_missed_rate": float(summary.get("deadline_missed", 0)),
        "on_time_workflow_completion_rate": float(
            summary.get("on_time_workflow_completed", 0)
        ),
        "late_workflow_completion_rate": float(
            summary.get("late_workflow_completed", 0)
        ),
        "unfinished_after_deadline_rate": float(
            summary.get("unfinished_after_deadline", 0)
        ),
        "service_failure_rate": float(summary["service_failures"] > 0),
        "service_failures": int(summary["service_failures"]),
        "failed_service_attempt_seconds_proxy": float(
            summary.get("failed_service_attempt_seconds_proxy", 0.0)
        ),
        "handoff_failure_rate": float(summary["handoff_failures"])
        / max(int(summary["handoff_count"]), 1),
        "handoff_failures": int(summary["handoff_failures"]),
        "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
        "completed_sample_elapsed_seconds": (
            float(summary["modeled_completion_seconds"]) if completed_workflow else ""
        ),
        "completed_sample_elapsed_coverage": float(completed_workflow),
        "total_transfer_mb": float(total_bytes) / 1_000_000.0,
        "model_prepare_mb": float(summary["model_transfer_bytes"]) / 1_000_000.0,
        "state_transfer_mb": float(summary["state_transfer_bytes"]) / 1_000_000.0,
        "input_transfer_mb": float(summary["input_transfer_bytes"]) / 1_000_000.0,
        "model_load_seconds": float(summary["model_load_seconds"]),
        "state_restore_seconds": float(summary["state_restore_seconds"]),
        "recompute_seconds": float(summary["recompute_seconds"]),
        "migration_attempts": int(summary["migration_attempts"]),
        "migration_successes": int(summary["migration_successes"]),
        "action4_attempts": action4_attempts,
        "action4_successes": action4_successes,
        "action4_failures": action4_attempts - action4_successes,
        "max_consecutive_no_progress_steps": max_no_progress_streak,
        "current_model_missing_decisions": int(sum(missing_current_counts.values())),
        "decision_overhead_ms": float(decision_ns) / 1_000_000.0,
        "reward": float(summary["reward"]),
        "reward_node_completion": float(summary["reward_node_completion"]),
        "reward_workflow_completion": float(summary["reward_workflow_completion"]),
        "reward_time_penalty": float(summary["reward_time_penalty"]),
        "reward_transfer_penalty": float(summary["reward_transfer_penalty"]),
        "reward_failure_penalty": float(summary["reward_failure_penalty"]),
        "reward_deadline_penalty": float(summary["reward_deadline_penalty"]),
        "reward_service_operation_time_penalty": float(
            summary.get("reward_service_operation_time_penalty", 0.0)
        ),
        "reward_recompute_penalty": float(
            summary.get("reward_recompute_penalty", 0.0)
        ),
        "reward_on_time_completion": float(
            summary.get("reward_on_time_completion", 0.0)
        ),
        "reward_profile": summary.get("reward_profile", "original_reward_v1"),
    }
    for profile, value in dict(summary.get("reward_totals_by_profile", {})).items():
        row[f"rescored_return_{profile}"] = float(value)
    for action_id in range(5):
        row[f"action_{action_id}"] = int(counts[action_id])
        row[f"current_missing_action_{action_id}"] = int(
            missing_current_counts[action_id]
        )
    return row


def _run_evaluation_episode(
    *,
    method: str,
    seed: int | None,
    config: dict[str, Any],
    instance: dict[str, Any],
    step_cap: int,
    agent: Any | None = None,
    rule: TwoStepCostRule | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    counts: Counter[int] = Counter()
    missing_counts: Counter[int] = Counter()
    action4_attempts = action4_successes = 0
    safe_prepare_attempts = invalid_prepare_attempts = 0
    feasible_target_prepare_attempts = infeasible_target_prepare_attempts = 0
    no_progress_streak = max_no_progress_streak = 0
    decision_ns = 0
    ledger: list[dict[str, Any]] = []
    while not env.terminated and env.step_index < step_cap:
        semantic = info["semantic_state"]
        prepared_prefix = (
            (semantic.get("calibrated_context") or {}).get("prepared_state_prefix")
            or {}
        )
        current_prepared = prepared_prefix.get("current") or {}
        target_prepared = prepared_prefix.get("predicted_target") or {}
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
        action_mask = list(info["action_mask"])
        target_prepare_feasible = False
        prepare_feasibility_reason = "action_4_masked_missing_distinct_target"
        if len(action_mask) > 4 and action_mask[4]:
            preview = env.clone_for_decision_model()
            _, _, _, _, preview_info = preview.step(4)
            preview_transition = dict(preview_info.get("transition", {}) or {})
            preview_events = list(preview_transition.get("cache_events", []) or [])
            if preview_events:
                prepare_feasibility_reason = str(
                    preview_events[0].get("reason", "unknown")
                )
                target_prepare_feasible = (
                    prepare_feasibility_reason != "contact_budget_exceeded"
                )
            else:
                prepare_feasibility_reason = "action_4_no_cache_event"
        before_completed = len(env.completed)
        started = time.perf_counter_ns()
        if agent is not None:
            action, action_info = agent.act(observation, info)
        else:
            assert rule is not None
            action = rule.select_action(env)
            action_info = {
                "raw_head_actions": {},
                "raw_head_actions_source": "not_applicable_model_based_planner",
                "raw_env_action": action,
                "projected_env_action": action,
                "final_env_action": action,
                "action_projection_applied": False,
                "aggregation_reason": "model_based_planner",
                "env_action_probs": [],
            }
        decision_ns += time.perf_counter_ns() - started
        counts[int(action)] += 1
        if not current_ready:
            missing_counts[int(action)] += 1
        observation, _, terminated, truncated, next_info = env.step(action)
        transition = dict(next_info.get("transition", {}))
        if int(action) == 4:
            action4_attempts += 1
            feasible_target_prepare_attempts += int(target_prepare_feasible)
            infeasible_target_prepare_attempts += int(not target_prepare_feasible)
            service_safe_prepare = bool(current_ready and target_prepare_feasible)
            safe_prepare_attempts += int(service_safe_prepare)
            invalid_prepare_attempts += int(not service_safe_prepare)
            action4_successes += int(
                bool(transition.get("service_completed"))
                and bool(transition.get("migration_success"))
            )
        progressed = len(env.completed) > before_completed
        no_progress_streak = 0 if progressed else no_progress_streak + 1
        max_no_progress_streak = max(max_no_progress_streak, no_progress_streak)
        ledger.append(
            {
                "split": instance["split"],
                "method": method,
                "seed": "rule" if seed is None else seed,
                "design_id": instance["design_id"],
                "step_index": int(transition.get("step_index", env.step_index - 1)),
                "node_id": transition.get("node_id"),
                "current_rsu_id": current_rsu_id,
                "target_rsu_id": target_rsu_id,
                "prediction_provenance": json.dumps(
                    semantic.get("predictions", {}).get("causal_provenance", {}), sort_keys=True
                ),
                "predicted_next_rsu_id": semantic.get("predictions", {}).get("predicted_next_rsu_by_vehicle", {}).get("veh_pilot"),
                "actual_next_rsu_id": (
                    instance["rsu_sequence"][min(int(transition.get("step_index", 0)) + 1, len(instance["rsu_sequence"]) - 1)]
                    if semantic.get("interface_profile") in {
                        "calibrated_workflow_interface_v3_prefix_only",
                        "calibrated_workflow_interface_v4_prepared_state_prefix",
                    }
                    else None
                ),
                "current_prepared_known": current_prepared.get("known", ""),
                "current_prepared_exists": current_prepared.get("exists", ""),
                "current_prepared_valid": current_prepared.get("valid", ""),
                "current_prepared_missing_completed_count": current_prepared.get(
                    "missing_completed_count", ""
                ),
                "current_prepared_missing_completed_fraction": current_prepared.get(
                    "missing_completed_fraction", ""
                ),
                "target_prepared_known": target_prepared.get("known", ""),
                "target_prepared_exists": target_prepared.get("exists", ""),
                "target_prepared_valid": target_prepared.get("valid", ""),
                "target_prepared_missing_completed_count": target_prepared.get(
                    "missing_completed_count", ""
                ),
                "target_prepared_missing_completed_fraction": target_prepared.get(
                    "missing_completed_fraction", ""
                ),
                "current_bundle_ready": current_ready,
                "target_bundle_ready": target_ready,
                "target_prepare_feasible": target_prepare_feasible,
                "prepare_feasibility_reason": prepare_feasibility_reason,
                "raw_head_actions": json.dumps(
                    action_info.get("raw_head_actions", {}), sort_keys=True
                ),
                "raw_head_actions_source": action_info.get(
                    "raw_head_actions_source", "unspecified"
                ),
                "raw_env_action": int(action_info.get("raw_env_action", action)),
                "projected_env_action": int(
                    action_info.get("projected_env_action", action)
                ),
                "executed_action": int(action),
                "projection_applied": bool(
                    action_info.get("action_projection_applied", False)
                ),
                "aggregation_reason": action_info.get("aggregation_reason", ""),
                "env_action_probs": json.dumps(
                    action_info.get("env_action_probs", []), sort_keys=True
                ),
                "policy_log_prob": action_info.get("log_prob", ""),
                "executed_action_log_prob": action_info.get(
                    "env_action_log_prob", ""
                ),
                "service_completed": bool(transition.get("service_completed")),
                "migration_success": bool(transition.get("migration_success")),
                "state_ready": bool(transition.get("state_ready")),
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
                "service_operation_seconds": float(
                    transition.get("service_operation_seconds", 0.0) or 0.0
                ),
                "clock_seconds_after": float(
                    transition.get("clock_seconds_after", 0.0) or 0.0
                ),
                "deadline_miss_event": bool(
                    transition.get("deadline_miss_event", False)
                ),
                "reward_components": json.dumps(
                    transition.get("reward_components", {}), sort_keys=True
                ),
                "reward_components_by_profile": json.dumps(
                    transition.get("reward_components_by_profile", {}),
                    sort_keys=True,
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
            "safe_prepare_attempts": safe_prepare_attempts,
            "invalid_prepare_attempts": invalid_prepare_attempts,
            "feasible_target_prepare_attempts": feasible_target_prepare_attempts,
            "infeasible_target_prepare_attempts": infeasible_target_prepare_attempts,
            "realized_prepare_rate": float(action4_successes)
            / max(action4_attempts, 1),
            "current_missing_action4_rate": float(missing_counts[4])
            / max(sum(missing_counts.values()), 1),
        }
    )
    return row, ledger


def _evaluate_agent(
    agent: Any,
    method: str,
    seed: int | None,
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    step_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_deterministic = bool(getattr(agent, "_deterministic_action", False))
    agent._deterministic_action = True
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for instance in instances:
        row, episode_ledger = _run_evaluation_episode(
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


def _evaluate_rule(
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    step_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule = TwoStepCostRule()
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for instance in instances:
        row, episode_ledger = _run_evaluation_episode(
            method=rule.method_name,
            seed=None,
            config=config,
            instance=instance,
            step_cap=step_cap,
            rule=rule,
        )
        rows.append(row)
        ledger.extend(episode_ledger)
    return rows, ledger


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    result: dict[str, Any] = {}
    extra_metrics = (
        "action4_attempts",
        "action4_successes",
        "action4_failures",
        "max_consecutive_no_progress_steps",
        "current_model_missing_decisions",
        "completed_sample_elapsed_coverage",
    )
    for method, method_rows in sorted(grouped.items()):
        by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            by_window[str(row["window_id"])].append(row)
        metrics: dict[str, Any] = {}
        for metric in (*PRIMARY_METRICS, *extra_metrics):
            values = [
                float(np.mean([float(item[metric]) for item in window_rows]))
                for window_rows in by_window.values()
            ]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        completed_elapsed = [
            float(row["completed_sample_elapsed_seconds"])
            for row in method_rows
            if row["completed_sample_elapsed_seconds"] != ""
        ]
        metrics["completed_sample_elapsed_seconds"] = {
            "mean": float(np.mean(completed_elapsed)) if completed_elapsed else None,
            "sample_n": len(completed_elapsed),
            "episode_n": len(method_rows),
        }
        result[method] = {
            "outer_unit": "source_window",
            "outer_n": len(by_window),
            "metrics": metrics,
        }
    return result


def _stratified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    fields = (
        "handoff_pressure",
        "link_error_class",
        "capacity",
        "sharing",
        "topology_class",
    )
    for field in fields:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["method"]), str(row[field]))].append(row)
        for (method, value), group in sorted(grouped.items()):
            result.append(
                {
                    "stratum": field,
                    "value": value,
                    "method": method,
                    "row_n": len(group),
                    "source_window_n": len({str(row["window_id"]) for row in group}),
                    "workflow_completion_rate": float(
                        np.mean([float(row["workflow_completion_rate"]) for row in group])
                    ),
                    "node_coverage_rate": float(
                        np.mean([float(row["node_coverage_rate"]) for row in group])
                    ),
                    "action4_failure_rate": float(
                        sum(int(row["action4_failures"]) for row in group)
                        / max(sum(int(row["action4_attempts"]) for row in group), 1)
                    ),
                    "max_consecutive_no_progress_steps": float(
                        np.mean(
                            [
                                float(row["max_consecutive_no_progress_steps"])
                                for row in group
                            ]
                        )
                    ),
                }
            )
    return result


def _write_curve_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {"sa_ghmappo": "#1367d1", "mappo": "#d16900", "ppo": "#2b8a3e"}
    width, height = 900, 480
    left, right, top, bottom = 65, 25, 40, 65
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["method"])][int(row["episode"])].append(
            float(row["workflow_completed"])
        )
    x0, y0, x1, y1 = left, height - bottom, width - right, top
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">Bounded training completion: cumulative mean</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = y0 - value * (y0 - y1)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    max_episode = max(int(row["episode"]) for row in rows)
    legend_x = 100
    for method in ("sa_ghmappo", "mappo", "ppo"):
        cumulative: list[float] = []
        points: list[str] = []
        for episode in sorted(grouped[method]):
            cumulative.extend(grouped[method][episode])
            value = float(np.mean(cumulative))
            x = x0 + episode / max_episode * (x1 - x0)
            y = y0 - value * (y0 - y1)
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="2"/>')
        svg.append(f'<line x1="{legend_x}" y1="{height - 24}" x2="{legend_x + 24}" y2="{height - 24}" stroke="{colors[method]}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 30}" y="{height - 19}" font-family="sans-serif" font-size="12">{method}</text>')
        legend_x += 230
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


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
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json",
    )
    parser.add_argument(
        "--manifest",
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json",
    )
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    manifest_path = ROOT_DIR / args.manifest
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "checkpoints").mkdir()
    started_at = time.monotonic()
    _write_json(
        output_root / "run_status.json",
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )

    config, base_config_path = _load_experiment_config(config_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("frozen manifest config hash mismatch")
    if manifest.get("resolved_config_sha256") != _canonical_sha256(config):
        raise RuntimeError("frozen manifest resolved-config hash mismatch")
    if config["interface_profile"] != "calibrated_workflow_interface_v2":
        raise RuntimeError("repaired interface profile is not active")
    training = config["training"]
    methods = list(training["agents"])
    seeds = [int(seed) for seed in training["seeds"]]
    episodes = int(training["episodes"])
    candidates = {int(value) for value in training["checkpoint_episode_candidates"]}
    step_cap = int(training["budget"]["episode_max_steps"])
    ceiling = len(methods) * len(seeds) * episodes * step_cap
    if episodes > 192 or max(candidates) != episodes:
        raise RuntimeError("episode/checkpoint budget exceeds frozen bound")
    if ceiling > int(training["budget"]["total_training_step_cap"]):
        raise RuntimeError("training interaction ceiling exceeds frozen cap")
    if set(methods) != {"sa_ghmappo", "ppo", "mappo"}:
        raise RuntimeError("unexpected learning-method set")
    wall_clock_cap = float(training["wall_clock_cap_seconds"])
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
        raise RuntimeError("unexpected frozen split counts")

    training_curves: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    checkpoint_selection: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    action_ledger: list[dict[str, Any]] = []
    total_steps = 0
    total_updates = 0

    for method in methods:
        for seed in seeds:
            if time.monotonic() - started_at > wall_clock_cap:
                raise TimeoutError("frozen wall-clock cap exceeded before next seed")
            seed_started = time.monotonic()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = _build_agent(method, seed, config)
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
                rollout, summary = _collect_training_episode(
                    agent, config, instance, step_cap
                )
                pending.extend(rollout)
                seed_steps += len(rollout)
                total_steps += len(rollout)
                training_curves.append(
                    {
                        "method": method,
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
                    }
                )
                if (episode_index + 1) % int(training["update_every_episodes"]) == 0:
                    update_records.append(_json_safe(agent.learn(pending)))
                    total_updates += 1
                    pending = []
                if episode_index + 1 in candidates:
                    checkpoint = (
                        output_root
                        / "checkpoints"
                        / f"{method}_seed{seed}_episode{episode_index + 1}.pt"
                    )
                    agent.save(str(checkpoint))
                    dev_rows, _ = _evaluate_agent(
                        agent, method, seed, config, splits["dev"], step_cap
                    )
                    candidate_records.append(
                        {
                            "method": method,
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
            selected = output_root / "checkpoints" / f"{method}_seed{seed}_selected.pt"
            agent.save(str(selected))
            for split in ("regression", "frozen_check"):
                rows, ledger = _evaluate_agent(
                    agent, method, seed, config, splits[split], step_cap
                )
                evaluation_rows.extend(rows)
                action_ledger.extend(ledger)
            checkpoint_selection.extend(candidate_records)
            training_summary.append(
                {
                    "method": method,
                    "seed": seed,
                    "episodes": episodes,
                    "actual_steps": seed_steps,
                    "updates": len(update_records),
                    "elapsed_seconds": time.monotonic() - seed_started,
                    "selected_episode": best["episode"],
                    "selected_checkpoint": str(selected.relative_to(output_root)),
                    "selected_checkpoint_sha256": _sha256(selected),
                    "last_update": update_records[-1] if update_records else None,
                }
            )
            print(
                json.dumps(
                    {
                        "completed": method,
                        "seed": seed,
                        "actual_steps": seed_steps,
                        "selected_episode": best["episode"],
                    }
                ),
                flush=True,
            )

    if total_steps > int(training["budget"]["total_training_step_cap"]):
        raise RuntimeError("observed training steps exceeded frozen cap")
    for split in ("regression", "frozen_check"):
        rows, ledger = _evaluate_rule(config, splits[split], step_cap)
        evaluation_rows.extend(rows)
        action_ledger.extend(ledger)

    aggregate = {
        split: _aggregate([row for row in evaluation_rows if row["split"] == split])
        for split in ("regression", "frozen_check")
    }
    strata = _stratified(evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_csv(output_root / "action_ledger.csv", action_ledger)
    _write_json(output_root / "aggregate.json", aggregate)
    _write_csv(output_root / "stratified_results.csv", strata)
    _write_json(output_root / "training_curves.json", training_curves)
    _write_csv(output_root / "training_curves.csv", training_curves)
    _write_curve_svg(output_root / "training_completion_curve.svg", training_curves)
    _write_json(output_root / "training_summary.json", training_summary)
    _write_json(output_root / "checkpoint_selection.json", checkpoint_selection)
    _write_json(
        output_root / "checkpoint_compatibility.json",
        {
            "historical_checkpoint_state_dict_structurally_loadable": True,
            "historical_checkpoint_semantically_comparable_under_new_profile": False,
            "reason": [
                "encoder input semantics changed under an explicit profile",
                "hierarchical deterministic selection changed from env-marginal argmax to independent-head argmax",
                "PPO actor likelihood changed from canonical heads to executed environment action",
            ],
            "required_action": "retrain_all_learned_methods_under_repaired_profile",
        },
    )
    elapsed_seconds = time.monotonic() - started_at
    run_manifest = {
        "schema_version": "calibrated_continuous_workflow_interface_repair_run_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {"path": args.config, "sha256": _sha256(config_path)},
        "base_config": (
            {
                "path": str(base_config_path.relative_to(ROOT_DIR)),
                "sha256": _sha256(base_config_path),
            }
            if base_config_path is not None
            else None
        ),
        "workload_manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
        "interface_profile": config["interface_profile"],
        "mobility_progression": config["mobility_progression"],
        "methods": methods,
        "rules": ["two_step_cost_rule"],
        "seeds": seeds,
        "episodes_per_method_seed": episodes,
        "episode_max_steps": step_cap,
        "training_step_ceiling": ceiling,
        "actual_training_steps": total_steps,
        "actual_updates": total_updates,
        "elapsed_seconds": elapsed_seconds,
        "wall_clock_cap_seconds": wall_clock_cap,
        "checkpoint_selection": training["checkpoint_selection"],
        "regression_outer_n": len(splits["regression"]),
        "frozen_check_outer_n": len(splits["frozen_check"]),
        "regression_previously_exposed": True,
        "frozen_check_independence": "development_validation_only_unused_dev_plan_source_group",
        "claim_boundary": config["claim_boundary"],
        "real_model_generate_calls": 0,
        "downloads": 0,
        "old_holdout_calls": 0,
        "reward_changed": False,
        "sa_ablation": None,
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "methods": methods + ["two_step_cost_rule"],
            "evaluation_rows": len(evaluation_rows),
            "action_ledger_rows": len(action_ledger),
            "actual_training_steps": total_steps,
            "actual_updates": total_updates,
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
    print(json.dumps({"output_root": str(output_root), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
