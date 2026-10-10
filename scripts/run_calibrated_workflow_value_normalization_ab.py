"""Run the single authorized, bounded development PopArt A/B once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
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
    _evaluate_agent,
)
from scripts.run_calibrated_workflow_service_reward_alignment import (  # noqa: E402
    _resolved_arm_config,
)
from src.encoders.calibrated_workflow_features import bundle_ready, rsu_by_id  # noqa: E402
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)
from src.trainers.ppo_buffer import PPORolloutBuffer  # noqa: E402


DEFAULT_AUTHORIZATION = (
    "configs/experiment/calibrated_workflow_value_normalization_ab_authorized_v1.json"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _python_identity() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().tolist()
    return payload


def _integrity(output_root: Path) -> None:
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_ab_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(output_root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "checkpoint": "checkpoints" in path.relative_to(output_root).parts,
                }
                for path in files
            ],
        },
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
        -float(
            np.mean(
                [
                    float(int(row["max_consecutive_no_progress_steps"]) >= 2)
                    for row in rows
                ]
            )
        ),
        -elapsed,
        -mean("total_transfer_mb"),
        -mean("recompute_seconds"),
        -mean("invalid_prepare_attempts"),
    )


def _validate_intervals(instances: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "source_segment_id",
        "time_index_start",
        "time_index_end",
        "frame_offset",
        "window_length",
    }
    missing: list[str] = []
    overlap_pairs: list[list[str]] = []
    for row in instances:
        interval = row.get("source_interval", {})
        absent = sorted(required - set(interval))
        if absent:
            missing.append(f"{row.get('design_id')}:{','.join(absent)}")
        expected_prefix = f"window_{interval.get('source_segment_id', '')}_off{interval.get('frame_offset', '')}_len{interval.get('window_length', '')}_"
        if not str(row.get("window_id", "")).startswith(expected_prefix):
            missing.append(f"{row.get('design_id')}:window_id_identity")
    for index, left in enumerate(instances):
        a = left["source_interval"]
        for right in instances[index + 1 :]:
            b = right["source_interval"]
            if a["source_segment_id"] != b["source_segment_id"]:
                continue
            overlap = not (
                int(a["time_index_end"]) < int(b["time_index_start"])
                or int(b["time_index_end"]) < int(a["time_index_start"])
            )
            if overlap:
                overlap_pairs.append([str(left["design_id"]), str(right["design_id"])])
    return {
        "required_identity_missing": missing,
        "raw_interval_overlap_pairs": overlap_pairs,
        "all_identity_fields_present": not missing,
        "all_intervals_pairwise_disjoint": not overlap_pairs,
    }


def _new_episode_state(
    *,
    config: dict[str, Any],
    train_instances: list[dict[str, Any]],
    order: list[int],
    rng: random.Random,
    episode_index: int,
) -> dict[str, Any]:
    if episode_index % len(order) == 0:
        rng.shuffle(order)
    instance = deepcopy(train_instances[order[episode_index % len(order)]])
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    return {
        "env": env,
        "observation": observation,
        "info": info,
        "episode_index": episode_index + 1,
        "segment_index": 0,
    }


def _collect_exact_update_batch(
    *,
    agent: Any,
    config: dict[str, Any],
    train_instances: list[dict[str, Any]],
    order: list[int],
    rng: random.Random,
    state: dict[str, Any] | None,
    episodes_started: int,
    transition_count: int,
    step_cap: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    completed_episode_rows: list[dict[str, Any]] = []
    buffer = PPORolloutBuffer()
    while len(rows) < transition_count:
        if state is None:
            state = _new_episode_state(
                config=config,
                train_instances=train_instances,
                order=order,
                rng=rng,
                episode_index=episodes_started,
            )
            episodes_started += 1
        env = state["env"]
        observation = state["observation"]
        info = state["info"]
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
        state["observation"] = next_observation
        state["info"] = next_info
        boundary = len(rows) + len(buffer) == transition_count
        outer_cap = bool(
            not terminated and not truncated and int(env.step_index) >= step_cap
        )
        if terminated or truncated or outer_cap or boundary:
            last_value = (
                0.0
                if terminated
                else float(agent.evaluate_value(next_observation, next_info))
            )
            buffer.finalize(
                last_value=last_value,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            segment = buffer.to_training_rows()
            state["segment_index"] += 1
            for row in segment:
                row["training_episode_index"] = state["episode_index"]
                row["rollout_segment_index"] = state["segment_index"]
            rows.extend(segment)
            buffer = PPORolloutBuffer()
        if terminated or truncated or outer_cap:
            summary = env.summary()
            completed_episode_rows.append(
                {
                    "training_episode_index": state["episode_index"],
                    "design_id": summary["design_id"],
                    "steps": summary["steps"],
                    "workflow_completed": summary["workflow_completed"],
                    "on_time_workflow_completed": summary[
                        "on_time_workflow_completed"
                    ],
                    "deadline_missed": summary["deadline_missed"],
                    "service_failures": summary["service_failures"],
                    "completed_nodes": summary["completed_nodes"],
                    "node_count": summary["node_count"],
                    "reward": summary["reward"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated or outer_cap),
                    "runner_outer_cap": outer_cap,
                }
            )
            state = None
    if len(rows) != transition_count:
        raise RuntimeError("exact transition batch invariant failed")
    return rows, state, episodes_started, completed_episode_rows


def _training_signal_row(
    *,
    arm: str,
    method: str,
    seed: int,
    update_index: int,
    global_step: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    action_info = dict(row.get("action_info", {}))
    projection = dict(action_info.get("action_projection", {}))
    decision_info = dict(row.get("decision_info", {}))
    semantic = dict(decision_info.get("semantic_state", {}))
    vehicles = list(semantic.get("vehicles", []) or [])
    primary_vehicle_id = semantic.get("primary_vehicle_id")
    primary_vehicle = next(
        (
            vehicle
            for vehicle in vehicles
            if primary_vehicle_id is not None
            and str(vehicle.get("vehicle_id", "")) == str(primary_vehicle_id)
        ),
        vehicles[0] if vehicles else {},
    )
    current_rsu = rsu_by_id(semantic, primary_vehicle.get("associated_rsu_id"))
    current_ready = bundle_ready(
        semantic,
        current_rsu,
        semantic.get("current_workflow_node"),
    )
    return {
        "arm": arm,
        "method": method,
        "seed": seed,
        "update_index": update_index,
        "global_cell_step": global_step,
        "training_episode_index": row["training_episode_index"],
        "rollout_segment_index": row["rollout_segment_index"],
        "action": int(row["action"]),
        "reward": float(row["reward"]),
        "terminated": bool(row["terminated"]),
        "truncated": bool(row["truncated"]),
        "value_prediction_denormalized": float(row["value"]),
        "value_target_denormalized": float(row["return"]),
        "advantage_raw": float(row["advantage"]),
        "old_log_prob": float(row["log_prob"]),
        "old_env_action_log_prob": float(
            action_info.get(
                "env_action_log_prob",
                projection.get("masked_env_action_log_prob", row["log_prob"]),
            )
        ),
        "env_action_probs": json.dumps(
            action_info.get("env_action_probs", []), sort_keys=True
        ),
        "current_bundle_ready": bool(current_ready),
    }


def _aggregate_evaluation(
    rows: list[dict[str, Any]], *, by_seed: bool
) -> list[dict[str, Any]]:
    metrics = (
        "on_time_workflow_completion_rate",
        "workflow_completion_rate",
        "unfinished_after_deadline_rate",
        "service_failure_rate",
        "failed_service_attempt_seconds_proxy",
        "total_transfer_mb",
        "recompute_seconds",
        "invalid_prepare_attempts",
        "max_consecutive_no_progress_steps",
        "current_missing_action4_rate",
        "completed_sample_elapsed_coverage",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_key = str(row["seed"]) if by_seed else "pooled"
        grouped[
            (str(row["arm"]), str(row["method"]), seed_key, str(row["split"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (arm, method, seed, split), group in sorted(grouped.items()):
        completed_elapsed = [
            float(row["completed_sample_elapsed_seconds"])
            for row in group
            if row.get("completed_sample_elapsed_seconds", "") != ""
        ]
        output.append(
            {
                "arm": arm,
                "method": method,
                "seed": seed,
                "split": split,
                "episode_count": len(group),
                **{
                    metric: float(np.mean([float(row[metric]) for row in group]))
                    for metric in metrics
                },
                "consecutive_no_progress_episode_rate": float(
                    np.mean(
                        [
                            int(row["max_consecutive_no_progress_steps"]) >= 2
                            for row in group
                        ]
                    )
                ),
                "completed_elapsed_seconds": (
                    float(np.mean(completed_elapsed)) if completed_elapsed else None
                ),
                "completed_elapsed_sample_count": len(completed_elapsed),
                "service_failure_attempt_rate": float(
                    sum(int(row["service_failures"]) for row in group)
                    / max(
                        sum(
                            sum(int(row[f"action_{action_id}"]) for action_id in range(5))
                            for row in group
                        ),
                        1,
                    )
                ),
                "service_failure_episode_rate": float(
                    np.mean([float(row["service_failure_rate"]) for row in group])
                ),
            }
        )
    return output


def _analyze_gates(
    *,
    evaluation_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    final_updates = [row for row in update_rows if int(row["update_index"]) == 24]
    by_cell = {
        (row["arm"], row["method"], int(row["seed"])): row for row in final_updates
    }
    mechanism_deltas: list[dict[str, Any]] = []
    for method in methods:
        for seed in (7, 17, 29, 43, 61):
            control = by_cell[("control_raw_critic_target", method, seed)]
            candidate = by_cell[("candidate_popart_critic_target", method, seed)]
            control_ratio = float(control["weighted_value_to_policy_grad_ratio"])
            candidate_ratio = float(candidate["weighted_value_to_policy_grad_ratio"])
            mechanism_deltas.append(
                {
                    "method": method,
                    "seed": seed,
                    "value_to_policy_ratio_delta": candidate_ratio - control_ratio,
                    "explained_variance_delta": float(candidate["explained_variance"])
                    - float(control["explained_variance"]),
                }
            )
    ratio_delta_median = float(
        np.median([row["value_to_policy_ratio_delta"] for row in mechanism_deltas])
    )
    ev_delta_median = float(
        np.median([row["explained_variance_delta"] for row in mechanism_deltas])
    )
    mechanism_pass = ratio_delta_median < 0.0 and ev_delta_median > 0.0

    evaluation_splits = {"regression", "frozen_check"}
    pooled = [row for row in evaluation_rows if row["split"] in evaluation_splits]
    method_service: dict[str, Any] = {}
    service_pass = True
    any_service_improvement = False
    for method in methods:
        control = [
            row
            for row in pooled
            if row["arm"] == "control_raw_critic_target" and row["method"] == method
        ]
        candidate = [
            row
            for row in pooled
            if row["arm"] == "candidate_popart_critic_target" and row["method"] == method
        ]
        values = {
            "control_completion": float(
                np.mean([float(row["workflow_completion_rate"]) for row in control])
            ),
            "candidate_completion": float(
                np.mean([float(row["workflow_completion_rate"]) for row in candidate])
            ),
            "control_on_time": float(
                np.mean([float(row["on_time_workflow_completion_rate"]) for row in control])
            ),
            "candidate_on_time": float(
                np.mean([float(row["on_time_workflow_completion_rate"]) for row in candidate])
            ),
            "control_failure_episode_rate": float(
                np.mean([float(row["service_failure_rate"]) for row in control])
            ),
            "candidate_failure_episode_rate": float(
                np.mean([float(row["service_failure_rate"]) for row in candidate])
            ),
            "control_failure_attempt_rate": float(
                sum(int(row["service_failures"]) for row in control)
                / max(
                    sum(
                        sum(int(row[f"action_{action_id}"]) for action_id in range(5))
                        for row in control
                    ),
                    1,
                )
            ),
            "candidate_failure_attempt_rate": float(
                sum(int(row["service_failures"]) for row in candidate)
                / max(
                    sum(
                        sum(int(row[f"action_{action_id}"]) for action_id in range(5))
                        for row in candidate
                    ),
                    1,
                )
            ),
            "control_no_progress": float(
                np.mean(
                    [
                        int(row["max_consecutive_no_progress_steps"]) >= 2
                        for row in control
                    ]
                )
            ),
            "candidate_no_progress": float(
                np.mean(
                    [
                        int(row["max_consecutive_no_progress_steps"]) >= 2
                        for row in candidate
                    ]
                )
            ),
        }
        values["completion_delta"] = (
            values["candidate_completion"] - values["control_completion"]
        )
        values["on_time_delta"] = values["candidate_on_time"] - values["control_on_time"]
        values["failure_episode_rate_delta"] = (
            values["candidate_failure_episode_rate"]
            - values["control_failure_episode_rate"]
        )
        values["failure_attempt_rate_delta"] = (
            values["candidate_failure_attempt_rate"]
            - values["control_failure_attempt_rate"]
        )
        values["no_progress_delta"] = (
            values["candidate_no_progress"] - values["control_no_progress"]
        )
        method_pass = bool(
            values["completion_delta"] >= 0.0 and values["on_time_delta"] >= 0.0
        )
        service_pass = service_pass and method_pass
        any_service_improvement = any_service_improvement or bool(
            values["completion_delta"] > 0.0
            or values["failure_episode_rate_delta"] < 0.0
            or values["failure_attempt_rate_delta"] < 0.0
        )
        values["no_completion_harm_pass"] = method_pass
        method_service[method] = values
    service_pass = service_pass and any_service_improvement

    def _current_missing_action4(arm: str) -> dict[str, float]:
        rows = [
            row
            for row in behavior_rows
            if row["arm"] == arm and row["split"] in evaluation_splits
        ]
        executed_numerator = sum(
            int(row["executed_action"]) == 4 and not bool(row["current_bundle_ready"])
            for row in rows
        )
        raw_argmax_numerator = sum(
            int(row["raw_env_action"]) == 4 and not bool(row["current_bundle_ready"])
            for row in rows
        )
        eligible = [row for row in rows if not bool(row["current_bundle_ready"])]
        action4_probs: list[float] = []
        for row in eligible:
            probabilities = json.loads(str(row.get("env_action_probs", "[]")))
            if len(probabilities) > 4:
                action4_probs.append(float(probabilities[4]))
        denominator = len(eligible)
        return {
            "decision_count": float(denominator),
            "executed_rate": float(executed_numerator) / max(denominator, 1),
            "raw_argmax_rate": float(raw_argmax_numerator) / max(denominator, 1),
            "mean_probability": (
                float(np.mean(action4_probs)) if action4_probs else 0.0
            ),
            "probability_coverage": float(len(action4_probs)) / max(denominator, 1),
        }

    control_action4 = _current_missing_action4("control_raw_critic_target")
    candidate_action4 = _current_missing_action4("candidate_popart_critic_target")
    failure_episode_delta_all = float(
        np.mean(
            [
                values["failure_episode_rate_delta"]
                for values in method_service.values()
            ]
        )
    )
    failure_attempt_delta_all = float(
        np.mean(
            [
                values["failure_attempt_rate_delta"]
                for values in method_service.values()
            ]
        )
    )
    no_progress_delta_all = float(
        np.mean(
            [
                values["no_progress_delta"]
                for values in method_service.values()
            ]
        )
    )
    behavior_pass = bool(
        candidate_action4["mean_probability"] < control_action4["mean_probability"]
        and candidate_action4["raw_argmax_rate"] < control_action4["raw_argmax_rate"]
        and failure_episode_delta_all <= 0.0
        and failure_attempt_delta_all <= 0.0
        and no_progress_delta_all <= 0.0
    )
    all_pass = mechanism_pass and behavior_pass and service_pass
    return {
        "mechanism": {
            "pass": mechanism_pass,
            "paired_cell_deltas": mechanism_deltas,
            "median_value_to_policy_ratio_delta": ratio_delta_median,
            "median_explained_variance_delta": ev_delta_median,
        },
        "behavior": {
            "pass": behavior_pass,
            "control_current_missing_action4": control_action4,
            "candidate_current_missing_action4": candidate_action4,
            "mean_failure_episode_rate_delta": failure_episode_delta_all,
            "mean_failure_attempt_rate_delta": failure_attempt_delta_all,
            "mean_no_progress_delta": no_progress_delta_all,
        },
        "service": {
            "pass": service_pass,
            "any_completion_or_failure_improvement": any_service_improvement,
            "by_method": method_service,
        },
        "overall": {
            "pass": all_pass,
            "verdict": (
                "SUPPORTED_DEVELOPMENT_CANDIDATE"
                if all_pass
                else "FALSIFIED_OR_NOT_PROMOTED"
            ),
            "automatic_next_stage_authorized": False,
        },
        "optimizer_record_count": len(optimizer_rows),
    }


def _run(
    *,
    authorization_path: Path,
    output_root: Path,
    started_at: float,
) -> None:
    authorization = _load_json(authorization_path)
    if output_root.name != str(authorization["output_run_id"]):
        raise RuntimeError("authorization output_run_id does not match output root")
    design_path = ROOT_DIR / authorization["design_config"]
    if _sha256(design_path) != authorization["design_config_sha256"]:
        raise RuntimeError("authorized design config hash mismatch")
    design = _load_json(design_path)
    if not authorization["execution_authorized"] or design["execution_authorized"]:
        raise RuntimeError("authorization/design status mismatch")
    if authorization["launch_count"] != 1 or authorization["automatic_retry"]:
        raise RuntimeError("single-launch identity drift")
    base_path = ROOT_DIR / design["base_config"]
    manifest_path = ROOT_DIR / design["workload_manifest"]
    base, _ = _load_experiment_config(base_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(base_path):
        raise RuntimeError("base config hash mismatch")
    if manifest["resolved_config_sha256"] != _canonical_sha256(base):
        raise RuntimeError("resolved base config hash mismatch")
    interval_identity = _validate_intervals(list(manifest["instances"]))
    if not interval_identity["all_identity_fields_present"]:
        raise RuntimeError("frozen window identity validation failed")
    if not interval_identity["all_intervals_pairwise_disjoint"]:
        raise RuntimeError("frozen source intervals overlap")

    protocol = design["development_ab"]
    methods = list(protocol["learned_methods"])
    seeds = [int(seed) for seed in protocol["seeds"]]
    if methods != ["sa_ghmappo", "mappo", "ppo"] or seeds != [7, 17, 29, 43, 61]:
        raise RuntimeError("method/seed identity drift")
    if (
        int(protocol["environment_steps_per_cell"]) != 1440
        or int(protocol["update_opportunities_per_cell"]) != 24
        or int(protocol["transitions_per_update"]) != 60
        or int(protocol["ppo_epochs_per_update"]) != 4
        or int(protocol["minibatch_size"]) != 32
        or int(protocol["optimizer_steps_per_cell"]) != 192
        or list(protocol["checkpoint_update_candidates"]) != [6, 12, 18, 24]
    ):
        raise RuntimeError("scientific budget identity drift")
    arms = list(design["arms"])
    if [arm["arm"] for arm in arms] != [
        "control_raw_critic_target",
        "candidate_popart_critic_target",
    ]:
        raise RuntimeError("arm identity drift")
    expected_cells = len(arms) * len(methods) * len(seeds)
    if expected_cells != authorization["training_cell_count"]:
        raise RuntimeError("cell budget mismatch")
    expected_steps = expected_cells * int(protocol["environment_steps_per_cell"])
    expected_updates = expected_cells * int(protocol["update_opportunities_per_cell"])
    expected_optimizer = expected_cells * int(protocol["optimizer_steps_per_cell"])
    if (
        expected_steps != authorization["total_environment_steps"]
        or expected_updates != authorization["total_update_opportunities"]
        or expected_optimizer != authorization["total_optimizer_steps"]
    ):
        raise RuntimeError("aggregate budget mismatch")
    if protocol["hyperparameter_search_trials_per_arm"] != 0:
        raise RuntimeError("hyperparameter search is forbidden")

    config = _resolved_arm_config(base, {
        "service_aligned_reward": _load_json(
            ROOT_DIR / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json"
        )["service_aligned_reward"]
    }, design["reward_profile"])
    splits = {
        split: [row for row in manifest["instances"] if row["split"] == split]
        for split in ("train", "dev", "regression", "frozen_check")
    }
    if {key: len(rows) for key, rows in splits.items()} != {
        "train": 12,
        "dev": 4,
        "regression": 12,
        "frozen_check": 8,
    }:
        raise RuntimeError("split count drift")

    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir()
    training_signal_rows: list[dict[str, Any]] = []
    training_episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    wall_cap = float(authorization["scientific_wall_clock_cap_seconds"])

    for arm in arms:
        arm_name = str(arm["arm"])
        popart_enabled = arm["critic_target_normalization"] == "popart_running_mean_std"
        for method in methods:
            for seed in seeds:
                if time.monotonic() - started_at > wall_cap:
                    raise TimeoutError("scientific wall-clock cap exceeded")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                agent = _build_agent(
                    method,
                    seed,
                    config,
                    agent_overrides={
                        "value_normalization_enabled": popart_enabled,
                        "popart_min_std": 1.0,
                        "popart_max_abs_target": 1.0e20,
                    },
                )
                order = list(range(len(splits["train"])))
                rng = random.Random(seed)
                active_state: dict[str, Any] | None = None
                episodes_started = 0
                cell_step = 0
                candidates: list[dict[str, Any]] = []
                cell_updates: list[dict[str, Any]] = []
                for update_index in range(1, 25):
                    batch, active_state, episodes_started, episodes = (
                        _collect_exact_update_batch(
                            agent=agent,
                            config=config,
                            train_instances=splits["train"],
                            order=order,
                            rng=rng,
                            state=active_state,
                            episodes_started=episodes_started,
                            transition_count=60,
                            step_cap=int(protocol["episode_max_steps"]),
                            gamma=float(config["training"]["gamma"]),
                            gae_lambda=float(config["training"]["gae_lambda"]),
                        )
                    )
                    for row in batch:
                        cell_step += 1
                        training_signal_rows.append(
                            _training_signal_row(
                                arm=arm_name,
                                method=method,
                                seed=seed,
                                update_index=update_index,
                                global_step=cell_step,
                                row=row,
                            )
                        )
                    for row in episodes:
                        training_episode_rows.append(
                            {"arm": arm_name, "method": method, "seed": seed, **row}
                        )
                    update = _json_safe(agent.learn(batch))
                    if update.get("policy_update_skipped"):
                        raise RuntimeError("unexpected skipped policy update")
                    if int(update.get("optimizer_step_count", -1)) != 8:
                        raise RuntimeError("optimizer-step budget drift")
                    policy_grad = float(
                        np.mean(
                            [
                                float(row["policy_grad_norm"])
                                for row in update["optimizer_step_records"]
                            ]
                        )
                    )
                    value_grad = float(
                        np.mean(
                            [
                                float(row["weighted_value_grad_norm"])
                                for row in update["optimizer_step_records"]
                            ]
                        )
                    )
                    update_row = {
                        "arm": arm_name,
                        "method": method,
                        "seed": seed,
                        "update_index": update_index,
                        **{
                            key: value
                            for key, value in update.items()
                            if key != "optimizer_step_records"
                        },
                        "weighted_value_to_policy_grad_ratio": value_grad
                        / max(policy_grad, 1e-12),
                    }
                    update_rows.append(update_row)
                    cell_updates.append(update_row)
                    for optimizer_row in update["optimizer_step_records"]:
                        optimizer_rows.append(
                            {
                                "arm": arm_name,
                                "method": method,
                                "seed": seed,
                                "update_index": update_index,
                                **optimizer_row,
                            }
                        )
                    if update_index in (6, 12, 18, 24):
                        checkpoint_path = (
                            checkpoints
                            / f"{arm_name}_{method}_seed{seed}_update{update_index}.pt"
                        )
                        agent.save(str(checkpoint_path))
                        dev_rows, _ = _evaluate_agent(
                            agent,
                            method,
                            seed,
                            config,
                            splits["dev"],
                            int(protocol["episode_max_steps"]),
                        )
                        score = list(_selection_score(dev_rows))
                        candidate = {
                            "arm": arm_name,
                            "method": method,
                            "seed": seed,
                            "update_index": update_index,
                            "score": score,
                            "score_fields": list(protocol["selection_order"]),
                            "checkpoint": str(checkpoint_path.relative_to(output_root)),
                            "checkpoint_sha256": _sha256(checkpoint_path),
                            "reward_used": False,
                            "evaluation_or_holdout_used": False,
                            "dev_rows": dev_rows,
                        }
                        candidates.append(candidate)
                        checkpoint_rows.append(candidate)
                if cell_step != 1440 or len(cell_updates) != 24:
                    raise RuntimeError("per-cell interaction/update budget drift")
                best = max(
                    candidates,
                    key=lambda row: (tuple(row["score"]), -int(row["update_index"])),
                )
                agent.load(str(output_root / best["checkpoint"]))
                selected_path = (
                    checkpoints / f"{arm_name}_{method}_seed{seed}_selected.pt"
                )
                agent.save(str(selected_path))
                for split in authorization["falsification_evaluation_splits"]:
                    rows, ledger = _evaluate_agent(
                        agent,
                        method,
                        seed,
                        config,
                        splits[split],
                        int(protocol["episode_max_steps"]),
                    )
                    evaluation_rows.extend(
                        {"arm": arm_name, **row} for row in rows
                    )
                    behavior_rows.extend(
                        {"arm": arm_name, **row} for row in ledger
                    )
                training_summary.append(
                    {
                        "arm": arm_name,
                        "method": method,
                        "seed": seed,
                        "environment_steps": cell_step,
                        "updates": len(cell_updates),
                        "optimizer_steps": sum(
                            int(row["optimizer_step_count"]) for row in cell_updates
                        ),
                        "episodes_started": episodes_started,
                        "episodes_completed_or_truncated": sum(
                            row["arm"] == arm_name
                            and row["method"] == method
                            and int(row["seed"]) == seed
                            for row in training_episode_rows
                        ),
                        "active_partial_episode_at_budget": active_state is not None,
                        "selected_update": int(best["update_index"]),
                        "selected_checkpoint": str(selected_path.relative_to(output_root)),
                        "selected_checkpoint_sha256": _sha256(selected_path),
                        "value_normalization_enabled": popart_enabled,
                    }
                )
                print(
                    json.dumps(
                        {
                            "arm": arm_name,
                            "method": method,
                            "seed": seed,
                            "steps": cell_step,
                            "updates": len(cell_updates),
                            "selected_update": best["update_index"],
                        }
                    ),
                    flush=True,
                )

    if len(training_signal_rows) != expected_steps:
        raise RuntimeError("aggregate environment-step count mismatch")
    if len(update_rows) != expected_updates or len(optimizer_rows) != expected_optimizer:
        raise RuntimeError("aggregate update/optimizer-step count mismatch")
    if not all(np.isfinite(float(row["reward"])) for row in training_signal_rows):
        raise RuntimeError("non-finite training reward")
    if not all(np.isfinite(float(row["value_loss"])) for row in update_rows):
        raise RuntimeError("non-finite value loss")

    aggregates = _aggregate_evaluation(evaluation_rows, by_seed=False)
    per_seed_aggregates = _aggregate_evaluation(evaluation_rows, by_seed=True)
    gates = _analyze_gates(
        evaluation_rows=evaluation_rows,
        behavior_rows=behavior_rows,
        update_rows=update_rows,
        optimizer_rows=optimizer_rows,
        methods=methods,
    )
    elapsed = time.monotonic() - started_at
    run_manifest = {
        "schema_version": "calibrated_workflow_value_normalization_ab_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_identity": _python_identity(),
        "authorization_config": {
            "path": str(authorization_path.relative_to(ROOT_DIR)),
            "sha256": _sha256(authorization_path),
        },
        "design_config": {
            "path": str(design_path.relative_to(ROOT_DIR)),
            "sha256": _sha256(design_path),
        },
        "base_config": {"path": design["base_config"], "sha256": _sha256(base_path)},
        "workload_manifest": {
            "path": design["workload_manifest"],
            "sha256": _sha256(manifest_path),
        },
        "window_identity_validation": interval_identity,
        "methods": methods,
        "seeds": seeds,
        "arms": arms,
        "budget": protocol,
        "actual_environment_steps": len(training_signal_rows),
        "actual_updates": len(update_rows),
        "actual_optimizer_steps": len(optimizer_rows),
        "elapsed_seconds": elapsed,
        "automatic_retry": False,
        "launch_count": 1,
        "formal_or_holdout_reads": 0,
        "real_model_generate_calls": 0,
        "downloads": 0,
        "hyperparameter_search_trials": 0,
        "claim_boundary": authorization["claim_boundary"],
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(output_root / "training_summary.json", training_summary)
    _write_csv(output_root / "training_signal_ledger.csv", training_signal_rows)
    _write_csv(output_root / "training_episode_rows.csv", training_episode_rows)
    _write_json(output_root / "update_records.json", update_rows)
    _write_csv(output_root / "optimizer_step_records.csv", optimizer_rows)
    _write_json(output_root / "checkpoint_selection.json", checkpoint_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_csv(output_root / "behavior_ledger.csv", behavior_rows)
    _write_csv(output_root / "evaluation_aggregate.csv", aggregates)
    _write_csv(output_root / "evaluation_per_seed_aggregate.csv", per_seed_aggregates)
    _write_json(output_root / "falsification_gates.json", gates)
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "scientific_execution_complete": True,
            "actual_environment_steps": len(training_signal_rows),
            "actual_updates": len(update_rows),
            "actual_optimizer_steps": len(optimizer_rows),
            "checkpoint_files_committed_or_uploaded": False,
            "evaluation_rows": len(evaluation_rows),
            "behavior_rows": len(behavior_rows),
            "falsification_verdict": gates["overall"]["verdict"],
            "elapsed_seconds": elapsed,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-config", default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    authorization_path = (ROOT_DIR / args.authorization_config).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    authorization = _load_json(authorization_path)
    if output_root.name != str(authorization["output_run_id"]):
        raise ValueError("authorization output_run_id does not match output root")
    output_root.mkdir(parents=True)
    started_at = time.monotonic()
    _write_json(
        output_root / "runner_entered.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_ab_runner_entry_v1",
            "status": "entered",
            "entered_at": datetime.now(timezone.utc).isoformat(),
            "run_id": authorization["output_run_id"],
            "pid": os.getpid(),
            "interpreter": sys.executable,
            "git_commit": _git_commit(),
        },
    )
    _write_json(
        output_root / "run_status.json",
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "launch_count": 1,
            "automatic_retry": False,
        },
    )
    try:
        _run(
            authorization_path=authorization_path,
            output_root=output_root,
            started_at=started_at,
        )
    except Exception as exc:
        _write_json(
            output_root / "failure_receipt.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started_at,
                "automatic_retry": False,
            },
        )
        _write_json(
            output_root / "run_status.json",
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "failure_receipt": "failure_receipt.json",
            },
        )
        _integrity(output_root)
        raise
    print(json.dumps({"status": "complete", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
