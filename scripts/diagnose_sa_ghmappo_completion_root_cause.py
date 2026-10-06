"""Bounded, read-only root-cause diagnosis for the calibrated SA-GHMAPPO pilot.

The script never trains or modifies a checkpoint.  It consumes the frozen v2
config/manifest plus an explicit pilot artifact root, verifies their hashes,
runs at most 18 diagnostic episodes, and writes a new isolated analysis bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.agents.registry import build_agent
from src.agents.sa_ghmappo_core import 聚合层级动作
from src.encoders.fusion_encoder import FlatSemanticEncoder
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    ImmediateCostRule,
    TwoStepCostRule,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs/experiment/calibrated_continuous_workflow_pilot_v2.json"
DEFAULT_MANIFEST = ROOT_DIR / "configs/experiment/calibrated_continuous_workflow_pilot_v2_manifest.json"
DIAGNOSTIC_SEEDS = (7, 17, 29)
MODES = ("deterministic_env_argmax", "fixed_env_sample")
MAX_EPISODES = 18
MAX_EPISODE_STEPS = 24
DAG_ENUMERATION_DEPTH = 3
DAG_TOTAL_BRANCH_CAP = 2048


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_agent(seed: int, training: dict[str, Any], checkpoint: Path, *, deterministic: bool) -> Any:
    agent = build_agent(
        "sa_ghmappo",
        random_seed=int(seed),
        learning_rate=float(training["learning_rate"]),
        clip_ratio=float(training["clip_ratio"]),
        entropy_coef=float(training["entropy_coef"]),
        value_coef=float(training["value_coef"]),
        batch_size=32,
        train_epochs=4,
        deterministic_action=bool(deterministic),
    )
    agent.load(str(checkpoint))
    agent._deterministic_action = bool(deterministic)
    return agent


def _head_probability_pushforward(
    head_probabilities: dict[str, list[float]],
    action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Enumerate the actual 3x2x2 head space and aggregate by env action."""
    env_probabilities = [0.0] * 5
    contributions: dict[str, list[dict[str, Any]]] = {str(action): [] for action in range(5)}
    for slow, fast, event in itertools.product(range(3), range(2), range(2)):
        head_actions = {"slow": slow, "fast": fast, "event": event}
        env_action, reason = 聚合层级动作(
            head_actions=head_actions,
            use_hierarchy=True,
            event_head_enabled=True,
            adapter_prefetch_enabled=True,
        )
        probability = (
            float(head_probabilities["slow"][slow])
            * float(head_probabilities["fast"][fast])
            * float(head_probabilities["event"][event])
        )
        env_probabilities[env_action] += probability
        contributions[str(env_action)].append(
            {"head_actions": head_actions, "probability": probability, "reason": reason}
        )
    masked = list(env_probabilities)
    if action_mask and any(action_mask):
        masked = [probability if bool(action_mask[index]) else 0.0 for index, probability in enumerate(masked)]
        denominator = sum(masked)
        if denominator > 0.0:
            masked = [probability / denominator for probability in masked]
    return {
        "unmasked_env_probabilities": env_probabilities,
        "masked_env_probabilities": masked,
        "contributions": contributions,
    }


def _policy_snapshot(agent: Any, info: dict[str, Any]) -> dict[str, Any]:
    semantic_state = info["semantic_state"]
    action_mask = list(info["action_mask"])
    with torch.no_grad():
        output = agent._forward_policy(
            semantic_state,
            run_metadata={"policy_evaluation_mode": "raw_policy"},
        )
        logits = {
            head: [float(value) for value in output[f"{head}_logits"].tolist()]
            for head in ("slow", "fast", "event")
        }
        probabilities = {
            head: [float(value) for value in torch.softmax(output[f"{head}_logits"], dim=-1).tolist()]
            for head in ("slow", "fast", "event")
        }
        head_argmax = {
            head: int(torch.argmax(output[f"{head}_logits"]).item())
            for head in ("slow", "fast", "event")
        }
        head_argmax_action, head_argmax_reason = 聚合层级动作(
            head_actions=head_argmax,
            use_hierarchy=True,
            event_head_enabled=True,
            adapter_prefetch_enabled=True,
        )
        env_logits = agent._masked_flat_logits(
            agent._hierarchical_env_action_scores(output),
            action_mask,
        )
        env_probabilities = [float(value) for value in torch.softmax(env_logits, dim=-1).tolist()]
    pushforward = _head_probability_pushforward(probabilities, action_mask)
    return {
        "raw_head_logits": logits,
        "raw_head_probabilities": probabilities,
        "independent_head_argmax": head_argmax,
        "independent_head_argmax_action": int(head_argmax_action),
        "independent_head_argmax_reason": head_argmax_reason,
        "pushforward": pushforward,
        "declared_env_action_probabilities": env_probabilities,
        "declared_env_action_argmax": int(np.argmax(env_probabilities)),
        "max_pushforward_error": max(
            abs(left - right)
            for left, right in zip(env_probabilities, pushforward["masked_env_probabilities"])
        ),
    }


def _select_fixed_instances(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [row for row in manifest["instances"] if row["split"] in {"train", "dev"}]
    described: list[dict[str, Any]] = []
    for row in candidates:
        env = CalibratedContinuousWorkflowEnv(config, row)
        env.reset()
        node = env._current_node()
        current_rsu = env._current_rsu_id()
        described.append(
            {
                "instance": row,
                "current_ready": env._bundle_ready(current_rsu, str(node["required_adapter"])),
                "contact_budget_seconds": env._contact_budget_seconds(),
                "target_rsu_id": env._predicted_handoff_target(),
            }
        )
    ready = next(item for item in described if item["current_ready"])
    missing = next(
        item
        for item in described
        if not item["current_ready"] and item["instance"]["design_id"] != ready["instance"]["design_id"]
    )
    excluded = {ready["instance"]["design_id"], missing["instance"]["design_id"]}
    minimum_contact = min(
        item["contact_budget_seconds"]
        for item in described
        if item["target_rsu_id"] and item["instance"]["design_id"] not in excluded
    )
    imminent = next(
        item
        for item in described
        if item["target_rsu_id"]
        and item["instance"]["design_id"] not in excluded
        and item["contact_budget_seconds"] == minimum_contact
    )
    return [
        {"category": "current_model_ready", **ready},
        {"category": "current_model_missing", **missing},
        {"category": "imminent_handoff_min_contact", **imminent},
    ]


def _finalize_advantages(
    rows: list[dict[str, Any]],
    *,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> None:
    gae = 0.0
    for index in reversed(range(len(rows))):
        row = rows[index]
        next_value = last_value if index == len(rows) - 1 else float(rows[index + 1]["value"])
        next_non_terminal = 0.0 if bool(row["terminated"]) else 1.0
        delta = float(row["reward"]) + gamma * next_value * next_non_terminal - float(row["value"])
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        row["advantage"] = float(gae)
        row["return"] = float(gae + float(row["value"]))


def _run_episode(
    *,
    config: dict[str, Any],
    instance: dict[str, Any],
    checkpoint: Path,
    seed: int,
    category: str,
    mode: str,
    episode_ordinal: int,
) -> dict[str, Any]:
    deterministic = mode == "deterministic_env_argmax"
    sampling_seed = int(seed * 1000 + episode_ordinal)
    random.seed(sampling_seed)
    np.random.seed(sampling_seed)
    torch.manual_seed(sampling_seed)
    agent = _build_agent(seed, config["training"], checkpoint, deterministic=deterministic)
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    rows: list[dict[str, Any]] = []
    terminated = truncated = False
    step_cap = min(MAX_EPISODE_STEPS, int(instance["max_steps"]))
    while not terminated and not truncated and len(rows) < step_cap:
        snapshot = _policy_snapshot(agent, info)
        semantic = info["semantic_state"]
        node = semantic["current_workflow_node"]
        current_rsu = env._current_rsu_id()
        target_rsu = env._predicted_handoff_target()
        current_bundle_ready_before = env._bundle_ready(current_rsu, str(node["required_adapter"]))
        target_bundle_ready_before = (
            env._bundle_ready(target_rsu, str(node["required_adapter"])) if target_rsu else None
        )
        before_node_index = env.node_index
        before_step_index = env.step_index
        before_clock = env.clock_seconds
        before_position = float(semantic["vehicles"][0]["position_x"])
        before_sequence = list(semantic["predictions"]["next_rsu_sequence"]["veh_pilot"])
        value = float(agent.evaluate_value(observation, info))
        action, action_info = agent.act(observation, info)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        transition = next_info["transition"]
        next_semantic = next_info["semantic_state"]
        declared_probability = float(snapshot["declared_env_action_probabilities"][int(action)])
        buffer_probability = math.exp(float(action_info["log_prob"]))
        env_log_probability = math.exp(float(action_info["env_action_log_prob"]))
        rows.append(
            {
                "step": before_step_index,
                "node_id": str(node["node_id"]),
                "completed_node_ids_before": list(semantic["workflow"]["completed_node_ids"]),
                "current_rsu_id": current_rsu,
                "target_rsu_id": target_rsu,
                "required_base_model": str(node["required_base_model"]),
                "required_adapter": str(node["required_adapter"]),
                "current_bundle_ready": current_bundle_ready_before,
                "target_bundle_ready_before": target_bundle_ready_before,
                "action_mask": list(info["action_mask"]),
                "policy_snapshot": snapshot,
                "reported_raw_head_actions": dict(action_info["raw_head_actions"]),
                "reported_projected_head_actions": dict(action_info["projected_head_actions"]),
                "reported_final_head_actions": dict(action_info["head_actions"]),
                "raw_env_action": int(action_info["raw_env_action"]),
                "projected_env_action": int(action_info["projected_env_action"]),
                "executed_env_action": int(action),
                "aggregation_reason": str(action_info["aggregation_reason"]),
                "projection_applied": bool(action_info["action_projection_applied"]),
                "guard_action_delta": bool(action_info["guard_action_delta"]),
                "buffer_log_prob": float(action_info["log_prob"]),
                "env_action_log_prob": float(action_info["env_action_log_prob"]),
                "declared_executed_action_probability": declared_probability,
                "exp_buffer_log_prob": buffer_probability,
                "exp_env_action_log_prob": env_log_probability,
                "buffer_vs_executed_probability_error": abs(buffer_probability - declared_probability),
                "env_log_prob_vs_executed_probability_error": abs(env_log_probability - declared_probability),
                "ppo_update_identity": {
                    "stored_action": int(action),
                    "stored_head_actions": dict(action_info["head_actions"]),
                    "stored_log_prob_source": "weighted_canonical_head_log_prob",
                    "stored_env_action_log_prob_available": True,
                    "env_action_ppo_enabled": bool(agent._env_action_ppo_enabled),
                    "env_action_ppo_coef": float(agent._env_action_ppo_coef),
                },
                "reward": float(reward),
                "value": value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "transition": _json_safe(transition),
                "outcomes": {
                    "node_completed": bool(transition["service_completed"]),
                    "state_ready": bool(transition["state_ready"]),
                    "state_restored_seconds": float(transition["state_restore_seconds"]),
                    "service_failure": not bool(transition["service_completed"]),
                    "migration_success": bool(transition["migration_success"]),
                },
                "exogenous_progress": {
                    "step_index_before": before_step_index,
                    "step_index_after": env.step_index,
                    "node_index_before": before_node_index,
                    "node_index_after": env.node_index,
                    "clock_seconds_before": before_clock,
                    "clock_seconds_after": env.clock_seconds,
                    "contact_budget_seconds_before": float(semantic["calibrated_context"]["contact_budget_seconds"]),
                    "contact_budget_seconds_after": float(next_semantic["calibrated_context"]["contact_budget_seconds"]),
                    "vehicle_position_before": before_position,
                    "vehicle_position_after": float(next_semantic["vehicles"][0]["position_x"]),
                    "predicted_sequence_before": before_sequence,
                    "predicted_sequence_after": list(next_semantic["predictions"]["next_rsu_sequence"]["veh_pilot"]),
                    "mobility_advanced": bool(
                        float(next_semantic["vehicles"][0]["position_x"]) != before_position
                        or list(next_semantic["predictions"]["next_rsu_sequence"]["veh_pilot"]) != before_sequence
                    ),
                },
            }
        )
        observation, info = next_observation, next_info
    if not terminated and not truncated and len(rows) >= step_cap:
        truncated = True
    last_value = 0.0 if terminated else float(agent.evaluate_value(observation, info))
    _finalize_advantages(
        rows,
        last_value=last_value,
        gamma=float(config["training"]["gamma"]),
        gae_lambda=float(config["training"]["gae_lambda"]),
    )
    return {
        "category": category,
        "design_id": instance["design_id"],
        "split": instance["split"],
        "seed": int(seed),
        "sampling_seed": sampling_seed,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "step_cap": step_cap,
        "steps": rows,
        "last_value": last_value,
        "summary": env.summary(),
    }


def _negative_advantage_update_witness(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    witness = None
    for episode in episodes:
        for row in episode["steps"]:
            if row["outcomes"]["service_failure"] and float(row["advantage"]) < 0.0:
                witness = (episode, row)
                break
        if witness:
            break
    if witness is None:
        return {"available": False, "reason": "no service-failure step with negative diagnostic GAE"}
    episode, row = witness
    target_action = int(row["executed_env_action"])
    head_targets = dict(row["reported_final_head_actions"])
    original_logits = {
        head: torch.tensor(values, dtype=torch.float64, requires_grad=True)
        for head, values in row["policy_snapshot"]["raw_head_logits"].items()
    }

    def env_probs(logits_by_head: dict[str, torch.Tensor]) -> torch.Tensor:
        event = torch.log_softmax(logits_by_head["event"], dim=-1)
        slow = torch.log_softmax(logits_by_head["slow"], dim=-1)
        fast = torch.log_softmax(logits_by_head["fast"], dim=-1)
        scores = torch.stack(
            [
                event[0] + slow[1],
                event[0] + slow[2],
                event[0] + slow[0] + fast[1],
                event[0] + slow[0] + fast[0],
                event[1],
            ]
        )
        mask = torch.tensor(row["action_mask"], dtype=torch.bool)
        scores = scores.masked_fill(~mask, -1.0e9)
        return torch.softmax(scores, dim=-1)

    before = env_probs(original_logits)
    canonical_log_prob = sum(
        torch.log_softmax(original_logits[head], dim=-1)[int(head_targets[head])]
        for head in ("slow", "fast", "event")
    )
    canonical_loss = canonical_log_prob  # negative advantage: gradient descent lowers selected likelihood
    canonical_loss.backward()
    learning_rate = 0.05
    canonical_gradients = {head: original_logits[head].grad.detach().clone() for head in original_logits}
    canonical_after_logits = {
        head: (original_logits[head].detach() - learning_rate * canonical_gradients[head]).requires_grad_(False)
        for head in original_logits
    }
    canonical_after = env_probs(canonical_after_logits)

    env_logits = {
        head: torch.tensor(values, dtype=torch.float64, requires_grad=True)
        for head, values in row["policy_snapshot"]["raw_head_logits"].items()
    }
    env_loss = torch.log(env_probs(env_logits)[target_action])
    env_loss.backward()
    env_gradients = {
        head: (
            env_logits[head].grad.detach().clone()
            if env_logits[head].grad is not None
            else torch.zeros_like(env_logits[head])
        )
        for head in env_logits
    }
    env_after_logits = {
        head: (env_logits[head].detach() - learning_rate * env_gradients[head]).requires_grad_(False)
        for head in env_logits
    }
    env_after = env_probs(env_after_logits)
    return {
        "available": True,
        "episode_identity": {
            key: episode[key]
            for key in ("category", "design_id", "split", "seed", "sampling_seed", "mode")
        },
        "step": row["step"],
        "diagnostic_advantage": row["advantage"],
        "executed_env_action": target_action,
        "canonical_head_targets": head_targets,
        "learning_rate_for_local_sign_check": learning_rate,
        "before_env_probabilities": before.detach().tolist(),
        "canonical_head_loss": {
            "gradients": {head: values.tolist() for head, values in canonical_gradients.items()},
            "after_env_probabilities": canonical_after.detach().tolist(),
        },
        "executed_env_action_loss": {
            "gradients": {head: values.tolist() for head, values in env_gradients.items()},
            "after_env_probabilities": env_after.detach().tolist(),
        },
        "interpretation": (
            "The configured PPO loss uses the canonical reverse-mapped head tuple. "
            "An executed-action loss uses the summed pushforward probability instead."
        ),
    }


def _audit_h1(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for episode in episodes for row in episode["steps"]]
    first_failure = next(
        (
            {
                "episode_identity": {
                    key: episode[key]
                    for key in ("category", "design_id", "split", "seed", "sampling_seed", "mode")
                },
                "step": row,
                "next_step": episode["steps"][index + 1] if index + 1 < len(episode["steps"]) else None,
            }
            for episode in episodes
            for index, row in enumerate(episode["steps"])
            if row["outcomes"]["service_failure"]
        ),
        None,
    )
    repeated_failure_rows = [
        row
        for row in all_rows
        if row["outcomes"]["service_failure"] and not row["exogenous_progress"]["mobility_advanced"]
    ]
    return {
        "episode_count": len(episodes),
        "step_count": len(all_rows),
        "max_episode_budget": MAX_EPISODES,
        "max_step_budget": MAX_EPISODE_STEPS,
        "max_pushforward_probability_error": max(
            float(row["policy_snapshot"]["max_pushforward_error"]) for row in all_rows
        ),
        "max_env_log_prob_error": max(
            float(row["env_log_prob_vs_executed_probability_error"]) for row in all_rows
        ),
        "max_buffer_log_prob_error": max(
            float(row["buffer_vs_executed_probability_error"]) for row in all_rows
        ),
        "buffer_log_prob_mismatch_step_count": sum(
            float(row["buffer_vs_executed_probability_error"]) > 1e-5 for row in all_rows
        ),
        "head_argmax_vs_env_argmax_step_count": sum(
            int(row["policy_snapshot"]["independent_head_argmax_action"])
            != int(row["policy_snapshot"]["declared_env_action_argmax"])
            for row in all_rows
        ),
        "failure_without_mobility_progress_count": len(repeated_failure_rows),
        "first_failure_witness": first_failure,
        "negative_advantage_update_witness": _negative_advantage_update_witness(episodes),
    }


def _original_score(case: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    completed = bool(case["workflow_completed"])
    late = bool(completed and float(case["elapsed_seconds"]) > float(case["deadline_seconds"]))
    components = {
        "reward_node_completion": float(objective["node_completion_reward"]) * int(case["completed_nodes"]),
        "reward_workflow_completion": float(objective["workflow_completion_reward"]) if completed else 0.0,
        "reward_time_penalty": -float(objective["time_weight"]) * float(case["elapsed_seconds"]),
        "reward_transfer_penalty": -float(objective["transfer_gib_weight"])
        * float(case["transfer_bytes"])
        / float(1024**3),
        "reward_failure_penalty": -float(objective["failure_penalty"]) * int(case["service_failures"]),
        "reward_deadline_penalty": -float(objective["deadline_penalty"]) if late else 0.0,
    }
    return {
        **case,
        "late": late,
        "reward_components": components,
        "original_reward": float(sum(components.values())),
    }


def _audit_h2(
    config: dict[str, Any],
    manifest: dict[str, Any],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    objective = config["objective"]
    by_design = {row["design_id"]: row for row in manifest["instances"]}
    component_errors = []
    unfinished_overdue = []
    for row in evaluation_rows:
        component_sum = sum(
            float(row[field])
            for field in (
                "reward_node_completion",
                "reward_workflow_completion",
                "reward_time_penalty",
                "reward_transfer_penalty",
                "reward_failure_penalty",
                "reward_deadline_penalty",
            )
        )
        expected_node = float(objective["node_completion_reward"]) * int(row["completed_nodes"])
        expected_workflow = float(objective["workflow_completion_reward"]) * int(row["workflow_completion_rate"])
        expected_failure = -float(objective["failure_penalty"]) * int(row["service_failures"])
        component_errors.append(
            max(
                abs(component_sum - float(row["reward"])),
                abs(expected_node - float(row["reward_node_completion"])),
                abs(expected_workflow - float(row["reward_workflow_completion"])),
                abs(expected_failure - float(row["reward_failure_penalty"])),
            )
        )
        deadline = float(by_design[row["design_id"]]["deadline_seconds"])
        if not bool(row["workflow_completion_rate"]) and float(row["modeled_completion_seconds"]) > deadline:
            unfinished_overdue.append(
                {
                    "method": row["method"],
                    "seed": row["seed"],
                    "design_id": row["design_id"],
                    "elapsed_seconds": row["modeled_completion_seconds"],
                    "deadline_seconds": deadline,
                    "deadline_violation_rate": row["deadline_violation_rate"],
                    "reward_deadline_penalty": row["reward_deadline_penalty"],
                }
            )
    gib = 1024**3
    cases = [
        {
            "case": "on_time_complete",
            "completed_nodes": 5,
            "total_nodes": 5,
            "workflow_completed": True,
            "elapsed_seconds": 50.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": int(2.0 * gib),
            "service_failures": 0,
            "external_truncation": False,
        },
        {
            "case": "same_completion_later",
            "completed_nodes": 5,
            "total_nodes": 5,
            "workflow_completed": True,
            "elapsed_seconds": 120.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": int(2.0 * gib),
            "service_failures": 0,
            "external_truncation": False,
        },
        {
            "case": "same_completion_more_interruptions",
            "completed_nodes": 5,
            "total_nodes": 5,
            "workflow_completed": True,
            "elapsed_seconds": 50.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": int(2.0 * gib),
            "service_failures": 2,
            "external_truncation": False,
        },
        {
            "case": "lower_transfer_unfinished",
            "completed_nodes": 4,
            "total_nodes": 5,
            "workflow_completed": False,
            "elapsed_seconds": 20.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": 0,
            "service_failures": 0,
            "external_truncation": True,
        },
        {
            "case": "repeated_prepare_without_mobility_progress",
            "completed_nodes": 0,
            "total_nodes": 5,
            "workflow_completed": False,
            "elapsed_seconds": 12.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": int(0.15 * gib),
            "service_failures": 6,
            "external_truncation": True,
        },
        {
            "case": "over_deadline_then_truncated",
            "completed_nodes": 4,
            "total_nodes": 5,
            "workflow_completed": False,
            "elapsed_seconds": 80.0,
            "deadline_seconds": 60.0,
            "transfer_bytes": 0,
            "service_failures": 0,
            "external_truncation": True,
        },
    ]
    scored = [_original_score(case, objective) for case in cases]
    ranking = sorted(scored, key=lambda row: (-float(row["original_reward"]), row["case"]))
    complete_rows = [row for row in evaluation_rows if bool(row["workflow_completion_rate"])]
    incomplete_rows = [row for row in evaluation_rows if not bool(row["workflow_completion_rate"])]
    inversions = []
    for incomplete in incomplete_rows:
        worse_complete = next(
            (
                complete
                for complete in complete_rows
                if float(incomplete["reward"]) > float(complete["reward"])
            ),
            None,
        )
        if worse_complete:
            inversions.append(
                {
                    "incomplete": {
                        key: incomplete[key]
                        for key in ("method", "seed", "design_id", "completed_nodes", "node_count", "reward")
                    },
                    "completed": {
                        key: worse_complete[key]
                        for key in ("method", "seed", "design_id", "completed_nodes", "node_count", "reward")
                    },
                }
            )
        if len(inversions) >= 10:
            break
    return {
        "evaluation_row_count": len(evaluation_rows),
        "max_component_recalculation_error": max(component_errors, default=0.0),
        "node_reward_is_once_per_completed_node": True,
        "workflow_reward_requires_completion": True,
        "deadline_check_only_on_completion": True,
        "external_truncation_terminal_cost_present": False,
        "unfinished_overdue_count": len(unfinished_overdue),
        "unfinished_overdue_examples": unfinished_overdue[:10],
        "observed_incomplete_over_completed_reward_inversions": inversions,
        "unit_cases": scored,
        "original_reward_ranking": [row["case"] for row in ranking],
        "bootstrap_boundary": (
            "PPORolloutBuffer keeps bootstrap for non-terminated truncation; an external timeout cost "
            "must not be implemented by marking every truncation terminated."
        ),
    }


def _sync_node_dependencies(instance: dict[str, Any], edges: list[list[str]]) -> None:
    predecessors = {str(node["node_id"]): [] for node in instance["nodes"]}
    successors = {str(node["node_id"]): [] for node in instance["nodes"]}
    for source, target in edges:
        successors[str(source)].append(str(target))
        predecessors[str(target)].append(str(source))
    for node in instance["nodes"]:
        node_id = str(node["node_id"])
        node["predecessors"] = predecessors[node_id]
        node["successors"] = successors[node_id]
    instance["edges"] = [[str(source), str(target)] for source, target in edges]


def _reachable(edges: list[list[str]], source: str, target: str) -> bool:
    adjacency: dict[str, list[str]] = {}
    for left, right in edges:
        adjacency.setdefault(str(left), []).append(str(right))
    pending = [str(source)]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node == str(target):
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency.get(node, []))
    return False


def _transform_prune_future_edge(instance: dict[str, Any]) -> dict[str, Any] | None:
    result = deepcopy(instance)
    current = str(result["execution_order"][0])
    for edge in result["edges"]:
        if current not in {str(edge[0]), str(edge[1])}:
            edges = [list(item) for item in result["edges"] if list(item) != list(edge)]
            _sync_node_dependencies(result, edges)
            return result
    return None


def _transform_add_future_edge(instance: dict[str, Any]) -> dict[str, Any] | None:
    result = deepcopy(instance)
    order = [str(item) for item in result["execution_order"]]
    existing = {tuple(map(str, edge)) for edge in result["edges"]}
    for left_index in range(1, len(order)):
        for right_index in range(left_index + 1, len(order)):
            candidate = (order[left_index], order[right_index])
            if candidate not in existing and not _reachable(result["edges"], candidate[1], candidate[0]):
                edges = [list(item) for item in result["edges"]] + [list(candidate)]
                _sync_node_dependencies(result, edges)
                return result
    return None


def _transform_swap_incomparable_order(instance: dict[str, Any]) -> dict[str, Any] | None:
    result = deepcopy(instance)
    order = [str(item) for item in result["execution_order"]]
    for index in range(1, len(order) - 1):
        left, right = order[index], order[index + 1]
        if not _reachable(result["edges"], left, right) and not _reachable(result["edges"], right, left):
            order[index], order[index + 1] = right, left
            result["execution_order"] = order
            return result
    return None


def _transform_swap_future_demands(instance: dict[str, Any]) -> dict[str, Any] | None:
    result = deepcopy(instance)
    current = str(result["execution_order"][0])
    future_nodes = [node for node in result["nodes"] if str(node["node_id"]) != current]
    for left_index, left in enumerate(future_nodes):
        for right in future_nodes[left_index + 1 :]:
            if str(left["required_adapter"]) != str(right["required_adapter"]):
                for field in ("required_adapter", "required_base_model"):
                    left[field], right[field] = right[field], left[field]
                return result
    return None


def _planner_objective(env: CalibratedContinuousWorkflowEnv) -> tuple[Any, ...]:
    metrics = env.metrics
    return (
        -int(metrics["completed_nodes"]),
        int(metrics["service_failures"]),
        int(metrics["deadline_violations"]),
        float(metrics["modeled_completion_seconds"]),
        int(metrics["model_transfer_bytes"] + metrics["state_transfer_bytes"] + metrics["input_transfer_bytes"]),
    )


def _enumerate_first_actions(
    env: CalibratedContinuousWorkflowEnv,
    *,
    depth: int,
    branch_counter: list[int],
    branch_cap: int,
) -> dict[str, Any]:
    results: dict[int, tuple[Any, ...]] = {}

    def visit(state: CalibratedContinuousWorkflowEnv, remaining: int) -> tuple[Any, ...]:
        if remaining <= 0 or state.terminated or branch_counter[0] >= branch_cap:
            return _planner_objective(state)
        scores = []
        for action in state.valid_actions():
            if branch_counter[0] >= branch_cap:
                break
            branch_counter[0] += 1
            child = state.clone_for_decision_model()
            _, _, terminated, truncated, _ = child.step(action)
            score = _planner_objective(child) if terminated or truncated else visit(child, remaining - 1)
            scores.append(score)
        return min(scores) if scores else _planner_objective(state)

    for action in env.valid_actions():
        if branch_counter[0] >= branch_cap:
            break
        branch_counter[0] += 1
        child = env.clone_for_decision_model()
        _, _, terminated, truncated, _ = child.step(action)
        results[int(action)] = (
            _planner_objective(child)
            if terminated or truncated or depth <= 1
            else visit(child, depth - 1)
        )
    if not results:
        return {"scores": {}, "best_actions": [], "branch_cap_reached": True}
    best_score = min(results.values())
    return {
        "scores": {str(action): list(score) for action, score in results.items()},
        "best_actions": sorted(action for action, score in results.items() if score == best_score),
        "branch_cap_reached": branch_counter[0] >= branch_cap,
    }


def _encoder_signature(agent: Any, semantic_state: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        encoded = agent._network.encoder(semantic_state)
        output = agent._forward_policy(
            semantic_state,
            run_metadata={"policy_evaluation_mode": "raw_policy"},
        )
    return {
        "graph_embedding": encoded["shared_embedding"].detach().cpu().tolist(),
        "slow_logits": output["slow_logits"].detach().cpu().tolist(),
        "fast_logits": output["fast_logits"].detach().cpu().tolist(),
        "event_logits": output["event_logits"].detach().cpu().tolist(),
    }


def _max_signature_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
        for key in left
    }


def _relabel_instance(instance: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(instance)
    mapping = {
        str(node_id): f"perm_{index:02d}"
        for index, node_id in enumerate(reversed(result["execution_order"]))
    }
    for node in result["nodes"]:
        node["node_id"] = mapping[str(node["node_id"])]
        node["predecessors"] = [mapping[str(item)] for item in node.get("predecessors", [])]
        node["successors"] = [mapping[str(item)] for item in node.get("successors", [])]
    result["edges"] = [[mapping[str(left)], mapping[str(right)]] for left, right in result["edges"]]
    result["execution_order"] = [mapping[str(item)] for item in result["execution_order"]]
    result["nodes"] = list(reversed(result["nodes"]))
    return result


def _audit_h3(
    config: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    transformations: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any] | None]]] = [
        ("prune_first_future_edge", _transform_prune_future_edge),
        ("add_first_legal_future_edge", _transform_add_future_edge),
        ("swap_first_incomparable_future_execution", _transform_swap_incomparable_order),
        ("swap_first_distinct_future_model_demand", _transform_swap_future_demands),
    ]
    candidates = [row for row in manifest["instances"] if row["split"] in {"train", "dev"}]
    agent = _build_agent(7, config["training"], checkpoint, deterministic=True)
    branch_counter = [0]
    used: set[str] = set()
    pairs = []
    for transform_name, transform in transformations:
        selected = None
        for candidate in candidates:
            if candidate["design_id"] in used:
                continue
            variant = transform(candidate)
            if variant is not None:
                selected = (candidate, variant)
                break
        if selected is None:
            pairs.append({"transform": transform_name, "available": False})
            continue
        original, variant = selected
        used.add(original["design_id"])
        sides = []
        for label, instance in (("original", original), ("variant", variant)):
            env = CalibratedContinuousWorkflowEnv(config, instance)
            _, info = env.reset()
            enumeration = _enumerate_first_actions(
                env,
                depth=DAG_ENUMERATION_DEPTH,
                branch_counter=branch_counter,
                branch_cap=DAG_TOTAL_BRANCH_CAP,
            )
            immediate_action = ImmediateCostRule().select_action(env)
            two_step_action = TwoStepCostRule().select_action(env)
            signature = _encoder_signature(agent, info["semantic_state"])
            action, action_info = agent.act(np.zeros(9, dtype=np.float32), info)
            sides.append(
                {
                    "label": label,
                    "enumeration": enumeration,
                    "immediate_rule_action": int(immediate_action),
                    "two_step_rule_action": int(two_step_action),
                    "sa_env_argmax_action": int(action),
                    "sa_env_probabilities": action_info["env_action_probs"],
                    "encoder_signature": signature,
                    "current_node": deepcopy(info["semantic_state"]["current_workflow_node"]),
                    "current_rsu_id": env._current_rsu_id(),
                    "current_bundle_ready": env._bundle_ready(
                        env._current_rsu_id(),
                        str(env._current_node()["required_adapter"]),
                    ),
                }
            )
        pairs.append(
            {
                "transform": transform_name,
                "available": True,
                "design_id": original["design_id"],
                "sides": sides,
                "enumerated_first_action_changed": sides[0]["enumeration"]["best_actions"]
                != sides[1]["enumeration"]["best_actions"],
                "immediate_rule_changed": sides[0]["immediate_rule_action"] != sides[1]["immediate_rule_action"],
                "two_step_rule_changed": sides[0]["two_step_rule_action"] != sides[1]["two_step_rule_action"],
                "sa_action_changed": sides[0]["sa_env_argmax_action"] != sides[1]["sa_env_argmax_action"],
                "encoder_max_abs_diff": _max_signature_diff(
                    sides[0]["encoder_signature"], sides[1]["encoder_signature"]
                ),
            }
        )
    invariant_base = candidates[0]
    invariant_variant = _relabel_instance(invariant_base)
    invariant_sides = []
    for label, instance in (("original", invariant_base), ("relabel_and_reverse_node_list", invariant_variant)):
        env = CalibratedContinuousWorkflowEnv(config, instance)
        _, info = env.reset()
        enumeration = _enumerate_first_actions(
            env,
            depth=DAG_ENUMERATION_DEPTH,
            branch_counter=branch_counter,
            branch_cap=DAG_TOTAL_BRANCH_CAP,
        )
        signature = _encoder_signature(agent, info["semantic_state"])
        action, action_info = agent.act(np.zeros(9, dtype=np.float32), info)
        invariant_sides.append(
            {
                "label": label,
                "enumeration": enumeration,
                "sa_env_argmax_action": int(action),
                "sa_env_probabilities": action_info["env_action_probs"],
                "encoder_signature": signature,
            }
        )
    return {
        "pair_count": sum(bool(pair.get("available")) for pair in pairs),
        "enumeration_depth": DAG_ENUMERATION_DEPTH,
        "total_branch_cap": DAG_TOTAL_BRANCH_CAP,
        "branches_consumed": branch_counter[0],
        "branch_cap_reached": branch_counter[0] >= DAG_TOTAL_BRANCH_CAP,
        "pairs": pairs,
        "identifier_and_list_order_invariance": {
            "design_id": invariant_base["design_id"],
            "sides": invariant_sides,
            "reasonable_action_invariant": invariant_sides[0]["enumeration"]["best_actions"]
            == invariant_sides[1]["enumeration"]["best_actions"],
            "sa_action_invariant": invariant_sides[0]["sa_env_argmax_action"]
            == invariant_sides[1]["sa_env_argmax_action"],
            "encoder_max_abs_diff": _max_signature_diff(
                invariant_sides[0]["encoder_signature"], invariant_sides[1]["encoder_signature"]
            ),
        },
        "environment_execution_contract": {
            "node_selection": "fixed execution_order[node_index]",
            "dependency_edges_gate_legal_node_selection": False,
            "policy_selects_node": False,
            "policy_selects_prepare_object": False,
            "edge_effects": "encoder message passing/frontier plus handoff prefix recompute ancestry",
        },
    }


def _audit_adaptation(
    config: dict[str, Any],
    instance: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    semantic = deepcopy(info["semantic_state"])
    current_rsu_id = env._current_rsu_id()
    rsu_index = next(index for index, row in enumerate(semantic["rsus"]) if row["rsu_id"] == current_rsu_id)
    base_rsu = semantic["rsus"][rsu_index]
    flat = FlatSemanticEncoder()
    agent = _build_agent(7, config["training"], checkpoint, deterministic=True)

    def vectors(state: dict[str, Any]) -> dict[str, np.ndarray]:
        with torch.no_grad():
            graph = agent._network.encoder(state)["shared_embedding"].detach().cpu().numpy()
        return {
            "flat_actor": flat._build_feature_tensor(state).detach().cpu().numpy(),
            "flat_centralized": flat._build_centralized_feature_tensor(state).detach().cpu().numpy(),
            "graph_shared": graph,
        }

    baseline = vectors(semantic)
    perturbations = {}
    for name, mutate in (
        ("cache_capacity_x2", lambda row: row.__setitem__("cache_capacity", float(row["cache_capacity"]) * 2.0)),
        ("cache_used_bytes_half", lambda row: row.__setitem__("cache_used_bytes", int(row["cache_used_bytes"]) // 2)),
        (
            "adapter_count_plus_one",
            lambda row: row.__setitem__("cached_adapter_ids", list(row["cached_adapter_ids"]) + ["diagnostic_extra"]),
        ),
    ):
        changed = deepcopy(semantic)
        mutate(changed["rsus"][rsu_index])
        current = vectors(changed)
        perturbations[name] = {
            key: float(np.max(np.abs(current[key] - baseline[key])))
            for key in baseline
        }
    return {
        "design_id": instance["design_id"],
        "current_rsu_id": current_rsu_id,
        "cache_capacity_value": base_rsu["cache_capacity"],
        "cache_capacity_unit": "bytes",
        "cache_used_bytes_value": base_rsu["cache_used_bytes"],
        "cache_used_bytes_unit": "bytes",
        "adapter_count": len(base_rsu["cached_adapter_ids"]),
        "true_used_fraction": float(base_rsu["cache_used_bytes"]) / float(base_rsu["cache_capacity"]),
        "flat_critic_implemented_fraction": float(len(base_rsu["cached_adapter_ids"]))
        / float(base_rsu["cache_capacity"]),
        "observation_used_fraction": float(observation[6]),
        "single_field_perturbation_max_abs_diff": perturbations,
        "agent_consumes_array_observation": False,
    }


def _planner_permission_audit() -> dict[str, Any]:
    return {
        "clone_primitive": "deepcopy of full CalibratedContinuousWorkflowEnv internal state",
        "retained_internal_state": [
            "full frozen instance including all future nodes, edges, execution_order and rsu_sequence",
            "all typed-cache residents/LRU stamps/capacities",
            "completed set, node_index, step_index, clock, last execution RSU and prepared state",
            "exact deterministic transition and cost implementation",
        ],
        "decision_time_estimate_boundary": {
            "link_rate": "estimated_mbps via clone_for_decision_model",
            "actual_mbps_used_by_preview": False,
            "mobility": "exact frozen future rsu_sequence is retained; no separate estimated sequence is present",
            "future_workflow_demand": "full exact frozen DAG, order, model demands and costs are retained",
        },
        "objective": [
            "maximize completed nodes",
            "minimize service failures",
            "minimize deadline violations",
            "minimize elapsed time",
            "minimize transferred bytes",
        ],
        "learning_method_difference": {
            "transition_model": "model-free actor; no exact clone rollout in the frozen profile",
            "objective": "scalar reward rather than planner lexicographic tuple",
            "consumed_input": "semantic encoder drops some public byte/typed/link fields",
        },
        "classification": "strong model-based baseline with different model capability and objective; retain, label, do not weaken",
    }


def _verify_inputs(
    config_path: Path,
    manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    config_hash = _sha256(config_path)
    manifest_hash = _sha256(manifest_path)
    run_manifest = _read_json(artifact_root / "run_manifest.json")
    integrity = _read_json(artifact_root / "artifact_integrity.json")
    expected = {row["path"]: row for row in integrity["files"]}
    selected = {}
    for seed in DIAGNOSTIC_SEEDS:
        relative = f"checkpoints/sa_ghmappo_seed{seed}_selected.pt"
        path = artifact_root / relative
        actual_hash = _sha256(path)
        if relative not in expected or actual_hash != expected[relative]["sha256"]:
            raise RuntimeError(f"checkpoint integrity mismatch: {relative}")
        selected[str(seed)] = {
            "path": str(path),
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
    if config_hash != run_manifest["config"]["sha256"]:
        raise RuntimeError("config hash does not match frozen run manifest")
    if manifest_hash != run_manifest["workload_manifest"]["sha256"]:
        raise RuntimeError("workload manifest hash does not match frozen run manifest")
    return {
        "run_source_git_commit": run_manifest["git_commit"],
        "config": {"path": str(config_path), "sha256": config_hash},
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "pilot_artifact_root": str(artifact_root),
        "selected_checkpoints": selected,
    }


def _write_integrity(output_root: Path, filenames: list[str]) -> None:
    files = []
    for filename in filenames:
        path = output_root / filename
        files.append({"path": filename, "sha256": _sha256(path), "bytes": path.stat().st_size})
    _write_json(output_root / "artifact_integrity.json", {"schema_version": "artifact_integrity_v1", "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot_artifact_root", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT_DIR / "artifacts/analysis/sa_ghmappo_completion_root_cause_20261006_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    artifact_root = args.pilot_artifact_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_verification = _verify_inputs(config_path, manifest_path, artifact_root)
    config = _read_json(config_path)
    manifest = _read_json(manifest_path)
    selected_instances = _select_fixed_instances(config, manifest)

    episodes = []
    episode_ordinal = 0
    for selected in selected_instances:
        for seed in DIAGNOSTIC_SEEDS:
            checkpoint = Path(input_verification["selected_checkpoints"][str(seed)]["path"])
            for mode in MODES:
                episode_ordinal += 1
                episodes.append(
                    _run_episode(
                        config=config,
                        instance=selected["instance"],
                        checkpoint=checkpoint,
                        seed=seed,
                        category=selected["category"],
                        mode=mode,
                        episode_ordinal=episode_ordinal,
                    )
                )
    if len(episodes) > MAX_EPISODES:
        raise RuntimeError("diagnostic episode budget exceeded")

    evaluation_rows = _read_json(artifact_root / "evaluation_rows.json")
    h1 = _audit_h1(episodes)
    h2 = _audit_h2(config, manifest, evaluation_rows)
    seed7_checkpoint = Path(input_verification["selected_checkpoints"]["7"]["path"])
    h3 = _audit_h3(config, manifest, seed7_checkpoint)
    adaptation = _audit_adaptation(config, selected_instances[0]["instance"], seed7_checkpoint)
    planner = _planner_permission_audit()
    metadata = {
        "schema_version": "sa_ghmappo_completion_root_cause_diagnosis_v1",
        "diagnosis_git_commit": _git_commit(),
        "input_verification": input_verification,
        "selected_instances": [
            {
                "category": row["category"],
                "design_id": row["instance"]["design_id"],
                "split": row["instance"]["split"],
                "current_ready": row["current_ready"],
                "contact_budget_seconds": row["contact_budget_seconds"],
                "target_rsu_id": row["target_rsu_id"],
                "selection_rule": "first manifest-order match; imminent uses first minimum-contact distinct instance",
            }
            for row in selected_instances
        ],
        "budgets": {
            "new_training": 0,
            "real_model_generate": 0,
            "downloads": 0,
            "old_holdout": 0,
            "diagnostic_episodes": len(episodes),
            "episode_step_cap": MAX_EPISODE_STEPS,
            "reward_unit_cases": len(h2["unit_cases"]),
            "dag_pair_count": h3["pair_count"],
            "dag_enumeration_depth": DAG_ENUMERATION_DEPTH,
            "dag_branch_cap": DAG_TOTAL_BRANCH_CAP,
            "automatic_retries": 0,
        },
    }
    files = {
        "metadata.json": metadata,
        "h1_episode_traces.json": episodes,
        "h1_probability_and_update_audit.json": h1,
        "h2_reward_audit.json": h2,
        "h3_dag_information_value_audit.json": h3,
        "encoder_unit_adaptation_audit.json": adaptation,
        "planner_permission_audit.json": planner,
    }
    for filename, payload in files.items():
        _write_json(output_root / filename, _json_safe(payload))
    receipt = {
        "status": "complete",
        "diagnostic_only": True,
        "episode_count": len(episodes),
        "reward_case_count": len(h2["unit_cases"]),
        "dag_pair_count": h3["pair_count"],
        "new_training": 0,
        "real_model_generate": 0,
        "downloads": 0,
        "old_holdout": 0,
        "automatic_retries": 0,
    }
    _write_json(output_root / "completion_receipt.json", receipt)
    _write_integrity(output_root, [*files, "completion_receipt.json"])
    print(json.dumps({"output_root": str(output_root), **receipt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
