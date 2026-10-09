"""Read-only diagnosis for the calibrated workflow reward-learning run.

The script never updates model parameters.  It verifies the recorded checkpoint
hashes, replays the committed evaluation action ledger, recomputes values/GAE,
and probes loss-gradient scales on a deterministic development sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.distributions import Categorical

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _build_agent,
    _collect_training_episode,
)
from scripts.run_calibrated_workflow_service_reward_alignment import (  # noqa: E402
    _resolved_arm_config,
)
from src.encoders.calibrated_workflow_features import (  # noqa: E402
    bundle_ready,
    rsu_by_id,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)
from src.trainers.ppo_buffer import PPORolloutBuffer  # noqa: E402


RUN_ID = "calibrated_workflow_service_reward_learning_diagnosis_20261009_v1"
SOURCE_RUN_ID = "calibrated_workflow_service_reward_alignment_20261006_v2"
EXECUTION_COMMIT = "cc1ecb44d998465c86350ebf4c216e13600f52e2"
BASELINE_COMMIT = "70a83afb5d3d3b3fb40fcaeaf825a82e4b74c416"
METHOD_ORDER = {"sa_ghmappo": 0, "mappo": 1, "ppo": 2}
SPLIT_ORDER = {"regression": 0, "frozen_check": 1}


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return float(np.mean(data)) if data else None


def _std(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return float(np.std(data)) if data else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    data = [float(value) for value in values]
    return float(np.quantile(data, q)) if data else None


def _fraction(flags: Iterable[bool]) -> float | None:
    data = [bool(flag) for flag in flags]
    return float(np.mean(data)) if data else None


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _verify_code_identity(source_root: Path) -> dict[str, Any]:
    identity = _load_json(source_root / "execution_code_identity.json")
    critical: dict[str, Any] = {}
    for relative, expected in identity["critical_file_sha256"].items():
        path = ROOT_DIR / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"critical execution file drift: {relative}")
        critical[relative] = {"expected": expected, "actual": actual, "match": True}

    agent_files = [
        "src/agents/sa_ghmappo_core.py",
        "src/agents/sa_ghmappo_agent.py",
        "src/agents/mappo_agent.py",
        "src/agents/ppo_agent.py",
        "src/agents/registry.py",
        "src/trainers/ppo_buffer.py",
    ]
    agents: dict[str, Any] = {}
    for relative in agent_files:
        current = _sha256(ROOT_DIR / relative)
        payload = subprocess.run(
            ["git", "show", f"{EXECUTION_COMMIT}:{relative}"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
        ).stdout
        at_execution = _bytes_sha256(payload)
        if current != at_execution:
            raise RuntimeError(f"agent/trainer code differs from execution commit: {relative}")
        agents[relative] = {
            "current_sha256": current,
            "execution_commit_sha256": at_execution,
            "match": True,
        }
    return {
        "recorded_execution_commit": identity["base_git_commit"],
        "expected_execution_commit": EXECUTION_COMMIT,
        "delivery_baseline_commit": BASELINE_COMMIT,
        "critical_execution_files": critical,
        "agent_and_buffer_files_recovered_from_execution_commit": agents,
    }


def _integrity_lookup(source_root: Path) -> dict[str, dict[str, Any]]:
    integrity = _load_json(source_root / "artifact_integrity.json")
    return {str(row["path"]): dict(row) for row in integrity["files"]}


def _verify_checkpoints(
    *,
    source_root: Path,
    checkpoint_root: Path,
    training_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = _integrity_lookup(source_root)
    expected_summary_hashes = {
        str(row["selected_checkpoint"]): str(row["selected_checkpoint_sha256"])
        for row in training_summary
    }
    checkpoint_records = [
        row for relative, row in lookup.items() if relative.startswith("checkpoints/")
    ]
    verified: list[dict[str, Any]] = []
    for record in sorted(checkpoint_records, key=lambda row: str(row["path"])):
        relative = str(record["path"])
        path = checkpoint_root / relative
        integrity_expected = str(record["sha256"])
        actual = _sha256(path)
        if actual != integrity_expected:
            raise RuntimeError(f"checkpoint hash mismatch: {relative}")
        if relative in expected_summary_hashes and actual != expected_summary_hashes[relative]:
            raise RuntimeError(f"selected checkpoint summary hash mismatch: {relative}")
        verified.append(
            {
                "relative_path": relative,
                "sha256": actual,
                "bytes": path.stat().st_size,
                "selected_checkpoint": relative in expected_summary_hashes,
                "uploaded_or_copied": False,
            }
        )
    return {
        "checkpoint_root": str(checkpoint_root),
        "verified_checkpoint_count": len(verified),
        "verified_selected_checkpoint_count": sum(row["selected_checkpoint"] for row in verified),
        "all_hashes_match_source_integrity_and_selected_training_summary": True,
        "checkpoints": verified,
    }


def _group_ledger(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] == "two_step_cost_rule":
            continue
        key = (
            row["reward_arm"],
            row["method"],
            int(row["seed"]),
            row["split"],
            row["design_id"],
        )
        grouped[key].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["step_index"]))
    return grouped


def _gae(
    rewards: list[float],
    values: list[float],
    terminated: list[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in reversed(range(len(rewards))):
        next_value = last_value if index == len(rewards) - 1 else values[index + 1]
        non_terminal = 0.0 if terminated[index] else 1.0
        delta = rewards[index] + gamma * next_value * non_terminal - values[index]
        running = delta + gamma * gae_lambda * non_terminal * running
        advantages[index] = float(running)
    returns = [float(value + advantage) for value, advantage in zip(values, advantages)]
    return advantages, returns


def _replay_evaluation_ledger(
    *,
    experiment: dict[str, Any],
    base_config: dict[str, Any],
    manifest: dict[str, Any],
    source_root: Path,
    checkpoint_root: Path,
    training_summary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with (source_root / "behavior_ledger.csv").open(encoding="utf-8", newline="") as handle:
        ledger_rows = list(csv.DictReader(handle))
    grouped = _group_ledger(ledger_rows)
    instances = {str(row["design_id"]): row for row in manifest["instances"]}
    evaluation_rows = _load_json(source_root / "evaluation_rows.json")
    outcomes = {
        (
            str(row["reward_arm"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["design_id"]),
        ): row
        for row in evaluation_rows
        if row["method"] != "two_step_cost_rule"
    }
    summaries = {
        (str(row["reward_arm"]), str(row["method"]), int(row["seed"])): row
        for row in training_summary
    }
    arm_configs = {
        str(arm["arm"]): _resolved_arm_config(
            base_config,
            experiment,
            str(arm["reward_profile"]),
        )
        for arm in experiment["reward_arms"]
    }

    replay_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for cell_key in sorted(
        summaries,
        key=lambda key: (key[0], METHOD_ORDER[key[1]], key[2]),
    ):
        arm, method, seed = cell_key
        config = arm_configs[arm]
        summary = summaries[cell_key]
        agent = _build_agent(method, seed, config)
        agent.load(str(checkpoint_root / summary["selected_checkpoint"]))
        for episode_key in sorted(
            (key for key in grouped if key[:3] == cell_key),
            key=lambda key: (SPLIT_ORDER[key[3]], key[4]),
        ):
            _, _, _, split, design_id = episode_key
            recorded = grouped[episode_key]
            instance = deepcopy(instances[design_id])
            env = CalibratedContinuousWorkflowEnv(config, instance)
            observation, info = env.reset()
            rewards: list[float] = []
            values: list[float] = []
            terminated_flags: list[bool] = []
            annotations: list[dict[str, Any]] = []
            for expected in recorded:
                value = float(agent.evaluate_value(observation, info))
                action = int(expected["executed_action"])
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                transition = dict(next_info.get("transition", {}) or {})
                expected_components = json.loads(expected["reward_components"])
                if (
                    int(transition.get("step_index", -1)) != int(expected["step_index"])
                    or abs(float(reward) - sum(float(v) for v in expected_components.values())) > 1e-8
                    or bool(transition.get("service_completed"))
                    != (expected["service_completed"] == "True")
                ):
                    mismatch_count += 1
                rewards.append(float(reward))
                values.append(value)
                terminated_flags.append(bool(terminated))
                deadline = float(instance["deadline_seconds"])
                current_ready = expected["current_bundle_ready"] == "True"
                prepare_feasible = expected["target_prepare_feasible"] == "True"
                progressed = expected["progressed"] == "True"
                annotations.append(
                    {
                        "split": split,
                        "design_id": design_id,
                        "step_index": int(expected["step_index"]),
                        "action": action,
                        "current_bundle_ready": current_ready,
                        "target_bundle_ready": expected["target_bundle_ready"] == "True",
                        "target_prepare_feasible": prepare_feasible,
                        "deadline_phase": (
                            "before"
                            if float(expected["clock_seconds_after"]) <= deadline
                            else "after"
                        ),
                        "progressed": progressed,
                        "consecutive_no_progress": int(expected["no_progress_streak"]) >= 2,
                        "unwise_action4": bool(action == 4 and (not current_ready or not prepare_feasible)),
                        "service_failed": not (expected["service_completed"] == "True"),
                        "env_action_probs": json.loads(expected["env_action_probs"] or "[]"),
                    }
                )
                observation, info = next_observation, next_info
                if terminated or truncated:
                    break
            if len(rewards) != len(recorded):
                raise RuntimeError(f"ledger replay length mismatch: {episode_key}")
            terminated = bool(env.terminated)
            last_value = 0.0 if terminated else float(agent.evaluate_value(observation, info))
            variants = {
                "configured": (float(config["training"]["gamma"]), float(config["training"]["gae_lambda"])),
                "lambda_1": (float(config["training"]["gamma"]), 1.0),
                "undiscounted": (1.0, 1.0),
                "no_bootstrap": (float(config["training"]["gamma"]), float(config["training"]["gae_lambda"])),
            }
            computed: dict[str, tuple[list[float], list[float]]] = {}
            for name, (gamma, gae_lambda) in variants.items():
                computed[name] = _gae(
                    rewards,
                    values,
                    terminated_flags,
                    0.0 if name == "no_bootstrap" else last_value,
                    gamma,
                    gae_lambda,
                )
            configured_advantages, configured_returns = computed["configured"]
            outcome = outcomes[episode_key]
            for index, annotation in enumerate(annotations):
                row = {
                    "reward_arm": arm,
                    "method": method,
                    "seed": seed,
                    **annotation,
                    "reward": rewards[index],
                    "value_prediction": values[index],
                    "value_target": configured_returns[index],
                    "critic_error": configured_returns[index] - values[index],
                    "advantage_raw": configured_advantages[index],
                    "episode_completed": bool(float(outcome["workflow_completion_rate"])),
                    "episode_on_time": bool(float(outcome["on_time_workflow_completion_rate"])),
                    "episode_truncated": not terminated,
                }
                for variant, (variant_advantages, variant_returns) in computed.items():
                    row[f"advantage_{variant}"] = variant_advantages[index]
                    row[f"return_{variant}"] = variant_returns[index]
                replay_rows.append(row)
            episode_rows.append(
                {
                    "reward_arm": arm,
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "design_id": design_id,
                    "steps": len(rewards),
                    "terminated": terminated,
                    "truncated": not terminated,
                    "workflow_completed": bool(float(outcome["workflow_completion_rate"])),
                    "workflow_completed_on_time": bool(
                        float(outcome["on_time_workflow_completion_rate"])
                    ),
                    "last_observation_bootstrap_value": last_value,
                    **{
                        f"first_return_{name}": variant_returns[0]
                        for name, (_, variant_returns) in computed.items()
                    },
                    "bootstrap_delta_at_first_step": configured_returns[0]
                    - computed["no_bootstrap"][1][0],
                }
            )

    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        by_cell[(row["reward_arm"], row["method"], row["seed"])].append(row)
    for rows in by_cell.values():
        advantages = np.asarray([float(row["advantage_raw"]) for row in rows], dtype=np.float64)
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for row, value in zip(rows, normalized):
            row["advantage_normalized_offline_cell"] = float(value)

    return replay_rows, episode_rows, {
        "replayed_episode_count": len(episode_rows),
        "replayed_step_count": len(replay_rows),
        "transition_or_reward_mismatch_count": mismatch_count,
        "action_source": "committed behavior_ledger.csv executed_action",
        "value_source": "hash-verified selected checkpoint offline forward",
        "gae_source": "offline recomputation; not original training minibatches",
    }


def _gradient_vector(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flattened = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(flattened) if flattened else torch.zeros(1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return None
    return float(torch.dot(left, right).item() / denominator.item())


def _losses_for_rollout(agent: Any, rollout: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_states = [agent._extract_semantic_state(row["decision_info"]) for row in rollout]
    run_metadata = [dict(row["decision_info"].get("run_metadata", {}) or {}) for row in rollout]
    masks = [agent._extract_action_mask(row["decision_info"]) for row in rollout]
    actions = torch.as_tensor([int(row["action"]) for row in rollout], dtype=torch.long)
    returns = torch.as_tensor([float(row["return"]) for row in rollout], dtype=torch.float32)
    raw_advantages = np.asarray([float(row["advantage"]) for row in rollout], dtype=np.float32)
    normalized_advantages = torch.as_tensor(
        (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8),
        dtype=torch.float32,
    )
    outputs = [
        agent._forward_policy(state, run_metadata=metadata)
        for state, metadata in zip(semantic_states, run_metadata)
    ]
    logits = torch.stack(
        [
            agent._masked_flat_logits(
                agent._hierarchical_env_action_scores(output)
                if agent._use_hierarchy
                else output["flat_logits"],
                mask,
            )
            for output, mask in zip(outputs, masks)
        ]
    )
    distribution = Categorical(logits=logits)
    new_log_probs = distribution.log_prob(actions)
    old_log_probs = new_log_probs.detach()
    ratios = torch.exp(new_log_probs - old_log_probs)
    surrogate = torch.minimum(
        ratios * normalized_advantages,
        torch.clamp(ratios, 1.0 - agent._clip_ratio, 1.0 + agent._clip_ratio)
        * normalized_advantages,
    )
    actor_loss = -surrogate.mean()
    entropy = distribution.entropy().mean()
    policy_loss = actor_loss - agent._entropy_coef * entropy
    predictions = torch.stack([output["value"] for output in outputs])
    value_loss = torch.mean((returns - predictions) ** 2)
    weighted_value_loss = agent._value_coef * value_loss
    auxiliary_loss = agent._compute_auxiliary_loss(
        batch_states=semantic_states,
        batch_outputs=outputs,
    )
    weighted_auxiliary_loss = agent._auxiliary_coef * auxiliary_loss
    return_std = torch.std(returns, unbiased=False).clamp_min(1.0)
    scale_normalized_weighted_value_loss = weighted_value_loss / return_std.square()
    total_loss = policy_loss + weighted_value_loss + weighted_auxiliary_loss
    return {
        "policy_loss_tensor": policy_loss,
        "weighted_value_loss_tensor": weighted_value_loss,
        "weighted_auxiliary_loss_tensor": weighted_auxiliary_loss,
        "scale_normalized_weighted_value_loss_tensor": scale_normalized_weighted_value_loss,
        "total_loss_tensor": total_loss,
        "actor_loss": float(actor_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "weighted_value_loss": float(weighted_value_loss.detach().item()),
        "auxiliary_loss": float(auxiliary_loss.detach().item()),
        "weighted_auxiliary_loss": float(weighted_auxiliary_loss.detach().item()),
        "return_mean": float(returns.mean().item()),
        "return_std": float(return_std.item()),
        "value_prediction_mean": float(predictions.mean().detach().item()),
        "advantage_raw_mean": float(raw_advantages.mean()),
        "advantage_raw_std": float(raw_advantages.std()),
    }


def _gradient_probe(
    *,
    experiment: dict[str, Any],
    base_config: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dev_instances = [row for row in manifest["instances"] if row["split"] == "dev"]
    if [row["design_id"] for row in dev_instances] != ["dev_00", "dev_01", "dev_02", "dev_03"]:
        raise RuntimeError("deterministic development sample identity drift")
    rows: list[dict[str, Any]] = []
    for arm in experiment["reward_arms"]:
        arm_name = str(arm["arm"])
        config = _resolved_arm_config(base_config, experiment, str(arm["reward_profile"]))
        for method in experiment["training"]["agents"]:
            for seed in experiment["training"]["seeds"]:
                checkpoint = checkpoint_root / "checkpoints" / f"{arm_name}_{method}_seed{seed}_episode48.pt"
                agent = _build_agent(method, int(seed), config)
                agent.load(str(checkpoint))
                agent._deterministic_action = True
                rollout: list[dict[str, Any]] = []
                for instance in dev_instances:
                    episode, _ = _collect_training_episode(agent, config, instance, 24)
                    rollout.extend(episode)
                losses = _losses_for_rollout(agent, rollout)
                named_parameters = [
                    (name, parameter)
                    for name, parameter in agent._network.named_parameters()
                    if parameter.requires_grad
                ]
                parameters = [parameter for _, parameter in named_parameters]
                encoder_indices = [
                    index for index, (name, _) in enumerate(named_parameters) if name.startswith("encoder.")
                ]
                vectors = {
                    "policy": _gradient_vector(losses["policy_loss_tensor"], parameters, retain_graph=True),
                    "value": _gradient_vector(losses["weighted_value_loss_tensor"], parameters, retain_graph=True),
                    "auxiliary": _gradient_vector(
                        losses["weighted_auxiliary_loss_tensor"], parameters, retain_graph=True
                    ),
                    "scale_normalized_value": _gradient_vector(
                        losses["scale_normalized_weighted_value_loss_tensor"],
                        parameters,
                        retain_graph=True,
                    ),
                    "total": _gradient_vector(losses["total_loss_tensor"], parameters, retain_graph=True),
                }
                encoder_parameters = [parameters[index] for index in encoder_indices]
                encoder_vectors = {
                    "policy": _gradient_vector(
                        losses["policy_loss_tensor"], encoder_parameters, retain_graph=True
                    ),
                    "value": _gradient_vector(
                        losses["weighted_value_loss_tensor"], encoder_parameters, retain_graph=True
                    ),
                    "auxiliary": _gradient_vector(
                        losses["weighted_auxiliary_loss_tensor"], encoder_parameters, retain_graph=True
                    ),
                    "scale_normalized_value": _gradient_vector(
                        losses["scale_normalized_weighted_value_loss_tensor"],
                        encoder_parameters,
                        retain_graph=False,
                    ),
                }
                norms = {
                    name: float(torch.linalg.vector_norm(vector).item())
                    for name, vector in vectors.items()
                }
                encoder_norms = {
                    name: float(torch.linalg.vector_norm(vector).item())
                    for name, vector in encoder_vectors.items()
                }
                missing_rollout = []
                for sample in rollout:
                    semantic = agent._extract_semantic_state(sample["decision_info"])
                    node = semantic.get("current_workflow_node") or {}
                    vehicle = (semantic.get("vehicles") or [{}])[0]
                    current_rsu_id = vehicle.get("associated_rsu_id")
                    ready = bool(
                        bundle_ready(
                            semantic,
                            rsu_by_id(semantic, current_rsu_id),
                            node,
                        )
                    )
                    if not ready:
                        missing_rollout.append(sample)
                missing_policy_auxiliary_cosine = None
                if len(missing_rollout) >= 2 and float(agent._auxiliary_coef) > 0.0:
                    missing_losses = _losses_for_rollout(agent, missing_rollout)
                    missing_policy = _gradient_vector(
                        missing_losses["policy_loss_tensor"], parameters, retain_graph=True
                    )
                    missing_auxiliary = _gradient_vector(
                        missing_losses["weighted_auxiliary_loss_tensor"],
                        parameters,
                        retain_graph=False,
                    )
                    missing_policy_auxiliary_cosine = _cosine(
                        missing_policy,
                        missing_auxiliary,
                    )
                clip_scale = min(1.0, float(agent._max_grad_norm) / max(norms["total"], 1e-12))
                rows.append(
                    {
                        "reward_arm": arm_name,
                        "method": method,
                        "seed": int(seed),
                        "checkpoint_episode": 48,
                        "development_instances": 4,
                        "rollout_steps": len(rollout),
                        "return_mean": losses["return_mean"],
                        "return_std": losses["return_std"],
                        "value_prediction_mean": losses["value_prediction_mean"],
                        "advantage_raw_mean": losses["advantage_raw_mean"],
                        "advantage_raw_std": losses["advantage_raw_std"],
                        "policy_loss": losses["policy_loss"],
                        "weighted_value_loss": losses["weighted_value_loss"],
                        "weighted_auxiliary_loss": losses["weighted_auxiliary_loss"],
                        "policy_grad_norm": norms["policy"],
                        "value_grad_norm": norms["value"],
                        "auxiliary_grad_norm": norms["auxiliary"],
                        "total_grad_norm": norms["total"],
                        "global_clip_max_norm": float(agent._max_grad_norm),
                        "estimated_global_clip_scale": clip_scale,
                        "value_to_policy_grad_norm_ratio": norms["value"] / max(norms["policy"], 1e-12),
                        "scale_normalized_value_to_policy_grad_norm_ratio": norms["scale_normalized_value"]
                        / max(norms["policy"], 1e-12),
                        "encoder_value_to_policy_grad_norm_ratio": encoder_norms["value"]
                        / max(encoder_norms["policy"], 1e-12),
                        "encoder_scale_normalized_value_to_policy_grad_norm_ratio": encoder_norms[
                            "scale_normalized_value"
                        ]
                        / max(encoder_norms["policy"], 1e-12),
                        "policy_value_gradient_cosine": _cosine(vectors["policy"], vectors["value"]),
                        "policy_auxiliary_gradient_cosine": _cosine(
                            vectors["policy"], vectors["auxiliary"]
                        ),
                        "encoder_policy_auxiliary_gradient_cosine": _cosine(
                            encoder_vectors["policy"], encoder_vectors["auxiliary"]
                        ),
                        "current_missing_step_count": len(missing_rollout),
                        "current_missing_policy_auxiliary_gradient_cosine": (
                            missing_policy_auxiliary_cosine
                        ),
                    }
                )
    return rows, {
        "sample_rule": (
            "For every reward/method/seed cell, load the episode-48 checkpoint; run "
            "deterministic policy inference on all four dev instances in manifest order "
            "dev_00..dev_03; concatenate complete per-episode GAE rows; do not select by outcome."
        ),
        "parameter_updates": 0,
        "checkpoint_selection_dependency": "none; episode 48 is the first frozen candidate",
        "loss_reconstruction_boundary": (
            "Full-batch, ratio=1 offline PPO gradient at the frozen checkpoint. It is not an "
            "exact reproduction of original minibatch order, old policies, or optimizer steps."
        ),
    }


def _candidate_policy_trace(
    *,
    experiment: dict[str, Any],
    base_config: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Forward all candidates on states fixed by the first candidate's dev rollout."""

    dev_instances = [row for row in manifest["instances"] if row["split"] == "dev"]
    output: list[dict[str, Any]] = []
    for arm in experiment["reward_arms"]:
        arm_name = str(arm["arm"])
        config = _resolved_arm_config(base_config, experiment, str(arm["reward_profile"]))
        for method in experiment["training"]["agents"]:
            for seed in experiment["training"]["seeds"]:
                first = _build_agent(method, int(seed), config)
                first.load(
                    str(
                        checkpoint_root
                        / "checkpoints"
                        / f"{arm_name}_{method}_seed{seed}_episode48.pt"
                    )
                )
                first._deterministic_action = True
                fixed_rows: list[dict[str, Any]] = []
                for instance in dev_instances:
                    episode, _ = _collect_training_episode(first, config, instance, 24)
                    fixed_rows.extend(episode)
                for episode in (48, 96, 144, 192):
                    agent = _build_agent(method, int(seed), config)
                    agent.load(
                        str(
                            checkpoint_root
                            / "checkpoints"
                            / f"{arm_name}_{method}_seed{seed}_episode{episode}.pt"
                        )
                    )
                    probability_rows: list[dict[str, Any]] = []
                    for sample in fixed_rows:
                        semantic = agent._extract_semantic_state(sample["decision_info"])
                        metadata = dict(sample["decision_info"].get("run_metadata", {}) or {})
                        mask = agent._extract_action_mask(sample["decision_info"])
                        with torch.no_grad():
                            policy_output = agent._forward_policy(semantic, run_metadata=metadata)
                            scores = (
                                agent._hierarchical_env_action_scores(policy_output)
                                if agent._use_hierarchy
                                else policy_output["flat_logits"]
                            )
                            logits = agent._masked_flat_logits(scores, mask)
                            probabilities = torch.softmax(logits, dim=-1).tolist()
                        node = semantic.get("current_workflow_node") or {}
                        vehicle = (semantic.get("vehicles") or [{}])[0]
                        current_rsu_id = vehicle.get("associated_rsu_id")
                        current_ready = bool(
                            bundle_ready(
                                semantic,
                                rsu_by_id(semantic, current_rsu_id),
                                node,
                            )
                        )
                        probability_rows.append(
                            {
                                "current_ready": current_ready,
                                "probabilities": probabilities,
                                "argmax": int(np.argmax(probabilities)),
                            }
                        )
                    for state_group, selected in (
                        ("all", probability_rows),
                        (
                            "current_missing",
                            [row for row in probability_rows if not row["current_ready"]],
                        ),
                    ):
                        if not selected:
                            continue
                        output.append(
                            {
                                "reward_arm": arm_name,
                                "method": method,
                                "seed": int(seed),
                                "checkpoint_episode": episode,
                                "state_group": state_group,
                                "fixed_state_count": len(selected),
                                **{
                                    f"action_{action}_mean_probability": _mean(
                                        row["probabilities"][action] for row in selected
                                    )
                                    for action in range(5)
                                },
                                **{
                                    f"action_{action}_argmax_rate": _fraction(
                                        row["argmax"] == action for row in selected
                                    )
                                    for action in range(5)
                                },
                            }
                        )
    return output, {
        "state_sampling_rule": (
            "For each reward/method/seed cell, the first candidate (episode 48) runs "
            "deterministically over all four dev instances in manifest order. The resulting "
            "decision states are frozen. Episodes 48/96/144/192 are then forward-only scored "
            "on those identical states without environment evaluation or parameter updates."
        ),
        "probability_definition": (
            "masked executed-action distribution before deterministic evaluation guards"
        ),
        "parameter_updates": 0,
        "evaluation_outcomes_generated": 0,
    }


def _aggregate_diagnostic_matrix(
    *,
    experiment: dict[str, Any],
    training_summary: list[dict[str, Any]],
    training_curves: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curves: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in training_curves:
        curves[(row["reward_arm"], row["method"], int(row["seed"]))].append(row)
    replay: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        replay[(row["reward_arm"], row["method"], int(row["seed"]))].append(row)
    episodes: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        episodes[(row["reward_arm"], row["method"], int(row["seed"]))].append(row)
    gradients = {
        (row["reward_arm"], row["method"], int(row["seed"])): row
        for row in gradient_rows
    }
    output: list[dict[str, Any]] = []
    for summary in training_summary:
        key = (summary["reward_arm"], summary["method"], int(summary["seed"]))
        rows = replay[key]
        invalid = [row for row in rows if row["unwise_action4"]]
        stalled = [row for row in rows if row["consecutive_no_progress"]]
        update = dict(summary["last_update"] or {})
        grad = gradients[key]
        curve = curves[key]
        episode = episodes[key]
        output.append(
            {
                "reward_arm": key[0],
                "method": key[1],
                "seed": key[2],
                "training_episodes": int(summary["episodes"]),
                "training_steps": int(summary["actual_steps"]),
                "training_updates": int(summary["updates"]),
                "training_completion_frequency": _fraction(bool(row["workflow_completed"]) for row in curve),
                "training_truncation_frequency": _fraction(not bool(row["workflow_completed"]) for row in curve),
                "selected_episode": int(summary["selected_episode"]),
                "evaluation_steps_replayed": len(rows),
                "evaluation_truncated_episodes": sum(bool(row["truncated"]) for row in episode),
                "value_target_mean_offline": _mean(row["value_target"] for row in rows),
                "value_prediction_mean_offline": _mean(row["value_prediction"] for row in rows),
                "critic_error_rmse_offline": float(
                    np.sqrt(np.mean([float(row["critic_error"]) ** 2 for row in rows]))
                ),
                "advantage_raw_mean_offline": _mean(row["advantage_raw"] for row in rows),
                "advantage_raw_std_offline": _std(row["advantage_raw"] for row in rows),
                "advantage_normalized_p05_offline": _quantile(
                    (row["advantage_normalized_offline_cell"] for row in rows), 0.05
                ),
                "advantage_normalized_p95_offline": _quantile(
                    (row["advantage_normalized_offline_cell"] for row in rows), 0.95
                ),
                "unwise_action4_count": len(invalid),
                "unwise_action4_positive_raw_advantage_rate": _fraction(
                    float(row["advantage_raw"]) > 0.0 for row in invalid
                ),
                "unwise_action4_positive_normalized_advantage_rate": _fraction(
                    float(row["advantage_normalized_offline_cell"]) > 0.0 for row in invalid
                ),
                "unwise_action4_mean_raw_advantage": _mean(row["advantage_raw"] for row in invalid),
                "consecutive_no_progress_count": len(stalled),
                "consecutive_no_progress_positive_raw_advantage_rate": _fraction(
                    float(row["advantage_raw"]) > 0.0 for row in stalled
                ),
                "last_logged_policy_loss": (
                    float(update.get("env_action_ppo_loss", 0.0))
                    if bool(update.get("executed_action_ppo_only", False))
                    else float(update.get("actor_loss", 0.0))
                ),
                "last_logged_value_loss": float(update.get("value_loss", 0.0)),
                "last_logged_auxiliary_loss": float(update.get("auxiliary_loss", 0.0)),
                "last_logged_approx_kl": float(update.get("approx_kl", 0.0)),
                "last_logged_clip_fraction": float(update.get("clip_fraction", 0.0)),
                "last_logged_policy_entropy": float(update.get("policy_entropy", 0.0)),
                "last_logged_explained_variance": float(update.get("explained_variance", 0.0)),
                "dev48_value_to_policy_grad_norm_ratio": grad[
                    "value_to_policy_grad_norm_ratio"
                ],
                "dev48_encoder_value_to_policy_grad_norm_ratio": grad[
                    "encoder_value_to_policy_grad_norm_ratio"
                ],
                "dev48_estimated_global_clip_scale": grad["estimated_global_clip_scale"],
                "dev48_policy_auxiliary_gradient_cosine": grad[
                    "policy_auxiliary_gradient_cosine"
                ],
            }
        )
    output.sort(key=lambda row: (row["reward_arm"], METHOD_ORDER[row["method"]], row["seed"]))
    return output


def _behavior_groups(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, Any]] = [
        ("current_ready", lambda row: row["current_bundle_ready"]),
        ("current_missing", lambda row: not row["current_bundle_ready"]),
        ("target_prepare_feasible", lambda row: row["target_prepare_feasible"]),
        ("target_prepare_infeasible", lambda row: not row["target_prepare_feasible"]),
        ("before_deadline", lambda row: row["deadline_phase"] == "before"),
        ("after_deadline", lambda row: row["deadline_phase"] == "after"),
        ("progress", lambda row: row["progressed"]),
        ("no_progress", lambda row: not row["progressed"]),
        ("consecutive_no_progress", lambda row: row["consecutive_no_progress"]),
        ("unwise_action4", lambda row: row["unwise_action4"]),
    ]
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        by_cell[(row["reward_arm"], row["method"], row["seed"])].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(by_cell.items(), key=lambda item: (item[0][0], METHOD_ORDER[item[0][1]], item[0][2])):
        for group_name, predicate in groups:
            selected = [row for row in rows if predicate(row)]
            if not selected:
                continue
            action_counts = [sum(int(row["action"]) == action for row in selected) for action in range(5)]
            probabilities = [
                _mean(
                    row["env_action_probs"][action]
                    for row in selected
                    if len(row["env_action_probs"]) > action
                )
                for action in range(5)
            ]
            output.append(
                {
                    "reward_arm": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "state_group": group_name,
                    "step_count": len(selected),
                    **{f"action_{action}_choice_rate": action_counts[action] / len(selected) for action in range(5)},
                    **{f"action_{action}_mean_probability": probabilities[action] for action in range(5)},
                    "mean_raw_advantage": _mean(row["advantage_raw"] for row in selected),
                    "positive_raw_advantage_rate": _fraction(
                        float(row["advantage_raw"]) > 0.0 for row in selected
                    ),
                    "mean_normalized_advantage": _mean(
                        row["advantage_normalized_offline_cell"] for row in selected
                    ),
                    "service_failure_rate": _fraction(row["service_failed"] for row in selected),
                    "episode_completion_rate": _fraction(row["episode_completed"] for row in selected),
                }
            )
    return output


def _gae_sensitivity(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_cell[(row["reward_arm"], row["method"], int(row["seed"]))].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(
        by_cell.items(),
        key=lambda item: (item[0][0], METHOD_ORDER[item[0][1]], item[0][2]),
    ):
        cohorts = (
            ("all", rows),
            (
                "completed_after_deadline",
                [
                    row
                    for row in rows
                    if row["workflow_completed"] and not row["workflow_completed_on_time"]
                ],
            ),
        )
        for cohort, selected in cohorts:
            if not selected:
                continue
            configured = _mean(row["first_return_configured"] for row in selected)
            lambda_1 = _mean(row["first_return_lambda_1"] for row in selected)
            undiscounted = _mean(row["first_return_undiscounted"] for row in selected)
            no_bootstrap = _mean(row["first_return_no_bootstrap"] for row in selected)
            output.append(
                {
                    "reward_arm": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "cohort": cohort,
                    "episode_count": len(selected),
                    "first_return_configured_mean": configured,
                    "first_return_lambda_1_mean": lambda_1,
                    "first_return_undiscounted_mean": undiscounted,
                    "first_return_no_bootstrap_mean": no_bootstrap,
                    "lambda_1_minus_configured": lambda_1 - configured,
                    "undiscounted_minus_configured": undiscounted - configured,
                    "bootstrap_minus_no_bootstrap": configured - no_bootstrap,
                }
            )
    return output


def _checkpoint_curves(source_root: Path) -> list[dict[str, Any]]:
    records = _load_json(source_root / "checkpoint_selection.json")
    output: list[dict[str, Any]] = []
    for row in records:
        score = list(row["score"])
        output.append(
            {
                "reward_arm": row["reward_arm"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "dev_on_time_completion": float(score[0]),
                "dev_total_completion": float(score[1]),
                "dev_unfinished_after_deadline": -float(score[2]),
                "dev_failed_service": -float(score[3]),
                "dev_handoff_failure": -float(score[4]),
                "dev_completed_elapsed": -float(score[5]),
                "dev_transfer_mb": -float(score[6]),
                "dev_recompute_seconds": -float(score[7]),
                "dev_invalid_prepare": -float(score[8]),
            }
        )
    output.sort(key=lambda row: (row["reward_arm"], METHOD_ORDER[row["method"]], row["seed"], row["episode"]))
    return output


def _reward_contributions(source_root: Path) -> list[dict[str, Any]]:
    with (source_root / "reward_decomposition.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                key: (
                    value
                    if key in {"split", "reward_arm", "method"}
                    else int(value)
                    if key == "episode_n"
                    else float(value)
                )
                for key, value in row.items()
            }
        )
    return output


def _truncation_witness(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truncated = [row for row in episode_rows if row["truncated"]]
    configured_deltas = [float(row["bootstrap_delta_at_first_step"]) for row in truncated]
    return {
        "actual_semantics": {
            "workflow_complete": "terminated=true",
            "failed_service_attempt": "recoverable step event; it does not terminate the episode",
            "instance_horizon": (
                "env.step returns truncated=true when instance.max_steps is reached before completion"
            ),
            "runner_outer_cap": (
                "runner also stops at step_cap=24; all frozen instances have max_steps<=24"
            ),
            "bootstrap_observation": (
                "the final non-reset observation returned by the last env.step; reset is not called before evaluate_value"
            ),
            "gae_episode_boundary": "one PPORolloutBuffer is finalized per episode before rows are concatenated",
        },
        "evaluation_budget_alignment": {
            "training": "non-terminated truncations bootstrap continuation value",
            "checkpoint_selection_and_evaluation": "completion is zero when the same finite horizon expires",
            "verdict": "semantic objective mismatch, but low training incidence prevents assigning it as the primary cause",
        },
        "replayed_truncated_episode_count": len(truncated),
        "bootstrap_value_mean": _mean(row["last_observation_bootstrap_value"] for row in truncated),
        "bootstrap_first_step_target_delta_mean": _mean(configured_deltas),
        "bootstrap_first_step_target_delta_min": min(configured_deltas) if configured_deltas else None,
        "bootstrap_first_step_target_delta_max": max(configured_deltas) if configured_deltas else None,
        "systematic_overestimate_verdict": (
            "not established: selected critics are globally low versus completed targets; truncation bootstrap is positive, "
            "but the true beyond-budget continuation return is unobserved"
        ),
    }


def _minimal_witness(replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in replay_rows
        if row["reward_arm"] == "service_aligned_v1"
        and row["unwise_action4"]
        and row["episode_completed"]
    ]
    candidates.sort(
        key=lambda row: (
            METHOD_ORDER[row["method"]],
            row["seed"],
            SPLIT_ORDER[row["split"]],
            row["design_id"],
            row["step_index"],
        )
    )
    witness = dict(candidates[0])
    return {
        "selection_rule": (
            "Within service_aligned_v1, sort method=SA/MAPPO/PPO, then seed, split, "
            "design_id and step_index; among episodes that complete, select the first action 4 "
            "for which current bundle is missing or target preparation is infeasible. Advantage "
            "sign and magnitude are not selection keys."
        ),
        "witness": witness,
    }


def _source_field_availability() -> dict[str, Any]:
    return {
        "direct_logs": [
            "per-episode training reward/completion/steps/service_failures",
            "all candidate dev summaries and selected checkpoint hashes",
            "selected evaluation behavior ledger and reward components",
            "last update loss/KL/clip/entropy/explained-variance record per cell",
            "selected and candidate checkpoint network/optimizer states",
        ],
        "offline_recomputed": [
            "selected-ledger value predictions, GAE targets/advantages and gamma-lambda variants",
            "state-conditioned advantage/action-probability summaries",
            "episode-48 deterministic dev full-batch component gradient norms",
        ],
        "unrecoverable_for_exact_original_update": [
            "training transition/action ledger and per-step reward decomposition",
            "all 24 update records (only the last record was persisted)",
            "original minibatch permutations and transient old-policy snapshots",
            "pre-clip and post-clip gradient norms from the original optimizer steps",
        ],
    }


def _artifact_integrity(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.iterdir()):
        if path.name == "artifact_integrity.json" or not path.is_file():
            continue
        rows.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "service_reward_learning_diagnosis_integrity_v1",
            "run_id": RUN_ID,
            "files": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=f"artifacts/benchmarks/{SOURCE_RUN_ID}",
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--output-root",
        default=f"artifacts/analysis/{RUN_ID}",
    )
    args = parser.parse_args()

    source_root = (ROOT_DIR / args.source_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = (ROOT_DIR / args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)

    experiment = _load_json(
        ROOT_DIR / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json"
    )
    base_config, _ = _load_experiment_config(ROOT_DIR / experiment["base_config"])
    manifest = _load_json(ROOT_DIR / experiment["workload_manifest"])
    training_summary = _load_json(source_root / "training_summary.json")
    training_curves = _load_json(source_root / "training_curves.json")

    code_identity = _verify_code_identity(source_root)
    checkpoint_identity = _verify_checkpoints(
        source_root=source_root,
        checkpoint_root=checkpoint_root,
        training_summary=training_summary,
    )
    replay_rows, episode_rows, replay_receipt = _replay_evaluation_ledger(
        experiment=experiment,
        base_config=base_config,
        manifest=manifest,
        source_root=source_root,
        checkpoint_root=checkpoint_root,
        training_summary=training_summary,
    )
    if replay_receipt["transition_or_reward_mismatch_count"] != 0:
        raise RuntimeError("committed behavior ledger failed exact environment replay")
    gradient_rows, gradient_protocol = _gradient_probe(
        experiment=experiment,
        base_config=base_config,
        manifest=manifest,
        checkpoint_root=checkpoint_root,
    )
    candidate_policy_rows, candidate_policy_protocol = _candidate_policy_trace(
        experiment=experiment,
        base_config=base_config,
        manifest=manifest,
        checkpoint_root=checkpoint_root,
    )
    matrix = _aggregate_diagnostic_matrix(
        experiment=experiment,
        training_summary=training_summary,
        training_curves=training_curves,
        replay_rows=replay_rows,
        episode_rows=episode_rows,
        gradient_rows=gradient_rows,
    )
    behavior_groups = _behavior_groups(replay_rows)
    gae_sensitivity = _gae_sensitivity(episode_rows)
    checkpoint_curves = _checkpoint_curves(source_root)
    reward_contributions = _reward_contributions(source_root)

    _write_json(output_root / "code_identity.json", code_identity)
    _write_json(output_root / "checkpoint_identity.json", checkpoint_identity)
    _write_json(output_root / "replay_receipt.json", replay_receipt)
    _write_csv(output_root / "diagnostic_matrix.csv", matrix)
    _write_json(output_root / "diagnostic_matrix.json", matrix)
    _write_csv(output_root / "behavior_groups.csv", behavior_groups)
    _write_csv(output_root / "gae_sensitivity.csv", gae_sensitivity)
    _write_csv(output_root / "checkpoint_curves.csv", checkpoint_curves)
    _write_csv(output_root / "reward_contributions.csv", reward_contributions)
    _write_csv(output_root / "gradient_probe.csv", gradient_rows)
    _write_json(
        output_root / "gradient_probe_protocol.json",
        {**gradient_protocol, "rows": gradient_rows},
    )
    _write_csv(output_root / "candidate_policy_trace.csv", candidate_policy_rows)
    _write_json(
        output_root / "candidate_policy_trace_protocol.json",
        {**candidate_policy_protocol, "rows": candidate_policy_rows},
    )
    _write_json(output_root / "truncation_witness.json", _truncation_witness(episode_rows))
    _write_json(output_root / "minimal_witness.json", _minimal_witness(replay_rows))
    _write_json(output_root / "source_field_availability.json", _source_field_availability())
    _write_json(
        output_root / "diagnosis_receipt.json",
        {
            "schema_version": "service_reward_learning_diagnosis_v1",
            "run_id": RUN_ID,
            "source_run_id": SOURCE_RUN_ID,
            "baseline_commit": BASELINE_COMMIT,
            "execution_commit": EXECUTION_COMMIT,
            "read_only_model_analysis": True,
            "parameter_updates": 0,
            "training_runs": 0,
            "evaluation_metric_runs": 0,
            "holdout_reads": 0,
            "downloads": 0,
            "checkpoint_files_copied_or_uploaded": 0,
            "diagnostic_cell_count": len(matrix),
            "replayed_episode_count": len(episode_rows),
            "replayed_step_count": len(replay_rows),
            "gradient_probe_cell_count": len(gradient_rows),
            "candidate_policy_trace_row_count": len(candidate_policy_rows),
        },
    )
    (output_root / "command_log.txt").write_text(
        " ".join(
            [
                "python",
                "scripts/diagnose_calibrated_workflow_service_reward_learning.py",
                "--source-root",
                args.source_root,
                "--checkpoint-root",
                str(checkpoint_root),
                "--output-root",
                args.output_root,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _artifact_integrity(output_root)
    print(json.dumps({"run_id": RUN_ID, "status": "complete", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
