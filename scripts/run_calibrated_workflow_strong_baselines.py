"""Prepare a guarded, common development path for learned and rule baselines.

The checked-in protocol is not authorized for a scientific run. Unit tests use
synthetic instances through the small cell helpers below.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_reward_alignment import (  # noqa: E402
    _seed_rows,
    _summary_rows,
    _write_csv,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _build_agent,
    _evaluate_agent,
    _evaluate_rule,
    _git_commit,
    _integrity,
    _sha256,
    _write_json,
)
from scripts.run_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    _collect_exact_update_batch,
    _selection_score,
    _training_signal_row,
    _validate_intervals,
)
from src.agents.registry import ALGO_REGISTRY, build_agent  # noqa: E402
from src.envs.core.causal_rsu_predictor import canonical_hash, fit_predictor  # noqa: E402
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv  # noqa: E402


DEFAULT_CONFIG = (
    "configs/experiment/calibrated_workflow_strong_baselines_development_v1.json"
)
LEARNED_METHODS = ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl")
HEURISTIC_METHOD = "popularity_cache_heuristic"
MODEL_BASED_METHOD = "two_step_cost_rule"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class PublicContractAgent:
    """Check the shared observation/mask boundary without changing a policy."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def _deterministic_action(self) -> bool:
        return bool(getattr(self.agent, "_deterministic_action", False))

    @_deterministic_action.setter
    def _deterministic_action(self, value: bool) -> None:
        self.agent._deterministic_action = bool(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        if info is None or not isinstance(info.get("semantic_state"), dict):
            raise RuntimeError("public semantic_state is missing")
        mask = info.get("action_mask")
        if not isinstance(mask, (list, tuple)) or len(mask) != 5:
            raise RuntimeError("public five-action mask is missing")
        if len(observation) != 9:
            raise RuntimeError("public nine-value observation contract drift")
        action, action_info = self.agent.act(observation, info)
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < 5 or not mask[int(action)]:
            raise RuntimeError("policy returned an action outside the public mask")
        if not isinstance(action_info, dict):
            raise RuntimeError("policy action_info is missing")
        if getattr(self.agent, "support_level", "") == "heuristic":
            action_info = dict(action_info)
            action_info.setdefault("aggregation_reason", action_info.get("heuristic_reason", ""))
            action_info.setdefault("raw_head_actions_source", "public_state_heuristic")
            action_info.setdefault("raw_env_action", int(action))
            action_info.setdefault("projected_env_action", int(action))
            action_info.setdefault("final_env_action", int(action))
            action_info.setdefault("action_projection_applied", False)
        return int(action), action_info


def _build_learned(method: str, seed: int, config: dict[str, Any], *, popart_enabled: bool) -> PublicContractAgent:
    if method not in LEARNED_METHODS:
        raise ValueError(f"unsupported learned method: {method}")
    agent = _build_agent(
        method,
        seed,
        config,
        agent_overrides={"value_normalization_enabled": popart_enabled},
    )
    if not hasattr(agent, "learn") or not hasattr(agent, "save") or not hasattr(agent, "load"):
        raise RuntimeError(f"learned method lacks training/checkpoint contract: {method}")
    return PublicContractAgent(agent)


def _source_fields(instance: dict[str, Any]) -> dict[str, Any]:
    interval = instance["source_interval"]
    return {
        "source_segment_id": str(interval["source_segment_id"]),
        "source_frame_offset": int(interval["frame_offset"]),
        "source_window_length": int(interval["window_length"]),
        "source_time_start": int(interval["time_index_start"]),
        "source_time_end": int(interval["time_index_end"]),
        "window_id": str(instance["window_id"]),
        "workflow_id": str(instance["workflow_id"]),
    }


def _annotate_rows(
    rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    reward_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_design = {str(instance["design_id"]): instance for instance in instances}
    for item in (*rows, *ledger):
        source = _source_fields(by_design[str(item["design_id"])])
        item.update(source)
        item["reward_arm"] = reward_profile
    return rows, ledger


def _evaluate_popularity(
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    step_cap: int,
    reward_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for instance in instances:
        # Its adapter counter is episode-local. Reusing an object would leak
        # development history into the next independent window.
        agent = PublicContractAgent(build_agent(HEURISTIC_METHOD))
        if agent.agent._adapter_counts:
            raise RuntimeError("popularity memory was not reset")
        one_rows, one_ledger = _evaluate_agent(
            agent, HEURISTIC_METHOD, None, config, [instance], step_cap
        )
        one_rows[0]["heuristic_memory_scope"] = "one_instance_fresh_agent"
        annotated, annotated_ledger = _annotate_rows(
            one_rows, one_ledger, [instance], reward_profile
        )
        rows.extend(annotated)
        ledger.extend(annotated_ledger)
    return rows, ledger


def _run_learned_cell(
    *,
    method: str,
    seed: int,
    config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
    checkpoints: Path,
    reward_profile: str,
    popart_enabled: bool,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = _build_learned(method, seed, config, popart_enabled=popart_enabled)
    parameter_count = sum(int(parameter.numel()) for parameter in agent._network.parameters())
    order = list(range(len(splits["train"])))
    if not order or not splits["dev"]:
        raise RuntimeError("train and dev instances are required")
    rng = random.Random(seed)
    state: dict[str, Any] | None = None
    episodes_started = 0
    updates: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    training_episodes: list[dict[str, Any]] = []
    training_signals: list[dict[str, Any]] = []
    environment_steps = 0
    started_at = time.monotonic()
    transition_count = int(protocol["transitions_per_update"])
    update_count = int(protocol["update_opportunities_per_method_seed"])
    optimizer_steps_per_update = int(protocol["ppo_epochs_per_update"]) * math.ceil(
        transition_count / int(protocol["minibatch_size"])
    )
    checkpoints.mkdir(parents=True, exist_ok=True)
    for update_index in range(1, update_count + 1):
        batch, state, episodes_started, episode_rows = _collect_exact_update_batch(
            agent=agent,
            config=config,
            train_instances=splits["train"],
            order=order,
            rng=rng,
            state=state,
            episodes_started=episodes_started,
            transition_count=transition_count,
            step_cap=int(protocol["episode_max_steps"]),
            gamma=float(config["training"]["gamma"]),
            gae_lambda=float(config["training"]["gae_lambda"]),
        )
        environment_steps += len(batch)
        if config.get("interface_profile") == "calibrated_workflow_interface_v3_prefix_only":
            training_signals.extend(
                {**_training_signal_row(
                    arm="original_reward_v1", method=method, seed=seed,
                    update_index=update_index,
                    global_step=environment_steps - len(batch) + offset,
                    row=row,
                ), "prediction_provenance": json.dumps(
                    row["decision_info"]["semantic_state"]["predictions"]["causal_provenance"],
                    sort_keys=True,
                )}
                for offset, row in enumerate(batch, start=1)
            )
        train_by_design = {str(instance["design_id"]): instance for instance in splits["train"]}
        training_episodes.extend(
            {"method": method, "seed": seed, **row, **_source_fields(train_by_design[str(row["design_id"])])}
            for row in episode_rows
        )
        result = agent.learn(batch)
        if result.get("policy_update_skipped") or int(result.get("optimizer_step_count", -1)) != optimizer_steps_per_update:
            raise RuntimeError(f"{method} optimizer-step contract differs from frozen protocol")
        records = list(result.get("optimizer_step_records", []))
        if len(records) != optimizer_steps_per_update:
            raise RuntimeError("optimizer receipts are incomplete")
        optimizer_rows.extend({"method": method, "seed": seed, "update_index": update_index, **row} for row in records)
        updates.append({
            "method": method,
            "seed": seed,
            "update_index": update_index,
            "environment_steps": environment_steps,
            "optimizer_step_count": len(records),
            "value_normalization_enabled": bool(result.get("value_normalization_enabled", False)),
            "value_loss": float(result.get("value_loss", 0.0)),
        })
        if update_index in protocol["checkpoint_update_candidates"]:
            path = checkpoints / f"{method}_seed{seed}_update{update_index}.pt"
            agent.save(str(path))
            dev_rows, _ = _evaluate_agent(
                agent, method, seed, config, splits["dev"], int(protocol["episode_max_steps"])
            )
            candidate_rows.append({
                "method": method,
                "seed": seed,
                "update_index": update_index,
                "score": list(_selection_score(dev_rows)),
                "checkpoint": str(Path(checkpoints.name) / path.name),
                "checkpoint_sha256": _sha256(path),
                "dev_rows": _annotate_rows(dev_rows, [], splits["dev"], reward_profile)[0],
                "reward_used_for_selection": False,
                "evaluation_split_used_for_selection": False,
            })
    expected_steps = int(protocol["environment_steps_per_method_seed"])
    if environment_steps != expected_steps:
        raise RuntimeError("environment-step budget drift")
    if sum(row["optimizer_step_count"] for row in updates) != int(protocol["expected_optimizer_steps_per_method_seed"]):
        raise RuntimeError("optimizer-step budget drift")
    if not candidate_rows:
        raise RuntimeError("no development checkpoint candidates")
    best = max(candidate_rows, key=lambda row: (tuple(row["score"]), -int(row["update_index"])))
    agent.load(str(checkpoints.parent / best["checkpoint"]))
    selected = checkpoints / f"{method}_seed{seed}_selected.pt"
    agent.save(str(selected))
    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for split in protocol["evaluation_splits"]:
        rows, ledger = _evaluate_agent(
            agent, method, seed, config, splits[split], int(protocol["episode_max_steps"])
        )
        rows, ledger = _annotate_rows(rows, ledger, splits[split], reward_profile)
        evaluation_rows.extend(rows)
        behavior_rows.extend(ledger)
    return {
        "summary": {
            "method": method,
            "seed": seed,
            "support_level": "trainable",
            "parameter_count": parameter_count,
            "environment_steps": environment_steps,
            "update_opportunities": len(updates),
            "optimizer_steps": len(optimizer_rows),
            "episodes_started": episodes_started,
            "episodes_completed_or_truncated": len(training_episodes),
            "active_partial_episode_at_budget": state is not None,
            "selected_update": int(best["update_index"]),
            "selected_checkpoint": str(Path(checkpoints.name) / selected.name),
            "selected_checkpoint_sha256": _sha256(selected),
            "elapsed_seconds": time.monotonic() - started_at,
            "value_normalization_enabled": popart_enabled,
        },
        "updates": updates,
        "optimizer_rows": optimizer_rows,
        "candidates": candidate_rows,
        "training_episodes": training_episodes,
        "training_signals": training_signals,
        "evaluation_rows": evaluation_rows,
        "behavior_rows": behavior_rows,
    }


def _validate_protocol(design: dict[str, Any]) -> None:
    protocol = design["training"]
    version = design["schema_version"]
    if version not in {"calibrated_workflow_strong_baselines_development_v1", "calibrated_workflow_strong_baselines_development_v2_prefix_only"}:
        raise RuntimeError("baseline protocol version drift")
    if design["scope"] != "consumed_36_instance_development_only":
        raise RuntimeError("development-only scope drift")
    if design["seeds"] != [7, 17, 29, 43, 61]:
        raise RuntimeError("learned seed identity drift")
    if tuple(design["learned_methods"]) != LEARNED_METHODS:
        raise RuntimeError("learned-method identity drift")
    if design["heuristic_methods"] != [HEURISTIC_METHOD] or design["model_based_methods"] != [MODEL_BASED_METHOD]:
        raise RuntimeError("rule-method identity drift")
    if design["reward_profile"] not in {"original_reward_v1", "service_aligned_v1"}:
        raise RuntimeError("unexpected reward profile")
    if design["critic_target_normalization"] not in {"raw_disabled", "popart_running_mean_std"}:
        raise RuntimeError("unexpected normalization profile")
    if design["critic_target_normalization"] != "raw_disabled" and not design["claim_boundary"]["popart_promoted"]:
        raise RuntimeError("PopArt requires a separately frozen promotion decision")
    if version.endswith("v2_prefix_only"):
        if design["reward_profile"] != "original_reward_v1" or design["critic_target_normalization"] != "raw_disabled":
            raise RuntimeError("prefix-only development arm is frozen to original reward and raw critic")
        predictor = design["causal_predictor"]
        if predictor["schema_version"] != "prefix_transition_counts_v1" or predictor["training_fit"] != "train_only_leave_one_instance_out_for_train":
            raise RuntimeError("causal predictor protocol drift")
        if design["workload_manifest_sha256"] != "b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc":
            raise RuntimeError("source manifest identity drift")
    if int(protocol["environment_steps_per_method_seed"]) != int(protocol["transitions_per_update"]) * int(protocol["update_opportunities_per_method_seed"]):
        raise RuntimeError("step/update budget mismatch")
    if (
        int(protocol["environment_steps_per_method_seed"]) != 1440
        or int(protocol["transitions_per_update"]) != 60
        or int(protocol["update_opportunities_per_method_seed"]) != 24
        or int(protocol["ppo_epochs_per_update"]) != 4
        or int(protocol["minibatch_size"]) != 32
        or int(protocol["episode_max_steps"]) != 24
    ):
        raise RuntimeError("development budget identity drift")
    expected_optimizer = int(protocol["update_opportunities_per_method_seed"]) * int(protocol["ppo_epochs_per_update"]) * math.ceil(int(protocol["transitions_per_update"]) / int(protocol["minibatch_size"]))
    if expected_optimizer != int(protocol["expected_optimizer_steps_per_method_seed"]):
        raise RuntimeError("optimizer-step budget mismatch")
    candidates = list(protocol["checkpoint_update_candidates"])
    if candidates != [6, 12, 18, 24]:
        raise RuntimeError("invalid checkpoint opportunities")
    if int(protocol["hyperparameter_search_trials_per_method"]) != 0 or bool(protocol["automatic_retry"]):
        raise RuntimeError("search/retry is forbidden")
    if protocol["evaluation_splits"] != ["regression", "frozen_check"]:
        raise RuntimeError("development evaluation split drift")
    claim = design["claim_boundary"]
    if claim["formal"] or claim["holdout"] or claim["independent_test"] or int(claim["old_holdout_reads"]) != 0:
        raise RuntimeError("development claim boundary drift")
    for method in (*LEARNED_METHODS, HEURISTIC_METHOD):
        expected_support = "heuristic" if method == HEURISTIC_METHOD else "trainable"
        if ALGO_REGISTRY[method]["support_level"] != expected_support:
            raise RuntimeError(f"registry support-level drift: {method}")


def _audit_public_prefix_invariance(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> int:
    comparisons = 0
    for rows in splits.values():
        for instance in rows:
            sequence = list(instance["rsu_sequence"])
            for index in range(min(len(sequence), int(instance["max_steps"]))):
                altered = deepcopy(instance)
                alternative = next((item for item in instance["rsu_ids"] if item != sequence[index]), sequence[index])
                altered["rsu_sequence"][index + 1 :] = [alternative] * (len(sequence) - index - 1)
                original_env = CalibratedContinuousWorkflowEnv(config, instance)
                altered_env = CalibratedContinuousWorkflowEnv(config, altered)
                original_env.step_index = altered_env.step_index = index
                original_observation, altered_observation = original_env._observation(), altered_env._observation()
                original_info, altered_info = original_env._info(), altered_env._info()
                if not np.array_equal(original_observation, altered_observation) or original_info != altered_info:
                    raise RuntimeError(f"PREDICTION_PREFIX_PERMISSION_BLOCKER: {instance['design_id']} at step {index}")
                comparisons += 1
    return comparisons


def _load_inputs(design: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_path = ROOT_DIR / design["base_config"]
    manifest_path = ROOT_DIR / design["workload_manifest"]
    causal = design["schema_version"].endswith("v2_prefix_only")
    if causal and _sha256(manifest_path) != design["workload_manifest_sha256"]:
        raise RuntimeError("causal source manifest hash mismatch")
    config, _ = _load_experiment_config(base_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(base_path) or manifest["resolved_config_sha256"] != _canonical_sha256(config):
        raise RuntimeError("frozen workload/base identity mismatch")
    identity = _validate_intervals(list(manifest["instances"]))
    if not identity["all_identity_fields_present"] or not identity["all_intervals_pairwise_disjoint"]:
        raise RuntimeError("source interval identity/overlap gate failed")
    splits = {
        name: [row for row in manifest["instances"] if row["split"] == name]
        for name in ("train", "dev", "regression", "frozen_check")
    }
    if {name: len(rows) for name, rows in splits.items()} != {"train": 12, "dev": 4, "regression": 12, "frozen_check": 8}:
        raise RuntimeError("consumed development split identity drift")
    missing_forecast = [
        str(row["design_id"])
        for row in manifest["instances"]
        if "predicted_rsu_sequence" not in row
    ]
    if missing_forecast and not causal:
        raise RuntimeError(
            "PREDICTION_FUTURE_LEAK_BLOCKER: "
            f"{len(missing_forecast)}/{len(manifest['instances'])} instances omit "
            "predicted_rsu_sequence; the environment then exposes future "
            "actual rsu_sequence as public predictions/contact budget. "
            "A separately audited causal forecast contract is required."
        )
    if causal:
        predictor = design["causal_predictor"]
        ordered_train = sorted(splits["train"], key=lambda row: str(row["design_id"]))
        source = [{"design_id": row["design_id"], "rsu_sequence": row["rsu_sequence"]} for row in ordered_train]
        if canonical_hash(source) != predictor["train_source_projection_sha256"]:
            raise RuntimeError("train-only predictor source identity mismatch")
        full = fit_predictor([row["rsu_sequence"] for row in ordered_train])
        if full["sha256"] != predictor["full_model_sha256"]:
            raise RuntimeError("causal predictor fitted hash mismatch")
        loo = {
            str(row["design_id"]): fit_predictor([other["rsu_sequence"] for other in ordered_train if other["design_id"] != row["design_id"]])
            for row in ordered_train
        }
        bundle_hash = canonical_hash({"full": full["sha256"], "loo": {key: value["sha256"] for key, value in loo.items()}})
        if bundle_hash != predictor["model_bundle_sha256"]:
            raise RuntimeError("causal predictor leave-one-out bundle hash mismatch")
        splits = {
            name: [{**deepcopy(row), "causal_predictor_model": loo[str(row["design_id"])] if name == "train" else full} for row in rows]
            for name, rows in splits.items()
        }
        config["interface_profile"] = "calibrated_workflow_interface_v3_prefix_only"
        identity["causal_predictor"] = {
            "train_source_projection_sha256": predictor["train_source_projection_sha256"],
            "full_model_sha256": full["sha256"],
            "model_bundle_sha256": bundle_hash,
            "train_instance_predictions": "leave_one_instance_out",
            "dev_and_evaluation_predictions": "train_only_full_fit",
        }
        identity["public_prefix_suffix_tamper_comparisons"] = _audit_public_prefix_invariance(config, splits)
    if design["reward_profile"] == "service_aligned_v1":
        reward = _load_json(ROOT_DIR / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json")["service_aligned_reward"]
        config["objective"].setdefault("reward_profiles", {})["service_aligned_v1"] = reward
    config["reward_profile"] = design["reward_profile"]
    return config, splits, identity


def _execute(design: dict[str, Any], output_root: Path, *, design_path: Path, command: list[str]) -> None:
    _validate_protocol(design)
    if not design["execution_authorized"] or not design["claim_boundary"]["scientific_execution_authorized"]:
        raise RuntimeError("scientific development execution is not authorized")
    config, splits, identity = _load_inputs(design)
    protocol = design["training"]
    profile = design["reward_profile"]
    popart_enabled = design["critic_target_normalization"] == "popart_running_mean_std"
    checkpoints = output_root / "checkpoints"
    cell_results = []
    for method in LEARNED_METHODS:
        for seed in design["seeds"]:
            cell_results.append(_run_learned_cell(
                method=method,
                seed=int(seed),
                config=config,
                splits=splits,
                protocol=protocol,
                checkpoints=checkpoints,
                reward_profile=profile,
                popart_enabled=popart_enabled,
            ))
    evaluation_rows = [row for cell in cell_results for row in cell["evaluation_rows"]]
    behavior_rows = [row for cell in cell_results for row in cell["behavior_rows"]]
    for split in protocol["evaluation_splits"]:
        heuristic_rows, heuristic_ledger = _evaluate_popularity(
            config, splits[split], int(protocol["episode_max_steps"]), profile
        )
        evaluation_rows.extend(heuristic_rows)
        behavior_rows.extend(heuristic_ledger)
        rule_rows, rule_ledger = _evaluate_rule(
            config, splits[split], int(protocol["episode_max_steps"])
        )
        rule_rows, rule_ledger = _annotate_rows(rule_rows, rule_ledger, splits[split], profile)
        evaluation_rows.extend(rule_rows)
        behavior_rows.extend(rule_ledger)
    if len({(row["split"], row["method"], row["seed"], row["design_id"]) for row in evaluation_rows}) != len(evaluation_rows):
        raise RuntimeError("duplicate evaluation row identity")
    if not evaluation_rows or not behavior_rows:
        raise RuntimeError("empty evaluation or behavior artifact")
    _write_json(output_root / "training_summary.json", [cell["summary"] for cell in cell_results])
    _write_json(output_root / "update_records.json", [row for cell in cell_results for row in cell["updates"]])
    _write_csv(output_root / "optimizer_step_records.csv", [row for cell in cell_results for row in cell["optimizer_rows"]])
    _write_json(output_root / "checkpoint_selection.json", [row for cell in cell_results for row in cell["candidates"]])
    _write_json(output_root / "training_episode_rows.json", [row for cell in cell_results for row in cell["training_episodes"]])
    if design["schema_version"].endswith("v2_prefix_only"):
        _write_csv(output_root / "training_signal_rows.csv", [row for cell in cell_results for row in cell["training_signals"]])
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output_root / "behavior_ledger.csv", behavior_rows)
    _write_csv(output_root / "service_metric_summary.csv", _summary_rows(evaluation_rows))
    _write_csv(output_root / "seed_summary.csv", _seed_rows(evaluation_rows))
    _write_json(output_root / "run_manifest.json", {
        "schema_version": "calibrated_workflow_strong_baselines_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "design_config": {"path": str(design_path.relative_to(ROOT_DIR)), "sha256": _sha256(design_path)},
        "base_config": {"path": design["base_config"], "sha256": _sha256(ROOT_DIR / design["base_config"])},
        "workload_manifest": {"path": design["workload_manifest"], "sha256": _sha256(ROOT_DIR / design["workload_manifest"])},
        "source_interval_validation": identity,
        "reward_profile": profile,
        "critic_target_normalization": design["critic_target_normalization"],
        "learned_methods": list(LEARNED_METHODS),
        "heuristic_methods": [HEURISTIC_METHOD],
        "model_based_methods": [MODEL_BASED_METHOD],
        "seeds": design["seeds"],
        "training_budget": protocol,
        "capability_labels": design["capability_labels"],
        "heuristic_memory_scope": "fresh_per_instance_no_checkpoint_no_training_seed",
        "claim_boundary": design["claim_boundary"],
        "interface_profile": config["interface_profile"],
        "command": command,
    })
    _write_json(output_root / "completion_receipt.json", {
        "status": "complete",
        "learned_cells": len(cell_results),
        "learned_environment_steps": sum(cell["summary"]["environment_steps"] for cell in cell_results),
        "learned_optimizer_steps": sum(cell["summary"]["optimizer_steps"] for cell in cell_results),
        "heuristic_training_steps": 0,
        "heuristic_checkpoint_count": 0,
        "evaluation_rows": len(evaluation_rows),
        "behavior_rows": len(behavior_rows),
        "formal_or_holdout_reads": 0,
        "scientific_execution_complete": True,
    })
    _integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output_root")
    parser.add_argument("--expected_git_commit")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise RuntimeError("select exactly one of --preflight or --run")
    design_path = (ROOT_DIR / args.config).resolve()
    design = _load_json(design_path)
    _validate_protocol(design)
    _, splits, identity = _load_inputs(design)
    if args.preflight:
        print(json.dumps({
            "status": "preflight_only",
            "execution_authorized": bool(design["execution_authorized"]),
            "split_counts": {key: len(value) for key, value in splits.items()},
            "source_identity": identity,
            "scientific_steps": 0,
        }))
        return
    if not design["execution_authorized"] or not design["claim_boundary"]["scientific_execution_authorized"]:
        raise RuntimeError("scientific development execution is not authorized")
    if not args.output_root or not args.expected_git_commit:
        raise RuntimeError("run requires output root and frozen Git commit")
    if args.expected_git_commit != _git_commit():
        raise RuntimeError("frozen Git commit mismatch")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError("scientific run requires a clean checkout")
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    started = time.monotonic()
    _write_json(output_root / "runner_entered.json", {
        "status": "entered", "entered_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "design_sha256": _sha256(design_path),
        "scientific_steps_at_entry": 0,
    })
    _write_json(output_root / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        _execute(design, output_root, design_path=design_path, command=list(sys.argv))
    except Exception as exc:
        _write_json(output_root / "failure_receipt.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started, "automatic_retry": False,
        })
        _write_json(output_root / "run_status.json", {"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat()})
        _integrity(output_root)
        raise
    _write_json(output_root / "run_status.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    _integrity(output_root)
    print(json.dumps({"status": "complete", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
