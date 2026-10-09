"""Read-only development diagnosis of the causal strong-baseline run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import (
    LEARNED_METHODS, _build_learned, _load_inputs, _load_json,
)
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv
from src.envs.core.causal_rsu_predictor import canonical_hash

SOURCE_COMMIT = "a08869388f9962149198054b5f8a0d6dd4d07eae"
RUN_ID = "cscwd_causal_strong_baselines_dev_20261009_v1"
SOURCE = ROOT / "artifacts/experiments" / RUN_ID
TERMINAL = ROOT / "artifacts/experiments" / f"{RUN_ID}_supervisor/terminal_receipt.json"
OUTPUT = ROOT / "artifacts/analysis/cscwd_sa_behavior_diagnosis_20261009_v1"
DESIGN = ROOT / "configs/experiment/calibrated_workflow_strong_baselines_development_v2_prefix_only.json"
PLAN = ROOT / "docs/project/cscwd_sa_behavior_diagnosis_plan_20261009.md"
METHODS = (*LEARNED_METHODS,)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty machine table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) not in {"True", "False"}:
        raise RuntimeError(f"invalid recorded boolean: {value}")
    return str(value) == "True"


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-6)


def _public_state_hash(observation: np.ndarray, info: dict[str, Any]) -> str:
    return canonical_hash({
        "observation": [float(item) for item in observation],
        "semantic_state": info["semantic_state"], "action_mask": info["action_mask"],
    })


def _load_sources() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    terminal = _load_json(TERMINAL)
    receipt = _load_json(SOURCE / "completion_receipt.json")
    run_manifest = _load_json(SOURCE / "run_manifest.json")
    if terminal.get("status") != "PASS" or receipt.get("scientific_execution_complete") is not True:
        raise RuntimeError("frozen scientific run is not complete")
    if run_manifest.get("git_commit") != SOURCE_COMMIT or run_manifest.get("interface_profile") != "calibrated_workflow_interface_v3_prefix_only":
        raise RuntimeError("scientific source commit/interface mismatch")
    if _sha256(DESIGN) != run_manifest["design_config"]["sha256"]:
        raise RuntimeError("frozen design hash mismatch")
    inventory = {item["path"]: item for item in _load_json(SOURCE / "artifact_integrity.json")["files"]}
    selected = (
        "run_manifest.json", "completion_receipt.json", "evaluation_rows.csv",
        "behavior_ledger.csv", "training_summary.json", "checkpoint_selection.json",
    )
    for name in selected:
        item = inventory[name]
        path = SOURCE / name
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"source artifact hash mismatch: {name}")
    eval_rows = _rows(SOURCE / "evaluation_rows.csv")
    ledger = _rows(SOURCE / "behavior_ledger.csv")
    training = _load_json(SOURCE / "training_summary.json")
    if len(eval_rows) != 440 or len(ledger) != 3308 or len(training) != 20:
        raise RuntimeError("frozen row/cell count mismatch")
    for cell in training:
        path = SOURCE / cell["selected_checkpoint"]
        item = inventory[cell["selected_checkpoint"]]
        if _sha256(path) != cell["selected_checkpoint_sha256"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch: {path.name}")
    if {row["method"] for row in eval_rows} != set(METHODS) | {"popularity_cache_heuristic", "two_step_cost_rule"}:
        raise RuntimeError("method identity mismatch")
    if len({(r["split"], r["method"], r["seed"], r["design_id"]) for r in eval_rows}) != 440:
        raise RuntimeError("duplicate evaluation identity")
    design = _load_json(DESIGN)
    config, splits, source_identity = _load_inputs(design)
    if source_identity["public_prefix_suffix_tamper_comparisons"] != 474:
        raise RuntimeError("causal prediction preflight drift")
    return config, splits, eval_rows, ledger, training, run_manifest


def _distance(current: str, sequence: list[str]) -> str:
    for index, candidate in enumerate(sequence, start=1):
        if candidate != current:
            return "1" if index == 1 else "2" if index == 2 else "3+"
    return "unknown"


def _replay(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], eval_rows: list[dict[str, str]], ledger: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = {row["design_id"]: row for name in ("regression", "frozen_check") for row in splits[name]}
    by_episode: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        by_episode[(row["split"], row["method"], row["seed"], row["design_id"])].append(row)
    eval_by_key = {(row["split"], row["method"], row["seed"], row["design_id"]): row for row in eval_rows}
    if set(by_episode) != set(eval_by_key):
        raise RuntimeError("behavior/evaluation episode identities differ")
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for key, records in sorted(by_episode.items()):
        split, method, seed, design_id = key
        env = CalibratedContinuousWorkflowEnv(config, instances[design_id])
        observation, info = env.reset()
        victim_history: set[tuple[str, str]] = set()
        prepared_targets: set[str] = set()
        episode_reloads = 0
        episode_reuses = 0
        first_failure: int | None = None
        for record in sorted(records, key=lambda item: int(item["step_index"])):
            index = int(record["step_index"])
            if index != env.step_index or env.terminated:
                raise RuntimeError(f"replay step drift: {key} {index}")
            semantic = info["semantic_state"]
            prediction = semantic["predictions"]["causal_provenance"]
            if prediction != json.loads(record["prediction_provenance"]):
                raise RuntimeError(f"prediction replay drift: {key} {index}")
            current = env._current_rsu_id()
            target = env._predicted_handoff_target()
            node = env._current_node()
            adapter = str(node["required_adapter"])
            current_ready = env._bundle_ready(current, adapter)
            target_ready = bool(target and env._bundle_ready(target, adapter))
            if current_ready != _bool(record["current_bundle_ready"]) or target_ready != _bool(record["target_bundle_ready"]):
                raise RuntimeError(f"cache readiness replay drift: {key} {index}")
            target_before = list(env.caches[target].residents) if target else []
            reuse = current_ready and current in prepared_targets
            if reuse:
                episode_reuses += 1
                prepared_targets.discard(current)
            before_nodes = len(env.completed)
            public_hash = _public_state_hash(observation, info)
            action = int(record["executed_action"])
            if not info["action_mask"][action]:
                raise RuntimeError(f"recorded action outside public mask: {key} {index}")
            observation, _, terminated, truncated, info = env.step(action)
            event = info["transition"]
            progressed = len(env.completed) > before_nodes
            for field in ("service_completed", "migration_success", "state_ready"):
                if bool(event[field]) != _bool(record[field]):
                    raise RuntimeError(f"{field} replay drift: {key} {index}")
            if progressed != _bool(record["progressed"]):
                raise RuntimeError(f"progress replay drift: {key} {index}")
            for field in ("model_transfer_bytes", "state_transfer_bytes"):
                if int(event[field]) != int(record[field]):
                    raise RuntimeError(f"{field} replay drift: {key} {index}")
            for field, source_field in (("recompute_seconds", "recompute_seconds"), ("clock_seconds_after", "clock_seconds_after")):
                if not _same_float(event[field], record[source_field]):
                    raise RuntimeError(f"{field} replay drift: {key} {index}")
            events = list(event["cache_events"])
            victims = [(str(item["rsu_id"]), str(victim)) for item in events for victim in item.get("victims", []) if item.get("committed")]
            admissions = [(str(item["rsu_id"]), str(admitted)) for item in events for admitted in item.get("admitted", []) if item.get("committed")]
            reloaded = [pair for pair in admissions if pair in victim_history]
            episode_reloads += len(reloaded)
            victim_history.update(victims)
            if action in (1, 4) and target and any(item.get("committed") and item.get("rsu_id") == target for item in events):
                prepared_targets.add(target)
            if not event["service_completed"] and first_failure is None:
                first_failure = index
            target_after = list(env.caches[target].residents) if target else []
            steps.append({
                "split": split, "method": method, "seed": seed, "design_id": design_id,
                "step_index": index, "public_state_sha256": public_hash,
                "current_ready": current_ready, "target_ready": target_ready,
                "target_prepare_feasible": _bool(record["target_prepare_feasible"]),
                "predicted_handoff_distance": _distance(current, prediction["sequence"]),
                "predicted_target": target, "executed_action": action,
                "raw_env_action": int(record["raw_env_action"]),
                "projected_env_action": int(record["projected_env_action"]),
                "projection_applied": _bool(record["projection_applied"]),
                "service_completed": bool(event["service_completed"]),
                "migration_prepare_committed": bool(event["migration_success"]),
                "target_residents_before": target_before,
                "target_residents_after": target_after,
                "target_resident_changed": target_before != target_after,
                "cache_victims": victims, "cache_admissions": admissions,
                "victim_reload_count": len(reloaded), "prepared_target_reused": reuse,
                "node_progressed": progressed,
                "no_progress_streak": int(record["no_progress_streak"]),
                "model_transfer_bytes": int(event["model_transfer_bytes"]),
                "state_transfer_bytes": int(event["state_transfer_bytes"]),
                "recompute_seconds": float(event["recompute_seconds"]),
            })
            if terminated or truncated:
                if index != max(int(item["step_index"]) for item in records):
                    raise RuntimeError(f"premature replay episode end: {key}")
                break
        evaluation = eval_by_key[key]
        summary = env.summary()
        for field in ("workflow_completion_rate", "on_time_workflow_completion_rate", "service_failures"):
            actual = summary["workflow_completed" if field == "workflow_completion_rate" else "on_time_workflow_completed" if field == "on_time_workflow_completion_rate" else field]
            if not _same_float(actual, evaluation[field]):
                raise RuntimeError(f"episode summary replay drift: {key} {field}")
        episodes.append({
            "split": split, "method": method, "seed": seed, "design_id": design_id,
            "steps": len(records), "completed": bool(summary["workflow_completed"]),
            "on_time": bool(summary["on_time_workflow_completed"]),
            "first_service_failure_step": first_failure,
            "max_no_progress_streak": max(int(item["no_progress_streak"]) for item in records),
            "no_progress_step_count": sum(not _bool(item["progressed"]) for item in records),
            "prepared_target_reuse_count": episode_reuses,
            "victim_reload_count": episode_reloads,
            "total_transfer_mb": float(evaluation["total_transfer_mb"]),
            "recompute_seconds": float(evaluation["recompute_seconds"]),
            "modeled_completion_seconds": float(evaluation["modeled_completion_seconds"]),
        })
    if len(steps) != 3308 or len(episodes) != 440:
        raise RuntimeError("replay row count drift")
    return steps, episodes


def _prediction_audit(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], ledger: list[dict[str, str]], run_manifest: dict[str, Any]) -> dict[str, Any]:
    instances = {row["design_id"]: row for name in ("regression", "frozen_check") for row in splits[name]}
    result = Counter()
    horizon = int(config["prediction_horizon"])
    full_hash = run_manifest["source_interval_validation"]["causal_predictor"]["full_model_sha256"]
    for row in ledger:
        instance = instances[row["design_id"]]
        index = int(row["step_index"])
        sequence = instance["rsu_sequence"]
        forecast = json.loads(row["prediction_provenance"])
        if forecast["predictor_sha256"] != full_hash or forecast["prefix_end_index"] != index or forecast["prefix_sha256"] != canonical_hash(sequence[: index + 1]):
            raise RuntimeError("prediction provenance mismatch")
        result["recorded_decisions"] += 1
        if index + 1 >= len(sequence):
            result["next_step_censored"] += 1
            continue
        current = sequence[index]
        actual_next = sequence[index + 1]
        actual_handoff = actual_next != current
        result["next_step_evaluable"] += 1
        result["actual_next_handoff" if actual_handoff else "actual_next_stay"] += 1
        if forecast["status"] == "unknown":
            result["unknown"] += 1
            continue
        result["known"] += 1
        predicted = list(forecast["sequence"])
        if not predicted:
            raise RuntimeError("known prediction has empty sequence")
        predicted_next = predicted[0]
        result["known_next_correct" if predicted_next == actual_next else "known_next_wrong"] += 1
        result["known_actual_handoff" if actual_handoff else "known_actual_stay"] += 1
        result["known_handoff_correct" if actual_handoff and predicted_next == actual_next else "known_stay_correct" if not actual_handoff and predicted_next == actual_next else "known_handoff_wrong" if actual_handoff else "known_stay_wrong"] += 1
        future = sequence[index + 1 : index + 1 + horizon]
        actual_first = next(((offset, item) for offset, item in enumerate(future, 1) if item != current), None)
        predicted_first = next(((offset, item) for offset, item in enumerate(predicted, 1) if item != current), None)
        if actual_first:
            result["actual_handoff_within_observed_horizon"] += 1
            if predicted_first:
                result["predicted_handoff_when_actual"] += 1
                result["handoff_target_correct" if predicted_first[1] == actual_first[1] else "handoff_target_wrong"] += 1
                result["handoff_time_correct" if predicted_first[0] == actual_first[0] else "handoff_time_wrong"] += 1
                result["target_and_time_correct" if predicted_first == actual_first else "target_or_time_wrong"] += 1
        if len(future) < horizon:
            result["horizon_censored"] += 1
    return {"counts": dict(sorted(result.items())), "known_coverage_evaluable": result["known"] / max(result["next_step_evaluable"], 1),
            "known_next_accuracy": result["known_next_correct"] / max(result["known"], 1),
            "known_stay_accuracy": result["known_stay_correct"] / max(result["known_actual_stay"], 1),
            "known_handoff_accuracy": result["known_handoff_correct"] / max(result["known_actual_handoff"], 1),
            "handoff_target_accuracy_when_predicted": result["handoff_target_correct"] / max(result["predicted_handoff_when_actual"], 1),
            "handoff_time_accuracy_when_predicted": result["handoff_time_correct"] / max(result["predicted_handoff_when_actual"], 1)}


def _first_divergences(steps: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    outcome = {(item["split"], item["method"], item["seed"], item["design_id"]): item for item in episodes}
    for item in steps:
        if item["method"] in METHODS:
            by_episode[(item["split"], item["method"], item["seed"], item["design_id"])].append(item)
    result = []
    for (split, method, seed, design_id), left in sorted(by_episode.items()):
        if method != "sa_ghmappo":
            continue
        for comparator in ("ppo", "mappo", "dt_handoff_drl"):
            right = by_episode[(split, comparator, seed, design_id)]
            pair = next(((a, b) for a, b in zip(left, right) if a["executed_action"] != b["executed_action"]), None)
            if pair is None:
                result.append({"split": split, "seed": seed, "design_id": design_id, "comparator": comparator,
                               "first_divergence_step": None, "same_public_state": None, "sa_action": None, "comparator_action": None,
                               "sa_completed": outcome[(split, method, seed, design_id)]["completed"],
                               "comparator_completed": outcome[(split, comparator, seed, design_id)]["completed"]})
                continue
            a, b = pair
            result.append({"split": split, "seed": seed, "design_id": design_id, "comparator": comparator,
                           "first_divergence_step": a["step_index"],
                           "same_public_state": a["public_state_sha256"] == b["public_state_sha256"],
                           "sa_action": a["executed_action"], "comparator_action": b["executed_action"],
                           "sa_current_ready": a["current_ready"], "sa_target_prepare_feasible": a["target_prepare_feasible"],
                           "sa_handoff_distance": a["predicted_handoff_distance"],
                           "sa_completed": outcome[(split, method, seed, design_id)]["completed"],
                           "comparator_completed": outcome[(split, comparator, seed, design_id)]["completed"]})
    if len(result) != 300:
        raise RuntimeError("first-divergence pair count drift")
    return result


def _state_candidates(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates = []
    for split in ("train", "dev"):
        for instance in splits[split]:
            for schedule in ("cache_first", "periodic_prepare"):
                env = CalibratedContinuousWorkflowEnv(config, instance)
                observation, info = env.reset()
                while not env.terminated and env.step_index < min(24, int(instance["max_steps"])):
                    current = env._current_rsu_id()
                    target = env._predicted_handoff_target()
                    node = env._current_node()
                    current_ready = env._bundle_ready(current, str(node["required_adapter"]))
                    target_ready = bool(target and env._bundle_ready(target, str(node["required_adapter"])))
                    feasible = False
                    if info["action_mask"][4]:
                        preview = env.clone_for_decision_model()
                        _, _, _, _, preview_info = preview.step(4)
                        cache_events = preview_info["transition"].get("cache_events", [])
                        feasible = bool(cache_events and cache_events[0].get("reason") != "contact_budget_exceeded")
                    distance = _distance(current, info["semantic_state"]["predictions"]["causal_provenance"]["sequence"])
                    bucket = (current_ready, feasible, distance, target_ready)
                    candidates.append({"split": split, "design_id": instance["design_id"], "schedule": schedule,
                                       "step_index": env.step_index, "bucket": bucket,
                                       "public_state_sha256": _public_state_hash(observation, info),
                                       "observation": observation.copy(), "info": deepcopy(info)})
                    if not current_ready:
                        action = 0
                    elif schedule == "periodic_prepare" and env.step_index % 3 == 0 and info["action_mask"][4]:
                        action = 4
                    else:
                        action = 3
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
    buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
    for item in sorted(candidates, key=lambda row: (row["split"], row["design_id"], row["schedule"], row["step_index"])):
        buckets[item["bucket"]].append(item)
    selected = []
    while len(selected) < 60 and any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < 60:
                selected.append(buckets[bucket].popleft())
    return selected


def _state_snapshot(agent: Any) -> dict[str, Any]:
    return {
        "network": deepcopy(agent._network.state_dict()),
        "optimizer": deepcopy(agent._optimizer.state_dict()),
        "popart": deepcopy(agent._popart.state_dict()) if agent._popart is not None else None,
        "learned_transition_model": deepcopy(agent._learned_transition_model.state_dict()) if agent._learned_transition_model is not None else None,
    }


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_deep_equal(a, b) for a, b in zip(left, right))
    return left == right


def _forward(config: dict[str, Any], selected: list[dict[str, Any]], training: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(selected) > 60:
        raise RuntimeError("common-state budget exceeded")
    rows = []
    audit = []
    for cell in training:
        method, seed = str(cell["method"]), int(cell["seed"])
        agent = _build_learned(method, seed, config, popart_enabled=False)
        path = SOURCE / cell["selected_checkpoint"]
        if _sha256(path) != cell["selected_checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed before forward")
        agent.load(str(path))
        agent._deterministic_action = True
        before = _state_snapshot(agent)
        for index, state in enumerate(selected):
            info = deepcopy(state["info"])
            action, details = agent.act(state["observation"], info)
            if not info["action_mask"][int(action)]:
                raise RuntimeError("forward action outside common mask")
            rows.append({
                "method": method, "seed": seed, "common_state_index": index,
                "split": state["split"], "design_id": state["design_id"],
                "schedule": state["schedule"], "step_index": state["step_index"],
                "bucket": list(state["bucket"]), "public_state_sha256": state["public_state_sha256"],
                "action_mask": list(info["action_mask"]),
                "env_action_probs": details.get("env_action_probs"),
                "raw_head_actions": details.get("raw_head_actions"),
                "raw_head_actions_source": details.get("raw_head_actions_source"),
                "raw_env_action": details.get("raw_env_action"),
                "projected_env_action": details.get("projected_env_action"),
                "projection_applied": details.get("action_projection_applied"),
                "final_env_action": int(action),
                "aggregation_reason": details.get("aggregation_reason"),
                "policy_evaluation_mode": details.get("policy_evaluation_mode"),
            })
        unchanged = _deep_equal(before, _state_snapshot(agent)) and _sha256(path) == cell["selected_checkpoint_sha256"]
        audit.append({"method": method, "seed": seed, "forward_calls": len(selected), "state_unchanged": unchanged,
                      "checkpoint_sha256": cell["selected_checkpoint_sha256"]})
        if not unchanged:
            raise RuntimeError(f"forward mutated network/optimizer/normalization: {method} {seed}")
    if len(rows) > 1200:
        raise RuntimeError("total forward budget exceeded")
    return rows, audit


def _aggregate(eval_rows: list[dict[str, str]], steps: list[dict[str, Any]], episodes: list[dict[str, Any]], divergences: list[dict[str, Any]], forward: list[dict[str, Any]]) -> dict[str, Any]:
    eval_by_key = {(row["split"], row["method"], row["seed"], row["design_id"]): row for row in eval_rows}
    by_cell = defaultdict(list)
    for item in episodes:
        if item["method"] in METHODS:
            by_cell[(item["method"], item["seed"])].append(item)
    seed_rows = []
    for (method, seed), group in sorted(by_cell.items()):
        assert len(group) == 20
        group_steps = [r for r in steps if r["method"] == method and r["seed"] == seed]
        failed = [r for r in group if not r["completed"]]
        seed_rows.append({
            "method": method, "seed": seed, "instances": 20,
            "completed": sum(r["completed"] for r in group), "on_time": sum(r["on_time"] for r in group),
            "failed_episodes": len(failed), "failure_episodes_first_service_failure": sum(r["first_service_failure_step"] is not None for r in failed),
            "mean_no_progress_steps": mean(r["no_progress_step_count"] for r in group),
            "max_no_progress_streak": max(r["max_no_progress_streak"] for r in group),
            "prepared_target_reuses": sum(r["prepared_target_reuse_count"] for r in group),
            "victim_reloads": sum(r["victim_reload_count"] for r in group),
            "action4_steps": sum(r["executed_action"] == 4 for r in group_steps),
            "action4_service_failure_with_target_change": sum(r["executed_action"] == 4 and not r["service_completed"] and r["target_resident_changed"] for r in group_steps),
            "action4_service_failure_without_migration": sum(r["executed_action"] == 4 and not r["service_completed"] and not r["migration_prepare_committed"] for r in group_steps),
            "projection_steps": sum(r["projection_applied"] for r in group_steps),
            "all_instance_mean_transfer_mb": mean(r["total_transfer_mb"] for r in group),
            "completed_only_mean_elapsed_seconds": mean(r["modeled_completion_seconds"] for r in group if r["completed"]) if any(r["completed"] for r in group) else None,
            "completed_elapsed_coverage": sum(r["completed"] for r in group) / 20,
        })
    paired = []
    for row in eval_rows:
        if row["method"] != "sa_ghmappo":
            continue
        for method in ("ppo", "mappo", "dt_handoff_drl"):
            peer = eval_by_key[(row["split"], method, row["seed"], row["design_id"])]
            both = float(row["workflow_completion_rate"]) == float(peer["workflow_completion_rate"]) == 1.0
            paired.append({
                "split": row["split"], "seed": row["seed"], "design_id": row["design_id"], "comparator": method,
                "sa_completed": float(row["workflow_completion_rate"]) == 1.0,
                "peer_completed": float(peer["workflow_completion_rate"]) == 1.0,
                "both_completed": both,
                "elapsed_seconds_sa_minus_peer": float(row["completed_sample_elapsed_seconds"]) - float(peer["completed_sample_elapsed_seconds"]) if both else None,
                "transfer_mb_sa_minus_peer_all_instances": float(row["total_transfer_mb"]) - float(peer["total_transfer_mb"]),
                "transfer_mb_sa_minus_peer_both_completed": float(row["total_transfer_mb"]) - float(peer["total_transfer_mb"]) if both else None,
                "recompute_seconds_sa_minus_peer_both_completed": float(row["recompute_seconds"]) - float(peer["recompute_seconds"]) if both else None,
            })
    pair_summary = []
    for method in ("ppo", "mappo", "dt_handoff_drl"):
        group = [r for r in paired if r["comparator"] == method]
        both = [r for r in group if r["both_completed"]]
        pair_summary.append({
            "comparator": method, "pairs": len(group), "sa_completed": sum(r["sa_completed"] for r in group),
            "peer_completed": sum(r["peer_completed"] for r in group), "both_completed": len(both),
            "both_completed_coverage": len(both) / len(group),
            "mean_elapsed_seconds_sa_minus_peer_both_completed": mean(r["elapsed_seconds_sa_minus_peer"] for r in both) if both else None,
            "mean_transfer_mb_sa_minus_peer_all_instances": mean(r["transfer_mb_sa_minus_peer_all_instances"] for r in group),
            "mean_transfer_mb_sa_minus_peer_both_completed": mean(r["transfer_mb_sa_minus_peer_both_completed"] for r in both) if both else None,
            "mean_recompute_seconds_sa_minus_peer_both_completed": mean(r["recompute_seconds_sa_minus_peer_both_completed"] for r in both) if both else None,
        })
    by_method = defaultdict(list)
    for r in steps:
        if r["method"] in METHODS:
            by_method[r["method"]].append(r)
    strata = []
    for method, rows in sorted(by_method.items()):
        bucketed = defaultdict(list)
        for r in rows:
            bucketed[(r["current_ready"], r["target_prepare_feasible"], r["predicted_handoff_distance"], r["target_resident_changed"], r["migration_prepare_committed"], r["node_progressed"])].append(r)
        for bucket, group in sorted(bucketed.items(), key=lambda item: str(item[0])):
            strata.append({"method": method, "current_ready": bucket[0], "target_prepare_feasible": bucket[1],
                           "handoff_distance": bucket[2], "target_resident_changed": bucket[3],
                           "migration_commit": bucket[4], "node_progressed": bucket[5],
                           "steps": len(group), "action4_steps": sum(r["executed_action"] == 4 for r in group),
                           "service_failures": sum(not r["service_completed"] for r in group),
                           "victim_reloads": sum(r["victim_reload_count"] for r in group)})
    forward_summary = []
    for method in METHODS:
        group = [r for r in forward if r["method"] == method]
        forward_summary.append({"method": method, "forward_rows": len(group),
                                "final_action_counts": dict(Counter(str(r["final_env_action"]) for r in group)),
                                "raw_action_counts": dict(Counter(str(r["raw_env_action"]) for r in group)),
                                "projection_count": sum(bool(r["projection_applied"]) for r in group)})
    return {"seed_rows": seed_rows, "paired_summary": pair_summary, "paired_rows": paired,
            "strata": strata, "first_divergence_rows": divergences,
            "first_divergence_summary": {method: {"pairs": sum(r["comparator"] == method for r in divergences),
                 "same_public_state_at_first_divergence": sum(r["comparator"] == method and r["same_public_state"] is True for r in divergences),
                 "no_divergence": sum(r["comparator"] == method and r["first_divergence_step"] is None for r in divergences)}
                 for method in ("ppo", "mappo", "dt_handoff_drl")},
            "forward_summary": forward_summary}


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"create-only diagnostic root exists: {OUTPUT}")
    config, splits, eval_rows, ledger, training, manifest = _load_sources()
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "diagnosis_identity.json", {"source_commit": SOURCE_COMMIT, "run_id": RUN_ID,
        "run_manifest_sha256": _sha256(SOURCE / "run_manifest.json"), "plan_sha256": _sha256(PLAN),
        "script_sha256": _sha256(Path(__file__)), "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(training), "original_evaluation_rows": len(eval_rows),
        "original_behavior_rows": len(ledger), "synthetic_step_counterexamples_budget": 12})
    replay, episodes = _replay(config, splits, eval_rows, ledger)
    prediction = _prediction_audit(config, splits, ledger, manifest)
    divergences = _first_divergences(replay, episodes)
    selected = _state_candidates(config, splits)
    forward, forward_audit = _forward(config, selected, training)
    aggregate = _aggregate(eval_rows, replay, episodes, divergences, forward)
    aggregate["prediction_audit"] = prediction
    aggregate["common_state_count"] = len(selected)
    aggregate["forward_calls"] = len(forward)
    aggregate["synthetic_step_counterexamples_used"] = 0
    aggregate["source_run_id"] = RUN_ID
    aggregate["source_commit"] = SOURCE_COMMIT
    aggregate["claim_boundary"] = "consumed_development_diagnostic_only_no_new_evaluation_no_paper_table"
    with (OUTPUT / "replay_step_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in replay:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(OUTPUT / "episode_rows.json", episodes)
    _write_json(OUTPUT / "common_state_selection.json", [{key: value for key, value in row.items() if key not in {"observation", "info"}} for row in selected])
    _write_json(OUTPUT / "common_state_forward.json", forward)
    _write_json(OUTPUT / "forward_immutability_audit.json", forward_audit)
    _write_json(OUTPUT / "machine_summary.json", aggregate)
    _write_csv(OUTPUT / "seed_summary.csv", aggregate["seed_rows"])
    _write_csv(OUTPUT / "paired_summary.csv", aggregate["paired_summary"])
    _write_csv(OUTPUT / "mechanism_strata.csv", aggregate["strata"])
    _write_json(OUTPUT / "completion_receipt.json", {"status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
        "replayed_steps": len(replay), "replayed_episodes": len(episodes), "common_states": len(selected),
        "forward_calls": len(forward), "checkpoint_states_unchanged": all(r["state_unchanged"] for r in forward_audit),
        "synthetic_step_counterexamples_used": 0, "training_steps": 0, "formal_or_holdout_reads": 0})
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "artifact_integrity.json")
    _write_json(OUTPUT / "artifact_integrity.json", {"files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files]})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise RuntimeError("select exactly one of --preflight or --run")
    if args.preflight:
        _, _, eval_rows, ledger, training, manifest = _load_sources()
        print(json.dumps({"status": "preflight_only", "source_commit": manifest["git_commit"],
                          "evaluation_rows": len(eval_rows), "behavior_rows": len(ledger),
                          "selected_checkpoints": len(training), "forward_calls": 0}))
    else:
        summary = run()
        print(json.dumps({"status": "complete", "output": str(OUTPUT),
                          "common_states": summary["common_state_count"], "forward_calls": summary["forward_calls"]}))


if __name__ == "__main__":
    main()
