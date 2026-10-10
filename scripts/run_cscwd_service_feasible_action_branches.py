"""Bounded, create-only action0/2/4 branches for the frozen v4 SA diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_prepared_state_event_chain import (
    OUTPUT as PRIOR_OUTPUT, SOURCE, _bool, _csv, _preflight, _public_state_hash,
    _same, _write_csv,
)
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv

OUTPUT = ROOT / "artifacts/analysis/cscwd_service_feasible_action_branches_20261010_v1"
PLAN = ROOT / "docs/project/cscwd_service_feasible_action_branch_plan_20261010.md"
STATE_FIELDS = frozenset({
    "config", "instance", "_mask_builder", "_object_catalog", "_adapter_to_bundle", "_rng",
    "_decision_model_mode", "nodes", "node_map", "execution_order", "completed", "node_index",
    "step_index", "clock_seconds", "last_execution_rsu", "prepared_state", "metrics",
    "_deadline_miss_recorded", "_reward_totals_by_profile", "caches",
})
BRANCH_ACTIONS = (0, 2, 4)
MAX_SOURCE_FAILURES = 24
MAX_CONTROLS = 6
MAX_BRANCHES = 90
MAX_STEPS = 2160
MAX_POLICY_FORWARDS = 2160


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, np.ndarray):
        return _stable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("nonfinite snapshot value")
        return value
    raise TypeError(f"unserializable snapshot value: {type(value).__name__}")


def _hash(value: Any) -> str:
    raw = json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _snapshot_hash(env: CalibratedContinuousWorkflowEnv, observation: Any, info: dict[str, Any]) -> str:
    actual_fields = frozenset(vars(env))
    if actual_fields != STATE_FIELDS:
        raise RuntimeError(f"environment mutable field contract drift: {sorted(actual_fields ^ STATE_FIELDS)}")
    payload = {key: getattr(env, key) for key in STATE_FIELDS - {"_mask_builder", "_rng", "caches"}}
    payload["_rng"] = env._rng.bit_generator.state
    payload["_mask_builder"] = [vars(item) for item in env._mask_builder._schema._actions]
    payload["caches"] = {key: vars(value) for key, value in env.caches.items()}
    payload["policy_input_observation"] = [float(item) for item in observation]
    payload["policy_input_info"] = info
    return _hash(payload)


def _source_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (str(row["checkpoint_view"]), str(row["split"]), str(row["method"]),
            str(row["seed"]), str(row["design_id"]))


def _source_id(row: dict[str, Any]) -> str:
    return "|".join((*_source_key(row), str(row["step_index"])))


def _rebuild(config: dict[str, Any], instance: dict[str, Any], records: list[dict[str, str]],
             step: int) -> tuple[CalibratedContinuousWorkflowEnv, Any, dict[str, Any], dict[str, str]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    for row in sorted(records, key=lambda item: int(item["step_index"])):
        actual_step = int(row["step_index"])
        if actual_step != env.step_index:
            raise RuntimeError("recorded prefix step drift")
        if actual_step == step:
            return env, observation, info, row
        if actual_step > step:
            break
        observation, _, _, _, info = env.step(int(row["executed_action"]))
        transition = info["transition"]
        if bool(transition["service_completed"]) != _bool(row["service_completed"]):
            raise RuntimeError("recorded prefix service drift")
        if not _same(transition["clock_seconds_after"], row["clock_seconds_after"]):
            raise RuntimeError("recorded prefix clock drift")
    raise RuntimeError(f"selected source step absent: {instance['design_id']} {step}")


def _checkpoint(view: str, seed: str, inventory: dict[str, dict[str, Any]]) -> tuple[Path, str]:
    name = f"checkpoints/sa_ghmappo_seed{seed}_{view}.pt"
    if name not in inventory:
        raise RuntimeError(f"checkpoint missing from inventory: {name}")
    path = SOURCE / name
    digest = inventory[name]["sha256"]
    if _sha256(path) != digest:
        raise RuntimeError(f"checkpoint hash drift: {name}")
    return path, digest


def _select_sources(config: dict[str, Any], instances: dict[str, dict[str, Any]],
                    episodes: dict[tuple[str, ...], list[dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_manifest = json.loads((PRIOR_OUTPUT / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name, digest in prior_manifest["output_files"].items():
        if _sha256(PRIOR_OUTPUT / name) != digest:
            raise RuntimeError(f"prior analysis artifact drift: {name}")
    if prior_manifest["auxiliary_probe"] != {"first_failure_states": 24, "new_policy_forwards": 0,
                                               "action_branches": 0, "suffix_steps": 0}:
        raise RuntimeError("prior probe identity drift")
    probes = _csv(PRIOR_OUTPUT / "auxiliary_probe_rows.csv")
    if len(probes) != MAX_SOURCE_FAILURES:
        raise RuntimeError("first-failure probe denominator drift")
    sources = []
    for row in probes:
        source = {"source_kind": "failure", "checkpoint_view": row["checkpoint_view"],
                  "split": row["split"], "method": row["method"], "seed": row["seed"],
                  "design_id": row["design_id"], "step_index": int(row["first_failure_step"]),
                  "prior_public_state_sha256": row["public_state_sha256"],
                  "original_event_target": int(row["pseudo_event_target"])}
        sources.append(source)
    label_agent = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    candidates = [row for rows in episodes.values() for row in rows
                  if row["checkpoint_view"] == "selected" and row["method"] == "sa_ghmappo"
                  and row["executed_action"] == "4" and _bool(row["current_bundle_ready"])
                  and _bool(row["service_completed"]) and _bool(row["migration_success"])]
    candidates.sort(key=lambda row: (row["split"], row["design_id"], int(row["seed"]), int(row["step_index"])))
    selected_controls = []
    for candidate in candidates:
        if len(selected_controls) >= MAX_CONTROLS:
            break
        key = _source_key(candidate)
        env, observation, info, recorded = _rebuild(config, instances[candidate["design_id"]],
                                                     episodes[key], int(candidate["step_index"]))
        targets = label_agent._build_mechanism_targets(info["semantic_state"])
        if int(targets["event_target"]) != 1:
            continue
        source = {"source_kind": "control", "checkpoint_view": "selected",
                  "split": candidate["split"], "method": "sa_ghmappo", "seed": candidate["seed"],
                  "design_id": candidate["design_id"], "step_index": int(candidate["step_index"]),
                  "prior_public_state_sha256": _public_state_hash(observation, info),
                  "original_event_target": 1}
        selected_controls.append(source)
    if len(selected_controls) != MAX_CONTROLS:
        raise RuntimeError(f"insufficient deterministic positive controls: {len(selected_controls)}")
    if len({_source_id(row) for row in sources + selected_controls}) != len(sources) + len(selected_controls):
        raise RuntimeError("duplicate selected source identity")
    sources.extend(selected_controls)
    return sources, {"prior_analysis_manifest_sha256": _sha256(PRIOR_OUTPUT / "analysis_manifest.json"),
                     "positive_control_candidate_count": len(candidates)}


def _prepare() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, ...], list[dict[str, str]]], list[dict[str, Any]], dict[str, Any]]:
    config, instances, _, episodes, science = _preflight()
    inventory = {row["path"]: row for row in json.loads((SOURCE / "artifact_integrity.json").read_text(encoding="utf-8"))["files"]}
    sources, prior = _select_sources(config, instances, episodes)
    mappings = []
    for source in sources:
        key = _source_key(source)
        env, observation, info, recorded = _rebuild(config, instances[source["design_id"]],
                                                     episodes[key], source["step_index"])
        if _public_state_hash(observation, info) != source["prior_public_state_sha256"]:
            raise RuntimeError(f"public state source drift: {_source_id(source)}")
        if env._decision_model_mode:
            raise RuntimeError("branch snapshot unexpectedly in estimated decision-model mode")
        full_hash = _snapshot_hash(env, observation, info)
        if _snapshot_hash(env.clone(), observation, info) != full_hash:
            raise RuntimeError("clone snapshot is not identical")
        checkpoint_path, checkpoint_hash = _checkpoint(source["checkpoint_view"], source["seed"], inventory)
        targets = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)._build_mechanism_targets(info["semantic_state"])
        if int(targets["event_target"]) != source["original_event_target"]:
            raise RuntimeError("source pseudo label drift")
        current = env._current_rsu_id()
        node = env._current_node()
        current_ready = env._bundle_ready(current, str(node["required_adapter"]))
        target = env._predicted_handoff_target()
        target_ready = bool(target and env._bundle_ready(target, str(node["required_adapter"])))
        mapping = dict(source)
        mapping.update({"source_id": _source_id(source), "full_state_sha256": full_hash,
                        "checkpoint": str(checkpoint_path.relative_to(SOURCE)),
                        "checkpoint_sha256": checkpoint_hash,
                        "branch_key": _hash([full_hash, checkpoint_hash]),
                        "public_state_sha256": _public_state_hash(observation, info),
                        "action_mask": list(info["action_mask"]), "original_action": int(recorded["executed_action"]),
                        "current_bundle_ready": current_ready, "target_bundle_ready": target_ready,
                        "current_rsu": current, "target_rsu": target or "",
                        "deadline_remaining_seconds": float(env.instance["deadline_seconds"]) - float(env.clock_seconds),
                        "completed_prefix": list(env.completed), "prepared_state": env.prepared_state,
                        "candidate_event_target": int(targets["event_target"]) if current_ready else 0,
                        "candidate_event_soft_target": float(targets["event_soft_target"]) if current_ready else 0.0})
        mappings.append(mapping)
    unique = {row["branch_key"] for row in mappings}
    if len(mappings) != 30 or len(unique) > 30 or len(unique) * 3 > MAX_BRANCHES:
        raise RuntimeError("source/branch budget overflow")
    receipt = {"schema_version": "cscwd_service_feasible_action_branch_preflight_v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": _sha256(PLAN),
               "science": science, "prior": prior, "source_rows": len(mappings),
               "failure_source_rows": MAX_SOURCE_FAILURES, "positive_control_source_rows": MAX_CONTROLS,
               "unique_branch_states": len(unique), "maximum_branch_slots": len(unique) * 3,
               "max_env_steps": MAX_STEPS, "max_policy_forwards": MAX_POLICY_FORWARDS,
               "mappings": mappings}
    return config, instances, episodes, mappings, receipt


def preflight() -> None:
    if OUTPUT.exists():
        raise RuntimeError("create-only output root already exists")
    _, _, _, _, receipt = _prepare()
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "preflight_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "source_rows": receipt["source_rows"],
                      "unique_branch_states": receipt["unique_branch_states"],
                      "maximum_branch_slots": receipt["maximum_branch_slots"]}, ensure_ascii=False))


def _delta(after: dict[str, Any], before: dict[str, Any], field: str) -> float:
    return float(after[field]) - float(before[field])


def _run_branch(config: dict[str, Any], instance: dict[str, Any], env_snapshot: CalibratedContinuousWorkflowEnv,
                observation: Any, info: dict[str, Any], mapping: dict[str, Any], action: int,
                budget: dict[str, int]) -> dict[str, Any]:
    if not info["action_mask"][action]:
        return {"branch_key": mapping["branch_key"], "action": action, "status": "NA_MASKED",
                "source_id": mapping["source_id"], "steps": []}
    env = env_snapshot.clone()
    if _snapshot_hash(env, observation, info) != mapping["full_state_sha256"]:
        raise RuntimeError("branch clone identity mismatch")
    agent = _build_learned("sa_ghmappo", int(mapping["seed"]), config, popart_enabled=False)
    agent.load(str(SOURCE / mapping["checkpoint"]))
    agent._deterministic_action = True
    random.seed(314159)
    np.random.seed(314159)
    torch.manual_seed(314159)
    before = env.summary()
    rows = []
    first = True
    step_cap = min(int(env.instance["max_steps"]), 24)
    while not env.terminated and env.step_index < step_cap:
        if budget["env_steps"] >= MAX_STEPS:
            raise RuntimeError("environment branch step cap reached")
        current_step = env.step_index
        current_rsu = env._current_rsu_id()
        target_rsu = env._predicted_handoff_target()
        node = env._current_node()
        required = str(node["required_adapter"])
        current_ready = env._bundle_ready(current_rsu, required)
        target_ready = bool(target_rsu and env._bundle_ready(target_rsu, required))
        public_hash = _public_state_hash(observation, info)
        if first:
            chosen = action
            origin = "forced_legal_first_action"
        else:
            if budget["policy_forwards"] >= MAX_POLICY_FORWARDS:
                raise RuntimeError("policy forward cap reached")
            chosen, _ = agent.act(observation, info)
            budget["policy_forwards"] += 1
            origin = "frozen_raw_deterministic_sa"
        if not info["action_mask"][int(chosen)]:
            raise RuntimeError("policy returned masked action")
        before_completed = len(env.completed)
        observation, reward, terminated, truncated, info = env.step(int(chosen))
        budget["env_steps"] += 1
        transition = info["transition"]
        rows.append({"step_index": current_step, "source": origin, "action": int(chosen),
                     "public_state_sha256": public_hash, "current_rsu": current_rsu,
                     "predicted_target_rsu": target_rsu or "", "current_bundle_ready": current_ready,
                     "target_bundle_ready": target_ready, "service_completed": bool(transition["service_completed"]),
                     "node_progressed": len(env.completed) > before_completed,
                     "target_model_stage_events": transition["cache_events"],
                     "state_transfer_status": (transition["state_transfer"] or {}).get("status", ""),
                     "state_commit": bool(transition["migration_success"]),
                     "clock_seconds_after": float(transition["clock_seconds_after"]),
                     "step_cost_seconds": float(transition["step_cost_seconds"]),
                     "reward": float(reward), "reward_components": transition["reward_components"],
                     "model_transfer_bytes": int(transition["model_transfer_bytes"]),
                     "state_transfer_bytes": int(transition["state_transfer_bytes"]),
                     "input_transfer_bytes": int(transition["input_transfer_bytes"]),
                     "recompute_seconds": float(transition["recompute_seconds"]),
                     "terminated": bool(terminated), "truncated": bool(truncated)})
        first = False
        if terminated or truncated:
            break
    after = env.summary()
    return {"branch_key": mapping["branch_key"], "action": action, "status": "COMPLETE",
            "source_id": mapping["source_id"], "checkpoint_sha256": mapping["checkpoint_sha256"],
            "full_state_sha256": mapping["full_state_sha256"], "deadline_remaining_seconds": mapping["deadline_remaining_seconds"],
            "steps": rows, "outcome": {
                "workflow_completed": bool(after["workflow_completed"]),
                "on_time_workflow_completed": bool(after["on_time_workflow_completed"]),
                "service_failures": int(_delta(after, before, "service_failures")),
                "elapsed_seconds": _delta(after, before, "modeled_completion_seconds"),
                "model_transfer_bytes": int(_delta(after, before, "model_transfer_bytes")),
                "state_transfer_bytes": int(_delta(after, before, "state_transfer_bytes")),
                "input_transfer_bytes": int(_delta(after, before, "input_transfer_bytes")),
                "recompute_seconds": _delta(after, before, "recompute_seconds"),
                "original_reward": _delta(after, before, "reward"),
                "ending_step_index": int(env.step_index), "ending_completed_nodes": int(after["completed_nodes"]),
                "ending_deadline_missed": bool(after["deadline_missed"]),
            }}


def _service_tuple(outcome: dict[str, Any]) -> tuple[int, int, int]:
    return (int(outcome["workflow_completed"]), int(outcome["on_time_workflow_completed"]),
            -int(outcome["service_failures"]))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    maximize = ("workflow_completed", "on_time_workflow_completed", "original_reward")
    minimize = ("service_failures", "elapsed_seconds", "model_transfer_bytes", "state_transfer_bytes",
                "input_transfer_bytes", "recompute_seconds")
    no_worse = all(float(left[field]) >= float(right[field]) - 1e-8 for field in maximize)
    no_worse = no_worse and all(float(left[field]) <= float(right[field]) + 1e-8 for field in minimize)
    strict = any(float(left[field]) > float(right[field]) + 1e-8 for field in maximize)
    strict = strict or any(float(left[field]) < float(right[field]) - 1e-8 for field in minimize)
    return no_worse and strict


def _gate(mappings: list[dict[str, Any]], branches: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mappings:
        by_key[row["branch_key"]].append(row)
    failures = []
    mixed = []
    state_rows = []
    for key, origins in sorted(by_key.items()):
        outcomes = {action: branches[(key, action)] for action in BRANCH_ACTIONS}
        affected = any(row["source_kind"] == "failure" and row["original_event_target"] == 1
                       and not row["current_bundle_ready"] for row in origins)
        controls = any(row["source_kind"] == "control" for row in origins)
        valid = {action: branch for action, branch in outcomes.items() if branch["status"] == "COMPLETE"}
        pareto = [action for action, branch in valid.items()
                  if not any(other_action != action and _dominates(other["outcome"], branch["outcome"])
                             for other_action, other in valid.items())]
        row = {"branch_key": key, "source_ids": [item["source_id"] for item in origins],
               "affected_failure_state": affected, "positive_control_state": controls,
               "action0_legal": 0 in valid, "action2_legal": 2 in valid, "action4_legal": 4 in valid,
               "pareto_nondominated_actions": pareto,
               "service_tuples": {str(action): _service_tuple(branch["outcome"]) for action, branch in valid.items()}}
        if controls:
            if any(item["candidate_event_target"] != 1 for item in origins if item["source_kind"] == "control"):
                failures.append({"branch_key": key, "reason": "positive_event_label_not_preserved"})
            if 4 not in valid or not valid[4]["steps"][0]["service_completed"] or not valid[4]["steps"][0]["state_commit"]:
                failures.append({"branch_key": key, "reason": "positive_action4_service_or_commit_failed"})
        if affected:
            if 4 not in valid:
                mixed.append({"branch_key": key, "reason": "action4_masked_semantics_unobserved"})
            else:
                if valid[4]["steps"][0]["service_completed"]:
                    failures.append({"branch_key": key, "reason": "current_missing_action4_completed_service"})
                service_alts = [action for action in (0, 2) if action in valid and valid[action]["steps"][0]["service_completed"]]
                if not service_alts:
                    failures.append({"branch_key": key, "reason": "no_current_service_success_alternative"})
                else:
                    action4 = valid[4]["outcome"]
                    if _service_tuple(action4) > max(_service_tuple(valid[action]["outcome"]) for action in service_alts):
                        mixed.append({"branch_key": key, "reason": "action4_strictly_better_service_outcome"})
                    if 4 in pareto:
                        mixed.append({"branch_key": key, "reason": "action4_pareto_nondominated"})
                    if not any(_dominates(valid[action]["outcome"], action4) for action in service_alts):
                        mixed.append({"branch_key": key, "reason": "no_service_feasible_alternative_pareto_dominates_action4"})
        state_rows.append(row)
    status = "FAIL" if failures else "MIXED" if mixed else "PASS"
    return {"schema_version": "cscwd_service_feasible_action_branch_gate_v1", "status": status,
            "state_rows": state_rows, "hard_failures": failures, "mixed_counterexamples": mixed,
            "affected_unique_states": sum(row["affected_failure_state"] for row in state_rows),
            "positive_control_unique_states": sum(row["positive_control_state"] for row in state_rows),
            "claim_boundary": "bounded development feasibility only; no training or paper claim authorized by this output alone"}


def run() -> None:
    preflight_path = OUTPUT / "preflight_receipt.json"
    if not preflight_path.is_file() or (OUTPUT / "analysis_manifest.json").exists():
        raise RuntimeError("frozen preflight absent or analysis already complete")
    frozen_bytes = preflight_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    config, instances, episodes, mappings, current = _prepare()
    if current["plan_sha256"] != frozen["plan_sha256"] or current["mappings"] != frozen["mappings"]:
        raise RuntimeError("create-only preflight state/source identity drift")
    if (OUTPUT / "runner_entered.json").exists():
        raise RuntimeError("runner already entered; inspect partial branches rather than restarting")
    _write_json(OUTPUT / "runner_entered.json", {"at": datetime.now(timezone.utc).isoformat(),
                                                 "preflight_sha256": hashlib.sha256(frozen_bytes).hexdigest()})
    branch_dir = OUTPUT / "branches"
    branch_dir.mkdir()
    by_key = {}
    for mapping in mappings:
        by_key.setdefault(mapping["branch_key"], mapping)
    budget = {"env_steps": 0, "policy_forwards": 0}
    results = {}
    try:
        for branch_key, mapping in sorted(by_key.items()):
            source = mapping
            env, observation, info, recorded = _rebuild(config, instances[source["design_id"]],
                                                         episodes[_source_key(source)], int(source["step_index"]))
            if _snapshot_hash(env, observation, info) != mapping["full_state_sha256"]:
                raise RuntimeError("run-time snapshot drift")
            for action in BRANCH_ACTIONS:
                branch = _run_branch(config, instances[source["design_id"]], env, observation, info,
                                     mapping, action, budget)
                if action == int(recorded["executed_action"]) and branch["status"] == "COMPLETE":
                    first = branch["steps"][0]
                    for field, name in (("service_completed", "service_completed"), ("state_commit", "migration_success")):
                        if bool(first[field]) != _bool(recorded[name]):
                            raise RuntimeError(f"original first-step boolean mismatch: {mapping['source_id']} {field}")
                    for field in ("model_transfer_bytes", "state_transfer_bytes", "recompute_seconds", "clock_seconds_after"):
                        if not _same(first[field], recorded[field]):
                            raise RuntimeError(f"original first-step metric mismatch: {mapping['source_id']} {field}")
                path = branch_dir / f"{branch_key}_a{action}.json"
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(branch, handle, ensure_ascii=False, sort_keys=True, indent=2)
                    handle.write("\n")
                results[(branch_key, action)] = branch
    except Exception as exc:
        _write_json(OUTPUT / "failure_receipt.json", {"at": datetime.now(timezone.utc).isoformat(),
                                                      "error_type": type(exc).__name__, "error": str(exc),
                                                      "branches_written": len(results), "budget": budget})
        raise
    gate = _gate(mappings, results)
    _write_json(OUTPUT / "gate_report.json", gate)
    map_rows = [{key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                 for key, value in row.items()} for row in mappings]
    _write_csv(OUTPUT / "source_to_state_rows.csv", map_rows)
    summary_rows = []
    trajectory_rows = []
    for (key, action), branch in sorted(results.items()):
        row = {"branch_key": key, "action": action, "status": branch["status"], "source_id": branch["source_id"],
               "step_count": len(branch["steps"]), "first_service_completed": branch["steps"][0]["service_completed"] if branch["steps"] else "",
               "first_state_commit": branch["steps"][0]["state_commit"] if branch["steps"] else "",
               "first_model_stage_events_json": json.dumps(branch["steps"][0]["target_model_stage_events"], sort_keys=True) if branch["steps"] else ""}
        row.update(branch.get("outcome", {}))
        summary_rows.append(row)
        for event in branch["steps"]:
            trajectory_rows.append({"branch_key": key, "branch_action": action, **{
                name: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for name, value in event.items()}})
    summary_fields = list(dict.fromkeys(field for row in summary_rows for field in row))
    _write_csv(OUTPUT / "branch_summary_rows.csv",
               [{field: row.get(field, "") for field in summary_fields} for row in summary_rows])
    _write_csv(OUTPUT / "branch_trajectory_rows.csv", trajectory_rows)
    files = ["preflight_receipt.json", "runner_entered.json", "gate_report.json",
             "source_to_state_rows.csv", "branch_summary_rows.csv", "branch_trajectory_rows.csv"]
    files += [str(path.relative_to(OUTPUT)) for path in sorted(branch_dir.glob("*.json"))]
    manifest = {"schema_version": "cscwd_service_feasible_action_branch_manifest_v1",
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
                "gate_status": gate["status"], "plan_sha256": _sha256(PLAN),
                "source_science_commit": frozen["science"]["source_commit"],
                "source_rows": len(mappings), "unique_branch_states": len(by_key),
                "branch_slots": len(results), "legal_branches": sum(row["status"] == "COMPLETE" for row in results.values()),
                "masked_na_branches": sum(row["status"] == "NA_MASKED" for row in results.values()),
                "env_steps": budget["env_steps"], "policy_forwards": budget["policy_forwards"],
                "files": {name: _sha256(OUTPUT / name) for name in files}}
    _write_json(OUTPUT / "analysis_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("gate_status", "source_rows", "unique_branch_states",
                                                   "branch_slots", "legal_branches", "env_steps", "policy_forwards")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    preflight() if args.preflight else run()


if __name__ == "__main__":
    main()
