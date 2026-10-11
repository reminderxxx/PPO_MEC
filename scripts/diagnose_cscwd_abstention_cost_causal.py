"""Hash-bound, bounded replay and first-divergence branches for CSCWD abstention."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
B_ROOT = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC")
for entry in (str(ROOT), str(B_ROOT)):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

from scripts.run_calibrated_workflow_event_aux_abstention_ab import _load_candidate_inputs
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from scripts.run_cscwd_service_feasible_action_branches import _snapshot_hash
from scripts.diagnose_cscwd_prepared_state_event_chain import _bool, _public_state_hash, _same, _write_csv
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, _network_seconds

PLAN = ROOT / "docs/project/cscwd_abstention_cost_causal_audit_plan_20261010.md"
OUTPUT = ROOT / "artifacts/analysis/cscwd_abstention_cost_causal_20261010_v2"
BASE = B_ROOT / "artifacts/experiments"
CANDIDATE = BASE / "cscwd_event_aux_abstention_ab_20261010_v1"
CONTROL = BASE / "cscwd_causal_prepared_state_visibility_matched_20261010_v1"
PROTOCOL = B_ROOT / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json"
SPLITS = ("regression", "frozen_check")
VIEWS = ("selected", "update96")
MAX_SOURCES = 12
MAX_BRANCHES = 36
MAX_BRANCH_STEPS = 864


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return str(row["checkpoint_view"]), str(row["split"]), int(row["seed"]), str(row["design_id"])


def _sort(key: tuple[str, str, int, str]) -> tuple[int, int, int, str]:
    return VIEWS.index(key[0]), SPLITS.index(key[1]), key[2], key[3]


def _inventory(root: Path, manifest_sha: str, integrity_sha: str, expected_count: int) -> dict[str, dict[str, Any]]:
    if _sha256(root / "run_manifest.json") != manifest_sha or _sha256(root / "artifact_integrity.json") != integrity_sha:
        raise RuntimeError(f"science manifest/integrity hash drift: {root.name}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "completion_receipt.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or not completion.get("scientific_execution_complete"):
        raise RuntimeError(f"scientific run incomplete: {root.name}")
    if manifest.get("interface_profile") != "calibrated_workflow_interface_v4_prepared_state_prefix" or manifest.get("formal_or_holdout_reads") != 0:
        raise RuntimeError("profile or held-out read drift")
    entries = json.loads((root / "artifact_integrity.json").read_text(encoding="utf-8"))["files"]
    if len(entries) != expected_count:
        raise RuntimeError(f"inventory count drift: {root.name} {len(entries)}")
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"artifact hash drift: {path}")
    return {entry["path"]: entry for entry in entries}


def _inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    ci = _inventory(CANDIDATE, "d9f256b6277c6b358b7eafd329f118f1f56884b67d94640bfcb8775d29fa880c",
                    "fa03b3fe9eed9af66f96e10e10d90db380b13a28893716071a395a8e4eb129cc", 40)
    oi = _inventory(CONTROL, "48718e48dc55e6958c03b676634dca53e84fd21be251115d4809380a1441c615",
                    "b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1", 119)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["single_variable"]["implementation_commit"] != "46a68f11c289ccc304b88cdfed01c34ba5f61c3d":
        raise RuntimeError("candidate implementation identity drift")
    config, splits, identity, _ = _load_candidate_inputs(protocol)
    if identity["public_prefix_suffix_tamper_comparisons"] != 474:
        raise RuntimeError("causal prefix identity drift")
    instances = {str(row["design_id"]): row for split in SPLITS for row in splits[split]}
    return config, instances, ci, oi


def _tables() -> tuple[dict[str, dict[tuple[str, str, int, str], dict[str, str]]], dict[str, dict[tuple[str, str, int, str], list[dict[str, str]]]]]:
    evaluations: dict[str, dict[tuple[str, str, int, str], dict[str, str]]] = {}
    episodes: dict[str, dict[tuple[str, str, int, str], list[dict[str, str]]]] = {}
    for arm, root in (("candidate", CANDIDATE), ("control", CONTROL), ("ppo", CONTROL)):
        ev: dict[tuple[str, str, int, str], dict[str, str]] = {}
        ep: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
        for view in VIEWS:
            stem = f"candidate_{view}" if arm == "candidate" else f"new_{view}"
            rows = _csv(root / f"{stem}_evaluation_rows.csv")
            rows = [row for row in rows if row["method"] == ("ppo" if arm == "ppo" else "sa_ghmappo")]
            for row in rows:
                key = _key(row)
                if key in ev:
                    raise RuntimeError("duplicate evaluation identity")
                ev[key] = row
            for row in _csv(root / f"{stem}_behavior_ledger.csv"):
                if row["method"] != ("ppo" if arm == "ppo" else "sa_ghmappo"):
                    continue
                ep[_key(row)].append(row)
        if set(ev) != set(ep):
            raise RuntimeError(f"{arm} ledger/evaluation mismatch")
        if len(ev) != 200:
            raise RuntimeError(f"{arm} episode denominator mismatch")
        for rows in ep.values():
            rows.sort(key=lambda row: int(row["step_index"]))
        evaluations[arm], episodes[arm] = ev, ep
    return evaluations, episodes


def _first_divergence(left: list[dict[str, str]], right: list[dict[str, str]]) -> int | None:
    for position, (a, b) in enumerate(zip(left, right)):
        if int(a["step_index"]) != position or int(b["step_index"]) != position:
            raise RuntimeError("ledger step index drift")
        if a["executed_action"] != b["executed_action"]:
            return position
        for field in ("current_rsu_id", "node_id", "current_bundle_ready", "clock_seconds_after"):
            if (not _same(a[field], b[field])) if field == "clock_seconds_after" else (a[field] != b[field]):
                raise RuntimeError(f"same-action prefix state drift: {field}")
    return None


def _selected(evaluations, episodes) -> list[dict[str, Any]]:
    ce = evaluations["candidate"]
    oe = evaluations["control"]
    if set(ce) != set(oe):
        raise RuntimeError("candidate/control episode pairing drift")
    eligible = []
    for key in sorted(ce, key=_sort):
        c, o = ce[key], oe[key]
        ca, oa = episodes["candidate"][key], episodes["control"][key]
        step = _first_divergence(ca, oa)
        if step is None:
            if any(c[field] != o[field] for field in ("on_time_workflow_completion_rate", "service_failures", "model_prepare_mb")):
                raise RuntimeError(f"same actions changed outcome: {key}")
            continue
        eligible.append({"view": key[0], "split": key[1], "seed": key[2], "design_id": key[3],
                         "first_divergence_step": step, "candidate_action": int(ca[step]["executed_action"]),
                         "control_action": int(oa[step]["executed_action"]),
                         "on_time_delta": float(c["on_time_workflow_completion_rate"]) - float(o["on_time_workflow_completion_rate"]),
                         "model_mb_delta": float(c["model_prepare_mb"]) - float(o["model_prepare_mb"]),
                         "elapsed_delta": float(c["modeled_completion_seconds"]) - float(o["modeled_completion_seconds"])})
    chosen, seen = [], set()
    strata = [
        ("fixed_on_time_down_17", "update96", 17, "down", False),
        ("fixed_on_time_down_29", "update96", 29, "down", False),
        ("fixed_on_time_down_61", "update96", 61, "down", False),
        ("fixed_on_time_up_7", "update96", 7, "up", False),
        ("selected_on_time_up_7", "selected", 7, "up", False),
        ("selected_on_time_up_17", "selected", 17, "up", False),
        ("selected_on_time_up_29", "selected", 29, "up", False),
        ("selected_on_time_down_7", "selected", 7, "down", False),
        ("selected_on_time_down_17", "selected", 17, "down", False),
        ("selected_neutral_model_up_29", "selected", 29, "neutral", True),
        ("selected_neutral_model_up_43", "selected", 43, "neutral", True),
        ("fixed_neutral_model_up_61", "update96", 61, "neutral", True),
    ]
    for label, view, seed, direction, needs_bytes in strata:
        pool = [row for row in eligible if row["view"] == view and row["seed"] == seed
                and (row["on_time_delta"] > 0 if direction == "up" else row["on_time_delta"] < 0 if direction == "down" else row["on_time_delta"] == 0)
                and (not needs_bytes or row["model_mb_delta"] > 1e-9)
                and (row["split"], row["seed"], row["design_id"]) not in seen]
        pool.sort(key=lambda row: (SPLITS.index(row["split"]), row["seed"], row["design_id"]))
        if pool:
            row = {**pool[0], "stratum": label}
            chosen.append(row)
            seen.add((row["split"], row["seed"], row["design_id"]))
    if len(chosen) > MAX_SOURCES:
        raise RuntimeError("source cap exceeded")
    return chosen


def _prepare():
    config, instances, ci, oi = _inputs()
    evaluations, episodes = _tables()
    selected = _selected(evaluations, episodes)
    return config, instances, ci, oi, evaluations, episodes, selected


def preflight() -> None:
    if OUTPUT.exists():
        raise RuntimeError("create-only output already exists")
    _, _, ci, oi, evaluations, episodes, selected = _prepare()
    receipt = {"schema_version": "cscwd_abstention_cost_causal_preflight_v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "plan_sha256": _sha256(PLAN), "candidate_integrity_sha256": _sha256(CANDIDATE / "artifact_integrity.json"),
               "control_integrity_sha256": _sha256(CONTROL / "artifact_integrity.json"),
               "candidate_inventory_verified": len(ci), "control_inventory_verified": len(oi),
               "candidate_episodes": len(evaluations["candidate"]), "control_sa_ppo_episodes": len(evaluations["control"]) + len(evaluations["ppo"]),
               "selected": selected, "maximum_sources": MAX_SOURCES, "maximum_branches": MAX_BRANCHES,
               "maximum_counterfactual_env_steps": MAX_BRANCH_STEPS}
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "preflight_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "selected": len(selected), "strata": [row["stratum"] for row in selected]}))


def _checkpoint(root: Path, inventory: dict[str, dict[str, Any]], view: str, seed: int) -> tuple[Path, str]:
    name = f"checkpoints/sa_ghmappo_seed{seed}_{view}.pt"
    if name not in inventory:
        raise RuntimeError(f"checkpoint missing: {name}")
    path = root / name
    if _sha256(path) != inventory[name]["sha256"]:
        raise RuntimeError(f"checkpoint drift: {name}")
    return path, inventory[name]["sha256"]


def _rebuild_prefix(config, instance, candidate_rows, control_rows, step):
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    for index in range(step):
        a, b = candidate_rows[index], control_rows[index]
        if int(a["executed_action"]) != int(b["executed_action"]) or int(a["step_index"]) != env.step_index:
            raise RuntimeError("prefix action/step mismatch")
        observation, _, _, _, info = env.step(int(a["executed_action"]))
        transition = info["transition"]
        if not _same(transition["clock_seconds_after"], a["clock_seconds_after"]) or not _same(transition["clock_seconds_after"], b["clock_seconds_after"]):
            raise RuntimeError("prefix clock mismatch")
    if env.step_index != step or env._current_rsu_id() != candidate_rows[step]["current_rsu_id"] or env._current_rsu_id() != control_rows[step]["current_rsu_id"]:
        raise RuntimeError("first divergent state mismatch")
    if str(env._current_node()["node_id"]) != candidate_rows[step]["node_id"] or str(env._current_node()["node_id"]) != control_rows[step]["node_id"]:
        raise RuntimeError("first divergent DAG node mismatch")
    return env, observation, info


def _run_branch(env, observation, info, first_action, agent, source, budget):
    branch = env.clone()
    before = branch.summary()
    rows = []
    while not branch.terminated and branch.step_index < min(int(branch.instance["max_steps"]), 24):
        if budget["branch_env_steps"] >= MAX_BRANCH_STEPS:
            raise RuntimeError("counterfactual env.step budget exceeded")
        first = not rows
        if first:
            action = first_action
        else:
            action, _ = agent.act(observation, info)
            budget["policy_forwards"] += 1
        if not info["action_mask"][int(action)]:
            raise RuntimeError("branch action masked")
        prior_clock = branch.clock_seconds
        prior_node = str(branch._current_node()["node_id"])
        prior_rsu = branch._current_rsu_id()
        public_hash = _public_state_hash(observation, info)
        observation, reward, terminated, truncated, info = branch.step(int(action))
        budget["branch_env_steps"] += 1
        tr = info["transition"]
        rows.append({"step_index": budget["branch_env_steps"], "episode_step_index": branch.step_index - 1,
                     "source": "forced" if first else "frozen_candidate_raw_policy", "action": int(action),
                     "public_state_sha256": public_hash, "node_id": prior_node, "current_rsu": prior_rsu,
                     "clock_before": prior_clock, "clock_after": float(tr["clock_seconds_after"]),
                     "step_cost_seconds": float(tr["step_cost_seconds"]), "service_completed": bool(tr["service_completed"]),
                     "cache_events": tr["cache_events"], "state_transfer": tr["state_transfer"],
                     "model_transfer_bytes": int(tr["model_transfer_bytes"]), "state_transfer_bytes": int(tr["state_transfer_bytes"]),
                     "input_transfer_bytes": int(tr["input_transfer_bytes"]), "recompute_seconds": float(tr["recompute_seconds"]),
                     "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
        if terminated or truncated:
            break
    after = branch.summary()
    return {"source": source, "first_action": first_action, "steps": rows,
            "outcome": {"workflow_completed": bool(after["workflow_completed"]),
                        "on_time_workflow_completed": bool(after["on_time_workflow_completed"]),
                        "deadline_missed": bool(after["deadline_missed"]),
                        "service_failures": int(after["service_failures"] - before["service_failures"]),
                        "elapsed_seconds": float(after["modeled_completion_seconds"] - before["modeled_completion_seconds"]),
                        "model_transfer_bytes": int(after["model_transfer_bytes"] - before["model_transfer_bytes"]),
                        "state_transfer_bytes": int(after["state_transfer_bytes"] - before["state_transfer_bytes"]),
                        "input_transfer_bytes": int(after["input_transfer_bytes"] - before["input_transfer_bytes"]),
                        "recompute_seconds": float(after["recompute_seconds"] - before["recompute_seconds"]),
                        "ending_step_index": int(branch.step_index)}}


def _replay_episode(config, instance, records, evaluation, arm, key):
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    steps, loads = [], []
    active: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, str]] = {(rsu, obj) for rsu, cache in env.caches.items() for obj in cache.residents}
    catalog = env._object_catalog
    for index, row in enumerate(records):
        if env.step_index != index or int(row["step_index"]) != index:
            raise RuntimeError("recorded replay step drift")
        action = int(row["executed_action"])
        current, node = env._current_rsu_id(), env._current_node()
        current_ready = env._bundle_ready(current, str(node["required_adapter"]))
        if current != row["current_rsu_id"] or current_ready != _bool(row["current_bundle_ready"]):
            raise RuntimeError("recorded replay public state drift")
        if not info["action_mask"][action]:
            raise RuntimeError("recorded replay masked action")
        before_clock = env.clock_seconds
        observation, _, _, _, info = env.step(action)
        tr = info["transition"]
        for field in ("model_transfer_bytes", "state_transfer_bytes", "recompute_seconds", "clock_seconds_after", "service_operation_seconds"):
            if not _same(tr[field], row[field]):
                raise RuntimeError(f"recorded replay transition drift: {key} {index} {field}")
        if bool(tr["service_completed"]) != _bool(row["service_completed"]):
            raise RuntimeError("recorded replay service drift")
        admitted_bytes = 0
        for event in tr["cache_events"]:
            if not event["committed"]:
                continue
            rsu = str(event["rsu_id"])
            for victim in event["victims"]:
                active.pop((rsu, victim), None)
            for obj in event["admitted"]:
                obj = str(obj)
                amount = int(catalog[obj]["transfer_bytes"])
                admitted_bytes += amount
                same_before = (rsu, obj) in seen
                elsewhere_before = any(other != rsu and (other, obj) in seen for other in env.caches)
                placement = "eviction_reload" if same_before else "cross_rsu_placement" if elsewhere_before else "first_placement"
                load = {"arm": arm, "view": key[0], "split": key[1], "seed": key[2], "design_id": key[3],
                        "step_index": index, "action": action, "rsu_id": rsu, "object_id": obj,
                        "object_type": catalog[obj]["object_type"], "transfer_bytes": amount,
                        "placement": placement, "used_before_eviction_or_end": False, "victims": json.dumps(event["victims"]),
                        "before_residents": json.dumps(event["before_residents"]), "after_residents": json.dumps(event["after_residents"])}
                active[(rsu, obj)] = len(loads)
                seen.add((rsu, obj))
                loads.append(load)
        if admitted_bytes != int(tr["model_transfer_bytes"]):
            raise RuntimeError("model bytes not conserved across committed cache events")
        if tr["service_completed"] and action != 2:
            for obj in env._bundle_ids(str(node["required_adapter"])):
                position = active.get((current, obj))
                if position is not None:
                    loads[position]["used_before_eviction_or_end"] = True
        model = int(tr["model_transfer_bytes"])
        state = int(tr["state_transfer_bytes"])
        input_bytes = int(tr["input_transfer_bytes"])
        transfer_seconds = sum(_network_seconds(size, env._effective_link_mbps(), float(config["link"]["fixed_seconds"]))
                               for size in (model, state, input_bytes))
        fallback = float(config["vehicle"]["fallback_seconds"]) if action == 2 else 0.0
        compute = float(node["compute_seconds"]) if tr["service_completed"] else 0.0
        failed = float(config["objective"]["failed_service_seconds"]) if not tr["service_completed"] else 0.0
        model_load = sum(float(event["load_seconds"]) for event in tr["cache_events"] if event["committed"])
        restore = float(config["measured_time_seconds"]["state_restore_overhead"]) if tr["migration_success"] else 0.0
        expected_cost = fallback + compute + failed + float(tr["recompute_seconds"]) + transfer_seconds + model_load + restore
        if not _same(before_clock + expected_cost, tr["clock_seconds_after"]):
            raise RuntimeError(f"cost decomposition mismatch: {key} {index}: {expected_cost}")
        steps.append({"arm": arm, "view": key[0], "split": key[1], "seed": key[2], "design_id": key[3],
                      "step_index": index, "node_id": node["node_id"], "current_rsu": current, "action": action,
                      "current_bundle_ready": current_ready, "service_completed": bool(tr["service_completed"]),
                      "model_transfer_bytes": model, "state_transfer_bytes": state, "input_transfer_bytes": input_bytes,
                      "fallback_seconds": fallback, "compute_seconds": compute, "failed_service_seconds": failed,
                      "recompute_seconds": float(tr["recompute_seconds"]), "network_seconds": transfer_seconds,
                      "model_load_seconds": model_load, "state_restore_seconds": restore,
                      "step_cost_seconds": expected_cost, "clock_seconds_after": float(tr["clock_seconds_after"]),
                      "cache_events_json": json.dumps(tr["cache_events"], sort_keys=True)})
    summary = env.summary()
    if not _same(summary["model_transfer_bytes"] / 1_000_000, evaluation["model_prepare_mb"]):
        raise RuntimeError(f"episode model bytes do not match evaluation: {key}")
    if not _same(summary["modeled_completion_seconds"], evaluation["modeled_completion_seconds"]):
        raise RuntimeError(f"episode clock does not match evaluation: {key}")
    for load in loads:
        if load["action"] in (1, 4) and not load["used_before_eviction_or_end"]:
            load["category"] = "unused_preparation"
        elif load["placement"] == "eviction_reload":
            load["category"] = "eviction_reload"
        elif load["placement"] == "cross_rsu_placement":
            load["category"] = "cross_rsu_placement"
        elif load["action"] == 0 and load["used_before_eviction_or_end"]:
            load["category"] = "first_current_service_load"
        elif load["used_before_eviction_or_end"]:
            load["category"] = "first_useful_preparation"
        else:
            load["category"] = "other_unproductive_load"
    if sum(item["transfer_bytes"] for item in loads) != int(summary["model_transfer_bytes"]):
        raise RuntimeError("episode object-load ledger conservation failure")
    return steps, loads


def run() -> None:
    receipt_path = OUTPUT / "preflight_receipt.json"
    if not receipt_path.is_file() or (OUTPUT / "analysis_manifest.json").exists() or (OUTPUT / "runner_entered.json").exists():
        raise RuntimeError("frozen preflight missing or run already entered")
    frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
    config, instances, ci, oi, evaluations, episodes, selected = _prepare()
    if frozen["plan_sha256"] != _sha256(PLAN) or frozen["selected"] != selected:
        raise RuntimeError("preflight selection or plan drift")
    _write_json(OUTPUT / "runner_entered.json", {"at": datetime.now(timezone.utc).isoformat(),
                                                   "preflight_sha256": _sha256(receipt_path)})
    branches_dir = OUTPUT / "branches"
    branches_dir.mkdir()
    budget = {"branch_env_steps": 0, "policy_forwards": 0, "recorded_replay_steps": 0}
    branch_summary = []
    for source in selected:
        key = (source["view"], source["split"], source["seed"], source["design_id"])
        ca, oa = episodes["candidate"][key], episodes["control"][key]
        env, observation, info = _rebuild_prefix(config, instances[key[3]], ca, oa, source["first_divergence_step"])
        snapshot_sha = _snapshot_hash(env, observation, info)
        path, digest = _checkpoint(CANDIDATE, ci, key[0], key[2])
        old_config = {**config, "mechanism_aux_missing_current_event_abstention_enabled": False}
        old_path, old_digest = _checkpoint(CONTROL, oi, key[0], key[2])
        old_agent = _build_learned("sa_ghmappo", key[2], old_config, popart_enabled=False)
        old_agent.load(str(old_path)); old_agent._deterministic_action = True
        old_action, _ = old_agent.act(observation, deepcopy(info))
        if int(old_action) != source["control_action"]:
            raise RuntimeError("old checkpoint first action does not match recorded ledger")
        agent = _build_learned("sa_ghmappo", key[2], config, popart_enabled=False)
        agent.load(str(path)); agent._deterministic_action = True
        actual_action, _ = agent.act(observation, deepcopy(info))
        if int(actual_action) != source["candidate_action"]:
            raise RuntimeError("candidate checkpoint first action does not match recorded ledger")
        actions = list(dict.fromkeys((source["candidate_action"], source["control_action"], 2 if info["action_mask"][2] else 0)))
        actions = [action for action in actions if info["action_mask"][action]]
        if len(actions) > 3 or not actions:
            raise RuntimeError("invalid branch action set")
        for action in actions:
            if _snapshot_hash(env.clone(), observation, info) != snapshot_sha:
                raise RuntimeError("branch clone snapshot drift")
            branch = _run_branch(env, observation, info, action, agent, source, budget)
            if action == source["candidate_action"]:
                suffix = ca[source["first_divergence_step"]:]
                if len(branch["steps"]) != len(suffix):
                    raise RuntimeError("candidate factual suffix length drift")
                for event, recorded in zip(branch["steps"], suffix):
                    if event["action"] != int(recorded["executed_action"]) or not _same(event["clock_after"], recorded["clock_seconds_after"]):
                        raise RuntimeError("candidate factual suffix drift")
            label = f"{key[0]}_{key[1]}_seed{key[2]}_{key[3]}_step{source['first_divergence_step']}_a{action}"
            _write_json(branches_dir / f"{label}.json", {**branch, "snapshot_sha256": snapshot_sha,
                                                          "candidate_checkpoint_sha256": digest,
                                                          "old_checkpoint_sha256": old_digest})
            branch_summary.append({"source_id": label.rsplit("_a", 1)[0], "stratum": source["stratum"],
                                   "view": key[0], "split": key[1], "seed": key[2], "design_id": key[3],
                                   "first_divergence_step": source["first_divergence_step"],
                                   "candidate_action": source["candidate_action"], "control_action": source["control_action"],
                                   "branch_action": action, "snapshot_sha256": snapshot_sha,
                                   "candidate_checkpoint_sha256": digest, "old_checkpoint_sha256": old_digest,
                                   "branch_path": f"branches/{label}.json", **branch["outcome"]})
    if len(branch_summary) > MAX_BRANCHES or budget["branch_env_steps"] > MAX_BRANCH_STEPS:
        raise RuntimeError("counterfactual budget exceeded")
    all_steps, all_loads = [], []
    for arm in ("candidate", "control", "ppo"):
        root_config = config
        for key in sorted(evaluations[arm], key=_sort):
            steps, loads = _replay_episode(root_config, instances[key[3]], episodes[arm][key], evaluations[arm][key], arm, key)
            all_steps.extend(steps); all_loads.extend(loads)
            budget["recorded_replay_steps"] += len(steps)
    _write_csv(OUTPUT / "branch_summary_rows.csv", branch_summary)
    _write_csv(OUTPUT / "recorded_step_cost_rows.csv", all_steps)
    _write_csv(OUTPUT / "object_load_rows.csv", all_loads)
    summary = {"schema_version": "cscwd_abstention_cost_causal_summary_v1", "selected_sources": len(selected),
               "branch_count": len(branch_summary), **budget,
               "model_bytes_by_arm_view_category_type": {}, "fallback_steps_by_arm_view": {},
               "total_model_bytes_from_loads": sum(row["transfer_bytes"] for row in all_loads),
               "total_model_bytes_from_steps": sum(row["model_transfer_bytes"] for row in all_steps)}
    totals = defaultdict(int)
    for row in all_loads:
        totals[(row["arm"], row["view"], row["category"], row["object_type"])] += row["transfer_bytes"]
    summary["model_bytes_by_arm_view_category_type"] = {"|".join(key): value for key, value in sorted(totals.items())}
    falls = defaultdict(int)
    for row in all_steps:
        if row["action"] == 2:
            falls[(row["arm"], row["view"])] += 1
    summary["fallback_steps_by_arm_view"] = {"|".join(key): value for key, value in sorted(falls.items())}
    if summary["total_model_bytes_from_loads"] != summary["total_model_bytes_from_steps"]:
        raise RuntimeError("global model-load byte conservation failed")
    _write_json(OUTPUT / "summary.json", summary)
    files = ["preflight_receipt.json", "runner_entered.json", "branch_summary_rows.csv", "recorded_step_cost_rows.csv", "object_load_rows.csv", "summary.json"]
    files.extend(str(path.relative_to(OUTPUT)) for path in sorted(branches_dir.glob("*.json")))
    manifest = {"schema_version": "cscwd_abstention_cost_causal_manifest_v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete", "plan_sha256": _sha256(PLAN), "source_candidate_run_manifest_sha256": _sha256(CANDIDATE / "run_manifest.json"),
                "source_control_run_manifest_sha256": _sha256(CONTROL / "run_manifest.json"),
                "selected_sources": len(selected), "branch_count": len(branch_summary), **budget,
                "formal_holdout_reads": 0, "training_steps": 0,
                "files": {name: _sha256(OUTPUT / name) for name in files}}
    _write_json(OUTPUT / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "complete", "sources": len(selected), "branches": len(branch_summary), **budget}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one phase")
    preflight() if args.preflight else run()


if __name__ == "__main__":
    main()
