"""Read-only, single-pass replay of the frozen prepared-state development run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_sa_behavior import _bool, _public_state_hash
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _load_inputs, _load_json
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv

B_ROOT = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC")
SOURCE = B_ROOT / "artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1"
TERMINAL = B_ROOT / "artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1_supervisor/terminal_receipt.json"
OUTPUT = ROOT / "artifacts/analysis/cscwd_prepared_state_event_chain_diagnosis_20261010_v1"
DESIGN = ROOT / "configs/experiment/calibrated_workflow_strong_baselines_development_v2_prefix_only.json"
SCIENCE_COMMIT = "f46ec72b15f534ac44768a83ef6316c1cfcb6b58"
PROFILE = "calibrated_workflow_interface_v4_prepared_state_prefix"
VIEWS = ("selected", "update96")
METHODS = ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty analysis table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _same(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-6)


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (str(row["checkpoint_view"]), str(row["split"]), str(row["method"]),
            str(row["seed"]), str(row["design_id"]))


def _preflight() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, ...], dict[str, str]], dict[tuple[str, ...], list[dict[str, str]]], dict[str, Any]]:
    terminal = _load_json(TERMINAL)
    completion = _load_json(SOURCE / "completion_receipt.json")
    manifest = _load_json(SOURCE / "run_manifest.json")
    if terminal.get("status") != "PASS" or terminal.get("return_code") != 0:
        raise RuntimeError("scientific supervisor not PASS")
    if completion.get("status") != "complete" or completion.get("scientific_execution_complete") is not True:
        raise RuntimeError("scientific completion receipt missing")
    if manifest.get("git_commit") != SCIENCE_COMMIT or manifest.get("interface_profile") != PROFILE:
        raise RuntimeError("scientific identity/profile mismatch")
    if manifest.get("formal_or_holdout_reads") != 0:
        raise RuntimeError("unexpected formal/holdout reads")
    protocol = manifest["protocol"]
    if _sha256(B_ROOT / protocol["path"]) != protocol["sha256"]:
        raise RuntimeError("frozen protocol hash mismatch")
    inventory = _load_json(SOURCE / "artifact_integrity.json")["files"]
    if len(inventory) != 119:
        raise RuntimeError("scientific inventory count drift")
    for item in inventory:
        path = SOURCE / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"scientific artifact mismatch: {item['path']}")
    design = _load_json(DESIGN)
    config, splits, identity = _load_inputs(design)
    if identity["public_prefix_suffix_tamper_comparisons"] != 474:
        raise RuntimeError("public prefix audit drift")
    frozen = _load_json(B_ROOT / protocol["path"])["frozen_scientific_identity"]
    for field, expected in (("base_config", "base_config_sha256"), ("workload_manifest", "workload_manifest_sha256")):
        if _sha256(ROOT / frozen[field]) != frozen[expected]:
            raise RuntimeError(f"frozen {field} identity drift")
    config["interface_profile"] = PROFILE
    instances = {str(row["design_id"]): row for split in ("regression", "frozen_check") for row in splits[split]}
    evaluations: dict[tuple[str, ...], dict[str, str]] = {}
    episodes: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    counts: dict[str, Any] = {}
    for view in VIEWS:
        eval_rows = _csv(SOURCE / f"new_{view}_evaluation_rows.csv")
        ledger = _csv(SOURCE / f"new_{view}_behavior_ledger.csv")
        if len(eval_rows) != 400 or len({_key(row) for row in eval_rows}) != 400:
            raise RuntimeError(f"{view} episode count/identity drift")
        if {row["method"] for row in eval_rows} != set(METHODS):
            raise RuntimeError(f"{view} method set drift")
        for row in eval_rows:
            evaluations[_key(row)] = row
        for row in ledger:
            episodes[_key(row)].append(row)
        counts[view] = {"episodes": len(eval_rows), "steps": len(ledger)}
    if set(evaluations) != set(episodes):
        raise RuntimeError("evaluation/behavior episode mismatch")
    if len(evaluations) != 800 or sum(len(x) for x in episodes.values()) != 5632:
        raise RuntimeError("new evaluation replay count drift")
    if any(key[-1] not in instances for key in episodes):
        raise RuntimeError("unknown design identity")
    return config, instances, evaluations, episodes, {"source_commit": SCIENCE_COMMIT, "protocol_sha256": protocol["sha256"], "inventory_files_verified": len(inventory), "counts": counts, "formal_or_holdout_reads": 0}


def _failure_cause(action: int, current_ready: bool, cache_events: list[dict[str, Any]]) -> str:
    if current_ready:
        return "execution_inconsistency_or_unknown"
    if action in (1, 3, 4):
        return "current_bundle_missing_nonrepair_action"
    if action == 0:
        reasons = [str(e.get("reason", "unknown")) for e in cache_events]
        return "current_admission_" + (reasons[-1] if reasons else "unknown")
    return "execution_inconsistency_or_unknown"


def _handoff_cause(prior: list[dict[str, Any]], destination: str, ready: bool) -> str:
    if ready:
        return "prepared_state_ready"
    relevant = [event for event in prior if event["action"] == 4 and event["target_rsu"] == destination]
    if relevant:
        last = relevant[-1]
        if last["migration_success"]:
            return "prepared_prefix_stale_after_progress"
        if not last["service_completed"]:
            return "prepare_current_service_failed_no_commit"
        return "prepare_admission_or_contact_failed"
    if any(event["action"] == 4 and event["target_rsu"] != destination for event in prior):
        return "prepared_other_predicted_target"
    return "prepare_not_initiated"


def _replay(config: dict[str, Any], instances: dict[str, dict[str, Any]], evaluations: dict[tuple[str, ...], dict[str, str]], episodes: dict[tuple[str, ...], list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for key, records in sorted(episodes.items()):
        view, split, method, seed, design_id = key
        env = CalibratedContinuousWorkflowEnv(config, instances[design_id])
        observation, info = env.reset()
        history: list[dict[str, Any]] = []
        first_failure: dict[str, Any] | None = None
        for row in sorted(records, key=lambda value: int(value["step_index"])):
            step = int(row["step_index"])
            if env.terminated or env.step_index != step:
                raise RuntimeError(f"step mismatch: {key} {step}")
            current = env._current_rsu_id()
            target = env._predicted_handoff_target()
            node = env._current_node()
            required = str(node["required_adapter"])
            current_ready = env._bundle_ready(current, required)
            target_ready = bool(target and env._bundle_ready(target, required))
            if current != row["current_rsu_id"] or (target or "") != row["target_rsu_id"]:
                raise RuntimeError(f"RSU mismatch: {key} {step}")
            if current_ready != _bool(row["current_bundle_ready"]) or target_ready != _bool(row["target_bundle_ready"]):
                raise RuntimeError(f"bundle readiness mismatch: {key} {step}")
            prefix = info["semantic_state"]["calibrated_context"]["prepared_state_prefix"]
            for scope in ("current", "predicted_target"):
                stem = "current" if scope == "current" else "target"
                for field in ("known", "exists", "valid"):
                    if bool(prefix[scope][field]) != _bool(row[f"{stem}_prepared_{field}"]):
                        raise RuntimeError(f"public prepared field mismatch: {key} {step} {scope}.{field}")
                for field in ("missing_completed_count", "missing_completed_fraction"):
                    if not _same(prefix[scope][field], row[f"{stem}_prepared_{field}"]):
                        raise RuntimeError(f"public prepared count mismatch: {key} {step} {scope}.{field}")
            action = int(row["executed_action"])
            if not info["action_mask"][action]:
                raise RuntimeError(f"recorded action outside mask: {key} {step}")
            before_completed = len(env.completed)
            public_contact = float(env._contact_budget_seconds())
            physical_contact = float(env._physical_contact_budget_seconds())
            public_hash = _public_state_hash(observation, info)
            observation, _, _, _, info = env.step(action)
            transition = info["transition"]
            for field in ("service_completed", "migration_success", "state_ready"):
                if bool(transition[field]) != _bool(row[field]):
                    raise RuntimeError(f"transition mismatch: {key} {step} {field}")
            if (len(env.completed) > before_completed) != _bool(row["progressed"]):
                raise RuntimeError(f"progress mismatch: {key} {step}")
            for field in ("model_transfer_bytes", "state_transfer_bytes", "recompute_seconds", "service_operation_seconds", "clock_seconds_after"):
                if not _same(transition[field], row[field]):
                    raise RuntimeError(f"metric mismatch: {key} {step} {field}")
            cache_events = list(transition["cache_events"])
            causes = _failure_cause(action, current_ready, cache_events) if not transition["service_completed"] else ""
            if causes and first_failure is None:
                first_failure = {"step_index": step, "first_failure_cause": causes, "first_failure_action": action,
                                 "first_failure_current_ready": current_ready, "first_failure_target": target}
            actual_next = instances[design_id]["rsu_sequence"][min(step + 1, len(instances[design_id]["rsu_sequence"]) - 1)]
            future = instances[design_id]["rsu_sequence"][step + 1: step + 1 + int(config["prediction_horizon"])]
            actual_first = next((str(candidate) for candidate in future if str(candidate) != current), None)
            event = {"checkpoint_view": view, "split": split, "method": method, "seed": seed, "design_id": design_id,
                     "step_index": step, "public_state_sha256": public_hash, "node_id": str(node["node_id"]),
                     "current_rsu": current, "target_rsu": target or "", "actual_next_rsu_posthoc": actual_next,
                     "actual_first_handoff_target_posthoc": actual_first or "", "forecast_matches_first_handoff_posthoc": bool(target and target == actual_first),
                     "action": action, "current_bundle_ready": current_ready, "target_bundle_ready": target_ready,
                     "current_prepared_exists": bool(prefix["current"]["exists"]), "current_prepared_valid": bool(prefix["current"]["valid"]),
                     "target_prepared_exists": bool(prefix["predicted_target"]["exists"]), "target_prepared_valid": bool(prefix["predicted_target"]["valid"]),
                     "public_contact_budget_seconds": public_contact,
                     "physical_contact_budget_seconds_posthoc": physical_contact,
                     "service_completed": bool(transition["service_completed"]), "failure_cause": causes,
                     "migration_success": bool(transition["migration_success"]), "state_transfer_status": (transition["state_transfer"] or {}).get("status", ""),
                     "state_ready": bool(transition["state_ready"]), "handoff": bool(transition["handoff"]),
                     "progressed": len(env.completed) > before_completed, "completed_count_after": len(env.completed),
                     "cache_events_json": json.dumps(cache_events, ensure_ascii=False, sort_keys=True),
                     "model_transfer_bytes": int(transition["model_transfer_bytes"]), "state_transfer_bytes": int(transition["state_transfer_bytes"]),
                     "recompute_seconds": float(transition["recompute_seconds"]), "clock_seconds_after": float(transition["clock_seconds_after"]),
                     "target_prepare_feasible_preview": _bool(row["target_prepare_feasible"]), "target_prepare_reason_preview": row["prepare_feasibility_reason"],
                     "env_action_probs": row["env_action_probs"]}
            events.append(event)
            if event["handoff"]:
                source = history[-1]["current_rsu"] if history else ""
                prior = [old for old in history if old["current_rsu"] == source]
                relevant = [old for old in prior if old["action"] == 4 and old["target_rsu"] == current]
                last = relevant[-1] if relevant else None
                handoffs.append({"checkpoint_view": view, "split": split, "method": method, "seed": seed, "design_id": design_id,
                                 "arrival_step": step, "source_rsu": source, "destination_rsu": current, "arrival_action": action,
                                 "arrival_service_completed": event["service_completed"], "arrival_current_bundle_ready": current_ready,
                                 "arrival_state_ready": event["state_ready"], "arrival_recompute_seconds": event["recompute_seconds"],
                                 "arrival_public_current_prepared_exists": bool(prefix["current"]["exists"]),
                                 "arrival_public_current_prepared_valid": bool(prefix["current"]["valid"]),
                                 "prior_prepare_count_for_destination": len(relevant), "last_prepare_step_for_destination": last["step_index"] if last else "",
                                 "last_prepare_committed": last["migration_success"] if last else "",
                                 "last_prepare_current_service_completed": last["service_completed"] if last else "",
                                 "last_prepare_cache_events_json": last["cache_events_json"] if last else "",
                                 "prior_forecast_match_count": sum(old["target_rsu"] == current for old in prior),
                                 "prior_other_target_action4_count": sum(old["action"] == 4 and old["target_rsu"] != current for old in prior),
                                 "state_break_cause": _handoff_cause(prior, current, event["state_ready"]),
                                 "public_state_sha256": public_hash})
            history.append(event)
        summary = env.summary()
        recorded = evaluations[key]
        for field, source in (("service_failures", "service_failures"), ("handoff_failures", "handoff_failures"),
                              ("migration_attempts", "migration_attempts"), ("migration_successes", "migration_successes"),
                              ("completed_nodes", "completed_nodes"), ("recompute_seconds", "recompute_seconds"),
                              ("modeled_completion_seconds", "modeled_completion_seconds")):
            if not _same(summary[source], recorded[field]):
                raise RuntimeError(f"episode summary mismatch: {key} {field}")
        if not _same(int(summary["workflow_completed"]), recorded["workflow_completion_rate"]):
            raise RuntimeError(f"completion mismatch: {key}")
        episode_rows.append({"checkpoint_view": view, "split": split, "method": method, "seed": seed, "design_id": design_id,
                             "source_segment_id": recorded["source_segment_id"], "window_id": recorded["window_id"],
                             "workflow_id": recorded["workflow_id"], "steps": len(records), "service_failures": int(summary["service_failures"]),
                             "workflow_completed": bool(summary["workflow_completed"]), "on_time_workflow_completed": bool(summary["on_time_workflow_completed"]),
                             "handoff_count": int(summary["handoff_count"]), "handoff_failures": int(summary["handoff_failures"]),
                             "recompute_seconds": float(summary["recompute_seconds"]), "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
                             "model_transfer_mb": float(summary["model_transfer_bytes"]) / 1e6,
                             "state_transfer_mb": float(summary["state_transfer_bytes"]) / 1e6,
                             "total_transfer_mb": float(summary["total_transfer_bytes"]) / 1e6,
                             "first_failure_step": first_failure["step_index"] if first_failure else "",
                             "first_failure_cause": first_failure["first_failure_cause"] if first_failure else "",
                             "first_failure_action": first_failure["first_failure_action"] if first_failure else "",
                             "first_failure_current_ready": first_failure["first_failure_current_ready"] if first_failure else "",
                             "first_failure_target": first_failure["first_failure_target"] if first_failure else ""})
    if len(events) != 5632 or len(episode_rows) != 800:
        raise RuntimeError("replay result count mismatch")
    return events, handoffs, episode_rows, {"episodes_replayed": len(episode_rows), "steps_replayed": len(events), "handoff_arrivals": len(handoffs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    config, instances, evaluation, episodes, receipt = _preflight()
    if args.preflight:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return
    if OUTPUT.exists():
        raise RuntimeError("analysis output already exists; single-pass replay only")
    events, handoffs, episode_rows, replay = _replay(config, instances, evaluation, episodes)
    OUTPUT.mkdir(parents=True)
    _write_csv(OUTPUT / "event_chain_rows.csv", events)
    _write_csv(OUTPUT / "handoff_chain_rows.csv", handoffs)
    _write_csv(OUTPUT / "episode_diagnosis_rows.csv", episode_rows)
    receipt.update(replay)
    receipt.update({"schema_version": "cscwd_prepared_state_event_chain_diagnosis_v1",
                    "created_at": datetime.now(timezone.utc).isoformat(), "optional_action_branches": 0,
                    "output_files": {name: _sha256(OUTPUT / name) for name in
                                     ("event_chain_rows.csv", "handoff_chain_rows.csv", "episode_diagnosis_rows.csv")}})
    _write_json(OUTPUT / "analysis_manifest.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
