"""One-shot outcome-blind coverage of public phase-v2 prepare decision regions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces  # noqa: E402
from scripts.confirm_cscwd_phase_v2_repeated_action4 import (  # noqa: E402
    PILOT_BEHAVIOR_SHA, branch_metrics, canonical_sha, choose_continuation, event_row,
    public_decision_facts, public_projection, read_csv, read_json,
    replay_matches_pilot, sha, verify_inputs, write_json,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402


RUN_ID = "cscwd_phase_v2_public_prepare_regions_20261011_v1"
RULE_BEHAVIOR_SHA = "229ba1c059f7dd43b2948cc23ae27a952c9fa20bdfdcafdcb671b25089c8704b"
SOURCES = ("lankershim", "us_101")
CONDITIONS = ("default", "adverse_link", "ample_cache")
PREFIXES = ((), (0,), (1,), (4,))
ARM_PRIORITY = {"sa_candidate": 0, "sa_control": 1, "mappo": 2, "ppo": 3,
                "public_rule": 4}
MAX_STARTS = 12
MAX_BRANCHES = 36
MAX_ENV_STEPS = 288
MAX_CONTINUATION = 7
EPS = 1e-6


def episode_key(row: dict[str, str]) -> tuple[str, ...]:
    return (row["source_segment_id"], row["condition_id"], row["arm"], row["view"],
            row["seed"], row["window_id"], row["method"])


def precedence(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    row = candidate["row"]
    seed = int(row["seed"]) if row["seed"].isdigit() else 9999
    return (ARM_PRIORITY[row["arm"]], 0 if row["view"] == "selected" else 1,
            seed, row["method"])


def enumerate_public_starts(learned: list[dict[str, str]], rules: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    episodes: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in learned:
        if row["time_profile"] == "raw_ngsim_event_time_v2" and row["view"] == "selected":
            episodes[episode_key(row)].append(row)
    for row in rules:
        if (row["time_profile"] == "raw_ngsim_event_time_v2" and row["view"] == "fixed_rule"
                and row["arm"] == "public_rule"):
            episodes[episode_key(row)].append(row)
    candidates: defaultdict[tuple[str, str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    deeper: defaultdict[tuple[str, str, tuple[int, ...]], int] = defaultdict(int)
    for key, episode in episodes.items():
        episode.sort(key=lambda item: int(item["step"]))
        if [int(item["step"]) for item in episode] != list(range(len(episode))):
            raise RuntimeError("noncontiguous source episode in coverage inventory")
        prefix: list[int] = []
        for index, row in enumerate(episode):
            mask = json.loads(row["action_mask"])
            if len(mask) != 5 or not mask[int(row["executed_action"])]:
                raise RuntimeError("source public action mask drift")
            cell = (key[0], key[1], tuple(prefix))
            if len(prefix) <= 1:
                candidates[cell].append({"row": row, "history": episode[:index], "mask": mask})
            elif mask[3] and mask[4]:
                deeper[cell] += 1
            prefix.append(int(row["executed_action"]))
    inventory = []
    selected = []
    for source in SOURCES:
        for condition in CONDITIONS:
            for prefix in PREFIXES:
                cell = (source, condition, prefix)
                rows = candidates.get(cell, [])
                legal = [item for item in rows if item["mask"][3] and item["mask"][4]]
                if rows and any(item["mask"] != rows[0]["mask"] for item in rows):
                    raise RuntimeError("same prefix exposes inconsistent action masks")
                record = {"source_segment_id": source, "condition_id": condition,
                          "prefix_actions": list(prefix), "observed_row_count": len(rows),
                          "eligible_row_count": len(legal),
                          "public_action_mask": rows[0]["mask"] if rows else None}
                if not rows:
                    record["decision"] = "excluded_not_observed_in_frozen_behavior"
                elif not legal:
                    record["decision"] = "excluded_actions_3_and_4_not_both_legal"
                elif len(selected) >= MAX_STARTS:
                    record["decision"] = "excluded_frozen_12_start_cap"
                else:
                    chosen = min(legal, key=precedence)
                    original = int(chosen["row"]["executed_action"])
                    actions = list(dict.fromkeys((original, 3, 4)))
                    if not all(chosen["mask"][action] for action in actions):
                        raise RuntimeError("registered original branch action is illegal")
                    record.update(decision="selected", original_action=original,
                                  original_arm=chosen["row"]["arm"],
                                  original_view=chosen["row"]["view"],
                                  original_seed=chosen["row"]["seed"], branch_actions=actions,
                                  regression_exposed=prefix == (4,))
                    selected.append({**record, "history": chosen["history"], "row": chosen["row"]})
                inventory.append(record)
    for (source, condition, prefix), count in sorted(deeper.items()):
        inventory.append({"source_segment_id": source, "condition_id": condition,
                          "prefix_actions": list(prefix), "observed_row_count": count,
                          "eligible_row_count": count,
                          "decision": "excluded_prefix_depth_gt_one"})
    if len(selected) != 11 or sum(row["regression_exposed"] for row in selected) != 3:
        raise RuntimeError("frozen 11-start / 3-regression public coverage drift")
    if sum(len(row["branch_actions"]) for row in selected) > MAX_BRANCHES:
        raise RuntimeError("36-branch cap would be exceeded")
    max_possible_steps = sum(len(row["prefix_actions"]) +
                             len(row["branch_actions"]) * (MAX_CONTINUATION + 1)
                             for row in selected)
    if max_possible_steps > MAX_ENV_STEPS:
        raise RuntimeError("288-step cap could be exceeded")
    return selected, inventory


def public_features(info: dict[str, Any]) -> dict[str, Any]:
    facts = public_decision_facts(info)
    state = info["semantic_state"]
    context = state["calibrated_context"]
    predictions = state["predictions"]
    facts.update(
        estimated_current_contact_seconds=context.get("contact_budget_seconds"),
        prediction_uncertainty=predictions.get("prediction_uncertainty_by_vehicle", {}).get("veh_pilot"),
        remaining_deadline_seconds=context["time_contract"]["remaining_deadline_seconds"],
    )
    return facts


def prefix_history_pairs(prefix: list[int], history: list[dict[str, str]]) -> list[tuple[int, dict[str, str]]]:
    """Pair frozen actions and rows without relying on Python 3.10 zip(strict=...)."""
    if len(prefix) != len(history):
        raise RuntimeError("registered prefix/history length mismatch")
    return list(zip(prefix, history))


def posthoc_migration_use(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    first = ledger[0]
    target = None
    for phase in first["phase_events"]:
        if phase["phase"] == "state_transfer":
            target = str(phase["location"]).split("_to_")[-1]
            break
    used = False
    later_recompute_at_target = False
    if first["migration_success"] and target:
        for event in ledger[1:]:
            phases = event["phase_events"]
            if any(phase["phase"] == "current_node_compute" and phase["location"] == target
                   for phase in phases):
                used = True
                later_recompute_at_target = any(phase["phase"] == "prefix_recompute" for phase in phases)
                break
    return {"first_migration_status": first["migration_status"],
            "first_migration_success": first["migration_success"],
            "first_target_rsu_id_after_execution": target,
            "actual_target_departed_before_delivery": first["migration_status"] == "target_departed_before_delivery",
            "later_target_compute_observed": used,
            "later_target_compute_without_recompute": used and not later_recompute_at_target}


def region(action4: dict, action3: dict, migration: dict) -> str:
    if action4["censored_at_continuation_cap"] or action3["censored_at_continuation_cap"]:
        return "UNCERTAIN_CENSORED"
    service_nonharm = (int(action4["workflow_completed"]) >= int(action3["workflow_completed"])
                       and int(action4["on_time"]) >= int(action3["on_time"])
                       and action4["service_failures"] <= action3["service_failures"])
    strict_service = (int(action4["workflow_completed"]) > int(action3["workflow_completed"])
                      or int(action4["on_time"]) > int(action3["on_time"])
                      or action4["service_failures"] < action3["service_failures"])
    if not service_nonharm:
        return "SERVICE_HARM"
    if service_nonharm and strict_service:
        return "POSITIVE_SERVICE_WITH_COST_ACCOUNTED"
    if (service_nonharm and migration["later_target_compute_without_recompute"]
            and action4["recompute_seconds"] < action3["recompute_seconds"] - EPS):
        return "POSITIVE_MECHANISM_WITH_RESOURCE_TRADEOFF"
    if (not strict_service and action4["recompute_seconds"] >= action3["recompute_seconds"] - EPS
            and action4["elapsed_seconds"] >= action3["elapsed_seconds"] - EPS
            and action4["total_transfer_bytes"] > action3["total_transfer_bytes"]):
        return "NO_BENEFIT_COSTLY"
    return "MIXED_OR_UNCERTAIN"


def step_once(env: RawNGSIMPhaseV2Env, action: int, before: dict, *, start_id: str,
              condition_id: str, branch: str, budget: dict[str, int]) -> tuple[dict, dict, bool, bool]:
    if budget["env_steps"] >= MAX_ENV_STEPS:
        raise RuntimeError("288 environment-step hard cap reached")
    if not before["action_mask"][action]:
        raise RuntimeError("masked branch action")
    clock, index = env.clock_seconds, env.step_index
    _, _, terminated, truncated, after = env.step(action)
    budget["env_steps"] += 1
    event = dict(after["transition"])
    if int(event["action"]) != action or env.step_index != index + 1:
        raise RuntimeError("environment changed action or step identity")
    if not math.isclose(env.clock_seconds - clock, float(event["step_cost_seconds"]),
                        abs_tol=EPS, rel_tol=0.0):
        raise RuntimeError("real environment clock does not conserve")
    result = event_row(condition_id, branch, index, before, event, terminated, truncated)
    result["start_id"] = start_id
    return result, after, terminated, truncated


def run(protocol_path: Path, pilot_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError("create-only region output exists")
    plan = ROOT / "docs/project/cscwd_phase_v2_public_prepare_region_plan_20261011.md"
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                      text=True, check=True).stdout:
        raise RuntimeError("public region diagnostic requires a clean frozen commit")
    protocol, paths = verify_inputs(protocol_path, pilot_root)
    if sha(pilot_root / "rule_behavior_rows.csv") != RULE_BEHAVIOR_SHA:
        raise RuntimeError("frozen public-rule behavior identity mismatch")
    selected, inventory = enumerate_public_starts(
        read_csv(pilot_root / "behavior_rows.csv"), read_csv(pilot_root / "rule_behavior_rows.csv"))
    output_root.mkdir(parents=True)
    receipt = {"run_id": RUN_ID, "status": "STARTED",
               "started_at": datetime.now(timezone.utc).isoformat(),
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout.strip(),
               "plan_sha256": sha(plan), "pilot_behavior_sha256": PILOT_BEHAVIOR_SHA,
               "rule_behavior_sha256": RULE_BEHAVIOR_SHA,
               "selected_start_count": len(selected), "max_starts": MAX_STARTS,
               "max_branches": MAX_BRANCHES, "max_env_steps": MAX_ENV_STEPS}
    write_json(output_root / "start_receipt.json", receipt)
    write_json(output_root / "selection_inventory.json", inventory)
    budget = {"env_steps": 0}
    try:
        config, _ = _load_experiment_config(paths["base_config"])
        config["reward_profile"] = "original_reward_v1"
        config["interface_profile"] = "calibrated_workflow_interface_v4_prepared_state_prefix"
        workload = read_json(paths["workload_manifest"])
        instance = next(deepcopy(item) for item in workload["instances"] if item["design_id"] == "dev_01")
        phase = read_json(paths["phase_contract"])
        source = read_json(paths["source_manifest"])
        traces, trace_identity = load_traces(paths["raw_csv_path"], source["selected_windows"])
        trace_by_source = {item["source_segment_id"]: traces[item["window_id"]]
                           for item in source["selected_windows"]}
        if set(trace_by_source) != set(SOURCES):
            raise RuntimeError("both frozen development traces not reconstructed")
        starts = []
        branches = []
        events = []
        seen_public_sha = set()
        for item in selected:
            source_id, condition_id = item["source_segment_id"], item["condition_id"]
            prefix = item["prefix_actions"]
            start_id = f"{source_id}:{condition_id}:{','.join(map(str, prefix)) or 'reset'}"
            condition = next(row for row in phase["conditions"] if row["condition_id"] == condition_id)
            env = RawNGSIMPhaseV2Env(config, deepcopy(instance), trace_by_source[source_id],
                                     phase_contract=phase["physical_assumptions"], condition=condition)
            _, info = env.reset()
            prefix_events = []
            for action, old in prefix_history_pairs(prefix, item["history"]):
                step, info, done, truncated = step_once(env, action, info,
                                                        start_id=start_id, condition_id=condition_id,
                                                        branch="common_prefix", budget=budget)
                replay_matches_pilot(step, old)
                prefix_events.append(step)
                if done or truncated:
                    raise RuntimeError("selected public prefix ended before decision")
            if list(info["action_mask"]) != item["public_action_mask"]:
                raise RuntimeError("selected public action mask differs from archived mask")
            public_sha = canonical_sha(public_projection(info))
            if public_sha in seen_public_sha:
                raise RuntimeError("selected public states are not unique")
            seen_public_sha.add(public_sha)
            start = {key: item[key] for key in (
                "source_segment_id", "condition_id", "prefix_actions", "original_action",
                "original_arm", "original_view", "original_seed", "branch_actions", "regression_exposed")}
            start.update(start_id=start_id, public_state_sha256=public_sha,
                         public_features=public_features(info), prefix_steps=len(prefix_events))
            starts.append(start)
            events.extend(prefix_events)
            branch_by_action = {}
            for action in item["branch_actions"]:
                branch_env = deepcopy(env)
                branch_info = deepcopy(info)
                branch = f"action{action}"
                ledger = []
                step, branch_info, done, truncated = step_once(
                    branch_env, action, branch_info, start_id=start_id, condition_id=condition_id,
                    branch=branch, budget=budget)
                ledger.append(step)
                count = 0
                while not (done or truncated) and count < MAX_CONTINUATION:
                    next_action = choose_continuation(list(branch_info["action_mask"]))
                    step, branch_info, done, truncated = step_once(
                        branch_env, next_action, branch_info, start_id=start_id, condition_id=condition_id,
                        branch=branch, budget=budget)
                    ledger.append(step)
                    count += 1
                metrics = branch_metrics(ledger, branch_env.summary(), censored=not(done or truncated))
                migration = posthoc_migration_use(ledger)
                branch_by_action[action] = {"start_id": start_id, "first_action": action,
                                             "public_state_sha256": public_sha,
                                             "branch_env_steps": len(ledger), "metrics": metrics,
                                             "posthoc_migration": migration}
                branches.append(branch_by_action[action])
                events.extend(ledger)
            if 3 not in branch_by_action or 4 not in branch_by_action:
                raise RuntimeError("missing matched action3/action4 branch")
        if len(starts) > MAX_STARTS or len(branches) > MAX_BRANCHES or budget["env_steps"] > MAX_ENV_STEPS:
            raise RuntimeError("bounded public region execution exceeded cap")
        paired = []
        by_start: defaultdict[str, dict[int, dict]] = defaultdict(dict)
        for branch in branches:
            by_start[branch["start_id"]][branch["first_action"]] = branch
        for start in starts:
            arms = by_start[start["start_id"]]
            a3, a4 = arms[3]["metrics"], arms[4]["metrics"]
            contrasts = {key: a4[key] - a3[key] for key in (
                "service_failures", "migration_successes", "total_transfer_bytes",
                "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
                "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds",
                "elapsed_seconds", "reward")}
            contrasts["workflow_completed"] = int(a4["workflow_completed"]) - int(a3["workflow_completed"])
            contrasts["on_time"] = int(a4["on_time"]) - int(a3["on_time"])
            paired.append({"start_id": start["start_id"], "source_segment_id": start["source_segment_id"],
                           "condition_id": start["condition_id"], "prefix_actions": start["prefix_actions"],
                           "regression_exposed": start["regression_exposed"],
                           "public_state_sha256": start["public_state_sha256"],
                           "original_first_action": start["original_action"],
                           "original_arm": start["original_arm"],
                           "action4_minus_action3": contrasts,
                           "action4_region": region(a4, a3, arms[4]["posthoc_migration"]),
                           "action4_posthoc_migration": arms[4]["posthoc_migration"]})
        write_json(output_root / "starts.json", starts)
        write_json(output_root / "branches.json", branches)
        write_json(output_root / "paired_regions.json", paired)
        with (output_root / "event_ledger.jsonl").open("w") as handle:
            for row in events:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        counts: defaultdict[str, int] = defaultdict(int)
        for item in paired:
            counts[item["action4_region"]] += 1
        summary = {"run_id": RUN_ID, "status": "COMPLETE", "development_only": True,
                   "source_run_id": "cscwd_phase_v2_causal_public_pilot_20261011_v1",
                   "selection_inventory_count": len(inventory), "selected_start_count": len(starts),
                   "regression_exposed_start_count": sum(item["regression_exposed"] for item in starts),
                   "new_coverage_start_count": sum(not item["regression_exposed"] for item in starts),
                   "branch_count": len(branches), "environment_step_count": budget["env_steps"],
                   "event_ledger_rows": len(events), "region_counts": dict(sorted(counts.items())),
                   "trace_identity": trace_identity, "formal_or_holdout_reads": 0,
                   "training_steps": 0, "automatic_retry": False}
        write_json(output_root / "summary.json", summary)
        payloads = ("start_receipt.json", "selection_inventory.json", "starts.json", "branches.json",
                    "paired_regions.json", "event_ledger.jsonl", "summary.json")
        write_json(output_root / "artifact_integrity.json", {"files": [
            {"path": name, "bytes": (output_root / name).stat().st_size,
             "sha256": sha(output_root / name)} for name in payloads]})
        receipt.update(status="COMPLETE", finished_at=datetime.now(timezone.utc).isoformat(),
                       branch_count=len(branches), environment_step_count=budget["env_steps"],
                       summary_sha256=sha(output_root / "summary.json"),
                       integrity_sha256=sha(output_root / "artifact_integrity.json"))
        write_json(output_root / "completion_receipt.json", receipt)
        return summary
    except Exception as exc:
        receipt.update(status="FAIL", failed_at=datetime.now(timezone.utc).isoformat(),
                       environment_step_count=budget["env_steps"], error_type=type(exc).__name__,
                       error=str(exc), automatic_retry=False)
        write_json(output_root / "failure_receipt.json", receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", required=True, type=Path)
    parser.add_argument("--pilot-run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.pilot_protocol.resolve(), args.pilot_run_root.resolve(), args.output_root.resolve())
    print(json.dumps({"run_id": RUN_ID, "status": result["status"],
                      "selected_start_count": result["selected_start_count"],
                      "branch_count": result["branch_count"],
                      "environment_step_count": result["environment_step_count"],
                      "summary_sha256": sha(args.output_root.resolve() / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
