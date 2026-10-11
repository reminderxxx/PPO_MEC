"""Future single-run CSCWD region executor with full-state grouping and durable branches.

The CLI requires a separate, committed authorization manifest. This file is not
an authorization to run the real development traces.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces  # noqa: E402
from scripts.confirm_cscwd_phase_v2_public_prepare_regions import (  # noqa: E402
    MAX_BRANCHES, MAX_ENV_STEPS, MAX_STARTS, RULE_BEHAVIOR_SHA,
    SOURCES, enumerate_public_starts, posthoc_migration_use, prefix_history_pairs,
    public_features, region, step_once,
)
from scripts.confirm_cscwd_phase_v2_repeated_action4 import (  # noqa: E402
    branch_metrics, read_csv, read_json, replay_matches_pilot,
    sha, verify_inputs, write_json,
)
from scripts.cscwd_region_v3_evidence import (  # noqa: E402
    BRANCH_RECORD_VERSION, STATE_IDENTITY_VERSION, atomic_record,
    classify_identities, execute_and_persist, partial_evidence_status,
    read_verified_record, state_identities,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402


RUN_ID = "cscwd_phase_v2_public_prepare_regions_20261011_v3"
ORIGINAL_PLAN_SHA = "472808fc90a3af16f8fe2c4f76acb69a9921a70f50d02184baa1eb40ec9c1def"
V1_FAILURE_SHA = "55b826571c4b2474bf1f52ab228892d37855a850425ce20249b942100127e1ad"
V2_FAILURE_SHA = "4a19c502d5c5fb298601c211ed0fdb07d8a6b0b01ff829e164fc4bfda605c088"
SELECTION_SHA = "875163d7022ce209746c1e7dd954891d12da2f92f640cac7e815a0604a2d54cb"
EXECUTOR_BASE_COMMIT = "2705fa49a92ff79b83f7a09fc750221fdfe2ebaf"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_future_authorization(path: Path) -> tuple[dict[str, Any], str]:
    """Accept only a later explicit v3 manifest, never a v1/v2 receipt."""
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError("v3 authorization manifest must be in this repository") from exc
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
                      capture_output=True, check=False).returncode != 0:
        raise RuntimeError("v3 authorization manifest must be tracked in the frozen commit")
    manifest = read_json(path)
    required = {
        "run_id": RUN_ID,
        "status": "AUTHORIZED_FOR_SINGLE_V3_EXECUTION",
        "executor_base_commit": EXECUTOR_BASE_COMMIT,
        "original_plan_sha256": ORIGINAL_PLAN_SHA,
        "v1_failure_receipt_sha256": V1_FAILURE_SHA,
        "v2_failure_receipt_sha256": V2_FAILURE_SHA,
        "prior_selection_inventory_sha256": SELECTION_SHA,
        "candidate_cell_count": 24,
        "selected_public_cell_count": 11,
        "previously_exposed_regression_cell_count": 3,
        "new_coverage_same_us101_trace_cell_count": 8,
        "max_starts": MAX_STARTS,
        "max_branches": MAX_BRANCHES,
        "max_env_steps_including_prefix_replay": MAX_ENV_STEPS,
        "state_identity_version": STATE_IDENTITY_VERSION,
        "prefix_replay_steps_are_scientific_steps": True,
        "training_model_calls_formal_holdout_downloads": False,
        "automatic_retry": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("v3 explicit authorization manifest mismatch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXECUTOR_BASE_COMMIT, "HEAD"],
                      cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("v3 executor base commit is not in execution history")
    if sha(ROOT / "docs/project/cscwd_phase_v2_public_prepare_region_plan_20261011.md") != ORIGINAL_PLAN_SHA:
        raise RuntimeError("original scientific plan identity mismatch")
    for path_name, expected in (
        ("v1/failure_receipt.json", V1_FAILURE_SHA),
        ("v2/failure_receipt.json", V2_FAILURE_SHA),
    ):
        version, file_name = path_name.split("/")
        old = ROOT / f"artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_{version}" / file_name
        if sha(old) != expected:
            raise RuntimeError("prior failure receipt identity mismatch")
    if sha(ROOT / "artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v2/selection_inventory.json") != SELECTION_SHA:
        raise RuntimeError("prior public selection inventory identity mismatch")
    return manifest, sha(path)


def run(protocol_path: Path, pilot_root: Path, output_root: Path, authorization_path: Path) -> dict[str, Any]:
    if output_root != ROOT / "artifacts/analysis" / RUN_ID:
        raise RuntimeError("v3 output root does not match frozen run identity")
    if output_root.exists():
        raise FileExistsError("create-only v3 output root already exists")
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout:
        raise RuntimeError("v3 execution requires clean committed code")
    _, authorization_sha = verify_future_authorization(authorization_path)
    _, paths = verify_inputs(protocol_path, pilot_root)
    if sha(pilot_root / "rule_behavior_rows.csv") != RULE_BEHAVIOR_SHA:
        raise RuntimeError("public-rule behavior identity mismatch")
    selected, inventory = enumerate_public_starts(
        read_csv(pilot_root / "behavior_rows.csv"), read_csv(pilot_root / "rule_behavior_rows.csv"))
    if len(inventory) != 24 or len(selected) != 11:
        raise RuntimeError("frozen candidate coverage drift")
    if inventory != read_json(ROOT / "artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v2/selection_inventory.json"):
        raise RuntimeError("v3 selection differs from frozen v2 inventory")
    for item in selected:
        prefix_history_pairs(item["prefix_actions"], item["history"])

    output_root.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "run_id": RUN_ID, "status": "STARTED", "started_at": utc_now(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout.strip(),
        "authorization_sha256": authorization_sha,
        "original_plan_sha256": ORIGINAL_PLAN_SHA,
        "v1_failure_receipt_sha256": V1_FAILURE_SHA,
        "v2_failure_receipt_sha256": V2_FAILURE_SHA,
        "state_identity_version": STATE_IDENTITY_VERSION,
        "max_starts": MAX_STARTS, "max_branches": MAX_BRANCHES,
        "max_env_steps_including_prefix_replay": MAX_ENV_STEPS,
    }
    write_json(output_root / "start_receipt.json", receipt)
    write_json(output_root / "selection_inventory.json", inventory)
    budget = {"env_steps": 0}
    current_branch: str | None = None
    try:
        config, _ = _load_experiment_config(paths["base_config"])
        config["reward_profile"] = "original_reward_v1"
        config["interface_profile"] = "calibrated_workflow_interface_v4_prepared_state_prefix"
        workload = read_json(paths["workload_manifest"])
        instance = next(deepcopy(row) for row in workload["instances"] if row["design_id"] == "dev_01")
        phase = read_json(paths["phase_contract"])
        source = read_json(paths["source_manifest"])
        traces, trace_identity = load_traces(paths["raw_csv_path"], source["selected_windows"])
        trace_by_source = {item["source_segment_id"]: traces[item["window_id"]]
                           for item in source["selected_windows"]}
        if set(trace_by_source) != set(SOURCES):
            raise RuntimeError("frozen trace reconstruction mismatch")

        prepared = []
        identities = []
        for index, item in enumerate(selected):
            source_id, condition_id = item["source_segment_id"], item["condition_id"]
            prefix = item["prefix_actions"]
            start_id = f"{source_id}:{condition_id}:{','.join(map(str, prefix)) or 'reset'}"
            start_id_safe = f"s{index:02d}"
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
                    raise RuntimeError("selected prefix ended before decision")
            if list(info["action_mask"]) != item["public_action_mask"]:
                raise RuntimeError("selected public action mask drift")
            identity = {
                "start_id": start_id, "start_id_safe": start_id_safe,
                "source_segment_id": source_id, "condition_id": condition_id,
                "prefix_actions": prefix, "original_action": item["original_action"],
                "original_arm": item["original_arm"], "original_view": item["original_view"],
                "original_seed": item["original_seed"], "branch_actions": item["branch_actions"],
                "regression_exposed": item["regression_exposed"],
                "public_features": public_features(info), "prefix_env_steps": len(prefix_events),
                **state_identities(env, info),
            }
            atomic_record(output_root / "prefixes" / f"{start_id_safe}.json",
                          {"identity": identity, "events": prefix_events}, version=STATE_IDENTITY_VERSION)
            identities.append(identity)
            prepared.append((item, env, info))

        identities = classify_identities(identities)
        atomic_record(output_root / "identity_inventory.json",
                      {"selected_cell_count": len(identities), "prefix_env_steps": budget["env_steps"],
                       "rows": identities}, version=STATE_IDENTITY_VERSION)
        prefix_env_steps = budget["env_steps"]
        representative: dict[str, str] = {
            row["start_id"]: row["full_duplicate_of"] or row["start_id"] for row in identities
        }
        actions_by_representative: defaultdict[str, list[int]] = defaultdict(list)
        for row in identities:
            actions = actions_by_representative[representative[row["start_id"]]]
            for action in row["branch_actions"]:
                if action not in actions:
                    actions.append(action)
        tasks = []
        for row, (_, env, info) in zip(identities, prepared):
            if row["full_duplicate_of"] is not None:
                continue
            for action in actions_by_representative[row["start_id"]]:
                tasks.append({"start_id_safe": row["start_id_safe"], "start_id": row["start_id"],
                              "action": action, "identity": row, "env": env, "info": info})
        if len(tasks) > MAX_BRANCHES:
            raise RuntimeError("v3 representative branch cap exceeded")

        def execute_one(task: dict[str, Any]) -> dict[str, Any]:
            nonlocal current_branch
            current_branch = f"{task['start_id_safe']}:action{task['action']}"
            branch_env = deepcopy(task["env"])
            branch_info = deepcopy(task["info"])
            row = task["identity"]
            branch = f"action{task['action']}"
            ledger = []
            step, branch_info, done, truncated = step_once(
                branch_env, task["action"], branch_info,
                start_id=row["start_id"], condition_id=row["condition_id"],
                branch=branch, budget=budget)
            ledger.append(step)
            continuation = 0
            from scripts.confirm_cscwd_phase_v2_repeated_action4 import choose_continuation
            while not (done or truncated) and continuation < 7:
                next_action = choose_continuation(list(branch_info["action_mask"]))
                step, branch_info, done, truncated = step_once(
                    branch_env, next_action, branch_info,
                    start_id=row["start_id"], condition_id=row["condition_id"],
                    branch=branch, budget=budget)
                ledger.append(step)
                continuation += 1
            result = {
                "start_id": row["start_id"], "start_id_safe": row["start_id_safe"],
                "source_segment_id": row["source_segment_id"],
                "condition_id": row["condition_id"],
                "prefix_actions": row["prefix_actions"],
                "first_action": task["action"],
                "input_identity": {key: row[key] for key in (
                    "identity_version", "public_observation_sha256", "full_simulation_sha256")},
                "events": ledger,
                "metrics": branch_metrics(ledger, branch_env.summary(), censored=not (done or truncated)),
                "posthoc_migration": posthoc_migration_use(ledger),
            }
            return result

        execute_and_persist(tasks, execute_one, output_root / "branches")
        current_branch = None
        stored = {}
        for task in tasks:
            path = output_root / "branches" / f"{task['start_id_safe']}_action{task['action']}.json"
            result = read_verified_record(path, version=BRANCH_RECORD_VERSION)
            stored[(task["start_id"], task["action"])] = result
        paired = []
        counts: defaultdict[str, int] = defaultdict(int)
        distinct_counts: defaultdict[str, int] = defaultdict(int)
        for row in identities:
            representative_id = representative[row["start_id"]]
            a3 = stored[(representative_id, 3)]["metrics"]
            a4_result = stored[(representative_id, 4)]
            a4 = a4_result["metrics"]
            contrasts = {key: a4[key] - a3[key] for key in (
                "service_failures", "migration_successes", "total_transfer_bytes",
                "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
                "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds",
                "elapsed_seconds", "reward")}
            contrasts["workflow_completed"] = int(a4["workflow_completed"]) - int(a3["workflow_completed"])
            contrasts["on_time"] = int(a4["on_time"]) - int(a3["on_time"])
            label = region(a4, a3, a4_result["posthoc_migration"])
            counts[label] += 1
            if row["full_duplicate_of"] is None:
                distinct_counts[label] += 1
            paired.append({"start_id": row["start_id"], "representative_start_id": representative_id,
                           "full_duplicate_of": row["full_duplicate_of"],
                           "public_observation_group_id": row["public_observation_group_id"],
                           "source_segment_id": row["source_segment_id"],
                           "condition_id": row["condition_id"],
                           "prefix_actions": row["prefix_actions"],
                           "regression_exposed": row["regression_exposed"],
                           "original_first_action": row["original_action"],
                           "original_arm": row["original_arm"],
                           "action4_minus_action3": contrasts,
                           "action4_region": label,
                           "action4_posthoc_migration": a4_result["posthoc_migration"]})
        write_json(output_root / "paired_regions.json", paired)
        summary = {
            "run_id": RUN_ID, "status": "COMPLETE", "development_only": True,
            "selection_inventory_count": len(inventory), "selected_cell_count": len(identities),
            "full_distinct_state_count": sum(row["full_duplicate_of"] is None for row in identities),
            "regression_exposed_cell_count": sum(bool(row["regression_exposed"]) for row in identities),
            "completed_branch_count": len(stored), "prefix_environment_step_count": prefix_env_steps,
            "environment_step_count": budget["env_steps"],
            "region_counts_by_cell": dict(sorted(counts.items())),
            "region_counts_by_distinct_full_state": dict(sorted(distinct_counts.items())),
            "trace_identity": trace_identity, "training_steps": 0, "formal_or_holdout_reads": 0,
            "automatic_retry": False,
        }
        write_json(output_root / "summary.json", summary)
        payloads = [output_root / name for name in (
            "start_receipt.json", "selection_inventory.json", "identity_inventory.json",
            "paired_regions.json", "summary.json")]
        payloads += sorted((output_root / "prefixes").glob("*.json"))
        payloads += sorted((output_root / "branches").glob("*.json"))
        write_json(output_root / "artifact_integrity.json", {"files": [
            {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size,
             "sha256": sha(path)} for path in payloads]})
        receipt.update(status="COMPLETE", finished_at=utc_now(),
                       environment_step_count=budget["env_steps"],
                       completed_branch_count=len(stored),
                       summary_sha256=sha(output_root / "summary.json"),
                       integrity_sha256=sha(output_root / "artifact_integrity.json"))
        write_json(output_root / "completion_receipt.json", receipt)
        return summary
    except Exception as exc:
        receipt.update(status="FAIL", failed_at=utc_now(), error_type=type(exc).__name__,
                       error=str(exc), **partial_evidence_status(
                           output_root / "branches", current_branch=current_branch,
                           environment_step_count=budget["env_steps"]),
                       automatic_retry=False)
        write_json(output_root / "failure_receipt.json", receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", type=Path, required=True)
    parser.add_argument("--pilot-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pilot_protocol.resolve(), args.pilot_run_root.resolve(),
                 args.output_root.resolve(), args.authorization_manifest.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
