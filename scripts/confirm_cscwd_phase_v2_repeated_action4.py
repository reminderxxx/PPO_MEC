"""One-shot, development-only paired branch check of the second phase-v2 action 4."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces  # noqa: E402
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402


RUN_ID = "cscwd_phase_v2_repeated_action4_branches_20261011_v1"
PILOT_ID = "cscwd_phase_v2_causal_public_pilot_20261011_v1"
PILOT_COMMIT = "5a79ffd018d4c260bc7a8417229a440c7d591fe4"
PILOT_MANIFEST_SHA = "8750250575566d68bc4c3030994536da6211db9d8aa65a6ce1a3a2173b84ea28"
PILOT_BEHAVIOR_SHA = "b3975c212bf5e0e9d9f444e2f32367f530a27f84b3f0ec494384f45ea94b0eef"
PROTOCOL_SHA = "c2ff08fafee5523a669ec7a201dfe3558bd45b54589bb291383c2ba7a97246c7"
BASE_CONFIG_SHA = "20e82b0ea4ff5c7caf597d987bcef1f7523dc0c1c7ece21deba32aa7d47af923"
CONDITIONS = ("default", "adverse_link", "ample_cache")
SOURCE_WINDOW = "cscwd_newdev_us_101_v1183_t1118849293800_1118849412600"
MAX_STARTS = 12
MAX_BRANCHES = 24
MAX_ENV_STEPS = 192
MAX_CONTINUATION = 8
EPS = 1e-6


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_witnesses(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    cases: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (row["arm"] == "sa_candidate" and row["view"] == "selected"
                and row["time_profile"] == "raw_ngsim_event_time_v2"
                and row["source_segment_id"] == "us_101"):
            if row["window_id"] != SOURCE_WINDOW:
                raise RuntimeError("pilot source window drift")
            cases[(row["condition_id"], row["seed"])].append(row)
    if len(cases) != 15 or {key[0] for key in cases} != set(CONDITIONS):
        raise RuntimeError("pilot selected-v2-US101 grid drift")
    result: dict[str, list[dict[str, str]]] = {condition: [] for condition in CONDITIONS}
    for (condition, seed), episode in cases.items():
        episode.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in episode] != list(range(len(episode))):
            raise RuntimeError("noncontiguous pilot episode")
        if len(episode) < 2:
            raise RuntimeError("pilot episode has no second decision")
        first, second = episode[:2]
        repeated = (first["executed_action"] == "4" and first["migration_status"] == "delivered"
                    and first["migration_success"] == "True" and second["executed_action"] == "4"
                    and second["migration_status"] == "target_departed_before_delivery"
                    and second["migration_success"] == "False")
        if repeated:
            result[condition].append({"seed": seed, "first": first, "second": second})
        elif any(row["executed_action"] == "4" and row["migration_status"] == "target_departed_before_delivery"
                 for row in episode):
            raise RuntimeError("unregistered failed action4 shape in selected pilot")
    if sum(map(len, result.values())) != 11 or any(not result[condition] for condition in CONDITIONS):
        raise RuntimeError("11 pilot witnesses / 3 fixed condition strata changed")
    return result


def choose_continuation(mask: list[bool]) -> int:
    if len(mask) != 5:
        raise RuntimeError("five-action contract drift")
    if mask[3]:
        return 3
    if mask[2]:
        return 2
    raise RuntimeError("frozen public continuation has no legal action")


def branch_metrics(rows: list[dict[str, Any]], summary: dict[str, Any], *, censored: bool) -> dict[str, Any]:
    sums = {key: sum(int(row[key]) for row in rows) for key in (
        "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
        "backhaul_transfer_bytes", "partial_transfer_bytes",
    )}
    if sums["backhaul_transfer_bytes"] != sums["model_transfer_bytes"] + sums["state_transfer_bytes"]:
        raise RuntimeError("branch backhaul byte conservation failed")
    return {
        "workflow_completed": bool(summary["workflow_completed"]),
        "on_time": bool(summary.get("on_time_workflow_completed", 0)),
        "service_failures": sum(not row["service_completed"] for row in rows),
        "migration_successes": sum(row["migration_success"] for row in rows),
        "migration_attempts": sum(row["executed_action"] == 4 for row in rows),
        "total_transfer_bytes": sums["input_transfer_bytes"] + sums["model_transfer_bytes"] + sums["state_transfer_bytes"],
        "recompute_seconds": sum(row["recompute_seconds"] for row in rows),
        "elapsed_seconds": sum(row["step_cost_seconds"] for row in rows),
        "reward": sum(row["reward"] for row in rows),
        "censored_at_continuation_cap": censored,
        **sums,
    }


def locally_dominated(action4: dict[str, Any], action3: dict[str, Any]) -> bool:
    if action4["censored_at_continuation_cap"] or action3["censored_at_continuation_cap"]:
        return False
    return (int(action3["workflow_completed"]) >= int(action4["workflow_completed"])
            and int(action3["on_time"]) >= int(action4["on_time"])
            and action3["service_failures"] <= action4["service_failures"]
            and action3["total_transfer_bytes"] < action4["total_transfer_bytes"])


def verify_inputs(protocol_path: Path, pilot_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    if protocol_path.name != "cscwd_phase_v2_causal_public_pilot_v1.json" or sha(protocol_path) != PROTOCOL_SHA:
        raise RuntimeError("frozen B protocol identity mismatch")
    if pilot_root.name != PILOT_ID or sha(pilot_root / "run_manifest.json") != PILOT_MANIFEST_SHA:
        raise RuntimeError("frozen B pilot manifest mismatch")
    if sha(pilot_root / "behavior_rows.csv") != PILOT_BEHAVIOR_SHA:
        raise RuntimeError("frozen B pilot behavior mismatch")
    manifest = read_json(pilot_root / "run_manifest.json")
    if (manifest["status"] != "complete" or manifest["git_commit"] != PILOT_COMMIT
            or manifest["protocol_sha256"] != PROTOCOL_SHA or manifest["formal_or_holdout_reads"] != 0):
        raise RuntimeError("B pilot provenance drift")
    protocol = read_json(protocol_path)
    if protocol["run_id"] != PILOT_ID or protocol["data_assignment"]["workload_design_id"] != "dev_01":
        raise RuntimeError("B pilot design identity drift")
    inputs = protocol["inputs"]
    paths = {
        name: Path(value).resolve() if Path(value).is_absolute() else (ROOT / value).resolve()
        for name, value in inputs.items() if name != "sha256"
    }
    if sha(paths["base_config"]) != BASE_CONFIG_SHA:
        raise RuntimeError("B pilot base config identity drift")
    for name in ("workload_manifest", "phase_contract", "raw_csv_path", "source_manifest"):
        key = "raw_csv" if name == "raw_csv_path" else name
        if sha(paths[name]) != inputs["sha256"][key]:
            raise RuntimeError(f"B pilot frozen input mismatch: {name}")
    if sha(ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py") != inputs["sha256"]["phase_v2_environment"]:
        raise RuntimeError("phase-v2 producer source drift")
    return protocol, paths


def public_projection(info: dict[str, Any]) -> dict[str, Any]:
    return {"semantic_state": info["semantic_state"], "action_mask": list(info["action_mask"])}


def public_decision_facts(info: dict[str, Any]) -> dict[str, Any]:
    state = info["semantic_state"]
    context = state["calibrated_context"]
    return {
        "current_rsu_id": state["vehicles"][0]["associated_rsu_id"],
        "predicted_target_rsu_id": state["predictions"]["predicted_handoff_target_rsu_id_by_vehicle"]["veh_pilot"],
        "prepared_state_prefix": context["prepared_state_prefix"],
        "public_clock_seconds": context["time_contract"]["clock_seconds"],
        "action_mask": list(info["action_mask"]),
    }


def event_row(condition: str, branch: str, step: int, before: dict[str, Any], event: dict[str, Any],
              terminated: bool, truncated: bool) -> dict[str, Any]:
    phases = list(event.get("phase_events") or [])
    if not math.isclose(sum(float(phase["elapsed_seconds"]) for phase in phases),
                        float(event["step_cost_seconds"]), abs_tol=EPS, rel_tol=0.0):
        raise RuntimeError("phase/step clock conservation failed")
    if int(event["backhaul_transfer_bytes"]) != int(event["model_transfer_bytes"]) + int(event["state_transfer_bytes"]):
        raise RuntimeError("phase-v2 backhaul conservation failed")
    return {
        "condition_id": condition, "branch": branch, "step_from_reset": step,
        "public_state_sha256_before": canonical_sha(public_projection(before)),
        "public_action_mask_before": list(before["action_mask"]),
        "executed_action": int(event["action"]), "service_completed": bool(event["service_completed"]),
        "migration_success": bool(event.get("migration_success", False)),
        "migration_status": event.get("migration_status"), "failure_reason": event.get("failure_reason"),
        "input_transfer_bytes": int(event.get("input_transfer_bytes", 0)),
        "model_transfer_bytes": int(event.get("model_transfer_bytes", 0)),
        "state_transfer_bytes": int(event.get("state_transfer_bytes", 0)),
        "backhaul_transfer_bytes": int(event.get("backhaul_transfer_bytes", 0)),
        "partial_transfer_bytes": int(event.get("partial_transfer_bytes", 0)),
        "recompute_seconds": sum(float(phase["elapsed_seconds"]) for phase in phases if phase["phase"] == "prefix_recompute"),
        "step_cost_seconds": float(event["step_cost_seconds"]), "reward": float(event["reward"]),
        "terminated": terminated, "truncated": truncated, "phase_events": phases,
    }


def step_checked(env: RawNGSIMPhaseV2Env, action: int, info: dict[str, Any],
                 *, condition: str, branch: str, budget: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    if budget["env_steps"] >= MAX_ENV_STEPS:
        raise RuntimeError("192 environment-step hard cap reached")
    if not info["action_mask"][action]:
        raise RuntimeError("masked branch action")
    before_clock, before_index = env.clock_seconds, env.step_index
    _, _, terminated, truncated, after = env.step(action)
    budget["env_steps"] += 1
    event = dict(after["transition"])
    if event["action"] != action or env.step_index != before_index + 1:
        raise RuntimeError("environment action/step identity changed")
    if not math.isclose(env.clock_seconds - before_clock, float(event["step_cost_seconds"]),
                        abs_tol=EPS, rel_tol=0.0):
        raise RuntimeError("environment clock changed without ledger conservation")
    return event_row(condition, branch, before_index, info, event, terminated, truncated), after, terminated, truncated


def replay_matches_pilot(actual: dict[str, Any], row: dict[str, str]) -> None:
    for key in ("executed_action", "input_transfer_bytes", "model_transfer_bytes",
                "state_transfer_bytes", "backhaul_transfer_bytes", "partial_transfer_bytes"):
        if int(actual[key]) != int(row[key]):
            raise RuntimeError(f"original pilot replay mismatch: {key}")
    for key in ("migration_status",):
        if actual[key] != row[key]:
            raise RuntimeError(f"original pilot replay mismatch: {key}")
    for key in ("step_cost_seconds", "reward"):
        if not math.isclose(float(actual[key]), float(row[key]), abs_tol=EPS, rel_tol=0.0):
            raise RuntimeError(f"original pilot replay mismatch: {key}")


def run(protocol_path: Path, pilot_root: Path, output_root: Path, plan_path: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    if plan_path.name != "cscwd_phase_v2_repeated_action4_branch_plan_20261011.md":
        raise RuntimeError("wrong pre-run plan path")
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout:
        raise RuntimeError("diagnostic worktree must be clean at launch")
    protocol, paths = verify_inputs(protocol_path, pilot_root)
    witnesses = select_witnesses(read_csv(pilot_root / "behavior_rows.csv"))
    output_root.mkdir(parents=True)
    receipt = {
        "run_id": RUN_ID, "status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "plan_sha256": sha(plan_path), "pilot_behavior_sha256": PILOT_BEHAVIOR_SHA,
        "pilot_protocol_sha256": PROTOCOL_SHA, "max_starts": MAX_STARTS,
        "max_branches": MAX_BRANCHES, "max_env_steps": MAX_ENV_STEPS,
    }
    write_json(output_root / "start_receipt.json", receipt)
    budget = {"env_steps": 0}
    try:
        config, _ = _load_experiment_config(paths["base_config"])
        config["reward_profile"] = "original_reward_v1"
        config["interface_profile"] = "calibrated_workflow_interface_v4_prepared_state_prefix"
        workload = read_json(paths["workload_manifest"])
        instance = next(deepcopy(item) for item in workload["instances"] if item["design_id"] == "dev_01")
        phase = read_json(paths["phase_contract"])
        source = read_json(paths["source_manifest"])
        selected = [item for item in source["selected_windows"] if item["source_segment_id"] == "us_101"]
        if len(selected) != 1 or selected[0]["window_id"] != SOURCE_WINDOW:
            raise RuntimeError("US101 source identity drift")
        traces, trace_identity = load_traces(paths["raw_csv_path"], selected)
        trace = traces[SOURCE_WINDOW]
        starts = []
        events = []
        pairs = []
        public_hashes = set()
        for condition_id in CONDITIONS:
            condition = next(item for item in phase["conditions"] if item["condition_id"] == condition_id)
            env = RawNGSIMPhaseV2Env(config, deepcopy(instance), trace,
                                     phase_contract=phase["physical_assumptions"], condition=condition)
            _, initial = env.reset()
            prefix, before, done, truncated = step_checked(
                env, 4, initial, condition=condition_id, branch="common_prefix", budget=budget,
            )
            replay_matches_pilot(prefix, witnesses[condition_id][0]["first"])
            if done or truncated or not prefix["migration_success"] or prefix["migration_status"] != "delivered":
                raise RuntimeError("registered successful first migration did not replay")
            if not (before["action_mask"][3] and before["action_mask"][4]):
                raise RuntimeError("paired native actions not both publicly legal")
            public_sha = canonical_sha(public_projection(before))
            if public_sha in public_hashes:
                raise RuntimeError("condition-stratified public states not unique")
            public_hashes.add(public_sha)
            facts = public_decision_facts(before)
            starts.append({"condition_id": condition_id, "public_state_sha256": public_sha,
                           "public_decision_facts": facts,
                           "repeated_pilot_seeds": sorted(int(item["seed"]) for item in witnesses[condition_id]),
                           "pilot_repeated_count": len(witnesses[condition_id]),
                           "prefix_step": prefix})
            events.append(prefix)
            branch_results = {}
            for branch_action in (4, 3):
                branch = f"action{branch_action}"
                branch_env = deepcopy(env)
                branch_info = deepcopy(before)
                ledger = []
                event, branch_info, done, truncated = step_checked(
                    branch_env, branch_action, branch_info, condition=condition_id,
                    branch=branch, budget=budget,
                )
                if branch_action == 4:
                    replay_matches_pilot(event, witnesses[condition_id][0]["second"])
                ledger.append(event)
                continuation_count = 0
                while not (done or truncated) and continuation_count < MAX_CONTINUATION:
                    action = choose_continuation(list(branch_info["action_mask"]))
                    event, branch_info, done, truncated = step_checked(
                        branch_env, action, branch_info, condition=condition_id,
                        branch=branch, budget=budget,
                    )
                    ledger.append(event)
                    continuation_count += 1
                censored = not (done or truncated)
                metrics = branch_metrics(ledger, branch_env.summary(), censored=censored)
                branch_results[branch] = {"first_action": branch_action,
                                          "continuation_steps": continuation_count,
                                          "branch_env_steps": len(ledger), "metrics": metrics,
                                          "first_event": ledger[0]}
                events.extend(ledger)
            a4, a3 = branch_results["action4"]["metrics"], branch_results["action3"]["metrics"]
            pairs.append({
                "condition_id": condition_id, "source_segment_id": "us_101",
                "public_state_sha256": public_sha,
                "pilot_repeated_count": len(witnesses[condition_id]),
                "action4": branch_results["action4"], "action3": branch_results["action3"],
                "action3_minus_action4": {
                    key: a3[key] - a4[key] for key in (
                        "service_failures", "migration_successes", "total_transfer_bytes",
                        "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
                        "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds",
                        "elapsed_seconds", "reward",
                    )
                },
                "locally_dominated_in_frozen_continuation": locally_dominated(a4, a3),
            })
        if len(starts) > MAX_STARTS or len(pairs) * 2 > MAX_BRANCHES or budget["env_steps"] > MAX_ENV_STEPS:
            raise RuntimeError("bounded branch contract exceeded")
        if sum(item["pilot_repeated_count"] for item in starts) != 11:
            raise RuntimeError("pilot witness count drift after branching")
        write_json(output_root / "starts.json", starts)
        write_json(output_root / "paired_branches.json", pairs)
        with (output_root / "event_ledger.jsonl").open("w") as handle:
            for row in events:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "run_id": RUN_ID, "status": "COMPLETE", "pilot_run_id": PILOT_ID,
            "source_window_id": SOURCE_WINDOW, "trace_identity": trace_identity,
            "public_start_count": len(starts), "branch_count": len(pairs) * 2,
            "environment_step_count": budget["env_steps"], "event_ledger_rows": len(events),
            "original_pilot_repeated_attempts": 11,
            "condition_ids": list(CONDITIONS),
            "locally_dominated_conditions": [item["condition_id"] for item in pairs
                                             if item["locally_dominated_in_frozen_continuation"]],
            "formal_or_holdout_reads": 0, "training_steps": 0,
            "claim_scope": "development-only matched local branch, not independent paper evidence",
        }
        write_json(output_root / "summary.json", summary)
        payloads = ["start_receipt.json", "starts.json", "paired_branches.json", "event_ledger.jsonl", "summary.json"]
        write_json(output_root / "artifact_integrity.json", {
            "files": [{"path": name, "sha256": sha(output_root / name), "bytes": (output_root / name).stat().st_size}
                      for name in payloads]
        })
        receipt.update(status="COMPLETE", finished_at=datetime.now(timezone.utc).isoformat(),
                       environment_step_count=budget["env_steps"], branch_count=len(pairs) * 2,
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
    result = run(args.pilot_protocol.resolve(), args.pilot_run_root.resolve(), args.output_root.resolve(),
                 ROOT / "docs/project/cscwd_phase_v2_repeated_action4_branch_plan_20261011.md")
    print(json.dumps({"status": result["status"], "run_id": RUN_ID,
                      "environment_step_count": result["environment_step_count"],
                      "summary_sha256": sha(args.output_root.resolve() / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
