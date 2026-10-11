"""Bounded public-rule reachability on a frozen long raw-development source."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv, RawVehicleTrace

RAW_SHA = "ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c"
SOURCE_SHA = "34eee3a8d16f2dfe89f3a8d8dcd27f20159c0488e2f1a9ebc77be3aac747abde"
PARENT_SHA = "b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc"
RAW_ENV_SHA = "30666d7c874717d8464a64702c811116e96b931a5db6b57287c6c11d2dcc6575"
BASE_ENV_SHA = "c6ded6ce84230519160a40f29e276b5823eb204a35e31c35f914d1c6bfee3a59"
PUBLIC_ESTIMATOR_SHA = "ee6873d9b16e5b1e4ac0dcfab1eacce915d50aeeff8bf8d9a862093202984828"
DESIGN_IDS = ("regression_05", "dev_00", "dev_01", "regression_04")
METHODS = ("fallback", "serve", "causal_public_immediate_rule",
           "causal_public_two_step_rule", "prepare_if_legal_else_serve")
MAX_EPISODES = 60
MAX_STEPS = 1440
MAX_PREVIEWS = 5000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def segment(raw: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in str(raw)).strip("_")


def number(raw: str) -> float:
    return float(str(raw).replace(",", ""))


def load_public_rules(path: Path) -> dict:
    if sha(path) != PUBLIC_ESTIMATOR_SHA:
        raise RuntimeError("public rule file differs from pre-run frozen SHA")
    spec = importlib.util.spec_from_file_location("frozen_cscwd_public_estimator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen public estimator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "causal_public_immediate_rule": module.CausalPublicImmediateRule(),
        "causal_public_two_step_rule": module.CausalPublicTwoStepRule(),
    }


def load_traces(path: Path, selected: list[dict]) -> tuple[dict[str, RawVehicleTrace], dict]:
    if len(selected) > 3 or not selected:
        raise RuntimeError("source manifest does not contain 1–3 frozen selected windows")
    expected = {}
    for window in selected:
        if window["decision"] != "selected" or window["contiguous_frame_count"] != 1189:
            raise RuntimeError("source manifest contains nonqualified window")
        lo, hi = int(window["source_time_start"]), int(window["source_time_end"])
        if hi - lo != 118800:
            raise RuntimeError("selected source time span changed")
        expected[window["window_id"]] = window
    matched: dict[str, list[tuple[int, int, float, float]]] = {key: [] for key in expected}
    scanned = 0
    for chunk in pd.read_csv(
        path,
        usecols=["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "Location"],
        dtype=str,
        keep_default_na=False,
        chunksize=500_000,
    ):
        scanned += len(chunk)
        locations = chunk["Location"].map(segment)
        times = pd.to_numeric(chunk["Global_Time"].str.replace(",", "", regex=False), errors="coerce")
        vehicles = pd.to_numeric(chunk["Vehicle_ID"].str.replace(",", "", regex=False), errors="coerce")
        for window_id, window in expected.items():
            mask = (locations == window["source_segment_id"]) & (vehicles == int(window["vehicle_id"]))
            mask &= times.between(int(window["source_time_start"]), int(window["source_time_end"]))
            for row in chunk.loc[mask].itertuples(index=False):
                # read_csv usecols follows source header order for this raw file.
                values = dict(zip(chunk.columns, row))
                matched[window_id].append((int(number(values["Global_Time"])),
                                           int(number(values["Frame_ID"])),
                                           number(values["Local_X"]) * 0.3048,
                                           number(values["Local_Y"]) * 0.3048))
    traces = {}
    identities = {}
    for window_id, window in expected.items():
        rows = sorted(matched[window_id])
        lo = int(window["source_time_start"])
        if len(rows) != 1189 or [row[0] for row in rows] != [lo + 100 * i for i in range(1189)]:
            raise RuntimeError(f"selected raw timestamp/vehicle uniqueness mismatch: {window_id}")
        if [row[1] for row in rows] != list(range(int(window["source_frame_start"]),
                                                int(window["source_frame_end"]) + 1)):
            raise RuntimeError(f"selected raw Frame_ID mismatch: {window_id}")
        if any(not math.isfinite(x) or not math.isfinite(y) for _, _, x, y in rows):
            raise RuntimeError(f"selected raw coordinate nonfinite: {window_id}")
        traces[window_id] = RawVehicleTrace(
            tuple((row[0] - lo) / 1000.0 for row in rows),
            tuple((row[2], row[3]) for row in rows),
            str(window["vehicle_id"]),
        )
        identities[window_id] = {
            "source_segment_id": window["source_segment_id"],
            "vehicle_id": window["vehicle_id"],
            "Global_Time_start": lo,
            "Global_Time_end": int(window["source_time_end"]),
            "Frame_ID_start": int(window["source_frame_start"]),
            "Frame_ID_end": int(window["source_frame_end"]),
            "raw_source_row_count": len(rows),
            "raw_duration_seconds": traces[window_id].times_seconds[-1],
            "historical_train_dev_overlap_count": window["historical_train_dev_overlap_count"],
        }
    return traces, {"raw_rows_scanned": scanned, "window_identities": identities}


def requested_action(method: str, info: dict, public_rules: dict) -> tuple[int, int]:
    mask = list(info["action_mask"])
    if method == "fallback":
        action, estimates = 2, 0
    elif method == "serve":
        action, estimates = 3, 0
    elif method == "prepare_if_legal_else_serve":
        action, estimates = (4 if mask[4] else 3), 0
    elif method in public_rules:
        action = int(public_rules[method].select_action_from_info(info))
        estimates = 1 if method == "causal_public_immediate_rule" else 1 + sum(mask)
    else:
        raise RuntimeError(f"unexpected frozen method: {method}")
    return action, estimates


def legal_action(requested: int, mask: list[bool]) -> int:
    if 0 <= requested < 5 and mask[requested]:
        return requested
    return next(action for action in (2, 3, 0, 1, 4) if mask[action])


def episode(config: dict, instance: dict, trace: RawVehicleTrace, method: str,
            public_rules: dict) -> tuple[dict, list[dict], int, int]:
    env = RawNGSIMEventTimeEnv(config, deepcopy(instance), trace)
    _, info = env.reset()
    ledger = []
    previews = 0
    terminated = truncated = False
    for step in range(min(24, int(instance["max_steps"]))):
        mask = list(info["action_mask"])
        requested, estimates = requested_action(method, info, public_rules)
        executed = legal_action(requested, mask)
        previews += 1 + estimates  # one native preview inside each raw env.step
        state = info["semantic_state"]
        public_contact = state["calibrated_context"].get("contact_budget_seconds")
        actual_contact = env._physical_contact_budget_seconds() if executed != 2 else None
        before = env.clock_seconds
        node_id = env._current_node()["node_id"]
        _, _, terminated, truncated, info = env.step(executed)
        event = info["transition"]
        if int(event["action"]) != executed:
            raise RuntimeError("raw executor rewrote an action already checked against public mask")
        ledger.append({
            "step": step, "node_id_before": node_id,
            "requested_action": requested, "executed_action": executed,
            "public_mask": mask, "public_contact_budget_seconds": public_contact,
            "actual_contact_budget_seconds_privileged_diagnostic": actual_contact,
            "clock_seconds_before": before, "clock_seconds_after": env.clock_seconds,
            "trace_remaining_seconds_privileged_diagnostic": trace.times_seconds[-1] - before,
            "service_completed": bool(event["service_completed"]),
            "migration_success": bool(event.get("migration_success", False)),
            "state_transfer_status": dict(event.get("state_transfer") or {}).get("status"),
            "admission_rejection_reason": event.get("admission_rejection_reason"),
            "planned_step_cost_seconds": event.get("planned_step_cost_seconds"),
            "step_cost_seconds": float(event["step_cost_seconds"]),
            "model_transfer_bytes": int(event.get("model_transfer_bytes", 0)),
            "state_transfer_bytes": int(event.get("state_transfer_bytes", 0)),
            "input_transfer_bytes": int(event.get("input_transfer_bytes", 0)),
            "transfer_seconds": float(event.get("transfer_seconds", 0.0)),
            "model_load_seconds": float(event.get("model_load_seconds", 0.0)),
            "state_restore_seconds": float(event.get("state_restore_seconds", 0.0)),
            "recompute_seconds": float(event.get("recompute_seconds", 0.0)),
            "service_operation_seconds": float(event.get("service_operation_seconds", 0.0)),
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    summary = env.summary()
    result = {"method": method, "design_id": instance["design_id"],
              "source_duration_seconds": trace.times_seconds[-1],
              "steps": len(ledger), "terminated": bool(terminated), "truncated": bool(truncated),
              "workflow_completed": int(summary["workflow_completed"]),
              "completed_nodes": int(summary["completed_nodes"]),
              "service_failures": int(summary["service_failures"]),
              "migration_successes": int(summary["migration_successes"]),
              "model_transfer_bytes": int(summary["model_transfer_bytes"]),
              "state_transfer_bytes": int(summary["state_transfer_bytes"]),
              "input_transfer_bytes": int(summary["input_transfer_bytes"]),
              "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
              "deadline_seconds": float(instance["deadline_seconds"]),
              "on_time_complete": bool(summary["workflow_completed"] and
                                       float(summary["modeled_completion_seconds"]) <= float(instance["deadline_seconds"])),
              "admission_rejections": dict(Counter(str(x["admission_rejection_reason"])
                                                   for x in ledger if x["admission_rejection_reason"])),
              "preview_count_upper_bound": previews,
              "requested_executed_mismatches": sum(x["requested_action"] != x["executed_action"] for x in ledger)}
    return result, ledger, len(ledger), previews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--public-estimator-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.raw_csv_path) != RAW_SHA or sha(args.source_manifest) != SOURCE_SHA:
        raise RuntimeError("frozen raw/source identity mismatch")
    if sha(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json") != PARENT_SHA:
        raise RuntimeError("frozen workload identity mismatch")
    if sha(ROOT / "src/envs/core/raw_ngsim_event_time_env.py") != RAW_ENV_SHA:
        raise RuntimeError("raw environment changed")
    if sha(ROOT / "src/envs/core/calibrated_continuous_workflow_env.py") != BASE_ENV_SHA:
        raise RuntimeError("calibrated executor changed")
    public_rules = load_public_rules(args.public_estimator_path)
    source = json.loads(args.source_manifest.read_text())
    if source["decision"] != "READY_FOR_BOUNDED_REACHABILITY":
        raise RuntimeError("source qualification has not passed")
    selected = source["selected_windows"]
    traces, trace_identity = load_traces(args.raw_csv_path, selected)
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    parent = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instances = {row["design_id"]: row for row in parent["instances"]}
    if any(key not in instances for key in DESIGN_IDS):
        raise RuntimeError("frozen resource control missing")
    results, steps, previews = [], 0, 0
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    ledger_path = output / "step_ledger.jsonl"
    with ledger_path.open("w") as stream:
        for window in selected:
            for design_id in DESIGN_IDS:
                for method in METHODS:
                    result, ledger, count, preview_count = episode(
                        config, instances[design_id], traces[window["window_id"]], method, public_rules
                    )
                    result["window_id"] = window["window_id"]
                    results.append(result)
                    steps += count
                    previews += preview_count
                    if len(results) > MAX_EPISODES or steps > MAX_STEPS or previews > MAX_PREVIEWS:
                        raise RuntimeError("frozen reachability budget exceeded")
                    for event in ledger:
                        stream.write(json.dumps({"window_id": window["window_id"],
                                                 "design_id": design_id, "method": method,
                                                 **event}, sort_keys=True) + "\n")
    manifest = {"schema_version": "cscwd_new_development_reachability_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "script_sha256": sha(Path(__file__)), "source_manifest_sha256": SOURCE_SHA,
                "raw_source_sha256": RAW_SHA, "raw_environment_sha256": RAW_ENV_SHA,
                "calibrated_executor_sha256": BASE_ENV_SHA,
                "public_estimator_sha256": PUBLIC_ESTIMATOR_SHA,
                "public_estimator_path": str(args.public_estimator_path.resolve()),
                "profile": "raw_ngsim_event_time_v1", "claim_scope": "bounded_development_feasibility_not_training_or_ranking",
                "source_trace_identity": trace_identity,
                "methods": list(METHODS), "design_ids": list(DESIGN_IDS),
                "episode_count": len(results), "real_step_count": steps,
                "preview_count_upper_bound": previews,
                "step_ledger": {"path": ledger_path.name, "sha256": sha(ledger_path)},
                "results": results}
    result_path = output / "reachability_manifest.json"
    result_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(results), "steps": steps,
                      "previews_upper_bound": previews,
                      "workflow_completed": sum(x["workflow_completed"] for x in results)}, sort_keys=True))


if __name__ == "__main__":
    main()
