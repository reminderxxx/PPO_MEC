"""Bounded native phase diagnostic of the 12 frozen action-4 attempts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import (
    BASE_ENV_SHA, PARENT_SHA, PUBLIC_ESTIMATOR_SHA, RAW_ENV_SHA, RAW_SHA,
    SOURCE_SHA, load_public_rules, load_traces, sha,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, _network_seconds
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv


MANIFEST_SHA = "481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c"
LEDGER_SHA = "679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb"
MAX_STARTS = 12
MAX_EXTRA_STEPS = 288
EPS = 1e-7


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=EPS)


def diagnose(raw_csv: Path, source_manifest: Path, reachability_manifest: Path,
             estimator_path: Path) -> dict:
    required = (
        (raw_csv, RAW_SHA), (source_manifest, SOURCE_SHA),
        (reachability_manifest, MANIFEST_SHA), (estimator_path, PUBLIC_ESTIMATOR_SHA),
        (ROOT / "src/envs/core/raw_ngsim_event_time_env.py", RAW_ENV_SHA),
        (ROOT / "src/envs/core/calibrated_continuous_workflow_env.py", BASE_ENV_SHA),
        (ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json", PARENT_SHA),
    )
    for path, digest in required:
        if sha(path) != digest:
            raise RuntimeError(f"frozen input differs: {path}")
    reachability = json.loads(reachability_manifest.read_text())
    ledger_path = reachability_manifest.parent / reachability["step_ledger"]["path"]
    if reachability["step_ledger"]["sha256"] != LEDGER_SHA or sha(ledger_path) != LEDGER_SHA:
        raise RuntimeError("reachability step ledger differs")
    ledger = [json.loads(line) for line in ledger_path.open()]
    targets = [row for row in ledger if row["executed_action"] == 4]
    if (len(ledger), len(targets)) != (320, 12) or len(targets) > MAX_STARTS:
        raise RuntimeError("frozen step/action4 count changed")
    source = json.loads(source_manifest.read_text())
    traces, trace_identity = load_traces(raw_csv, source["selected_windows"])
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    workload = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instances = {str(row["design_id"]): row for row in workload["instances"]}
    estimator = load_public_rules(estimator_path)
    # Both rules share this pure public estimator module; use its module function.
    public_estimate = estimator["causal_public_immediate_rule"].select_action_from_info.__globals__["estimate_public_actions"]

    episode_rows = defaultdict(list)
    for row in ledger:
        episode_rows[(row["window_id"], row["design_id"], row["method"])].append(row)
    output_rows = []
    extra_steps = 0
    for target in targets:
        key = target["window_id"], target["design_id"], target["method"]
        env = RawNGSIMEventTimeEnv(config, deepcopy(instances[target["design_id"]]), traces[target["window_id"]])
        _, info = env.reset()
        episode = episode_rows[key]
        for row in episode[:int(target["step"])]:
            if not _close(env.clock_seconds, float(row["clock_seconds_before"])):
                raise RuntimeError("prefix replay clock-before mismatch")
            _, _, _, _, info = env.step(int(row["executed_action"]))
            extra_steps += 1
            event = info["transition"]
            if (int(event["action"]) != int(row["executed_action"])
                    or not _close(env.clock_seconds, float(row["clock_seconds_after"]))
                    or event.get("admission_rejection_reason") != row["admission_rejection_reason"]):
                raise RuntimeError("prefix replay transition mismatch")
            if extra_steps > MAX_EXTRA_STEPS:
                raise RuntimeError("phase diagnostic step budget exceeded")
        if not _close(env.clock_seconds, float(target["clock_seconds_before"])):
            raise RuntimeError("target replay clock mismatch")
        info = env._info()
        if info["action_mask"] != target["public_mask"] or not info["action_mask"][4]:
            raise RuntimeError("target public mask mismatch")
        public_budget = float(info["semantic_state"]["calibrated_context"]["contact_budget_seconds"])
        actual_budget = env._physical_contact_budget_seconds()
        if not (_close(public_budget, float(target["public_contact_budget_seconds"]))
                and _close(actual_budget, float(target["actual_contact_budget_seconds_privileged_diagnostic"]))):
            raise RuntimeError("contact budget replay mismatch")
        before_guard = _hash_json({"clock": env.clock_seconds, "cache": {k: v.residents for k, v in env.caches.items()},
                                   "prepared": env.prepared_state, "metrics": env.metrics})
        estimate = public_estimate(info["semantic_state"], info["action_mask"])
        if before_guard != _hash_json({"clock": env.clock_seconds, "cache": {k: v.residents for k, v in env.caches.items()},
                                       "prepared": env.prepared_state, "metrics": env.metrics}):
            raise RuntimeError("public estimator changed source environment")
        action4 = estimate["actions"]["4"]

        current = env._association(env._observed_xy())
        predicted_target = env._predicted_handoff_target()
        position = env._actual_xy(env.clock_seconds)
        vx, vy = env._public_velocity()
        distances = {rsu: math.hypot(position[0] - x, position[1] - y)
                     for rsu, x, y, _ in env.geometry}
        geometry = {"source": "raw_profile_first_point_plus_12m_y_offsets_radius_10m",
                    "radius_metres": 10.0, "current_rsu_id": current,
                    "predicted_target_rsu_id": predicted_target,
                    "distance_to_current_metres": distances.get(current),
                    "distance_to_predicted_target_metres": distances.get(predicted_target),
                    "speed_metres_per_second": math.hypot(vx, vy),
                    "associated_rsu_distances_metres": distances}

        preview = deepcopy(env)
        _, _, _, _, preview_info = CalibratedContinuousWorkflowEnv.step(preview, 4)
        extra_steps += 1
        if extra_steps > MAX_EXTRA_STEPS:
            raise RuntimeError("phase diagnostic step budget exceeded")
        event = preview_info["transition"]
        planned = float(event["step_cost_seconds"])
        if int(event["action"]) != 4 or not _close(planned, float(target["planned_step_cost_seconds"])):
            raise RuntimeError("native preview differs from frozen planned cost/action")
        cache_events = list(event.get("cache_events") or [])
        cache_event = cache_events[0] if cache_events else None
        node = env._current_node()
        state_bytes = int(node["state_bytes"])
        link_mbps = float(env._effective_link_mbps())
        fixed = float(config["link"]["fixed_seconds"])
        restore = float(config["measured_time_seconds"]["state_restore_overhead"])
        model_network = _network_seconds(int(cache_event["transfer_bytes"]), link_mbps, fixed) if cache_event else None
        model_load = float(cache_event["load_seconds"]) if cache_event else None
        state_network = _network_seconds(state_bytes, link_mbps, fixed)
        native_prepare = model_network + model_load + state_network + restore if cache_event else None
        phase = {
            "model_network_seconds": model_network,
            "model_load_seconds": model_load,
            "state_network_seconds": state_network,
            "state_restore_overhead_seconds": restore,
            "native_model_plus_state_prepare_seconds": native_prepare,
            "native_prepare_fits_actual_contact": native_prepare is not None and native_prepare <= actual_budget + EPS,
            "native_prepare_fits_public_contact": native_prepare is not None and native_prepare <= public_budget + EPS,
            "native_cache_admission_committed_after_preview": None if cache_event is None else bool(cache_event["committed"]),
            "native_cache_reason_after_preview": None if cache_event is None else cache_event.get("reason"),
            "native_state_transfer_status": dict(event.get("state_transfer") or {}).get("status"),
            "native_service_completed": bool(event["service_completed"]),
            "native_migration_success": bool(event.get("migration_success", False)),
            "native_step_cost_seconds": planned,
            "native_model_transfer_bytes": int(event["model_transfer_bytes"]),
            "native_state_transfer_bytes": int(event["state_transfer_bytes"]),
            "native_input_transfer_bytes": int(event["input_transfer_bytes"]),
            "native_transfer_seconds": float(event["transfer_seconds"]),
            "native_compute_seconds": float(node["compute_seconds"]),
            "native_recompute_seconds": float(event["recompute_seconds"]),
            "native_service_operation_seconds": float(event["service_operation_seconds"]),
            "native_model_load_seconds": float(event["model_load_seconds"]),
            "native_state_restore_seconds": float(event["state_restore_seconds"]),
        }
        if before_guard != _hash_json({"clock": env.clock_seconds, "cache": {k: v.residents for k, v in env.caches.items()},
                                       "prepared": env.prepared_state, "metrics": env.metrics}):
            raise RuntimeError("native clone preview changed source environment")
        output_rows.append({
            "window_id": target["window_id"], "design_id": target["design_id"],
            "method": target["method"], "step": target["step"],
            "clock_seconds_before": target["clock_seconds_before"],
            "public_state_sha256": _hash_json(info),
            "geometry": geometry,
            "public_contact_budget_seconds": public_budget,
            "actual_contact_budget_seconds_privileged_diagnostic": actual_budget,
            "trace_remaining_seconds_privileged_diagnostic": float(target["trace_remaining_seconds_privileged_diagnostic"]),
            "public_action4": {
                key: action4[key] for key in (
                    "legal", "current_service", "target_prepare", "target_state_commit",
                    "cost_status", "deadline_fit", "target_prepare_contact_fit",
                    "raw_full_step_contact_fit", "raw_execution_fit", "estimated_total_seconds",
                    "estimated_phase_seconds", "unknown_reasons",
                )
            },
            "phase": phase,
            "raw_rejection_reason": target["admission_rejection_reason"],
            "raw_recorded_model_bytes_after_rollback": int(target["model_transfer_bytes"]),
            "raw_recorded_state_bytes_after_rollback": int(target["state_transfer_bytes"]),
            "raw_recorded_step_cost_seconds": float(target["step_cost_seconds"]),
        })
    categories = (
        "current_service", "target_prepare", "target_state_commit", "cost_status",
        "deadline_fit", "raw_full_step_contact_fit", "raw_execution_fit",
    )
    return {
        "schema_version": "cscwd_new_development_action4_phase_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hashes": {"raw_csv": RAW_SHA, "source_manifest": SOURCE_SHA,
                         "reachability_manifest": MANIFEST_SHA, "step_ledger": LEDGER_SHA,
                         "raw_environment": RAW_ENV_SHA, "calibrated_executor": BASE_ENV_SHA,
                         "public_estimator": PUBLIC_ESTIMATOR_SHA},
        "scope": "12_existing_action4_rows_no_new_policy_episode_or_selection",
        "raw_trace_identity": trace_identity,
        "extra_environment_steps": extra_steps,
        "max_extra_environment_steps": MAX_EXTRA_STEPS,
        "action4_row_count": len(output_rows),
        "public_action4_category_counts": {
            field: dict(Counter(str(row["public_action4"][field]) for row in output_rows))
            for field in categories
        },
        "native_prepare_fits_actual_contact_count": sum(row["phase"]["native_prepare_fits_actual_contact"] for row in output_rows),
        "native_prepare_fits_public_contact_count": sum(row["phase"]["native_prepare_fits_public_contact"] for row in output_rows),
        "native_preview_migration_success_count": sum(row["phase"]["native_migration_success"] for row in output_rows),
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--reachability-manifest", required=True, type=Path)
    parser.add_argument("--public-estimator-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.raw_csv_path, args.source_manifest, args.reachability_manifest,
                      args.public_estimator_path)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    path = output / "action4_phase_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"path": str(path), "sha256": sha(path),
                      "extra_environment_steps": result["extra_environment_steps"]}, sort_keys=True))


if __name__ == "__main__":
    main()
