"""Read-only timing/phase audit of the frozen CSCWD raw NGSIM v4 check."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.data.mobility.ngsim_event_trace import FROZEN_DESIGN_IDS, load_frozen_traces
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, _network_seconds
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv, RawVehicleTrace

SOURCE_SHA = "c0b64fd64c602ef5351c5686d2f315bbd5784e63ebbd08902b297fc5bacc883f"
SUMMARY_SHA = "62c1f4c2c7fbb2e2763f6946c0fad29b4cd6a3705ebf01501a7d905d8b1fe30f"
REPLAY_STEP_CAP = 147
PREVIEW_CAP = 2000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def network_seconds(byte_count: int, mbps: float, fixed: float) -> float:
    return _network_seconds(byte_count, mbps, fixed)


def native_phase(env: RawNGSIMEventTimeEnv, action: int, *, estimated: bool) -> dict:
    clone = env.clone_for_decision_model() if estimated else deepcopy(env)
    _, _, _, _, info = CalibratedContinuousWorkflowEnv.step(clone, action)
    event = info["transition"]
    rate = clone._effective_link_mbps()
    fixed = float(clone.config["link"]["fixed_seconds"])
    model = network_seconds(int(event["model_transfer_bytes"]), rate, fixed)
    state = network_seconds(int(event["state_transfer_bytes"]), rate, fixed)
    input_time = network_seconds(int(event["input_transfer_bytes"]), rate, fixed)
    compute = float(env._current_node()["compute_seconds"]) if event["service_completed"] else 0.0
    fallback = float(env.config["vehicle"]["fallback_seconds"]) if action == 2 else 0.0
    failed = float(env.config["objective"]["failed_service_seconds"]) if not event["service_completed"] else 0.0
    load = float(event["model_load_seconds"])
    restore = float(event["state_restore_seconds"])
    recompute = float(event["recompute_seconds"])
    total = sum((model, state, input_time, compute, fallback, failed, load, restore, recompute))
    if not math.isclose(total, float(event["step_cost_seconds"]), abs_tol=1e-7):
        raise AssertionError(f"native time ledger mismatch: {total} != {event['step_cost_seconds']}")
    return {"action": action, "native_service_completed": bool(event["service_completed"]),
            "native_migration_success": bool(event["migration_success"]),
            "model_network_seconds": model, "state_network_seconds": state,
            "input_network_seconds": input_time, "model_load_seconds": load,
            "state_restore_seconds": restore, "recompute_seconds": recompute,
            "compute_seconds": compute, "vehicle_fallback_seconds": fallback,
            "failed_service_seconds": failed, "total_seconds": total,
            "model_transfer_bytes": int(event["model_transfer_bytes"]),
            "state_transfer_bytes": int(event["state_transfer_bytes"]),
            "input_transfer_bytes": int(event["input_transfer_bytes"]),
            "link_mbps": rate,
            "native_cache_events": [{"reason": x["reason"], "committed": x["committed"],
                                     "transfer_bytes": x["transfer_bytes"], "load_seconds": x["load_seconds"]}
                                    for x in event["cache_events"]],
            "native_state_status": (event.get("state_transfer") or {}).get("status")}


def prospective_prepare(env: RawNGSIMEventTimeEnv, action: int) -> dict | None:
    if action not in (1, 4):
        return None
    target = env._predicted_handoff_target()
    if target is None:
        return {"target": None, "bundle_admissible": False, "reason": "no_causal_target"}
    clone = deepcopy(env)
    event = clone._admit_bundle(target, str(clone._current_node()["required_adapter"]))
    model_seconds = clone._model_transfer_seconds(event)
    package_bytes = clone._state_package_bytes(clone._current_node()) if action == 4 else 0
    state_network = network_seconds(package_bytes, clone._effective_link_mbps(), float(clone.config["link"]["fixed_seconds"]))
    state_restore = float(clone.config["measured_time_seconds"]["state_restore_overhead"]) if action == 4 else 0.0
    return {"target": target, "bundle_admissible": bool(event["committed"]),
            "reason": event["reason"], "model_transfer_bytes": int(event["transfer_bytes"]),
            "model_network_plus_load_seconds": model_seconds if math.isfinite(model_seconds) else None,
            "state_package_bytes": package_bytes, "state_network_seconds": state_network,
            "state_restore_seconds": state_restore,
            "native_prepare_contact_test_seconds": model_seconds + state_network + state_restore if math.isfinite(model_seconds) else None}


def gate(env: RawNGSIMEventTimeEnv, action: int, cost: float) -> dict:
    trace_remaining = float(env.trace.times_seconds[-1] - env.clock_seconds)
    contact_remaining = float(env._physical_contact_budget_seconds()) if action != 2 else None
    trace_exceeded = bool(cost > trace_remaining + 1e-9)
    contact_exceeded = bool(contact_remaining is not None and cost > contact_remaining + 1e-9)
    if contact_remaining is None:
        first_boundary = "trace_end" if trace_exceeded else "none"
    elif min(trace_remaining, contact_remaining) >= cost - 1e-9:
        first_boundary = "none"
    else:
        first_boundary = "contact_end" if contact_remaining < trace_remaining - 1e-9 else "trace_end"
    return {"trace_remaining_seconds": trace_remaining, "current_contact_remaining_seconds": contact_remaining,
            "deadline_remaining_seconds": float(env.instance["deadline_seconds"] - env.clock_seconds),
            "trace_exceeded": trace_exceeded, "contact_exceeded": contact_exceeded,
            "first_boundary": first_boundary,
            "v4_priority_rejection": "trace_end_before_service_commit" if trace_exceeded else
                                     "current_rsu_contact_expires_before_commit" if contact_exceeded else None}


def future_clone_probe(config: dict, row: dict, trace: RawVehicleTrace) -> dict:
    prefix = trace.xy_metres[:2]
    slow_xy = prefix + tuple(prefix[-1] for _ in trace.xy_metres[2:])
    fast_xy = prefix + tuple((prefix[-1][0], prefix[-1][1] + 30.0) for _ in trace.xy_metres[2:])
    slow = RawNGSIMEventTimeEnv(config, row, RawVehicleTrace(trace.times_seconds, slow_xy, trace.vehicle_id))
    fast = RawNGSIMEventTimeEnv(config, row, RawVehicleTrace(trace.times_seconds, fast_xy, trace.vehicle_id))
    slow_public = slow._info()
    fast_public = fast._info()
    if slow_public != fast_public:
        raise AssertionError("synthetic future perturbation changed public decision state")
    _, _, _, _, slow_info = slow.clone_for_decision_model().step(3)
    _, _, _, _, fast_info = fast.clone_for_decision_model().step(3)
    return {"public_state_equal": True, "action": 3,
            "slow_preview_completed": bool(slow_info["transition"]["service_completed"]),
            "fast_preview_completed": bool(fast_info["transition"]["service_completed"]),
            "slow_preview_rejection": slow_info["transition"].get("admission_rejection_reason"),
            "fast_preview_rejection": fast_info["transition"].get("admission_rejection_reason"),
            "slow_physical_contact_seconds": slow._physical_contact_budget_seconds(),
            "fast_physical_contact_seconds": fast._physical_contact_budget_seconds(),
            "note": "synthetic suffix counterexample only; no source window or workload change"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    v4 = args.v4_root.resolve()
    if sha(v4 / "source_manifest.json") != SOURCE_SHA or sha(v4 / "summary.json") != SUMMARY_SHA:
        raise RuntimeError("v4 source/summary identity mismatch")
    prior_source = json.loads((v4 / "source_manifest.json").read_text())
    prior_summary = json.loads((v4 / "summary.json").read_text())
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    manifest = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instances = [next(row for row in manifest["instances"] if row["design_id"] == design_id)
                 for design_id in FROZEN_DESIGN_IDS]
    traces, source = load_frozen_traces(args.raw_csv_path, instances)
    if source != prior_source or set(traces) != set(FROZEN_DESIGN_IDS):
        raise RuntimeError("raw source interval identity changed")
    by_key = {(x["design_id"], x["profile"], int(x["fixed_action"])): x for x in prior_summary["results"]}
    initial: dict[str, dict] = {}
    ledger: list[dict] = []
    replay_steps = 0
    preview_count = 0
    mismatches = []
    for row in instances:
        design_id = row["design_id"]
        trace = traces[design_id]
        probe = RawNGSIMEventTimeEnv(config, row, trace)
        node_map = {str(node["node_id"]): node for node in row["nodes"]}
        initial_table = []
        for action in probe.valid_actions():
            actual_native = native_phase(probe, action, estimated=False)
            estimated_native = native_phase(probe, action, estimated=True)
            preview_count += 2
            initial_table.append({"action": action,
                                  "native_actual": actual_native,
                                  "native_estimated_link": estimated_native,
                                  "prospective_prepare": prospective_prepare(probe, action),
                                  "raw_gate": gate(probe, action, actual_native["total_seconds"]),
                                  "strict_success_compute_lower_bound_seconds": float(probe._current_node()["compute_seconds"])})
        initial[design_id] = {
            "source_interval": row["source_interval"],
            "selected_vehicle_id": trace.vehicle_id,
            "source_frame_count": len(trace.times_seconds),
            "source_frame_deltas_seconds_unique": sorted(set(round(b-a, 9) for a,b in zip(trace.times_seconds,trace.times_seconds[1:]))),
            "decision_start_seconds": trace.times_seconds[1],
            "trace_end_seconds": trace.times_seconds[-1],
            "trace_remaining_at_first_decision_seconds": trace.times_seconds[-1] - trace.times_seconds[1],
            "actual_contact_at_first_decision_seconds": probe._physical_contact_budget_seconds(),
            "forecast_contact_at_first_decision_seconds": probe._contact_budget_seconds(),
            "deadline_seconds": float(row["deadline_seconds"]),
            "workflow_compute_only_lower_bound_seconds": sum(float(node_map[key]["compute_seconds"]) for key in row["execution_order"]),
            "native_synthetic_sequence_length": len(row["rsu_sequence"]),
            "native_synthetic_slot_seconds": float(config["mobility_abstraction"]["decision_step_seconds"]),
            "native_synthetic_nominal_sequence_seconds": len(row["rsu_sequence"]) * float(config["mobility_abstraction"]["decision_step_seconds"]),
            "legal_action_phase_table": initial_table,
        }
        for profile in prior_summary["profiles"]:
            for requested in range(5):
                env = (CalibratedContinuousWorkflowEnv(config, row) if profile == "decision_step_original"
                       else RawNGSIMEventTimeEnv(config, row, trace))
                counts: Counter[int] = Counter()
                reasons: Counter[str] = Counter()
                masked = services = migrations = 0
                terminated = truncated = False
                for _ in range(min(int(row["max_steps"]), 24)):
                    before = float(env.clock_seconds)
                    raw_native = None
                    raw_gate = None
                    if profile == "raw_ngsim_event_time_v1":
                        mask = env._info()["action_mask"]
                        effective = requested if mask[requested] else (2 if mask[2] else 3 if mask[3] else next(i for i,x in enumerate(mask) if x))
                        raw_native = native_phase(env, effective, estimated=False)
                        raw_gate = gate(env, effective, raw_native["total_seconds"])
                        before_cache = deepcopy(env.caches)
                        before_prepared = deepcopy(env.prepared_state)
                        before_node_index = env.node_index
                        before_bytes = tuple(env.metrics[key] for key in ("model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes"))
                        preview_count += 1
                    _, _, terminated, truncated, info = env.step(requested)
                    replay_steps += 1
                    event = info["transition"]
                    executed = int(event["action"])
                    counts[executed] += 1
                    masked += int(executed != requested)
                    services += int(bool(event["service_completed"]))
                    migrations += int(bool(event.get("migration_success", False)))
                    reason = event.get("admission_rejection_reason")
                    if reason:
                        reasons[str(reason)] += 1
                    if raw_native is not None:
                        if not math.isclose(float(env.clock_seconds) - before, float(event["step_cost_seconds"]), abs_tol=1e-7):
                            raise AssertionError("v4 transition clock/step-cost mismatch")
                        if reason:
                            after_bytes = tuple(env.metrics[key] for key in ("model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes"))
                            if (env.caches != before_cache or env.prepared_state != before_prepared
                                    or env.node_index != before_node_index or after_bytes != before_bytes
                                    or event["service_completed"]):
                                raise AssertionError("rejected v4 action committed resources or DAG state")
                        elif not math.isclose(float(event["step_cost_seconds"]), raw_native["total_seconds"], abs_tol=1e-7):
                            raise AssertionError("accepted v4 step diverged from native stage sum")
                        if reason != raw_gate["v4_priority_rejection"]:
                            mismatches.append({"kind": "gate_reason", "design_id": design_id,
                                               "requested": requested, "step": env.step_index})
                        ledger.append({"design_id": design_id, "requested_action": requested,
                                       "executed_action": executed, "step_index": env.step_index - 1,
                                       "clock_before_seconds": before,
                                       "clock_after_seconds": float(env.clock_seconds),
                                       "public_action_mask": mask,
                                       "native_actual_phase": raw_native,
                                       "physical_gate": raw_gate,
                                       "v4_rejection_reason": reason,
                                       "v4_service_completed": bool(event["service_completed"]),
                                       "v4_elapsed_seconds": float(event["step_cost_seconds"]),
                                       "terminated": terminated, "truncated": truncated})
                    if terminated or truncated:
                        break
                summary = env.summary()
                expected = by_key[(design_id, profile, requested)]
                observed = {"steps": summary["steps"], "terminated": terminated,
                            "truncated": truncated, "completed_nodes": summary["completed_nodes"],
                            "workflow_completed": summary["workflow_completed"],
                            "service_failures": summary["service_failures"],
                            "model_transfer_bytes": summary["model_transfer_bytes"],
                            "state_transfer_bytes": summary["state_transfer_bytes"],
                            "input_transfer_bytes": summary["input_transfer_bytes"],
                            "modeled_completion_seconds": summary["modeled_completion_seconds"],
                            "reward": summary["reward"],
                            "executed_action_counts": {str(k):v for k,v in counts.items()},
                            "masked_request_count": masked,
                            "service_completed_events": services,
                            "migration_success_events": migrations,
                            "admission_rejections": dict(reasons)}
                for key, value in observed.items():
                    old = expected[key]
                    same = math.isclose(float(value), float(old), abs_tol=1e-7) if isinstance(value, float) else value == old
                    if not same:
                        mismatches.append({"kind": "episode_summary", "design_id": design_id,
                                           "profile": profile, "requested": requested, "field": key,
                                           "expected": old, "observed": value})
    if replay_steps != REPLAY_STEP_CAP or preview_count > PREVIEW_CAP:
        raise RuntimeError(f"replay/preview cap violated: {replay_steps}/{preview_count}")
    if mismatches:
        raise AssertionError(f"v4 replay mismatch: {mismatches[:3]}")
    probe = future_clone_probe(config, instances[0], traces[FROZEN_DESIGN_IDS[0]])
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only diagnostic output exists: {output}")
    output.mkdir(parents=True)
    result = {"schema_version": "cscwd_raw_time_root_cause_diagnosis_v1",
              "parent_v4_source_sha256": SOURCE_SHA, "parent_v4_summary_sha256": SUMMARY_SHA,
              "parent_v4_git_commit": prior_summary["git_commit"],
              "replayed_episodes": len(by_key), "replayed_real_steps": replay_steps,
              "additional_policy_episodes": 0, "extra_native_previews": preview_count,
              "summary_mismatches": 0, "initial_action_phase_tables": initial,
              "raw_replayed_step_ledger": ledger,
              "decision_clone_future_probe": probe}
    (output / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "replayed_steps": replay_steps,
                      "raw_ledger_rows": len(ledger), "extra_previews": preview_count,
                      "summary_mismatches": 0, "decision_clone_future_probe": probe}, ensure_ascii=False))


if __name__ == "__main__":
    main()
