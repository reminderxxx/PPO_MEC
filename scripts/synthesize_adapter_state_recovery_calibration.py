"""Synthesize the frozen native long-tail ledger with bounded calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
EXPECTED_PREVIOUS_SHA256 = "041402eda14c3d85635342022c9027e3c57fd9b966f07646a169b16c5d1713db"
LONG_INSTANCES = ("tail_length__long", "tail_adapter_structure__long")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def native_rows(previous: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = {
        (row["instance_id"], row["rule_id"]): row
        for row in previous["episodes"]
    }
    rows = []
    for instance_id in LONG_INSTANCES:
        ablated = episodes[(instance_id, "remaining_reuse_information_ablated")]
        prepare = episodes[(instance_id, "remaining_workflow_handoff_prepare")]
        ablated_transfer = sum(ablated["summary"]["transfer_bytes_by_type"].values())
        prepare_transfer = sum(prepare["summary"]["transfer_bytes_by_type"].values())
        realized_steps = [
            row for row in prepare["steps"] if row["cache_event"].get("migration_realized") is True
        ]
        state_bytes_raw = sum(
            int(round(float(row["cache_event"].get("state_migration_size_mb", 0.0)) * MIB))
            for row in prepare["steps"]
        )
        rows.append(
            {
                "instance_id": instance_id,
                "ablated_first_action": ablated["steps"][0]["action_id"],
                "prepare_first_action": prepare["steps"][0]["action_id"],
                "prepare_action_semantics": {
                    "cache_action": prepare["steps"][0]["control_action"]["cache_action"],
                    "migration_action": prepare["steps"][0]["control_action"]["migration_action"],
                    "changes_location_or_state": "places the current-node adapter at the predicted handoff target and records a prepare history entry; it does not serialize, transfer, or import an application/runtime state object",
                },
                "ablated_model_transfer_bytes": ablated_transfer,
                "prepare_model_transfer_bytes": prepare_transfer,
                "avoided_synthetic_model_transfer_bytes": ablated_transfer - prepare_transfer,
                "node_completion": {
                    "ablated": ablated["summary"]["node_completion_count"],
                    "prepare": prepare["summary"]["node_completion_count"],
                    "node_total": prepare["summary"]["node_total"],
                },
                "service_failure_count": {
                    "ablated": ablated["summary"]["service_failure_count"],
                    "prepare": prepare["summary"]["service_failure_count"],
                },
                "native_migration_realized_step_count": len(realized_steps),
                "native_state_bytes_raw": state_bytes_raw,
                "native_state_bytes_interpretation": "unavailable_missing_workflow_state_object_or_payload_accounting_not_zero_cost",
                "equal_completion": (
                    ablated["summary"]["node_completion_count"]
                    == prepare["summary"]["node_completion_count"]
                ),
                "equal_failures": (
                    ablated["summary"]["service_failure_count"]
                    == prepare["summary"]["service_failure_count"]
                ),
            }
        )
    return rows


def sensitivity_grid(plan: dict[str, Any], avoided_bytes: int) -> list[dict[str, Any]]:
    rows = []
    for link_mbps in plan["decision_calibration"]["link_rate_interval_mbps_assumption"]:
        bits_per_second = float(link_mbps) * 1_000_000.0
        avoided_transfer_ms = avoided_bytes * 8.0 / bits_per_second * 1000.0
        for fixed_ms in plan["decision_calibration"][
            "fixed_one_way_transport_overhead_ms_assumption"
        ]:
            maximum_state_bytes = max(
                0.0,
                avoided_bytes - (float(fixed_ms) / 1000.0) * bits_per_second / 8.0,
            )
            rows.append(
                {
                    "link_mbps_assumption": float(link_mbps),
                    "fixed_state_transport_overhead_ms_assumption": float(fixed_ms),
                    "avoided_synthetic_model_transfer_ms": avoided_transfer_ms,
                    "network_only_break_even_state_bytes": maximum_state_bytes,
                    "network_only_break_even_state_mib": maximum_state_bytes / MIB,
                    "condition": "true_state_bytes must be below this threshold before adding save, restore, target rebuild, model load, queue, retransmission, or adapter-switch costs",
                    "wireless_measurement": False,
                }
            )
    return rows


def synthesize(
    previous_path: Path,
    measurement_path: Path,
    inventory_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    previous_sha256 = sha256_file(previous_path)
    if previous_sha256 != EXPECTED_PREVIOUS_SHA256:
        raise ValueError("frozen previous paired witness hash mismatch")
    previous = load_json(previous_path)
    measurement = load_json(measurement_path)
    inventory = load_json(inventory_path)
    plan = load_json(plan_path)
    rows = native_rows(previous)
    if any(row["avoided_synthetic_model_transfer_bytes"] != 104 * MIB for row in rows):
        raise ValueError("frozen long-tail avoided bytes changed")
    if not all(row["equal_completion"] and row["equal_failures"] for row in rows):
        raise ValueError("frozen equal-outcome native comparison changed")
    warmup_source = measurement["warmup"]["rows"].get("source", {})
    warmup_target = measurement["warmup"]["rows"].get("target", {})
    diagnostic_state = warmup_source.get("state") or {}
    return {
        "artifact_run_id": plan["artifact_run_id"],
        "evidence_type": "layered_native_ledger_and_failed_calibration_sensitivity",
        "inputs": {
            "previous_paired_witness": {
                "path": str(previous_path),
                "sha256": previous_sha256,
            },
            "measurement": {
                "path": str(measurement_path),
                "sha256": sha256_file(measurement_path),
            },
            "adapter_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "measurement_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
        },
        "action4_semantics": {
            "contract": "semantic_discrete_5",
            "changes": "prefetches the current workflow node adapter to the predicted handoff target and emits migration mode=prepare",
            "does_not_change": [
                "current workflow node identity",
                "future-node adapter identity",
                "application state serialization/import",
                "production action codec"
            ],
            "native_zero_state_bytes_meaning": "no workflow-state payload/object bytes were emitted; this is missing state evidence, not measured zero cost",
            "calibration_success_does_not_imply_native_support": True,
        },
        "native_synthetic_ledger_unchanged": rows,
        "adapter_compatibility": {
            "status": "unavailable",
            "local_adapter_count": len(inventory["local_inventory"]["adapter_config_paths"]),
            "candidate_weight_bytes": inventory["minimum_new_weight_budget"][
                "adapter_weight_bytes"
            ],
            "downloaded_weight_bytes": measurement["adapter_execution"][
                "downloaded_weight_bytes"
            ],
            "executed_adapters": 0,
            "compatibility_claim": False,
            "task_quality_claim": False,
            "blockers": [
                "new adapter weights were not authorized",
                "PEFT/accelerate are absent from the frozen environment",
                "both shortlisted task-relevant adapter configs declare task_type=null and require compatibility review"
            ],
        },
        "workflow_state_recovery": {
            "status": "failed_warmup_before_formal_measurements",
            "source_process_saved_state": warmup_source.get("status") == "pass",
            "source_pid": warmup_source.get("pid"),
            "target_pid": warmup_target.get("pid"),
            "distinct_processes": warmup_source.get("pid") != warmup_target.get("pid"),
            "source_state_bytes_diagnostic_only": diagnostic_state.get("bytes"),
            "source_state_sha256_diagnostic_only": diagnostic_state.get("sha256"),
            "source_state_serialization_ns_warmup_only": diagnostic_state.get(
                "serialization_ns"
            ),
            "source_state_durable_save_ns_warmup_only": diagnostic_state.get(
                "durable_write_ns"
            ),
            "target_failure": {
                "error_type": warmup_target.get("error_type"),
                "error": warmup_target.get("error"),
            },
            "failure_reason": "the fixed base generated 'lane status.' rather than the pre-frozen expected normalized value 'clear'; target validation rejected the state before model load or downstream execution",
            "independent_process_restore_witness": False,
            "final_output_equivalence": "unavailable",
            "formal_measurement_count": measurement["measurement_count"],
            "save_restore_load_cost_calibration": "unavailable",
            "diagnostic_state_is_complete_validated_migration_state": False,
        },
        "calibration_and_sensitivity": {
            "validated_state_bytes": None,
            "measured_save_ns": None,
            "measured_restore_ns": None,
            "measured_target_rebuild_ns": None,
            "measured_target_model_load_ns": None,
            "measured_adapter_switch_ns": None,
            "measured_network_time_ns": None,
            "byte_only_condition": "true_validated_state_bytes < 109051904",
            "time_condition": "state_save + state_transfer + state_restore + target_rebuild + target_model_load + adapter_switch + waiting < avoided_model_transfer + avoided_model_load + avoided_switch",
            "finite_assumption_grid": sensitivity_grid(plan, 104 * MIB),
            "grid_scope": "network-only algebra on the original synthetic 104 MiB avoided bytes; it excludes all unavailable measured costs and is not a deployment result",
            "real_model_byte_substitution_performed": False,
        },
        "decision_effect": {
            "native_witness": "unchanged: action 4 retains equal completion/failures and 104 MiB fewer synthetic model bytes in both long-tail instances",
            "after_calibration": "unverifiable because neither a validated recoverable state nor real adapter load/switch/network costs are available",
            "support_or_weaken": "the native mechanism witness remains intact, but the failed recovery and unavailable adapters weaken readiness for a real-workload benefit claim",
            "algorithm_superiority": "not evaluated",
            "next_small_scale_method_comparison_ready": False,
            "blocking_conditions": [
                "authorized and reviewed real adapter pair",
                "actual A→B→A load/switch/execute witness",
                "pre-frozen workflow whose continuous and restored paths both satisfy task correctness",
                "three valid save/restore/load measurements",
                "explicit network assumption or measurement kept separate from local disk timing"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-paired", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    output = synthesize(
        args.previous_paired,
        args.measurement,
        args.inventory,
        args.plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode())
        handle.write(b"\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "adapter_status": output["adapter_compatibility"]["status"],
                "recovery_status": output["workflow_state_recovery"]["status"],
                "next_comparison_ready": output["decision_effect"][
                    "next_small_scale_method_comparison_ready"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
