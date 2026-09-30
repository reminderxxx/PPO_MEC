"""Audit native cache ledgers and the legal action surface for workflow value.

This script only consumes the previously delivered four-configuration logs.  It
does not rerun their request matrix or replace unavailable timing fields with 0.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.specs.action_schema import ActionAdapter, ActionMaskBuilder, ActionSchema  # noqa: E402


MIB = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _object_type(row: dict[str, Any]) -> str:
    return str(row.get("object_type") or "unavailable")


def ledger_for_config(rows: list[dict[str, Any]], semantics_label: str) -> dict[str, Any]:
    admitted_count: Counter[str] = Counter()
    admitted_bytes: Counter[str] = Counter()
    evicted_count: Counter[str] = Counter()
    evicted_bytes: Counter[str] = Counter()
    transfer_bytes: Counter[str] = Counter()
    load_occurrences: Counter[str] = Counter()
    repeated_load_count: Counter[str] = Counter()
    repeated_load_bytes: Counter[str] = Counter()
    base_reuse_requests = 0
    base_initial_load_requests = 0
    statuses: Counter[str] = Counter()

    for row in rows:
        statuses[str(row["atomic_transaction_status"])] += 1
        for object_type, value in row.get(
            "transfer_bytes_by_type_under_explicit_mapping", {}
        ).items():
            transfer_bytes[str(object_type)] += int(value)
        admitted = list(row.get("admitted_typed_objects") or [])
        evicted = list(row.get("evicted_typed_objects") or [])
        for item in admitted:
            object_type = _object_type(item)
            size = int(round(float(item["resident_size_mb"]) * MIB))
            admitted_count[object_type] += 1
            admitted_bytes[object_type] += size
            object_id = str(item["object_id"])
            load_occurrences[object_id] += 1
            if load_occurrences[object_id] > 1:
                repeated_load_count[object_type] += 1
                repeated_load_bytes[object_type] += size
        for item in evicted:
            object_type = _object_type(item)
            size = int(round(float(item["resident_size_mb"]) * MIB))
            evicted_count[object_type] += 1
            evicted_bytes[object_type] += size

        base_id = next(
            (
                str(object_id)
                for object_id in row["dependency_bundle"]["ordered_object_ids"]
                if str(object_id).startswith("base:")
            ),
            None,
        )
        resident_before_ids = set(row["resident_before"]["native_object_ids"])
        if base_id is not None and base_id in resident_before_ids:
            base_reuse_requests += 1
        elif any(_object_type(item) == "base_model" for item in admitted):
            base_initial_load_requests += 1

    request_count = len(rows)
    success = sum(bool(row.get("service_success")) for row in rows)
    failure = sum(bool(row.get("service_failure")) for row in rows)
    return {
        "config_id": rows[0]["config_id"],
        "semantics_label": semantics_label,
        "typed_eviction_semantics": rows[0]["typed_eviction_semantics"],
        "request_count": request_count,
        "service_success_count": success,
        "service_failure_count": failure,
        "completed_request_count": success,
        "workflow_completion_count": None,
        "workflow_completion_availability": "unavailable_request_replay_has_no_workflow_progression",
        "transaction_status_counts": dict(sorted(statuses.items())),
        "actual_load_count_by_type": dict(sorted(admitted_count.items())),
        "actual_load_bytes_by_type": dict(sorted(admitted_bytes.items())),
        "actual_transfer_bytes_by_type": dict(sorted(transfer_bytes.items())),
        "actual_transfer_bytes_total": sum(transfer_bytes.values()),
        "evicted_object_count_by_type": dict(sorted(evicted_count.items())),
        "evicted_bytes_by_type": dict(sorted(evicted_bytes.items())),
        "evicted_bytes_total": sum(evicted_bytes.values()),
        "repeated_load_count_by_type": dict(sorted(repeated_load_count.items())),
        "repeated_load_bytes_by_type": dict(sorted(repeated_load_bytes.items())),
        "repeated_load_bytes_total": sum(repeated_load_bytes.values()),
        "base_reuse_request_count": base_reuse_requests,
        "base_initial_load_request_count": base_initial_load_requests,
        "transfer_bytes_per_completed_request": (
            float(sum(transfer_bytes.values())) / float(success) if success else None
        ),
        "waiting_cost": None,
        "execution_cost": None,
        "state_migration_cost": None,
        "state_migration_applicability": "not_applicable_no_handoff_or_state_request_in_four_config_replay",
        "timing_availability": "unavailable_no_wait_execution_or_transfer_time_fields",
        "hardware_measurement": False,
        "ledger_source": "native CacheEvent/transaction fields under explicit MiB mapping",
    }


def build_native_ledger(old_path: Path, candidate_path: Path) -> dict[str, Any]:
    old_rows = read_jsonl(old_path)
    candidate_rows = read_jsonl(candidate_path)
    if len(old_rows) != 288 or len(candidate_rows) != 288:
        raise ValueError("expected exactly 288 old and 288 candidate request rows")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for label, rows in (("old", old_rows), ("candidate", candidate_rows)):
        for row in rows:
            grouped[(label, str(row["config_id"]))].append(row)
    ledgers = [
        ledger_for_config(rows, label)
        for (label, _), rows in sorted(grouped.items())
    ]
    candidate_ledgers = [row for row in ledgers if row["semantics_label"] == "candidate"]
    old_ledgers = [row for row in ledgers if row["semantics_label"] == "old"]
    if {row["completed_request_count"] for row in candidate_ledgers} != {72}:
        raise AssertionError("candidate equal-completion comparison requires 72 successes each")
    same_completion = sorted(
        (
            {
                "config_id": row["config_id"],
                "completed_request_count": row["completed_request_count"],
                "transfer_bytes_total": row["actual_transfer_bytes_total"],
                "base_transfer_bytes": row["actual_transfer_bytes_by_type"].get("base_model", 0),
                "adapter_transfer_bytes": row["actual_transfer_bytes_by_type"].get("adapter", 0),
                "evicted_bytes_total": row["evicted_bytes_total"],
                "repeated_load_bytes_total": row["repeated_load_bytes_total"],
                "base_reuse_request_count": row["base_reuse_request_count"],
            }
            for row in candidate_ledgers
        ),
        key=lambda row: (row["transfer_bytes_total"], row["config_id"]),
    )
    return {
        "artifact_run_id": "remaining_workflow_decision_value_audit_20260930_v1",
        "evidence_type": "native_simulation_ledger",
        "input_files": {
            "old": {
                "path": str(old_path),
                "sha256": sha256_file(old_path),
                "row_count": len(old_rows),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
                "row_count": len(candidate_rows),
            },
        },
        "ledgers": ledgers,
        "same_completion_comparison": {
            "scope": "candidate semantics only; all four configurations complete 72 requests",
            "rows": same_completion,
            "minimum_transfer_config": same_completion[0]["config_id"],
            "guard": "old semantics is excluded because its lower transfer coincides with 36 or 60 failed services",
        },
        "old_low_transfer_interpretation": [
            {
                "config_id": row["config_id"],
                "completed_requests": row["completed_request_count"],
                "failed_requests": row["service_failure_count"],
                "transfer_bytes": row["actual_transfer_bytes_total"],
                "interpretation": "rejected service suppressed transfer; not an achieved saving at equal completion",
            }
            for row in sorted(old_ledgers, key=lambda item: item["config_id"])
        ],
        "unavailable_fields_are_null_not_zero": [
            "workflow_completion_count",
            "waiting_cost",
            "execution_cost",
            "state_migration_cost",
        ],
    }


def action_reachability() -> dict[str, Any]:
    state = {
        "primary_vehicle_id": "veh_1",
        "vehicles": [{"vehicle_id": "veh_1", "associated_rsu_id": "rsu_a"}],
        "rsus": [
            {"rsu_id": "rsu_a", "cached_adapter_ids": []},
            {"rsu_id": "rsu_b", "cached_adapter_ids": []},
        ],
        "current_workflow_node": {
            "node_id": "n0",
            "required_base_model": "b0",
            "required_adapter": "b0.a0",
        },
        "predictions": {
            "next_rsu_sequence": {"veh_1": ["rsu_b"]},
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
        },
    }
    schema = ActionSchema.default_vec_workflow_schema()
    mask_info = ActionMaskBuilder(schema).build_mask_info(state)
    adapter = ActionAdapter(schema)
    decoded = []
    for action_id in range(schema.discrete_action_count):
        control = adapter.decode(action_id, state).to_dict()
        decoded.append(
            {
                "action_id": action_id,
                "action_name": schema.action_name(action_id),
                "mask_allowed": bool(mask_info["mask"][action_id]),
                "control_action": control,
            }
        )
    migration_modes = sorted(
        {str(row["control_action"]["migration_action"].get("mode", "keep")) for row in decoded}
    )
    cache_adapter_ids = sorted(
        {
            str(row["control_action"]["cache_action"].get("adapter_id"))
            for row in decoded
            if row["control_action"]["cache_action"].get("adapter_id") is not None
        }
    )
    return {
        "evidence_type": "native_action_contract_audit",
        "action_contract": "semantic_discrete_5",
        "schema": schema.to_dict(),
        "sample_state": state,
        "mask_info": mask_info,
        "decoded_actions": decoded,
        "reachable_migration_modes": migration_modes,
        "reachable_cache_adapter_ids": cache_adapter_ids,
        "future_adapter_direct_prepare": {
            "reachable": False,
            "reason": "all cache actions bind adapter_id to the current workflow node",
        },
        "explicit_state_migrate": {
            "reachable": "migrate" in migration_modes,
            "reason": "ActionAdapter emits keep or prepare, never migrate",
        },
        "continue_old_rsu_and_forward_result": {
            "reachable": False,
            "reason": "no action or result-forwarding field exists in semantic_discrete_5",
        },
        "bounded_consequence": "performance witnesses requiring an explicit migrate action or old-RSU result forwarding must stop; current-adapter prefetch remains testable",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_native_ledger(args.old_log, args.candidate_log)
    actions = action_reachability()
    write_json(args.output_root / "native_four_config_ledger.json", ledger)
    write_json(args.output_root / "native_action_reachability.json", actions)
    print(json.dumps({
        "ledger_rows": len(ledger["ledgers"]),
        "same_completion_rows": len(ledger["same_completion_comparison"]["rows"]),
        "reachable_migration_modes": actions["reachable_migration_modes"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
