"""One-shot, development-only sharing x migration contract execution."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces, sha
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_phase_v2_env import (
    FACTORIAL_PROFILE_VERSION,
    RawNGSIMPhaseV2Env,
)

POLICY_ORDER = (4, 3, 2, 0, 1)
GATE_PATH = ROOT / "configs/experiment/cscwd_sharing_migration_factorial_gate_20261011.json"
SCRIPT_PATH = Path(__file__).resolve()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise RuntimeError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        stream.write("\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_gate(gate: dict, raw_csv: Path) -> tuple[str, dict, list, dict]:
    if gate["contract_version"] != FACTORIAL_PROFILE_VERSION:
        raise RuntimeError("factorial contract version mismatch")
    if gate["execution_authorized"] is not True or gate["gate_status"] != "AUTHORIZED_SINGLE_DEVELOPMENT_RUN":
        raise RuntimeError("factorial execution gate is closed")
    if git("status", "--porcelain"):
        raise RuntimeError("execution requires a clean worktree")
    head = git("rev-parse", "HEAD")
    implementation = gate["fixed_implementation_commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", implementation, head], cwd=ROOT).returncode:
        raise RuntimeError("fixed implementation commit is not an ancestor")
    if git("diff", "--name-only", implementation, head, "--", "scripts/run_cscwd_sharing_migration_factorial.py", "src/envs/core/raw_ngsim_phase_v2_env.py"):
        raise RuntimeError("executor or environment changed after implementation freeze")
    if sha(SCRIPT_PATH) != gate["fixed_policy"]["consumer_identity_sha256"]:
        raise RuntimeError("fixed public policy/executor file hash mismatch")
    if tuple(gate["fixed_policy"]["action_order"]) != POLICY_ORDER:
        raise RuntimeError("fixed public action order changed")
    if gate["budget"] != {
        "independent_source_count_max": 1,
        "workflow_count_max": 1,
        "episode_count_max": 4,
        "real_env_step_count_max": 96,
        "train_steps": 0,
        "model_calls": 0,
        "formal_or_holdout_access": False,
    }:
        raise RuntimeError("factorial budget changed")
    expected_arms = [(False, False), (True, False), (False, True), (True, True)]
    observed_arms = [
        (row["base_sharing_enabled"], row["state_migration_enabled"])
        for row in gate["arms"]
    ]
    if observed_arms != expected_arms or [row["arm_id"] for row in gate["arms"]] != ["S0M0", "S1M0", "S0M1", "S1M1"]:
        raise RuntimeError("four-arm order or factors changed")
    if raw_csv.resolve() != Path(gate["source"]["raw_csv_path"]).resolve():
        raise RuntimeError("raw CSV path differs from frozen gate")
    if sha(raw_csv) != gate["source"]["source_csv_sha256"]:
        raise RuntimeError("raw CSV SHA-256 differs from frozen gate")
    source_path = ROOT / gate["source"]["source_manifest_path"]
    if sha(source_path) != gate["source"]["source_manifest_sha256"]:
        raise RuntimeError("source manifest SHA-256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["excluded_source_segments"] != ["peachtree", "i_80"]:
        raise RuntimeError("excluded source segments changed")
    if source["raw_source_sha256"] != gate["source"]["source_csv_sha256"]:
        raise RuntimeError("source manifest raw SHA mismatch")
    selected = source["selected_windows"]
    if len(selected) != 2 or any(row["decision"] != "selected" for row in selected):
        raise RuntimeError("approved two-source selection changed")
    for row in selected:
        lo, hi = int(row["source_time_start"]), int(row["source_time_end"])
        if len(str(lo)) != 13 or hi - lo != 118800 or row["contiguous_frame_count"] != 1189:
            raise RuntimeError("source raw time identity changed")
        for start, end in source["merged_forbidden_intervals"][row["source_segment_id"]]:
            if max(lo, int(start)) <= min(hi, int(end)):
                raise RuntimeError("source overlaps protected raw interval")
    config_path = ROOT / gate["workload"]["config_path"]
    manifest_path = ROOT / gate["workload"]["manifest_path"]
    protocol_path = ROOT / gate["physical_condition"]["protocol_path"]
    for path, expected in (
        (config_path, gate["workload"]["config_sha256"]),
        (manifest_path, gate["workload"]["manifest_sha256"]),
        (protocol_path, gate["physical_condition"]["protocol_sha256"]),
    ):
        if sha(path) != expected:
            raise RuntimeError(f"frozen input SHA mismatch: {path}")
    config, _ = _load_experiment_config(config_path)
    workload = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = [row for row in workload["instances"] if row["design_id"] == gate["workload"]["design_id"]]
    if len(instances) != 1:
        raise RuntimeError("frozen DAG instance is not unique")
    instance = deepcopy(instances[0])
    if len({node["required_adapter"] for node in instance["nodes"]}) < 2:
        raise RuntimeError("DAG has no two-adapter sharing opportunity")
    instance["cache_capacity_bytes"] = int(gate["physical_condition"]["common_cache_capacity_bytes"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    conditions = [row for row in protocol["conditions"] if row["condition_id"] == gate["physical_condition"]["condition_id"]]
    if len(conditions) != 1 or conditions[0] != gate["physical_condition"]["condition"]:
        raise RuntimeError("physical condition changed")
    traces, raw_identities = load_traces(raw_csv, selected)
    public_candidates = []
    for row in selected:
        environment = RawNGSIMPhaseV2Env(
            config, instance, traces[row["window_id"]],
            phase_contract=protocol["physical_assumptions"], condition=conditions[0],
            factorial_profile={"base_sharing_enabled": True, "state_migration_enabled": True},
        )
        public_candidates.append({"window_id": row["window_id"], "first_action_mask": environment._info()["action_mask"]})
    eligible = [item["window_id"] for item in public_candidates if item["first_action_mask"][4]]
    if not eligible or eligible[0] != gate["source"]["window_id"]:
        raise RuntimeError("outcome-blind first-action-4 source selection changed")
    source_row = next(row for row in selected if row["window_id"] == eligible[0])
    frozen_identity = gate["source"]
    for key, source_key in (
        ("segment", "source_segment_id"), ("vehicle_id", "vehicle_id"),
        ("raw_global_time_start", "source_time_start"), ("raw_global_time_end", "source_time_end"),
        ("historical_train_dev_overlap_count", "historical_train_dev_overlap_count"),
    ):
        if frozen_identity[key] != source_row[source_key]:
            raise RuntimeError(f"frozen source identity changed: {key}")
    if gate["source"]["historical_train_dev_overlap_count"] == 0:
        raise RuntimeError("exposed development source was mislabelled independent")
    return head, {
        "source": source_row,
        "raw_identity": raw_identities[eligible[0]],
        "public_candidate_masks": public_candidates,
        "trace": traces[eligible[0]],
    }, [config, instance, protocol["physical_assumptions"], conditions[0]], gate


def fixed_action(info: dict) -> int:
    mask = info["action_mask"]
    return next(action for action in POLICY_ORDER if mask[action])


def run(gate: dict, head: str, source: dict, inputs: list, root: Path) -> None:
    config, instance, phase_contract, condition = inputs
    trace = source["trace"]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "run_manifest.json", {
        "status": "STARTED", "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_git_commit": head, "fixed_implementation_commit": gate["fixed_implementation_commit"],
        "gate_path": str(GATE_PATH.relative_to(ROOT)), "gate_sha256": sha(GATE_PATH),
        "executor_sha256": sha(SCRIPT_PATH), "source": source["source"],
        "raw_identity": source["raw_identity"], "public_candidate_masks": source["public_candidate_masks"],
        "instance_identity_sha256": digest(instance),
        "phase_contract_sha256": digest(phase_contract), "physical_condition_sha256": digest(condition),
        "arms": gate["arms"], "fixed_policy": gate["fixed_policy"], "budget": gate["budget"],
        "development_only": True, "historically_exposed": True,
    })
    receipts: list[dict] = []
    total_steps = 0
    try:
        selected_public_mask = next(
            row["first_action_mask"] for row in source["public_candidate_masks"]
            if row["window_id"] == source["source"]["window_id"]
        )
        for arm in gate["arms"]:
            arm_id = arm["arm_id"]
            environment = RawNGSIMPhaseV2Env(
                config, deepcopy(instance), trace,
                phase_contract=phase_contract, condition=condition,
                factorial_profile={
                    "base_sharing_enabled": arm["base_sharing_enabled"],
                    "state_migration_enabled": arm["state_migration_enabled"],
                },
            )
            initial_info = environment._info()
            if initial_info["action_mask"] != selected_public_mask:
                raise RuntimeError("four-arm initial action permissions are not identical")
            if environment.instance["nodes"] != instance["nodes"] or environment.config["measured_identity"] != config["measured_identity"]:
                raise RuntimeError("four-arm task or model function identity changed")
            initial_residents = {
                rsu_id: list(cache.residents) for rsu_id, cache in environment.caches.items()
            }
            logical_residents = {
                rsu_id: sorted({environment.private_base_sources.get(item, item) for item in values})
                for rsu_id, values in initial_residents.items()
            }
            if logical_residents != {
                rsu_id: sorted(set(values)) for rsu_id, values in instance["initial_residents"].items()
            }:
                raise RuntimeError("initial logical cache resources differ between arms")
            initial_clock = environment.clock_seconds
            write_json(root / arm_id / "initial.json", {
                "arm": arm, "instance_identity_sha256": digest(environment.instance),
                "trace_identity_sha256": digest(source["raw_identity"]),
                "initial_action_mask": initial_info["action_mask"],
                "initial_logical_residents": logical_residents,
                "initial_physical_residents": initial_residents,
                "initial_resident_bytes_by_rsu": {
                    rsu_id: environment._resident_bytes(cache)
                    for rsu_id, cache in environment.caches.items()
                },
                "initial_clock_seconds": initial_clock,
            })
            terminal = truncated = False
            records = []
            while not terminal and not truncated:
                if total_steps >= gate["budget"]["real_env_step_count_max"]:
                    raise RuntimeError("global real environment step cap reached")
                before = environment._info()
                action = fixed_action(before)
                node_before = before["semantic_state"]["current_workflow_node"]
                _, _, terminal, truncated, after = environment.step(action)
                total_steps += 1
                event = after["transition"]
                if event["action"] != action or event.get("action_projection_applied"):
                    raise RuntimeError("fixed public action was projected")
                if event.get("factorial_profile", {}).get("base_sharing_enabled") != arm["base_sharing_enabled"] or event.get("factorial_profile", {}).get("state_migration_enabled") != arm["state_migration_enabled"]:
                    raise RuntimeError("factorial switch not consumed by transition")
                if arm["state_migration_enabled"] is False and event["state_transfer_bytes"]:
                    raise RuntimeError("migration-off transferred workflow state")
                if event["backhaul_transfer_bytes"] != event["model_transfer_bytes"] + event["state_transfer_bytes"]:
                    raise RuntimeError("backhaul byte conservation failed")
                if event["time_profile"] == "raw_ngsim_event_time_v2" and event["action"] != 2:
                    if not math.isclose(event["step_cost_seconds"], sum(row["elapsed_seconds"] for row in event["phase_events"]), abs_tol=1e-6):
                        raise RuntimeError("phase clock conservation failed")
                record = {
                    "arm_id": arm_id, "step_index": total_steps, "arm_step_index": len(records),
                    "public_action_mask_before": before["action_mask"],
                    "public_state_sha256_before": digest(before["semantic_state"]),
                    "node_before": node_before, "fixed_policy_action": action,
                    "transition": event, "episode_metrics_after": after["episode_metrics"],
                    "public_action_mask_after": after["action_mask"],
                    "terminated": terminal, "truncated": truncated,
                }
                write_json(root / arm_id / f"step_{len(records):03d}.json", record)
                records.append(record)
                if len(records) > instance["max_steps"]:
                    raise RuntimeError("instance step cap exceeded")
            summary = environment.summary()
            if not math.isclose(
                summary["modeled_completion_seconds"],
                initial_clock + sum(row["transition"]["step_cost_seconds"] for row in records),
                abs_tol=1e-6,
            ):
                raise RuntimeError("episode modeled time does not match step ledger")
            if summary["total_transfer_bytes"] != sum(
                row["transition"]["model_transfer_bytes"]
                + row["transition"]["state_transfer_bytes"]
                + row["transition"]["input_transfer_bytes"] for row in records
            ):
                raise RuntimeError("episode transfer bytes do not match step ledger")
            environment._validate_all_caches()
            deliveries = [
                (index, row["transition"]["predicted_target_rsu_id"])
                for index, row in enumerate(records) if row["transition"].get("migration_success")
            ]
            followup_consumptions = sum(
                any(
                    later["transition"]["execution_owner_rsu_id"] == target
                    and later["transition"]["service_completed"]
                    and later["transition"]["recompute_seconds"] == 0
                    for later in records[index + 1:]
                ) for index, target in deliveries
            )
            base_reuse = 0
            base_admissions = 0
            for row in records:
                for event in row["transition"].get("cache_events", []):
                    if event.get("status") != "completed" or not event.get("committed"):
                        continue
                    admitted = list(event.get("admitted", []))
                    base_admissions += sum(
                        environment._object_catalog[item]["object_type"] == "base_model"
                        for item in admitted
                    )
                    if any(environment._object_catalog[item]["object_type"] == "adapter" for item in admitted):
                        bundle = environment._bundle_ids(str(event["adapter_id"]))
                        base_id = bundle[0]
                        base_reuse += int(base_id in event.get("before_residents", []) and base_id not in admitted)
            receipt = {
                "arm_id": arm_id, "factors": arm, "status": "COMPLETE",
                "steps": len(records), "terminated": terminal, "truncated": truncated,
                "summary": summary, "base_admissions": base_admissions,
                "base_reuse_on_new_adapter_admission": base_reuse,
                "migration_deliveries": len(deliveries),
                "delivered_target_followup_service_without_recompute": followup_consumptions,
                "step_file_sha256": {
                    f"step_{index:03d}.json": sha(root / arm_id / f"step_{index:03d}.json")
                    for index in range(len(records))
                },
            }
            write_json(root / arm_id / "receipt.json", receipt)
            receipts.append(receipt)
        if len(receipts) != 4:
            raise RuntimeError("four arms not complete")
        by_id = {row["arm_id"]: row for row in receipts}
        comparable = all(row["terminated"] and not row["truncated"] for row in receipts)
        endpoints = (
            "workflow_completed", "on_time_workflow_completed", "service_failures",
            "total_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
            "input_transfer_bytes", "backhaul_transfer_bytes", "modeled_completion_seconds",
            "recompute_seconds", "model_load_seconds", "state_restore_seconds",
        )
        effects = {}
        for endpoint in endpoints:
            y = {key: float(row["summary"][endpoint]) for key, row in by_id.items()}
            effects[endpoint] = {
                "descriptive_sharing_main": ((y["S1M0"] - y["S0M0"]) + (y["S1M1"] - y["S0M1"])) / 2,
                "descriptive_migration_main": ((y["S0M1"] - y["S0M0"]) + (y["S1M1"] - y["S1M0"])) / 2,
                "descriptive_difference_in_differences": (y["S1M1"] - y["S1M0"]) - (y["S0M1"] - y["S0M0"]),
                "comparable_completed_endpoint": comparable,
                "cost_advantage_interpretable": comparable and endpoint not in {"workflow_completed", "on_time_workflow_completed", "service_failures"},
            }
        summary = {
            "status": "COMPLETE", "run_id": root.name, "development_only": True,
            "historically_exposed": True, "independent_outer_cluster_count": 1,
            "execution_git_commit": head, "total_real_env_steps": total_steps,
            "four_arm_count": len(receipts), "all_four_completed_without_truncation": comparable,
            "arm_receipts": {key: str(Path(key) / "receipt.json") for key in by_id},
            "endpoints": {key: row["summary"] for key, row in by_id.items()},
            "mechanism_counts": {key: {
                "base_admissions": row["base_admissions"],
                "base_reuse_on_new_adapter_admission": row["base_reuse_on_new_adapter_admission"],
                "migration_deliveries": row["migration_deliveries"],
                "delivered_target_followup_service_without_recompute": row["delivered_target_followup_service_without_recompute"],
            } for key, row in by_id.items()},
            "descriptive_effects": effects,
            "claim_boundary": "one historically exposed US101 development trace; no statistical generalization or paper claim",
        }
        write_json(root / "summary.json", summary)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        write_json(root / "artifact_integrity.json", {
            "status": "COMPLETE", "file_count": len(files),
            "files": {str(path.relative_to(root)): sha(path) for path in files},
        })
        write_json(root / "completion_receipt.json", {
            "status": "COMPLETE", "summary_sha256": sha(root / "summary.json"),
            "artifact_integrity_sha256": sha(root / "artifact_integrity.json"),
            "total_real_env_steps": total_steps,
        })
    except Exception as error:
        write_json(root / "failure_receipt.json", {
            "status": "PARTIAL_UNVERIFIABLE", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "total_real_env_steps": total_steps,
            "completed_arm_receipts": [row["arm_id"] for row in receipts],
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    head, source, inputs, gate = validate_gate(gate, args.raw_csv)
    output_root = ROOT / gate["output"]["create_only_root"]
    if output_root.exists():
        raise RuntimeError("create-only result root already exists")
    run(gate, head, source, inputs, output_root)
    print(canonical({"status": "COMPLETE", "run_id": output_root.name, "result_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
