"""Compare continuous, restart, and suffix-recovery paths for the frozen two-node workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_two_node_workflow_suffix_recovery import (  # noqa: E402
    current_commit,
    directory_bytes,
    execute_continuous,
    execute_source,
    execute_target,
    inventory_tree,
    read_json,
    run_child,
    run_negative_checks,
    sha256_file,
    production_action4_input_identity,
    production_action4_workflow,
    state_identity,
    write_json,
)
from src.runtime.workflow_suffix_recovery import canonical_json_bytes  # noqa: E402
from src.runtime.production_action4_state import import_action4_state  # noqa: E402


def execute_restart(plan_path: Path, run_root: Path) -> dict[str, Any]:
    receipt = execute_continuous(plan_path, run_root)
    receipt["role"] = "restart"
    receipt["restart_semantics"] = "independent target process re-read the original input and re-executed n0 then n1"
    receipt["target_access"] = {
        "source_image_access_count": receipt["n0_input"]["image_access_count"],
        "source_process_memory_access": False,
        "saved_state_package_access": False,
    }
    return receipt


def compare_receipts(
    continuous: dict[str, Any],
    source: dict[str, Any],
    restart: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    n0_receipts = (continuous, source, restart)
    n1_receipts = (continuous, restart, target)
    n0_texts = [item["n0_output"]["decoded_text"] for item in n0_receipts]
    n0_tokens = [item["n0_output"]["token_ids"] for item in n0_receipts]
    checks = {
        "all_n0_text_equal": len(set(n0_texts)) == 1,
        "all_n0_token_ids_equal": all(item == n0_tokens[0] for item in n0_tokens[1:]),
        "all_n1_prompt_equal": len({item["n1_input"]["prompt"] for item in n1_receipts}) == 1,
        "all_n1_rendered_prompt_equal": len({item["n1_input"]["rendered_prompt"] for item in n1_receipts}) == 1,
        "all_n1_input_hash_equal": len({item["n1_input"]["content_sha256"] for item in n1_receipts}) == 1,
        "all_n1_input_ids_equal": all(item["n1_input"]["input_ids"] == n1_receipts[0]["n1_input"]["input_ids"] for item in n1_receipts[1:]),
        "all_n1_token_ids_equal": all(item["n1_output"]["token_ids"] == n1_receipts[0]["n1_output"]["token_ids"] for item in n1_receipts[1:]),
        "identity_equal": continuous["identity"] == source["identity"] == restart["identity"] == target["identity"],
        "active_adapter_equal": len({tuple(item["adapter_status"]["active_adapters"]) for item in (continuous, source, restart, target)}) == 1,
        "continuous_completed_two_nodes": continuous["node_call_counts"] == {"n0": 1, "n1": 1},
        "restart_completed_two_nodes": restart["node_call_counts"] == {"n0": 1, "n1": 1},
        "recovery_executed_suffix_only": target["node_call_counts"] == {"n0": 0, "n1": 1},
        "target_consumed_source_intermediate": target["restored_n0_output"] == {
            "raw_text": source["n0_output"]["decoded_text"],
            "token_ids": source["n0_output"]["token_ids"],
        },
        "recovery_target_source_image_access_zero": target["target_access"]["source_image_access_count"] == 0,
        "restart_target_source_image_access_one": restart["target_access"]["source_image_access_count"] == 1,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _sum_load(receipt: dict[str, Any]) -> float:
    return sum(float(value) for value in receipt["model_load_timings"].values())


def cost_decomposition(
    plan: dict[str, Any],
    continuous: dict[str, Any],
    source: dict[str, Any],
    restart: dict[str, Any],
    target: dict[str, Any],
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    phase_by_role = {item["role"]: item for item in phases}
    legacy_state = source["state"]
    state = source["production_action4_state_export"]
    image_bytes = int(plan["source_input_provenance"]["bytes"])
    actual_image_bytes = Path(plan["source_input_provenance"]["path"]).stat().st_size
    if actual_image_bytes != image_bytes:
        raise RuntimeError(f"source image byte drift: expected {image_bytes}, got {actual_image_bytes}")
    n0_intermediate_bytes = len(canonical_json_bytes(source["n0_output"]))
    base_path = Path(plan["identity"]["base"]["path"])
    adapter_path = Path(plan["identity"]["adapter"]["path"])
    static = {
        "base_directory_bytes": directory_bytes(base_path),
        "base_weight_bytes": (base_path / "model.safetensors").stat().st_size,
        "adapter_directory_bytes": directory_bytes(adapter_path),
        "adapter_weight_bytes": (adapter_path / "adapter_model.safetensors").stat().st_size,
    }
    arms = {
        "continuous": {
            "executed_nodes": ["n0", "n1"],
            "generate_calls": 2,
            "model_load_seconds": _sum_load(continuous),
            "n0_inference_seconds": continuous["n0_output"]["inference_seconds"],
            "n1_input_rebuild_seconds": None,
            "n1_inference_seconds": continuous["n1_output"]["inference_seconds"],
            "process_elapsed_seconds": continuous["process_elapsed_seconds"],
            "supervisor_child_wall_seconds": phase_by_role["continuous"]["wall_seconds"],
            "dynamic_transfer_bytes_if_remote": 0,
            "queue_wait_seconds": None,
            "queue_wait_status": "unavailable_single_process_technical_run_no_rsu_queue_model",
        },
        "interrupted_then_restart_from_n0": {
            "executed_nodes_after_interrupt": ["n0", "n1"],
            "generate_calls_including_shared_prefix": 3,
            "shared_source_prefix_process_seconds": source["process_elapsed_seconds"],
            "target_model_load_seconds": _sum_load(restart),
            "target_n0_inference_seconds": restart["n0_output"]["inference_seconds"],
            "target_n1_inference_seconds": restart["n1_output"]["inference_seconds"],
            "path_process_elapsed_seconds": source["process_elapsed_seconds"] + restart["process_elapsed_seconds"],
            "path_supervisor_child_wall_seconds": phase_by_role["source"]["wall_seconds"] + phase_by_role["restart"]["wall_seconds"],
            "dynamic_transfer_bytes_if_remote": image_bytes,
            "dynamic_transfer_type": "original_input_image",
            "queue_wait_seconds": None,
            "queue_wait_status": "unavailable_single_process_technical_run_no_rsu_queue_model",
        },
        "interrupted_then_restore_and_execute_n1_only": {
            "executed_nodes_after_interrupt": ["n1"],
            "generate_calls_including_shared_prefix": 2,
            "shared_source_prefix_process_seconds": source["process_elapsed_seconds"],
            "serialize_and_save_seconds": state["serialize_and_save_seconds"],
            "state_restore_validation_seconds": target["state_restore_validation_seconds"],
            "input_reconstruction_seconds": target["input_reconstruction_seconds"],
            "target_model_load_seconds": _sum_load(target),
            "target_n1_inference_seconds": target["n1_output"]["inference_seconds"],
            "path_process_elapsed_seconds": source["process_elapsed_seconds"] + target["process_elapsed_seconds"],
            "path_supervisor_child_wall_seconds": phase_by_role["source"]["wall_seconds"] + phase_by_role["target"]["wall_seconds"],
            "dynamic_transfer_bytes_if_remote": state["package_bytes"],
            "dynamic_transfer_type": "state_package",
            "queue_wait_seconds": None,
            "queue_wait_status": "unavailable_single_process_technical_run_no_rsu_queue_model",
        },
    }
    sensitivity = []
    for link_mbps in plan["network_sensitivity"]["link_mbps"]:
        for fixed in plan["network_sensitivity"]["fixed_one_way_seconds"]:
            restart_network = image_bytes * 8.0 / (float(link_mbps) * 1_000_000.0) + float(fixed)
            recovery_network = state["package_bytes"] * 8.0 / (float(link_mbps) * 1_000_000.0) + float(fixed)
            sensitivity.append({
                "link_mbps_assumption": float(link_mbps),
                "fixed_one_way_seconds_assumption": float(fixed),
                "restart_input_network_seconds": restart_network,
                "recovery_state_network_seconds": recovery_network,
                "recovery_minus_restart_network_seconds": recovery_network - restart_network,
                "measured_network": False,
            })
    return {
        "arms": arms,
        "byte_classes": {
            "dynamic_state_payload_bytes": state["payload_bytes"],
            "dynamic_state_file_bytes": state["state_file_bytes"],
            "dynamic_state_manifest_bytes": state["manifest_bytes"],
            "dynamic_state_package_bytes": state["package_bytes"],
            "intermediate_n0_record_bytes": n0_intermediate_bytes,
            "original_input_image_bytes": image_bytes,
            "model_weights_and_directories": static,
            "legacy_v1_technical_package_bytes_not_consumed_by_target": legacy_state["package_bytes"],
        },
        "target_prestaged_model": {
            "measured_local_path": True,
            "actual_network_transfer_bytes": 0,
            "note": "all static resources were already local; process load time is measured but is not network time",
        },
        "target_missing_model": {
            "measured_path": False,
            "base_plus_adapter_directory_bytes": static["base_directory_bytes"] + static["adapter_directory_bytes"],
            "applies_equally_to_restart_and_recovery": True,
            "network_time_formula": "(base_directory_bytes + adapter_directory_bytes + dynamic_bytes) * 8 / link_bits_per_second + fixed_one_way_seconds",
        },
        "network_sensitivity": sensitivity,
        "break_even": {
            "dynamic_network_byte_condition_for_recovery": f"state_package_bytes < input_image_bytes ({state['package_bytes']} < {image_bytes})",
            "full_time_condition": "serialize + save + state_transfer + restore + input_rebuild + suffix_execution < input_transfer + prefix_reexecution + suffix_execution",
            "network_is_measured": False,
        },
    }


def execute_supervisor(plan_path: Path, run_root: Path, expected_commit: str) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run root: {run_root}")
    actual_commit = current_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(f"execution commit mismatch: expected {expected_commit}, got {actual_commit}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")
    plan = read_json(plan_path)
    run_root.mkdir(parents=True)
    started_wall, started = time.time(), time.monotonic()
    negative = run_negative_checks(plan)
    write_json(run_root / "preflight_negative_checks.json", negative)
    phases = []
    terminal: dict[str, Any]
    try:
        if negative["status"] != "PASS":
            raise RuntimeError("preflight negative checks failed")
        script = Path(__file__).resolve()
        for role in ("continuous", "source", "restart", "target"):
            phase = run_child(script, role, plan_path, run_root, int(plan["budget"]["process_timeout_seconds"]))
            phases.append(phase)
            write_json(run_root / f"{role}_process.json", phase)
            if phase["returncode"] != 0 or phase["timed_out"] or phase["receipt_status"] != "PASS":
                raise RuntimeError(f"{role} failed; no retry: {phase}")
            if role == "source":
                missing_model_negative = import_action4_state(
                    package_dir=run_root / "production_action4_state_package",
                    expected_workflow=production_action4_workflow(),
                    expected_identity=state_identity(plan),
                    expected_input_identity=production_action4_input_identity(plan),
                    target_model_ready=False,
                    target_rsu_id="technical_target_process",
                )
                write_json(
                    run_root / "production_action4_missing_model_negative.json",
                    missing_model_negative,
                )
        continuous = read_json(run_root / "continuous_receipt.json")
        source = read_json(run_root / "source_receipt.json")
        restart = read_json(run_root / "restart_receipt.json")
        target = read_json(run_root / "target_receipt.json")
        comparison = compare_receipts(continuous, source, restart, target)
        generate_attempted = sum(item["generate_attempted"] for item in (continuous, source, restart, target))
        generate_completed = sum(item["generate_completed"] for item in (continuous, source, restart, target))
        wall = time.monotonic() - started
        budget_pass = (
            generate_attempted == int(plan["budget"]["planned_generate_calls"])
            and generate_completed == int(plan["budget"]["planned_generate_calls"])
            and generate_attempted <= int(plan["budget"]["hard_generate_ceiling"])
            and wall <= float(plan["budget"]["scientific_total_timeout_seconds"])
        )
        pids = {item["pid"] for item in (continuous, source, restart, target)}
        ordered = all(phases[index]["ended_at_unix"] <= phases[index + 1]["started_at_unix"] for index in range(3))
        costs = cost_decomposition(plan, continuous, source, restart, target, phases)
        terminal = {
            "status": "PASS" if comparison["status"] == "PASS" and budget_pass and len(pids) == 4 and ordered else "FAIL",
            "classification": plan["classification"],
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": wall,
            "automatic_retry": False,
            "phases": phases,
            "process_ids_distinct": len(pids) == 4,
            "process_ordering_pass": ordered,
            "generate_attempted": generate_attempted,
            "generate_completed": generate_completed,
            "budget_compliance": "pass" if budget_pass else "fail",
            "comparisons": comparison,
            "cost_decomposition": costs,
            "negative_checks": negative,
            "production_action4_missing_model_negative": missing_model_negative,
            "task_correctness": "unavailable",
            "task_correctness_reason": "technical input has no applicable ALPR/Helmet ground-truth label",
            "production_action_4_state_export_import_called": True,
            "claims": {
                "workflow_suffix_recovery_fidelity": comparison["status"].lower(),
                "restart_comparison": comparison["status"].lower(),
                "production_action_4_real_migration": "validated technical state transfer through shared production action-4 coordinator",
                "wireless_transfer": "formula_only_not_measured",
                "algorithm_comparison": "not_evaluated",
            },
        }
    except BaseException as error:
        terminal = {
            "status": "FAIL",
            "classification": plan.get("classification"),
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "phases": phases,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "negative_checks": negative,
        }
    write_json(run_root / "terminal_receipt.json", terminal)
    files = inventory_tree(run_root, exclude={"integrity_manifest.json"})
    write_json(run_root / "integrity_manifest.json", {
        "schema_version": "ppo_mec.artifact_integrity.v1",
        "run_root": str(run_root),
        "file_count": len(files),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "files": files,
    })
    return 0 if terminal["status"] == "PASS" else 1


def role_main(role: str, plan_path: Path, run_root: Path) -> int:
    import torch

    torch.set_num_threads(1)
    try:
        if role == "continuous":
            receipt = execute_continuous(plan_path, run_root)
        elif role == "source":
            receipt = execute_source(plan_path, run_root)
        elif role == "restart":
            receipt = execute_restart(plan_path, run_root)
        elif role == "target":
            receipt = execute_target(plan_path, run_root)
        else:
            raise ValueError(f"unsupported role: {role}")
    except BaseException as error:
        receipt = {
            "status": "FAIL",
            "role": role,
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "task_correctness": "unavailable",
        }
    write_json(run_root / f"{role}_receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["supervisor", "continuous", "source", "restart", "target"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    run_root = args.run_root.resolve()
    if args.role == "supervisor":
        if not args.expected_commit:
            raise SystemExit("--expected-commit is required for supervisor")
        return execute_supervisor(plan_path, run_root, args.expected_commit)
    return role_main(args.role, plan_path, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
