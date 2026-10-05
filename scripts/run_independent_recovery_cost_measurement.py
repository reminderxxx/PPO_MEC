"""Run the frozen independent restart/recovery cost check with at most 24 calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
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
    execute_source,
    generate,
    import_action4_state,
    inventory_tree,
    load_runtime,
    prepare_n0,
    prepare_n1,
    production_action4_input_identity,
    production_action4_workflow,
    read_json,
    run_negative_checks,
    sha256_file,
    state_identity,
    verify_static_resources,
    write_json,
)


ARM_NAMES = ("restart", "recovery")


def _condition_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["condition_id"]): item for item in plan["conditions"]}


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    conditions = _condition_map(plan)
    if set(conditions) != {
        "target_model_prepared",
        "target_model_requires_local_preparation",
    }:
        raise ValueError("the frozen plan must contain exactly the two declared conditions")
    if plan["scope"]["training_forbidden"] is not True:
        raise ValueError("training must remain forbidden")
    if plan["scope"]["downloads_forbidden"] is not True:
        raise ValueError("downloads must remain forbidden")
    if plan["scope"]["old_holdout_read_forbidden"] is not True:
        raise ValueError("old holdout access must remain forbidden")
    order = list(plan["execution_order"])
    if len(order) != 6:
        raise ValueError("the frozen plan must contain six condition-repeat entries")
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(order, start=1):
        if int(item["sequence"]) != index:
            raise ValueError("execution sequence must be contiguous and frozen")
        condition_id = str(item["condition_id"])
        if condition_id not in conditions:
            raise ValueError(f"unknown condition: {condition_id}")
        key = (condition_id, str(item["repeat_id"]))
        if key in seen:
            raise ValueError(f"duplicate condition-repeat: {key}")
        seen.add(key)
        if sorted(item["arm_order"]) != sorted(ARM_NAMES):
            raise ValueError(f"invalid arm order: {item['arm_order']}")
    per_repeat = plan["budget"]["generate_calls_per_repeat"]
    expected_per_repeat = sum(int(per_repeat[key]) for key in ("source", *ARM_NAMES))
    planned = len(order) * expected_per_repeat
    if planned != int(plan["budget"]["planned_generate_calls"]):
        raise ValueError("execution order and planned generate calls disagree")
    if planned > int(plan["budget"]["hard_generate_ceiling"]):
        raise ValueError("planned generate calls exceed the hard ceiling")
    if planned != 24:
        raise ValueError("this protocol is frozen to exactly 24 generate calls")
    for condition in conditions.values():
        predictions = condition["predictions"]
        expected = min(
            ARM_NAMES,
            key=lambda arm: float(predictions[f"{arm}_scored_completion_seconds"]),
        )
        if expected != predictions["predicted_cheaper_action"]:
            raise ValueError(f"inconsistent frozen action prediction for {condition['condition_id']}")
    return {"planned_generate_calls": planned, "condition_repeat_count": len(order)}


def _model_load_seconds(receipt: dict[str, Any]) -> float:
    return sum(float(value) for value in receipt["model_load_timings"].values())


def _execute_arm(
    role: str,
    condition: dict[str, Any],
    base_plan_path: Path,
    repeat_root: Path,
) -> dict[str, Any]:
    if role not in ARM_NAMES:
        raise ValueError(f"unsupported arm role: {role}")
    child_started_wall = time.time()
    child_started = time.monotonic()
    base_plan = read_json(base_plan_path)
    checked = verify_static_resources(base_plan, include_source_image=role == "restart")

    processor = model = timings = status = None
    setup_model_load_wall = 0.0
    if not bool(condition["model_load_in_action_window"]):
        mark = time.monotonic()
        processor, model, timings, status = load_runtime(base_plan)
        setup_model_load_wall = time.monotonic() - mark

    action_started = time.monotonic()
    if bool(condition["model_load_in_action_window"]):
        processor, model, timings, status = load_runtime(base_plan)
    assert processor is not None and model is not None
    assert timings is not None and status is not None

    if role == "restart":
        n0_inputs, n0_input = prepare_n0(base_plan, processor)
        n0 = generate(model, processor, n0_inputs, base_plan["generation"]["n0"])
        n1_inputs, n1_input = prepare_n1(base_plan, processor, n0["decoded_text"])
        n1 = generate(model, processor, n1_inputs, base_plan["generation"]["n1"])
        restore = None
        restored_n0 = None
        node_counts = {"n0": 1, "n1": 1}
        generate_count = 2
        dynamic_bytes = int(base_plan["source_input_provenance"]["bytes"])
        target_access = {
            "source_image_access_count": 1,
            "saved_state_package_access": False,
            "source_process_memory_access": False,
        }
    else:
        production_import = import_action4_state(
            package_dir=repeat_root / "production_action4_state_package",
            expected_workflow=production_action4_workflow(),
            expected_identity=state_identity(base_plan),
            expected_input_identity=production_action4_input_identity(base_plan),
            target_model_ready=True,
            target_rsu_id="technical_target_process",
        )
        if not production_import.get("migration_success", False):
            raise RuntimeError(f"production action 4 import failed: {production_import}")
        rebuild_started = time.monotonic()
        restored_n0 = production_import["node_outputs"]["n0"]
        n1_inputs, n1_input = prepare_n1(base_plan, processor, restored_n0["raw_text"])
        rebuild_seconds = time.monotonic() - rebuild_started
        n1 = generate(model, processor, n1_inputs, base_plan["generation"]["n1"])
        n0_input = None
        n0 = None
        restore = {
            "state_restore_validation_seconds": float(
                production_import["restore_validation_seconds"]
            ),
            "input_reconstruction_seconds": rebuild_seconds,
            "production_action4_state_import": production_import,
        }
        node_counts = {"n0": 0, "n1": 1}
        generate_count = 1
        dynamic_bytes = int(production_import["package_bytes"])
        target_access = {
            "source_image_access_count": 0,
            "saved_state_package_access": True,
            "source_process_memory_access": False,
        }
    action_wall = time.monotonic() - action_started
    return {
        "status": "PASS",
        "role": role,
        "condition_id": condition["condition_id"],
        "pid": os.getpid(),
        "started_at_unix": child_started_wall,
        "child_process_wall_seconds": time.monotonic() - child_started,
        "action_wall_seconds": action_wall,
        "model_load_in_action_window": bool(condition["model_load_in_action_window"]),
        "setup_model_load_wall_seconds": setup_model_load_wall,
        "model_load_timings": timings,
        "model_load_seconds": _model_load_seconds({"model_load_timings": timings}),
        "static_resource_hashes": checked,
        "identity": state_identity(base_plan),
        "adapter_status": status,
        "node_call_counts": node_counts,
        "generate_attempted": generate_count,
        "generate_completed": generate_count,
        "dynamic_transfer_bytes_if_remote": dynamic_bytes,
        "n0_input": n0_input,
        "n0_output": n0,
        "n1_input": n1_input,
        "n1_output": n1,
        "restored_n0_output": restored_n0,
        "restore": restore,
        "target_access": target_access,
        "task_correctness": "unavailable",
        "automatic_retry": False,
    }


def _child_main(
    role: str,
    condition_id: str,
    plan_path: Path,
    repeat_root: Path,
) -> int:
    import torch

    torch.set_num_threads(1)
    receipt_path = repeat_root / f"{role}_receipt.json"
    try:
        plan = read_json(plan_path)
        validate_plan(plan)
        conditions = _condition_map(plan)
        base_plan_path = (ROOT / plan["base_plan_path"]).resolve()
        if role == "source":
            receipt = execute_source(base_plan_path, repeat_root)
            receipt["condition_id"] = condition_id
        else:
            receipt = _execute_arm(
                role,
                conditions[condition_id],
                base_plan_path,
                repeat_root,
            )
    except BaseException as error:
        receipt = {
            "status": "FAIL",
            "role": role,
            "condition_id": condition_id,
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
    write_json(receipt_path, receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "PASS" else 1


def _run_child(
    role: str,
    condition_id: str,
    plan_path: Path,
    repeat_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--condition-id",
        condition_id,
        "--plan",
        str(plan_path),
        "--run-root",
        str(repeat_root),
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started_wall = time.time()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        timed_out = True
    receipt_path = repeat_root / f"{role}_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"status": "MISSING"}
    return {
        "role": role,
        "condition_id": condition_id,
        "command": command,
        "pid": receipt.get("pid"),
        "started_at_unix": started_wall,
        "ended_at_unix": time.time(),
        "wall_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "receipt_status": receipt.get("status"),
    }


def _compare_outputs(
    source: dict[str, Any],
    restart: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_restart_n0_text_equal": source["n0_output"]["decoded_text"]
        == restart["n0_output"]["decoded_text"],
        "source_restart_n0_tokens_equal": source["n0_output"]["token_ids"]
        == restart["n0_output"]["token_ids"],
        "recovery_consumed_source_n0": recovery["restored_n0_output"]
        == {
            "raw_text": source["n0_output"]["decoded_text"],
            "token_ids": source["n0_output"]["token_ids"],
        },
        "suffix_prompt_equal": restart["n1_input"]["prompt"]
        == recovery["n1_input"]["prompt"],
        "suffix_rendered_prompt_equal": restart["n1_input"]["rendered_prompt"]
        == recovery["n1_input"]["rendered_prompt"],
        "suffix_input_ids_equal": restart["n1_input"]["input_ids"]
        == recovery["n1_input"]["input_ids"],
        "suffix_input_hash_equal": restart["n1_input"]["content_sha256"]
        == recovery["n1_input"]["content_sha256"],
        "suffix_output_tokens_equal": restart["n1_output"]["token_ids"]
        == recovery["n1_output"]["token_ids"],
        "identity_equal": source["identity"] == restart["identity"] == recovery["identity"],
        "restart_calls_exact": restart["node_call_counts"] == {"n0": 1, "n1": 1},
        "recovery_calls_exact": recovery["node_call_counts"] == {"n0": 0, "n1": 1},
        "recovery_source_image_access_zero": recovery["target_access"][
            "source_image_access_count"
        ]
        == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _network_seconds(byte_count: int, frozen: dict[str, Any]) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (
        float(frozen["effective_link_mbps_assumption"]) * 1_000_000.0
    ) + float(frozen["positive_dynamic_transfer_fixed_latency_seconds_assumption"])


def _score_repeat(
    plan: dict[str, Any],
    condition: dict[str, Any],
    source: dict[str, Any],
    restart: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    frozen = plan["frozen_cost_estimates"]
    predictions = condition["predictions"]
    measured: dict[str, dict[str, Any]] = {}
    for arm, receipt in (("restart", restart), ("recovery", recovery)):
        network = _network_seconds(int(receipt["dynamic_transfer_bytes_if_remote"]), frozen)
        actual_scored = float(receipt["action_wall_seconds"]) + network
        predicted = float(predictions[f"{arm}_scored_completion_seconds"])
        signed = predicted - actual_scored
        measured[arm] = {
            "predicted_local_action_wall_seconds": float(
                predictions[f"{arm}_local_action_wall_seconds"]
            ),
            "measured_local_action_wall_seconds": float(receipt["action_wall_seconds"]),
            "dynamic_transfer_bytes": int(receipt["dynamic_transfer_bytes_if_remote"]),
            "simulated_dynamic_network_seconds": network,
            "network_measured": False,
            "predicted_scored_completion_seconds": predicted,
            "measured_scored_completion_seconds": actual_scored,
            "signed_prediction_error_seconds": signed,
            "absolute_prediction_error_seconds": abs(signed),
            "relative_absolute_prediction_error": abs(signed) / actual_scored,
            "model_load_seconds": float(receipt["model_load_seconds"]),
            "model_load_in_action_window": bool(receipt["model_load_in_action_window"]),
            "n0_inference_seconds": (
                float(receipt["n0_output"]["inference_seconds"])
                if receipt["n0_output"] is not None
                else 0.0
            ),
            "n1_inference_seconds": float(receipt["n1_output"]["inference_seconds"]),
        }
    actual_cheaper = min(
        ARM_NAMES,
        key=lambda arm: measured[arm]["measured_scored_completion_seconds"],
    )
    predicted_action = str(predictions["predicted_cheaper_action"])
    wrong_choice_cost = (
        measured[predicted_action]["measured_scored_completion_seconds"]
        - measured[actual_cheaper]["measured_scored_completion_seconds"]
    )
    state_export = source["production_action4_state_export"]
    restore = recovery["restore"]
    state_component = (
        float(state_export["serialize_and_save_seconds"])
        + float(restore["state_restore_validation_seconds"])
        + float(restore["input_reconstruction_seconds"])
    )
    return {
        "predicted_cheaper_action": predicted_action,
        "measured_cheaper_action": actual_cheaper,
        "decision_match": predicted_action == actual_cheaper,
        "wrong_choice_cost_seconds": wrong_choice_cost,
        "arms": measured,
        "components": {
            "source_state_serialize_and_save_seconds": float(
                state_export["serialize_and_save_seconds"]
            ),
            "recovery_restore_validation_seconds": float(
                restore["state_restore_validation_seconds"]
            ),
            "recovery_input_reconstruction_seconds": float(
                restore["input_reconstruction_seconds"]
            ),
            "state_serialize_restore_rebuild_seconds": state_component,
            "state_package_bytes": int(state_export["package_bytes"]),
            "rerun_input_bytes": int(plan["frozen_cost_estimates"]["rerun_input_bytes"]),
            "queue_wait_seconds": None,
            "queue_wait_status": "unavailable_single_host_no_rsu_queue_model",
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition_id in sorted({row["condition_id"] for row in rows}):
        selected = [row for row in rows if row["condition_id"] == condition_id]
        arms: dict[str, Any] = {}
        for arm in ARM_NAMES:
            raw = [row["scoring"]["arms"][arm] for row in selected]
            arms[arm] = {
                "measured_local_action_wall_seconds_raw": [
                    item["measured_local_action_wall_seconds"] for item in raw
                ],
                "measured_scored_completion_seconds_raw": [
                    item["measured_scored_completion_seconds"] for item in raw
                ],
                "signed_prediction_error_seconds_raw": [
                    item["signed_prediction_error_seconds"] for item in raw
                ],
                "absolute_prediction_error_seconds_raw": [
                    item["absolute_prediction_error_seconds"] for item in raw
                ],
                "relative_absolute_prediction_error_raw": [
                    item["relative_absolute_prediction_error"] for item in raw
                ],
                "model_load_seconds_raw": [item["model_load_seconds"] for item in raw],
                "n0_inference_seconds_raw": [item["n0_inference_seconds"] for item in raw],
                "n1_inference_seconds_raw": [item["n1_inference_seconds"] for item in raw],
                "median_measured_local_action_wall_seconds": statistics.median(
                    item["measured_local_action_wall_seconds"] for item in raw
                ),
                "median_absolute_prediction_error_seconds": statistics.median(
                    item["absolute_prediction_error_seconds"] for item in raw
                ),
            }
        by_condition[condition_id] = {
            "repeat_count": len(selected),
            "decision_matches": sum(bool(row["scoring"]["decision_match"]) for row in selected),
            "wrong_choice_cost_seconds_raw": [
                row["scoring"]["wrong_choice_cost_seconds"] for row in selected
            ],
            "output_fidelity_passes": sum(
                row["comparison"]["status"] == "PASS" for row in selected
            ),
            "state_serialize_restore_rebuild_seconds_raw": [
                row["scoring"]["components"]["state_serialize_restore_rebuild_seconds"]
                for row in selected
            ],
            "state_package_bytes_raw": [
                row["scoring"]["components"]["state_package_bytes"] for row in selected
            ],
            "arms": arms,
        }
    total_matches = sum(bool(row["scoring"]["decision_match"]) for row in rows)
    observed_actions = sorted({row["scoring"]["measured_cheaper_action"] for row in rows})
    return {
        "repeat_count": len(rows),
        "decision_matches": total_matches,
        "decision_match_rate": total_matches / len(rows),
        "observed_cheaper_actions": observed_actions,
        "decision_boundary_observed": len(observed_actions) > 1,
        "decision_boundary_note": (
            "both action sides observed"
            if len(observed_actions) > 1
            else "all checked conditions fell on one action side; no decision boundary was tested"
        ),
        "by_condition": by_condition,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sequence",
        "condition_id",
        "repeat_id",
        "predicted_action",
        "measured_action",
        "decision_match",
        "wrong_choice_cost_seconds",
        "restart_predicted_seconds",
        "restart_measured_seconds",
        "restart_signed_error_seconds",
        "recovery_predicted_seconds",
        "recovery_measured_seconds",
        "recovery_signed_error_seconds",
        "output_fidelity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            scoring = row["scoring"]
            writer.writerow(
                {
                    "sequence": row["sequence"],
                    "condition_id": row["condition_id"],
                    "repeat_id": row["repeat_id"],
                    "predicted_action": scoring["predicted_cheaper_action"],
                    "measured_action": scoring["measured_cheaper_action"],
                    "decision_match": scoring["decision_match"],
                    "wrong_choice_cost_seconds": scoring["wrong_choice_cost_seconds"],
                    "restart_predicted_seconds": scoring["arms"]["restart"][
                        "predicted_scored_completion_seconds"
                    ],
                    "restart_measured_seconds": scoring["arms"]["restart"][
                        "measured_scored_completion_seconds"
                    ],
                    "restart_signed_error_seconds": scoring["arms"]["restart"][
                        "signed_prediction_error_seconds"
                    ],
                    "recovery_predicted_seconds": scoring["arms"]["recovery"][
                        "predicted_scored_completion_seconds"
                    ],
                    "recovery_measured_seconds": scoring["arms"]["recovery"][
                        "measured_scored_completion_seconds"
                    ],
                    "recovery_signed_error_seconds": scoring["arms"]["recovery"][
                        "signed_prediction_error_seconds"
                    ],
                    "output_fidelity": row["comparison"]["status"],
                }
            )


def execute_supervisor(plan_path: Path, run_root: Path, expected_commit: str) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run root: {run_root}")
    actual_commit = current_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")
    plan = read_json(plan_path)
    validation = validate_plan(plan)
    base_plan_path = (ROOT / plan["base_plan_path"]).resolve()
    base_plan = read_json(base_plan_path)
    run_root.mkdir(parents=True)
    write_json(run_root / "frozen_plan.json", plan)
    negative = run_negative_checks(base_plan)
    write_json(run_root / "preflight_negative_checks.json", negative)
    started_wall = time.time()
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    terminal: dict[str, Any]
    try:
        if negative["status"] != "PASS":
            raise RuntimeError("preflight negative checks failed")
        conditions = _condition_map(plan)
        timeout = int(plan["budget"]["process_timeout_seconds"])
        for item in plan["execution_order"]:
            condition_id = str(item["condition_id"])
            repeat_id = str(item["repeat_id"])
            repeat_root = run_root / condition_id / repeat_id
            repeat_root.mkdir(parents=True)
            roles = ["source", *item["arm_order"]]
            local_processes = []
            for role in roles:
                process = _run_child(
                    role,
                    condition_id,
                    plan_path,
                    repeat_root,
                    timeout,
                )
                processes.append(process)
                local_processes.append(process)
                write_json(repeat_root / f"{role}_process.json", process)
                if (
                    process["returncode"] != 0
                    or process["timed_out"]
                    or process["receipt_status"] != "PASS"
                ):
                    raise RuntimeError(
                        f"{condition_id}/{repeat_id}/{role} failed; no retry: {process}"
                    )
            source = read_json(repeat_root / "source_receipt.json")
            restart = read_json(repeat_root / "restart_receipt.json")
            recovery = read_json(repeat_root / "recovery_receipt.json")
            comparison = _compare_outputs(source, restart, recovery)
            if comparison["status"] != "PASS":
                raise RuntimeError(f"{condition_id}/{repeat_id} output fidelity failed")
            scoring = _score_repeat(
                plan,
                conditions[condition_id],
                source,
                restart,
                recovery,
            )
            row = {
                "sequence": int(item["sequence"]),
                "condition_id": condition_id,
                "repeat_id": repeat_id,
                "arm_order": list(item["arm_order"]),
                "processes": local_processes,
                "process_ids_distinct": len(
                    {int(process["pid"]) for process in local_processes}
                )
                == 3,
                "generate_attempted": sum(
                    int(receipt["generate_attempted"])
                    for receipt in (source, restart, recovery)
                ),
                "generate_completed": sum(
                    int(receipt["generate_completed"])
                    for receipt in (source, restart, recovery)
                ),
                "comparison": comparison,
                "scoring": scoring,
                "runtime_state": plan["runtime_state"],
            }
            write_json(repeat_root / "repeat_receipt.json", row)
            completed.append(row)

        total_attempted = sum(int(item["generate_attempted"]) for item in completed)
        total_completed = sum(int(item["generate_completed"]) for item in completed)
        scientific_wall = time.monotonic() - started
        budget_pass = (
            total_attempted == int(plan["budget"]["planned_generate_calls"])
            and total_completed == int(plan["budget"]["planned_generate_calls"])
            and total_attempted <= int(plan["budget"]["hard_generate_ceiling"])
            and scientific_wall <= float(plan["budget"]["scientific_total_timeout_seconds"])
        )
        aggregate = _aggregate(completed)
        write_json(run_root / "all_measurements.json", {"rows": completed})
        _write_csv(run_root / "all_measurements.csv", completed)
        write_json(run_root / "aggregate_summary.json", aggregate)
        all_pids = [int(process["pid"]) for process in processes]
        terminal = {
            "status": "PASS"
            if budget_pass
            and all(item["comparison"]["status"] == "PASS" for item in completed)
            and len(all_pids) == len(set(all_pids))
            else "FAIL",
            "classification": "independent fixed-implementation descriptive cost check",
            "parsed_baseline_commit": plan["parsed_baseline_commit"],
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "frozen_plan_copy_sha256": sha256_file(run_root / "frozen_plan.json"),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": scientific_wall,
            "automatic_retry": False,
            "generate_attempted": total_attempted,
            "generate_completed": total_completed,
            "budget_compliance": "pass" if budget_pass else "fail",
            "all_process_ids_distinct": len(all_pids) == len(set(all_pids)),
            "condition_repeat_count": len(completed),
            "plan_validation": validation,
            "aggregate": aggregate,
            "same_host_not_real_rsu_network_measurement": True,
            "network_time_status": "frozen_assumption_only",
            "queue_time_status": "unavailable",
            "task_correctness": "unavailable",
            "uncovered_conditions": plan["uncovered_conditions"],
        }
    except BaseException as error:
        terminal = {
            "status": "FAIL",
            "parsed_baseline_commit": plan.get("parsed_baseline_commit"),
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "completed_condition_repeats": len(completed),
            "completed_rows": completed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(run_root / "terminal_receipt.json", terminal)
    files = inventory_tree(run_root, exclude={"integrity_manifest.json"})
    write_json(
        run_root / "integrity_manifest.json",
        {
            "schema_version": "ppo_mec.artifact_integrity.v1",
            "run_root": str(run_root),
            "file_count": len(files),
            "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
            "files": files,
        },
    )
    return 0 if terminal["status"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=["supervisor", "source", *ARM_NAMES], required=True
    )
    parser.add_argument("--condition-id")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.resolve()
    run_root = args.run_root.resolve()
    if args.role == "supervisor":
        if not args.expected_commit:
            raise SystemExit("--expected-commit is required for supervisor")
        return execute_supervisor(plan_path, run_root, args.expected_commit)
    if not args.condition_id:
        raise SystemExit("--condition-id is required for child roles")
    return _child_main(args.role, args.condition_id, plan_path, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
