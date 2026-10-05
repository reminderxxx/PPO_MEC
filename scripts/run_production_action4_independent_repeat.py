"""Run fixed-implementation paired restart/recovery repetitions with frozen arm order."""

from __future__ import annotations

import argparse
import json
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
    inventory_tree,
    read_json,
    run_child,
    run_negative_checks,
    sha256_file,
    write_json,
)


def _sum_load(receipt: dict[str, Any]) -> float:
    return sum(float(value) for value in receipt["model_load_timings"].values())


def _compare_repeat(
    source: dict[str, Any], restart: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "source_restart_n0_text_equal": source["n0_output"]["decoded_text"]
        == restart["n0_output"]["decoded_text"],
        "source_restart_n0_tokens_equal": source["n0_output"]["token_ids"]
        == restart["n0_output"]["token_ids"],
        "target_consumed_source_n0": target["restored_n0_output"]
        == {
            "raw_text": source["n0_output"]["decoded_text"],
            "token_ids": source["n0_output"]["token_ids"],
        },
        "restart_target_n1_prompt_equal": restart["n1_input"]["prompt"]
        == target["n1_input"]["prompt"],
        "restart_target_n1_rendered_prompt_equal": restart["n1_input"]["rendered_prompt"]
        == target["n1_input"]["rendered_prompt"],
        "restart_target_n1_input_ids_equal": restart["n1_input"]["input_ids"]
        == target["n1_input"]["input_ids"],
        "restart_target_n1_hash_equal": restart["n1_input"]["content_sha256"]
        == target["n1_input"]["content_sha256"],
        "restart_target_n1_tokens_equal": restart["n1_output"]["token_ids"]
        == target["n1_output"]["token_ids"],
        "identity_equal": source["identity"] == restart["identity"] == target["identity"],
        "restart_calls_exact": restart["node_call_counts"] == {"n0": 1, "n1": 1},
        "recovery_calls_exact": target["node_call_counts"] == {"n0": 0, "n1": 1},
        "recovery_source_image_access_zero": target["target_access"]["source_image_access_count"]
        == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _repeat_costs(
    source: dict[str, Any],
    restart: dict[str, Any],
    target: dict[str, Any],
    phase_by_role: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    state = source["production_action4_state_export"]
    restart_wall = float(phase_by_role["source"]["wall_seconds"]) + float(
        phase_by_role["restart"]["wall_seconds"]
    )
    recovery_wall = float(phase_by_role["source"]["wall_seconds"]) + float(
        phase_by_role["target"]["wall_seconds"]
    )
    return {
        "source": {
            "model_load_seconds": _sum_load(source),
            "n0_inference_seconds": source["n0_output"]["inference_seconds"],
            "serialize_and_save_seconds": state["serialize_and_save_seconds"],
            "child_wall_seconds": phase_by_role["source"]["wall_seconds"],
        },
        "restart": {
            "model_load_seconds": _sum_load(restart),
            "n0_inference_seconds": restart["n0_output"]["inference_seconds"],
            "n1_inference_seconds": restart["n1_output"]["inference_seconds"],
            "child_wall_seconds": phase_by_role["restart"]["wall_seconds"],
            "interrupted_path_wall_seconds": restart_wall,
            "dynamic_transfer_bytes_if_remote": 192757,
        },
        "recovery": {
            "model_load_seconds": _sum_load(target),
            "restore_validation_seconds": target["state_restore_validation_seconds"],
            "input_reconstruction_seconds": target["input_reconstruction_seconds"],
            "n1_inference_seconds": target["n1_output"]["inference_seconds"],
            "child_wall_seconds": phase_by_role["target"]["wall_seconds"],
            "interrupted_path_wall_seconds": recovery_wall,
            "dynamic_transfer_bytes_if_remote": state["package_bytes"],
        },
        "recovery_minus_restart_path_wall_seconds": recovery_wall - restart_wall,
        "queue_wait_seconds": None,
        "queue_wait_status": "unavailable_single_host_no_rsu_queue_model",
    }


def execute(plan_path: Path, run_root: Path, expected_commit: str) -> int:
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
    base_plan_path = (ROOT / plan["base_plan_path"]).resolve()
    base_plan = read_json(base_plan_path)
    budget = plan["budget"]
    repetitions = plan["repetitions"]
    expected_calls = len(repetitions) * 4
    if expected_calls != int(budget["planned_generate_calls"]):
        raise RuntimeError("repeat plan and generate budget disagree")
    if int(budget["planned_generate_calls"]) > int(budget["hard_generate_ceiling"]):
        raise RuntimeError("planned generate calls exceed hard ceiling")

    run_root.mkdir(parents=True)
    started_wall = time.time()
    started = time.monotonic()
    negative = run_negative_checks(base_plan)
    write_json(run_root / "preflight_negative_checks.json", negative)
    terminal: dict[str, Any]
    completed_repetitions = []
    all_pids: list[int] = []
    try:
        if negative["status"] != "PASS":
            raise RuntimeError("preflight negative checks failed")
        role_script = ROOT / "scripts/run_two_node_workflow_reexecution_comparison.py"
        for repetition in repetitions:
            repeat_root = run_root / repetition["repeat_id"]
            repeat_root.mkdir()
            phases = []
            for role in ["source", *repetition["arm_order"]]:
                phase = run_child(
                    role_script,
                    role,
                    base_plan_path,
                    repeat_root,
                    int(budget["process_timeout_seconds"]),
                )
                phases.append(phase)
                write_json(repeat_root / f"{role}_process.json", phase)
                if (
                    phase["returncode"] != 0
                    or phase["timed_out"]
                    or phase["receipt_status"] != "PASS"
                ):
                    raise RuntimeError(
                        f"{repetition['repeat_id']} {role} failed; no retry: {phase}"
                    )
                all_pids.append(int(phase["pid"]))
            source = read_json(repeat_root / "source_receipt.json")
            restart = read_json(repeat_root / "restart_receipt.json")
            target = read_json(repeat_root / "target_receipt.json")
            comparison = _compare_repeat(source, restart, target)
            if comparison["status"] != "PASS":
                raise RuntimeError(f"{repetition['repeat_id']} output comparison failed")
            attempted = sum(
                item["generate_attempted"] for item in (source, restart, target)
            )
            completed = sum(
                item["generate_completed"] for item in (source, restart, target)
            )
            phase_by_role = {item["role"]: item for item in phases}
            row = {
                "repeat_id": repetition["repeat_id"],
                "arm_order": repetition["arm_order"],
                "phases": phases,
                "comparison": comparison,
                "generate_attempted": attempted,
                "generate_completed": completed,
                "process_ids_distinct_within_repeat": len({item["pid"] for item in phases})
                == len(phases),
                "runtime_state": plan["runtime_state"],
                "costs": _repeat_costs(source, restart, target, phase_by_role),
            }
            write_json(repeat_root / "repeat_receipt.json", row)
            completed_repetitions.append(row)

        total_attempted = sum(item["generate_attempted"] for item in completed_repetitions)
        total_completed = sum(item["generate_completed"] for item in completed_repetitions)
        wall = time.monotonic() - started
        budget_pass = (
            total_attempted == int(budget["planned_generate_calls"])
            and total_completed == int(budget["planned_generate_calls"])
            and total_attempted <= int(budget["hard_generate_ceiling"])
            and wall <= float(budget["scientific_total_timeout_seconds"])
        )
        terminal = {
            "status": "PASS"
            if budget_pass and len(set(all_pids)) == len(all_pids)
            else "FAIL",
            "classification": "fixed-implementation paired local-host repeat",
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "base_plan_sha256": sha256_file(base_plan_path),
            "role_script_sha256": sha256_file(role_script),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": wall,
            "automatic_retry": False,
            "generate_attempted": total_attempted,
            "generate_completed": total_completed,
            "budget_compliance": "pass" if budget_pass else "fail",
            "all_process_ids_distinct": len(set(all_pids)) == len(all_pids),
            "runtime_state": plan["runtime_state"],
            "repetitions": completed_repetitions,
            "same_host_not_real_rsu_network_measurement": True,
            "task_correctness": "unavailable",
        }
    except BaseException as error:
        terminal = {
            "status": "FAIL",
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "completed_repetitions": completed_repetitions,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    return execute(args.plan.resolve(), args.run_root.resolve(), args.expected_commit)


if __name__ == "__main__":
    raise SystemExit(main())
