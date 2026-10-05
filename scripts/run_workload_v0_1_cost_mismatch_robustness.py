"""Run the frozen workload-v0.1 cost-mismatch diagnostic without model calls."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_measurement_calibrated_vec_workload import sha256_file, write_json  # noqa: E402
from src.data.workflow.measurement_calibrated_vec_workload import (  # noqa: E402
    _objective,
    _simulate,
    generate_workloads,
)


def _network_seconds(byte_count: int, mbps: float, fixed_seconds: float) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (mbps * 1_000_000.0) + fixed_seconds


def _estimated_costs(
    robustness: dict[str, Any],
    baseline: dict[str, Any],
    point: dict[str, Any],
) -> dict[str, float]:
    estimate = robustness["decision_time_estimate"]
    model = baseline["calibration"]["model"]
    common = 0.0
    if point["estimated_target_model"] == "missing":
        static_bytes = int(model["base_network_directory_bytes"]["value"]) + int(
            model["adapter_network_directory_bytes"]["value"]
        )
        common = float(estimate["target_model_load_seconds"]) + _network_seconds(
            static_bytes,
            float(estimate["link_mbps"]),
            float(estimate["fixed_one_way_seconds"]),
        )
    restart = common + float(estimate["prefix_recompute_seconds"]) + _network_seconds(
        int(estimate["input_bytes"]),
        float(estimate["link_mbps"]),
        float(estimate["fixed_one_way_seconds"]),
    )
    recover = common + float(estimate["recovery_overhead_seconds"]) + _network_seconds(
        int(estimate["state_bytes"]),
        float(estimate["link_mbps"]),
        float(estimate["fixed_one_way_seconds"]),
    )
    return {
        "estimated_common_model_prepare_seconds": common,
        "estimated_restart_incremental_seconds": restart,
        "estimated_recovery_incremental_seconds": recover,
    }


def _realized_incremental_costs(
    robustness: dict[str, Any],
    baseline: dict[str, Any],
    point: dict[str, Any],
) -> dict[str, float]:
    constants = robustness["execution_constants"]
    model = baseline["calibration"]["model"]
    common = 0.0
    if point["actual_target_model"] == "missing":
        static_bytes = int(model["base_network_directory_bytes"]["value"]) + int(
            model["adapter_network_directory_bytes"]["value"]
        )
        common = float(point["actual_target_model_load_seconds"]) + _network_seconds(
            static_bytes,
            float(point["actual_link_mbps"]),
            float(constants["fixed_one_way_seconds"]),
        )
    restart = common + float(point["actual_prefix_recompute_seconds"]) + _network_seconds(
        int(constants["input_bytes"]),
        float(point["actual_link_mbps"]),
        float(constants["fixed_one_way_seconds"]),
    )
    recover = common + float(point["actual_recovery_overhead_seconds"]) + _network_seconds(
        int(constants["state_bytes"]),
        float(point["actual_link_mbps"]),
        float(constants["fixed_one_way_seconds"]),
    )
    return {
        "realized_common_model_prepare_seconds": common,
        "realized_restart_incremental_seconds": restart,
        "realized_recovery_incremental_seconds": recover,
        "realized_recovery_minus_restart_seconds": recover - restart,
    }


def evaluate_design(
    robustness: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    workloads = generate_workloads(baseline)
    template = next(
        item
        for item in workloads
        if item["design_id"] == robustness["baseline_design_id"]
        and item["seed"] == robustness["baseline_seed"]
    )
    result_rows: list[dict[str, Any]] = []
    for point in robustness["design_points"]:
        workload = deepcopy(template)
        workload["design_id"] = point["design_id"]
        workload["seed"] = int(robustness["baseline_seed"])
        workload["factors"]["design_id"] = point["design_id"]
        workload["factors"]["restore_cost"] = "high"
        workload["factors"]["initial_target_model"] = point["actual_target_model"]
        workload["rsu_resources"]["initial_target_model"] = point["actual_target_model"]
        workload["link"]["mbps"] = float(point["actual_link_mbps"])
        for index, workflow in enumerate(workload["workflows"]):
            workflow["workflow_id"] = f"{point['design_id']}__workflow_{index}"

        execution_config = deepcopy(baseline)
        execution_config["calibration"]["dynamic"]["state_package_bytes"]["value"] = int(
            robustness["execution_constants"]["state_bytes"]
        )
        execution_config["calibration"]["time_seconds"]["prefix_compute"]["value"] = float(
            point["actual_prefix_recompute_seconds"]
        )
        execution_config["calibration"]["time_seconds"]["target_model_load"]["value"] = float(
            point["actual_target_model_load_seconds"]
        )
        execution_config["calibration"]["time_seconds"]["high_restore_stress"]["value"] = float(
            point["actual_recovery_overhead_seconds"]
        )

        count = len(workload["workflows"])
        estimated = _estimated_costs(robustness, baseline, point)
        realized = _realized_incremental_costs(robustness, baseline, point)
        overheads: dict[str, float] = {}

        started = time.perf_counter()
        method_decisions: dict[str, list[str]] = {
            "current_restart_rule": ["restart"] * count
        }
        overheads["current_restart_rule"] = time.perf_counter() - started

        started = time.perf_counter()
        local_choice = (
            "recover"
            if estimated["estimated_recovery_incremental_seconds"]
            < estimated["estimated_restart_incremental_seconds"]
            else "restart"
        )
        method_decisions["mechanism_aware_local_incremental_cost"] = [local_choice] * count
        overheads["mechanism_aware_local_incremental_cost"] = time.perf_counter() - started

        started = time.perf_counter()
        candidates = [
            _simulate(workload, execution_config, list(decisions))
            for decisions in product(("restart", "recover"), repeat=count)
        ]
        oracle = min(
            candidates,
            key=lambda item: (_objective(item["summary"]), tuple(item["decisions"])),
        )
        method_decisions["post_hoc_exact_oracle"] = list(oracle["decisions"])
        overheads["post_hoc_exact_oracle"] = time.perf_counter() - started

        evaluated = {
            method: _simulate(workload, execution_config, decisions)
            for method, decisions in method_decisions.items()
        }
        oracle_summary = evaluated["post_hoc_exact_oracle"]["summary"]
        for method, result in evaluated.items():
            summary = result["summary"]
            result_rows.append(
                {
                    "design_id": point["design_id"],
                    "label": point["label"],
                    "source_type": point["source_type"],
                    "seed": workload["seed"],
                    "method": method,
                    "decisions": list(result["decisions"]),
                    "decision_overhead_seconds": overheads[method],
                    "actual_prefix_recompute_seconds": point[
                        "actual_prefix_recompute_seconds"
                    ],
                    "actual_recovery_overhead_seconds": point[
                        "actual_recovery_overhead_seconds"
                    ],
                    "actual_link_mbps": point["actual_link_mbps"],
                    "actual_target_model": point["actual_target_model"],
                    "estimated_target_model": point["estimated_target_model"],
                    "actual_target_model_load_seconds": point[
                        "actual_target_model_load_seconds"
                    ],
                    **estimated,
                    **realized,
                    **summary,
                    "workflow_results": result["workflow_results"],
                    "oracle_information_boundary": (
                        "post_hoc_realized_cost_oracle"
                        if method == "post_hoc_exact_oracle"
                        else "online_or_fixed_rule"
                    ),
                    "wrong_decision_count_vs_oracle": sum(
                        left != right
                        for left, right in zip(result["decisions"], oracle["decisions"])
                    ),
                    "completion_time_penalty_vs_oracle_seconds": float(
                        summary["total_completion_seconds"]
                    )
                    - float(oracle_summary["total_completion_seconds"]),
                    "makespan_penalty_vs_oracle_seconds": float(summary["makespan_seconds"])
                    - float(oracle_summary["makespan_seconds"]),
                    "transfer_penalty_vs_oracle_bytes": int(summary["total_transfer_bytes"])
                    - int(oracle_summary["total_transfer_bytes"]),
                    "recompute_penalty_vs_oracle_seconds": float(
                        summary["repeated_compute_seconds"]
                    )
                    - float(oracle_summary["repeated_compute_seconds"]),
                }
            )
    return result_rows


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "design_id": row["design_id"],
        "label": row["label"],
        "source_type": row["source_type"],
        "method": row["method"],
        "decisions": "-".join(row["decisions"]),
        "actual_prefix_recompute_seconds": row["actual_prefix_recompute_seconds"],
        "actual_recovery_overhead_seconds": row["actual_recovery_overhead_seconds"],
        "actual_link_mbps": row["actual_link_mbps"],
        "actual_target_model": row["actual_target_model"],
        "estimated_target_model": row["estimated_target_model"],
        "realized_recovery_minus_restart_seconds": row[
            "realized_recovery_minus_restart_seconds"
        ],
        "completed_workflows": row["completed_workflows"],
        "deadline_violations": row["deadline_violations"],
        "total_completion_seconds": row["total_completion_seconds"],
        "makespan_seconds": row["makespan_seconds"],
        "total_queue_wait_seconds": row["total_queue_wait_seconds"],
        "total_transfer_bytes": row["total_transfer_bytes"],
        "repeated_compute_seconds": row["repeated_compute_seconds"],
        "decision_overhead_seconds": row["decision_overhead_seconds"],
        "wrong_decision_count_vs_oracle": row["wrong_decision_count_vs_oracle"],
        "completion_time_penalty_vs_oracle_seconds": row[
            "completion_time_penalty_vs_oracle_seconds"
        ],
        "makespan_penalty_vs_oracle_seconds": row["makespan_penalty_vs_oracle_seconds"],
        "transfer_penalty_vs_oracle_bytes": row["transfer_penalty_vs_oracle_bytes"],
        "recompute_penalty_vs_oracle_seconds": row[
            "recompute_penalty_vs_oracle_seconds"
        ],
        "oracle_information_boundary": row["oracle_information_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"execution commit mismatch: expected {args.expected_commit}, got {commit}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")

    config_path = args.config.resolve()
    robustness = json.loads(config_path.read_text(encoding="utf-8"))
    if robustness["status"] != "FROZEN_BEFORE_EXECUTION":
        raise RuntimeError("robustness matrix is not frozen")
    points = robustness["design_points"]
    if len(points) > int(robustness["scope"]["maximum_design_points"]):
        raise RuntimeError("design point ceiling exceeded")
    if len({item["design_id"] for item in points}) != len(points):
        raise RuntimeError("duplicate design_id")
    baseline_path = ROOT / robustness["baseline_workload_config"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    rows = evaluate_design(robustness, baseline)
    if len(rows) != len(points) * 3:
        raise AssertionError("result row count drift")
    output.mkdir(parents=True)
    write_json(output / "all_method_results.json", rows)
    csv_rows = [_csv_row(row) for row in rows]
    with (output / "all_method_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    local_rows = [
        row for row in rows if row["method"] == "mechanism_aware_local_incremental_cost"
    ]
    write_json(
        output / "summary.json",
        {
            "design_point_count": len(points),
            "method_result_count": len(rows),
            "local_oracle_match_count": sum(
                row["wrong_decision_count_vs_oracle"] == 0 for row in local_rows
            ),
            "local_oracle_mismatch_count": sum(
                row["wrong_decision_count_vs_oracle"] > 0 for row in local_rows
            ),
            "maximum_local_completion_time_penalty_seconds": max(
                row["completion_time_penalty_vs_oracle_seconds"] for row in local_rows
            ),
            "maximum_local_makespan_penalty_seconds": max(
                row["makespan_penalty_vs_oracle_seconds"] for row in local_rows
            ),
            "maximum_local_transfer_penalty_bytes": max(
                row["transfer_penalty_vs_oracle_bytes"] for row in local_rows
            ),
            "maximum_local_recompute_penalty_seconds": max(
                row["recompute_penalty_vs_oracle_seconds"] for row in local_rows
            ),
            "post_hoc_oracle_is_information_fair_online_baseline": False,
        },
    )
    write_json(
        output / "completion_receipt.json",
        {
            "status": "PASS",
            "git_commit": commit,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "baseline_config_sha256": sha256_file(baseline_path),
            "design_point_count": len(points),
            "method_result_count": len(rows),
            "real_model_generate_calls": 0,
            "training_runs": 0,
            "formal_runs": 0,
            "holdout_runs": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    files = []
    for path in sorted(output.iterdir()):
        if path.name == "integrity_manifest.json":
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        output / "integrity_manifest.json",
        {"status": "PASS", "file_count": len(files), "files": files},
    )
    print(json.dumps({"status": "PASS", "rows": len(rows), "wall_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
