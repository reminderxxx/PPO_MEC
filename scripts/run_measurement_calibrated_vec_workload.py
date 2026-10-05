"""Generate workload v0.1 and execute the frozen three-method diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.workflow.measurement_calibrated_vec_workload import (  # noqa: E402
    compare_methods,
    generate_workloads,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    transfer = row["transfer_bytes_by_type"]
    return {
        "design_id": row["design_id"],
        "seed": row["seed"],
        "sharing": row["sharing"],
        "capacity": row["capacity"],
        "initial_target_model": row["initial_target_model"],
        "restore_cost": row["restore_cost"],
        "method": row["method"],
        "decisions": "-".join(row["decisions"]),
        "completed_nodes": row["completed_nodes"],
        "completed_workflows": row["completed_workflows"],
        "total_completion_seconds": row["total_completion_seconds"],
        "makespan_seconds": row["makespan_seconds"],
        "total_queue_wait_seconds": row["total_queue_wait_seconds"],
        "base_transfer_bytes": transfer["base"],
        "adapter_transfer_bytes": transfer["adapter"],
        "input_transfer_bytes": transfer["input"],
        "state_transfer_bytes": transfer["state"],
        "total_transfer_bytes": row["total_transfer_bytes"],
        "deadline_violations": row["deadline_violations"],
        "repeated_compute_seconds": row["repeated_compute_seconds"],
        "service_failure_count": row["service_failure_count"],
        "cache_prepare_count": row["cache_prepare_count"],
        "decision_overhead_seconds": row["decision_overhead_seconds"],
        "offline_future_information": row["offline_future_information"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
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
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_COMPARISON":
        raise RuntimeError("workload config is not frozen")
    if len(config["design"]["points"]) > 12 or len(config["seeds"]) > 3:
        raise RuntimeError("design or seed ceiling exceeded")

    started = time.perf_counter()
    workloads = generate_workloads(config)
    expected = len(config["design"]["points"]) * len(config["seeds"])
    if len(workloads) != expected:
        raise AssertionError("workload cell count drift")
    rows = [row for workload in workloads for row in compare_methods(workload, config)]
    expected_rows = expected * len(config["methods"])
    if len(rows) != expected_rows:
        raise AssertionError("comparison row count drift")

    output.mkdir(parents=True)
    with (output / "workload_instances.jsonl").open("w", encoding="utf-8") as handle:
        for workload in workloads:
            handle.write(json.dumps(workload, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output / "all_method_results.json", rows)
    csv_rows = [flatten(row) for row in rows]
    with (output / "all_method_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(
        output / "data_card.json",
        {
            "name": "measurement-calibrated semi-synthetic VEC workload",
            "version": config["workload_version"],
            "classification": "development mechanism diagnostic",
            "not_a_new_real_dataset": True,
            "public_value_representativeness_license_independent_reproduction": "unavailable",
            "calibration": config["calibration"],
            "synthetic_assumptions": config["synthetic_assumptions"],
            "seeds": config["seeds"],
            "design": config["design"],
            "scope": config["scope"],
        },
    )
    write_json(
        output / "completion_receipt.json",
        {
            "status": "PASS",
            "git_commit": commit,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "design_point_count": len(config["design"]["points"]),
            "seed_count": len(config["seeds"]),
            "workload_instance_count": len(workloads),
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
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output / "integrity_manifest.json",
        {"status": "PASS", "file_count": len(files), "files": files},
    )
    print(json.dumps({"status": "PASS", "rows": len(rows), "wall_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
