#!/usr/bin/env python3
"""Run the dedicated transaction through the real non-holdout benchmark chain.

This is a test-only acceptance.  It consumes the frozen public/formal split,
loads real frozen checkpoints, and never names or opens the sealed holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.dedicated_holdout_execution import (
    CAPACITIES,
    HOLDOUT_COMMAND_PACKAGE_VERSION,
    PRIMARY_METRICS,
    canonical_sha256,
    file_sha256,
    read_json,
    validate_command_package,
)
from src.runtime.evaluation_only_execution import (
    build_evaluation_execution_contract,
    validate_execution_contract,
)


SOURCE_REFERENCE = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_evaluation_only/"
    "typed_model_cache_post_ablation_20260921_g14e07_pending/"
    "evaluation_model_source_reference.json"
)
TEST_AGENTS = ("reactive_lru", "sa_ghmappo")
TEST_SEED = 7


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def replace_scalar(command: list[str], flag: str, value: str) -> None:
    if command.count(flag) != 1:
        raise ValueError(f"expected one {flag}")
    command[command.index(flag) + 1] = value


def replace_slice(command: list[str], start: str, end: str, values: list[str]) -> None:
    if command.count(start) != 1 or command.count(end) != 1:
        raise ValueError(f"expected one {start}/{end}")
    left = command.index(start) + 1
    right = command.index(end)
    command[left:right] = values


def rehash_evaluation(contract: dict[str, Any]) -> None:
    plans = contract["command_plans"]
    plan_hash = canonical_sha256(plans)
    contract["command_plan_sha256"] = plan_hash
    context = contract["evaluation_execution_context"]
    context["created_for_run_identity"] = canonical_sha256({
        "evaluation_run_id": contract["evaluation_run_id"],
        "executor_commit": contract["executor_commit"],
        "model_source_reference_sha256": contract["model_source_reference_sha256"],
        "command_plan_sha256": plan_hash,
        "post_ablation_handoff_sha256": None,
    })
    context["command_expansion"].update(
        outer_expansion_sha256=plan_hash,
        resolved_command_matrix_sha256=plan_hash,
        phase_count=len(contract["phases"]),
        command_count=sum(len(plan["commands"]) for plan in plans.values()),
    )
    context.pop("context_sha256", None)
    context["context_sha256"] = canonical_sha256(context)
    contract["evaluation_execution_context_sha256"] = context["context_sha256"]
    contract.pop("execution_contract_sha256", None)
    contract["execution_contract_sha256"] = canonical_sha256(contract)


def build_package(
    *, executor_checkout: Path, executor_commit: str, work_root: Path
) -> dict[str, Any]:
    output_root = work_root / "public_entry_output"
    source = read_json(SOURCE_REFERENCE)
    evaluation = build_evaluation_execution_contract(
        source_reference=source,
        evaluation_run_id=output_root.name,
        evaluation_run_root=output_root,
        executor_checkout=executor_checkout,
        executor_commit=executor_commit,
        python_executable=sys.executable,
    )
    commands = deepcopy(evaluation["command_plans"]["formal_controller"]["commands"])
    for capacity, command in zip(CAPACITIES, commands):
        replace_slice(command, "--agents", "--seeds", list(TEST_AGENTS))
        replace_slice(command, "--seeds", "--seed_checkpoint_manifest_path", [str(TEST_SEED)])
        replace_scalar(
            command,
            "--output_root",
            str(output_root / "staging" / f"scientific_{capacity}"),
        )
        if "--non-formal-rehearsal" not in command:
            command.append("--non-formal-rehearsal")
    evaluation["command_plans"]["formal_controller"]["commands"] = commands
    rehash_evaluation(evaluation)
    validate_execution_contract(evaluation, model_source_reference=source)

    statistics = [
        sys.executable,
        str(executor_checkout / "scripts/analyze_top_journal_statistics.py"),
        "--rows_path", "{G14R22_ROWS_0}",
        "--rows_path", "{G14R22_ROWS_1}",
        "--rows_path", "{G14R22_ROWS_2}",
        "--candidate_agent", "sa_ghmappo",
        "--baseline_agents", "reactive_lru",
        "--metrics", *PRIMARY_METRICS,
        "--pair_keys", "seed", "window_id", "workflow_id", "capacity_label",
        "--outer_cluster_keys", "source_segment_run_id", "window_id",
        "--inner_cluster_keys", "seed", "workflow_id", "capacity_label",
        "--ci_method", "bca", "--bootstrap_samples", "20", "--random_seed", "1401",
        "--output_root", "{G14R22_STATISTICS_OUTPUT_ROOT}",
    ]
    package: dict[str, Any] = {
        "command_package_version": HOLDOUT_COMMAND_PACKAGE_VERSION,
        "executor_checkout": str(executor_checkout),
        "executor_commit": executor_commit,
        "executor_git_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=executor_checkout, text=True
        ).strip(),
        "python_executable": sys.executable,
        "output_root": str(output_root),
        "phases": ["scientific", "statistics", "publication", "integrity"],
        "scientific_matrix": {
            "profile": "g14r22c_real_non_holdout_test_only",
            "agents": list(TEST_AGENTS),
            "seeds": [TEST_SEED],
            "capacities": list(CAPACITIES),
            "split": "formal_public_non_holdout",
            "windows": 12,
            "workflows": 3,
            "rows_per_child": 72,
            "total_expected_rows": 216,
            "holm_family_size": 6,
            "formal_statistics_eligible": False,
        },
        "commands": {"scientific": commands, "statistics": statistics},
        "evaluation_execution_contract": evaluation,
        "resolved_execution_context": evaluation["evaluation_execution_context"],
        "model_source_reference": source,
        "automatic_retry_count": 0,
        "acceptance_non_holdout": True,
        "acceptance_request_sha256": canonical_sha256({
            "profile": "g14r22c_real_non_holdout_test_only",
            "executor_commit": executor_commit,
            "output_root": str(output_root),
        }),
        "acceptance_checkpoint_audit": {
            "status": "pass",
            "actual_scope": {"models": 3, "total_bytes": 0},
            "models_canonical_sha256": canonical_sha256([]),
            "models": [],
            "note": "The benchmark loader reopens and validates the three selected real checkpoints.",
        },
    }
    package["command_package_sha256"] = canonical_sha256(package)
    validate_command_package(package, acceptance=True)
    return package


def validate_capacity_identity_outputs(work_root: Path) -> dict[str, Any]:
    published = work_root / "public_entry_output" / "published"
    scientific = published / "scientific"
    csv_audit: list[dict[str, Any]] = []
    for capacity in CAPACITIES:
        path = scientific / capacity / "benchmark_rows.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 72:
            raise ValueError(f"unexpected row count for {capacity}: {len(rows)}")
        if {row.get("capacity_label") for row in rows} != {capacity}:
            raise ValueError(f"capacity_label did not reach final CSV for {capacity}")
        if {row.get("runtime_config_resource_id") for row in rows} != {
            f"runtime_config.{capacity}"
        }:
            raise ValueError(f"runtime resource identity did not reach final CSV for {capacity}")
        if any(not row.get("source_segment_run_id") for row in rows):
            raise ValueError(f"source_segment_run_id missing in final CSV for {capacity}")
        csv_audit.append(
            {
                "capacity_label": capacity,
                "path": str(path),
                "sha256": file_sha256(path),
                "row_count": len(rows),
                "source_segment_run_ids": sorted(
                    {str(row["source_segment_run_id"]) for row in rows}
                ),
            }
        )
    statistics_path = published / "statistics" / "paired_statistics.json"
    statistics = read_json(statistics_path)
    if statistics.get("identity_validation", {}).get("status") != "passed_fail_closed":
        raise ValueError("statistics capacity identity validation did not pass")
    if statistics.get("pair_keys") != ["seed", "window_id", "workflow_id", "capacity_label"]:
        raise ValueError("statistics pair key identity drift")
    if statistics.get("outer_cluster_keys") != ["source_segment_run_id", "window_id"]:
        raise ValueError("statistics outer key identity drift")
    if statistics.get("inner_cluster_keys") != ["seed", "workflow_id", "capacity_label"]:
        raise ValueError("statistics inner key identity drift")
    rows = statistics.get("rows", [])
    if len(rows) != 6:
        raise ValueError("acceptance statistics family must contain six rows")
    for row in rows:
        if row.get("total_pair_count") != 108:
            raise ValueError("acceptance statistics lost a capacity pair coordinate")
        if row.get("inner_cluster_count") != row.get("available_paired_count"):
            raise ValueError("acceptance statistics merged capacity inner clusters")
    return {
        "status": "passed",
        "fixture_capacity_injection_used": False,
        "producer_csv_audit": csv_audit,
        "statistics_path": str(statistics_path),
        "statistics_sha256": file_sha256(statistics_path),
        "validated_pair_coordinate_count": statistics["identity_validation"][
            "validated_pair_coordinate_count"
        ],
        "inner_cluster_counts": [row["inner_cluster_count"] for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--prepared-command-package-path", type=Path)
    parser.add_argument("--background-request-path", type=Path)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=False)
    package_path = args.work_root / "real_non_holdout_command_package.json"
    if args.prepared_command_package_path is not None:
        prepared_path = args.prepared_command_package_path.resolve()
        package = read_json(prepared_path)
        validate_command_package(package, acceptance=True)
        if (
            Path(package["executor_checkout"]).resolve()
            != args.executor_checkout.resolve()
            or package["executor_commit"] != args.executor_commit
            or Path(package["output_root"]).resolve()
            != (args.work_root.resolve() / "public_entry_output")
        ):
            raise ValueError("prepared non-holdout package execution identity mismatch")
        package_path.write_bytes(prepared_path.read_bytes())
    else:
        package = build_package(
            executor_checkout=args.executor_checkout.resolve(),
            executor_commit=args.executor_commit,
            work_root=args.work_root.resolve(),
        )
        write_json(package_path, package)
    background_request_path = (
        args.background_request_path.resolve()
        if args.background_request_path is not None else None
    )
    background_request_sha256 = (
        file_sha256(background_request_path)
        if background_request_path is not None else None
    )
    command = [
        sys.executable,
        str(args.executor_checkout.resolve() / "scripts/run_dedicated_public_holdout.py"),
        "--command-package-path", str(package_path),
        "--check", "acceptance",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=args.executor_checkout.resolve(),
        text=True,
        capture_output=True,
        check=False,
    )
    capacity_identity_validation: dict[str, Any] | None = None
    validation_failure: str | None = None
    if completed.returncode == 0:
        try:
            capacity_identity_validation = validate_capacity_identity_outputs(args.work_root)
        except Exception as exc:
            validation_failure = f"{type(exc).__name__}: {exc}"
    passed = completed.returncode == 0 and validation_failure is None
    receipt = {
        "acceptance_version": "g14r22c_real_non_holdout_benchmark_v1",
        "test_only": True,
        "formal_statistics_eligible": False,
        "holdout_opened": False,
        "holdout_policy_runs": 0,
        "executor_commit": args.executor_commit,
        "executor_git_tree": package["executor_git_tree"],
        "command_package_path": str(package_path),
        "command_package_sha256": file_sha256(package_path),
        "background_request_path": (
            str(background_request_path) if background_request_path is not None else None
        ),
        "background_request_sha256": background_request_sha256,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "passed": passed,
        "capacity_identity_validation": capacity_identity_validation,
        "validation_failure": validation_failure,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "coverage": {
            "benchmark_entry": "scripts/benchmark_main_results.py",
            "agents": list(TEST_AGENTS),
            "capacities": list(CAPACITIES),
            "unique_real_checkpoints_expected": 3,
            "real_checkpoint_validations_expected": 108,
            "real_policy_rollouts_expected": 108,
            "reactive_rollouts_expected": 108,
            "benchmark_rows_expected": 216,
            "corrected_statistics_rows_expected": 6,
        },
    }
    write_json(args.work_root / "acceptance_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if passed else (completed.returncode or 2)


if __name__ == "__main__":
    raise SystemExit(main())
