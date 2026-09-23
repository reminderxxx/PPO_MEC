#!/usr/bin/env python3
"""Run the dedicated transaction through the real non-holdout benchmark chain.

This is a test-only acceptance.  It consumes the frozen public/formal split,
loads real frozen checkpoints, and never names or opens the sealed holdout.
"""

from __future__ import annotations

import argparse
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
        replace_scalar(command, "--max_workflows", "1")
        replace_scalar(command, "--max_steps", "1")
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
        "--formal-agent-order-contract-path",
        str(executor_checkout / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/formal_agent_order_contract.json"),
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
            "profile": "g14r22a_real_non_holdout_test_only",
            "agents": list(TEST_AGENTS),
            "seeds": [TEST_SEED],
            "capacities": list(CAPACITIES),
            "split": "formal_public_non_holdout",
            "windows": 12,
            "workflows": 1,
            "rows_per_child": 24,
            "total_expected_rows": 72,
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
            "profile": "g14r22a_real_non_holdout_test_only",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=False)
    package = build_package(
        executor_checkout=args.executor_checkout.resolve(),
        executor_commit=args.executor_commit,
        work_root=args.work_root.resolve(),
    )
    package_path = args.work_root / "real_non_holdout_command_package.json"
    write_json(package_path, package)
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
    receipt = {
        "acceptance_version": "g14r22a_real_non_holdout_benchmark_v1",
        "test_only": True,
        "formal_statistics_eligible": False,
        "holdout_opened": False,
        "holdout_policy_runs": 0,
        "executor_commit": args.executor_commit,
        "executor_git_tree": package["executor_git_tree"],
        "command_package_path": str(package_path),
        "command_package_sha256": file_sha256(package_path),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "coverage": {
            "benchmark_entry": "scripts/benchmark_main_results.py",
            "agents": list(TEST_AGENTS),
            "capacities": list(CAPACITIES),
            "real_checkpoint_loads_expected": 3,
            "real_policy_rollouts_expected": 36,
            "reactive_rollouts_expected": 36,
            "benchmark_rows_expected": 72,
            "corrected_statistics_rows_expected": 6,
        },
    }
    write_json(args.work_root / "acceptance_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
