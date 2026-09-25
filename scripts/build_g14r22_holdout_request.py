#!/usr/bin/env python3
"""Build the unsigned G14R22 holdout request and exact command package.

This builder reads only frozen formal/model metadata and holdout interval/plan
metadata.  It does not issue a grant or token, open holdout, or run a policy.
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
    ALL_AGENTS,
    CAPACITIES,
    GRANT_VALIDITY_CONTRACT,
    HOLDOUT_COMMAND_PACKAGE_VERSION,
    HOLDOUT_REQUEST_VERSION,
    LEARNED_AGENTS,
    PRIMARY_METRICS,
    SEEDS,
    canonical_sha256,
    file_sha256,
    validate_command_package,
    validate_unsigned_request,
    verify_checkpoint_bytes,
)
from src.runtime.evaluation_only_execution import build_evaluation_execution_contract


G14R21 = ROOT / "artifacts/analysis/typed_model_cache_g14r21_corrected_statistics_holdout_audit_20260923_v1"
SOURCE_REFERENCE = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_evaluation_only/"
    "typed_model_cache_post_ablation_20260921_g14e07_pending/evaluation_model_source_reference.json"
)
SOURCE_RUN = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260906_152847_g14c_v16"
)
SCIENTIFIC = Path("/Users/howen/Projects/PPO_MEC/artifacts/execution_checkouts/g14r20_i_scientific_a6d1fd8")
OUTPUT_ROOT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_holdout/"
    "typed_model_cache_holdout_20260926_g14r22d_once"
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def frozen(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def replace_flag(command: list[str], flag: str, value: str) -> None:
    if command.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag}")
    command[command.index(flag) + 1] = value


def build(executor_commit: str, executor_checkout: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = read(SOURCE_REFERENCE)
    reference_hash = source.pop("source_reference_sha256")
    if reference_hash != canonical_sha256(source):
        raise ValueError("source reference hash mismatch")
    source["source_reference_sha256"] = reference_hash
    evaluation = build_evaluation_execution_contract(
        source_reference=source,
        evaluation_run_id=OUTPUT_ROOT.name,
        evaluation_run_root=OUTPUT_ROOT,
        executor_checkout=executor_checkout,
        executor_commit=executor_commit,
        python_executable=Path("/Users/howen/Projects/PPO_MEC/.venv/bin/python"),
    )
    commands = deepcopy(evaluation["command_plans"]["formal_controller"]["commands"])
    holdout_plan = SCIENTIFIC / "configs/experiment/typed_model_cache_formal_protocol_v1_20260820/sealed_holdout_window_plan.json"
    for command in commands:
        replace_flag(command, "--window_plan_path", str(holdout_plan))
        replace_flag(command, "--formal_window_split", "sealed_holdout")
        replace_flag(command, "--window-plan-resource-id", "window_plan.typed_model_cache.sealed_holdout")
        replace_flag(command, "--output_root", "{G14R22_CELL_OUTPUT_ROOT}")
        command.extend([
            "--dedicated-holdout-opening-receipt", "{G14R22_OPENING_RECEIPT}",
            "--dedicated-holdout-request-sha256", "{G14R22_REQUEST_SHA256}",
            "--dedicated-holdout-command-package-sha256", "{G14R22_COMMAND_PACKAGE_SHA256}",
        ])
    statistics = [
        "/Users/howen/Projects/PPO_MEC/.venv/bin/python",
        str(executor_checkout / "scripts/analyze_top_journal_statistics.py"),
        "--rows_path", "{G14R22_ROWS_0}",
        "--rows_path", "{G14R22_ROWS_1}",
        "--rows_path", "{G14R22_ROWS_2}",
        "--candidate_agent", "sa_ghmappo",
        "--baseline_agents", *[agent for agent in ALL_AGENTS if agent != "sa_ghmappo"],
        "--metrics", *PRIMARY_METRICS,
        "--pair_keys", "seed", "window_id", "workflow_id", "capacity_label",
        "--outer_cluster_keys", "source_segment_run_id", "window_id",
        "--inner_cluster_keys", "seed", "workflow_id", "capacity_label",
        "--ci_method", "bca", "--bootstrap_samples", "10000", "--random_seed", "1401",
        "--formal-agent-order-contract-path",
        str(executor_checkout / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/formal_agent_order_contract.json"),
        "--output_root", "{G14R22_STATISTICS_OUTPUT_ROOT}",
    ]
    matrix = {
        "agents": list(ALL_AGENTS), "learned_agents": list(LEARNED_AGENTS),
        "seeds": list(SEEDS), "capacities": list(CAPACITIES), "outer_windows": 12,
        "workflows": 3, "scientific_child_count": 3, "rows_per_child": 2700,
        "total_expected_rows": 8100, "checkpoint_count": 150,
        "primary_metrics": list(PRIMARY_METRICS), "holm_family_size": 84,
    }
    package: dict[str, Any] = {
        "command_package_version": HOLDOUT_COMMAND_PACKAGE_VERSION,
        "executor_checkout": str(executor_checkout),
        "executor_commit": executor_commit,
        "executor_git_tree": git("rev-parse", f"{executor_commit}^{{tree}}"),
        "python_executable": "/Users/howen/Projects/PPO_MEC/.venv/bin/python",
        "output_root": str(OUTPUT_ROOT),
        "phases": ["scientific", "statistics", "publication", "integrity"],
        "scientific_matrix": matrix,
        "commands": {"scientific": commands, "statistics": statistics},
        "evaluation_execution_contract": evaluation,
        "resolved_execution_context": evaluation["evaluation_execution_context"],
        "model_source_reference": source,
        "automatic_retry_count": 0,
        "failure_policy": "any failure after atomic opening is terminal and permanently consumed",
        "tested_wall_clock_upper_bound_seconds": None,
    }
    package["command_package_sha256"] = canonical_sha256(package)
    integrity = read(G14R21 / "artifact_integrity_manifest.json")
    corrected_package_sha256 = canonical_sha256(integrity["files"])
    seal = ROOT / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/holdout_seal_record.json"
    split = ROOT / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/split_manifest.json"
    protocol = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json"
    fixed_inputs = [
        ROOT / "docs/project/g14r21_corrected_statistics_holdout_audit.md",
        G14R21 / "artifact_integrity_manifest.json",
        G14R21 / "corrected_statistics/paired_statistics.json",
        G14R21 / "corrected_claim_map.json",
        seal, split, holdout_plan, protocol, SOURCE_REFERENCE,
        SOURCE_RUN / "generated_checkpoint_resource_registry.json",
    ]
    for capacity in CAPACITIES:
        fixed_inputs.extend([
            SOURCE_RUN / "checkpoint_manifests" / capacity / "seed_checkpoint_manifest.json",
            SOURCE_RUN / "checkpoint_manifests" / capacity / "checkpoint_provenance_manifest.json",
        ])
    request: dict[str, Any] = {
        "request_version": HOLDOUT_REQUEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_AUTHORIZATION_REVIEW",
        "grant_signed": False, "execution_authorized": False,
        "holdout_opened": False, "holdout_consumed_permanently": False,
        "requested_authority": "one exact one-time holdout execution only",
        "prior_package": {
            "identity": "G14R22 v4",
            "status": "audit_only",
            "approval_migrates": False,
        },
        "grant_validity_contract": GRANT_VALIDITY_CONTRACT,
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": executor_commit,
        "output_root": str(OUTPUT_ROOT),
        "checkpoint_source_reference": frozen(SOURCE_REFERENCE),
        "checkpoint_opening_requirement": {
            "byte_rehash_required": True, "exact_model_count": 150,
            "actual_use_scope": "all 150 learned-agent checkpoints; no replacement or subset",
        },
        "corrected_statistics_package": {
            "root": str(G14R21), "files_canonical_sha256": corrected_package_sha256,
            "paired_statistics_sha256": file_sha256(G14R21 / "corrected_statistics/paired_statistics.json"),
            "claim_map_sha256": file_sha256(G14R21 / "corrected_claim_map.json"),
            "generator": str(executor_checkout / "scripts/analyze_top_journal_statistics.py"),
            "holm_family_size": 84,
        },
        "frozen_science": matrix,
        "frozen_inputs": [frozen(path) for path in fixed_inputs],
        "failure_boundary": {
            "before_atomic_open": "not_consumed; no scientific child may start",
            "at_or_after_atomic_open": "permanently_consumed",
            "child_failure": "terminal_failure_no_retry_no_resume_no_reopen",
            "partial_output": "retained_in_staging_and_permanently_consumed",
            "success": "permanently_consumed",
        },
        "prohibited_changes": [
            "model/checkpoint replacement", "window replacement", "agent/seed/capacity reduction",
            "metric or Holm-family change", "training", "formal rerun", "automatic retry",
        ],
        "workload_and_timing": {
            "scientific_children": 3, "expected_rows": 8100, "statistics_comparisons": 84,
            "checkpoint_byte_rehash_models": 150, "checkpoint_byte_rehash_bytes": 105464376,
            "measured_checkpoint_rehash_seconds": 3,
            "historical_formal_science_lower_bound": "5h12m",
            "historical_statistics_integrity_lower_bound": "15m",
            "tested_full_run_upper_bound": None,
            "calendar_guarantee": None,
            "resource_limits": {
                "device": "cpu", "parallel_scientific_children": 1,
                "automatic_retries": 0, "maximum_total_compute_cpu_hours": 2500,
            },
        },
    }
    request["request_sha256"] = canonical_sha256(request)
    checklist = {
        "integrity_checklist_version": "g14r22d_v2",
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "pre_open": [
            "verify signed exact grant and independent review", "verify one-time token hash",
            "verify timezone-aware grant validity is at most 72 hours",
            "recheck grant expiry immediately before atomic rename",
            "verify frozen input bytes", "rehash exactly 150 checkpoint files",
            "verify output root absent", "atomically publish opening receipt and ledger",
        ],
        "post_open": [
            "run exactly three serial capacity children without retry", "publish only complete child payloads",
            "generate exactly 84 corrected statistics rows", "publish integrity manifest and terminal receipt",
        ],
        "permanent_consumption": "opening, failure, partial output, and success all preserve consumed=true after atomic open",
        "long_running_expiry": (
            "expiry after atomic open never kills or changes the scientific process; "
            "consumption stays permanent"
        ),
        "verdict": "READY_TO_REQUEST_HOLDOUT_EXECUTION_AUTHORIZATION",
        "grant_signed": False, "execution_authorized": False,
    }
    validate_command_package(package)
    validate_unsigned_request(request, package)
    checkpoint_audit = verify_checkpoint_bytes(SOURCE_REFERENCE)
    return request, package, checklist, checkpoint_audit


def build_background_derivation_contract(
    request: dict[str, Any], package: dict[str, Any], *, executor_checkout: Path,
    executor_commit: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "background_derivation_contract_version": "g14r22d_formal_background_derivation_v1",
        "status": "UNSIGNED_AWAITING_NEW_EXACT_GRANT_AND_TOKEN",
        "launch_allowed": False,
        "execution_mode": "formal_holdout",
        "executor_checkout": str(executor_checkout),
        "executor_commit": executor_commit,
        "executor_git_tree": git("rev-parse", f"{executor_commit}^{{tree}}"),
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "output_root": package["output_root"],
        "grant_validity_contract": GRANT_VALIDITY_CONTRACT,
        "derivation_command_template": [
            package["python_executable"],
            str(executor_checkout / "scripts/derive_g14r22_holdout_background_job.py"),
            "--request-path", "{G14R22D_UNSIGNED_REQUEST_PATH}",
            "--command-package-path", "{G14R22D_SCIENTIFIC_PACKAGE_PATH}",
            "--grant-path", "{G14R22D_NEW_SIGNED_GRANT_PATH}",
            "--one-time-token-file", "{G14R22D_NEW_TOKEN_PATH}",
            "--executor-checkout", str(executor_checkout),
            "--python-executable", package["python_executable"],
            "--derived-root", "{G14R22D_DERIVED_ROOT}",
            "--job-root", "{G14R22D_UNIQUE_JOB_ROOT}",
        ],
        "required_new_owner_action": (
            "issue a new exact grant/token for this v2 request; v4 approval is not transferable"
        ),
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "resume_allowed": False,
        "reopen_allowed": False,
        "automatic_lock_cleanup_allowed": False,
    }
    value["background_derivation_contract_sha256"] = canonical_sha256(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    request, package, checklist, checkpoint_audit = build(args.executor_commit, args.executor_checkout.resolve())
    args.output_root.mkdir(parents=True, exist_ok=False)
    write(args.output_root / "holdout_request_unsigned.json", request)
    write(args.output_root / "command_package.json", package)
    write(args.output_root / "integrity_checklist.json", checklist)
    write(args.output_root / "checkpoint_byte_audit_preopen.json", checkpoint_audit)
    write(
        args.output_root / "background_derivation_contract.json",
        build_background_derivation_contract(
            request,
            package,
            executor_checkout=args.executor_checkout.resolve(),
            executor_commit=args.executor_commit,
        ),
    )
    print(json.dumps({"status": "pass", "request_sha256": request["request_sha256"],
                      "command_package_sha256": package["command_package_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
