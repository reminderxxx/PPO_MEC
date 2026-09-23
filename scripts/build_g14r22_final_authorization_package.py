#!/usr/bin/env python3
"""Freeze the final unsigned G14R22 holdout authorization package.

The builder only reads the completed public non-holdout acceptance and frozen
formal inputs.  It never creates a grant/token, launches a background process,
or opens the sealed holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_g14r22_holdout_request import build as build_holdout_package
from scripts.run_g14r22b_background_job import validate_integrity
from src.evaluators.dedicated_holdout_execution import (
    CAPACITIES,
    PRIMARY_METRICS,
    canonical_sha256,
    file_sha256,
    validate_command_package,
    validate_unsigned_request,
)


CRITICAL_RUNTIME_PATHS = (
    "scripts/analyze_top_journal_statistics.py",
    "scripts/benchmark_main_results.py",
    "scripts/run_dedicated_public_holdout.py",
    "scripts/run_g14r22a_real_non_holdout_acceptance.py",
    "src/evaluators/dedicated_holdout_execution.py",
    "src/runtime/evaluation_only_execution.py",
)
EXPECTED_AGENTS = {"reactive_lru", "sa_ghmappo"}
EXPECTED_PAIR_KEYS = ["seed", "window_id", "workflow_id", "capacity_label"]
EXPECTED_OUTER_KEYS = ["source_segment_run_id", "window_id"]
EXPECTED_INNER_KEYS = ["seed", "workflow_id", "capacity_label"]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=checkout, text=True
    ).strip()


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def canonical_binding(path: Path, field: str) -> dict[str, Any]:
    value = binding(path)
    value["canonical_sha256"] = read_json(path)[field]
    return value


def finite_number(value: str | None) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def verify_runtime_equivalence(
    checkout: Path, acceptance_commit: str, final_commit: str
) -> dict[str, Any]:
    changed = subprocess.run(
        ["git", "diff", "--quiet", acceptance_commit, final_commit, "--", *CRITICAL_RUNTIME_PATHS],
        cwd=checkout,
        check=False,
    )
    if changed.returncode != 0:
        raise ValueError("critical holdout/benchmark runtime changed after successful acceptance")
    files = []
    for relative in CRITICAL_RUNTIME_PATHS:
        target = checkout / relative
        files.append({"path": relative, "sha256": file_sha256(target)})
    return {
        "status": "pass",
        "acceptance_executor_commit": acceptance_commit,
        "final_executor_commit": final_commit,
        "critical_runtime_diff_empty": True,
        "files": files,
        "files_canonical_sha256": canonical_sha256(files),
    }


def verify_successful_acceptance(root: Path, final_checkout: Path, final_commit: str) -> dict[str, Any]:
    job = root / "background_job"
    work = root / "real_non_holdout"
    public = work / "public_entry_output"
    published = public / "published"
    terminal_path = job / "terminal_receipt.json"
    acceptance_path = work / "acceptance_receipt.json"
    execution_path = public / "execution_receipt.json"
    manifest_path = public / "artifact_integrity_manifest.json"
    terminal = read_json(terminal_path)
    acceptance = read_json(acceptance_path)
    execution = read_json(execution_path)
    manifest = read_json(manifest_path)

    if terminal.get("status") != "SUCCEEDED" or terminal.get("child_return_code") != 0:
        raise ValueError("background job is not terminal SUCCEEDED/0")
    if acceptance.get("passed") is not True or acceptance.get("return_code") != 0:
        raise ValueError("scientific acceptance did not pass")
    if acceptance.get("holdout_opened") is not False or acceptance.get("holdout_policy_runs") != 0:
        raise ValueError("acceptance crossed the holdout boundary")
    if execution.get("status") != "completed_permanently_consumed":
        raise ValueError("dedicated acceptance execution did not complete")
    if execution.get("acceptance_non_holdout") is not True or execution.get("holdout_opened") is not False:
        raise ValueError("dedicated execution receipt is not non-holdout")

    completion = terminal.get("scientific_completion", {})
    expected_hashes = {
        acceptance_path: completion.get("scientific_receipt_sha256"),
        execution_path: completion.get("dedicated_execution_receipt_sha256"),
        manifest_path: completion.get("integrity_manifest_sha256"),
    }
    for path, expected in expected_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"terminal receipt hash mismatch: {path}")
    integrity = validate_integrity(manifest_path, published)
    if integrity != completion.get("integrity"):
        raise ValueError("recomputed publication inventory differs from terminal receipt")
    if manifest.get("file_count") != 247:
        raise ValueError("successful publication inventory must contain exactly 247 files")

    csv_rows: list[dict[str, str]] = []
    csv_audit = []
    checkpoint_audit = []
    for capacity in CAPACITIES:
        capacity_root = published / "scientific" / capacity
        rows_path = capacity_root / "benchmark_rows.csv"
        with rows_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 72:
            raise ValueError(f"unexpected CSV row count for {capacity}: {len(rows)}")
        if {row.get("capacity_label") for row in rows} != {capacity}:
            raise ValueError(f"capacity label drift in {capacity}")
        if {row.get("runtime_config_resource_id") for row in rows} != {
            f"runtime_config.{capacity}"
        }:
            raise ValueError(f"runtime resource drift in {capacity}")
        if {row.get("agent_name") for row in rows} != EXPECTED_AGENTS:
            raise ValueError(f"agent scope drift in {capacity}")
        learned = [row for row in rows if row["agent_name"] == "sa_ghmappo"]
        reactive = [row for row in rows if row["agent_name"] == "reactive_lru"]
        if len(learned) != 36 or len(reactive) != 36:
            raise ValueError(f"rollout count drift in {capacity}")
        if {row.get("checkpoint_provenance_status") for row in learned} != {"compatible"}:
            raise ValueError(f"checkpoint provenance did not pass in {capacity}")
        checkpoint_hashes = {row.get("checkpoint_sha256") for row in learned}
        if len(checkpoint_hashes) != 1 or None in checkpoint_hashes or "" in checkpoint_hashes:
            raise ValueError(f"learned rows do not bind one checkpoint in {capacity}")

        run_manifest_path = capacity_root / "run_manifest.json"
        run_manifest = read_json(run_manifest_path)
        provenance = run_manifest.get("checkpoint_provenance_validation", {}).get(
            "sa_ghmappo", {}
        ).get("7", {})
        checkpoint_path = Path(str(provenance.get("checkpoint_path", "")))
        checkpoint_hash = str(provenance.get("checkpoint_sha256", ""))
        if (
            provenance.get("status") != "compatible"
            or checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or checkpoint_hash not in checkpoint_hashes
            or file_sha256(checkpoint_path) != checkpoint_hash
        ):
            raise ValueError(f"actual checkpoint byte identity failed in {capacity}")
        checkpoint_audit.append(
            {
                "capacity_label": capacity,
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_hash,
                "compatible_rollout_rows": len(learned),
            }
        )
        csv_audit.append(
            {
                "capacity_label": capacity,
                "path": str(rows_path),
                "size_bytes": rows_path.stat().st_size,
                "sha256": file_sha256(rows_path),
                "row_count": len(rows),
                "learned_rollout_rows": len(learned),
                "reactive_rollout_rows": len(reactive),
                "window_count": len({row["window_id"] for row in rows}),
                "workflow_count": len({row["workflow_id"] for row in rows}),
                "source_segment_run_ids": sorted({row["source_segment_run_id"] for row in rows}),
            }
        )
        csv_rows.extend(rows)

    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in csv_rows:
        key = tuple(str(row[field]) for field in EXPECTED_PAIR_KEYS)
        agents = pairs.setdefault(key, {})
        agent = row["agent_name"]
        if agent in agents:
            raise ValueError(f"duplicate pair row: {key}/{agent}")
        agents[agent] = row
    if len(pairs) != 108 or any(set(rows) != EXPECTED_AGENTS for rows in pairs.values()):
        raise ValueError("capacity-aware candidate/baseline pair matrix is incomplete")

    delay_by_capacity = {capacity: 0 for capacity in CAPACITIES}
    delay_by_window: dict[str, int] = {}
    delay_by_workflow: dict[str, int] = {}
    for key, agents in pairs.items():
        if all(
            finite_number(agents[agent].get("end_to_end_workflow_delay"))
            for agent in EXPECTED_AGENTS
        ):
            capacity = key[3]
            window = key[1]
            workflow = key[2]
            delay_by_capacity[capacity] += 1
            delay_by_window[window] = delay_by_window.get(window, 0) + 1
            delay_by_workflow[workflow] = delay_by_workflow.get(workflow, 0) + 1
    if delay_by_capacity != {
        "constrained_288mb": 10,
        "medium_576mb": 17,
        "relaxed_864mb": 17,
    }:
        raise ValueError("finite delay pair provenance drift")

    statistics_path = published / "statistics" / "paired_statistics.json"
    statistics = read_json(statistics_path)
    if (
        statistics.get("pair_keys") != EXPECTED_PAIR_KEYS
        or statistics.get("outer_cluster_keys") != EXPECTED_OUTER_KEYS
        or statistics.get("inner_cluster_keys") != EXPECTED_INNER_KEYS
        or statistics.get("identity_validation", {}).get("status") != "passed_fail_closed"
        or statistics.get("identity_validation", {}).get("validated_pair_coordinate_count") != 108
    ):
        raise ValueError("statistics identity validation drift")
    statistic_rows = statistics.get("rows")
    if not isinstance(statistic_rows, list) or len(statistic_rows) != 6:
        raise ValueError("acceptance statistics must contain six comparisons")
    seen_metrics = set()
    for row in statistic_rows:
        metric = row.get("metric")
        seen_metrics.add(metric)
        expected_available = 44 if metric == "end_to_end_workflow_delay" else 108
        expected_outer = 9 if metric == "end_to_end_workflow_delay" else 12
        if (
            row.get("total_pair_count") != 108
            or row.get("available_paired_count") != expected_available
            or row.get("inner_cluster_count") != expected_available
            or row.get("outer_cluster_count") != expected_outer
            or row.get("candidate_only_available_drop_count") != 0
            or row.get("baseline_only_available_drop_count") != 0
        ):
            raise ValueError(f"statistics availability/cluster drift for {metric}")
        expected_both_unavailable = 64 if metric == "end_to_end_workflow_delay" else 0
        if row.get("both_unavailable_drop_count") != expected_both_unavailable:
            raise ValueError(f"statistics missingness provenance drift for {metric}")
    if seen_metrics != set(PRIMARY_METRICS):
        raise ValueError("statistics metric family drift")

    runtime_equivalence = verify_runtime_equivalence(
        final_checkout, str(terminal["executor_commit"]), final_commit
    )
    return {
        "status": "pass",
        "acceptance_root": str(root.resolve()),
        "background_terminal": binding(terminal_path),
        "scientific_acceptance_receipt": binding(acceptance_path),
        "dedicated_execution_receipt": binding(execution_path),
        "publication_integrity_manifest": binding(manifest_path),
        "publication_integrity_recalculation": integrity,
        "csv_audit": csv_audit,
        "actual_checkpoint_audit": checkpoint_audit,
        "rollout_evidence": {
            "total_rows": len(csv_rows),
            "learned_policy_rollout_rows": sum(
                row["agent_name"] == "sa_ghmappo" for row in csv_rows
            ),
            "reactive_rollout_rows": sum(
                row["agent_name"] == "reactive_lru" for row in csv_rows
            ),
            "complete_capacity_aware_pair_coordinates": len(pairs),
        },
        "statistics": {
            "path": str(statistics_path),
            "sha256": file_sha256(statistics_path),
            "comparison_count": len(statistic_rows),
            "pair_keys": EXPECTED_PAIR_KEYS,
            "outer_cluster_keys": EXPECTED_OUTER_KEYS,
            "inner_cluster_keys": EXPECTED_INNER_KEYS,
            "validated_pair_coordinate_count": 108,
            "inner_cluster_counts": [row["inner_cluster_count"] for row in statistic_rows],
            "delay_both_unavailable_pair_count": 64,
            "delay_finite_pair_count": 44,
            "delay_finite_by_capacity": delay_by_capacity,
            "delay_finite_outer_window_count": len(delay_by_window),
            "delay_finite_by_window": delay_by_window,
            "delay_finite_by_workflow": delay_by_workflow,
        },
        "runtime_equivalence": runtime_equivalence,
        "holdout_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-checkout", type=Path, required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    checkout = args.executor_checkout.resolve()
    acceptance_root = args.acceptance_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError("final authorization package root must be absent")
    if git(checkout, "rev-parse", "HEAD") != args.executor_commit:
        raise ValueError("final executor commit mismatch")
    if git(checkout, "status", "--porcelain"):
        raise ValueError("final executor checkout must be clean")
    executor_tree = git(checkout, "rev-parse", "HEAD^{tree}")

    acceptance = verify_successful_acceptance(
        acceptance_root, checkout, args.executor_commit
    )
    request, package, checklist, checkpoint_audit = build_holdout_package(
        args.executor_commit, checkout
    )
    package["authorization_prerequisite_evidence"] = {
        "successful_public_non_holdout_acceptance": {
            "terminal_receipt": acceptance["background_terminal"],
            "scientific_acceptance_receipt": acceptance[
                "scientific_acceptance_receipt"
            ],
            "publication_integrity_manifest": acceptance[
                "publication_integrity_manifest"
            ],
            "publication_files_canonical_sha256": acceptance[
                "publication_integrity_recalculation"
            ]["files_canonical_sha256"],
            "publication_file_count": acceptance[
                "publication_integrity_recalculation"
            ]["checked_file_count"],
            "statistics_sha256": acceptance["statistics"]["sha256"],
            "runtime_equivalence_sha256": acceptance["runtime_equivalence"][
                "files_canonical_sha256"
            ],
        },
        "scope_boundary": "public non-holdout test-only acceptance; no performance claim",
    }
    package.pop("command_package_sha256", None)
    package["command_package_sha256"] = canonical_sha256(package)
    validate_command_package(package)

    request["command_package_sha256"] = package["command_package_sha256"]
    request["authorization_prerequisite_evidence"] = package[
        "authorization_prerequisite_evidence"
    ]
    request["final_executor_git_tree"] = executor_tree
    request.pop("request_sha256", None)
    request["request_sha256"] = canonical_sha256(request)
    validate_unsigned_request(request, package)
    checklist["request_sha256"] = request["request_sha256"]
    checklist["command_package_sha256"] = package["command_package_sha256"]
    checklist["successful_public_non_holdout_acceptance_verified"] = True
    checklist["grant_signed"] = False
    checklist["execution_authorized"] = False

    output_root.mkdir(parents=True, exist_ok=False)
    request_path = output_root / "holdout_request_unsigned.json"
    package_path = output_root / "scientific_command_package.json"
    checklist_path = output_root / "integrity_checklist.json"
    checkpoint_path = output_root / "checkpoint_byte_audit_preopen.json"
    write_json(request_path, request)
    write_json(package_path, package)
    write_json(checklist_path, checklist)
    write_json(checkpoint_path, checkpoint_audit)

    recalculation = {
        "recalculation_version": "g14r22_final_publication_integrity_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source_manifest": acceptance["publication_integrity_manifest"],
        **acceptance["publication_integrity_recalculation"],
        "csv_audit": acceptance["csv_audit"],
        "statistics": acceptance["statistics"],
        "actual_checkpoint_audit": acceptance["actual_checkpoint_audit"],
        "rollout_evidence": acceptance["rollout_evidence"],
        "holdout_opened": False,
    }
    recalculation_path = output_root / "publication_integrity_recalculation.json"
    write_json(recalculation_path, recalculation)

    holdout_seal_path = (
        ROOT
        / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/holdout_seal_record.json"
    )
    holdout_seal = read_json(holdout_seal_path)
    formal_output = Path(str(package["output_root"]))
    if formal_output.exists() or holdout_seal.get("opened") is not False:
        raise ValueError("formal holdout is not sealed/unopened")
    protection = {
        "protection_snapshot_version": "g14r22_final_pre_authorization_v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "executor": {
            "checkout": str(checkout),
            "commit": args.executor_commit,
            "git_tree": executor_tree,
            "clean": True,
        },
        "successful_acceptance": {
            "terminal_status": "SUCCEEDED",
            "child_return_code": 0,
            "holdout_opened": False,
            "publication_file_count": 247,
            "publication_files_canonical_sha256": recalculation[
                "files_canonical_sha256"
            ],
        },
        "holdout_seal": {
            **binding(holdout_seal_path),
            "sealed": holdout_seal.get("sealed"),
            "opened": holdout_seal.get("opened"),
            "consumed_permanently": holdout_seal.get("consumed_permanently"),
            "one_time_execution_token_status": holdout_seal.get(
                "one_time_execution_token_status"
            ),
        },
        "formal_output_root": str(formal_output),
        "formal_output_root_absent": True,
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "grant_signed": False,
        "token_issued": False,
        "execution_authorized": False,
        "holdout_opened": False,
    }
    protection_path = output_root / "protection_snapshot.json"
    write_json(protection_path, protection)

    review = {
        "review_version": "g14r22_final_independent_readonly_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "literature_cutoff": "2026-09-21",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": output_root.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": args.executor_commit,
        "evidence_level": "E2_ARTIFACT_AUDITED_NON_HOLDOUT_ACCEPTANCE_AND_PREOPEN_ONLY",
        "status": "pass",
        "verdict": "READY_TO_REQUEST_HOLDOUT_EXECUTION_AUTHORIZATION",
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "acceptance_findings": {
            "terminal_succeeded": True,
            "publication_inventory_exact": True,
            "publication_file_count": 247,
            "csv_rows_by_capacity": {
                row["capacity_label"]: row["row_count"]
                for row in acceptance["csv_audit"]
            },
            "actual_checkpoint_count": len(acceptance["actual_checkpoint_audit"]),
            "learned_policy_rollout_rows": acceptance["rollout_evidence"][
                "learned_policy_rollout_rows"
            ],
            "capacity_aware_pair_coordinates": 108,
            "statistics_comparisons": 6,
            "inner_cluster_counts": acceptance["statistics"][
                "inner_cluster_counts"
            ],
            "delay_finite_pair_source": {
                "finite": 44,
                "both_unavailable": 64,
                "by_capacity": acceptance["statistics"][
                    "delay_finite_by_capacity"
                ],
                "outer_windows": acceptance["statistics"][
                    "delay_finite_outer_window_count"
                ],
            },
            "runtime_equivalence": acceptance["runtime_equivalence"],
        },
        "authorization_boundary": {
            "grant_signed": False,
            "token_issued": False,
            "execution_authorized": False,
            "holdout_opened": False,
            "formal_statistics_or_performance_claim": False,
        },
        "remaining_required_action": "independent owner review and exact grant/token issuance",
    }
    review_path = output_root / "independent_readonly_review.json"
    write_json(review_path, review)

    background = {
        "background_task_package_version": "g14r22_holdout_pre_authorization_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_UNSIGNED_AWAITING_GRANT_AND_TOKEN",
        "launch_allowed": False,
        "executor_checkout": str(checkout),
        "executor_commit": args.executor_commit,
        "executor_git_tree": executor_tree,
        "python_executable": package["python_executable"],
        "cwd": str(checkout),
        "request": canonical_binding(request_path, "request_sha256"),
        "scientific_command_package": canonical_binding(
            package_path, "command_package_sha256"
        ),
        "independent_readonly_review": binding(review_path),
        "protection_snapshot": binding(protection_path),
        "future_authorized_command_template": [
            package["python_executable"],
            str(checkout / "scripts/run_dedicated_public_holdout.py"),
            "--request-path",
            str(request_path),
            "--command-package-path",
            str(package_path),
            "--grant-path",
            "{G14R22_SIGNED_GRANT_PATH}",
            "--one-time-token-file",
            "{G14R22_ONE_TIME_TOKEN_PATH}",
            "--check",
            "execute",
        ],
        "unresolved_authorization_placeholders": [
            "{G14R22_SIGNED_GRANT_PATH}",
            "{G14R22_ONE_TIME_TOKEN_PATH}",
        ],
        "automatic_retry_count": 0,
        "grant_signed": False,
        "token_issued": False,
        "execution_authorized": False,
        "holdout_opened": False,
    }
    background["background_task_package_sha256"] = canonical_sha256(background)
    background_path = output_root / "background_task_package.json"
    write_json(background_path, background)

    previous = []
    for version in ("v1", "v2", "v3"):
        old_root = ROOT / (
            "artifacts/analysis/"
            f"typed_model_cache_g14r22_holdout_execution_contract_20260923_{version}"
        )
        old_request = old_root / "holdout_request_unsigned.json"
        old_package = old_root / "command_package.json"
        previous.append(
            {
                "root": str(old_root),
                "request": canonical_binding(old_request, "request_sha256"),
                "command_package": canonical_binding(
                    old_package, "command_package_sha256"
                ),
            }
        )
    supersession = {
        "supersession_version": "g14r22_final_authorization_v4",
        "status": "v1_v2_v3_retained_audit_only_must_not_be_signed_or_executed",
        "superseded": previous,
        "replacement_root": str(output_root),
        "replacement_request_sha256": request["request_sha256"],
        "replacement_command_package_sha256": package["command_package_sha256"],
        "replacement_executor_commit": args.executor_commit,
        "reason": "v4 uniquely binds the successful G14R22-C public non-holdout background acceptance and final clean executor",
        "grant_signed": False,
        "execution_authorized": False,
        "holdout_opened": False,
    }
    supersession_path = output_root / "superseded_request_record.json"
    write_json(supersession_path, supersession)

    inventory = []
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            inventory.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    receipt = {
        "receipt_version": "g14r22_final_authorization_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_AUTHORIZATION_REVIEW",
        "root": str(output_root),
        "request_sha256": request["request_sha256"],
        "request_file_sha256": file_sha256(request_path),
        "command_package_sha256": package["command_package_sha256"],
        "command_package_file_sha256": file_sha256(package_path),
        "background_task_package_sha256": background[
            "background_task_package_sha256"
        ],
        "background_task_package_file_sha256": file_sha256(background_path),
        "files": inventory,
        "files_canonical_sha256": canonical_sha256(inventory),
        "grant_signed": False,
        "token_issued": False,
        "execution_authorized": False,
        "holdout_opened": False,
    }
    receipt_path = output_root / "authorization_bundle_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
