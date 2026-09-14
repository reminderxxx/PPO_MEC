"""Build the fail-closed G14R20-I5-A acceptance and unsigned-request package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ORIGINAL_PROJECT_GRANT_PATH,
    ORIGINAL_REQUEST_PATH,
    ORIGINAL_RUN_ROOT,
    build_restricted_recovery_request,
    validate_restricted_recovery_request,
)


PROTECTED_USER_RELATIVE_PATHS = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_REPORT_LABELS = ("acceptance_gate", "targeted", "adjacent", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovery-execution-id", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--executor-checkout", required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--protected-start-snapshot", required=True)
    parser.add_argument("--historical-correction", required=True)
    parser.add_argument("--historical-full-junit", required=True)
    parser.add_argument("--i4-artifact-root", required=True)
    parser.add_argument("--skip-review", required=True)
    parser.add_argument("--supplemental-validation-receipt", required=True)
    for label in REQUIRED_REPORT_LABELS:
        option = label.replace("_", "-")
        parser.add_argument(f"--{option}-junit", required=True)
        parser.add_argument(f"--{option}-receipt", required=True)
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_create_only(path: Path, payload: object) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or non-regular {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def build_protected_snapshot(
    workspace_root: Path,
    *,
    snapshot_id: str,
    capture_kind: str,
    capture_source: str,
) -> dict[str, object]:
    workspace = workspace_root.resolve(strict=True)
    rows: list[dict[str, object]] = []
    for relative in PROTECTED_USER_RELATIVE_PATHS:
        path = (workspace / relative).resolve(strict=True)
        if path != workspace / relative or not path.is_file() or path.is_symlink():
            raise ValueError(f"protected path is not an in-place regular file: {path}")
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path),
                "size_bytes": stat.st_size,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
        )
    payload: dict[str, object] = {
        "protected_workspace_snapshot_version": "1.0.0",
        "snapshot_id": snapshot_id,
        "capture_kind": capture_kind,
        "captured_at": now_iso(),
        "capture_source": capture_source,
        "workspace_root": str(workspace),
        "files": rows,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    return payload


def validate_protected_snapshot(
    value: dict[str, object], *, expected_capture_kind: str | None = None
) -> dict[str, object]:
    if value.get("protected_workspace_snapshot_version") != "1.0.0":
        raise ValueError("unsupported protected snapshot version")
    if expected_capture_kind and value.get("capture_kind") != expected_capture_kind:
        raise ValueError("protected snapshot capture kind mismatch")
    if not isinstance(value.get("capture_source"), str) or not value["capture_source"]:
        raise ValueError("protected snapshot source is missing")
    try:
        datetime.fromisoformat(str(value["captured_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("protected snapshot capture time is invalid") from exc
    workspace = Path(str(value.get("workspace_root", "")))
    if not workspace.is_absolute():
        raise ValueError("protected snapshot workspace root must be absolute")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != len(PROTECTED_USER_RELATIVE_PATHS):
        raise ValueError("protected snapshot file coverage mismatch")
    by_relative: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("protected snapshot row must be an object")
        relative = row.get("relative_path")
        absolute = row.get("absolute_path")
        digest = row.get("sha256")
        if relative not in PROTECTED_USER_RELATIVE_PATHS or relative in by_relative:
            raise ValueError("protected snapshot relative path mismatch")
        if Path(str(absolute)) != workspace / str(relative):
            raise ValueError("protected snapshot absolute path mismatch")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError("protected snapshot file hash is invalid")
        for integer_field in ("size_bytes", "inode", "mtime_ns"):
            if not isinstance(row.get(integer_field), int):
                raise ValueError(f"protected snapshot {integer_field} is invalid")
        by_relative[str(relative)] = row
    expected_snapshot_hash = value.get("snapshot_sha256")
    without_hash = {key: item for key, item in value.items() if key != "snapshot_sha256"}
    if (
        not isinstance(expected_snapshot_hash, str)
        or canonical_sha256(without_hash) != expected_snapshot_hash
    ):
        raise ValueError("protected snapshot semantic hash mismatch")
    return value


def compare_protected_snapshots(
    start: dict[str, object], end: dict[str, object]
) -> dict[str, object]:
    if start["workspace_root"] != end["workspace_root"]:
        raise ValueError("protected snapshot workspace changed")
    start_rows = {str(row["relative_path"]): row for row in start["files"]}  # type: ignore[index]
    end_rows = {str(row["relative_path"]): row for row in end["files"]}  # type: ignore[index]
    comparisons = []
    for relative in PROTECTED_USER_RELATIVE_PATHS:
        before = start_rows[relative]
        after = end_rows[relative]
        comparisons.append(
            {
                "relative_path": relative,
                "absolute_path": after["absolute_path"],
                "start_sha256": before["sha256"],
                "end_sha256": after["sha256"],
                "content_unchanged": before["sha256"] == after["sha256"],
                "start_size_bytes": before["size_bytes"],
                "end_size_bytes": after["size_bytes"],
                "start_inode": before["inode"],
                "end_inode": after["inode"],
                "start_mtime_ns": before["mtime_ns"],
                "end_mtime_ns": after["mtime_ns"],
            }
        )
    return {
        "status": "pass" if all(row["content_unchanged"] for row in comparisons) else "fail",
        "scope": "g14r20_i5_a_revalidation_only",
        "historical_i5_inference_allowed": False,
        "start_snapshot_sha256": start["snapshot_sha256"],
        "end_snapshot_sha256": end["snapshot_sha256"],
        "files": comparisons,
    }


def junit_cases(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing JUnit evidence: {path}")
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(f"unsupported JUnit root: {root.tag}")
    cases: list[dict[str, object]] = []
    time_seconds = 0.0
    for case in root.findall(".//testcase"):
        classname = case.get("classname")
        name = case.get("name")
        if not classname or not name:
            raise ValueError("JUnit testcase identity is incomplete")
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        terminal_count = sum(item is not None for item in (failure, error, skipped))
        if terminal_count > 1:
            raise ValueError("JUnit testcase has multiple terminal states")
        terminal = failure if failure is not None else error if error is not None else skipped
        status = (
            "failure"
            if failure is not None
            else "error"
            if error is not None
            else "skipped"
            if skipped is not None
            else "passed"
        )
        time_seconds += float(case.get("time", "0"))
        cases.append(
            {
                "nodeid": f"{classname}::{name}",
                "status": status,
                "message": "" if terminal is None else str(terminal.get("message") or ""),
                "details": "" if terminal is None else str(terminal.text or ""),
            }
        )
    summary: dict[str, object] = {
        "tests": len(cases),
        "failures": sum(case["status"] == "failure" for case in cases),
        "errors": sum(case["status"] == "error" for case in cases),
        "skipped": sum(case["status"] == "skipped" for case in cases),
        "passed": sum(case["status"] == "passed" for case in cases),
        "time_seconds": time_seconds,
    }
    return summary, cases


def validate_skip_review(
    skip_review: dict[str, object], report_cases: dict[str, list[dict[str, object]]]
) -> dict[str, object]:
    if skip_review.get("skip_review_version") != "1.0.0":
        raise ValueError("unsupported skip review version")
    rows = skip_review.get("rows")
    if not isinstance(rows, list):
        raise ValueError("skip review rows are missing")
    reviewed: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("skip review row must be an object")
        key = (str(row.get("report_label")), str(row.get("nodeid")))
        if key in reviewed:
            raise ValueError("duplicate skip review row")
        if row.get("same_scope_substitute_verified") is not True:
            raise ValueError("skip lacks a verified same-scope substitute")
        if not row.get("reason") or not row.get("substitute_evidence"):
            raise ValueError("skip explanation or substitute evidence is missing")
        reviewed[key] = row
    observed = {
        (label, str(case["nodeid"]))
        for label, cases in report_cases.items()
        for case in cases
        if case["status"] == "skipped"
    }
    if observed != set(reviewed):
        raise ValueError("skip review does not exactly match observed skips")
    return {"observed_skip_count": len(observed), "rows": rows, "status": "pass"}


def validate_test_receipt(
    *,
    label: str,
    junit_path: Path,
    receipt_path: Path,
    expected_checkout: Path,
    expected_commit: str,
    expected_tree: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary, cases = junit_cases(junit_path)
    receipt = load_json_object(receipt_path, f"{label} test receipt")
    if receipt.get("restricted_recovery_test_receipt_version") != "1.0.0":
        raise ValueError(f"{label} test receipt version mismatch")
    expected = {
        "label": label,
        "executor_checkout": str(expected_checkout),
        "executor_commit": expected_commit,
        "executor_git_tree": expected_tree,
        "return_code": 0,
        "checkout_clean_before": True,
        "checkout_clean_after": True,
        "junit_absolute_path": str(junit_path.resolve()),
        "junit_sha256": file_sha256(junit_path),
        "junit_summary": summary,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"{label} test receipt {field} mismatch")
    if summary["tests"] == 0:
        raise ValueError(f"{label} JUnit contains an empty test set")
    if summary["failures"] or summary["errors"]:
        raise ValueError(f"{label} JUnit contains failures or errors")
    return {
        **summary,
        "junit_path": junit_path.name,
        "junit_sha256": file_sha256(junit_path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": file_sha256(receipt_path),
        "executor_commit": expected_commit,
        "status": "pass",
    }, cases


def validate_supplemental_validation_receipt(
    receipt: dict[str, object],
    *,
    expected_checkout: Path,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, object]:
    if receipt.get("restricted_recovery_validation_receipt_version") != "1.0.0":
        raise ValueError("supplemental validation receipt version mismatch")
    expected = {
        "executor_checkout": str(expected_checkout),
        "executor_commit": expected_commit,
        "executor_git_tree": expected_tree,
        "executor_identity_unchanged": True,
        "baseline_commit": "398799177b0189353b10ce90d53ffba13bfd1d92",
        "status": "pass",
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"supplemental validation receipt {field} mismatch")
    checks = receipt.get("checks")
    required = {
        "smoke",
        "compile_import",
        "target_imports",
        "diff_check",
        "clean_scope",
        "protected_scope_diff_check",
    }
    if not isinstance(checks, list):
        raise ValueError("supplemental validation checks are missing")
    by_name = {
        str(check.get("name")): check for check in checks if isinstance(check, dict)
    }
    if set(by_name) != required:
        raise ValueError("supplemental validation check membership mismatch")
    if any(check.get("status") != "pass" for check in by_name.values()):
        raise ValueError("supplemental validation contains a failed check")
    return receipt


def build_historical_anomaly_disposition(
    historical_path: Path,
    current_cases: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    summary, cases = junit_cases(historical_path)
    if (summary["failures"], summary["errors"], summary["skipped"]) != (62, 84, 1):
        raise ValueError("historical JUnit exception counts do not match the preserved report")
    current_status: dict[str, set[str]] = {}
    for report_cases in current_cases.values():
        for case in report_cases:
            current_status.setdefault(str(case["nodeid"]), set()).add(str(case["status"]))
    rows = []
    for case in cases:
        if case["status"] not in {"failure", "error", "skipped"}:
            continue
        text = f"{case['message']}\n{case['details']}"
        nodeid = str(case["nodeid"])
        if "cryptography" in text and "No module named" in text:
            classification = "missing_isolated_test_dependency_cryptography"
            disposition = "rerun_with_tmp_dependency_overlay"
        elif "Operation not permitted" in text and "ps" in text:
            classification = "sandbox_process_inspection_permission_denied"
            disposition = "rerun_with_explicit_process_inspection_permission"
        elif nodeid.startswith(
            "tests.test_continuation_executor_boundaries::test_native_faults_and_restarts["
        ) and "invocations.jsonl" in text:
            classification = "secondary_driver_evidence_missing_after_pre_dispatch_stop"
            disposition = "rerun_with_tmp_dependency_overlay_and_process_permission"
        elif nodeid.startswith("tests.test_evaluation_only_public_bootstrap::"):
            classification = "secondary_assertion_after_process_permission_pre_dispatch_stop"
            disposition = "rerun_with_explicit_process_inspection_permission"
        elif case["status"] == "skipped" and "G14R20_I5_PUBLIC_ACCEPTANCE=1" in text:
            classification = "conditional_public_acceptance_not_enabled"
            disposition = "same_scope_public_two_process_chain_rerun_enabled"
        else:
            raise ValueError(f"unclassified historical regression anomaly: {case['nodeid']}")
        rerun_statuses = sorted(current_status.get(str(case["nodeid"]), set()))
        if "passed" not in rerun_statuses:
            raise ValueError(f"historical anomaly lacks a passing rerun: {case['nodeid']}")
        rows.append(
            {
                "nodeid": case["nodeid"],
                "historical_status": case["status"],
                "classification": classification,
                "disposition": disposition,
                "current_rerun_statuses": rerun_statuses,
                "resolved": True,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1
    return {
        "version": "1.0.0",
        "historical_junit_absolute_path": str(historical_path.resolve()),
        "historical_junit_sha256": file_sha256(historical_path),
        "historical_summary": summary,
        "exception_or_skip_row_count": len(rows),
        "classification_counts": counts,
        "rows": rows,
        "status": "pass",
    }


def audit_i4_originals(root: Path) -> dict[str, object]:
    artifact_root = root.resolve(strict=True)
    expected_names = {
        "engineering_acceptance_and_recovery_review.json",
        "independent_verification_rerun.json",
    }
    manifest_path = artifact_root / "integrity_manifest.json"
    manifest = load_json_object(manifest_path, "I4 integrity manifest")
    manifest_rows = manifest.get("files")
    if not isinstance(manifest_rows, list):
        raise ValueError("I4 manifest files are missing")
    declared = {str(row.get("path")): row for row in manifest_rows if isinstance(row, dict)}
    if set(declared) != expected_names:
        raise ValueError("I4 manifest membership mismatch")
    rows = []
    for name in sorted(expected_names):
        path = artifact_root / name
        payload = load_json_object(path, f"I4 {name}")
        stat = path.stat()
        if declared[name].get("size_bytes") != stat.st_size:
            raise ValueError(f"I4 size mismatch: {name}")
        if declared[name].get("sha256") != file_sha256(path):
            raise ValueError(f"I4 hash mismatch: {name}")
        rows.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": file_sha256(path),
                "mtime_ns": stat.st_mtime_ns,
                "review_time": payload.get("reviewed_at") or payload.get("verified_at"),
            }
        )
    return {
        "version": "1.0.0",
        "status": "pass_current_complete_with_temporal_qualification",
        "artifact_root": str(artifact_root),
        "path_resolution_correction": (
            "The old I5 check searched the main/I5-relative artifacts/analysis path. "
            "The supplied I4 checkout absolute path is a different filesystem root."
        ),
        "temporal_qualification": (
            "The engineering review predates the old I5 stop; the independent rerun and "
            "current manifest update postdate it and are current completeness evidence, not "
            "proof that all three current bytes existed at the old stop."
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "files": rows,
        "old_package_modified_by_i5_a": False,
    }


def git_identity(checkout: Path) -> tuple[str, str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "README.md",
            "docs",
            "scripts",
            "src",
            "tests",
            "configs",
        ],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).strip()
    return head, tree, status


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_root)
    if output.exists():
        raise ValueError("acceptance output root must not already exist")
    executor_checkout = Path(args.executor_checkout).resolve(strict=True)
    if executor_checkout != ROOT.resolve():
        raise ValueError("builder checkout and executor checkout differ")
    head, tree, status = git_identity(executor_checkout)
    if head != args.executor_commit or status:
        raise ValueError("executor commit mismatch or code scope is not clean")

    historical_correction_path = Path(args.historical_correction).resolve(strict=True)
    historical_correction = load_json_object(
        historical_correction_path, "historical qualification correction"
    )
    if (
        historical_correction.get("status")
        != "CORRECTED_WITH_HISTORICAL_START_UNAVAILABLE"
        or historical_correction.get("historical_start_evidence") != "unavailable"
        or historical_correction.get("logged_interval_start_evidence") != "available"
        or historical_correction.get("old_qualification_stop_retained") is not True
    ):
        raise ValueError("historical correction evidence is incomplete")

    start_snapshot_path = Path(args.protected_start_snapshot).resolve(strict=True)
    start_snapshot = validate_protected_snapshot(
        load_json_object(start_snapshot_path, "protected start snapshot"),
        expected_capture_kind="g14r20_i5_a_revalidation_start",
    )
    end_snapshot = build_protected_snapshot(
        Path(str(start_snapshot["workspace_root"])),
        snapshot_id="g14r20_i5_a_revalidation_end",
        capture_kind="g14r20_i5_a_revalidation_end",
        capture_source="build_restricted_recovery_acceptance_artifacts.py live end observation",
    )
    protected = compare_protected_snapshots(start_snapshot, end_snapshot)
    if protected["status"] != "pass":
        raise ValueError("protected user-file content changed after the explicit start snapshot")

    report_paths: dict[str, tuple[Path, Path]] = {}
    for label in REQUIRED_REPORT_LABELS:
        report_paths[label] = (
            Path(getattr(args, f"{label}_junit")).resolve(strict=True),
            Path(getattr(args, f"{label}_receipt")).resolve(strict=True),
        )
    test_results: dict[str, dict[str, object]] = {}
    report_cases: dict[str, list[dict[str, object]]] = {}
    for label, (junit_path, receipt_path) in report_paths.items():
        result, cases = validate_test_receipt(
            label=label,
            junit_path=junit_path,
            receipt_path=receipt_path,
            expected_checkout=executor_checkout,
            expected_commit=head,
            expected_tree=tree,
        )
        test_results[label] = result
        report_cases[label] = cases
    skip_review = validate_skip_review(
        load_json_object(Path(args.skip_review), "skip review"), report_cases
    )
    supplemental_validation_path = Path(args.supplemental_validation_receipt).resolve(
        strict=True
    )
    supplemental_validation = validate_supplemental_validation_receipt(
        load_json_object(supplemental_validation_path, "supplemental validation receipt"),
        expected_checkout=executor_checkout,
        expected_commit=head,
        expected_tree=tree,
    )
    anomaly_disposition = build_historical_anomaly_disposition(
        Path(args.historical_full_junit).resolve(strict=True), report_cases
    )
    i4_audit = audit_i4_originals(Path(args.i4_artifact_root))

    public_summary_source = report_paths["targeted"][0].parent / "public_recovery_acceptance.json"
    public_summary = load_json_object(public_summary_source, "public acceptance counters")
    if (
        public_summary.get("negative_dispatch_count") != 0
        or public_summary.get("negative_recovery_write_count") != 0
        or public_summary.get("synthetic_dispatch_count") != 2
        or public_summary.get("successful_cell_process_count") != 2
    ):
        raise ValueError("public acceptance dispatch/write counters failed")

    request = build_restricted_recovery_request(
        original_request_path=ORIGINAL_REQUEST_PATH,
        original_project_grant_path=ORIGINAL_PROJECT_GRANT_PATH,
        original_run_root=ORIGINAL_RUN_ROOT,
        recovery_execution_id=args.recovery_execution_id,
        recovery_root=args.recovery_root,
        executor_checkout=str(executor_checkout),
        executor_commit=args.executor_commit,
        python_executable=args.python_executable,
    )
    validate_restricted_recovery_request(request, check_live=False)
    execution = request["recovery_execution"]
    output.mkdir(parents=True)

    write_create_only(output / "authorization_request_unsigned.json", request)
    write_create_only(output / "readonly_source_qualification.json", request["immutable_source_audit"])
    write_create_only(output / "scientific_invariance.json", request["scientific_invariance"])
    write_create_only(output / "historical_qualification_correction.json", historical_correction)
    write_create_only(output / "protected_revalidation_start_snapshot.json", start_snapshot)
    write_create_only(output / "protected_revalidation_end_snapshot.json", end_snapshot)
    write_create_only(output / "protected_workspace_results.json", protected)
    write_create_only(output / "i4_original_artifact_review.json", i4_audit)
    write_create_only(output / "regression_anomaly_disposition.json", anomaly_disposition)
    write_create_only(output / "skip_review.json", skip_review)
    shutil.copyfile(
        supplemental_validation_path, output / "supplemental_validation_receipt.json"
    )
    write_create_only(
        output / "identity_bridge.json",
        {
            "original_identity": request["original_identity"],
            "recovery_execution_identity": execution,
            "identities_are_equal": False,
            "bridge_sha256": canonical_sha256(
                {
                    "original": request["original_identity"],
                    "recovery": execution["recovery_execution_identity_sha256"],
                }
            ),
        },
    )
    write_create_only(
        output / "producer_consumer_matrix.json",
        {
            "version": "1.0.0",
            "external_original_cells": request["external_committed_cells"],
            "planned_recovery_cells": execution["cell_specs"],
            "deduplication": request["source_mapping_and_deduplication"],
            "statistics_consumer": {
                "input": "future ablation_recovery_handoff.json",
                "formal_controller_source": "three external original committed cells",
                "direct_legacy_single_root_consumption": False,
                "explicit_path_resolver_available_after_ablation_handoff": True,
                "executed_in_this_acceptance": False,
                "separate_authorization_required": True,
            },
        },
    )
    commands = {
        "version": "1.1.0",
        "working_directory": str(executor_checkout),
        "prepare_and_validate_authorized_now": True,
        "execute_authorized_now": False,
        "prepare": [
            args.python_executable,
            str(ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"),
            "--action",
            "prepare",
            "--original-request-path",
            str(ORIGINAL_REQUEST_PATH),
            "--original-project-grant-path",
            str(ORIGINAL_PROJECT_GRANT_PATH),
            "--original-run-root",
            str(ORIGINAL_RUN_ROOT),
            "--recovery-execution-id",
            args.recovery_execution_id,
            "--recovery-root",
            args.recovery_root,
            "--executor-checkout",
            str(executor_checkout),
            "--executor-commit",
            args.executor_commit,
            "--python-executable",
            args.python_executable,
            "--output-path",
            str(output / "authorization_request_unsigned.json"),
        ],
        "validate": [
            args.python_executable,
            str(ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"),
            "--action",
            "validate",
            "--output-path",
            str(output / "authorization_request_unsigned.json"),
        ],
        "future_execute_plan_requires_new_grant": [
            [
                args.python_executable,
                str(ROOT / "scripts/run_typed_model_cache_restricted_recovery.py"),
                "--authorization-request-path",
                str(output / "authorization_request_unsigned.json"),
                "--recovery-grant-path",
                "<not_issued_exact_recovery_grant.json>",
                "--cell-id",
                cell_id,
                "--check",
                "execute",
            ]
            for cell_id in execution["allowed_cell_ids"]
        ],
        "scientific_commands": execution["command_plan"]["commands"],
        "automatic_next_phase": False,
        "test_dependency_overlay_applies_to_future_commands": False,
    }
    write_create_only(output / "complete_command_plan.json", commands)

    for label, (junit_path, receipt_path) in report_paths.items():
        shutil.copyfile(junit_path, output / f"{label}_tests.junit.xml")
        shutil.copyfile(receipt_path, output / f"{label}_tests.receipt.json")
    shutil.copyfile(public_summary_source, output / "public_recovery_acceptance.json")

    origin = subprocess.check_output(
        [
            "git",
            "-C",
            str(executor_checkout),
            "rev-parse",
            "origin/codex/g14r20-i5-restricted-recovery",
        ],
        text=True,
    ).strip()
    acceptance = {
        "status": "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION",
        "reviewed_at": now_iso(),
        "literature_cutoff": "2026-06-21",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": "g14r20_i5_a_restricted_recovery_acceptance_20260914",
        "policy_version": "tmc_review_policy_v3_20260621",
        "evidence_level": "E2_ARTIFACT_AUDITED_SYNTHETIC_RECOVERY_EXECUTION",
        "paper_verdict": "Unverifiable",
        "base_commit": "398799177b0189353b10ce90d53ffba13bfd1d92",
        "executor_commit": head,
        "executor_git_tree": tree,
        "origin_branch_commit": origin,
        "executor_matches_request": head == args.executor_commit == execution["executor_commit"],
        "commit_pushed": origin == head,
        "clean_code_scope": not bool(status),
        "historical_start_evidence": historical_correction["historical_start_evidence"],
        "historical_protection_verdict": "UNVERIFIED",
        "historical_acceptance_requires_central_judgment": True,
        "old_qualification_stop_retained": True,
        "current_revalidation_protection": protected["status"],
        "i4_original_review": i4_audit["status"],
        "test_results": test_results,
        "supplemental_validation": {
            "status": supplemental_validation["status"],
            "checks": [check["name"] for check in supplemental_validation["checks"]],
        },
        "skip_review": skip_review,
        "historical_regression_disposition": {
            "status": anomaly_disposition["status"],
            "row_count": anomaly_disposition["exception_or_skip_row_count"],
            "classification_counts": anomaly_disposition["classification_counts"],
        },
        "public_acceptance": public_summary,
        "source_qualification": "pass_retention_only",
        "scientific_parameters_unchanged": True,
        "formal_performance_conclusion": False,
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    if not all(
        (
            acceptance["executor_matches_request"],
            acceptance["commit_pushed"],
            acceptance["clean_code_scope"],
            protected["status"] == "pass",
            anomaly_disposition["status"] == "pass",
            skip_review["status"] == "pass",
        )
    ):
        raise ValueError("final commit/push/protection/test acceptance gate failed")
    write_create_only(output / "acceptance_summary.json", acceptance)

    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "integrity_manifest.json":
            manifest_rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "version": "1.0.0",
        "artifact_run_id": acceptance["artifact_run_id"],
        "executor_commit": head,
        "origin_branch_commit": origin,
        "file_count": len(manifest_rows),
        "files": manifest_rows,
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_create_only(output / "integrity_manifest.json", manifest)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
