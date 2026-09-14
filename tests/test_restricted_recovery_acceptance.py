"""Fail-closed acceptance evidence tests for G14R20-I5-A."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.build_restricted_recovery_acceptance_artifacts import (
    build_historical_anomaly_disposition,
    build_protected_snapshot,
    compare_protected_snapshots,
    junit_cases,
    validate_protected_snapshot,
    validate_skip_review,
    validate_supplemental_validation_receipt,
    validate_test_receipt,
)
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256


def write_junit(path: Path, statuses: list[tuple[str, str, str]]) -> None:
    suite = ET.Element("testsuite")
    suite.set("tests", str(len(statuses)))
    suite.set("failures", str(sum(status == "failure" for _, status, _ in statuses)))
    suite.set("errors", str(sum(status == "error" for _, status, _ in statuses)))
    suite.set("skipped", str(sum(status == "skipped" for _, status, _ in statuses)))
    for name, status, message in statuses:
        case = ET.SubElement(suite, "testcase", classname="tests.test_gate", name=name, time="0")
        if status != "passed":
            terminal = ET.SubElement(case, status)
            terminal.set("message", message)
            terminal.text = message
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_receipt(
    path: Path,
    *,
    label: str,
    junit: Path,
    checkout: Path,
    commit: str = "a" * 40,
    tree: str = "b" * 40,
) -> dict[str, object]:
    summary, _ = junit_cases(junit)
    receipt = {
        "restricted_recovery_test_receipt_version": "1.0.0",
        "label": label,
        "executor_checkout": str(checkout),
        "executor_commit": commit,
        "executor_git_tree": tree,
        "return_code": 0,
        "checkout_clean_before": True,
        "checkout_clean_after": True,
        "junit_absolute_path": str(junit.resolve()),
        "junit_sha256": file_sha256(junit),
        "junit_summary": summary,
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def validate(tmp_path: Path, statuses: list[tuple[str, str, str]]) -> None:
    junit = tmp_path / "report.xml"
    receipt = tmp_path / "receipt.json"
    write_junit(junit, statuses)
    write_receipt(receipt, label="targeted", junit=junit, checkout=tmp_path)
    validate_test_receipt(
        label="targeted",
        junit_path=junit,
        receipt_path=receipt,
        expected_checkout=tmp_path,
        expected_commit="a" * 40,
        expected_tree="b" * 40,
    )


def test_passing_commit_bound_junit_is_accepted(tmp_path: Path) -> None:
    validate(tmp_path, [("test_pass", "passed", "")])


@pytest.mark.parametrize("status", ["failure", "error"])
def test_failure_and_error_junit_are_rejected(tmp_path: Path, status: str) -> None:
    with pytest.raises(ValueError, match="failures or errors"):
        validate(tmp_path, [("test_bad", status, "boom")])


def test_empty_junit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty test set"):
        validate(tmp_path, [])


def test_wrong_commit_and_mismatched_report_are_rejected(tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    receipt = tmp_path / "receipt.json"
    write_junit(junit, [("test_pass", "passed", "")])
    value = write_receipt(receipt, label="targeted", junit=junit, checkout=tmp_path)
    value["executor_commit"] = "c" * 40
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="executor_commit mismatch"):
        validate_test_receipt(
            label="targeted",
            junit_path=junit,
            receipt_path=receipt,
            expected_checkout=tmp_path,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
        )
    write_receipt(receipt, label="targeted", junit=junit, checkout=tmp_path)
    write_junit(junit, [("test_changed", "passed", "")])
    with pytest.raises(ValueError, match="junit_sha256 mismatch"):
        validate_test_receipt(
            label="targeted",
            junit_path=junit,
            receipt_path=receipt,
            expected_checkout=tmp_path,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
        )


def test_missing_receipt_is_rejected(tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    write_junit(junit, [("test_pass", "passed", "")])
    with pytest.raises(ValueError, match="missing or non-regular"):
        validate_test_receipt(
            label="targeted",
            junit_path=junit,
            receipt_path=tmp_path / "missing.json",
            expected_checkout=tmp_path,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
        )


def test_skip_requires_explanation_and_same_scope_substitute(tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    receipt = tmp_path / "receipt.json"
    write_junit(junit, [("test_public", "skipped", "conditional")])
    write_receipt(receipt, label="targeted", junit=junit, checkout=tmp_path)
    _, cases = validate_test_receipt(
        label="targeted",
        junit_path=junit,
        receipt_path=receipt,
        expected_checkout=tmp_path,
        expected_commit="a" * 40,
        expected_tree="b" * 40,
    )
    with pytest.raises(ValueError, match="exactly match"):
        validate_skip_review(
            {"skip_review_version": "1.0.0", "rows": []}, {"targeted": cases}
        )
    review = {
        "skip_review_version": "1.0.0",
        "rows": [
            {
                "report_label": "targeted",
                "nodeid": "tests.test_gate::test_public",
                "reason": "conditional path",
                "same_scope_substitute_verified": True,
                "substitute_evidence": "separate enabled public run",
            }
        ],
    }
    assert validate_skip_review(review, {"targeted": cases})["status"] == "pass"


def test_explicit_snapshot_detects_content_drift(tmp_path: Path) -> None:
    for relative in (
        "scripts/train_sa_ghmappo_real_sample.py",
        "src/agents/sa_ghmappo_agent.py",
        "src/agents/sa_ghmappo_core.py",
        "src/encoders/fusion_encoder.py",
        "src/evaluators/real_eval_support.py",
        "tests/test_algo_pool_contract.py",
        "tests/test_checkpoint_compat.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    start = build_protected_snapshot(
        tmp_path,
        snapshot_id="start",
        capture_kind="g14r20_i5_a_revalidation_start",
        capture_source="test capture",
    )
    validate_protected_snapshot(start, expected_capture_kind="g14r20_i5_a_revalidation_start")
    (tmp_path / "src/agents/sa_ghmappo_agent.py").write_text("drift", encoding="utf-8")
    end = build_protected_snapshot(
        tmp_path,
        snapshot_id="end",
        capture_kind="g14r20_i5_a_revalidation_end",
        capture_source="test end",
    )
    assert compare_protected_snapshots(start, end)["status"] == "fail"
    forged = deepcopy(start)
    forged["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        validate_protected_snapshot(forged)


def test_historical_146_anomalies_and_skip_are_individually_closed(tmp_path: Path) -> None:
    historical = tmp_path / "historical.xml"
    statuses = [
        (f"test_crypto_error_{index}", "error", "No module named 'cryptography'")
        for index in range(84)
    ]
    statuses += [
        (f"test_crypto_failure_{index}", "failure", "No module named 'cryptography'")
        for index in range(60)
    ]
    statuses += [
        (f"test_ps_{index}", "failure", "Operation not permitted: 'ps'")
        for index in range(2)
    ]
    statuses += [
        (
            "test_public",
            "skipped",
            "run from final clean commit with G14R20_I5_PUBLIC_ACCEPTANCE=1",
        )
    ]
    write_junit(historical, statuses)
    current = {
        "full": [
            {
                "nodeid": f"tests.test_gate::{name}",
                "status": "passed",
                "message": "",
                "details": "",
            }
            for name, _, _ in statuses
        ]
    }
    result = build_historical_anomaly_disposition(historical, current)
    assert result["exception_or_skip_row_count"] == 147
    assert result["status"] == "pass"


def test_supplemental_validation_receipt_is_commit_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    check_names = (
        "smoke",
        "compile_import",
        "target_imports",
        "diff_check",
        "clean_scope",
        "protected_scope_diff_check",
    )
    receipt = {
        "restricted_recovery_validation_receipt_version": "1.0.0",
        "executor_checkout": str(tmp_path),
        "executor_commit": "a" * 40,
        "executor_git_tree": "b" * 40,
        "executor_identity_unchanged": True,
        "baseline_commit": "398799177b0189353b10ce90d53ffba13bfd1d92",
        "status": "pass",
        "checks": [{"name": name, "status": "pass"} for name in check_names],
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    assert (
        validate_supplemental_validation_receipt(
            receipt,
            expected_checkout=tmp_path,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
        )["status"]
        == "pass"
    )
    receipt["checks"][0]["status"] = "fail"
    with pytest.raises(ValueError, match="failed check"):
        validate_supplemental_validation_receipt(
            receipt,
            expected_checkout=tmp_path,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
        )
