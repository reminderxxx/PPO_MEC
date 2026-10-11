from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.launch_calibrated_workflow_value_normalization_ab import (
    DEFAULT_INTERPRETER,
    ROOT_DIR,
    SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
    _build_probe_plan,
    _build_scientific_plan,
    _frozen_interpreter,
    _terminal_decision,
    _validate_scientific_completion_receipt,
)


AUTHORIZATION = (
    "configs/experiment/calibrated_workflow_value_normalization_ab_authorized_v2.json"
)


def test_scientific_plan_freezes_interpreter_and_run_identity(tmp_path: Path) -> None:
    output_root = tmp_path / "calibrated_workflow_value_normalization_ab_20261009_v2"
    supervisor_root = tmp_path / "launcher"
    args = Namespace(
        interpreter=DEFAULT_INTERPRETER,
        authorization_config=AUTHORIZATION,
        output_root=str(output_root),
        supervisor_root=str(supervisor_root),
        expected_commit="a" * 40,
    )

    plan = _build_scientific_plan(args)

    assert plan["interpreter"] == DEFAULT_INTERPRETER
    assert plan["child_argv"][0] == DEFAULT_INTERPRETER
    assert plan["run_id"] == output_root.name
    assert plan["output_root"] == str(output_root.resolve())
    assert plan["entry_receipt"] == str(output_root.resolve() / "runner_entered.json")
    assert plan["wall_clock_cap_seconds"] == 7200.0
    assert plan["automatic_retry"] is False


def test_scientific_plan_rejects_output_identity_drift(tmp_path: Path) -> None:
    args = Namespace(
        interpreter=DEFAULT_INTERPRETER,
        authorization_config=AUTHORIZATION,
        output_root=str(tmp_path / "wrong_run_id"),
        supervisor_root=str(tmp_path / "launcher"),
        expected_commit="a" * 40,
    )
    with pytest.raises(ValueError, match="output_run_id"):
        _build_scientific_plan(args)


@pytest.mark.parametrize(
    ("probe_case", "expected_exit"), [("success", 0), ("nonzero", 7)]
)
def test_probe_plan_records_expected_terminal_outcome(
    tmp_path: Path, probe_case: str, expected_exit: int
) -> None:
    args = Namespace(
        interpreter=DEFAULT_INTERPRETER,
        supervisor_root=str(tmp_path / probe_case),
        expected_commit="b" * 40,
        probe_case=probe_case,
    )
    plan = _build_probe_plan(args)
    assert plan["kind"] == "host_acceptance"
    assert plan["expected_exit_code"] == expected_exit
    assert plan["child_argv"][0] == DEFAULT_INTERPRETER
    assert plan["automatic_retry"] is False


def test_frozen_interpreter_does_not_accept_realpath_substitution() -> None:
    assert _frozen_interpreter(DEFAULT_INTERPRETER) == DEFAULT_INTERPRETER
    with pytest.raises(ValueError, match="identity drift"):
        _frozen_interpreter(str((ROOT_DIR / ".venv/bin/python").resolve()))


def test_scientific_receipt_contract_accepts_only_explicit_true_success() -> None:
    receipt = {
        "schema_version": SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "scientific_execution_complete": True,
    }
    result = _validate_scientific_completion_receipt(
        receipt,
        expected_schema=SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
    )
    assert result == {"complete": True, "errors": []}


@pytest.mark.parametrize(
    ("receipt", "expected_error"),
    [
        ({"status": "complete"}, "scientific_execution_complete_is_not_literal_true"),
        (
            {"status": "complete", "scientific_execution_complete": "true"},
            "scientific_execution_complete_is_not_literal_true",
        ),
        (
            {"status": "failed", "scientific_execution_complete": True},
            "status_is_not_complete",
        ),
        (
            {
                "schema_version": "wrong_schema",
                "status": "complete",
                "scientific_execution_complete": True,
            },
            "schema_version_mismatch",
        ),
        (None, "missing_or_non_object_receipt"),
    ],
)
def test_scientific_receipt_contract_fails_closed(receipt, expected_error) -> None:
    result = _validate_scientific_completion_receipt(
        receipt,
        expected_schema=SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
    )
    assert result["complete"] is False
    assert expected_error in result["errors"]


@pytest.mark.parametrize(
    ("return_code", "timed_out", "receipt", "case_name"),
    [
        (9, False, {"status": "complete", "scientific_execution_complete": True}, "nonzero"),
        (0, True, {"status": "complete", "scientific_execution_complete": True}, "timeout"),
        (-15, False, {"status": "complete", "scientific_execution_complete": True}, "interrupted"),
        (0, False, None, "partial_output_missing_receipt"),
        (0, False, {"status": "complete"}, "partial_receipt"),
    ],
)
def test_terminal_decision_rejects_nonterminal_or_partial_scientific_children(
    return_code, timed_out, receipt, case_name
) -> None:
    result = _terminal_decision(
        kind="scientific",
        return_code=return_code,
        expected_exit_code=0,
        timed_out=timed_out,
        child_receipt=receipt,
    )
    assert result["passed"] is False, case_name
    assert result["classification"] == "CHILD_EXECUTION_FAILURE_NO_RETRY"


def test_terminal_decision_accepts_explicit_scientific_success() -> None:
    result = _terminal_decision(
        kind="scientific",
        return_code=0,
        expected_exit_code=0,
        timed_out=False,
        child_receipt={
            "schema_version": SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "scientific_execution_complete": True,
        },
        expected_receipt_schema=SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
    )
    assert result["passed"] is True
    assert result["classification"] == "SCIENTIFIC_CHILD_COMPLETE"
