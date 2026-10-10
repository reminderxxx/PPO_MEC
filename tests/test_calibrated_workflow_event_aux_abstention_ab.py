"""Guard the abstention candidate protocol before independent authorization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_calibrated_workflow_event_aux_abstention_ab import (
    DEFAULT_CONTROL_ROOT,
    DEFAULT_PROTOCOL,
    execute,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_verified_independent_gate_preflight_does_no_scientific_work() -> None:
    receipt = preflight(ROOT / DEFAULT_PROTOCOL, DEFAULT_CONTROL_ROOT)
    assert receipt["status"] == "preflight_passed"
    assert receipt["execution_authorized"] is True
    assert receipt["authorization_state"] == (
        "authorized_after_independent_interface_time_contract_pass"
    )
    assert receipt["independent_gate_status"] == "PASS"
    assert receipt["independent_gate_verification"]["status"] == "PASS"
    assert receipt["independent_gate_verification"]["artifacts_verified"] is True
    assert receipt["scientific_steps"] == receipt["new_evaluation_episodes"] == 0
    assert receipt["control_rows"] == {
        "selected": 400,
        "update96": 400,
        "rules": 40,
    }
    assert receipt["network_identity"] == {
        "control_parameter_count": 165512,
        "candidate_parameter_count": 165512,
        "state_dict_keys": 60,
        "initialized_tensors_equal": True,
    }


def test_pending_protocol_copy_refuses_scientific_execution(
    tmp_path: Path,
) -> None:
    protocol = json.loads((ROOT / DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
    protocol["execution_authorized"] = False
    protocol["authorization_state"] = (
        "awaiting_independent_interface_time_contract_gate"
    )
    protocol["independent_interface_time_contract_gate"]["status"] = (
        "pending_a_review"
    )
    path = tmp_path / "pending_protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="not authorized by a PASS independent gate"):
        execute(path, DEFAULT_CONTROL_ROOT, output, ["test"])
    assert not output.exists()


def test_text_only_pass_cannot_bypass_gate_artifact_verification(
    tmp_path: Path,
) -> None:
    protocol = json.loads((ROOT / DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
    protocol["execution_authorized"] = True
    protocol["authorization_state"] = (
        "authorized_after_independent_interface_time_contract_pass"
    )
    protocol["independent_interface_time_contract_gate"]["status"] = "PASS"
    for field in (
        "commit",
        "tree",
        "report_path",
        "report_sha256",
        "manifest_path",
        "manifest_sha256",
        "time_contract_receipt_path",
        "time_contract_receipt_sha256",
        "abstention_receipt_path",
        "abstention_receipt_sha256",
    ):
        protocol["independent_interface_time_contract_gate"][field] = None
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing immutable identity fields"):
        preflight(path, DEFAULT_CONTROL_ROOT)
