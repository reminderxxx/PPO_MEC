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


def test_pending_independent_gate_preflight_does_no_scientific_work() -> None:
    receipt = preflight(ROOT / DEFAULT_PROTOCOL, DEFAULT_CONTROL_ROOT)
    assert receipt["status"] == "preflight_passed"
    assert receipt["execution_authorized"] is False
    assert receipt["authorization_state"] == (
        "awaiting_independent_interface_time_contract_gate"
    )
    assert receipt["independent_gate_status"] == "pending_a_review"
    assert receipt["independent_gate_verification"] == {
        "status": "pending_a_review",
        "artifacts_verified": False,
    }
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


def test_pending_independent_gate_refuses_scientific_execution(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="not authorized by a PASS independent gate"):
        execute(ROOT / DEFAULT_PROTOCOL, DEFAULT_CONTROL_ROOT, output, ["test"])
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
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing immutable identity fields"):
        preflight(path, DEFAULT_CONTROL_ROOT)
