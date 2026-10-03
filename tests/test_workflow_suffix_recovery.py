from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.runtime.workflow_suffix_recovery import (
    EXPECTED_EDGES,
    EXPECTED_NODES,
    EXPECTED_WORKFLOW_ID,
    StateValidationError,
    build_n1_prompt,
    canonical_json_bytes,
    loaded_adapter_names,
    seal_state_payload,
    sha256_bytes,
    sha256_file,
    validate_state_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter_state_loaded_status_20261003.json"
OLD_RECEIPT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/environments/adapter_state_acceptance_py39_v1/"
    "acceptance_run_20261003_v1/terminal_receipt.json"
)


def identity() -> dict:
    return {
        "base_weight_sha256": "base-hash",
        "adapter_name": "adapter_a",
        "adapter_weight_sha256": "adapter-hash",
        "processor_sha256": "processor-hash",
    }


def payload(description: str = "a line graph") -> dict:
    return {
        "workflow_id": EXPECTED_WORKFLOW_ID,
        "nodes": EXPECTED_NODES,
        "edges": EXPECTED_EDGES,
        "completed_nodes": ["n0"],
        "remaining_nodes": ["n1"],
        "next_node": "n1",
        "n0_output": {"raw_text": description, "token_ids": [1, 2, 3]},
        "suffix_input_material": description,
        "generation": {"n0": {"do_sample": False}, "n1": {"do_sample": False}},
        "identity": identity(),
        "rng_state": {"required": False, "reason": "greedy nodes and explicit boundary"},
        "tensor_state": {"required": False, "reason": "n1 consumes explicit text"},
    }


def reseal(changed_payload: dict) -> dict:
    return seal_state_payload(changed_payload)


def test_valid_state_and_two_legal_descriptions_produce_distinct_suffix_inputs() -> None:
    first = validate_state_envelope(reseal(payload("a line graph")), expected_identity=identity())
    second = validate_state_envelope(reseal(payload("a road intersection")), expected_identity=identity())
    first_prompt = build_n1_prompt("Observed: {n0_output}", first["suffix_input_material"])
    second_prompt = build_n1_prompt("Observed: {n0_output}", second["suffix_input_material"])
    assert first_prompt != second_prompt
    assert sha256_bytes(canonical_json_bytes(first_prompt)) != sha256_bytes(canonical_json_bytes(second_prompt))


def test_missing_intermediate_is_rejected() -> None:
    changed = payload()
    del changed["n0_output"]
    with pytest.raises(StateValidationError, match="n0_output is required"):
        validate_state_envelope(reseal(changed), expected_identity=identity())


def test_tampered_bytes_with_stale_hash_are_rejected() -> None:
    envelope = reseal(payload("a line graph"))
    envelope["payload"]["n0_output"]["raw_text"] = "tampered description"
    with pytest.raises(StateValidationError, match="payload_sha256 mismatch"):
        validate_state_envelope(envelope, expected_identity=identity())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("next_node", "n0", "next_node mismatch"),
        ("completed_nodes", [], "completed_nodes mismatch"),
    ],
)
def test_control_conflicts_are_rejected(field: str, value: object, message: str) -> None:
    changed = payload()
    changed[field] = value
    with pytest.raises(StateValidationError, match=message):
        validate_state_envelope(reseal(changed), expected_identity=identity())


def test_model_identity_conflict_is_rejected() -> None:
    changed = payload()
    changed["identity"]["base_weight_sha256"] = "wrong-base"
    with pytest.raises(StateValidationError, match="identity mismatch"):
        validate_state_envelope(reseal(changed), expected_identity=identity())


def test_existing_receipt_uses_available_adapters_and_negative_variants_fail() -> None:
    fixture = json.loads(FIXTURE.read_text())
    status = fixture["loaded_status"]
    assert loaded_adapter_names(status) == fixture["expected"]

    missing = copy.deepcopy(status)
    del missing["available_adapters"]
    with pytest.raises(StateValidationError, match="available_adapters is missing"):
        loaded_adapter_names(missing)

    conflict = copy.deepcopy(status)
    conflict["loaded_adapters"] = ["adapter_a"]
    with pytest.raises(StateValidationError, match="conflicts"):
        loaded_adapter_names(conflict)


@pytest.mark.skipif(not OLD_RECEIPT.exists(), reason="historical ignored receipt is not present")
def test_fixture_matches_existing_preserved_receipt() -> None:
    fixture = json.loads(FIXTURE.read_text())
    actual = json.loads(OLD_RECEIPT.read_text())
    assert sha256_file(OLD_RECEIPT) == fixture["source_receipt_sha256"]
    assert actual["primary"]["loaded_status"]["available_adapters"] == fixture["expected"]
    assert "loaded_adapters" not in actual["primary"]["loaded_status"]
