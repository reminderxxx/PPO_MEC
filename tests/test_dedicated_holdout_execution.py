from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluators.dedicated_holdout_execution import (
    ALL_AGENTS,
    CAPACITIES,
    HOLDOUT_COMMAND_PACKAGE_VERSION,
    HOLDOUT_LEDGER_VERSION,
    HOLDOUT_REQUEST_VERSION,
    LEARNED_AGENTS,
    PRIMARY_METRICS,
    SEEDS,
    HoldoutExecutionError,
    canonical_sha256,
    file_sha256,
    validate_command_package,
    validate_opening_receipt,
    validate_unsigned_request,
    verify_checkpoint_bytes,
)


def package() -> dict:
    value = {
        "command_package_version": HOLDOUT_COMMAND_PACKAGE_VERSION,
        "executor_checkout": "/tmp/executor",
        "executor_commit": "abc",
        "output_root": "/tmp/output",
        "phases": ["scientific", "statistics", "publication", "integrity"],
        "scientific_matrix": {
            "agents": list(ALL_AGENTS), "learned_agents": list(LEARNED_AGENTS),
            "seeds": list(SEEDS), "capacities": list(CAPACITIES), "outer_windows": 12,
            "workflows": 3, "scientific_child_count": 3, "rows_per_child": 2700,
            "total_expected_rows": 8100, "checkpoint_count": 150,
            "primary_metrics": list(PRIMARY_METRICS), "holm_family_size": 84,
        },
        "commands": {"scientific": [["child", str(i)] for i in range(3)], "statistics": ["stats"]},
        "automatic_retry_count": 0,
    }
    value["command_package_sha256"] = canonical_sha256(value)
    return value


def request(tmp_path: Path, command_package: dict) -> dict:
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}\n")
    value = {
        "request_version": HOLDOUT_REQUEST_VERSION,
        "status": "READY_FOR_AUTHORIZATION_REVIEW",
        "grant_signed": False, "execution_authorized": False,
        "holdout_opened": False, "holdout_consumed_permanently": False,
        "command_package_sha256": command_package["command_package_sha256"],
        "frozen_inputs": [{"path": str(frozen), "size_bytes": frozen.stat().st_size,
                           "sha256": file_sha256(frozen)}],
        "failure_boundary": {
            "before_atomic_open": "not_consumed; no scientific child may start",
            "at_or_after_atomic_open": "permanently_consumed",
            "child_failure": "terminal_failure_no_retry_no_resume_no_reopen",
            "partial_output": "retained_in_staging_and_permanently_consumed",
            "success": "permanently_consumed",
        },
    }
    value["request_sha256"] = canonical_sha256(value)
    return value


def test_exact_scientific_matrix_and_zero_retry() -> None:
    audit = validate_command_package(package())
    assert audit == {"status": "pass", "command_count": 4, "acceptance": False}
    bad = package()
    bad["scientific_matrix"]["holm_family_size"] = 83
    bad["command_package_sha256"] = canonical_sha256(
        {key: value for key, value in bad.items() if key != "command_package_sha256"}
    )
    with pytest.raises(HoldoutExecutionError, match="matrix drift"):
        validate_command_package(bad)


def test_unsigned_request_is_bound_but_not_authorized(tmp_path: Path) -> None:
    command_package = package()
    unsigned = request(tmp_path, command_package)
    assert validate_unsigned_request(unsigned, command_package)["status"] == "pass"
    unsigned["execution_authorized"] = True
    unsigned["request_sha256"] = canonical_sha256(
        {key: value for key, value in unsigned.items() if key != "request_sha256"}
    )
    with pytest.raises(HoldoutExecutionError, match="falsely asserts"):
        validate_unsigned_request(unsigned, command_package)


def test_checkpoint_byte_audit_covers_exact_150_coordinates(tmp_path: Path) -> None:
    models = []
    for capacity in CAPACITIES:
        for agent in LEARNED_AGENTS:
            for seed in SEEDS:
                target = tmp_path / f"{capacity}-{agent}-{seed}.pt"
                target.write_bytes(f"{capacity}:{agent}:{seed}".encode())
                models.append({
                    "capacity_label": capacity, "agent": agent, "seed": seed,
                    "checkpoint_path": str(target), "size_bytes": target.stat().st_size,
                    "checkpoint_sha256": file_sha256(target),
                })
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"models": models}))
    audit = verify_checkpoint_bytes(source)
    assert audit["status"] == "pass"
    assert audit["actual_scope"]["models"] == 150
    assert audit["actual_scope"]["learned_agents"] == list(LEARNED_AGENTS)


def test_opening_receipt_is_already_permanently_consumed(tmp_path: Path) -> None:
    receipt = {
        "ledger_version": HOLDOUT_LEDGER_VERSION,
        "event": "opened", "consumed_permanently": True,
        "request_sha256": "r", "command_package_sha256": "c",
        "checkpoint_byte_validation": {"models": 150, "status": "pass"},
    }
    target = tmp_path / "opening.json"
    target.write_text(json.dumps(receipt))
    assert validate_opening_receipt(target, request_sha256="r", command_package_sha256="c") == receipt
    receipt["consumed_permanently"] = False
    target.write_text(json.dumps(receipt))
    with pytest.raises(HoldoutExecutionError, match="invalid"):
        validate_opening_receipt(target, request_sha256="r", command_package_sha256="c")


def test_acceptance_package_cannot_name_sealed_split() -> None:
    value = package()
    value["scientific_matrix"] = {"profile": "tiny"}
    value["acceptance_non_holdout"] = True
    value["commands"]["scientific"][0].append("sealed_holdout")
    value["command_package_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "command_package_sha256"}
    )
    with pytest.raises(HoldoutExecutionError, match="holdout reference"):
        validate_command_package(value, acceptance=True)
