from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_remaining_workflow_decision_value import (
    action_reachability,
    build_native_ledger,
)
from scripts.measure_local_model_and_state_costs import (
    application_state_payload,
    measure_state_once,
)
from scripts.synthesize_remaining_workflow_decision_value import synthesize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/analysis/native_typed_cache_replacement_20260930_v1"
OUTPUT = ROOT / "artifacts/analysis/remaining_workflow_decision_value_audit_20260930_v1"


def test_native_ledger_preserves_equal_completion_and_rejection_guard() -> None:
    ledger = build_native_ledger(
        SOURCE / "four_config_old_request_log.jsonl",
        SOURCE / "four_config_candidate_request_log.jsonl",
    )
    rows = ledger["same_completion_comparison"]["rows"]
    assert len(rows) == 4
    assert {row["completed_request_count"] for row in rows} == {72}
    assert rows[0]["config_id"] == "blocked__sharing_on"
    assert all(row["interpretation"].startswith("rejected service") for row in ledger["old_low_transfer_interpretation"])


def test_native_action_contract_exposes_prepare_but_not_migrate_or_forward() -> None:
    audit = action_reachability()
    assert audit["reachable_migration_modes"] == ["keep", "prepare"]
    assert audit["explicit_state_migrate"]["reachable"] is False
    assert audit["future_adapter_direct_prepare"]["reachable"] is False
    assert audit["continue_old_rsu_and_forward_result"]["reachable"] is False


def test_explicit_application_state_round_trip(tmp_path: Path) -> None:
    payload = application_state_payload(
        {"repository": "test/model", "revision": "1", "weight_sha256": "a" * 64, "adapter": None}
    )
    row = measure_state_once(payload, tmp_path, 0)
    assert row["serialized_bytes"] > 0
    assert row["canonical_payload_equal"] is True
    assert row["continuation_key_equal"] is True


def test_paired_witness_stays_within_budget_and_two_step_is_sufficient() -> None:
    output = json.loads((OUTPUT / "paired_witness_results.json").read_text())
    assert output["episode_count"] == 16
    assert output["episode_count"] <= output["episode_budget"]["maximum_episodes"]
    assert sum(row["decision_changed"] for row in output["decision_changes"]) == 6
    assert output["two_step_matches_full_remaining_on_all_designed_instances"] is True


def test_synthesis_does_not_treat_missing_native_state_payload_as_zero() -> None:
    result = synthesize(
        json.loads((OUTPUT / "native_four_config_ledger.json").read_text()),
        json.loads((OUTPUT / "local_model_state_measurements.json").read_text()),
        json.loads((OUTPUT / "paired_witness_results.json").read_text()),
        json.loads((OUTPUT / "native_action_reachability.json").read_text()),
    )
    rows = result["sensitivity_analysis"]["rows"]
    assert len(rows) == 2
    assert {row["avoided_native_model_transfer_bytes"] for row in rows} == {104 * 1024 * 1024}
    assert all(
        row["native_state_transfer_availability"].startswith("unavailable_")
        for row in rows
    )
    assert result["answers"]["support_algorithm_change"].startswith("no;")
