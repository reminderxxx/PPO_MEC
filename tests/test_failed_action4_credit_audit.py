from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import audit_calibrated_workflow_failed_action4_credit as audit


ROOT = Path(__file__).resolve().parents[1]


def test_predeclared_witnesses_cover_benefit_rollback_and_externality() -> None:
    _, config, _ = audit._load_protocol()
    rows = audit._synthetic_witnesses(config)
    by_id = {row["case_id"]: row for row in rows}

    assert len(rows) == 6
    assert all(row["cap_predicate_triggered"] for row in rows)
    assert not any(row["public_state_exposes_actual_mbps"] for row in rows)

    reuse = by_id["committed_warm_then_reuse"]
    assert reuse["cache_event"]["committed"] is True
    assert reuse["cache_event"]["admitted"] == ["adapter:alpr"]
    assert reuse["future"][0]["transition"]["service_completed"] is True
    assert reuse["future"][0]["transition"]["model_transfer_bytes"] == 0

    rollback = by_id["contact_budget_rollback"]
    assert rollback["cache_event"]["committed"] is False
    assert rollback["cache_event"]["reason"] == "contact_budget_exceeded"
    assert rollback["target_cache_before"] == rollback["target_cache_after"]

    eviction = by_id["unused_warm_with_victim_reload"]
    assert eviction["cache_event"]["victims"] == ["adapter:helmet_shared"]
    assert eviction["future"][-1]["transition"]["model_transfer_bytes"] > 0

    unused = by_id["committed_warm_never_reused"]
    assert unused["cache_event"]["committed"] is True
    assert all(
        step["transition"]["current_rsu_id"] != "rsu_1" for step in unused["future"]
    )

    noop = by_id["target_ready_noop_lru_touch"]
    assert noop["target_cache_before"]["residents"] == noop["target_cache_after"]["residents"]
    assert noop["target_cache_before"]["last_used"] != noop["target_cache_after"]["last_used"]

    rejection = by_id["capacity_rejection_no_change"]
    assert rejection["cache_event"]["reason"] == "dependency_bundle_exceeds_total_capacity"
    assert rejection["target_cache_before"] == rejection["target_cache_after"]


def test_full_existing_ledger_replay_classifies_every_trigger(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    audit.run(
        ROOT
        / "artifacts/benchmarks/calibrated_workflow_value_normalization_ab_20261009_v2",
        output,
    )
    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))

    assert summary["existing_replay"]["behavior_row_count"] == 5293
    assert summary["existing_replay"]["episode_count"] == 600
    assert summary["existing_replay"]["replay_validation_mismatch_count"] == 0
    assert summary["existing_replay"]["trigger_count"] == 1143
    assert summary["committed_new_admission_count"] == 122
    assert summary["committed_new_admission_future_reuse_count"] == 118
    assert summary["decision"] == "CAP_PREDICATE_INSUFFICIENT_NOT_READY"


def test_audit_has_no_agent_checkpoint_or_optimizer_path() -> None:
    source = Path(inspect.getsourcefile(audit) or "").read_text(encoding="utf-8")
    assert "_build_agent" not in source
    assert ".learn(" not in source
    assert "optimizer.step" not in source
    assert "torch.load" not in source
