from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eviction_aware_recovery_validation import _preview_inputs
from scripts.run_shared_cache_recovery_coupling import run_branch
from src.runtime.eviction_aware_recovery import (
    RecoveryDecisionInputs,
    eviction_aware_recovery_decision,
    original_simple_threshold_decision,
    two_step_lookahead_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment/eviction_aware_recovery_v1.json"


def _inputs(**overrides) -> RecoveryDecisionInputs:
    values = {
        "action4_legal": True,
        "recovery_preview_feasible": True,
        "current_missing_object_ids": ("base:b0", "adapter:b0.a0"),
        "legal_victim_ids": ("adapter:b1.a0", "base:b1"),
        "near_term_dependency_ids": ("base:b1", "adapter:b1.a0"),
        "near_term_missing_before_ids": (),
        "near_term_missing_after_ids": ("base:b1", "adapter:b1.a0"),
        "transfer_bytes_by_object": {
            "base:b0": 96 * 1_048_576,
            "adapter:b0.a0": 8 * 1_048_576,
            "base:b1": 128 * 1_048_576,
            "adapter:b1.a0": 8 * 1_048_576,
        },
        "state_package_bytes": 2185,
        "rerun_input_bytes": 192757,
        "effective_link_mbps": 100.0,
        "positive_transfer_fixed_latency_seconds": 0.02,
        "state_restore_overhead_seconds": 0.006151,
        "prefix_recompute_seconds": 11.035304,
        "future_reload_cost_scale": 1.0,
    }
    values.update(overrides)
    return RecoveryDecisionInputs(**values)


def test_eviction_aware_counts_unique_objects_and_matches_two_step() -> None:
    inputs = _inputs()
    proposed = eviction_aware_recovery_decision(inputs, opt_in_enabled=True)
    lookahead = two_step_lookahead_decision(inputs)
    assert proposed["selected_first_action"] == 0
    assert proposed["induced_reload_object_ids"] == ["adapter:b1.a0", "base:b1"]
    assert proposed["induced_reload_bytes"] == 136 * 1_048_576
    assert proposed["restart_incremental_seconds"] == lookahead["restart_incremental_seconds"]
    assert proposed["recovery_incremental_seconds"] == lookahead["recovery_incremental_seconds"]
    assert proposed["selected_first_action"] == lookahead["selected_first_action"]


def test_shared_base_is_not_charged_twice() -> None:
    inputs = _inputs(
        current_missing_object_ids=("adapter:b0.a0",),
        legal_victim_ids=("adapter:b0.a1",),
        near_term_dependency_ids=("base:b0", "adapter:b0.a1", "base:b0"),
        near_term_missing_after_ids=("adapter:b0.a1", "adapter:b0.a1"),
        transfer_bytes_by_object={
            "base:b0": 96 * 1_048_576,
            "adapter:b0.a0": 8 * 1_048_576,
            "adapter:b0.a1": 8 * 1_048_576,
        },
    )
    result = eviction_aware_recovery_decision(inputs, opt_in_enabled=True)
    assert result["induced_reload_object_ids"] == ["adapter:b0.a1"]
    assert result["induced_reload_bytes"] == 8 * 1_048_576


def test_missing_input_and_unexplained_reload_fail_conservatively() -> None:
    missing = eviction_aware_recovery_decision(
        _inputs(effective_link_mbps=None), opt_in_enabled=True
    )
    assert missing["selected_first_action"] == 0
    assert missing["conservative_fallback"] is True
    unexplained = eviction_aware_recovery_decision(
        _inputs(legal_victim_ids=()), opt_in_enabled=True
    )
    assert unexplained["selected_first_action"] == 0
    assert unexplained["fallback_reason"] == "missing_invalid_or_unexplained_reload_input"


def test_opt_in_disabled_preserves_original_threshold_choice() -> None:
    inputs = _inputs()
    baseline = original_simple_threshold_decision(inputs)
    disabled = eviction_aware_recovery_decision(inputs, opt_in_enabled=False)
    assert baseline["selected_first_action"] == 4
    assert disabled["selected_first_action"] == baseline["selected_first_action"]
    assert disabled["opt_in_enabled"] is False


def test_native_preview_uses_legal_dependency_safe_victims(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    points = {row["instance_id"]: row for row in config["design_points"]}
    full_inputs, full_preview = _preview_inputs(
        points["d04_full_bundle_reload_b0_to_b1"], config, tmp_path
    )
    assert list(full_inputs.legal_victim_ids) == ["adapter:b1.a0", "base:b1"]
    assert full_preview["native_legal_victim_plan"]["sufficient"] is True
    shared_inputs, _ = _preview_inputs(
        points["d05_shared_base_adapter_only_reload"], config, tmp_path
    )
    assert list(shared_inputs.legal_victim_ids) == ["adapter:b0.a1"]
    assert list(shared_inputs.near_term_missing_after_ids) == ["adapter:b0.a1"]


def test_native_action_state_transitions_remain_complete_and_legal(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    instance = next(
        row
        for row in config["design_points"]
        if row["instance_id"] == "d04_full_bundle_reload_b0_to_b1"
    )
    branches = {
        action: run_branch(instance, config, tmp_path / f"action_{action}", action)
        for action in (0, 4)
    }
    assert all(branch["feasible"] for branch in branches.values())
    assert all(branch["summary"]["completed_node_count"] == 2 for branch in branches.values())
    assert branches[4]["steps"][0]["cache_event"]["evicted_object_ids"] == [
        "adapter:b1.a0",
        "base:b1",
    ]
