from __future__ import annotations

import json
from pathlib import Path

from scripts.run_symmetric_recovery_cost_validation import (
    _decision_inputs,
    _execute_cache_path,
    _load_config,
    _score_realized_branch,
)
from src.runtime.symmetric_recovery_cost import (
    LifecycleEventEstimate,
    PathLifecycleEstimate,
    SymmetricRecoveryDecisionInputs,
    eviction_aware_recovery_decision,
    original_simple_threshold_decision,
    two_step_lookahead_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment/eviction_aware_recovery_corrected_v2.json"


def _event(*, safe: bool = True) -> LifecycleEventEstimate:
    return LifecycleEventEstimate(
        event_id="current",
        required_adapter_id="b0.a0",
        pre_resident_object_ids=("base:b1", "adapter:b1.a0"),
        victim_object_ids=("adapter:b1.a0", "base:b1"),
        admitted_object_ids=("base:b0", "adapter:b0.a0"),
        post_resident_object_ids=("base:b0", "adapter:b0.a0"),
        transfer_bytes_by_object={"base:b0": 96 * 1_048_576, "adapter:b0.a0": 8 * 1_048_576},
        transaction_status="committed",
        dependency_safe=safe,
    )


def _inputs(*, safe: bool = True) -> SymmetricRecoveryDecisionInputs:
    common = (_event(safe=safe),)
    return SymmetricRecoveryDecisionInputs(
        action4_legal=True,
        restart_path=PathLifecycleEstimate(
            action_id=0,
            events=common,
            dynamic_transfer_bytes=192757,
            state_overhead_seconds=0.0,
            recompute_seconds=11.035304,
            successful_service_seconds=0.06,
        ),
        recovery_path=PathLifecycleEstimate(
            action_id=4,
            events=common,
            dynamic_transfer_bytes=2185,
            state_overhead_seconds=0.006151,
            recompute_seconds=0.0,
            successful_service_seconds=0.06,
        ),
        effective_link_mbps=100.0,
        positive_transfer_fixed_latency_seconds=0.02,
    )


def test_common_lifecycle_is_charged_once_to_each_path_and_cancels() -> None:
    inputs = _inputs()
    simple = original_simple_threshold_decision(inputs)
    proposed = eviction_aware_recovery_decision(inputs)
    lookahead = two_step_lookahead_decision(inputs)
    assert simple["selected_first_action"] == 4
    assert proposed["selected_first_action"] == lookahead["selected_first_action"] == 4
    restart = proposed["restart_path_estimate"]
    recovery = proposed["recovery_path_estimate"]
    assert restart["model_transfer_bytes"] == recovery["model_transfer_bytes"] == 104 * 1_048_576
    assert len(restart["event_costs"]) == len(recovery["event_costs"]) == 1


def test_illegal_or_dependency_unsafe_path_falls_back() -> None:
    result = eviction_aware_recovery_decision(_inputs(safe=False))
    assert result["selected_first_action"] == 0
    assert result["conservative_fallback"] is True
    assert result["fallback_reason"] == "missing_invalid_or_illegal_path_input"


def test_original_matrix_full_reload_paths_are_isolated_and_symmetric(tmp_path: Path) -> None:
    config = _load_config(CONFIG_PATH)
    instance = next(
        row
        for row in config["design_points"]
        if row["instance_id"] == "d04_full_bundle_reload_b0_to_b1"
    )
    previews = {
        action: _execute_cache_path(
            instance,
            config,
            tmp_path,
            action_id=action,
            purpose="decision_preview",
        )
        for action in (0, 4)
    }
    branches = {
        action: _execute_cache_path(
            instance,
            config,
            tmp_path,
            action_id=action,
            purpose="realized_score",
        )
        for action in (0, 4)
    }
    assert len({row["environment_identity"] for row in (*previews.values(), *branches.values())}) == 4
    for action in (0, 4):
        events = branches[action]["events"]
        assert [list(row["estimate"].victim_object_ids) for row in events] == [
            ["adapter:b1.a0", "base:b1"],
            ["adapter:b0.a0", "base:b0"],
        ]
        assert sum(
            sum(row["estimate"].transfer_bytes_by_object.values()) for row in events
        ) == 240 * 1_048_576
        assert branches[action]["terminal_resident_object_ids"] == [
            "base:b1",
            "adapter:b1.a0",
        ]
    inputs = _decision_inputs(previews, config=config, instance=instance)
    decision = eviction_aware_recovery_decision(inputs)
    assert decision["realized_score_used"] is False
    summaries = {
        action: _score_realized_branch(
            branches[action], action_id=action, config=config, instance=instance
        )
        for action in (0, 4)
    }
    assert summaries[0]["completed_node_count"] == summaries[4]["completed_node_count"] == 2
    assert summaries[0]["service_failure_count"] == summaries[4]["service_failure_count"] == 0
    assert summaries[0]["model_transfer_bytes"] == summaries[4]["model_transfer_bytes"]
    assert summaries[0]["modeled_completion_seconds"] > summaries[4]["modeled_completion_seconds"]


def test_inherited_original_design_is_byte_identified_and_complete() -> None:
    config = _load_config(CONFIG_PATH)
    assert config["actual_design_points"] == len(config["design_points"]) == 12
    assert config["inherits_frozen_design"]["sha256"] == (
        "6f0b9a5cdca4c3e72373e1d8bd595ce5fd637cbd236a57dbb14f88e960671b10"
    )
    assert {row["instance_id"] for row in config["design_points"]} >= {
        "d10_recompute_estimate_half",
        "d11_recompute_estimate_double",
    }
