from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.audit_calibrated_workflow_service_reward import (
    _bootstrap_audit,
    _score_case,
    _transition_invariance,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from scripts.run_calibrated_workflow_service_reward_alignment import _selection_score
from src.envs.core.calibrated_continuous_workflow_env import (
    SERVICE_ALIGNED_REWARD_PROFILE,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    experiment = json.loads(
        (ROOT / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json").read_text()
    )
    config, _ = _load_experiment_config(ROOT / experiment["base_config"])
    config["objective"].setdefault("reward_profiles", {})[
        SERVICE_ALIGNED_REWARD_PROFILE
    ] = experiment["service_aligned_reward"]
    manifest = json.loads((ROOT / experiment["workload_manifest"]).read_text())
    return experiment, config, manifest


def test_candidate_reward_orders_service_cases() -> None:
    _, config, _ = _inputs()
    common = {
        "total_nodes": 5,
        "deadline_seconds": 60.0,
        "transfer_bytes": 2 * 1024**3,
        "service_operation_seconds": 30.0,
        "recompute_seconds": 0.0,
        "service_failures": 0,
        "external_truncation": False,
    }
    on_time = _score_case(
        dict(common, case="on_time", completed_nodes=5, workflow_completed=True, elapsed_seconds=50.0),
        config["objective"],
    )
    late = _score_case(
        dict(common, case="late", completed_nodes=5, workflow_completed=True, elapsed_seconds=80.0),
        config["objective"],
    )
    failed = _score_case(
        dict(common, case="failed", completed_nodes=4, workflow_completed=False, elapsed_seconds=80.0),
        config["objective"],
    )
    assert on_time["candidate_return"] > late["candidate_return"] > failed["candidate_return"]


def test_reward_profile_does_not_change_transitions() -> None:
    _, config, manifest = _inputs()
    result = _transition_invariance(config, manifest["instances"][0])
    assert result["same_state_action_transition_trace"] is True


def test_external_truncation_bootstraps() -> None:
    assert _bootstrap_audit()["external_truncation_bootstraps"] is True


def test_checkpoint_selection_is_reward_value_independent() -> None:
    row = {
        "on_time_workflow_completion_rate": 1.0,
        "workflow_completion_rate": 1.0,
        "unfinished_after_deadline_rate": 0.0,
        "service_failure_rate": 0.0,
        "handoff_failure_rate": 0.0,
        "completed_sample_elapsed_seconds": 10.0,
        "total_transfer_mb": 1.0,
        "recompute_seconds": 0.0,
        "invalid_prepare_attempts": 0,
        "reward": -999.0,
    }
    changed = deepcopy(row)
    changed["reward"] = 999.0
    assert _selection_score([row]) == _selection_score([changed])


def test_frozen_identity_keeps_interface_and_auxiliary() -> None:
    experiment, config, _ = _inputs()
    assert experiment["interface_profile"] == "calibrated_workflow_interface_v2"
    assert experiment["hierarchical_action_contract"] == "independent_heads_executed_env_v2"
    assert experiment["claim_boundary"]["auxiliary_target_changed"] is False
    assert experiment["training"]["theoretical_total_step_cap"] == 82944
    assert config["interface_profile"] == experiment["interface_profile"]
