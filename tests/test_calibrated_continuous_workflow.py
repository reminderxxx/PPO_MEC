from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    ImmediateCostRule,
    TwoStepCostRule,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/experiment/calibrated_continuous_workflow_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _instance(*, state_bytes: int = 2195, target_ready: bool = True) -> dict:
    config = _config()
    bundle = list(config["adapter_to_bundle"]["alpr"])
    nodes = [
        {
            "node_id": "n0",
            "node_name": "R0",
            "required_base_model": "base:family_a",
            "required_adapter": "alpr",
            "input_size": 192757,
            "output_size": state_bytes,
            "input_bytes": 192757,
            "state_bytes": state_bytes,
            "compute_seconds": 3.807234,
            "predecessors": [],
            "successors": ["n1"],
        },
        {
            "node_id": "n1",
            "node_name": "R1",
            "required_base_model": "base:family_a",
            "required_adapter": "alpr",
            "input_size": 192757,
            "output_size": state_bytes,
            "input_bytes": 192757,
            "state_bytes": state_bytes,
            "compute_seconds": 3.807234,
            "predecessors": ["n0"],
            "successors": [],
        },
    ]
    return {
        "design_id": "test_00",
        "split": "test",
        "window_id": "w0",
        "workflow_id": "wf0",
        "trace_features": {"handoff_pressure": "high", "mean_speed_proxy": 20.0},
        "workflow_features": {"topology_class": "simple"},
        "factors": {"sharing": "shared_base", "capacity": "ample", "state_scale": 1, "initial_target": "test"},
        "rsu_ids": ["rsu_0", "rsu_1"],
        "rsu_sequence": ["rsu_0", "rsu_1"],
        "cache_capacity_bytes": int(config["capacity_bytes"]["shared_ample"]),
        "initial_residents": {"rsu_0": bundle, "rsu_1": bundle if target_ready else []},
        "nodes": nodes,
        "edges": [["n0", "n1"]],
        "execution_order": ["n0", "n1"],
        "deadline_seconds": 60.0,
        "max_steps": 4,
        "source_classes": {"measured": [], "trace_derived": [], "literature": [], "artificial": []},
    }


def test_action4_prepares_state_and_avoids_prefix_recompute() -> None:
    config = _config()
    prepared = CalibratedContinuousWorkflowEnv(config, _instance())
    _, _, terminated, _, info = prepared.step(4)
    assert not terminated
    assert info["transition"]["migration_success"] is True
    _, _, terminated, _, info = prepared.step(3)
    assert terminated
    assert info["transition"]["state_ready"] is True
    assert prepared.summary()["recompute_seconds"] == 0.0

    restart = CalibratedContinuousWorkflowEnv(config, _instance())
    restart.step(3)
    _, _, terminated, _, info = restart.step(3)
    assert terminated
    assert info["transition"]["state_ready"] is False
    assert restart.summary()["recompute_seconds"] == pytest.approx(
        config["measured_time_seconds"]["prefix_recompute_reference"]
    )


def test_dependency_safe_replacement_keeps_shared_base() -> None:
    config = _config()
    instance = _instance()
    instance["cache_capacity_bytes"] = int(config["capacity_bytes"]["shared_tight"])
    instance["initial_residents"]["rsu_1"] = list(config["adapter_to_bundle"]["helmet_shared"])
    env = CalibratedContinuousWorkflowEnv(config, instance)
    event = env._admit_bundle("rsu_1", "alpr")
    assert event["committed"] is True
    assert event["victims"] == ["adapter:helmet_shared"]
    assert set(env.caches["rsu_1"].residents) == {"base:family_a", "adapter:alpr"}


def test_failed_migration_rolls_back_atomic_cache_transition() -> None:
    config = _config()
    instance = _instance(target_ready=False)
    instance["cache_capacity_bytes"] = int(config["capacity_bytes"]["shared_tight"])
    instance["initial_residents"]["rsu_1"] = list(config["adapter_to_bundle"]["helmet_distinct"])
    before = deepcopy(instance["initial_residents"]["rsu_1"])
    env = CalibratedContinuousWorkflowEnv(config, instance)
    _, _, _, _, info = env.step(4)
    assert info["transition"]["migration_success"] is False
    assert env.caches["rsu_1"].residents == before
    assert env.summary()["cache_evictions"] == 0


def test_two_step_rule_is_valid_and_side_effect_free() -> None:
    env = CalibratedContinuousWorkflowEnv(_config(), _instance())
    rule = TwoStepCostRule()
    before = env.summary()
    action = rule.select_action(env)
    assert action in env.valid_actions()
    assert env.summary() == before


def test_decision_preview_uses_estimated_link_without_exposing_actual_link() -> None:
    config = _config()
    instance = _instance(target_ready=False)
    instance["link_profile"] = {
        "actual_mbps": 200.0,
        "estimated_mbps": 1000.0,
        "error_class": "optimistic",
    }
    env = CalibratedContinuousWorkflowEnv(config, instance)
    semantic = env._info()["semantic_state"]
    assert semantic["calibrated_context"]["link"]["estimated_mbps"] == 1000.0
    assert "actual_mbps" not in semantic["calibrated_context"]["link"]
    execution = env.clone()
    preview = env.clone_for_decision_model()
    execution.step(2)
    preview.step(2)
    assert execution.summary()["modeled_completion_seconds"] > preview.summary()[
        "modeled_completion_seconds"
    ]


def test_failed_current_service_does_not_commit_staged_migration_state() -> None:
    config = _config()
    instance = _instance(target_ready=True)
    instance["initial_residents"]["rsu_0"] = []
    env = CalibratedContinuousWorkflowEnv(config, instance)
    _, _, _, _, info = env.step(4)
    assert info["transition"]["service_completed"] is False
    assert info["transition"]["migration_success"] is False
    assert env.prepared_state == {}
    assert env.summary()["state_transfer_bytes"] == 0
    assert env.summary()["migration_successes"] == 0


def test_reward_components_sum_to_step_reward() -> None:
    env = CalibratedContinuousWorkflowEnv(_config(), _instance())
    _, reward, _, _, info = env.step(3)
    components = info["transition"]["reward_components"]
    assert reward == pytest.approx(sum(components.values()))
    assert env.summary()["reward"] == pytest.approx(
        sum(env.summary()[key] for key in components)
    )


@pytest.mark.parametrize("rule", [ImmediateCostRule(), TwoStepCostRule()])
def test_information_matched_rules_return_legal_actions(rule: object) -> None:
    env = CalibratedContinuousWorkflowEnv(_config(), _instance())
    action = rule.select_action(env)
    assert action in env.valid_actions()
