from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent
from scripts.run_calibrated_workflow_prepare_balance_ablation import (
    CANDIDATE_METHOD,
    _run_extended_evaluation_episode,
)
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
ABLATION_CONFIG = ROOT / "configs/experiment/calibrated_workflow_prepare_balance_ablation_v1.json"
MANIFEST = ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"


def _load_base_and_instance() -> tuple[dict, dict]:
    config, _ = _load_experiment_config(BASE_CONFIG)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    instance = next(row for row in manifest["instances"] if row["split"] == "dev")
    return config, instance


def test_candidate_changes_only_auxiliary_coefficient() -> None:
    config, _ = _load_base_and_instance()
    original = _build_agent("sa_ghmappo", 7, config)
    candidate = _build_agent(
        "sa_ghmappo", 7, config, agent_overrides={"auxiliary_coef": 0.0}
    )
    assert original._auxiliary_coef == 0.1
    assert candidate._auxiliary_coef == 0.0
    assert original._hierarchical_action_contract == candidate._hierarchical_action_contract
    assert original._executed_action_ppo_only is candidate._executed_action_ppo_only is True
    assert original._event_prepare_margin_boost == candidate._event_prepare_margin_boost
    assert (
        original._event_logit_sharpening_final_scale
        == candidate._event_logit_sharpening_final_scale
    )
    assert original._temporal_consistency_coef == candidate._temporal_consistency_coef
    original_state = original._network.state_dict()
    candidate_state = candidate._network.state_dict()
    assert original_state.keys() == candidate_state.keys()
    assert all(
        torch.equal(original_state[key], candidate_state[key])
        for key in original_state
    )


def test_frozen_ablation_does_not_override_environment_or_reward() -> None:
    experiment = json.loads(ABLATION_CONFIG.read_text(encoding="utf-8"))
    assert experiment["frozen_factor"] == {
        "name": "auxiliary_loss",
        "implementation_parameter": "auxiliary_coef",
        "baseline_value": 0.1,
        "candidate_value": 0.0,
        "other_algorithm_parameters_changed": False,
    }
    assert experiment["claim_boundary"]["reward_changed"] is False
    assert experiment["claim_boundary"]["environment_changed"] is False
    assert experiment["base_config"] == str(BASE_CONFIG.relative_to(ROOT))
    assert experiment["workload_manifest"] == str(MANIFEST.relative_to(ROOT))


def test_old_builder_call_and_executed_action_log_prob_remain_compatible() -> None:
    config, instance = _load_base_and_instance()
    agent = _build_agent("sa_ghmappo", 7, config)
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    action, action_info = agent.act(observation, info)
    assert 0 <= action < 5
    assert action_info["hierarchical_action_contract"] == "independent_heads_executed_env_v2"
    assert action_info["executed_action_ppo_only"] is True
    assert action_info["log_prob"] == action_info["env_action_log_prob"]


def test_extended_behavior_records_readiness_and_preserves_environment_metrics() -> None:
    config, instance = _load_base_and_instance()
    agent = _build_agent(
        "sa_ghmappo", 7, config, agent_overrides={"auxiliary_coef": 0.0}
    )
    agent._deterministic_action = True
    row, ledger = _run_extended_evaluation_episode(
        method=CANDIDATE_METHOD,
        seed=7,
        config=config,
        instance=instance,
        step_cap=24,
        agent=agent,
    )
    assert ledger
    assert all(
        "current_bundle_ready" in item
        and "target_bundle_ready" in item
        and "target_prepare_feasible" in item
        and "service_safe_prepare_state" in item
        for item in ledger
    )
    assert row["action4_attempts"] == sum(
        int(item["executed_action"] == 4) for item in ledger
    )
    assert row["safe_prepare_attempts"] + row["invalid_prepare_attempts"] == row[
        "action4_attempts"
    ]
    assert (
        row["feasible_target_prepare_attempts"]
        + row["infeasible_target_prepare_attempts"]
        == row["action4_attempts"]
    )
    summary = CalibratedContinuousWorkflowEnv(config, instance)
    _, _ = summary.reset()
    assert summary.config == config
