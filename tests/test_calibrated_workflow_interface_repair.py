from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from src.agents.registry import build_agent
from src.encoders import FlatSemanticEncoder
from src.encoders.dag_graph_encoder import DAGGraphEncoder
from src.encoders.rsu_state_encoder import RSUStateEncoder
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
)
MANIFEST_PATH = (
    ROOT
    / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
)


def _config_and_instance() -> tuple[dict, dict]:
    config, _ = _load_experiment_config(CONFIG_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return config, deepcopy(manifest["instances"][0])


def _empty_current_cache(env: CalibratedContinuousWorkflowEnv) -> None:
    current_rsu = env._current_rsu_id()
    env.caches[current_rsu].residents = []
    env.caches[current_rsu].last_used = {}


def test_byte_capacity_and_public_mechanism_features_are_consumed() -> None:
    config, instance = _config_and_instance()
    env = CalibratedContinuousWorkflowEnv(config, instance)
    _, info = env.reset()
    state = deepcopy(info["semantic_state"])
    assert state["interface_profile"] == "calibrated_workflow_interface_v2"
    assert state["calibrated_context"]["cache_capacity_unit"] == "bytes"

    flat = FlatSemanticEncoder()
    baseline_actor = flat._build_feature_tensor(state)
    baseline_critic = flat._build_centralized_feature_tensor(state)
    changed_bytes = deepcopy(state)
    changed_bytes["rsus"][0]["cache_used_bytes"] = 0
    changed_actor = flat._build_feature_tensor(changed_bytes)
    changed_critic = flat._build_centralized_feature_tensor(changed_bytes)
    assert not torch.equal(baseline_actor, changed_actor)
    assert not torch.equal(baseline_critic, changed_critic)
    assert 0.0 <= float(baseline_actor[10]) <= 1.0

    changed_state_size = deepcopy(state)
    changed_state_size["current_workflow_node"]["state_bytes"] *= 1000
    assert not torch.equal(
        baseline_actor, flat._build_feature_tensor(changed_state_size)
    )

    rsu_encoder = RSUStateEncoder()
    baseline_rsu, _ = rsu_encoder._build_rsu_feature_tensor(
        state, list(state["rsus"])
    )
    missing_base = deepcopy(state)
    required_base = missing_base["current_workflow_node"]["required_base_model"]
    for rsu in missing_base["rsus"]:
        rsu["typed_resident_object_ids"] = [
            item
            for item in rsu["typed_resident_object_ids"]
            if item != required_base
        ]
    missing_rsu, _ = rsu_encoder._build_rsu_feature_tensor(
        missing_base, list(missing_base["rsus"])
    )
    assert not torch.equal(baseline_rsu, missing_rsu)

    dag_encoder = DAGGraphEncoder()
    baseline_dag, _ = dag_encoder._build_node_feature_tensor(
        state, list(state["workflow"]["nodes"])
    )
    missing_dag, _ = dag_encoder._build_node_feature_tensor(
        missing_base, list(missing_base["workflow"]["nodes"])
    )
    assert not torch.equal(baseline_dag, missing_dag)


def test_deterministic_multihead_aggregation_uses_raw_head_argmax() -> None:
    agent = build_agent(
        "mappo",
        deterministic_action=True,
        hierarchical_action_contract="independent_heads_executed_env_v2",
        executed_action_ppo_only=True,
        env_action_ppo_enabled=True,
        env_action_ppo_coef=1.0,
    )
    output = {
        "slow_logits": torch.log(torch.tensor([0.32597, 0.50744, 0.16659])),
        "fast_logits": torch.log(torch.tensor([0.856507, 0.143493])),
        "event_logits": torch.log(torch.tensor([0.633515, 0.366485])),
    }
    env_scores = agent._hierarchical_env_action_scores(output)
    assert int(torch.argmax(env_scores).item()) == 4
    actions, _, _, _, projection = agent._sample_actions(
        output, deterministic=True, action_mask=[True] * 5
    )
    assert actions == {"slow": 1, "fast": 0, "event": 0}
    assert projection["raw_head_actions"] == actions
    assert projection["raw_head_actions_source"] == "independent_head_argmax"
    assert projection["projected_env_action"] == 0
    assert projection["projection_applied"] is False


def test_executed_action_log_prob_is_the_rollout_log_prob() -> None:
    config, instance = _config_and_instance()
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    agent = build_agent(
        "mappo",
        random_seed=7,
        hierarchical_action_contract="independent_heads_executed_env_v2",
        executed_action_ppo_only=True,
        env_action_ppo_enabled=True,
        env_action_ppo_coef=1.0,
    )
    _, action_info = agent.act(observation, info)
    assert action_info["executed_action_ppo_only"] is True
    assert action_info["log_prob"] == action_info["env_action_log_prob"]
    assert len(action_info["env_action_probs"]) == 5
    assert abs(sum(action_info["env_action_probs"]) - 1.0) < 1e-5


def test_action4_only_prepares_target_and_never_commits_state_on_service_failure() -> None:
    config, instance = _config_and_instance()
    instance["rsu_sequence"][:3] = ["rsu_0", "rsu_0", "rsu_1"]
    env = CalibratedContinuousWorkflowEnv(config, instance)
    env.reset()
    _empty_current_cache(env)
    before_node = env.node_index
    _, _, _, _, first = env.step(4)
    assert first["transition"]["service_completed"] is False
    assert first["transition"]["migration_success"] is False
    assert env.node_index == before_node
    assert env.step_index == 1
    assert env.prepared_state == {}
    _, _, _, _, second = env.step(4)
    assert second["transition"]["service_completed"] is False
    assert second["transition"]["migration_success"] is False
    assert second["transition"]["model_transfer_bytes"] == 0
    assert env.node_index == before_node
    assert env.prepared_state == {}


def test_current_fill_and_valid_prepare_have_distinct_effects() -> None:
    config, instance = _config_and_instance()
    fill_env = CalibratedContinuousWorkflowEnv(config, deepcopy(instance))
    fill_env.reset()
    _empty_current_cache(fill_env)
    _, _, _, _, fill_info = fill_env.step(0)
    assert fill_info["transition"]["service_completed"] is True
    assert fill_env.node_index == 1

    prepare_env = CalibratedContinuousWorkflowEnv(config, deepcopy(instance))
    prepare_env.reset()
    target = prepare_env._predicted_handoff_target()
    assert target is not None
    prepare_env.caches[target].residents = []
    prepare_env.caches[target].last_used = {}
    _, _, _, _, prepare_info = prepare_env.step(4)
    assert prepare_info["transition"]["service_completed"] is True
    assert prepare_info["transition"]["migration_success"] is True
    assert target in prepare_env.prepared_state


def test_failed_service_advances_mobility_but_not_workflow_node() -> None:
    config, instance = _config_and_instance()
    instance["rsu_sequence"][:2] = ["rsu_0", "rsu_1"]
    env = CalibratedContinuousWorkflowEnv(config, instance)
    env.reset()
    _empty_current_cache(env)
    node_before = env.node_index
    rsu_before = env._current_rsu_id()
    env.step(3)
    assert env.node_index == node_before
    assert env.step_index == 1
    assert env._current_rsu_id() != rsu_before
