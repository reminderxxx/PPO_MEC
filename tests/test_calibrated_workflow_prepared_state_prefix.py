"""Public prepared-state prefix contract for the opt-in causal profile."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from scripts.diagnose_cscwd_sa_behavior import _public_state_hash
from src.encoders.calibrated_workflow_features import prepared_state_prefix_features
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    PREPARED_STATE_PREFIX_PROFILE,
)
from src.envs.core.causal_rsu_predictor import fit_predictor


ROOT = Path(__file__).resolve().parents[1]
OLD_PROFILE = "calibrated_workflow_interface_v3_prefix_only"
METHODS = ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl")


def _fixture() -> tuple[dict, dict]:
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    manifest = json.loads((
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
    ).read_text())
    ordered = sorted(manifest["instances"], key=lambda row: str(row["design_id"]))
    model = fit_predictor([row["rsu_sequence"] for row in ordered if row["split"] == "train"])
    instance = deepcopy(next(row for row in ordered if row["design_id"] == "regression_08"))
    instance["causal_predictor_model"] = model
    return config, instance


def _env(config: dict, instance: dict, profile: str, actions: tuple[int, ...] = ()) -> CalibratedContinuousWorkflowEnv:
    env = CalibratedContinuousWorkflowEnv({**config, "interface_profile": profile}, instance)
    for action in actions:
        env.step(action)
    return env


def test_original_same_public_state_alias_is_separated_without_changing_cost() -> None:
    config, instance = _fixture()
    # B run regression_08, SA seeds 17 and 43, before their common action 0.
    ready_history, missing_history = (3, 3, 4, 4), (3, 3, 0, 0)
    old_ready = _env(config, instance, OLD_PROFILE, ready_history)
    old_missing = _env(config, instance, OLD_PROFILE, missing_history)
    assert old_ready._observation().tolist() == old_missing._observation().tolist()
    assert old_ready._info() == old_missing._info()
    assert _public_state_hash(old_ready._observation(), old_ready._info()) == (
        "86206ccfda557199bb67a41c0d3a7bcfc3b98a3817aa7de3875e71e6c971d62d"
    )

    new_ready = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE, ready_history)
    new_missing = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE, missing_history)
    ready_info, missing_info = new_ready._info(), new_missing._info()
    assert ready_info["action_mask"] == missing_info["action_mask"]
    assert ready_info["semantic_state"] != missing_info["semantic_state"]
    assert prepared_state_prefix_features(ready_info["semantic_state"])[:2] == [1.0, 1.0]
    assert prepared_state_prefix_features(missing_info["semantic_state"])[:2] == [0.0, 0.0]
    assert new_ready._info() == new_ready._info()  # stable repeated serialization
    assert new_missing._info() == new_missing._info()

    for old, new, expected_ready, expected_recompute in (
        (old_ready, new_ready, True, 0.0),
        (old_missing, new_missing, False, 30.21069466716467),
    ):
        old_observation, old_reward, _, _, old_info = old.step(0)
        new_observation, new_reward, _, _, new_info = new.step(0)
        assert new_info["transition"]["state_ready"] is expected_ready
        assert new_info["transition"]["recompute_seconds"] == pytest.approx(expected_recompute)
        assert old_observation.tolist() == new_observation.tolist()
        assert old_reward == new_reward
        assert old_info["transition"] == new_info["transition"]
        assert old.metrics == new.metrics and old.prepared_state == new.prepared_state


def test_identical_legal_action_sequence_keeps_physical_reward_and_cost_state() -> None:
    config, instance = _fixture()
    old = _env(config, instance, OLD_PROFILE)
    new = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    for desired in (4, 2, 1, 3, 0, 4):
        legal = old.valid_actions()
        assert legal == new.valid_actions()
        action = desired if desired in legal else legal[0]
        old_observation, old_reward, old_done, old_truncated, old_info = old.step(action)
        new_observation, new_reward, new_done, new_truncated, new_info = new.step(action)
        assert old_observation.tolist() == new_observation.tolist()
        assert (old_reward, old_done, old_truncated) == (new_reward, new_done, new_truncated)
        assert old_info["transition"] == new_info["transition"]
        assert old.metrics == new.metrics
        assert old.caches == new.caches and old.prepared_state == new.prepared_state
        assert old.completed == new.completed and old.clock_seconds == new.clock_seconds
        if old_done or old_truncated:
            break


def test_existence_validity_staleness_and_unknown_target() -> None:
    config, instance = _fixture()
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    public = env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]
    assert public["current"] == {
        "known": True, "exists": False, "valid": False,
        "missing_completed_count": 0, "missing_completed_fraction": 0.0,
    }
    current = env._current_rsu_id()
    target = env._predicted_handoff_target()
    assert target is not None and target != current
    env.prepared_state[target] = {"workflow_id": instance["workflow_id"], "completed_node_ids": []}
    target_ready = env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["predicted_target"]
    assert target_ready["known"] and target_ready["exists"] and target_ready["valid"]
    env.prepared_state[current] = {"workflow_id": instance["workflow_id"], "completed_node_ids": []}
    assert env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"]["valid"]
    env.completed = [env.execution_order[0]]
    status = env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"]
    assert status["exists"] and not status["valid"]
    assert status["missing_completed_count"] == 1
    assert status["missing_completed_fraction"] == pytest.approx(1 / len(env.execution_order))
    env.prepared_state[current]["completed_node_ids"] = list(env.completed)
    assert env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"]["valid"]
    env.prepared_state[current]["workflow_id"] = "wrong_workflow"
    assert not env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"]["valid"]

    progressed = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    progressed.prepared_state[progressed._current_rsu_id()] = {
        "workflow_id": instance["workflow_id"], "completed_node_ids": []
    }
    progressed.step(3)
    progressed_status = progressed._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"]
    assert progressed_status["exists"] and not progressed_status["valid"]
    assert progressed_status["missing_completed_count"] == 1

    unknown = deepcopy(instance)
    unknown["causal_predictor_model"] = fit_predictor([["rsu_1", "rsu_1"]])
    unknown_env = _env(config, unknown, PREPARED_STATE_PREFIX_PROFILE)
    assert unknown_env._predicted_handoff_target() is None
    target_status = unknown_env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["predicted_target"]
    assert target_status == {"known": False, "exists": False, "valid": False,
                             "missing_completed_count": 0, "missing_completed_fraction": 0.0}
    assert prepared_state_prefix_features(unknown_env._info()["semantic_state"])[-1] == 0.0


def test_failed_prepare_target_change_and_future_tamper() -> None:
    config, instance = _fixture()
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    current = env._current_rsu_id()
    target = env._predicted_handoff_target()
    assert target and target != current
    env.caches[current].residents = []
    env.caches[current].last_used = {}
    env.step(4)
    assert env.prepared_state == {}  # failed current service cannot commit state

    base = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    altered = deepcopy(instance)
    altered["rsu_sequence"][1:] = ["rsu_2"] * (len(altered["rsu_sequence"]) - 1)
    altered["link_profile"]["actual_mbps"] = 1.0
    changed = _env(config, altered, PREPARED_STATE_PREFIX_PROFILE)
    assert base._observation().tolist() == changed._observation().tolist()
    assert base._info() == changed._info()

    other = deepcopy(instance)
    alternative = next(rsu for rsu in instance["rsu_ids"] if rsu not in {current, target})
    other["causal_predictor_model"] = fit_predictor([[current, alternative, alternative]] * 2)
    other_env = _env(config, other, PREPARED_STATE_PREFIX_PROFILE)
    assert other_env._predicted_handoff_target() == alternative
    assert other_env._predicted_handoff_target() != target
    base.prepared_state[target] = {"workflow_id": instance["workflow_id"], "completed_node_ids": []}
    other_env.prepared_state = deepcopy(base.prepared_state)
    original_target = base._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["predicted_target"]
    changed_target = other_env._info()["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["predicted_target"]
    assert original_target["valid"] and changed_target["known"] and not changed_target["exists"]


@pytest.mark.parametrize("method", METHODS)
def test_all_learned_encoders_consume_new_fields_and_checkpoint_identity(method: str, tmp_path: Path) -> None:
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE, (3, 3, 4, 4))
    observation, info = env._observation(), env._info()
    altered = deepcopy(info)
    altered["semantic_state"]["calibrated_context"]["prepared_state_prefix"]["current"] = {
        "known": True, "exists": False, "valid": False,
        "missing_completed_count": 4, "missing_completed_fraction": 4 / 6,
    }
    agent = _build_learned(method, 7, config, popart_enabled=False)
    old_agent = _build_learned(method, 7, {**config, "interface_profile": OLD_PROFILE}, popart_enabled=False)
    new_params = sum(parameter.numel() for parameter in agent._network.parameters())
    old_params = sum(parameter.numel() for parameter in old_agent._network.parameters())
    assert new_params - old_params == (192 if method == "sa_ghmappo" else 448)
    encoder = agent._network.encoder
    if method == "sa_ghmappo":
        baseline, _ = encoder._rsu_encoder._build_rsu_feature_tensor(info["semantic_state"], info["semantic_state"]["rsus"])
        modified, _ = encoder._rsu_encoder._build_rsu_feature_tensor(altered["semantic_state"], altered["semantic_state"]["rsus"])
        assert baseline.shape[-1] == 13
    else:
        baseline = encoder._build_feature_tensor(info["semantic_state"])
        modified = encoder._build_feature_tensor(altered["semantic_state"])
        assert baseline.shape[-1] == 25
    assert not torch.equal(baseline, modified)
    with torch.no_grad():
        old_output = agent._network.forward_single(info["semantic_state"])
        new_output = agent._network.forward_single(altered["semantic_state"])
    assert not torch.equal(old_output["encoded"]["shared_embedding"], new_output["encoded"]["shared_embedding"])
    action, action_info = agent.act(observation, info)
    assert action_info["log_prob"] == action_info["env_action_log_prob"]
    with torch.no_grad():
        policy_output = agent._forward_policy(info["semantic_state"], run_metadata=info["run_metadata"])
        recomputed, _, _ = agent._env_action_distribution_statistics(
            policy_output=policy_output, env_action=action, action_mask=info["action_mask"]
        )
    assert action_info["log_prob"] == pytest.approx(round(float(recomputed.item()), 6), abs=1e-6)
    new_path = tmp_path / f"{method}_new.pt"
    old_path = tmp_path / f"{method}_old.pt"
    agent.save(str(new_path))
    old_agent.save(str(old_path))
    agent.load(str(new_path))
    with pytest.raises(ValueError, match="profile checkpoint mismatch"):
        agent.load(str(old_path))
    with pytest.raises(ValueError, match="profile checkpoint mismatch"):
        old_agent.load(str(new_path))
    with pytest.raises(ValueError, match="profile and agent encoder mismatch"):
        old_agent.act(observation, info)
