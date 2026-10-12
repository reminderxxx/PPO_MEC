"""Exact marginal PPO contract; synthetic checks, no performance claim."""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest
import torch
from torch.distributions import Categorical

from src.agents.registry import build_agent
from src.evaluators.real_eval_support import _filter_checkpoint_config


def _output():
    return {"slow_logits": torch.tensor([0.7, -0.4, 0.2], requires_grad=True),
            "fast_logits": torch.tensor([-0.6, 0.3], requires_grad=True),
            "event_logits": torch.tensor([0.4, -0.7], requires_grad=True)}


def _state():
    return {"time_index": 0, "observation_contract": "joint_request_observation_v2",
            "current_workflow_node": {"node_id": "n0", "required_adapter": "a", "input_size": 1., "output_size": 1.},
            "workflow": {"nodes": [{"node_id": "n0", "predecessors": [], "successors": []}],
                         "execution_order": ["n0"], "current_node_id": "n0", "completed_node_ids": []},
            "vehicles": [{"vehicle_id": "v0", "associated_rsu_id": "r0", "speed": 1.}],
            "rsus": [{"rsu_id": "r0", "cached_adapter_ids": [], "cache_capacity": 2}],
            "predictions": {}}


def _agent(name="mappo", **kwargs):
    return build_agent(name, actor_objective="masked_env_action_v1", train_epochs=1, batch_size=8, **kwargs)


def _rollout(agent):
    rows = []
    for mask, advantage in [([True]*5, 1.), ([True, False, True, True, False], -1.)]:
        info = {"semantic_state": _state(), "action_mask": mask,
                "run_metadata": {"policy_evaluation_mode": "raw_policy"}}
        action, details = agent.act([], info)
        rows.append({"decision_info": info, "action": action, "action_info": details,
                     "log_prob": details["log_prob"], "value": details["value"],
                     "advantage": advantage, "return": advantage, "reward": advantage})
    return rows


@pytest.mark.parametrize("mask", [list(bits) for bits in product([False, True], repeat=5) if any(bits)])
def test_all_31_masks_match_marginal_sampling_ratio_entropy_and_gradient(mask):
    agent = _agent()
    output = _output()
    e, s, f = [torch.softmax(output[f"{head}_logits"], -1) for head in ("event", "slow", "fast")]
    reference_probs = torch.stack([e[0]*s[1], e[0]*s[2], e[0]*s[0]*f[1], e[0]*s[0]*f[0], e[1]])
    reference_probs = reference_probs * torch.tensor(mask)
    reference_probs = reference_probs / reference_probs.sum()
    reference = Categorical(probs=reference_probs)
    action = next(i for i, valid in enumerate(mask) if valid)
    old_log = reference.log_prob(torch.tensor(action)).detach().reshape(1)
    stats = agent._exact_env_action_ppo_statistics(
        [output], [mask], torch.tensor([action]), old_log, torch.ones(1), reference_probs.detach()[None])
    assert torch.allclose(stats["ratio"], torch.ones(1), atol=2e-7)
    assert torch.allclose(stats["entropy"], reference.entropy(), atol=2e-7)
    assert stats["kl"].item() < 2e-7
    actual_grad = torch.autograd.grad(stats["actor_loss"], tuple(output.values()), retain_graph=True)
    expected_loss = -torch.exp(reference.log_prob(torch.tensor(action)) - old_log[0])
    reference_grad = torch.autograd.grad(expected_loss, tuple(output.values()))
    for actual, expected in zip(actual_grad, reference_grad):
        assert torch.allclose(actual, expected, atol=2e-7)
        if sum(mask) == 1:
            assert torch.allclose(actual, torch.zeros_like(actual), atol=2e-7)
    heads, _, _, _, projection = agent._sample_actions(output, deterministic=True, action_mask=mask)
    assert projection["projected_env_action"] == reference_probs.argmax().item()
    assert torch.allclose(torch.tensor(projection["masked_env_action_probs"]), reference_probs.detach(), atol=1e-6)


def test_irrelevant_fast_head_has_no_gradient_for_cache_fill():
    agent = _agent()
    output = _output()
    mask = [True, False, True, True, False]
    dist = Categorical(logits=agent._masked_flat_logits(agent._hierarchical_env_action_scores(output), mask))
    stats = agent._exact_env_action_ppo_statistics([output], [mask], torch.tensor([0]),
        dist.log_prob(torch.tensor(0)).detach()[None], torch.ones(1), dist.probs.detach()[None])
    stats["actor_loss"].backward()
    assert output["fast_logits"].grad.abs().max() < 1e-7


@pytest.mark.parametrize("name", ["ppo", "mappo", "sa_ghmappo", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl"])
def test_rollout_learn_and_checkpoint_restore(name, tmp_path, monkeypatch):
    agent = _agent(name, learning_rate=0.0)
    rows = _rollout(agent)
    # An old head objective must never contribute to the exact mode's actor loss.
    monkeypatch.setattr(agent, "_compute_hierarchical_actor_loss", lambda **kw: torch.tensor(12345.))
    result = agent.learn(rows)
    assert result["actor_objective"] == "masked_env_action_v1"
    assert abs(result["actor_loss"]) < 1e-5
    assert result["approx_kl"] < 1e-6
    assert result["clip_fraction"] == 0.0
    path = tmp_path / "policy.pt"
    agent.save(str(path))
    payload = torch.load(path, weights_only=False)
    filtered = _filter_checkpoint_config(name, payload["config"])
    assert filtered["actor_objective"] == "masked_env_action_v1"
    restored = build_agent(name, **filtered)
    restored.load(str(path))
    assert restored._actor_objective == "masked_env_action_v1"
    # Direct loads also restore this behavioral contract.
    direct = build_agent(name)
    direct.load(str(path))
    assert direct._actor_objective == "masked_env_action_v1"
    info = deepcopy(rows[0]["decision_info"])
    info["deterministic_policy"] = True
    action, details = agent.act([], info)
    other_action, other_details = restored.act([], info)
    assert action == other_action
    assert details["env_action_probs"] == other_details["env_action_probs"]


@pytest.mark.parametrize("failure", ["metadata", "old_probability", "deterministic", "empty_mask"])
def test_missing_or_inconsistent_behavior_fails_closed(failure):
    agent = _agent()
    rows = _rollout(agent)
    if failure == "metadata":
        rows[0]["decision_info"].pop("run_metadata")
    elif failure == "old_probability":
        rows[0]["action_info"].pop("env_action_log_prob")
    elif failure == "deterministic":
        rows[0]["action_info"]["policy_mode"] = "deterministic"
    else:
        rows[0]["decision_info"]["action_mask"] = [False]*5
    with pytest.raises(ValueError):
        agent.learn(rows)


def test_legacy_checkpoint_remains_legacy_and_supplemental_loss_is_rejected(tmp_path):
    legacy = build_agent("mappo")
    path = tmp_path / "legacy.pt"
    legacy.save(str(path))
    payload = torch.load(path, weights_only=False)
    payload["config"].pop("actor_objective")
    torch.save(payload, path)
    upgraded = _agent()
    upgraded.load(str(path))
    assert upgraded._actor_objective == "legacy_head"
    with pytest.raises(ValueError, match="replaces"):
        _agent(env_action_ppo_enabled=True, env_action_ppo_coef=1.0)


def test_changed_policy_clipping_and_full_distribution_kl():
    agent = _agent("ppo")
    old_logits = torch.tensor([0.3, -0.2, 0.7, 0.1, -0.6])
    new_logits = torch.tensor([1.8, 0.1, -0.5, 0.4, -0.3], requires_grad=True)
    old, new = Categorical(logits=old_logits), Categorical(logits=new_logits)
    actions = torch.tensor([0, 2])
    advantages = torch.tensor([1., -1.])
    stats = agent._exact_env_action_ppo_statistics(
        [{"flat_logits": new_logits}]*2, [[True]*5]*2, actions,
        old.log_prob(actions), advantages, old.probs.expand(2, -1))
    ratio = new.probs[actions] / old.probs[actions]
    expected_loss = -torch.minimum(ratio*advantages,
        ratio.clamp(1-agent._clip_ratio, 1+agent._clip_ratio)*advantages).mean()
    assert torch.allclose(stats["actor_loss"], expected_loss)
    assert torch.allclose(stats["kl"], torch.distributions.kl_divergence(old, new))
    stats["actor_loss"].backward()
    # Both chosen advantages are beyond their relevant clipping bound.
    assert new_logits.grad.abs().max() == 0


@pytest.mark.parametrize("name", ["ppo", "mappo", "sa_ghmappo", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl", "dqn"])
def test_joint_observation_routes_to_all_agents_and_checkpoint(name, tmp_path):
    kwargs = {"joint_observation_enabled": True}
    if name != "dqn":
        kwargs["actor_objective"] = "masked_env_action_v1"
    agent = build_agent(name, **kwargs)
    info = {"semantic_state": _state(), "action_mask": [True]*5,
            "run_metadata": {"policy_evaluation_mode": "raw_policy"}, "deterministic_policy": True,
            "centralized_critic_state": {"joint_workflows": {"v0": {
                "remaining_nodes": 1, "remaining_work_units": 1, "work_progress": 0,
                "state_version": 0, "parents_ready": True}}, "rsus": [], "associations": {"v0": "r0"}}}
    changed = deepcopy(info)
    changed["centralized_critic_state"]["joint_workflows"]["v0"].update(
        remaining_nodes=8, remaining_work_units=5, state_version=3)
    action, details = agent.act([], info)
    changed_action, changed_details = agent.act([], changed)
    assert action == changed_action
    if name == "dqn":
        assert details["q_values"] == changed_details["q_values"]
    else:
        assert details["env_action_probs"] == changed_details["env_action_probs"]
        if name != "ppo":
            assert details["value"] != changed_details["value"]
    assert "centralized_critic_state" not in info["semantic_state"]
    assert any("_joint_actor_projection" in key for key in agent._network.state_dict())
    path = tmp_path / "joint.pt"
    agent.save(str(path))
    payload = torch.load(path, weights_only=False)
    filtered = _filter_checkpoint_config(name, payload["config"])
    assert filtered["joint_observation_enabled"] is True
    restored = build_agent(name, **filtered)
    restored.load(str(path))
    restored_action, restored_details = restored.act([], info)
    assert action == restored_action
    probability_key = "q_values" if name == "dqn" else "env_action_probs"
    assert details[probability_key] == restored_details[probability_key]
    with pytest.raises(ValueError, match="joint_observation_enabled mismatch"):
        build_agent(name).load(str(path))


@pytest.mark.parametrize("name", ["ppo", "mappo", "sa_ghmappo", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl", "dqn"])
def test_v2_checkpoint_rejects_legacy_observation(name, tmp_path):
    agent = build_agent(name, joint_observation_enabled=True)
    path = tmp_path / "joint.pt"
    agent.save(str(path))
    config = torch.load(path, weights_only=False)["config"]
    restored = build_agent(name, **_filter_checkpoint_config(name, config))
    restored.load(str(path))
    state = _state()
    state.pop("observation_contract")
    with pytest.raises(ValueError, match="joint_request_observation_v2"):
        restored.act([], {"semantic_state": state})


@pytest.mark.parametrize("name", ["mappo", "sa_ghmappo", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl"])
@pytest.mark.parametrize("central", [None, {}, {"joint_workflows": {}, "rsus": []}])
def test_joint_centralized_critic_rejects_missing_context(name, central):
    agent = build_agent(name, joint_observation_enabled=True)
    with pytest.raises(ValueError, match="joint centralized critic"):
        agent.act([], {"semantic_state": _state(), "centralized_critic_state": central})


@pytest.mark.parametrize("name", ["ppo", "dqn"])
def test_noncentral_agents_accept_v2_without_centralized_context(name):
    agent = build_agent(name, joint_observation_enabled=True)
    agent.act([], {"semantic_state": _state(), "action_mask": [True]*5})


@pytest.mark.parametrize("failure", ["legacy_next", "missing_next"])
def test_dqn_rejects_non_v2_next_state_before_replay_insertion(failure):
    agent = build_agent("dqn", joint_observation_enabled=True)
    next_state = _state()
    next_state.pop("observation_contract")
    row = {"decision_info": {"semantic_state": _state()}, "action": 0,
           "terminated": False, "env_info": {"semantic_state": next_state}}
    if failure == "missing_next":
        row["env_info"] = {}
    with pytest.raises(ValueError):
        agent.learn([row])
    assert len(agent._replay) == 0
    row["env_info"] = {}
    row["terminated"] = True
    agent.learn([row])
    assert len(agent._replay) == 1
