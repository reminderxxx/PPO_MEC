"""No-update contracts for calibrated-workflow per-head gradient probes."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest
import torch
from torch import nn

from scripts.run_calibrated_workflow_strong_baselines import _build_learned
from src.envs.core.calibrated_continuous_workflow_env import (
    PREPARED_STATE_PREFIX_PROFILE,
)
from tests.test_calibrated_workflow_prepared_state_prefix import _env, _fixture


def _agent():
    config, instance = _fixture()
    config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
    config["mechanism_aux_missing_current_event_abstention_enabled"] = True
    return (
        _build_learned("sa_ghmappo", 7, config, popart_enabled=False),
        config,
        instance,
    )


def _states(config: dict, instance: dict) -> tuple[dict, dict]:
    env = _env(config, instance, PREPARED_STATE_PREFIX_PROFILE)
    states: list[dict] = []
    for step in range(4):
        assert 3 in env.valid_actions()
        env.step(3)
        if step in {1, 3}:
            states.append(deepcopy(env._info()["semantic_state"]))
    return states[0], states[1]


def _parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_vector(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for (_, parameter), gradient in zip(named_parameters, gradients)
        ]
    )


def _relation(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | str]:
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    dot = float(torch.dot(left, right).item())
    if left_norm == 0.0 or right_norm == 0.0:
        return {
            "status": "UNDEFINED_ZERO_NORM",
            "dot": dot,
            "left_norm": left_norm,
            "right_norm": right_norm,
        }
    return {
        "status": "DEFINED",
        "dot": dot,
        "cosine": dot / (left_norm * right_norm),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def _auxiliary_components(agent, states: list[dict], outputs: list[dict]):
    slow_terms: list[torch.Tensor] = []
    fast_terms: list[torch.Tensor] = []
    event_terms: list[torch.Tensor] = []
    for state, output in zip(states, outputs):
        target = agent._build_mechanism_targets(state)
        confidence = float(target["confidence_weight"])
        if confidence <= 1e-6:
            continue
        slow = nn.functional.cross_entropy(
            output["slow_logits"].unsqueeze(0),
            torch.tensor([target["slow_target"]], device=agent._device),
        )
        fast = nn.functional.cross_entropy(
            output["fast_logits"].unsqueeze(0),
            torch.tensor([target["fast_target"]], device=agent._device),
        )
        event = nn.functional.cross_entropy(
            output["event_logits"].unsqueeze(0),
            torch.tensor([target["event_target"]], device=agent._device),
        )
        temporal = nn.functional.binary_cross_entropy_with_logits(
            (output["event_logits"][1] - output["event_logits"][0]).unsqueeze(0),
            torch.tensor(
                [float(target["event_soft_target"])],
                dtype=torch.float32,
                device=agent._device,
            ),
        )
        event_weight = agent._event_auxiliary_supervision_weight(state)
        slow_terms.append(confidence * agent._auxiliary_slow_weight * slow)
        fast_terms.append(confidence * agent._auxiliary_fast_weight * fast)
        event_terms.append(
            confidence
            * event_weight
            * (
                agent._auxiliary_event_weight * event
                + agent._temporal_consistency_coef * temporal
            )
        )
    denominator = len(slow_terms)
    assert denominator > 0
    return tuple(torch.stack(terms).sum() / denominator for terms in (slow_terms, fast_terms, event_terms))


def test_executed_action_probability_is_exact_marginal_not_canonical_tuple() -> None:
    wrapper, _, _ = _agent()
    agent = wrapper.agent
    slow_logits = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    fast_logits = torch.tensor([0.8, -0.2], requires_grad=True)
    event_logits = torch.tensor([-0.3, 0.5], requires_grad=True)
    output = {
        "slow_logits": slow_logits,
        "fast_logits": fast_logits,
        "event_logits": event_logits,
    }
    actual = torch.softmax(agent._hierarchical_env_action_scores(output), dim=-1)
    slow, fast, event = (
        torch.softmax(logits, dim=-1)
        for logits in (slow_logits, fast_logits, event_logits)
    )
    expected = torch.stack(
        [
            event[0] * slow[1],
            event[0] * slow[2],
            event[0] * slow[0] * fast[1],
            event[0] * slow[0] * fast[0],
            event[1],
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.allclose(expected.sum(), torch.tensor(1.0), atol=1e-7)

    canonical_action4 = event[1] * slow[0] * fast[0]
    assert actual[4] > canonical_action4

    loss = -torch.log(actual[4])
    derivative = torch.autograd.grad(loss, event_logits)[0][1]
    epsilon = 1e-3

    def perturbed(delta: float) -> float:
        changed = event_logits.detach().clone()
        changed[1] += delta
        changed_output = {**output, "event_logits": changed}
        probability = torch.softmax(
            agent._hierarchical_env_action_scores(changed_output), dim=-1
        )[4]
        return float((-torch.log(probability)).item())

    finite_difference = (perturbed(epsilon) - perturbed(-epsilon)) / (2 * epsilon)
    assert float(derivative) < 0.0
    assert finite_difference == pytest.approx(float(derivative), rel=2e-3, abs=2e-4)


def test_auxiliary_head_decomposition_preserves_original_fixed_denominator() -> None:
    wrapper, config, instance = _agent()
    agent = wrapper.agent
    agent._temporal_consistency_coef = 0.7
    ready, missing = _states(config, instance)
    outputs = [
        agent._forward_policy(state, run_metadata={"policy_evaluation_mode": "raw_policy"})
        for state in (ready, missing)
    ]
    slow, fast, event = _auxiliary_components(agent, [ready, missing], outputs)
    actual = agent._compute_auxiliary_loss([ready, missing], outputs)
    assert torch.allclose(actual, slow + fast + event, atol=1e-7, rtol=1e-7)
    assert agent._event_auxiliary_supervision_weight(ready) == 1.0
    assert agent._event_auxiliary_supervision_weight(missing) == 0.0

    weighted = agent._auxiliary_coef * actual
    weighted_components = agent._auxiliary_coef * (slow + fast + event)
    assert torch.equal(weighted, weighted_components)


def test_same_graph_gradients_are_reproducible_and_zero_norm_is_undefined() -> None:
    wrapper, config, instance = _agent()
    agent = wrapper.agent
    ready, _ = _states(config, instance)
    actor_parameters = [
        (name, parameter)
        for name, parameter in agent._network.named_parameters()
        if name.startswith(("encoder.", "slow_actor.", "fast_actor.", "event_actor."))
    ]

    def vectors() -> tuple[torch.Tensor, torch.Tensor]:
        output = agent._forward_policy(
            ready, run_metadata={"policy_evaluation_mode": "raw_policy"}
        )
        executed_loss = -torch.log_softmax(
            agent._hierarchical_env_action_scores(output), dim=-1
        )[2]
        target = agent._build_mechanism_targets(ready)
        fast_loss = (
            agent._auxiliary_coef
            * float(target["confidence_weight"])
            * agent._auxiliary_fast_weight
            * nn.functional.cross_entropy(
                output["fast_logits"].unsqueeze(0),
                torch.tensor([target["fast_target"]], device=agent._device),
            )
        )
        return (
            _gradient_vector(executed_loss, actor_parameters, retain_graph=True),
            _gradient_vector(fast_loss, actor_parameters),
        )

    first = vectors()
    second = vectors()
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert _relation(first[0], first[1]) == _relation(second[0], second[1])
    zero_relation = _relation(first[0], torch.zeros_like(first[0]))
    assert zero_relation["status"] == "UNDEFINED_ZERO_NORM"
    assert "cosine" not in zero_relation


def test_head_conditioning_value_isolation_and_no_parameter_mutation() -> None:
    wrapper, config, instance = _agent()
    agent = wrapper.agent
    ready, _ = _states(config, instance)
    before = _parameter_hash(agent._network)
    output = agent._forward_policy(
        ready, run_metadata={"policy_evaluation_mode": "raw_policy"}
    )
    target = agent._build_mechanism_targets(ready)

    slow_actor_parameters = [
        (name, parameter)
        for name, parameter in agent._network.named_parameters()
        if name.startswith("slow_actor.")
    ]
    fast_loss = nn.functional.cross_entropy(
        output["fast_logits"].unsqueeze(0),
        torch.tensor([target["fast_target"]], device=agent._device),
    )
    fast_to_slow = _gradient_vector(
        fast_loss, slow_actor_parameters, retain_graph=True
    )
    assert torch.linalg.vector_norm(fast_to_slow) > 0.0

    event_loss = nn.functional.cross_entropy(
        output["event_logits"].unsqueeze(0),
        torch.tensor([target["event_target"]], device=agent._device),
    )
    event_to_slow = _gradient_vector(
        event_loss, slow_actor_parameters, retain_graph=True
    )
    assert torch.linalg.vector_norm(event_to_slow) > 0.0

    actor_only_parameters = [
        (name, parameter)
        for name, parameter in agent._network.named_parameters()
        if name.startswith(("slow_actor.", "fast_actor.", "event_actor."))
    ]
    value_loss = (output["value"] - torch.tensor(1.0, device=agent._device)).square()
    value_to_actor = _gradient_vector(value_loss, actor_only_parameters)
    assert torch.count_nonzero(value_to_actor) == 0
    assert _parameter_hash(agent._network) == before
