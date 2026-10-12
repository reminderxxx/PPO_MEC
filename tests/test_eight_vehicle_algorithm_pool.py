"""Boundary checks for the separate eight-vehicle diagnostic policy pool."""

import random

import pytest
import torch

from src.agents.dqn_agent import TraceDrivenMultiRequestQNetwork
from src.agents.mappo_agent import TraceDrivenMultiRequestMAPPO
from scripts.benchmark_eight_vehicle_algorithms import _q_update


@pytest.mark.parametrize("dueling", [False, True])
def test_q_network_excludes_uncovered_rsus(dueling):
    model = TraceDrivenMultiRequestQNetwork(dueling=dueling)
    features = torch.randn(2, 4, 10)
    mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
    values = model(features, mask)
    assert values.shape == (2, 4)
    assert int(values[0].argmax()) in (0, 2)
    assert int(values[1].argmax()) == 1
    assert torch.all(values[~mask] < -1e8)
    with pytest.raises(ValueError):
        model(features, torch.zeros_like(mask))


def test_q_update_accepts_terminal_and_legal_successor():
    torch.manual_seed(4)
    online = TraceDrivenMultiRequestQNetwork()
    target = TraceDrivenMultiRequestQNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    state = torch.randn(4, 10)
    mask = torch.tensor([True, False, False, False])
    replay = [(state, mask, 0, 0.1, state, mask, i % 2 == 0) for i in range(256)]
    loss = _q_update(online, target, optimizer, replay, double=True,
                     gamma=0.99, rng=random.Random(7))
    assert loss is not None and torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(before, after)
               for before, after in zip(target.parameters(), online.parameters()))


def test_trace_mappo_requires_joint_context_and_masks_actions():
    model = TraceDrivenMultiRequestMAPPO()
    features = torch.randn(3, 4, 10)
    mask = torch.tensor([[True, False, False, False]] * 3)
    context = torch.randn(3, 40)
    distribution, values = model.distribution_and_value(features, mask, context)
    other_distribution, other_values = model.distribution_and_value(features, mask, context + 1)
    assert distribution.sample().tolist() == [0, 0, 0]
    assert values.shape == (3,)
    assert torch.allclose(distribution.logits, other_distribution.logits)
    assert not torch.allclose(values, other_values)
    with pytest.raises(ValueError):
        model.distribution_and_value(features, mask, context[:2])
