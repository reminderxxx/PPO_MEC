from __future__ import annotations

import torch

from src.agents.ppo_agent import TraceDrivenMultiRequestPPO


def test_policy_never_selects_uncovered_rsu() -> None:
    policy = TraceDrivenMultiRequestPPO(hidden_size=8)
    features = torch.zeros((3, 4, len(policy.feature_names)))
    mask = torch.tensor([
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
    ])
    for deterministic in (False, True):
        for _ in range(10):
            actions, logprobs, values = policy.act(features, mask, deterministic=deterministic)
            assert actions.tolist() == [0, 1, 2]
            assert torch.isfinite(logprobs).all()
            assert torch.isfinite(values).all()
