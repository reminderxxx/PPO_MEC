"""Classical cache baselines sharing one reactive admission/control policy."""

from __future__ import annotations

from typing import Any

from src.agents.reactive_greedy_agent import ReactiveGreedyAgent


CONTROL_POLICY = "reactive_current_rsu_admission_v1"
BASELINE_SCOPE = "reactive placement/admission + selected eviction policy"


class ClassicalCacheAgent(ReactiveGreedyAgent):
    """Identity-preserving wrapper; eviction remains environment-owned."""

    family = "classical_cache"
    control_policy = CONTROL_POLICY
    baseline_scope = BASELINE_SCOPE

    def __init__(self, *, agent_name: str, eviction_policy: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent_name = str(agent_name)
        self.eviction_policy = str(eviction_policy)
        self.config.update({"agent_name": self.agent_name, "control_policy": self.control_policy, "eviction_policy": self.eviction_policy})

    def _action_info(self, action: int, reason: str) -> dict[str, Any]:
        return {**super()._action_info(action, reason), "policy_type": self.agent_name, "control_policy": self.control_policy, "required_eviction_policy": self.eviction_policy, "baseline_scope": self.baseline_scope}


def _make_classical_agent(agent_name: str, eviction_policy: str):
    class BoundClassicalCacheAgent(ClassicalCacheAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(agent_name=agent_name, eviction_policy=eviction_policy, **kwargs)
    BoundClassicalCacheAgent.__name__ = "".join(part.title() for part in agent_name.split("_")) + "Agent"
    return BoundClassicalCacheAgent


ReactiveLRUAgent = _make_classical_agent("reactive_lru", "lru")
ReactiveFIFOAgent = _make_classical_agent("reactive_fifo", "fifo")
ReactiveLFUAgent = _make_classical_agent("reactive_lfu", "lfu")
ReactiveAgingLFUAgent = _make_classical_agent("reactive_aging_lfu", "aging_lfu")
ReactiveRandomAgent = _make_classical_agent("reactive_random", "random")
