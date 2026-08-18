"""Unified algorithm registry."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.cache_offload_agent import CacheOffloadDRLAgent
from src.agents.classical_cache_agent import (
    BASELINE_SCOPE,
    CONTROL_POLICY,
    ReactiveAgingLFUAgent,
    ReactiveFIFOAgent,
    ReactiveLFUAgent,
    ReactiveLRUAgent,
    ReactiveRandomAgent,
)
from src.agents.dag_offload_agent import DAGOffloadDRLAgent
from src.agents.dqn_agent import DDQNAgent, DQNAgent, DuelingDDQNAgent, DuelingDQNAgent
from src.agents.dt_handoff_agent import DTHandoffDRLAgent
from src.agents.ippo_agent import IPPOAgent
from src.agents.mat_agent import ControllerMATAgent
from src.agents.mappo_agent import MAPPOAgent
from src.agents.popularity_cache_heuristic_agent import PopularityCacheHeuristicAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.qmix_agent import QMIXAgent
from src.agents.reactive_greedy_agent import ReactiveGreedyAgent
from src.agents.sa_ghmappo_agent import SAGHMAPPOAgent


ALGO_REGISTRY: dict[str, dict[str, Any]] = {
    "sa_ghmappo": {
        "class": SAGHMAPPOAgent,
        "support_level": "trainable",
        "priority_tier": "main_method",
        "checkpoint_required": True,
        "observation_contract": "graph_surrogate_semantic_encoder",
        "action_contract": "semantic_discrete_5_multi_head",
        "notes": "Main graph/hierarchical on-policy method.",
    },
    "ppo": {
        "class": PPOAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": PPOAgent.observation_contract,
        "action_contract": PPOAgent.action_contract,
        "notes": "Basic single-agent PPO baseline with a flat semantic encoder.",
    },
    "dqn": {
        "class": DQNAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": DQNAgent.observation_contract,
        "action_contract": DQNAgent.action_contract,
        "notes": "Replay-based DQN baseline over the five-way semantic discrete action space.",
    },
    "ddqn": {
        "class": DDQNAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": DDQNAgent.observation_contract,
        "action_contract": DDQNAgent.action_contract,
        "notes": "Double-DQN baseline over the five-way semantic discrete action space.",
    },
    "dueling_dqn": {
        "class": DuelingDQNAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": DuelingDQNAgent.observation_contract,
        "action_contract": DuelingDQNAgent.action_contract,
        "notes": "Dueling-DQN baseline with separate value and advantage streams over the five-way semantic action space.",
    },
    "dueling_ddqn": {
        "class": DuelingDDQNAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": DuelingDDQNAgent.observation_contract,
        "action_contract": DuelingDDQNAgent.action_contract,
        "notes": "Dueling Double-DQN baseline over the five-way semantic action space.",
    },
    "qmix": {
        "class": QMIXAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": QMIXAgent.observation_contract,
        "action_contract": QMIXAgent.action_contract,
        "notes": (
            "Controller-level QMIX value-decomposition baseline over cache, execution, and handoff-event "
            "controllers with a centralized monotonic mixer. This is paper-grade for the current controller-agent "
            "contract, not a vehicle-agent or RSU-agent QMIX wrapper."
        ),
    },
    "controller_mat": {
        "class": ControllerMATAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": ControllerMATAgent.observation_contract,
        "action_contract": ControllerMATAgent.action_contract,
        "notes": (
            "Controller-level Multi-Agent Transformer PPO baseline over cache, execution, and "
            "handoff-event controller tokens. This is paper-grade for the current controller-agent "
            "contract and excludes SA-GHMAPPO graph/surrogate/guard mechanisms."
        ),
    },
    "dag_offload_drl": {
        "class": DAGOffloadDRLAgent,
        "support_level": "trainable",
        "priority_tier": "tier1_domain",
        "checkpoint_required": True,
        "observation_contract": DAGOffloadDRLAgent.observation_contract,
        "action_contract": DAGOffloadDRLAgent.action_contract,
        "notes": (
            "Dependency-aware DAG task-offloading PPO baseline with scalar DAG workload features. "
            "It matches the current controller-agent contract and excludes SA-GHMAPPO graph message passing, "
            "surrogate fusion, mechanism auxiliary losses, and guards."
        ),
    },
    "cache_offload_drl": {
        "class": CacheOffloadDRLAgent,
        "support_level": "trainable",
        "priority_tier": "tier1_domain",
        "checkpoint_required": True,
        "observation_contract": CacheOffloadDRLAgent.observation_contract,
        "action_contract": CacheOffloadDRLAgent.action_contract,
        "notes": (
            "Model/adapter cache-aware offloading PPO baseline with cache occupancy, adapter readiness, "
            "cache-demand, and future-load scalars. It excludes SA-GHMAPPO graph/surrogate/guard mechanisms."
        ),
    },
    "dt_handoff_drl": {
        "class": DTHandoffDRLAgent,
        "support_level": "trainable",
        "priority_tier": "tier1_domain",
        "checkpoint_required": True,
        "observation_contract": DTHandoffDRLAgent.observation_contract,
        "action_contract": DTHandoffDRLAgent.action_contract,
        "notes": (
            "Digital-twin-assisted handoff/service-migration PPO baseline using raw predicted RSU sequence, "
            "dwell-time, confidence, future-load, and boundary-pressure scalars. It excludes SA-GHMAPPO "
            "calibrated surrogate gates, uncertainty-aware scaling, mechanism auxiliary losses, and guards."
        ),
    },
    "ippo": {
        "class": IPPOAgent,
        "support_level": "diagnostic",
        "priority_tier": "contract_blocked",
        "checkpoint_required": True,
        "observation_contract": IPPOAgent.observation_contract,
        "action_contract": IPPOAgent.action_contract,
        "notes": (
            "Diagnostic IPPO placeholder only. The current single-wrapper decision stream "
            "does not expose independent per-agent actions, so this is not a paper-grade baseline."
        ),
    },
    "mappo": {
        "class": MAPPOAgent,
        "support_level": "trainable",
        "priority_tier": "tier1",
        "checkpoint_required": True,
        "observation_contract": MAPPOAgent.observation_contract,
        "action_contract": MAPPOAgent.action_contract,
        "notes": (
            "Controller-level CTDE MAPPO baseline with separate cache, execution, and handoff-event actors "
            "and a centralized flat semantic critic. Current paper-grade runs require aggregation-reason "
            "controller head-credit, so each policy head is updated mainly for the environment action it "
            "controlled. This is paper-grade for the current controller-agent contract, not a vehicle-agent "
            "or RSU-agent MAPPO wrapper."
        ),
    },
    "reactive_greedy": {
        "class": ReactiveGreedyAgent,
        "support_level": "heuristic",
        "priority_tier": "heuristic",
        "checkpoint_required": False,
        "observation_contract": ReactiveGreedyAgent.observation_contract,
        "action_contract": ReactiveGreedyAgent.action_contract,
        "notes": "Reactive cache-fill and steady offload baseline over the shared semantic action contract.",
    },
    "popularity_cache_heuristic": {
        "class": PopularityCacheHeuristicAgent,
        "support_level": "heuristic",
        "priority_tier": "heuristic",
        "checkpoint_required": False,
        "observation_contract": PopularityCacheHeuristicAgent.observation_contract,
        "action_contract": PopularityCacheHeuristicAgent.action_contract,
        "notes": "Popularity-aware cache and simple prediction-aware offload heuristic baseline.",
    },
}

for _agent_name, _agent_class, _policy_name, _requires_seed in (
    ("reactive_lru", ReactiveLRUAgent, "lru", False),
    ("reactive_fifo", ReactiveFIFOAgent, "fifo", False),
    ("reactive_lfu", ReactiveLFUAgent, "lfu", False),
    ("reactive_aging_lfu", ReactiveAgingLFUAgent, "aging_lfu", False),
    ("reactive_random", ReactiveRandomAgent, "random", True),
):
    ALGO_REGISTRY[_agent_name] = {
        "class": _agent_class,
        "support_level": "heuristic",
        "priority_tier": "classical",
        "checkpoint_required": False,
        "family": "classical_cache",
        "control_policy": CONTROL_POLICY,
        "required_eviction_policy": _policy_name,
        "eviction_policy": _policy_name,
        "eviction_policy_version": "1.0.0",
        "capacity_contract": "cache_capacity_contract_v1",
        "observation_contract": ReactiveGreedyAgent.observation_contract,
        "action_contract": ReactiveGreedyAgent.action_contract,
        "uses_prediction": False,
        "uses_learning": False,
        "requires_seed": _requires_seed,
        "baseline_scope": BASELINE_SCOPE,
        "notes": f"{BASELINE_SCOPE}; not full cooperative caching.",
    }


def validate_agent_eviction_binding(agent_name: str, cache_capacity_profile: dict[str, Any] | None, *, run_seed: int | None = None) -> dict[str, Any] | None:
    """Bind classical identity to the environment policy and fail on mismatch."""
    spec = ALGO_REGISTRY.get(agent_name)
    required = None if spec is None else spec.get("required_eviction_policy")
    if required is None:
        return cache_capacity_profile
    profile = dict(cache_capacity_profile or {})
    actual = str(profile.get("eviction_policy") or "").strip().lower()
    if actual != required:
        raise ValueError(f"agent {agent_name} requires eviction_policy={required}, got {actual or '<missing>'}")
    if required == "random":
        supplied = profile.get("eviction_policy_seed")
        if supplied is None:
            if run_seed is None:
                raise ValueError("reactive_random requires eviction_policy_seed or run_seed")
            profile["eviction_policy_seed"] = int(run_seed)
        elif run_seed is not None and int(supplied) != int(run_seed):
            raise ValueError("reactive_random policy seed must equal benchmark run seed")
    return profile


def build_agent(agent_name: str, **kwargs: Any) -> BaseAgent:
    """Build an agent by registry name."""
    if agent_name not in ALGO_REGISTRY:
        raise KeyError(f"unknown agent: {agent_name}")
    agent_class = ALGO_REGISTRY[agent_name]["class"]
    return agent_class(**kwargs)


def list_registered_agents() -> list[str]:
    """Return all registered live agent names."""
    return sorted(ALGO_REGISTRY.keys())


def list_trainable_agents() -> list[str]:
    """Return agents that can currently train/evaluate on the live wrapper."""
    return sorted(
        agent_name
        for agent_name, spec in ALGO_REGISTRY.items()
        if spec.get("support_level") == "trainable"
    )


def list_evaluable_agents() -> list[str]:
    """Return agents that can currently run through evaluation/benchmark."""
    runnable_levels = {"trainable", "heuristic", "diagnostic"}
    return sorted(
        agent_name
        for agent_name, spec in ALGO_REGISTRY.items()
        if spec.get("support_level") in runnable_levels
    )


def checkpoint_required_agents() -> set[str]:
    """Return registered agents that require a checkpoint for evaluation."""
    return {
        agent_name
        for agent_name, spec in ALGO_REGISTRY.items()
        if bool(spec.get("checkpoint_required", False))
    }


def get_algo_spec(agent_name: str) -> dict[str, Any]:
    """Return serializable registry metadata for one algorithm."""
    if agent_name not in ALGO_REGISTRY:
        raise KeyError(f"unknown agent: {agent_name}")
    spec = dict(ALGO_REGISTRY[agent_name])
    spec.pop("class", None)
    return spec
