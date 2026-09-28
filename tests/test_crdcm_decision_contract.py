from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.agents.crdcm_agent import (
    CRDCMMAPPOAgent,
    CRDCMPPOAgent,
    CRDCMSAGHMAPPOAgent,
)
from src.agents.crdcm_heuristic_agent import CRDCMCriticalPathHeuristicAgent
from src.agents.ppo_agent import PPOAgent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.encoders.crdcm_observation import (
    CRDCM_FEATURE_NAMES,
    finalize_crdcm_observation,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"


def _profile() -> dict[str, object]:
    return {
        "profile_id": "crdcm_decision_test_v1",
        "base_sharing_enabled": True,
        "workflow_state_migration_enabled": True,
        "crdcm_observation_enabled": True,
        "crdcm_observation_contract_version": "crdcm_observation_v1",
    }


def _env(max_steps: int = 3) -> VecWorkflowCoreEnv:
    workflow = ToyWorkflowGenerator().generate()
    compatible_adapters = [
        "adapter_perception",
        "adapter_tracking",
        "adapter_fusion",
        "adapter_intent",
        "adapter_control",
        "adapter_perception",
    ]
    for node, adapter_id in zip(workflow.nodes, compatible_adapters):
        node.required_base_model = "veh_base_v1"
        node.required_adapter = adapter_id
    return VecWorkflowCoreEnv(
        adapter_catalog=AdapterCatalog.from_json(CATALOG),
        workflow_state=workflow,
        max_steps=max_steps,
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": 360.0,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
        },
        mechanism_profile=_profile(),
    )


def _payload(
    *,
    current_ready: bool = True,
    target_ready: bool = False,
    target_available: bool = True,
    confidence: float = 0.8,
    eta: int | None = 2,
    reuse: int = 3,
    critical_path: int = 5,
    capacity_conflict: bool = False,
    state_required: bool = True,
    state_ready: bool = False,
    migration_enabled: bool = True,
) -> dict:
    return finalize_crdcm_observation(
        {
            "observed_at_time_index": 10,
            "update_phase": "pre_action_after_causal_prediction",
            "units": {"cache_capacity": "mb", "time": "mobility_step"},
            "sources": {"remaining_dag": "known_submitted_workflow_dag"},
            "required_request": {
                "node_id": "n2",
                "required_base_model_id": "veh_base_v1",
                "required_adapter_id": "adapter_perception",
            },
            "current_rsu": {
                "rsu_id": "rsu_a",
                "capacity_remaining": 140.0,
                "occupancy_rate": 0.6,
                "required_bundle": {
                    "base_ready": current_ready,
                    "adapter_ready": current_ready,
                    "missing_resident_mb": 0.0 if current_ready else 96.0,
                },
            },
            "predicted_target_rsu": {
                "rsu_id": "rsu_b" if target_available else None,
                "capacity_remaining": 140.0,
                "occupancy_rate": 0.6,
                "required_bundle": {
                    "base_ready": target_ready,
                    "adapter_ready": target_ready,
                    "missing_resident_mb": 0.0 if target_ready else 96.0,
                    "eviction_shortfall_mb": 40.0 if capacity_conflict else 0.0,
                },
            },
            "remaining_dag": {
                "remaining_nodes": 8,
                "remaining_ratio": 0.8,
                "frontier_size": 2,
                "critical_path_length": critical_path,
                "critical_path_pressure": critical_path / 8.0,
                "current_adapter_remaining_reuse_count": reuse,
                "current_adapter_remaining_reuse_ratio": reuse / 8.0,
                "completed_ratio": 0.2,
            },
            "handoff_prediction": {
                "target_rsu_id": "rsu_b" if target_available else None,
                "target_available": target_available,
                "target_differs_from_current": target_available,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "eta_steps": eta,
            },
            "migration": {
                "enabled": migration_enabled,
                "state_required": state_required,
                "state_ready": state_ready,
                "capacity_conflict": capacity_conflict,
                "risk": 0.8,
            },
        }
    )


def test_environment_produces_causal_typed_crdcm_observation() -> None:
    state, _ = _env().reset()
    observation = state["crdcm_observation"]
    assert observation["contract_version"] == "crdcm_observation_v1"
    assert observation["oracle_future_fields_actor_visible"] is False
    assert observation["outcome_fields_present"] is False
    assert observation["units"]["cache_capacity"] == "mb"
    assert observation["update_phase"] == "pre_action_after_causal_prediction"
    assert len(observation["feature_vector"]) == len(CRDCM_FEATURE_NAMES) == 28
    assert observation["required_request"]["required_adapter_id"]
    assert "capacity_remaining" in observation["current_rsu"]
    assert "critical_path_length" in observation["remaining_dag"]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"current_ready": False}, 0),
        ({"state_ready": False}, 4),
        ({"state_ready": True}, 1),
        ({"capacity_conflict": True}, 3),
        ({"reuse": 0, "critical_path": 2}, 3),
        ({"target_available": False, "state_required": False}, 3),
        ({"target_ready": True, "state_ready": True}, 3),
    ],
)
def test_strong_heuristic_controlled_sensitivity(changes: dict, expected: int) -> None:
    agent = CRDCMCriticalPathHeuristicAgent()
    action, details = agent.act(
        None,
        {
            "semantic_state": {"crdcm_observation": _payload(**changes)},
            "action_mask": [True, True, True, True, True],
        },
    )
    assert action == expected
    assert details["crdcm_decision_contract"]["actual_executed_action"] == action
    assert details["crdcm_decision_contract"]["actor_credit_source"] == "non_learning_heuristic"


@pytest.mark.parametrize(
    "agent_class",
    [CRDCMPPOAgent, CRDCMMAPPOAgent, CRDCMSAGHMAPPOAgent],
)
def test_same_crdcm_vector_reaches_each_learned_policy(agent_class: type) -> None:
    semantic_state, _ = _env().reset()
    first = deepcopy(semantic_state)
    second = deepcopy(semantic_state)
    second["crdcm_observation"] = _payload(
        current_ready=False,
        capacity_conflict=True,
        reuse=0,
        critical_path=1,
    )
    agent = agent_class(deterministic_action=True, random_seed=17)
    with torch.no_grad():
        first_output = agent._forward_policy(first)
        first_trace = deepcopy(agent._crdcm_last_forward_trace)
        second_output = agent._forward_policy(second)
        second_trace = deepcopy(agent._crdcm_last_forward_trace)
    key = "flat_logits" if not agent._use_hierarchy else "event_logits"
    assert not torch.equal(first_output[key], second_output[key])
    assert first_trace["feature_vector"] != second_trace["feature_vector"]
    assert first_trace["post_crdcm_head_logits"] != second_trace["post_crdcm_head_logits"]


def test_policy_action_trace_and_external_override_credit_are_consistent() -> None:
    recorder = EpisodeRecorder()
    trainer = MARLOnPolicyTrainer(
        env=GymVecEnv(core_env=_env(max_steps=2), recorder=recorder),
        agent=CRDCMPPOAgent(deterministic_action=True, random_seed=19),
        recorder=recorder,
        max_steps=2,
    )
    summary, rollout = trainer.collect_episode(
        run_metadata={"diagnostic_only": True},
        collect_model_targets=False,
    )
    trace = summary["policy_decision_trace_v2"]
    assert len(trace) == len(rollout) == 2
    for row, traced in zip(rollout, trace):
        assert traced["actual_executed_action"] == row["action"]
        assert traced["feature_vector"]
        assert traced["post_crdcm_head_logits"]
        assert traced["credit_provenance"]["actor_credit_source"] in {
            "raw_policy_sample",
            "external_override_masked",
        }

    action_info = deepcopy(rollout[0]["action_info"])
    policy_action = action_info["crdcm_decision_contract"]["aggregated_policy_action"]
    overridden = trainer._finalize_crdcm_action_credit(
        action_info=action_info,
        executed_action=(policy_action + 1) % 5,
    )
    assert overridden["crdcm_decision_contract"]["actor_credit_weight"] == 0.0
    assert overridden["crdcm_decision_contract"]["actor_credit_source"] == "external_override_masked"


def test_crdcm_checkpoint_rejects_legacy_and_round_trips_new(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_ppo.pt"
    PPOAgent(deterministic_action=True).save(str(legacy_path))
    crdcm = CRDCMPPOAgent(deterministic_action=True, random_seed=23)
    with pytest.raises(ValueError, match="legacy/non-CRDCM"):
        crdcm.load(str(legacy_path))

    crdcm_path = tmp_path / "crdcm_ppo.pt"
    crdcm.save(str(crdcm_path))
    restored = CRDCMPPOAgent(deterministic_action=True, random_seed=23)
    restored.load(str(crdcm_path))
    for key, value in crdcm._crdcm_residual.state_dict().items():
        assert torch.equal(value, restored._crdcm_residual.state_dict()[key])
