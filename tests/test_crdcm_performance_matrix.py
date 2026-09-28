from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_crdcm_performance_matrix as matrix_runner
from src.agents.crdcm_agent import CRDCMPPOAgent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/crdcm_performance_matrix_v2.yaml"
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"


def _env() -> VecWorkflowCoreEnv:
    workflow = ToyWorkflowGenerator().generate()
    adapters = [
        "adapter_perception",
        "adapter_tracking",
        "adapter_fusion",
        "adapter_intent",
        "adapter_control",
        "adapter_perception",
    ]
    for node, adapter_id in zip(workflow.nodes, adapters):
        node.required_base_model = "veh_base_v1"
        node.required_adapter = adapter_id
    return VecWorkflowCoreEnv(
        adapter_catalog=AdapterCatalog.from_json(CATALOG),
        workflow_state=workflow,
        max_steps=2,
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": 360.0,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
        },
        mechanism_profile={
            "profile_id": "crdcm_matrix_test_v2",
            "base_sharing_enabled": True,
            "workflow_state_migration_enabled": True,
            "crdcm_observation_enabled": True,
            "crdcm_observation_contract_version": "crdcm_observation_v1",
        },
    )


def test_frozen_matrix_budget_and_condition_mapping_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    matrix_runner._validate_scientific_config(config)
    assert config["budget"]["training_episodes"] == 768
    assert config["budget"]["evaluation_episodes"] == 156
    assert config["budget"]["full_execution_episode_cap"] == 924
    assert config["budget"]["full_execution_environment_step_cap"] == 18480
    assert config["seeds"] == [1401, 1402, 1403]


def test_signal_off_is_same_wrapper_with_zero_new_residual() -> None:
    state, _ = _env().reset()
    full = CRDCMPPOAgent(
        deterministic_action=True,
        random_seed=1401,
        crdcm_feature_mode="full",
    )
    off = CRDCMPPOAgent(
        deterministic_action=True,
        random_seed=1401,
        crdcm_feature_mode="signal_off",
    )
    off._network.load_state_dict(full._network.state_dict())
    off._crdcm_residual.load_state_dict(full._crdcm_residual.state_dict())
    with torch.no_grad():
        full_output = full._forward_policy(state)
        off_output = off._forward_policy(state)
    assert torch.count_nonzero(off_output["crdcm_residual"]).item() == 0
    assert not torch.equal(full_output["flat_logits"], off_output["flat_logits"])
    assert off._crdcm_last_forward_trace["feature_mode"] == "signal_off"


def test_external_override_masks_actor_but_keeps_critic_training() -> None:
    recorder = EpisodeRecorder()
    agent = CRDCMPPOAgent(
        deterministic_action=False,
        random_seed=1401,
        crdcm_feature_mode="full",
        train_epochs=1,
        batch_size=2,
    )
    trainer = MARLOnPolicyTrainer(
        env=GymVecEnv(core_env=_env(), recorder=recorder),
        agent=agent,
        recorder=recorder,
        max_steps=2,
    )
    _, rollout = trainer.collect_episode(
        run_metadata={"test": "crdcm_actor_credit"},
        collect_model_targets=False,
    )
    masked_rollout = deepcopy(rollout)
    for row in masked_rollout:
        contract = row["action_info"]["crdcm_decision_contract"]
        policy_action = int(contract["aggregated_policy_action"])
        executed_action = (policy_action + 1) % 5
        row["action_info"] = trainer._finalize_crdcm_action_credit(
            action_info=row["action_info"],
            executed_action=executed_action,
        )
        row["action"] = executed_action
        finalized = row["action_info"]["crdcm_decision_contract"]
        assert finalized["actual_executed_action"] == executed_action
        assert finalized["actor_credit_source"] == "external_override_masked"
        assert finalized["sampled_from_raw_policy_distribution"] is False
    final_layer = agent._crdcm_residual[-1]
    actor_weights_before = final_layer.weight[:5].detach().clone()
    value_weights_before = final_layer.weight[5].detach().clone()
    stats = agent.learn(masked_rollout)
    assert stats["actor_update_skipped"] is True
    assert stats["external_override_masked_count"] == len(masked_rollout)
    assert stats["actor_credit_eligible_count"] == 0
    assert stats["critic_credit_sample_count"] == len(masked_rollout)
    assert stats["critic_only_sample_count"] == len(masked_rollout)
    assert stats["critic_update_skipped"] is False
    assert torch.equal(actor_weights_before, final_layer.weight[:5])
    assert not torch.equal(value_weights_before, final_layer.weight[5])
    assert all(
        torch.isfinite(parameter).all()
        for module in (agent._network, agent._crdcm_residual)
        for parameter in module.parameters()
    )


def test_checkpoint_feature_mode_round_trip_and_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full.pt"
    full = CRDCMPPOAgent(random_seed=1401, crdcm_feature_mode="full")
    full.save(str(checkpoint))
    restored = CRDCMPPOAgent(random_seed=1401, crdcm_feature_mode="full")
    restored.load(str(checkpoint))
    mismatch = CRDCMPPOAgent(random_seed=1401, crdcm_feature_mode="signal_off")
    with pytest.raises(ValueError, match="feature mode mismatch"):
        mismatch.load(str(checkpoint))


def test_lfs_pointer_is_rejected_before_content_use(tmp_path: Path) -> None:
    pointer = tmp_path / "batch_task.csv"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc\n"
        "size 802261444\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        matrix_runner._validate_source(
            pointer,
            {
                "size_bytes": 802261444,
                "sha256": "6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc",
            },
            "workflow",
        )


def test_preflight_import_failure_writes_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "failed_launch"

    def fail(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise RuntimeError("entrypoint import preflight failed")

    monkeypatch.setattr(matrix_runner, "_build_manifest", fail)
    manifest, manifest_path, digest = matrix_runner._prepare_output(
        CONFIG,
        tmp_path,
        output_root,
        smoke=False,
    )
    assert manifest is manifest_path is digest is None
    receipt = json.loads(
        (output_root / "startup_failure_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "FAILED"
    assert receipt["phase"] == "preflight_before_training"
    assert receipt["training_episode_count"] == 0
    assert "entrypoint import preflight failed" in receipt["reason"]
