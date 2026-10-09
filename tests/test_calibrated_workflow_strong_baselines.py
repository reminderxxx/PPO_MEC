from __future__ import annotations

import json
import csv
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.analyze_calibrated_workflow_service_reward_alignment import _seed_rows, _summary_rows
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from scripts.run_calibrated_workflow_strong_baselines import (
    _annotate_rows,
    _build_learned,
    _evaluate_popularity,
    _run_learned_cell,
    _validate_protocol,
    _write_csv,
)
from scripts.run_calibrated_workflow_interface_repair import _evaluate_rule
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_instance(split: str, index: int) -> dict:
    start = index * 1000
    return {
        "design_id": f"synthetic_{split}_{index}",
        "split": split,
        "window_id": f"window_synthetic_run_{index:03d}_off0_len8_t{start}_{start + 700}",
        "workflow_id": f"synthetic_dag_{index}",
        "source_interval": {
            "source_segment_id": f"synthetic_run_{index:03d}",
            "frame_offset": 0,
            "window_length": 8,
            "time_index_start": start,
            "time_index_end": start + 700,
        },
        "trace_features": {
            "estimated_handoff_count": 1,
            "active_vehicle_count_mean": 2.0,
            "mean_speed_proxy": 5.0,
            "handoff_pressure": "medium",
        },
        "workflow_features": {
            "num_tasks": 1,
            "num_edges": 0,
            "root_count": 1,
            "topology_class": "simple",
        },
        "factors": {
            "sharing": "shared_base",
            "capacity": "tight",
            "state_scale": 1,
            "initial_target": "competitor",
            "actual_mbps": 1000.0,
            "estimated_mbps": 1000.0,
            "prediction_quality": "matched",
            "prediction_confidence": 0.9,
            "prediction_uncertainty": 0.1,
        },
        "rsu_ids": ["rsu_0", "rsu_1"],
        "rsu_sequence": ["rsu_0", "rsu_0", "rsu_1", "rsu_1", "rsu_0", "rsu_0", "rsu_1", "rsu_1"],
        "link_profile": {"actual_mbps": 1000.0, "estimated_mbps": 1000.0, "error_class": "matched"},
        "prediction_profile": {"confidence": 0.9, "uncertainty": 0.1, "quality": "matched"},
        "cache_capacity_bytes": 1_169_449_264,
        "initial_residents": {
            "rsu_0": ["base:family_a", "adapter:helmet_shared"],
            "rsu_1": ["base:family_a"],
        },
        "nodes": [{
            "node_id": "n0",
            "node_name": "synthetic",
            "required_base_model": "base:family_a",
            "required_adapter": "helmet_shared",
            "input_size": 100,
            "output_size": 10,
            "input_bytes": 100,
            "state_bytes": 10,
            "compute_seconds": 1.0,
            "predecessors": [],
            "successors": [],
            "trace_duration_raw": 1,
            "trace_plan_mem": 0.3,
        }],
        "edges": [],
        "execution_order": ["n0"],
        "deadline_seconds": 10.0,
        "max_steps": 8,
        "source_classes": {"measured": [], "trace_derived": [], "literature": [], "artificial": ["entire_test_fixture"]},
    }


@pytest.fixture
def config() -> dict:
    resolved, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    resolved["reward_profile"] = "original_reward_v1"
    return resolved


def test_dt_shared_observation_mask_and_checkpoint(config: dict, tmp_path: Path) -> None:
    instance = _synthetic_instance("train", 1)
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    agent = _build_learned("dt_handoff_drl", 7, config, popart_enabled=False)
    assert agent._executed_action_ppo_only is True
    assert agent._hierarchical_action_contract == "independent_heads_executed_env_v2"
    action, action_info = agent.act(observation, info)
    assert len(observation) == 9 and len(info["action_mask"]) == 5
    assert info["action_mask"][action] and "log_prob" in action_info
    bad = deepcopy(info)
    bad["action_mask"] = [False] * 5
    with pytest.raises(RuntimeError, match="outside the public mask"):
        agent.act(observation, bad)
    checkpoint = tmp_path / "dt.pt"
    agent.save(str(checkpoint))
    agent.load(str(checkpoint))
    assert checkpoint.is_file()


def test_dt_one_synthetic_update_through_common_artifact_chain(config: dict, tmp_path: Path) -> None:
    splits = {
        "train": [_synthetic_instance("train", 1)],
        "dev": [_synthetic_instance("dev", 2)],
        "regression": [_synthetic_instance("regression", 3)],
        "frozen_check": [_synthetic_instance("frozen_check", 4)],
    }
    protocol = {
        "transitions_per_update": 4,
        "update_opportunities_per_method_seed": 1,
        "environment_steps_per_method_seed": 4,
        "ppo_epochs_per_update": 4,
        "minibatch_size": 32,
        "expected_optimizer_steps_per_method_seed": 4,
        "checkpoint_update_candidates": [1],
        "episode_max_steps": 8,
        "evaluation_splits": ["regression", "frozen_check"],
    }
    result = _run_learned_cell(
        method="dt_handoff_drl", seed=7, config=config, splits=splits,
        protocol=protocol, checkpoints=tmp_path / "checkpoints",
        reward_profile="original_reward_v1", popart_enabled=False,
    )
    assert result["summary"]["environment_steps"] == 4
    assert result["summary"]["optimizer_steps"] == 4
    assert result["summary"]["selected_update"] == 1
    assert result["summary"]["value_normalization_enabled"] is False
    assert len(result["candidates"]) == 1 and len(result["evaluation_rows"]) == 2
    assert result["summary"]["selected_checkpoint_sha256"]
    assert (tmp_path / result["candidates"][0]["checkpoint"]).is_file()
    assert (tmp_path / result["summary"]["selected_checkpoint"]).is_file()
    assert result["training_episodes"]
    assert all(row["method"] == "dt_handoff_drl" and row["seed"] == 7 and "source_time_start" in row for row in result["training_episodes"])
    assert all(row["source_segment_id"].startswith("synthetic_run_") for row in result["evaluation_rows"])
    assert all("source_time_start" in row for row in result["behavior_rows"])
    assert _summary_rows(result["evaluation_rows"])
    assert _seed_rows(result["evaluation_rows"])
    popularity, _ = _evaluate_popularity(config, splits["regression"], 8, "original_reward_v1")
    two_step, two_step_ledger = _evaluate_rule(config, splits["regression"], 8)
    two_step, two_step_ledger = _annotate_rows(two_step, two_step_ledger, splits["regression"], "original_reward_v1")
    combined = result["evaluation_rows"] + popularity + two_step
    _write_csv(tmp_path / "evaluation_rows.csv", combined)
    with (tmp_path / "evaluation_rows.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 4
    assert {row["method"] for row in saved} == {"dt_handoff_drl", "popularity_cache_heuristic", "two_step_cost_rule"}
    assert all(row["source_time_start"] and row["source_time_end"] for row in saved)
    assert len(_summary_rows(combined)) == 4
    assert len(_seed_rows(combined)) == 2


def test_popularity_resets_for_every_synthetic_instance(config: dict) -> None:
    instances = [_synthetic_instance("regression", 5), _synthetic_instance("regression", 6)]
    rows, ledger = _evaluate_popularity(config, instances, 8, "original_reward_v1")
    assert len(rows) == 2 and ledger
    assert all(row["seed"] == "rule" and row["heuristic_memory_scope"] == "one_instance_fresh_agent" for row in rows)
    assert all(row["source_time_start"] >= 5000 for row in ledger)
    assert all(row["aggregation_reason"] for row in ledger)
    assert not _seed_rows(rows)
    assert len(_summary_rows(rows)) == 1
    again, _ = _evaluate_popularity(config, instances, 8, "original_reward_v1")
    assert [{key: row[key] for key in ("action_0", "action_1", "action_2", "action_3", "action_4")} for row in rows] == [
        {key: row[key] for key in ("action_0", "action_1", "action_2", "action_3", "action_4")} for row in again
    ]


def test_development_protocol_is_unapproved_and_raw_by_default() -> None:
    design = json.loads((ROOT / "configs/experiment/calibrated_workflow_strong_baselines_development_v1.json").read_text())
    _validate_protocol(design)
    assert design["execution_authorized"] is False
    assert design["critic_target_normalization"] == "raw_disabled"
    assert design["claim_boundary"]["formal"] is False
    promoted = deepcopy(design)
    promoted["critic_target_normalization"] = "popart_running_mean_std"
    with pytest.raises(RuntimeError, match="separately frozen"):
        _validate_protocol(promoted)
