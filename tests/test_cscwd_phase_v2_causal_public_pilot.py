"""Interface and budget gates for the one-shot raw phase-v2 pilot."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from scripts.run_cscwd_phase_v2_causal_public_pilot import (
    _build_cell_agent, _collect_batch, _completion_receipt_payload, _phase_env,
)
from scripts.analyze_cscwd_phase_v2_causal_public_pilot import analyze
from src.agents.causal_public_action_estimator import (
    CausalPublicImmediateRule, CausalPublicTwoStepRule,
    estimate_public_actions, public_prepare_advantage_label,
)
from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace


ROOT = Path(__file__).resolve().parents[1]


def _setup():
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    config["interface_profile"] = "calibrated_workflow_interface_v4_prepared_state_prefix"
    phase = json.loads(
        (ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text()
    )
    instance = next(deepcopy(row) for row in manifest["instances"] if row["design_id"] == "dev_01")
    return config, phase, instance


def _trace(count: int = 1200, speed: float = 7.4) -> RawVehicleTrace:
    times = tuple(round(index * 0.1, 9) for index in range(count))
    return RawVehicleTrace(
        times, tuple((0.0, speed * seconds) for seconds in times), "pilot_fixture"
    )


def test_protocol_freezes_one_variable_and_exact_budget():
    protocol = json.loads(
        (ROOT / "configs/experiment/cscwd_phase_v2_causal_public_pilot_v1.json").read_text()
    )
    assert protocol["single_variable"]["name"] == "mechanism_aux_causal_public_prepare_advantage_enabled"
    assert protocol["single_variable"]["reward_change"] is False
    assert protocol["training"]["total_environment_steps"] == 20 * 1152 == 23040
    assert protocol["training"]["total_optimizer_steps"] == 20 * 192 == 3840
    assert protocol["evaluation"]["checkpoint_selection_episode_count"] == 20 * 4
    assert protocol["evaluation"]["final_learned_episode_count"] == 20 * 2 * 8
    assert protocol["claim_boundary"]["independent_unit_count"] == 2
    assert protocol["verdict"]["automatic_next_stage"] is False


def test_future_pilot_completion_receipt_matches_supervisor_contract():
    receipt = _completion_receipt_payload(
        expected_commit="a" * 40,
        elapsed_seconds=1.25,
    )
    assert receipt["schema_version"] == "scientific_completion_receipt_v1"
    assert receipt["status"] == "complete"
    assert receipt["scientific_execution_complete"] is True
    assert receipt["environment_steps"] == 23040
    assert receipt["optimizer_steps"] == 3840
    assert receipt["checkpoint_selection_episodes"] == 80
    assert receipt["final_evaluation_episodes"] == 320


def test_actual_phase_v2_public_consumer_and_sa_action_likelihood():
    config, phase, instance = _setup()
    env = _phase_env(config, instance, _trace(), phase, "default")
    observation, info = env.reset()
    estimate = estimate_public_actions(info["semantic_state"], info["action_mask"])
    label = public_prepare_advantage_label(info["semantic_state"], info["action_mask"])
    assert estimate["schema_version"] == "causal_public_action_estimator_v3"
    assert label["schema_version"] == "causal_public_prepare_advantage_v3"
    assert label["decision"] in {"prepare", "serve", "abstain"}
    immediate = CausalPublicImmediateRule().select_action_from_info(info)
    two_step = CausalPublicTwoStepRule().select_action_from_info(info)
    assert two_step == immediate
    assert info["action_mask"][immediate]
    torch.manual_seed(7)
    agent = _build_cell_agent(
        {"arm": "sa_candidate", "method": "sa_ghmappo", "candidate_enabled": True},
        7, config,
    )
    action, action_info = agent.act(observation, deepcopy(info))
    assert info["action_mask"][action]
    assert np.isfinite(float(action_info["log_prob"]))


@pytest.mark.parametrize(
    "cell",
    [
        {"arm": "sa_control", "method": "sa_ghmappo", "candidate_enabled": False},
        {"arm": "sa_candidate", "method": "sa_ghmappo", "candidate_enabled": True},
        {"arm": "ppo", "method": "ppo", "candidate_enabled": False},
        {"arm": "mappo", "method": "mappo", "candidate_enabled": False},
    ],
)
def test_actual_phase_v2_rollout_bootstrap_and_optimizer_contract(cell):
    config, phase, instance = _setup()
    torch.manual_seed(7)
    agent = _build_cell_agent(cell, 7, config)
    batch, state, episodes_started, episodes = _collect_batch(
        agent=agent,
        env_factory=lambda: _phase_env(config, instance, _trace(), phase, "default"),
        state=None, episodes_started=0, transition_count=48, step_cap=24,
        gamma=float(config["training"]["gamma"]),
        gae_lambda=float(config["training"]["gae_lambda"]),
    )
    assert len(batch) == 48
    assert episodes_started >= 1
    assert all(np.isfinite(float(row["bootstrap_value"])) for row in batch)
    assert all(not (row["segment_terminated"] and row["bootstrap_value"] != 0.0) for row in batch)
    assert state is None or state["episode_index"] == episodes_started
    result = agent.learn(batch)
    assert not result["policy_update_skipped"]
    assert result["optimizer_step_count"] == 8
    assert len(result["optimizer_step_records"]) == 8
    assert all(np.isfinite(float(row["env_action_ppo_loss"])) for row in result["optimizer_step_records"])


def test_analysis_is_descriptive_and_never_auto_continues(tmp_path):
    run = tmp_path / "cscwd_phase_v2_causal_public_pilot_20261011_v1"
    run.mkdir()
    (run / "completion_receipt.json").write_text(json.dumps({"status": "complete"}))
    (run / "run_manifest.json").write_text(json.dumps({"status": "complete", "git_commit": "a" * 40}))
    (run / "artifact_integrity.json").write_text(json.dumps({"files": []}))
    rows = []
    cases = [
        (profile, condition, segment)
        for profile, conditions in (
            ("raw_ngsim_event_time_v2", ("default", "adverse_link", "ample_cache")),
            ("raw_ngsim_event_time_v1", ("mechanism_off_default",)),
        )
        for condition in conditions for segment in ("lankershim", "us_101")
    ]
    for arm in ("sa_control", "sa_candidate", "ppo", "mappo"):
        for view in ("selected", "fixed_update24"):
            for seed in (7, 17, 29, 43, 61):
                for profile, condition, segment in cases:
                    rows.append({
                        "arm": arm, "view": view, "seed": seed,
                        "time_profile": profile, "condition_id": condition,
                        "source_segment_id": segment,
                        "workflow_completion_rate": 1.0,
                        "on_time_workflow_completion_rate": 1.0,
                        "service_failure_rate": 0.0,
                        "total_transfer_mb": 1.0, "recompute_seconds": 0.0,
                        "partial_transfer_bytes": 0.0, "decision_overhead_ms": 1.0,
                    })
    with (run / "evaluation_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = analyze(run, tmp_path / "analysis")
    assert summary["verdict"] == "MIXED"
    assert summary["automatic_next_stage"] is False
    assert summary["paper_ready"] is False
    assert summary["paired_case_count"] == 80
