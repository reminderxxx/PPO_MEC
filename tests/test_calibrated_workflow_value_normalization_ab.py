from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent, _load_json
from scripts.run_calibrated_workflow_service_reward_alignment import (
    _resolved_arm_config,
)
from scripts.run_calibrated_workflow_value_normalization_ab import (
    _aggregate_evaluation,
    _analyze_gates,
    _collect_exact_update_batch,
    _selection_score,
    _validate_intervals,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT_DIR / "configs/experiment/calibrated_workflow_value_normalization_ab_v1.json"
AUTHORIZATION_PATH = (
    ROOT_DIR
    / "configs/experiment/calibrated_workflow_value_normalization_ab_authorized_v1.json"
)


def _resolved_service_config_and_manifest() -> tuple[dict, dict]:
    design = _load_json(DESIGN_PATH)
    base, _ = _load_experiment_config(ROOT_DIR / design["base_config"])
    reward_config = _load_json(
        ROOT_DIR
        / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json"
    )
    return (
        _resolved_arm_config(
            base,
            {"service_aligned_reward": reward_config["service_aligned_reward"]},
            design["reward_profile"],
        ),
        _load_json(ROOT_DIR / design["workload_manifest"]),
    )


def test_authorized_budget_is_exact_and_single_variable() -> None:
    design = _load_json(DESIGN_PATH)
    authorization = _load_json(AUTHORIZATION_PATH)
    protocol = design["development_ab"]
    cells = len(design["arms"]) * len(protocol["learned_methods"]) * len(
        protocol["seeds"]
    )

    assert cells == authorization["training_cell_count"] == 30
    assert cells * protocol["environment_steps_per_cell"] == 43_200
    assert cells * protocol["update_opportunities_per_cell"] == 720
    assert cells * protocol["optimizer_steps_per_cell"] == 5_760
    assert protocol["checkpoint_update_candidates"] == [6, 12, 18, 24]
    assert design["single_variable"] == "critic_target_popart_normalization"
    assert design["candidate_contract"]["reward_unchanged"] is True
    assert design["candidate_contract"]["actor_loss_unchanged"] is True
    assert design["candidate_contract"]["auxiliary_loss_unchanged"] is True


def test_exact_collector_returns_sixty_transitions_without_cross_episode_gae() -> None:
    config, manifest = _resolved_service_config_and_manifest()
    train_instances = [
        row for row in manifest["instances"] if row["split"] == "train"
    ]
    agent = _build_agent("ppo", 7, config)

    rows, state, episodes_started, episode_rows = _collect_exact_update_batch(
        agent=agent,
        config=config,
        train_instances=train_instances,
        order=list(range(len(train_instances))),
        rng=random.Random(7),
        state=None,
        episodes_started=0,
        transition_count=60,
        step_cap=24,
        gamma=float(config["training"]["gamma"]),
        gae_lambda=float(config["training"]["gae_lambda"]),
    )

    assert len(rows) == 60
    assert episodes_started >= 1
    assert all("return" in row and "advantage" in row for row in rows)
    assert all(row["training_episode_index"] >= 1 for row in rows)
    assert len({row["training_episode_index"] for row in rows}) >= 2
    assert episode_rows
    assert state is None or state["episode_index"] == episodes_started


def test_checkpoint_selection_score_does_not_use_reward() -> None:
    row = {
        "on_time_workflow_completion_rate": 0.5,
        "workflow_completion_rate": 0.75,
        "unfinished_after_deadline_rate": 0.25,
        "service_failure_rate": 0.1,
        "max_consecutive_no_progress_steps": 1,
        "completed_sample_elapsed_seconds": 8.0,
        "total_transfer_mb": 3.0,
        "recompute_seconds": 0.5,
        "invalid_prepare_attempts": 1,
        "reward": -10_000.0,
    }
    changed = dict(row, reward=10_000.0)
    assert _selection_score([row]) == _selection_score([changed])


def test_interval_validator_uses_raw_identity_and_rejects_overlap() -> None:
    _, manifest = _resolved_service_config_and_manifest()
    result = _validate_intervals(list(manifest["instances"]))
    assert result["all_identity_fields_present"] is True
    assert result["all_intervals_pairwise_disjoint"] is True

    duplicate = [dict(row) for row in manifest["instances"][:2]]
    duplicate[1] = dict(duplicate[1], source_interval=dict(duplicate[0]["source_interval"]))
    overlap = _validate_intervals(duplicate)
    assert overlap["all_intervals_pairwise_disjoint"] is False


def test_evaluation_aggregate_reports_attempt_and_episode_failure_rates() -> None:
    base = {
        "arm": "control_raw_critic_target",
        "method": "ppo",
        "seed": 7,
        "split": "regression",
        "on_time_workflow_completion_rate": 0.0,
        "workflow_completion_rate": 0.0,
        "unfinished_after_deadline_rate": 1.0,
        "service_failure_rate": 1.0,
        "service_failures": 2,
        "failed_service_attempt_seconds_proxy": 1.0,
        "total_transfer_mb": 0.0,
        "recompute_seconds": 0.0,
        "invalid_prepare_attempts": 0,
        "max_consecutive_no_progress_steps": 2,
        "current_missing_action4_rate": 0.0,
        "completed_sample_elapsed_coverage": 0.0,
        "completed_sample_elapsed_seconds": "",
        "action_0": 2,
        "action_1": 2,
        "action_2": 2,
        "action_3": 2,
        "action_4": 2,
    }
    aggregate = _aggregate_evaluation([base], by_seed=True)[0]
    assert aggregate["service_failure_attempt_rate"] == pytest.approx(0.2)
    assert aggregate["service_failure_episode_rate"] == pytest.approx(1.0)
    assert aggregate["seed"] == "7"


def test_behavior_gate_uses_probability_and_raw_argmax_not_only_execution() -> None:
    methods = ["sa_ghmappo", "mappo", "ppo"]
    update_rows = []
    evaluation_rows = []
    behavior_rows = []
    for method in methods:
        for seed in (7, 17, 29, 43, 61):
            for arm, ratio, explained in (
                ("control_raw_critic_target", 10.0, 0.0),
                ("candidate_popart_critic_target", 1.0, 0.2),
            ):
                update_rows.append(
                    {
                        "arm": arm,
                        "method": method,
                        "seed": seed,
                        "update_index": 24,
                        "weighted_value_to_policy_grad_ratio": ratio,
                        "explained_variance": explained,
                    }
                )
                evaluation_rows.append(
                    {
                        "arm": arm,
                        "method": method,
                        "seed": seed,
                        "split": "regression",
                        "workflow_completion_rate": 1.0,
                        "on_time_workflow_completion_rate": 1.0,
                        "service_failure_rate": 0.0,
                        "service_failures": 0,
                        "max_consecutive_no_progress_steps": 0,
                        **{f"action_{action_id}": 1 for action_id in range(5)},
                    }
                )
                probability = 0.7 if arm.startswith("control") else 0.1
                behavior_rows.append(
                    {
                        "arm": arm,
                        "split": "regression",
                        "current_bundle_ready": False,
                        "executed_action": 0,
                        "raw_env_action": 4 if arm.startswith("control") else 0,
                        "env_action_probs": json.dumps(
                            [0.075, 0.075, 0.075, 0.075, probability]
                        ),
                    }
                )

    result = _analyze_gates(
        evaluation_rows=evaluation_rows,
        behavior_rows=behavior_rows,
        update_rows=update_rows,
        optimizer_rows=[],
        methods=methods,
    )
    assert result["behavior"]["pass"] is True
    assert result["behavior"]["candidate_current_missing_action4"][
        "mean_probability"
    ] < result["behavior"]["control_current_missing_action4"]["mean_probability"]
