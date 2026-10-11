"""Result gates for the calibration-abstention scientific A/B."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_cscwd_positive_prepare_calibration_abstention_ab import analyze


ARMS = ("sa_public_v3", "sa_calibration_abstention", "ppo", "mappo")
VIEWS = ("selected", "fixed_update24")
SEEDS = (7, 17, 29, 43, 61)
CASES = tuple(
    (profile, condition, segment)
    for profile, conditions in (
        ("raw_ngsim_event_time_v2", ("default", "adverse_link", "ample_cache")),
        ("raw_ngsim_event_time_v1", ("mechanism_off_default",)),
    )
    for condition in conditions
    for segment in ("lankershim", "us_101")
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> None:
    root.mkdir()
    (root / "completion_receipt.json").write_text(json.dumps({
        "status": "complete", "scientific_execution_complete": True,
    }))
    (root / "run_manifest.json").write_text(json.dumps({
        "status": "complete",
        "git_commit": "a" * 40,
        "schema_version": (
            "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_manifest_v1"
        ),
        "calibration_abstention_scope": {
            "raw_producer_confidence_calibrated": False,
            "new_arm_effect": "removes_all_positive_prepare_auxiliary_supervision",
            "fine_grained_uncertainty_control": False,
            "three_head_structure_contribution_tested": False,
            "single_variable": (
                "mechanism_aux_positive_prepare_calibration_abstention_enabled"
            ),
        },
    }))
    (root / "artifact_integrity.json").write_text(json.dumps({"files": []}))

    evaluation = []
    behavior = []
    for arm in ARMS:
        candidate = arm == "sa_calibration_abstention"
        for view in VIEWS:
            for seed in SEEDS:
                for profile, condition, segment in CASES:
                    transfer = 90 if candidate else 100
                    elapsed = 9.0 if candidate else 10.0
                    evaluation.append({
                        "arm": arm, "view": view, "method": arm, "seed": seed,
                        "time_profile": profile, "condition_id": condition,
                        "source_segment_id": segment,
                        "workflow_completion_rate": 1.0,
                        "on_time_workflow_completion_rate": 1.0,
                        "service_failure_rate": 0.0, "service_failures": 0,
                        "unfinished_rate": 0.0, "truncated": False,
                        "completed_sample_elapsed_seconds": elapsed,
                        "elapsed_seconds_for_failed_or_truncated": "",
                        "total_transfer_mb": transfer / 1_000_000,
                        "input_transfer_bytes": transfer,
                        "model_transfer_bytes": transfer,
                        "state_transfer_bytes": transfer,
                        "backhaul_transfer_bytes": transfer,
                        "partial_transfer_bytes": 0,
                        "recompute_seconds": elapsed,
                        "migration_successes": 0,
                        "reward": 2.0 if candidate else 1.0,
                        "reward_node_completion": 1.0,
                        "reward_workflow_completion": 1.0,
                        "reward_time_penalty": -elapsed,
                        "reward_transfer_penalty": -transfer,
                        "reward_failure_penalty": 0.0,
                        "reward_deadline_penalty": 0.0,
                    })
                    behavior.append({
                        "arm": arm, "view": view, "method": arm, "seed": seed,
                        "time_profile": profile, "condition_id": condition,
                        "source_segment_id": segment,
                        "migration_success": False, "failure_reason": "",
                    })
    _write_csv(root / "evaluation_rows.csv", evaluation)
    _write_csv(root / "behavior_rows.csv", behavior)

    signals = []
    for arm in ARMS:
        for index in range(5760):
            signals.append({
                "arm": arm,
                "global_step": index + 1,
                "event_label": (
                    "abstain" if arm == "sa_calibration_abstention" else "prepare"
                ),
                "event_supervision_weight": (
                    0.0 if arm == "sa_calibration_abstention" else 1.0
                ),
            })
    _write_csv(root / "training_signal_rows.csv", signals)
    _write_csv(root / "training_updates.csv", [
        {"arm": arm, "seed": seed, "update_index": update}
        for arm in ARMS for seed in SEEDS for update in range(1, 25)
    ])
    _write_csv(root / "rule_evaluation_rows.csv", [{
        "method": "causal_public_immediate_rule", "reward": 0.0,
    }])


def test_analysis_reports_reward_and_resources_without_overclaim(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(run)
    summary = analyze(run, tmp_path / "analysis")
    assert summary["verdict"] == "SUPPORTED_DEVELOPMENT_CANDIDATE"
    assert summary["candidate_positive_prepare_supervised_count"] == 0
    assert summary["control_positive_prepare_supervised_count"] == 5760
    assert summary["reward_lead_view_count"] == 2
    assert summary["reward_tradeoff"] is False
    assert summary["any_seed_view_service_harm"] is False
    assert summary["paper_ready"] is False
    assert "three-head structural contribution" in summary["claim_boundary"]
