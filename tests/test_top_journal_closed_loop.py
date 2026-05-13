from __future__ import annotations

import json
from pathlib import Path

from scripts.run_top_journal_closed_loop import (
    FORMAL_MIN_SETTINGS,
    ROOT_DIR,
    audit_formal_contract,
    evaluate_mode_gate,
    latest_training_summary_for_seed,
    select_sa_checkpoint,
)


TEST_ROOT = ROOT_DIR / "artifacts" / "tmp_validation" / "top_journal_closed_loop_tests"


def _test_path(test_name: str, filename: str) -> Path:
    path = TEST_ROOT / test_name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _metric(mean: float) -> dict[str, float]:
    return {"mean": mean}


def _agent_metrics(**overrides: float) -> dict[str, dict[str, float]]:
    metrics = {
        "total_reward": 10.0,
        "workflow_continuity_rate": 1.0,
        "handoff_failure_rate": 0.0,
        "backhaul_traffic_cost": 1.0,
        "handoff_ready_ratio": 1.0,
        "mechanism_realization_rate": 1.0,
        "adapter_state_migration_overhead": 0.0,
    }
    metrics.update(overrides)
    return {name: _metric(value) for name, value in metrics.items()}


def _write_aggregate(path: Path, *, sa_reward: float, ppo_reward: float, popularity_reward: float) -> None:
    payload = {
        "window_mode": "mixed_informative",
        "canonical_paper_protocol": True,
        "smoke_checkpoint_warnings": [],
        "episode_count": 18,
        "aggregate_by_agent": {
            "sa_ghmappo": {"metrics": _agent_metrics(total_reward=sa_reward)},
            "ppo": {"metrics": _agent_metrics(total_reward=ppo_reward)},
            "mappo": {"metrics": _agent_metrics(total_reward=ppo_reward - 0.5)},
            "popularity_cache_heuristic": {"metrics": _agent_metrics(total_reward=popularity_reward)},
        },
        "sa_advantage_diagnosis": {
            "minimum_success_reached": sa_reward > popularity_reward,
            "blockers": [] if sa_reward > popularity_reward else ["total_reward_not_above_popularity"],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_gate_passes_when_sa_beats_trainable_and_popularity() -> None:
    aggregate_path = _test_path("gate_pass", "aggregate_summary.json")
    _write_aggregate(aggregate_path, sa_reward=12.0, ppo_reward=10.0, popularity_reward=11.0)

    report = evaluate_mode_gate(
        aggregate_path=aggregate_path,
        baseline_agents=["ppo", "mappo"],
        tolerances={
            "continuity": 0.02,
            "handoff_failure": 0.01,
            "backhaul": 0.0,
            "mechanism": 0.02,
            "reward": 1e-6,
        },
        quick=False,
    )

    assert report["passed"] is True
    assert report["blockers"] == []


def test_gate_blocks_reward_regression_against_popularity() -> None:
    aggregate_path = _test_path("gate_block", "aggregate_summary.json")
    _write_aggregate(aggregate_path, sa_reward=10.5, ppo_reward=9.0, popularity_reward=11.0)

    report = evaluate_mode_gate(
        aggregate_path=aggregate_path,
        baseline_agents=["ppo", "mappo"],
        tolerances={
            "continuity": 0.02,
            "handoff_failure": 0.01,
            "backhaul": 0.0,
            "mechanism": 0.02,
            "reward": 1e-6,
        },
        quick=False,
    )

    assert report["passed"] is False
    assert "sa_total_reward_not_above_popularity" in report["blockers"]
    assert "benchmark_minimum_success_not_reached" in report["blockers"]


def test_select_sa_checkpoint_prefers_reward_tiebreak_guardrail_policy() -> None:
    tiebreak = _test_path("checkpoint_selection", "best_tiebreak.pt")
    retained = _test_path("checkpoint_selection", "best_retained.pt")
    reward = _test_path("checkpoint_selection", "best_reward.pt")
    tiebreak.write_bytes(b"checkpoint")
    retained.write_bytes(b"checkpoint")
    reward.write_bytes(b"checkpoint")

    checkpoint, field_name = select_sa_checkpoint(
        {
            "best_by_reward_tiebreak_score_path": str(tiebreak),
            "best_by_retained_mechanism_score_path": str(retained),
            "best_by_reward_path": str(reward),
        }
    )

    assert checkpoint == tiebreak
    assert field_name == "best_by_reward_tiebreak_score_path"


def test_formal_contract_blocks_quick_or_tiny_runs() -> None:
    class Args:
        quick = True
        primary_vehicle_selection = "handoff_pressure"
        seeds = [7]
        benchmark_modes = ["mixed_informative"]

    settings = dict(FORMAL_MIN_SETTINGS)
    settings["sa_episodes"] = 2

    audit = audit_formal_contract(Args(), settings)

    assert audit["ready"] is False
    assert "quick_mode" in audit["blockers"]
    assert "fewer_than_3_seeds:1" in audit["blockers"]
    assert "missing_required_benchmark_modes:full_stratified" in audit["blockers"]
    assert "sa_episodes_below_formal_min:2<96" in audit["blockers"]


def test_formal_contract_accepts_canonical_formal_budget() -> None:
    class Args:
        quick = False
        primary_vehicle_selection = "handoff_pressure"
        seeds = [7, 13, 29]
        benchmark_modes = ["mixed_informative", "full_stratified"]

    audit = audit_formal_contract(Args(), dict(FORMAL_MIN_SETTINGS))

    assert audit["ready"] is True
    assert audit["blockers"] == []


def test_latest_training_summary_for_seed_filters_seed_suffix() -> None:
    seed7 = _test_path("resume_summary", "agent/run_a_seed7/train_summary.json")
    seed13 = _test_path("resume_summary", "agent/run_b_seed13/train_summary.json")
    seed7.write_text("{}", encoding="utf-8")
    seed13.write_text("{}", encoding="utf-8")

    assert latest_training_summary_for_seed(seed7.parents[2], "train_summary.json", 13) == seed13
