from __future__ import annotations

import json
from pathlib import Path

from scripts.run_top_journal_closed_loop import (
    FORMAL_MIN_SETTINGS,
    ROOT_DIR,
    audit_formal_contract,
    effective_reward_positive_offset,
    effective_settings,
    evaluate_mode_gate,
    latest_training_summary_for_seed,
    select_sa_checkpoint,
)
from src.evaluators.main_results_support import build_window_context_agent_overrides, summary_to_row


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


def test_summary_row_reports_offset_adjusted_reward() -> None:
    summary = {
        "run_info": {
            "agent_name": "sa_ghmappo",
            "seed": 7,
            "window_id": "w0",
            "workflow_id": "wf0",
            "primary_vehicle_selection": "handoff_pressure",
            "reward_positive_offset": 5.0,
            "checkpoint_metadata": {},
        },
        "episode_success": True,
        "reward_breakdown": {"total": {"sum": 12.0}},
        "system_metrics": {
            "end_to_end_workflow_delay": 1.0,
            "workflow_continuity_rate": 1.0,
            "handoff_failure_rate": 0.0,
            "handoff_ready_ratio": 1.0,
            "adapter_warm_hit_ratio": 1.0,
            "cross_rsu_cold_start_frequency": 0.0,
            "backhaul_traffic_cost": 0.0,
            "adapter_state_migration_overhead": 0.0,
            "predictive_prefetch_precision": 1.0,
        },
        "handoff_summary": {
            "handoff_ready_count": 1,
            "migration_during_handoff_count": 0,
            "handoff_total_count": 1,
            "migration_prepare_count": 1,
        },
        "prefetch_summary": {"true_predictive_prefetch_count": 1},
        "prefetch_validation_summary": {
            "validated_predictive_prefetch_count": 1,
            "prefetch_validated_hit_count": 1,
            "prefetch_expired_miss_count": 0,
        },
        "agent_action_diagnostics": {},
        "step_trace": [
            {"reward_dict": {"positive_offset": 5.0}},
            {"reward_dict": {"positive_offset": 5.0}},
        ],
    }

    row = summary_to_row(summary)

    assert row["episode_step_count"] == 2
    assert row["reward_positive_offset_component"] == 10.0
    assert row["offset_adjusted_total_reward"] == 2.0


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


def test_effective_settings_honor_v6_sa_profile_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v6_strong_competition"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["train_window_count"] == 6


def test_effective_settings_honor_v7_sa_profile_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v7_latency_fallback"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["train_window_count"] == 6


def test_effective_settings_honor_v8_frozen_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v8_strict_full"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 96
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v9_pareto_safe_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v9_pareto_safe"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 96
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v10_mappo_rl_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v10_mappo_rl"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v11_mappo_reward_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v11_mappo_reward"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_v11_selects_best_reward_checkpoint_before_tiebreak() -> None:
    reward_path = _test_path("reward_first_checkpoint", "best_by_reward.pt")
    tiebreak_path = _test_path("reward_first_checkpoint", "best_by_reward_tiebreak_score.pt")
    reward_path.write_text("reward", encoding="utf-8")
    tiebreak_path.write_text("tiebreak", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v11_mappo_reward",
            "best_by_reward_path": str(reward_path),
            "best_by_reward_tiebreak_score_path": str(tiebreak_path),
        }
    )

    assert checkpoint_path == reward_path
    assert selection_field == "best_by_reward_path"


def test_v11_idle_window_enables_no_rsu_local_fallback_override() -> None:
    idle_overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v11_mappo_reward",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert idle_overrides["idle_popularity_no_rsu_local_fallback_enabled"] is True
    assert idle_overrides["idle_popularity_no_rsu_local_requires_low_context"] is False

    mechanism_overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v11_mappo_reward",
        run_metadata={"window_class": "mechanism_activating"},
    )

    assert "idle_popularity_no_rsu_local_fallback_enabled" not in mechanism_overrides


def test_effective_settings_honor_v12_learned_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v12_learned_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v13_prd_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v13_prd_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v14_net_utility_prd_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v14_net_utility_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v15_terminal_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v15_terminal_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v16_conservative_terminal_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v16_conservative_terminal_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v17_dag_aware_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v17_dag_aware_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v18_counterfactual_option_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v18_counterfactual_option"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v19_handoff_risk_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v19_handoff_risk_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v20_idle_execution_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v20_idle_execution_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v21_efficiency_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v21_efficiency_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v22_validated_utility_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v22_validated_utility_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v23_counterfactual_constrained_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v23_counterfactual_constrained_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v24_tail_risk_constrained_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v24_tail_risk_constrained_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v25_opportunity_risk_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v25_opportunity_risk_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v26_safe_counterfactual_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20


def test_effective_settings_honor_v27_segmented_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v27_conservative_advantage_imitation"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000


def test_effective_settings_honor_v28_executable_handoff_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v28_credit_focal_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v29_dt_fused_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v29_dt_fused_credit_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v30_dt_prior_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v30_dt_prior_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v31_handoff_pacing_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v31_handoff_pacing_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v32_dt_continuation_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v32_dt_continuation_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v33_env_action_ppo_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v33_env_action_ppo_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v34_adaptive_wait_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v34_adaptive_wait_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v35_guard_relaxed_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v35_guard_relaxed_action_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v36_counterfactual_margin_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v36_counterfactual_margin_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v37_advantage_gated_counterfactual_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v37_advantage_gated_counterfactual_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v38_undiscounted_dt_prior_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v38_undiscounted_dt_prior_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v39_delayed_credit_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v39_delayed_credit_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v40_advantage_weighted_behavior_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v40_advantage_weighted_behavior_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v41_conservative_recovery_budget() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v41_conservative_recovery_mappo"
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22


def test_effective_settings_honor_v42_completion_aligned_budget_and_offset() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v42_completion_aligned_mappo"
        reward_positive_offset = None
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["sa_update_every"] == 8
    assert settings["baseline_update_every"] == 8
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22
    assert effective_reward_positive_offset(Args()) == 0.0


def test_effective_settings_honor_v43_strict_opportunity_budget_and_offset() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v43_strict_opportunity_mappo"
        reward_positive_offset = None
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22
    assert effective_reward_positive_offset(Args()) == 0.0


def test_effective_settings_honor_v44_opportunity_constrained_budget_and_offset() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v44_opportunity_constrained_mappo"
        reward_positive_offset = None
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22
    assert effective_reward_positive_offset(Args()) == 0.0


def test_effective_settings_honor_v45_balanced_refresh_budget_and_offset() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v45_balanced_refresh_mappo"
        reward_positive_offset = None
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22
    assert effective_reward_positive_offset(Args()) == 0.0


def test_effective_settings_honor_v46_net_utility_constrained_budget_and_offset() -> None:
    class Args:
        quick = False
        sa_profile = "top_journal_mechanism_v46_net_utility_constrained_mappo"
        reward_positive_offset = None
        sa_episodes = None
        baseline_episodes = None
        sa_update_every = None
        baseline_update_every = None
        sa_batch_size = None
        baseline_batch_size = None
        max_mobility_rows = None
        max_workflows = None
        window_length = None
        window_count = None
        train_window_count = None
        window_scan_stride = None
        max_steps = None
        min_tasks = None
        max_tasks = None

    settings = effective_settings(Args())

    assert settings["sa_episodes"] == 128
    assert settings["baseline_episodes"] == 96
    assert settings["train_window_count"] == 20
    assert settings["max_mobility_rows"] == 5000000
    assert settings["max_steps"] == 22
    assert effective_reward_positive_offset(Args()) == 0.0


def test_v12_selects_reward_checkpoint_and_skips_v11_window_override() -> None:
    reward_path = _test_path("v12_reward_first_checkpoint", "best_by_reward.pt")
    continuity_path = _test_path("v12_reward_first_checkpoint", "best_by_continuity.pt")
    reward_path.write_text("reward", encoding="utf-8")
    continuity_path.write_text("continuity", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v12_learned_option",
            "best_by_reward_path": str(reward_path),
            "best_by_continuity_path": str(continuity_path),
        }
    )

    assert checkpoint_path == reward_path
    assert selection_field == "best_by_reward_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v12_learned_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v13_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v13_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v13_reward_first_checkpoint", "best_by_reward.pt")
    continuity_path = _test_path("v13_reward_first_checkpoint", "best_by_continuity.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")
    continuity_path.write_text("continuity", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v13_prd_option",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
            "best_by_continuity_path": str(continuity_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v13_prd_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v13_latest_checkpoint_selection_falls_back_to_reward() -> None:
    reward_path = _test_path("v13_latest_fallback_checkpoint", "best_by_reward.pt")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v13_prd_option",
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == reward_path
    assert selection_field == "best_by_reward_path"


def test_v14_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v14_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v14_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v14_net_utility_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v14_net_utility_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v15_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v15_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v15_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v15_terminal_option",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v15_terminal_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v16_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v16_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v16_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v16_conservative_terminal_option",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v16_conservative_terminal_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v17_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v17_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v17_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v17_dag_aware_option",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v17_dag_aware_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v18_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v18_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v18_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v18_counterfactual_option",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v18_counterfactual_option",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v19_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v19_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v19_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v19_handoff_risk_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v19_handoff_risk_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v20_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v20_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v20_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v20_idle_execution_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v20_idle_execution_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v21_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v21_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v21_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v21_efficiency_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v21_efficiency_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v22_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v22_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v22_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v22_validated_utility_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v22_validated_utility_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v23_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v23_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v23_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v23_counterfactual_constrained_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v23_counterfactual_constrained_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v24_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v24_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v24_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v24_tail_risk_constrained_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v24_tail_risk_constrained_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v25_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v25_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v25_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v25_opportunity_risk_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v25_opportunity_risk_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v26_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v26_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v26_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v26_mechanism_safe_counterfactual_prd",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v26_mechanism_safe_counterfactual_prd",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v27_selects_latest_checkpoint_and_skips_v11_window_override() -> None:
    latest_path = _test_path("v27_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v27_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v27_conservative_advantage_imitation",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v27_conservative_advantage_imitation",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v28_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v28_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v28_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v28_credit_focal_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v28_credit_focal_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v29_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v29_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v29_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v29_dt_fused_credit_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v29_dt_fused_credit_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v30_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v30_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v30_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v30_dt_prior_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v30_dt_prior_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v31_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v31_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v31_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v31_handoff_pacing_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v31_handoff_pacing_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v32_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v32_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v32_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v32_dt_continuation_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v32_dt_continuation_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v33_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v33_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v33_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v33_env_action_ppo_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v33_env_action_ppo_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v34_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v34_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v34_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v34_adaptive_wait_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v34_adaptive_wait_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v35_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v35_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v35_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v35_guard_relaxed_action_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v35_guard_relaxed_action_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v36_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v36_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v36_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v36_counterfactual_margin_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v36_counterfactual_margin_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v37_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v37_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v37_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v37_advantage_gated_counterfactual_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v37_advantage_gated_counterfactual_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v38_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v38_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v38_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v38_undiscounted_dt_prior_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v38_undiscounted_dt_prior_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v39_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v39_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v39_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v39_delayed_credit_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v39_delayed_credit_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v40_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v40_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v40_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v40_advantage_weighted_behavior_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v40_advantage_weighted_behavior_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v41_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v41_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v41_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v41_conservative_recovery_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v41_conservative_recovery_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v42_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v42_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v42_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v42_completion_aligned_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v42_completion_aligned_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v43_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v43_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v43_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v43_strict_opportunity_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v43_strict_opportunity_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v44_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v44_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v44_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v44_opportunity_constrained_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v44_opportunity_constrained_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v45_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v45_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v45_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v45_balanced_refresh_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v45_balanced_refresh_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_v46_selects_latest_checkpoint_and_skips_window_override() -> None:
    latest_path = _test_path("v46_latest_first_checkpoint", "latest.pt")
    reward_path = _test_path("v46_reward_first_checkpoint", "best_by_reward.pt")
    latest_path.write_text("latest", encoding="utf-8")
    reward_path.write_text("reward", encoding="utf-8")

    checkpoint_path, selection_field = select_sa_checkpoint(
        {
            "config_profile": "top_journal_mechanism_v46_net_utility_constrained_mappo",
            "latest_checkpoint_path": str(latest_path),
            "best_by_reward_path": str(reward_path),
        }
    )

    assert checkpoint_path == latest_path
    assert selection_field == "latest_checkpoint_path"

    overrides = build_window_context_agent_overrides(
        agent_name="sa_ghmappo",
        checkpoint_profile="top_journal_mechanism_v46_net_utility_constrained_mappo",
        run_metadata={"window_class": "idle_or_sparse"},
    )

    assert overrides == {}


def test_latest_training_summary_for_seed_filters_seed_suffix() -> None:
    seed7 = _test_path("resume_summary", "agent/run_a_seed7/train_summary.json")
    seed13 = _test_path("resume_summary", "agent/run_b_seed13/train_summary.json")
    seed7.write_text("{}", encoding="utf-8")
    seed13.write_text("{}", encoding="utf-8")

    assert latest_training_summary_for_seed(seed7.parents[2], "train_summary.json", 13) == seed13
