from __future__ import annotations

import random

from scripts.analyze_top_journal_statistics import holm_adjust, summarize_deltas
from scripts.train_sa_ghmappo_real_sample import compute_pareto_safe_checkpoint_score


def test_hierarchical_bootstrap_uses_window_as_outer_cluster() -> None:
    deltas: list[float] = []
    outer_clusters: list[tuple[str, ...]] = []
    inner_clusters: list[tuple[str, ...]] = []
    for window_index, window_delta in enumerate([-10.0, -8.0, 8.0, 14.0]):
        for seed in [7, 13, 29]:
            for workflow_id in ["j_3", "j_8"]:
                deltas.append(window_delta + (seed % 3) * 0.1)
                outer_clusters.append((f"window_id=w{window_index}",))
                inner_clusters.append((f"seed={seed}", f"workflow_id={workflow_id}"))

    summary = summarize_deltas(
        deltas,
        bootstrap_samples=2000,
        rng=random.Random(7),
        outer_clusters=outer_clusters,
        inner_clusters=inner_clusters,
        ci_method="bca",
    )

    assert summary["bootstrap_unit"] == "hierarchical"
    assert summary["paired_count"] == 24
    assert summary["cluster_count"] == 4
    assert summary["outer_cluster_count"] == 4
    assert summary["inner_cluster_count"] == 24
    assert summary["bca_available"] is True
    assert summary["ci95_method"] == "bca"


def test_hierarchical_interval_is_wider_for_correlated_window_rows() -> None:
    deltas: list[float] = []
    outer_clusters: list[tuple[str, ...]] = []
    inner_clusters: list[tuple[str, ...]] = []
    for window_index, window_delta in enumerate([-12.0, -8.0, 8.0, 16.0]):
        for replicate in range(12):
            deltas.append(window_delta)
            outer_clusters.append((f"window_id=w{window_index}",))
            inner_clusters.append((f"replicate={replicate}",))

    pair_summary = summarize_deltas(
        deltas,
        bootstrap_samples=3000,
        rng=random.Random(7),
        ci_method="percentile",
    )
    hierarchical_summary = summarize_deltas(
        deltas,
        bootstrap_samples=3000,
        rng=random.Random(7),
        outer_clusters=outer_clusters,
        inner_clusters=inner_clusters,
        ci_method="percentile",
    )

    pair_width = pair_summary["ci95_high"] - pair_summary["ci95_low"]
    hierarchical_width = hierarchical_summary["ci95_high"] - hierarchical_summary["ci95_low"]
    assert hierarchical_width > pair_width


def test_hierarchical_bootstrap_is_deterministic() -> None:
    deltas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    outer = [("window_id=w0",)] * 3 + [("window_id=w1",)] * 3
    inner = [(f"seed={index}",) for index in range(6)]

    first = summarize_deltas(
        deltas,
        bootstrap_samples=1000,
        rng=random.Random(7),
        outer_clusters=outer,
        inner_clusters=inner,
    )
    second = summarize_deltas(
        deltas,
        bootstrap_samples=1000,
        rng=random.Random(7),
        outer_clusters=outer,
        inner_clusters=inner,
    )

    assert first == second


def test_holm_adjust_controls_the_full_family() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])

    assert adjusted == [0.04, 0.09, 0.09, 0.20]


def test_pareto_safe_score_penalizes_failure_and_backhaul_regressions() -> None:
    reference = {
        "total_reward": 80.0,
        "workflow_continuity_rate": 0.9,
        "handoff_failure_rate": 0.05,
        "backhaul_traffic_cost": 100.0,
    }
    safe_candidate = {
        "total_reward": 85.0,
        "workflow_continuity_rate": 0.92,
        "handoff_failure_rate": 0.04,
        "backhaul_traffic_cost": 90.0,
        "handoff_ready_ratio": 0.4,
        "mechanism_realization_rate": 0.4,
        "adapter_state_migration_overhead": 0.1,
    }
    unsafe_candidate = {
        **safe_candidate,
        "total_reward": 88.0,
        "handoff_failure_rate": 0.10,
        "backhaul_traffic_cost": 130.0,
    }

    safe_score = compute_pareto_safe_checkpoint_score(safe_candidate, reference_metrics=reference)
    unsafe_score = compute_pareto_safe_checkpoint_score(unsafe_candidate, reference_metrics=reference)

    assert safe_score["noninferiority_penalty"] == 0.0
    assert unsafe_score["noninferiority_penalty"] > 0.0
    assert safe_score["score"] > unsafe_score["score"]
