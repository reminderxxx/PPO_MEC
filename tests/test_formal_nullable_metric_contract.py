from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_algo_pool_real_sample, train_algo_pool_real_sample
from scripts.analyze_top_journal_statistics import summarize_deltas
from scripts.manage_typed_model_cache_formal_artifacts import dev_select
from scripts.run_typed_model_cache_formal_dev_selection import (
    dev_select as dev_selection_runner_consumer,
)
from src.metrics.formal_nullable_metrics import (
    NullableMetricContractError,
    canonical_sha256,
    reduce_nullable_metric_rows,
    reduce_nullable_values,
    validate_end_to_end_delay_reason,
)
from src.runtime.formal_invalid_run_registry import (
    PermanentlyInvalidFormalReferenceError,
    reject_permanently_invalid_formal_references,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("values", "mean", "available", "unavailable", "status"),
    [
        ([None], None, 0, 1, "unavailable_all_values"),
        ([0.0], 0.0, 1, 0, "available_complete"),
        ([None, 6.0], 6.0, 1, 1, "available_partial"),
        ([], None, 0, 0, "unavailable_no_rows"),
    ],
)
def test_frozen_reducer_examples(values, mean, available, unavailable, status) -> None:
    result = reduce_nullable_values(values, field="metric")
    assert result == {
        "total_count": len(values),
        "available_count": available,
        "unavailable_count": unavailable,
        "availability_status": status,
        "mean": mean,
    }


def test_required_missing_is_distinct_from_explicit_null() -> None:
    means, availability = reduce_nullable_metric_rows([{"metric": None}], ["metric"])
    assert means == {"metric": None}
    assert availability["metric"]["unavailable_count"] == 1
    with pytest.raises(NullableMetricContractError, match="required.*missing"):
        reduce_nullable_metric_rows([{}], ["metric"])


@pytest.mark.parametrize("value", [True, "not-a-number", float("nan"), float("inf"), float("-inf")])
def test_invalid_formal_values_fail_fast(value) -> None:
    with pytest.raises(NullableMetricContractError):
        reduce_nullable_values([value], field="metric")


def test_csv_blank_round_trip_and_canonical_hash() -> None:
    restored = next(csv.DictReader(["metric,other\n", ",0\n"]))
    means, availability = reduce_nullable_metric_rows(
        [restored],
        ["metric", "other"],
        csv_empty_is_null=True,
        numeric_strings_allowed=True,
    )
    assert means == {"metric": None, "other": 0.0}
    assert availability["metric"]["unavailable_count"] == 1
    assert availability["other"]["available_count"] == 1
    payload = {"mean_metrics": means, "availability": availability}
    assert canonical_sha256(payload) == canonical_sha256(json.loads(json.dumps(payload, allow_nan=False)))


def test_delay_value_and_reason_are_consistent() -> None:
    validate_end_to_end_delay_reason(6.0, "available_completed_workflow")
    validate_end_to_end_delay_reason(None, "unavailable_failed_or_incomplete_workflow")
    validate_end_to_end_delay_reason(None, "unavailable_right_censored_workflow")
    with pytest.raises(NullableMetricContractError):
        validate_end_to_end_delay_reason(0.0, "unavailable_failed_or_incomplete_workflow")


@pytest.mark.parametrize(
    "producer",
    [train_algo_pool_real_sample, eval_algo_pool_real_sample],
)
def test_algo_pool_mean_metrics_stays_scalar_and_preserves_zero(producer) -> None:
    row = {name: 2.0 for name in producer.SUMMARY_METRICS}
    row["end_to_end_workflow_delay"] = None
    row["workflow_continuity_rate"] = 0.0
    means = producer.metric_means([row])
    availability = producer.metric_availability([row])
    assert means["end_to_end_workflow_delay"] is None
    assert means["workflow_continuity_rate"] == 0.0
    assert all(value is None or isinstance(value, float) for value in means.values())
    assert availability["end_to_end_workflow_delay"]["unavailable_count"] == 1
    assert availability["workflow_continuity_rate"]["available_count"] == 1


def _candidate(
    update: int,
    digest: str,
    *,
    hit: float | None = 0.8,
    continuity: float | None = 0.7,
    transfer: float | None = 4.0,
    delay: float | None = 6.0,
) -> dict:
    return {
        "agent_name": "agent",
        "seed": 7,
        "capacity_label": "capacity",
        "update_index": update,
        "checkpoint_sha256": digest,
        "full_service_ready_byte_hit_rate": hit,
        "workflow_continuity_rate": continuity,
        "transfer_mb_per_request": transfer,
        "end_to_end_workflow_delay": delay,
        "non_formal_rehearsal": True,
    }


def _select(tmp_path: Path, rows: list[dict]) -> dict:
    (tmp_path / "checkpoint_candidates.json").write_text(
        json.dumps(rows, allow_nan=False), encoding="utf-8"
    )
    return dev_select(
        tmp_path,
        {
            "typed_model_cache_formal_protocol_version": "1.0.0",
            "hashes": {"semantic_sha256": "p" * 64},
            "training_budget": {
                "checkpoint_selection": {"metric_rule": "frozen"}
            },
        },
    )


def test_dev_selection_finite_delay_beats_null_and_null_is_not_zero(tmp_path: Path) -> None:
    result = _select(
        tmp_path,
        [_candidate(4, "b" * 64, delay=None), _candidate(8, "a" * 64, delay=9.0)],
    )
    assert result["selected"][0]["update_index"] == 8
    assert result["selection_metric_candidate_availability"]["end_to_end_workflow_delay"] == {
        "candidate_count": 2,
        "available_candidate_count": 1,
        "unavailable_candidate_count": 1,
        "dimension_participated": True,
    }


def test_dev_selection_skips_both_null_then_uses_update_sha_tie_break(tmp_path: Path) -> None:
    result = _select(
        tmp_path,
        [_candidate(8, "b" * 64, delay=None), _candidate(4, "c" * 64, delay=None)],
    )
    assert result["selected"][0]["update_index"] == 4
    assert not result["selection_metric_candidate_availability"]["end_to_end_workflow_delay"]["dimension_participated"]


def test_dev_selection_uses_checkpoint_sha_after_equal_update(tmp_path: Path) -> None:
    result = _select(
        tmp_path,
        [_candidate(4, "c" * 64, delay=None), _candidate(4, "a" * 64, delay=None)],
    )
    assert result["selected"][0]["checkpoint_sha256"] == "a" * 64


def test_dev_selection_runner_and_artifact_manager_share_exact_consumer() -> None:
    assert dev_selection_runner_consumer is dev_select


def test_dev_selection_skips_null_dimension_and_obeys_next_lower_is_better(tmp_path: Path) -> None:
    result = _select(
        tmp_path,
        [
            _candidate(4, "b" * 64, hit=None, continuity=0.8, transfer=9.0),
            _candidate(8, "a" * 64, hit=None, continuity=0.8, transfer=3.0),
        ],
    )
    assert result["selected"][0]["update_index"] == 8


def test_zero_pair_statistics_are_unavailable() -> None:
    summary = summarize_deltas([], 10, random.Random(7))
    assert summary["paired_count"] == 0
    for field in (
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "sign_test_pvalue",
    ):
        assert summary[field] is None


def test_statistics_reports_nullable_pair_coverage_and_lower_direction(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.csv"
    fieldnames = ["agent_name", "seed", "window_id", "workflow_id", "end_to_end_workflow_delay"]
    rows = [
        {"agent_name": "candidate", "seed": 7, "window_id": "w0", "workflow_id": "x", "end_to_end_workflow_delay": 4},
        {"agent_name": "baseline", "seed": 7, "window_id": "w0", "workflow_id": "x", "end_to_end_workflow_delay": 6},
        {"agent_name": "candidate", "seed": 7, "window_id": "w1", "workflow_id": "x", "end_to_end_workflow_delay": ""},
        {"agent_name": "baseline", "seed": 7, "window_id": "w1", "workflow_id": "x", "end_to_end_workflow_delay": 7},
        {"agent_name": "candidate", "seed": 7, "window_id": "w2", "workflow_id": "x", "end_to_end_workflow_delay": 5},
        {"agent_name": "baseline", "seed": 7, "window_id": "w2", "workflow_id": "x", "end_to_end_workflow_delay": ""},
        {"agent_name": "candidate", "seed": 7, "window_id": "w3", "workflow_id": "x", "end_to_end_workflow_delay": ""},
        {"agent_name": "baseline", "seed": 7, "window_id": "w3", "workflow_id": "x", "end_to_end_workflow_delay": ""},
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "statistics"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_top_journal_statistics.py"),
            "--rows_path", str(rows_path),
            "--candidate_agent", "candidate",
            "--baseline_agents", "baseline",
            "--metrics", "end_to_end_workflow_delay",
            "--pair_keys", "seed", "window_id", "workflow_id",
            "--bootstrap_samples", "20",
            "--output_root", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "paired_statistics.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["higher_is_better"] is False
    assert row["raw_mean_delta_candidate_minus_baseline"] == -2.0
    assert row["mean_delta"] == 2.0
    assert row["total_pair_count"] == 4
    assert row["available_paired_count"] == 1
    assert row["candidate_only_available_drop_count"] == 1
    assert row["baseline_only_available_drop_count"] == 1
    assert row["both_unavailable_drop_count"] == 1
    assert row["holm_available_family_size"] == 1


def test_v12_run_and_staging_checkpoint_references_are_rejected() -> None:
    with pytest.raises(PermanentlyInvalidFormalReferenceError, match="g14c_v12"):
        reject_permanently_invalid_formal_references(
            [
                ROOT
                / "artifacts/experiments/typed_model_cache_formal"
                / "typed_model_cache_formal_20260902_162203_g14c_v12"
                / ".staging/train/cell/checkpoints/update_0032.pt"
            ]
        )
