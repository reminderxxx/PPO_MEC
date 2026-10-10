"""Synthetic boundary checks for the frozen raw-source eligibility gate."""

import pytest

from scripts.audit_cscwd_development_window_eligibility import (
    TARGET_FRAMES,
    allowed_contiguous_spans,
    gap_ms,
    qualify,
)


def _row(length: int, start: int = 1_500_000_000_000) -> dict:
    return {
        "plan_role": "dev",
        "plan_index": 0,
        "window_id": "synthetic",
        "source_segment_id": "us_101",
        "frame_offset": 0,
        "window_length": length,
        "time_index_start": start,
        "time_index_end": start + 100 * (length - 1),
    }


def _hits(row: dict) -> dict:
    start = row["time_index_start"]
    return {start + 100 * index: [(7, 1000 + index), (9, 2000 + index)]
            for index in range(row["window_length"])}


def test_short_authorized_interval_stays_ineligible_even_with_extra_raw_frames() -> None:
    row = _row(24)
    hits = _hits(row)
    hits[row["time_index_end"] + 100] = [(7, 1024)]
    result = qualify(row, hits, [], {"minimum_compute_only_seconds": 16.298588})
    assert result["selected_vehicle_id"] == 7
    assert result["selected_vehicle_contiguous_frames_from_start"] == 24
    assert result["decision_time_available_seconds"] == pytest.approx(2.2)
    assert result["qualified"] is False
    assert "below_any_workflow_compute_only_lower_bound" in result["rejection_reasons"]
    assert "below_frozen_engineering_source_duration" in result["rejection_reasons"]


def test_frozen_frame_target_is_eligible_only_without_split_overlap() -> None:
    row = _row(TARGET_FRAMES)
    threshold = {"minimum_compute_only_seconds": 16.298588}
    clean = qualify(row, _hits(row), [], threshold)
    assert clean["qualified"] is True
    assert clean["selected_vehicle_decision_available_seconds"] >= 118.7
    forbidden = [{**row, "plan_role": "formal"}]
    blocked = qualify(row, _hits(row), forbidden, threshold)
    assert blocked["qualified"] is False
    assert blocked["rejection_reasons"] == ["formal_or_hidden_interval_overlap"]


def test_disjoint_windows_cannot_be_stitched_across_a_gap() -> None:
    first = _row(24)
    second = _row(24, first["time_index_end"] + 2_500)
    assert gap_ms(first, second) == 2_500
    assert len(allowed_contiguous_spans([first, second])) == 2
