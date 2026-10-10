"""Synthetic source interval and provenance boundaries for new development data."""

import pandas as pd

from scripts.freeze_cscwd_new_development_source import (
    WINDOW_FRAMES,
    allowed_fragments,
    merge_blocks,
    qualify,
)


def _rows(vehicle_id: int, *, start: int = 1_500_000_000_000, count: int = WINDOW_FRAMES) -> pd.DataFrame:
    return pd.DataFrame({
        "source_segment_id": ["lankershim"] * count,
        "Vehicle_ID": [vehicle_id] * count,
        "Frame_ID": range(1, count + 1),
        "Global_Time": [start + 100 * index for index in range(count)],
        "Local_X": [1.0] * count,
        "Local_Y": [2.0] * count,
        "source_csv_row": range(count),
    })


def test_formal_interval_and_embargo_split_a_long_run() -> None:
    start = 1_500_000_000_000
    end = start + 1999 * 100
    blocks = merge_blocks([(start + 900 * 100 - 2400, start + 922 * 100 + 2400)])
    fragments = allowed_fragments(start, end, blocks)
    assert len(fragments) == 2
    data = _rows(7, start=start, count=2000)
    candidates, selected, _ = qualify(data, {"lankershim": blocks}, [])
    assert not selected
    assert all(row["rejection_reasons"] == ["below_frozen_1189_frame_source_span"]
               for row in candidates)


def test_same_time_vehicles_select_smallest_eligible_id_once() -> None:
    data = pd.concat([_rows(9), _rows(7)], ignore_index=True)
    data.sort_values(["source_segment_id", "Vehicle_ID", "Global_Time", "Frame_ID"],
                     kind="mergesort", inplace=True, ignore_index=True)
    candidates, selected, _ = qualify(data, {"lankershim": []}, [])
    assert len(selected) == 1
    assert selected[0]["vehicle_id"] == 7
    assert selected[0]["contiguous_frame_count"] == WINDOW_FRAMES
    assert next(row for row in candidates if row["vehicle_id"] == 9)["rejection_reasons"] == [
        "frozen_segment_quota_reached"
    ]
