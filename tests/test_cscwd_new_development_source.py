"""Synthetic source interval and provenance boundaries for new development data."""

import pandas as pd

from scripts.freeze_cscwd_new_development_source import (
    WINDOW_FRAMES,
    allowed_fragments,
    load_source,
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


def test_source_reader_accepts_quoted_thousands_in_coordinates(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(
        "Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y,Location\n"
        '7,1,1500000000000,"1,004.323",2.5,us-101\n'
        '7,2,1500000000100,"1,005.323",2.6,us-101\n'
    )
    data, prefix, scanned, invalid = load_source(path)
    assert scanned == 2 and invalid == 0
    assert prefix == {"us_101": [1_500_000_000_000, 1_500_000_000_100]}
    assert data["Local_X"].tolist() == [1004.323, 1005.323]
