import json
from pathlib import Path

from scripts.audit_window_independence import build_report


def write_summary(path: Path, offsets: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_window_plan": [
                    {"window_id": f"w{offset}", "frame_offset": offset, "window_length": 24}
                    for offset in offsets
                ]
            }
        ),
        encoding="utf-8",
    )


def write_segmented_summary(path: Path, windows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"selected_window_plan": windows}), encoding="utf-8")


def test_independent_plans_pass(tmp_path: Path) -> None:
    formal = tmp_path / "formal.json"
    holdout = tmp_path / "holdout.json"
    write_summary(formal, [0, 30])
    write_summary(holdout, [60, 90])
    assert build_report(formal, holdout)["passed"] is True


def test_overlapping_plans_fail(tmp_path: Path) -> None:
    formal = tmp_path / "formal.json"
    holdout = tmp_path / "holdout.json"
    write_summary(formal, [0, 30])
    write_summary(holdout, [20, 90])
    report = build_report(formal, holdout)
    assert report["passed"] is False
    assert report["formal_holdout_overlaps"]


def test_different_source_segments_do_not_conflict_on_reused_time_indices(tmp_path: Path) -> None:
    formal = tmp_path / "formal.json"
    holdout = tmp_path / "holdout.json"
    write_segmented_summary(
        formal,
        [
            {
                "window_id": "alpha",
                "frame_offset": 0,
                "window_length": 24,
                "time_index_start": 100,
                "time_index_end": 123,
                "source_segment_id": "alpha",
                "segment_frame_start": 0,
                "segment_frame_end": 23,
            }
        ],
    )
    write_segmented_summary(
        holdout,
        [
            {
                "window_id": "beta",
                "frame_offset": 0,
                "window_length": 24,
                "time_index_start": 100,
                "time_index_end": 123,
                "source_segment_id": "beta",
                "segment_frame_start": 0,
                "segment_frame_end": 23,
            }
        ],
    )

    assert build_report(formal, holdout, gap=24)["passed"] is True
