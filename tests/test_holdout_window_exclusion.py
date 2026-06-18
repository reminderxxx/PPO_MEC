import json
from pathlib import Path

from scripts.benchmark_main_results import load_excluded_window_intervals


def test_load_excluded_window_intervals_from_aggregate(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "aggregate_summary.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "selected_window_plan": [
                    {"frame_offset": 10, "window_length": 24},
                    {"frame_offset": 80, "window_length": 12},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_excluded_window_intervals([str(aggregate_path)]) == [(10, 33), (80, 91)]
