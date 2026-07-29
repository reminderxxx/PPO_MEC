import json
from pathlib import Path

from scripts.benchmark_main_results import (
    load_excluded_window_intervals,
    load_frozen_benchmark_window_payload,
)


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


def test_load_frozen_benchmark_window_payload_uses_plan_without_scan(tmp_path: Path) -> None:
    plan_path = tmp_path / "aggregate_summary.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol_version": "formal_v1",
                "split": "formal",
                "outcome_blind_selection": True,
                "mobility_source_path": "/tmp/ngsim.csv",
                "selected_window_plan": [
                    {
                        "window_id": "window_a",
                        "frame_offset": 10,
                        "window_length": 24,
                        "window_class": "idle_or_sparse",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    mobility_source_path, payload = load_frozen_benchmark_window_payload(plan_path)

    assert mobility_source_path == "/tmp/ngsim.csv"
    assert payload["frozen_window_plan_protocol_version"] == "formal_v1"
    assert payload["frozen_window_plan_split"] == "formal"
    assert payload["outcome_blind_selection"] is True
    assert payload["idle_or_sparse_windows"][0]["window_id"] == "window_a"
