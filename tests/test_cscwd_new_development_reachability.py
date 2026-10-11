"""Synthetic loader and legality checks for bounded raw reachability."""

from scripts.audit_cscwd_new_development_reachability import legal_action, load_traces


def test_selected_trace_loader_requires_exact_vehicle_time_and_frame(tmp_path) -> None:
    start = 1_500_000_000_000
    path = tmp_path / "raw.csv"
    with path.open("w") as stream:
        stream.write("Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y,Location\n")
        for index in range(1189):
            stream.write(f'7,{20 + index},{start + 100 * index},"1,004.323",{index},us-101\n')
    window = {"window_id": "synthetic", "decision": "selected",
              "source_segment_id": "us_101", "vehicle_id": 7,
              "source_time_start": start, "source_time_end": start + 118800,
              "source_frame_start": 20, "source_frame_end": 1208,
              "contiguous_frame_count": 1189,
              "historical_train_dev_overlap_count": 0}
    traces, metadata = load_traces(path, [window])
    assert traces["synthetic"].times_seconds[-1] == 118.8
    assert traces["synthetic"].xy_metres[0][0] == 1004.323 * 0.3048
    assert metadata["window_identities"]["synthetic"]["raw_source_row_count"] == 1189


def test_action_fallback_uses_public_mask() -> None:
    assert legal_action(4, [False, False, True, False, False]) == 2
    assert legal_action(3, [False, False, True, True, False]) == 3
