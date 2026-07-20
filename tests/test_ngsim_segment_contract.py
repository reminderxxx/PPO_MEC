from __future__ import annotations

from pathlib import Path

from src.data.mobility.ngsim_provider import NGSIMProvider
from src.envs.specs import VehicleState
from src.evaluators.real_sample_support import scan_mobility_windows


def test_ngsim_provider_groups_by_location_and_global_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "ngsim.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y,v_Vel,Location",
                "1,10,1000,0.0,0.0,10.0,alpha",
                "2,10,1000,1.0,0.0,10.0,alpha",
                "3,10,2000,50.0,0.0,12.0,beta",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    provider = NGSIMProvider(csv_path=csv_path, max_rows=3)
    frames = provider.get_loaded_frames()

    assert len(frames) == 2
    assert frames[0]["source_segment_id"] == "alpha"
    assert frames[0]["time_index"] == 1000
    assert len(frames[0]["vehicles"]) == 2
    assert frames[0]["vehicles"][0].vehicle_id.startswith("alpha:")
    assert frames[1]["source_segment_id"] == "beta"
    assert frames[1]["time_index"] == 2000
    assert len(frames[1]["vehicles"]) == 1


def test_scan_mobility_windows_does_not_cross_source_segments() -> None:
    frames = []
    for index in range(4):
        frames.append(
            {
                "time_index": index,
                "source_segment_id": "alpha",
                "source_location": "alpha",
                "segment_frame_index": index,
                "vehicles": [
                    VehicleState(
                        vehicle_id="alpha:1",
                        position_x=float(index),
                        position_y=0.0,
                        speed=1.0,
                        base_model_id="veh_base_v1",
                    )
                ],
            }
        )
    for index in range(4):
        frames.append(
            {
                "time_index": 100 + index,
                "source_segment_id": "beta",
                "source_location": "beta",
                "segment_frame_index": index,
                "vehicles": [
                    VehicleState(
                        vehicle_id="beta:1",
                        position_x=float(index),
                        position_y=10.0,
                        speed=1.0,
                        base_model_id="veh_base_v1",
                    )
                ],
            }
        )

    windows = scan_mobility_windows(
        frames=frames,
        layout_candidates=["tight_x"],
        frame_offset=0,
        window_length=3,
        stride=1,
    )

    assert windows
    assert all(window["source_segment_id"] in {"alpha", "beta"} for window in windows)
    assert all(window["segment_frame_end"] - window["segment_frame_start"] == 2 for window in windows)
    assert {window["frame_offset"] for window in windows}.isdisjoint({2, 3})
