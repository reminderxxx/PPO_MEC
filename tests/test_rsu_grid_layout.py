from src.data.mobility.rsu_mapper import RSUMapper
from src.envs.specs import VehicleState
from src.evaluators.real_sample_support import build_sample_rsus


def test_auto_grid_tight_covers_two_dimensional_mobility_extent() -> None:
    frames = [
        {
            "time_index": 0,
            "vehicles": [
                VehicleState("v0", 0.0, 0.0, 1.0, "base"),
                VehicleState("v1", 1000.0, 0.0, 1.0, "base"),
                VehicleState("v2", 0.0, 1000.0, 1.0, "base"),
                VehicleState("v3", 1000.0, 1000.0, 1.0, "base"),
                VehicleState("v4", 500.0, 500.0, 1.0, "base"),
            ],
        }
    ]

    rsus, metadata = build_sample_rsus(frames=frames, rsu_layout="auto_grid_tight")
    associations = RSUMapper(rsus).associate(frames[0]["vehicles"])

    assert metadata["chosen_rsu_axis"] == "grid"
    assert metadata["rsu_count"] == 16
    assert all(rsu_id is not None for rsu_id in associations.values())
