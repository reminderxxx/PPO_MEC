from copy import deepcopy

from src.data.mobility.rsu_mapper import RSUMapper
from src.envs.core.joint_causal_observation import CausalMobilityPredictor, SharedStateChannel
from src.envs.specs import RSUState, VehicleState


def test_future_load_projects_positions_and_scores_only_mature_forecasts():
    rsus = [RSUState("a", 0, 0, 100), RSUState("b", 10, 0, 100)]
    mapper = RSUMapper(rsus)
    predictor = CausalMobilityPredictor(horizon=3)
    for step, x in enumerate([0, 2, 4, 6, 8]):
        vehicles = [VehicleState("v", x, 0, 2, "base")]
        result = predictor.predict(vehicles, mapper.associate(vehicles), rsus, mapper, step)
        assert predictor.predict(vehicles, mapper.associate(vehicles), rsus, mapper, step) == result
        if step == 1:
            assert mapper.associate(vehicles)["v"] == "a"
            assert result["future_load"] == {"a": 0.0, "b": 1.0}
            assert result["predicted_next_rsu_by_vehicle"]["v"] == "b"
            assert result["next_rsu_sequence"]["v"] == ["a", "b", "b"]
            assert result["prediction_confidence_by_vehicle"]["v"] == 0
        if step < 3:
            assert result["prediction_observed_sample_count_by_vehicle"]["v"] == 0
    assert result["prediction_observed_sample_count_by_vehicle"]["v"] == 2
    assert result["prediction_confidence_by_vehicle"]["v"] == 0.5


def test_future_tail_cannot_change_prefix_forecasts():
    rsus = [RSUState("a", 0, 0, 100), RSUState("b", 10, 0, 100)]
    mapper = RSUMapper(rsus)
    results = []
    for tail in ([6, 8], [-20, -40]):
        predictor = CausalMobilityPredictor()
        prefix = []
        for step, x in enumerate([0, 2, 4] + tail):
            vehicles = [VehicleState("v", x, 0, 2, "base")]
            prefix.append(predictor.predict(vehicles, mapper.associate(vehicles), rsus, mapper, step))
        results.append(prefix)
    assert results[0][:3] == results[1][:3]


def test_shared_snapshot_has_delay_cost_and_no_live_remote_leak():
    channel = SharedStateChannel(interval_frames=2, delay_frames=1)
    rsus = [{"rsu_id": "a", "cached_adapter_ids": ["old"]},
            {"rsu_id": "b", "cached_adapter_ids": ["remote_old"]}]
    first = channel.snapshot(rsus, 0)
    assert not first["metadata"]["available"]
    assert first["metadata"]["transmitted_bytes"] > 0
    assert channel.snapshot(rsus, 0) == first
    assert "cached_adapter_ids" not in channel.actor_rsus(rsus, "a", first)[1]
    changed = deepcopy(rsus)
    changed[1]["cached_adapter_ids"] = ["remote_new"]
    second = channel.snapshot(changed, 1)
    assert second["metadata"]["age_frames"] == 1
    assert second["metadata"]["transmitted_bytes"] == 0
    assert channel.actor_rsus(changed, "a", second)[1]["cached_adapter_ids"] == ["remote_old"]
    channel.snapshot(changed, 2)
    fourth = channel.snapshot(changed, 3)
    assert fourth["rsus"][1]["cached_adapter_ids"] == ["remote_new"]
    assert fourth["metadata"]["total_transmitted_bytes"] > first["metadata"]["total_transmitted_bytes"]


def test_disabled_sharing_retains_local_data_without_remote_information_or_cost():
    channel = SharedStateChannel(enabled=False)
    rsus = [{"rsu_id": "a", "cached_adapter_ids": ["local"]},
            {"rsu_id": "b", "cached_adapter_ids": ["secret"]}]
    snap = channel.snapshot(rsus, 0)
    assert snap["metadata"]["total_transmitted_bytes"] == 0
    rows = channel.actor_rsus(rsus, "a", snap)
    assert rows[0]["cached_adapter_ids"] == ["local"]
    assert "cached_adapter_ids" not in rows[1]
