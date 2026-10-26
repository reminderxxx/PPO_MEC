import hashlib
import json

import pytest

from scripts.analyze_handoff_state_runtime import analyze
from scripts.run_realtime_handoff_recovery import forecast_target
from src.envs.specs import RSUState


def test_incomplete_study_cannot_be_turned_into_summary(tmp_path):
    (tmp_path/"completion.json").write_text(json.dumps({"status":"FAIL"}))
    with pytest.raises(ValueError,match="not complete"):analyze(tmp_path)


@pytest.mark.parametrize("drift",["inputs","plan"])
def test_frozen_input_drift_fails_before_reduction(tmp_path,drift):
    (tmp_path/"completion.json").write_text(json.dumps({"status":"PASS"}))
    for name in ("inputs","plan"):(tmp_path/f"{name}.json").write_text("{}")
    digest=hashlib.sha256(b"{}").hexdigest()
    (tmp_path/"execution_freeze.json").write_text(json.dumps({"inputs_sha256":digest,"plan_sha256":digest}))
    (tmp_path/f"{drift}.json").write_text('{"changed":true}')
    with pytest.raises(ValueError,match="drift"):analyze(tmp_path)


def test_realtime_prediction_uses_physical_time_and_no_future_position():
    def observation(time,x):return {"time_index":time,"vehicle":{"vehicle_id":"v",
        "position_x":x,"position_y":0,"speed":1,"base_model_id":"veh_base_v1"}}
    rsus=[RSUState("rsu_a",0,0,60),RSUState("rsu_b",100,0,60)]
    assert forecast_target(observation(0,40),observation(100,41),rsus,6)=="rsu_b"
    assert forecast_target(observation(0,40),observation(100,41),rsus,.2)=="rsu_a"
    with pytest.raises(ValueError):forecast_target(observation(0,40),observation(0,41),rsus,6)
