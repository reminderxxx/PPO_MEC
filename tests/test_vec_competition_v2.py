"""Physical, causal and competition acceptance for the frozen eight requests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest

from src.envs.core.vec_competition_contract import ControllerAction, NearestCoveredAccess, RequestSpec
from src.envs.core.vec_competition_runtime import VECCompetitionRuntime
from src.envs.core.vec_competition_policies import ConstantVelocityPrefetchController
from src.evaluators.vec_competition_audit import audit_competition_run
from src.evaluators.vec_frozen_scenario import _source_path, load_frozen_scenario
from src.evaluators.vec_competition_scenario import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/experiment/vec_competition_peachtree_4rsu_8vehicle_v2.json"


@pytest.fixture(scope="module")
def frozen_scenario():
    return load_frozen_scenario(PROFILE, ROOT)[0]


def test_real_window_has_eight_unique_physically_continuous_tracks(frozen_scenario) -> None:
    scenario = frozen_scenario
    assert len(scenario.traces) == len(scenario.workflows) == len(scenario.requests) == 8
    assert len(scenario.rsus) == 4
    assert len({workflow.workflow_id for workflow in scenario.workflows.values()}) == 8
    assert all(len(trace) == 120 for trace in scenario.traces.values())
    assert all(request.arrival_s == 0.0 for request in scenario.requests.values())
    scenario.validate()


def test_eight_requests_share_resources_and_deliver_causally(frozen_scenario) -> None:
    first = VECCompetitionRuntime(frozen_scenario)
    result = first.run()
    second = VECCompetitionRuntime(frozen_scenario)
    second.run()
    assert result["simultaneous_requests_at_start"] == result["generated_requests"] == 8
    assert result["rsu_count"] == 4 and result["vehicle_count"] == 8
    assert result["completed_nodes"] == result["total_nodes"] == 70
    assert result["delivered_on_time"] + result["delivered_late"] == 8
    assert result["handoffs"] >= 8 and result["state_migrations"] >= 1
    assert result["compute_queue_wait_s"] > 0
    assert first.ledger_digest() == second.ledger_digest()
    assert audit_competition_run(frozen_scenario, first.ledger, result)["passed"]
    assert len(first.association_timeline) == 120


def test_one_vehicle_workload_changes_competitors_completion_time(frozen_scenario) -> None:
    baseline = VECCompetitionRuntime(frozen_scenario).run()
    heavier = deepcopy(frozen_scenario)
    heavier.workflows["peachtree:832"].nodes[0].output_size *= 100
    changed = VECCompetitionRuntime(heavier).run()
    others = [vehicle_id for vehicle_id in baseline["per_vehicle"] if vehicle_id != "peachtree:832"]
    assert any(
        changed["per_vehicle"][vid]["delivered_s"] is None
        or changed["per_vehicle"][vid]["delivered_s"] > baseline["per_vehicle"][vid]["delivered_s"]
        for vid in others
    )
    assert changed["delivered_on_time"] <= baseline["delivered_on_time"]


def test_two_large_uplinks_compete_for_same_radio(frozen_scenario) -> None:
    baseline = deepcopy(frozen_scenario)
    heavier = deepcopy(frozen_scenario)
    for scenario in (baseline, heavier):
        scenario.workflows["peachtree:835"].nodes[0].input_size = 20_000
    heavier.workflows["peachtree:837"].nodes[0].input_size = 20_000

    def completed_uplink(scenario):
        env = VECCompetitionRuntime(scenario)
        env.run()
        node_id = scenario.workflows["peachtree:835"].nodes[0].node_id
        return min(event["time_s"] for event in env.ledger if event["event"] == "flow_completed" and event["kind"] == "uplink" and event["vehicle_id"] == "peachtree:835" and event["node_id"] == node_id)

    assert completed_uplink(heavier) > completed_uplink(baseline)


def test_independent_auditor_rejects_fabricated_transfer(frozen_scenario) -> None:
    env = VECCompetitionRuntime(frozen_scenario)
    result = env.run()
    forged = deepcopy(env.ledger)
    progress = next(event for event in forged if event["event"] == "flow_progress")
    progress["sent_bytes"] += 1_000_000
    assert not audit_competition_run(frozen_scenario, forged, result)["passed"]


def test_frozen_road_and_identity_fail_closed(frozen_scenario) -> None:
    cloned = deepcopy(frozen_scenario)
    ids = sorted(cloned.traces)
    cloned.traces[ids[1]] = cloned.traces[ids[0]]
    with pytest.raises(ValueError, match="cloned vehicle trajectory"):
        cloned.validate()
    illegal = deepcopy(frozen_scenario)
    vehicle_id = ids[0]
    trace = list(illegal.traces[vehicle_id])
    trace[1] = replace(trace[1], road_key="unmapped:teleport")
    illegal.traces[vehicle_id] = tuple(trace)
    with pytest.raises(ValueError, match="illegal road transition"):
        illegal.validate()


def test_raw_duplicate_vehicle_time_is_rejected(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text())
    vehicle_ids = profile["mobility"]["vehicle_ids"]
    fake_csv = tmp_path / "ngsim_duplicate.csv"
    columns = ["Location", "Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "Section_ID", "Lane_ID", "Direction"]
    with fake_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, vehicle_id in enumerate(vehicle_ids):
            for time_ms in (0, 100):
                writer.writerow({"Location": "peachtree", "Vehicle_ID": vehicle_id.split(":", 1)[1], "Global_Time": time_ms, "Local_X": index, "Local_Y": time_ms / 100, "v_Vel": 10, "Section_ID": 1, "Lane_ID": 1, "Direction": 2})
        writer.writerow({"Location": "peachtree", "Vehicle_ID": vehicle_ids[0].split(":", 1)[1], "Global_Time": 0, "Local_X": 900, "Local_Y": 900, "v_Vel": 10, "Section_ID": 1, "Lane_ID": 1, "Direction": 4})
    profile["mobility"].update({"relative_path": str(fake_csv), "sha256": sha256_file(fake_csv), "first_time_ms": 0, "frame_count": 2, "rows_to_scan": 17, "allowed_road_edges": []})
    profile["workflow"]["relative_path"] = str(_source_path(ROOT, profile["workflow"]["relative_path"]))
    profile["model_catalog"]["relative_path"] = str(_source_path(ROOT, profile["model_catalog"]["relative_path"]))
    fake_profile = tmp_path / "profile.json"
    fake_profile.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="row_count=2"):
        load_frozen_scenario(fake_profile, tmp_path)


def test_deadline_failure_stays_in_request_denominator(frozen_scenario) -> None:
    scenario = deepcopy(frozen_scenario)
    vehicle_id = sorted(scenario.requests)[0]
    scenario.requests[vehicle_id] = RequestSpec(vehicle_id, 0.0, 0.2)
    env = VECCompetitionRuntime(scenario)
    result = env.run()
    assert result["generated_requests"] == 8
    assert result["deadline_missed"] >= 1
    assert result["per_vehicle"][vehicle_id]["status"] != "on_time"
    assert result["delivered_on_time"] < 8


def test_coverage_outage_is_visible_and_blocks_delivery(frozen_scenario) -> None:
    class OutageAfterStart(NearestCoveredAccess):
        version = "test_outage"

        def associate(self, point, current_rsu_id, rsus):
            return None if point.time_s >= 0.1 else super().associate(point, current_rsu_id, rsus)

    env = VECCompetitionRuntime(frozen_scenario, access_policy=OutageAfterStart())
    result = env.run()
    assert result["unfinished"] == 8
    assert sum(event["event"] == "coverage_lost" for event in env.ledger) == 8
    assert sum(event["event"] == "request_unfinished" for event in env.ledger) == 8


def test_cancelled_base_load_releases_dependent_reservation(frozen_scenario) -> None:
    env = VECCompetitionRuntime(frozen_scenario)
    base_flow = next(flow_id for flow_id, flow in env.flows.items() if flow.kind == "model_load" and flow.object_id == "base:veh_base_v1" and flow.target_rsu_id == "rsu_d")
    env.cancel_flow(base_flow)
    assert ("rsu_d", "base:veh_base_v1") not in env.reserved
    assert ("rsu_d", "adapter:batch_type_1") not in env.reserved
    assert not any(flow.kind == "model_load" and flow.target_rsu_id == "rsu_d" and flow.object_id == "adapter:batch_type_1" for flow in env.flows.values())


def test_observation_contains_only_current_state(frozen_scenario) -> None:
    env = VECCompetitionRuntime(frozen_scenario)
    observation = env.observe()
    assert observation["time_s"] == 0.0
    assert "future_trace" not in observation
    assert len(observation["vehicles"]) == 8
    assert env.action_mask("peachtree:832", "task_1")["prepare"] == sorted(r.rsu_id for r in frozen_scenario.rsus)


def test_predictive_plugin_only_uses_past_and_records_error(frozen_scenario) -> None:
    env = VECCompetitionRuntime(frozen_scenario)
    controller = ConstantVelocityPrefetchController(frozen_scenario.rsus)
    assert controller.actions(env.observe()) == ()  # no velocity can be inferred at t=0
    result = env.run(controller)
    made = [event for event in env.ledger if event["event"] == "prediction_made"]
    resolved = [event for event in env.ledger if event["event"] == "prediction_resolved"]
    assert made and all(event["time_s"] > 0 for event in made)
    assert result["resolved_predictions"] == len(resolved)
    assert all("eta_error_s" in event and "target_correct" in event for event in resolved)


def test_wrong_prediction_keeps_transfer_exposure_in_ledger(frozen_scenario) -> None:
    scenario = deepcopy(frozen_scenario)
    # Give this vehicle a later adapter that no car requests at t=0, so the
    # deliberately wrong prepare causes incremental bytes at the wrong site.
    node = scenario.workflows["peachtree:835"].nodes[1]
    node.required_adapter = "adapter_intent"

    class WrongTargetOnce:
        version = "test_wrong_target"

        def __init__(self):
            self.sent = False

        def actions(self, observation):
            if self.sent:
                return ()
            self.sent = True
            return (ControllerAction("prepare", "peachtree:835", node.node_id, "rsu_a", 0.5, self.version),)

    env = VECCompetitionRuntime(scenario)
    result = env.run(WrongTargetOnce())
    assert result["wrong_predictions"] >= 1
    assert result["wrong_target_prefetch_bytes_sent_at_handoff"] > 0
