"""Causal checks for the four-RSU/eight-vehicle replay contract."""

from pathlib import Path

import pytest

from src.data.model_catalog.adapter_catalog import AdapterCatalog, RSUTypedCacheProfile
from src.envs.core.vec_competition_env import CompetitionScenario, RoadsideUnit, TracePoint, VECCompetitionEnv
from src.envs.specs import WorkflowGraphState, WorkflowNode


CATALOG = Path(__file__).resolve().parents[1] / "src/data/model_catalog/typed_model_cache_controlled.json"


def make_scenario(*, cold_start: bool = False) -> CompetitionScenario:
    catalog = AdapterCatalog.from_json(CATALOG)
    resident = ["base:veh_base_v1"] if cold_start else ["base:veh_base_v1", "adapter:batch_type_1"]
    catalog.rsu_typed_cache_profiles = [RSUTypedCacheProfile(f"rsu_{letter}", list(resident)) for letter in "abcd"]
    rsus = tuple(RoadsideUnit(f"rsu_{letter}", index * 100.0, 0.0, 70.0, 512.0, 1, 10.0, 1000.0, 40.0) for index, letter in enumerate("abcd"))
    traces = {}
    workflows = {}
    for index in range(8):
        vehicle_id = f"corridor:{index}"
        traces[vehicle_id] = tuple(TracePoint(1000 + step * 100, 0.0 if step < 4 else 100.0, float(index), 10.0) for step in range(80))
        first = WorkflowNode("first", "first", "veh_base_v1", "adapter_batch_type_1", 1000, 1000, [], ["second"])
        second = WorkflowNode("second", "second", "veh_base_v1", "adapter_batch_type_1", 1000, 1000, ["first"], [])
        workflows[vehicle_id] = WorkflowGraphState(f"job_{index}", [first, second], [("first", "second")], ["first", "second"], current_node_id="first")
    return CompetitionScenario("corridor", traces, workflows, rsus, catalog)


def test_concurrent_replay_handoff_state_and_reproducibility() -> None:
    scenario = make_scenario()
    first = VECCompetitionEnv(scenario)
    summary = first.run()
    second = VECCompetitionEnv(scenario)
    second.run()
    assert summary["vehicle_count"] == 8 and summary["rsu_count"] == 4
    assert summary["handoffs"] == 8
    assert summary["state_migrations"] >= 1
    assert summary["completed_nodes"] == summary["total_nodes"] == 16
    assert summary["delivered_workflows"] == 8
    assert first.ledger_digest() == second.ledger_digest()
    assert any(e["event"] == "transfer_completed" and e.get("kind") == "state_migration" and e["duration_ms"] > 0 for e in first.ledger)


def test_cold_adapter_cannot_execute_before_transfer_completes() -> None:
    env = VECCompetitionEnv(make_scenario(cold_start=True))
    for _ in range(5):
        env.step()
    assert not any(e["event"] == "compute_started" for e in env.ledger)
    assert any(e["event"] == "transfer_requested" and e.get("kind") == "model_load" for e in env.ledger)
    assert "adapter:batch_type_1" not in env.residents["rsu_a"]


def test_requires_exactly_eight_synchronised_traces() -> None:
    scenario = make_scenario()
    scenario.traces.pop(next(iter(scenario.traces)))
    with pytest.raises(ValueError, match="exactly four RSUs and eight"):
        scenario.validate()
