"""Synthetic, fixed two-RSU physical-environment acceptance run.

The scenario is a deterministic engineering fixture. It does not measure a
real model, vehicle radio, or production deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.physical_vec_workflow_env import PhysicalVecWorkflowCoreEnv
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.physical_event_metrics import PHYSICAL_EVENT_SCHEMA_VERSION, reduce_physical_events
from src.runtime.physical_cost_profile import canonical_identity


SCENARIO_ID = "two_rsu_two_vehicle_small_dag_synthetic_v1"
SEED = 71
ACTION_SEQUENCE = (0, 0, 0, 1, 0, 0, 4, 3, 3, 3, 3, 3)
PARAMETERS = {
    "control_step_s": 1.0,
    "backhaul_bytes_per_s": 100_000_000.0,
    "wireless_bytes_per_s": 100_000_000.0,
    "compute_slots_per_rsu": 1,
    "compute_rate_work_units_per_s": 32.0,
    "cache_capacity_bytes_per_rsu": 400_000_000,
}


def _graph() -> WorkflowGraphState:
    nodes = [
        WorkflowNode("n0", "perception", "veh_base_v1", "adapter_perception", 16, 32, successors=["n1"]),
        WorkflowNode("n1", "tracking", "veh_base_v1", "adapter_tracking", 32, 24, predecessors=["n0"]),
    ]
    return WorkflowGraphState("acceptance_wf", nodes, [("n0", "n1")], ["n0", "n1"], current_node_id="n0")


def _frames() -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for step in range(40):
        x_primary = 0.0 if step < 7 else 100.0
        frames.append({
            "time_index": step,
            "vehicles": [
                {"vehicle_id": "vehicle-1", "position_x": x_primary, "position_y": 0.0, "speed": 10.0,
                 "base_model_id": "veh_base_v1", "active_workflow_id": "acceptance_wf"},
                {"vehicle_id": "vehicle-2", "position_x": 0.0, "position_y": 0.0, "speed": 0.0,
                 "base_model_id": "veh_base_v1", "active_workflow_id": "background_fixture"},
            ],
        })
    return frames


def _background_jobs() -> list[dict[str, Any]]:
    return [
        {"time_s": 0.0, "vehicle_id": "vehicle-2", "kind": "transfer",
         "resource_id": "backhaul", "size_bytes": 120_000_000},
        {"time_s": 0.0, "vehicle_id": "vehicle-2", "kind": "compute",
         "resource_id": "rsu-0", "work_units": 96.0},
    ]


def _run_once(*, shared_load: bool) -> dict[str, Any]:
    env = PhysicalVecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=_frames()),
        workflow_state=_graph(),
        rsu_states=[
            RSUState("rsu-0", 0.0, 0.0, 60.0),
            RSUState("rsu-1", 100.0, 0.0, 60.0),
        ],
        max_steps=40,
        background_jobs=_background_jobs() if shared_load else [],
        **PARAMETERS,
    )
    wrapped = GymVecEnv(core_env=env)
    _, reset_info = wrapped.reset(seed=SEED)
    steps: list[dict[str, Any]] = []
    final_state: dict[str, Any] = reset_info["semantic_state"]
    for index in range(40):
        action = ACTION_SEQUENCE[index] if index < len(ACTION_SEQUENCE) else 3
        _, reward, terminated, truncated, info = wrapped.step(action)
        final_state = info["semantic_state"]
        steps.append({
            "step": index + 1,
            "action_id": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "physical_events": info.get("physical_events", []),
            "physical_state": final_state.get("physical_state"),
        })
        if terminated or truncated:
            break
    events = env.export_physical_events()
    summary = reduce_physical_events(events)
    return {
        "steps": steps,
        "events": events,
        "summary": summary.to_dict(),
        "final_state": final_state,
    }


def _assert_acceptance(main: dict[str, Any], replay: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    assert main["events"] == replay["events"], "same seed and inputs must replay the same event ledger"
    assert main["summary"] == replay["summary"], "independent ledger reduction must replay identically"
    events = main["events"]
    assert events, "scenario emitted no physical events"
    assert main["summary"]["requests"], "scenario emitted no request metrics"

    model_wait = [row for row in events if row.get("event_type") == "wait_started" and row.get("wait_reason") == "waiting_model"]
    assert model_wait, "model readiness must produce a real wait event"
    model_transfers = [row for row in events if row.get("event_type") == "transfer_started" and row.get("transfer_kind") == "model_load"]
    assert model_transfers, "cache preparation must start model transfer"
    completions = {row.get("operation_id"): row for row in events if row.get("event_type") == "transfer_completed"}
    assert any(
        op.get("operation_id") in completions
        and completions[op["operation_id"]]["time_s"] - op["time_s"] > PARAMETERS["control_step_s"]
        for op in model_transfers
    ), "at least one cache load must cross a control step"

    def first_primary_model_completion(run: dict[str, Any]) -> float:
        started = {row["operation_id"] for row in run["events"]
                   if row.get("event_type") == "transfer_started" and row.get("transfer_kind") == "model_load"
                   and str(row.get("request_id", "")).startswith("acceptance_wf:vehicle-1:prepare:")}
        times = [row["time_s"] for row in run["events"]
                 if row.get("event_type") == "transfer_completed" and row.get("operation_id") in started]
        assert times, "primary vehicle model transfer did not finish"
        return min(times)

    assert first_primary_model_completion(main) > first_primary_model_completion(baseline), (
        "second vehicle sharing the bottleneck must delay primary completion"
    )
    def first_node_start(run: dict[str, Any], node_id: str) -> float:
        return min(row["time_s"] for row in run["events"]
                   if row.get("event_type") == "compute_started" and row.get("node_id") == node_id)
    assert first_node_start(main, "n0") > first_node_start(baseline, "n0"), (
        "second vehicle occupying the compute slot must delay primary compute"
    )

    compute = [row for row in events if row.get("event_type") == "compute_started"]
    assert len(compute) >= 2, "small DAG must run both nodes"
    n0_done = min(row["time_s"] for row in events
                  if row.get("event_type") == "compute_completed" and row.get("node_id") == "n0")
    n1_start = min(row["time_s"] for row in compute if row.get("node_id") == "n1")
    assert n1_start >= n0_done, "successor started before predecessor output existed"
    physical_state = main["final_state"].get("physical_state", {})
    node_rows = {row["node_id"]: row for row in physical_state.get("nodes", [])}
    assert node_rows.get("n0", {}).get("output_object_id"), "predecessor output identity missing"
    assert node_rows.get("n1", {}).get("status") == "completed", "successor did not complete"
    assert node_rows["n0"]["output_object_id"] in node_rows["n1"]["input_object_ids"], (
        "successor input must be the actual predecessor output object"
    )

    handoffs = physical_state.get("handoff_receipts", [])
    assert node_rows["n1"]["deployment_rsu_id"] == "rsu-0", "running task moved with access handoff"
    assert node_rows["n1"]["started_at"] < 7 < node_rows["n1"]["completed_at"], (
        "old-RSU compute must continue across the t=7 access handoff"
    )
    assert any(row.get("kind") == "forward_result" and row.get("delivered_to_current_access")
               and row.get("source_rsu_id") == "rsu-0" and row.get("target_rsu_id") == "rsu-1"
               and row.get("object_id") == node_rows["n1"]["output_object_id"]
               for row in handoffs), (
        "handoff must continue old-RSU work and forward its result to current access"
    )
    assert main["summary"]["availability"] == "available", "request did not complete"
    return {
        "deterministic_replay": True,
        "shared_resource_contention": True,
        "cross_step_cache_wait": True,
        "dag_output_dependency": True,
        "handoff_continue_and_forward": True,
        "independent_event_reduction": True,
    }


def run_acceptance(output_path: str | Path) -> dict[str, Any]:
    """Run fixture, assert the mechanism, and write a fully auditable JSON."""
    main = _run_once(shared_load=True)
    replay = _run_once(shared_load=True)
    baseline = _run_once(shared_load=False)
    checks = _assert_acceptance(main, replay, baseline)
    scenario = {
        "scenario_id": SCENARIO_ID,
        "environment_version": "physical_vec_env_v1",
        "physical_event_schema_version": PHYSICAL_EVENT_SCHEMA_VERSION,
        "parameter_evidence": "synthetic",
        "time_unit": "s",
        "size_unit": "byte",
        "seed": SEED,
        "parameters": PARAMETERS,
        "actions": ACTION_SEQUENCE,
        "frames": _frames(),
        "workflow": _graph().to_dict(),
        "rsus": [RSUState("rsu-0", 0.0, 0.0, 60.0).to_dict(), RSUState("rsu-1", 100.0, 0.0, 60.0).to_dict()],
        "background_jobs": _background_jobs(),
    }
    result = {
        "scenario": scenario,
        "scenario_sha256": canonical_identity(scenario),
        "checks": checks,
        "event_ledger": main["events"],
        "independent_event_summary": main["summary"],
        "step_trace": main["steps"],
        "final_state": main["final_state"],
        "baseline_summary": baseline["summary"],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    result = run_acceptance(args.output_path)
    print(json.dumps({"output_path": str(args.output_path), "checks": result["checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
