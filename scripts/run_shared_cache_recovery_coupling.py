"""Run the frozen shared-cache × workflow-recovery coupling witness.

This script executes only normal ``GymVecEnv.step`` transitions.  It does not
train an agent and does not mutate the cache outside the environment contract.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_native_typed_cache_probe import build_audit_catalog  # noqa: E402
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.data.model_catalog.adapter_catalog import RSUTypedCacheProfile  # noqa: E402
from src.envs.core.cache_eviction import (  # noqa: E402
    TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode  # noqa: E402
from src.envs.wrappers.gym_vec_env import GymVecEnv  # noqa: E402


MIB = 1_048_576
METHOD_ORDER = (
    "current_simple_threshold",
    "existing_two_step_rule",
    "offline_enumeration_reference",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["execution_constraints"]["rl_training_forbidden"] is not True:
        raise ValueError("frozen protocol must forbid RL training")
    if len(config["design_points"]) > int(config["maximum_design_points"]):
        raise ValueError("design-point budget exceeded")
    if len(config["design_points"]) != int(config["actual_design_points"]):
        raise ValueError("actual_design_points does not match design_points")
    return config


def _placement_object_ids(catalog: Any, adapter_ids: list[str]) -> list[str]:
    residents: list[str] = []
    for adapter_id in adapter_ids:
        placement = catalog.resolve_typed_placement_plan(
            adapter_id=adapter_id,
            resident_object_ids=residents,
        )
        for object_id in placement.ordered_object_ids:
            if object_id not in residents:
                residents.append(object_id)
    return residents


def _build_workflow(instance: dict[str, Any], catalog: Any) -> WorkflowGraphState:
    adapters = list(instance["workflow_adapters"])
    nodes: list[WorkflowNode] = []
    for index, adapter_id in enumerate(adapters):
        typed = catalog.get_typed_adapter(adapter_id)
        node_id = f"n{index}"
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                node_name=f"frozen coupling witness {adapter_id}",
                required_base_model=str(typed.required_base_model_id),
                required_adapter=adapter_id,
                input_size=1,
                output_size=1,
                predecessors=[f"n{index - 1}"] if index else [],
                successors=[f"n{index + 1}"] if index + 1 < len(adapters) else [],
            )
        )
    return WorkflowGraphState(
        workflow_id=f"shared_cache_recovery::{instance['instance_id']}",
        nodes=nodes,
        edges=[("n0", "n1")],
        execution_order=["n0", "n1"],
        current_node_id="n0",
    )


def build_env(
    instance: dict[str, Any],
    config: dict[str, Any],
    package_root: Path,
) -> GymVecEnv:
    native = config["native_contract"]
    catalog = build_audit_catalog(sharing_enabled=True)
    source_residents = _placement_object_ids(catalog, list(instance["source_initial_adapters"]))
    target_residents = _placement_object_ids(catalog, list(instance["target_initial_adapters"]))
    catalog.rsu_typed_cache_profiles = [
        RSUTypedCacheProfile(rsu_id="rsu_a", resident_object_ids=source_residents),
        RSUTypedCacheProfile(rsu_id="rsu_b", resident_object_ids=target_residents),
    ]
    workflow = _build_workflow(instance, catalog)
    first_adapter, next_adapter = list(instance["workflow_adapters"])
    first_base = str(catalog.get_typed_adapter(first_adapter).required_base_model_id)
    next_base = str(catalog.get_typed_adapter(next_adapter).required_base_model_id)
    frames = [
        {
            "time_index": index,
            "vehicles": [
                {
                    "vehicle_id": "veh_coupling_witness",
                    "position_x": float(position),
                    "position_y": 0.0,
                    "speed": 40.0,
                    "base_model_id": first_base,
                    "active_workflow_id": workflow.workflow_id,
                }
            ],
        }
        for index, position in enumerate(native["vehicle_positions"])
    ]
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=workflow,
        adapter_catalog=catalog,
        rsu_states=[
            RSUState("rsu_a", 0.0, 0.0, float(native["coverage_radius"])),
            RSUState("rsu_b", 100.0, 0.0, float(native["coverage_radius"])),
        ],
        max_steps=int(native["maximum_steps"]),
        cache_capacity_profile={
            "model_cache_profile_id": native["cache_profile"],
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(instance["capacity_mib"]),
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            "telemetry_enabled": True,
        },
        workflow_state_migration={
            "enabled": True,
            "package_root": str(package_root),
            "identity": {"base": next_base, "adapter": next_adapter, "next_node": "n1"},
            "input_identity": {
                "kind": "frozen_controlled_witness",
                "sha256": hashlib.sha256(instance["instance_id"].encode("utf-8")).hexdigest(),
            },
            "node_outputs": {
                "n0": {
                    "kind": "abstract_typed_node_output",
                    "node_id": "n0",
                    "value": f"prefix::{instance['instance_id']}",
                }
            },
        },
    )
    return GymVecEnv(core)


def _residents(snapshot: dict[str, Any], rsu_id: str) -> list[dict[str, Any]]:
    for row in snapshot.get("rsus", []):
        if row.get("rsu_id") == rsu_id:
            return deepcopy(row.get("residents", []))
    return []


def _resident_ids(snapshot: dict[str, Any], rsu_id: str) -> list[str]:
    return [str(row["object_id"]) for row in _residents(snapshot, rsu_id)]


def _transfer_bytes_from_event(event: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in (event.get("transfer_mb_by_type") or {}).items():
        if key == "workflow_state":
            continue
        result[str(key)] = int(round(float(value) * MIB))
    return result


def _objective(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["completed_node_count"]),
        int(summary["deadline_violation_count"]),
        float(summary["modeled_completion_seconds"]),
        int(summary["total_transfer_bytes"]),
        float(summary["modeled_recompute_seconds"]),
        int(summary["service_failure_count"]),
        int(summary["first_action_id"]),
    )


def run_branch(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    first_action: int,
) -> dict[str, Any]:
    branch_id = f"first_action_{first_action}"
    package_root = output_root / "runtime_state_packages" / instance["instance_id"] / branch_id
    env = build_env(instance, config, package_root)
    _, reset = env.reset(seed=int(config["seed"]))
    state = deepcopy(reset["semantic_state"])
    mask = [bool(value) for value in reset["action_mask"]]
    initial_snapshot = deepcopy(reset["cache_trace_snapshot"])
    steps: list[dict[str, Any]] = []
    for step_index, action in enumerate((first_action, 0, 0), start=1):
        if action >= len(mask) or not mask[action]:
            return {
                "feasible": False,
                "instance_id": instance["instance_id"],
                "branch_id": branch_id,
                "invalid_step": step_index,
                "action_id": action,
                "pre_action_mask": mask,
            }
        pre_snapshot = env.core_env.export_cache_trace_snapshot()
        pre_workflow = deepcopy(state["workflow"])
        pre_mask = list(mask)
        _, reward, terminated, truncated, info = env.step(action)
        state = deepcopy(info["semantic_state"])
        mask = [bool(value) for value in info["action_mask"]]
        event = deepcopy(info["cache_event"])
        completed_now = sorted(
            set(state["workflow"]["completed_node_ids"])
            - set(pre_workflow["completed_node_ids"])
        )
        steps.append(
            {
                "step_index": step_index,
                "time_index": int(state["time_index"]),
                "action_id": int(action),
                "action_name": info["action_name"],
                "action_allowed": True,
                "action_invalid": bool(info["action_invalid"]),
                "pre_action_mask": pre_mask,
                "eligible_actions": [index for index, allowed in enumerate(pre_mask) if allowed],
                "observed_pre_action": deepcopy(info["decision_observation_trace_record"]),
                "control_action": deepcopy(info["control_action"]),
                "pre_cache_snapshot": deepcopy(pre_snapshot),
                "post_cache_snapshot": deepcopy(info["cache_trace_snapshot"]),
                "completed_node_ids_this_step": completed_now,
                "completed_node_ids_after_step": list(state["workflow"]["completed_node_ids"]),
                "current_node_id_after_step": state["workflow"].get("current_node_id"),
                "cache_event": event,
                "transfer_bytes_by_type": _transfer_bytes_from_event(event),
                "production_action4_state_transfer": deepcopy(info["production_action4_state_transfer"]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    transfer_by_type: dict[str, int] = {}
    successful_state_package_bytes = 0
    positive_transfer_events = 0
    service_failures = 0
    successful_requests = 0
    for step in steps:
        step_model_bytes = 0
        for object_type, count in step["transfer_bytes_by_type"].items():
            transfer_by_type[object_type] = transfer_by_type.get(object_type, 0) + int(count)
            step_model_bytes += int(count)
        state_transfer = step["production_action4_state_transfer"]
        if state_transfer.get("migration_success"):
            successful_state_package_bytes = max(
                successful_state_package_bytes,
                int(state_transfer.get("package_bytes", 0)),
            )
        if step_model_bytes > 0:
            positive_transfer_events += 1
        event = step["cache_event"]
        if event.get("event_type") == "request":
            if event.get("service_success"):
                successful_requests += 1
            else:
                service_failures += 1
    if successful_state_package_bytes > 0:
        transfer_by_type["workflow_state"] = successful_state_package_bytes
        positive_transfer_events += 1
    total_transfer_bytes = sum(transfer_by_type.values())
    cost = config["cost_model"]
    transfer_seconds = total_transfer_bytes * 8.0 / (float(cost["effective_link_mbps"]) * 1_000_000.0)
    transfer_seconds += positive_transfer_events * float(cost["positive_transfer_fixed_latency_seconds"])
    state_overhead_seconds = 0.0
    if successful_state_package_bytes > 0:
        state_overhead_seconds = float(cost["state_serialize_seconds_reference"]) + float(
            cost["state_restore_seconds_reference"]
        )
    recompute_seconds = float(cost["prefix_recompute_seconds"]) if service_failures else 0.0
    modeled_completion_seconds = (
        transfer_seconds
        + state_overhead_seconds
        + recompute_seconds
        + successful_requests * float(cost["successful_request_service_seconds"])
    )
    completed_nodes = list(state["workflow"]["completed_node_ids"])
    deadline_violation = int(
        len(completed_nodes) < len(state["workflow"]["execution_order"])
        or modeled_completion_seconds > float(cost["deadline_seconds"])
    )
    summary = {
        "instance_id": instance["instance_id"],
        "branch_id": branch_id,
        "first_action_id": int(first_action),
        "actions_executed": [int(step["action_id"]) for step in steps],
        "completed_node_ids": completed_nodes,
        "completed_node_count": len(completed_nodes),
        "simulator_step_count": len(steps),
        "service_failure_count": service_failures,
        "successful_request_count": successful_requests,
        "total_transfer_bytes": total_transfer_bytes,
        "transfer_bytes_by_type": transfer_by_type,
        "modeled_transfer_seconds": transfer_seconds,
        "modeled_state_overhead_seconds": state_overhead_seconds,
        "modeled_recompute_seconds": recompute_seconds,
        "modeled_completion_seconds": modeled_completion_seconds,
        "deadline_seconds": float(cost["deadline_seconds"]),
        "deadline_violation_count": deadline_violation,
        "wall_clock_network_measurement": False,
        "native_completion": len(completed_nodes) == len(state["workflow"]["execution_order"]),
    }
    summary["objective_key"] = list(_objective(summary))
    return {
        "feasible": True,
        "instance_id": instance["instance_id"],
        "branch_id": branch_id,
        "initial_state": {
            "workflow": deepcopy(reset["semantic_state"]["workflow"]),
            "current_workflow_node": deepcopy(reset["semantic_state"]["current_workflow_node"]),
            "cache_snapshot": initial_snapshot,
            "target_resident_object_ids": _resident_ids(initial_snapshot, "rsu_b"),
            "action_mask": [bool(value) for value in reset["action_mask"]],
            "eligible_actions": [
                index for index, allowed in enumerate(reset["action_mask"]) if allowed
            ],
        },
        "steps": steps,
        "final_state": {
            "workflow": deepcopy(state["workflow"]),
            "cache_snapshot": env.core_env.export_cache_trace_snapshot(),
            "target_resident_object_ids": _resident_ids(
                env.core_env.export_cache_trace_snapshot(), "rsu_b"
            ),
        },
        "summary": summary,
    }


def _threshold_decision(
    instance: dict[str, Any],
    config: dict[str, Any],
    no_prepare_branch: dict[str, Any],
) -> dict[str, Any]:
    catalog = build_audit_catalog(sharing_enabled=True)
    current_adapter = str(instance["workflow_adapters"][0])
    target_residents = list(no_prepare_branch["initial_state"]["target_resident_object_ids"])
    placement = catalog.resolve_typed_placement_plan(
        adapter_id=current_adapter,
        resident_object_ids=target_residents,
    )
    missing_rows = [
        catalog.get_typed_object(object_id)
        for object_id in placement.ordered_object_ids
        if object_id not in target_residents
    ]
    model_prepare_bytes = int(
        round(sum(float(row.transfer_size_mb) for row in missing_rows) * MIB)
    )
    cost = config["cost_model"]
    state_bytes = int(cost["state_package_bytes_reference"])
    total_bytes = model_prepare_bytes + state_bytes
    transfer_seconds = total_bytes * 8.0 / (float(cost["effective_link_mbps"]) * 1_000_000.0)
    if total_bytes:
        transfer_seconds += float(cost["positive_transfer_fixed_latency_seconds"])
    estimate = (
        transfer_seconds
        + float(cost["state_serialize_seconds_reference"])
        + float(cost["state_restore_seconds_reference"])
    )
    action4_legal = bool(no_prepare_branch["initial_state"]["action_mask"][4])
    selected = 4 if action4_legal and estimate < float(cost["prefix_recompute_seconds"]) else 0
    return {
        "selected_first_action": selected,
        "observed_current_target_resident_object_ids": target_residents,
        "observed_current_required_adapter": current_adapter,
        "observed_action4_legal": action4_legal,
        "ex_ante_missing_current_model_objects": [row.object_id for row in missing_rows],
        "ex_ante_current_model_prepare_bytes": model_prepare_bytes,
        "ex_ante_state_package_bytes": state_bytes,
        "ex_ante_prepare_plus_restore_seconds": estimate,
        "ex_ante_prefix_recompute_seconds": float(cost["prefix_recompute_seconds"]),
        "future_demand_prediction_used": False,
        "future_realization_used": False,
    }


def _two_step_decision(instance: dict[str, Any], no_prepare_branch: dict[str, Any]) -> dict[str, Any]:
    current_adapter, declared_next_adapter = list(instance["workflow_adapters"])
    mask = no_prepare_branch["initial_state"]["action_mask"]
    selected = 1 if current_adapter == declared_next_adapter and bool(mask[1]) else 0
    return {
        "selected_first_action": selected,
        "observed_current_adapter": current_adapter,
        "observed_declared_next_adapter": declared_next_adapter,
        "declared_workflow_graph_used": True,
        "future_arrival_realization_used": False,
        "future_radio_realization_used": False,
        "future_compute_realization_used": False,
    }


def _method_result(
    method: str,
    branch: dict[str, Any],
    decision: dict[str, Any],
    *,
    oracle: bool,
) -> dict[str, Any]:
    return {
        "method": method,
        "online": not oracle,
        "oracle": oracle,
        "decision_record": decision,
        "branch_id": branch["branch_id"],
        "summary": deepcopy(branch["summary"]),
        "legal_actions": [
            {
                "step_index": step["step_index"],
                "selected_action": step["action_id"],
                "eligible_actions": step["eligible_actions"],
                "allowed": step["action_allowed"],
            }
            for step in branch["steps"]
        ],
        "state_transition_trace": deepcopy(branch),
    }


def _instance_results(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    branches = {
        0: run_branch(instance, config, output_root, 0),
        4: run_branch(instance, config, output_root, 4),
    }
    if not all(branch["feasible"] for branch in branches.values()):
        raise AssertionError(f"frozen branch infeasible for {instance['instance_id']}: {branches}")
    threshold = _threshold_decision(instance, config, branches[0])
    two_step = _two_step_decision(instance, branches[0])
    if two_step["selected_first_action"] not in branches:
        raise AssertionError("frozen two-step rule selected an action outside the enumerated branches")
    oracle_action = min(branches, key=lambda action: _objective(branches[action]["summary"]))
    oracle_decision = {
        "selected_first_action": int(oracle_action),
        "candidate_first_actions": [0, 4],
        "candidate_objective_keys": {
            str(action): list(_objective(branches[action]["summary"])) for action in (0, 4)
        },
        "realized_future_used": True,
        "online_method": False,
        "complexity_claim_allowed": False,
    }
    methods = {
        "current_simple_threshold": _method_result(
            "current_simple_threshold",
            branches[int(threshold["selected_first_action"])],
            threshold,
            oracle=False,
        ),
        "existing_two_step_rule": _method_result(
            "existing_two_step_rule",
            branches[int(two_step["selected_first_action"])],
            two_step,
            oracle=False,
        ),
        "offline_enumeration_reference": _method_result(
            "offline_enumeration_reference",
            branches[int(oracle_action)],
            oracle_decision,
            oracle=True,
        ),
    }
    after_first = {
        str(action): _resident_ids(branches[action]["steps"][0]["post_cache_snapshot"], "rsu_b")
        for action in (0, 4)
    }
    later_step_transfer = {
        str(action): sum(branches[action]["steps"][1]["transfer_bytes_by_type"].values())
        for action in (0, 4)
    }
    coupling = {
        "target_residents_after_first_action": after_first,
        "later_request_model_transfer_bytes": later_step_transfer,
        "first_action_changes_target_residents": after_first["0"] != after_first["4"],
        "first_action_changes_later_model_transfer": later_step_transfer["0"] != later_step_transfer["4"],
        "why_current_decision_affects_later_cost": (
            "the legal action-4 transaction commits a different finite-capacity target resident set, "
            "which the next native request consumes under the same LRU/dependency contract"
            if after_first["0"] != after_first["4"] and later_step_transfer["0"] != later_step_transfer["4"]
            else "no later model-transfer difference was realized at this frozen point"
        ),
        "restore_cost_only_explanation": False,
    }
    return {
        "instance": deepcopy(instance),
        "branches": {str(key): value for key, value in branches.items()},
        "methods": methods,
        "coupling_test": coupling,
    }


def _csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for method in METHOD_ORDER:
            item = result["methods"][method]
            summary = item["summary"]
            rows.append(
                {
                    "instance_id": result["instance"]["instance_id"],
                    "classification": result["instance"]["classification"],
                    "method": method,
                    "online": item["online"],
                    "oracle": item["oracle"],
                    "first_action_id": summary["first_action_id"],
                    "actions_executed": " ".join(str(value) for value in summary["actions_executed"]),
                    "completed_node_count": summary["completed_node_count"],
                    "simulator_step_count": summary["simulator_step_count"],
                    "service_failure_count": summary["service_failure_count"],
                    "total_transfer_bytes": summary["total_transfer_bytes"],
                    "modeled_recompute_seconds": summary["modeled_recompute_seconds"],
                    "modeled_completion_seconds": summary["modeled_completion_seconds"],
                    "deadline_violation_count": summary["deadline_violation_count"],
                    "changes_later_model_transfer": result["coupling_test"][
                        "first_action_changes_later_model_transfer"
                    ],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_event_witness(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = next(item for item in results if item["instance"]["instance_id"].startswith("C_"))
    return {
        "instance_id": candidate["instance"]["instance_id"],
        "question": "why does the current decision affect a later cost?",
        "answer": candidate["coupling_test"]["why_current_decision_affects_later_cost"],
        "recompute_rule": "sum native transfer bytes from cache_event.transfer_mb_by_type, use exact package_bytes for successful workflow-state import, then apply the frozen cost_model",
        "branches": {
            action: {
                "initial_target_residents": branch["initial_state"]["target_resident_object_ids"],
                "first_action": branch["summary"]["first_action_id"],
                "step_events": [
                    {
                        "step_index": step["step_index"],
                        "action_id": step["action_id"],
                        "pre_target_residents": _resident_ids(step["pre_cache_snapshot"], "rsu_b"),
                        "admitted_object_ids": step["cache_event"].get("admitted_object_ids", []),
                        "evicted_object_ids": step["cache_event"].get("evicted_object_ids", []),
                        "post_target_residents": _resident_ids(step["post_cache_snapshot"], "rsu_b"),
                        "service_success": step["cache_event"].get("service_success"),
                        "migration_realized": step["cache_event"].get("migration_realized"),
                        "completed_node_ids_this_step": step["completed_node_ids_this_step"],
                        "transfer_bytes_by_type": step["transfer_bytes_by_type"],
                        "state_transfer_status": step["production_action4_state_transfer"].get("status"),
                        "state_package_bytes": step["production_action4_state_transfer"].get(
                            "package_bytes", 0
                        ),
                    }
                    for step in branch["steps"]
                ],
                "summary": branch["summary"],
            }
            for action, branch in candidate["branches"].items()
        },
        "claim_boundary": {
            "native_facts": "legal actions, cache commits, residents, evictions, bytes, service success, node completion",
            "modeled_facts": "100 Mbps transfer time, same-host state overhead, technical-workload recompute time, deadline",
            "not_claimed": "real wireless gain, shared bandwidth/compute contention, cross-workflow persistence, online oracle availability",
        },
    }


def _integrity_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "integrity_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = _load_config(config_path)
    actual_commit = _git("rev-parse", "HEAD")
    if actual_commit != args.expected_git_commit:
        raise SystemExit(f"git commit mismatch: {actual_commit} != {args.expected_git_commit}")
    if _git("status", "--porcelain"):
        raise SystemExit("refusing to run from a dirty worktree")
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    try:
        frozen_protocol = {
            "config": deepcopy(config),
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256_file(config_path),
            "execution_git_commit": actual_commit,
            "git_branch": _git("branch", "--show-current"),
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
        }
        _write_json(output_root / "frozen_protocol.json", frozen_protocol)

        results = [
            _instance_results(instance, config, output_root)
            for instance in config["design_points"]
        ]
        payload = {
            "run_id": config["run_id"],
            "execution_git_commit": actual_commit,
            "result_count": len(results),
            "results": results,
        }
        _write_json(output_root / "all_method_results.json", payload)
        _write_csv(output_root / "all_method_results.csv", _csv_rows(results))
        _write_json(output_root / "event_witness.json", _build_event_witness(results))

        completion = {
            "run_id": config["run_id"],
            "status": "COMPLETED",
            "single_execution": True,
            "rl_started": False,
            "design_point_count": len(results),
            "native_branch_execution_count": len(results) * 2,
            "all_actions_legal": all(
                legal["allowed"]
                for result in results
                for method in result["methods"].values()
                for legal in method["legal_actions"]
            ),
            "all_branches_completed_equal_work": all(
                all(branch["summary"]["completed_node_count"] == 2 for branch in result["branches"].values())
                for result in results
            ),
            "coupling_instances": [
                result["instance"]["instance_id"]
                for result in results
                if result["coupling_test"]["first_action_changes_later_model_transfer"]
            ],
            "method_first_actions": {
                result["instance"]["instance_id"]: {
                    method: item["summary"]["first_action_id"]
                    for method, item in result["methods"].items()
                }
                for result in results
            },
        }
        _write_json(output_root / "completion_receipt.json", completion)
        _write_json(
            output_root / "integrity_manifest.json",
            {
                "run_id": config["run_id"],
                "hash_algorithm": "sha256",
                "files": _integrity_rows(output_root),
            },
        )
    except Exception:
        shutil.rmtree(output_root)
        raise


if __name__ == "__main__":
    main()
