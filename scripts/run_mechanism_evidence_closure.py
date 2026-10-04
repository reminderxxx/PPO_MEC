"""Build the bounded A/C mechanism evidence package from a frozen plan."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_native_typed_cache_probe import build_audit_catalog  # noqa: E402
from scripts.audit_remaining_workflow_decision_value import ledger_for_config, read_jsonl  # noqa: E402
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.data.model_catalog.adapter_catalog import RSUTypedCacheProfile  # noqa: E402
from src.envs.core.cache_eviction import TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU  # noqa: E402
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode  # noqa: E402
from src.envs.wrappers.gym_vec_env import GymVecEnv  # noqa: E402


MIB = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _group_by_config(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_id"]), []).append(row)
    return grouped


def build_experiment_a(plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan["experiment_a"]
    old_path = ROOT / spec["input_old_log"]
    candidate_path = ROOT / spec["input_candidate_log"]
    old = _group_by_config(read_jsonl(old_path))
    candidate = _group_by_config(read_jsonl(candidate_path))
    expected = set(spec["configurations"])
    if set(old) != expected or set(candidate) != expected:
        raise AssertionError("four-configuration identity drift")
    ledgers = []
    for label, grouped in (("old", old), ("candidate", candidate)):
        for config_id in spec["configurations"]:
            rows = grouped[config_id]
            if len(rows) != int(spec["request_count_per_configuration"]):
                raise AssertionError(f"request count drift for {label}/{config_id}")
            ledgers.append(ledger_for_config(rows, label))

    timing_rows = []
    model = spec["cost_model"]
    for ledger in ledgers:
        for link_mbps in model["sensitivity_link_mbps"]:
            load_wait = float(ledger["actual_transfer_bytes_total"]) * 8.0 / (float(link_mbps) * 1_000_000.0)
            service = float(ledger["service_success_count"]) * float(model["successful_service_seconds"])
            rejected = float(ledger["service_failure_count"]) * float(model["rejected_request_seconds"])
            timing_rows.append({
                "semantics_label": ledger["semantics_label"],
                "config_id": ledger["config_id"],
                "link_mbps_assumption": float(link_mbps),
                "model_load_wait_seconds_simulated": load_wait,
                "successful_service_seconds_simulated": service,
                "rejected_request_seconds_simulated": rejected,
                "serial_completion_seconds_simulated": load_wait + service + rejected,
                "measured_wall_clock": False,
            })

    candidates = [row for row in ledgers if row["semantics_label"] == "candidate"]
    if {row["completed_request_count"] for row in candidates} != {72}:
        raise AssertionError("candidate arms do not have equal completion")
    by_id = {row["config_id"]: row for row in candidates}
    primary_timing = {
        row["config_id"]: row
        for row in timing_rows
        if row["semantics_label"] == "candidate"
        and row["link_mbps_assumption"] == float(model["primary_link_mbps"])
    }
    return {
        "classification": "native_simulation_ledger_plus_separate_analytical_time_model",
        "input_identity": {
            "old": {"path": spec["input_old_log"], "sha256": sha256_file(old_path), "rows": sum(map(len, old.values()))},
            "candidate": {"path": spec["input_candidate_log"], "sha256": sha256_file(candidate_path), "rows": sum(map(len, candidate.values()))},
        },
        "ledgers": ledgers,
        "time_model": {**model, "hardware_or_network_measurement": False},
        "timing_sensitivity": timing_rows,
        "equal_completion_comparison": {
            "baseline": "interleaved__sharing_on",
            "best_native_transfer": "blocked__sharing_on",
            "completed_requests_each": 72,
            "base_transfer_bytes_saved": (
                by_id["interleaved__sharing_on"]["actual_load_bytes_by_type"]["base_model"]
                - by_id["blocked__sharing_on"]["actual_load_bytes_by_type"]["base_model"]
            ),
            "total_transfer_bytes_saved": (
                by_id["interleaved__sharing_on"]["actual_transfer_bytes_total"]
                - by_id["blocked__sharing_on"]["actual_transfer_bytes_total"]
            ),
            "primary_simulated_completion_seconds_saved": (
                primary_timing["interleaved__sharing_on"]["serial_completion_seconds_simulated"]
                - primary_timing["blocked__sharing_on"]["serial_completion_seconds_simulated"]
            ),
        },
        "old_semantics_guard": "lower bytes are inseparable from 36 or 60 rejected services and are not an equal-completion advantage",
    }


def build_workflow(scenario: dict[str, Any], plan: dict[str, Any], catalog: Any) -> WorkflowGraphState:
    adapters = plan["experiment_c"]["workflow_design_points"][scenario["tail"]]
    nodes = []
    edges = []
    for index, adapter_id in enumerate(adapters):
        node_id = f"n{index}"
        previous = f"n{index - 1}" if index else None
        following = f"n{index + 1}" if index + 1 < len(adapters) else None
        typed = catalog.get_typed_adapter(adapter_id)
        nodes.append(WorkflowNode(
            node_id=node_id,
            node_name=f"mechanism closure {adapter_id}",
            required_base_model=str(typed.required_base_model_id),
            required_adapter=adapter_id,
            input_size=1,
            output_size=1,
            predecessors=[previous] if previous else [],
            successors=[following] if following else [],
        ))
        if previous:
            edges.append((previous, node_id))
    return WorkflowGraphState(
        workflow_id=f"mechanism_closure::{scenario['scenario_id']}",
        nodes=nodes,
        edges=edges,
        execution_order=[node.node_id for node in nodes],
        current_node_id="n0",
    )


def build_env(scenario: dict[str, Any], plan: dict[str, Any]) -> GymVecEnv:
    catalog = build_audit_catalog(sharing_enabled=bool(scenario["sharing"]))
    catalog.rsu_typed_cache_profiles = [
        RSUTypedCacheProfile(rsu_id="rsu_a", resident_object_ids=[]),
        RSUTypedCacheProfile(rsu_id="rsu_b", resident_object_ids=[]),
    ]
    workflow = build_workflow(scenario, plan, catalog)
    first_adapter = plan["experiment_c"]["workflow_design_points"][scenario["tail"]][0]
    first_base = str(catalog.get_typed_adapter(first_adapter).required_base_model_id)
    frames = [
        {
            "time_index": index,
            "vehicles": [{
                "vehicle_id": "veh_1",
                "position_x": position,
                "position_y": 0.0,
                "speed": 40.0,
                "base_model_id": first_base,
                "active_workflow_id": workflow.workflow_id,
            }],
        }
        for index, position in enumerate((0.0, 40.0, 80.0, 100.0, 100.0, 100.0))
    ]
    capacity = plan["experiment_c"]["capacity_design_points_mib"][scenario["capacity"]]
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=workflow,
        adapter_catalog=catalog,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 55.0), RSUState("rsu_b", 100.0, 0.0, 55.0)],
        max_steps=int(plan["experiment_c"]["maximum_steps"]),
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(capacity),
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            "telemetry_enabled": True,
        },
    )
    return GymVecEnv(core)


def _transfer_bytes(event: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(round(float(value) * MIB))
        for key, value in (event.get("transfer_mb_by_type") or {}).items()
    }


def run_sequence(scenario: dict[str, Any], plan: dict[str, Any], actions: tuple[int, ...]) -> dict[str, Any]:
    env = build_env(scenario, plan)
    _, reset = env.reset(seed=int(plan["experiment_c"]["seed"]))
    state = deepcopy(reset["semantic_state"])
    mask = list(reset["action_mask"])
    steps = []
    for step_index in range(int(plan["experiment_c"]["maximum_steps"])):
        action = int(actions[step_index])
        if action >= len(mask) or not mask[action]:
            return {"feasible": False, "invalid_at_step": step_index + 1, "actions": list(actions)}
        before = deepcopy(state["workflow"])
        _, reward, terminated, truncated, info = env.step(action)
        state = deepcopy(info["semantic_state"])
        mask = list(info["action_mask"])
        event = deepcopy(info["cache_event"])
        completed_now = sorted(set(state["workflow"]["completed_node_ids"]) - set(before["completed_node_ids"]))
        steps.append({
            "step_index": step_index + 1,
            "action_id": action,
            "action_name": info["action_name"],
            "action_allowed": True,
            "action_invalid": bool(info["action_invalid"]),
            "control_action": info["control_action"],
            "completed_node_ids": completed_now,
            "reward": float(reward),
            "cache_event": event,
            "transfer_bytes_by_type": _transfer_bytes(event),
            "migration_prepare_requested": bool(info["metrics_protocol"].get("migration_prepare_requested")),
            "migration_prepare_realized": bool(info["metrics_protocol"].get("migration_prepare_realized")),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    workflow = state["workflow"]
    transfer: dict[str, int] = {}
    for step in steps:
        for key, value in step["transfer_bytes_by_type"].items():
            transfer[key] = transfer.get(key, 0) + int(value)
    state_cost = int(plan["experiment_c"]["state_cost_design_points"][scenario["state_cost"]]["bytes"])
    prepare_count = sum(step["migration_prepare_requested"] for step in steps)
    native_transfer = sum(transfer.values())
    effective = native_transfer + prepare_count * state_cost
    completed = len(workflow["completed_node_ids"])
    total = len(workflow["execution_order"])
    primary_link = float(plan["experiment_a"]["cost_model"]["primary_link_mbps"])
    service_seconds = float(plan["experiment_a"]["cost_model"]["successful_service_seconds"])
    success_count = sum(bool(step["cache_event"].get("service_success")) for step in steps)
    return {
        "feasible": True,
        "actions_planned": list(actions),
        "actions_executed": [step["action_id"] for step in steps],
        "steps": steps,
        "summary": {
            "node_completion_count": completed,
            "node_total": total,
            "workflow_completed": bool(workflow["is_completed"]),
            "deadline_violation_count": 0 if workflow["is_completed"] else 1,
            "unfinished_node_count": total - completed,
            "service_success_count": success_count,
            "service_failure_count": sum(not bool(step["cache_event"].get("service_success")) for step in steps),
            "native_transfer_bytes_by_type": dict(sorted(transfer.items())),
            "native_transfer_bytes_total": native_transfer,
            "state_cost_bytes_per_prepare_design_point": state_cost,
            "prepare_request_count": prepare_count,
            "effective_transfer_bytes_total": effective,
            "step_count": len(steps),
            "simulated_completion_seconds": effective * 8.0 / (primary_link * 1_000_000.0) + success_count * service_seconds,
            "simulated_time_guard": "analytical assumption, not measured wall clock or RSU/network latency",
        },
    }


def objective_key(episode: dict[str, Any]) -> tuple[Any, ...]:
    summary = episode["summary"]
    return (
        -int(summary["node_completion_count"]),
        -int(summary["workflow_completed"]),
        int(summary["deadline_violation_count"]),
        int(summary["service_failure_count"]),
        int(summary["effective_transfer_bytes_total"]),
        int(summary["step_count"]),
        tuple(episode["actions_executed"]),
    )


def run_policy(
    scenario: dict[str, Any],
    plan: dict[str, Any],
    selector: Callable[[dict[str, Any], list[bool]], int],
) -> dict[str, Any]:
    env = build_env(scenario, plan)
    _, reset = env.reset(seed=int(plan["experiment_c"]["seed"]))
    state = deepcopy(reset["semantic_state"])
    mask = list(reset["action_mask"])
    actions = []
    for _ in range(int(plan["experiment_c"]["maximum_steps"])):
        action = int(selector(state, mask))
        if action >= len(mask) or not mask[action]:
            action = next(index for index, allowed in enumerate(mask) if allowed)
        actions.append(action)
        _, _, terminated, truncated, info = env.step(action)
        state = deepcopy(info["semantic_state"])
        mask = list(info["action_mask"])
        if terminated or truncated:
            break
    padded = tuple(actions + [0] * (int(plan["experiment_c"]["maximum_steps"]) - len(actions)))
    return run_sequence(scenario, plan, padded)


def current_selector(state: dict[str, Any], mask: list[bool]) -> int:
    return 0


def two_step_selector(state: dict[str, Any], mask: list[bool]) -> int:
    workflow = state["workflow"]
    node_by_id = {node["node_id"]: node for node in workflow["nodes"]}
    current_id = workflow.get("current_node_id")
    order = list(workflow["execution_order"])
    if current_id not in order:
        return 0
    index = order.index(current_id)
    current_adapter = node_by_id[current_id]["required_adapter"]
    if index + 1 < len(order):
        next_adapter = node_by_id[order[index + 1]]["required_adapter"]
        if next_adapter == current_adapter and len(mask) > 1 and mask[1]:
            return 1
    return 0


def enumerate_full_suffix(scenario: dict[str, Any], plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    action_ids = tuple(int(item) for item in plan["experiment_c"]["action_ids"])
    horizon = int(plan["experiment_c"]["maximum_steps"])
    feasible = []
    for actions in product(action_ids, repeat=horizon):
        episode = run_sequence(scenario, plan, actions)
        if episode["feasible"]:
            feasible.append(episode)
    if not feasible:
        raise AssertionError(f"no feasible action sequence for {scenario['scenario_id']}")
    best = min(feasible, key=objective_key)
    return best, {
        "enumerated_sequence_count": len(action_ids) ** horizon,
        "feasible_sequence_count": len(feasible),
        "objective_key": list(objective_key(best)[:-1]) + [best["actions_executed"]],
    }


def enumerate_current_only(scenario: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    horizon = int(plan["experiment_c"]["maximum_steps"])
    for action in plan["experiment_c"]["action_ids"]:
        episode = run_sequence(scenario, plan, tuple([int(action)] + [0] * (horizon - 1)))
        if episode["feasible"]:
            first = episode["steps"][0]
            step_transfer = sum(first["transfer_bytes_by_type"].values())
            state_cost = episode["summary"]["state_cost_bytes_per_prepare_design_point"] if first["migration_prepare_requested"] else 0
            episode["current_only_objective"] = (
                -len(first["completed_node_ids"]),
                int(not bool(first["cache_event"].get("service_success"))),
                step_transfer + state_cost,
                int(action),
            )
            candidates.append(episode)
    return min(candidates, key=lambda item: item["current_only_objective"])


def _method_row(scenario: dict[str, Any], method: str, episode: dict[str, Any]) -> dict[str, Any]:
    summary = episode["summary"]
    return {
        "scenario_id": scenario["scenario_id"],
        "sharing": bool(scenario["sharing"]),
        "capacity": scenario["capacity"],
        "tail": scenario["tail"],
        "state_cost": scenario["state_cost"],
        "method": method,
        "actions": "-".join(map(str, episode["actions_executed"])),
        "first_action": episode["actions_executed"][0],
        "node_completion_count": summary["node_completion_count"],
        "node_total": summary["node_total"],
        "workflow_completed": summary["workflow_completed"],
        "deadline_violation_count": summary["deadline_violation_count"],
        "service_failure_count": summary["service_failure_count"],
        "native_transfer_bytes_total": summary["native_transfer_bytes_total"],
        "state_cost_bytes_total": summary["prepare_request_count"] * summary["state_cost_bytes_per_prepare_design_point"],
        "effective_transfer_bytes_total": summary["effective_transfer_bytes_total"],
        "simulated_completion_seconds": summary["simulated_completion_seconds"],
    }


def build_experiment_c(plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan["experiment_c"]
    if len(spec["scenarios"]) != int(spec["scenario_count"]) or len(spec["scenarios"]) > 12:
        raise AssertionError("scenario matrix is not the frozen <=12 design")
    rows = []
    details = []
    ablations = []
    for scenario in spec["scenarios"]:
        current = run_policy(scenario, plan, current_selector)
        two_step = run_policy(scenario, plan, two_step_selector)
        full, enumeration = enumerate_full_suffix(scenario, plan)
        current_only = enumerate_current_only(scenario, plan)
        episodes = {
            "current_request_rule": current,
            "existing_two_step_lookahead": two_step,
            "bounded_full_suffix_enumeration": full,
        }
        rows.extend(_method_row(scenario, method, episode) for method, episode in episodes.items())
        details.append({"scenario": scenario, "methods": episodes, "full_enumeration": enumeration})
        ablations.append({
            "scenario_id": scenario["scenario_id"],
            "current_only_first_action": current_only["actions_executed"][0],
            "full_suffix_first_action": full["actions_executed"][0],
            "decision_changed": current_only["actions_executed"][0] != full["actions_executed"][0],
            "current_only_summary": current_only["summary"],
            "full_suffix_summary": full["summary"],
        })
    by = {(row["scenario_id"], row["method"]): row for row in rows}
    boundaries = []
    for scenario in spec["scenarios"]:
        sid = scenario["scenario_id"]
        current = by[(sid, "current_request_rule")]
        two = by[(sid, "existing_two_step_lookahead")]
        full = by[(sid, "bounded_full_suffix_enumeration")]
        boundaries.append({
            "scenario_id": sid,
            "two_step_matches_full_first_action": two["first_action"] == full["first_action"],
            "two_step_matches_full_primary_metrics": all(
                two[key] == full[key]
                for key in ("node_completion_count", "deadline_violation_count", "service_failure_count", "effective_transfer_bytes_total")
            ),
            "full_completion_delta_vs_current": full["node_completion_count"] - current["node_completion_count"],
            "full_effective_transfer_delta_vs_current": full["effective_transfer_bytes_total"] - current["effective_transfer_bytes_total"],
            "full_is_no_better_than_two_step": (
                full["node_completion_count"] <= two["node_completion_count"]
                and full["effective_transfer_bytes_total"] >= two["effective_transfer_bytes_total"]
            ),
        })
    return {
        "classification": spec["classification"],
        "scenario_count": len(spec["scenarios"]),
        "method_count": 3,
        "rows": rows,
        "scenario_details": details,
        "information_value_ablation": ablations,
        "profit_loss_boundaries": boundaries,
        "algorithm_promotion_guard": {
            "two_step_matches_full_all_primary_metrics": all(row["two_step_matches_full_primary_metrics"] for row in boundaries),
            "rl_authorized": False,
            "reason": "this bounded diagnostic cannot authorize large-scale training; simple-policy sufficiency is reported directly",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    actual_commit = current_commit()
    if actual_commit != args.expected_commit:
        raise RuntimeError(f"execution commit mismatch: expected {args.expected_commit}, got {actual_commit}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")
    started = time.monotonic()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    experiment_a = build_experiment_a(plan)
    experiment_c = build_experiment_c(plan)
    write_json(output / "experiment_a_shared_cache_cost.json", experiment_a)
    write_json(output / "experiment_c_decision_space.json", experiment_c)
    write_csv(output / "experiment_c_all_scenarios.csv", experiment_c["rows"])
    elapsed = time.monotonic() - started
    receipt = {
        "status": "PASS" if elapsed <= float(plan["wall_limits_seconds"]["a_and_c"]) else "FAIL",
        "artifact_run_id": output.name,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "git_commit": actual_commit,
        "wall_seconds": elapsed,
        "wall_limit_seconds": plan["wall_limits_seconds"]["a_and_c"],
        "real_model_generate_calls": 0,
        "training_runs": 0,
        "formal_runs": 0,
        "holdout_runs": 0,
    }
    write_json(output / "completion_receipt.json", receipt)
    files = inventory(output)
    write_json(output / "artifact_integrity_manifest.json", {
        "status": "PASS",
        "file_count": len(files),
        "files": files,
    })
    if receipt["status"] != "PASS":
        raise RuntimeError("A/C wall limit exceeded")
    print(json.dumps({"status": "PASS", "wall_seconds": elapsed, "scenario_rows": len(experiment_c["rows"])}))


if __name__ == "__main__":
    main()
