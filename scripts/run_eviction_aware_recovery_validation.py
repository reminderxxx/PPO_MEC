"""Run the frozen bounded validation of eviction-aware workflow recovery."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shared_cache_recovery_coupling import (  # noqa: E402
    MIB,
    _objective,
    _resident_ids,
    build_env,
    run_branch,
)
from src.envs.core.cache_eviction import CACHE_CAPACITY_EPSILON  # noqa: E402
from src.runtime.eviction_aware_recovery import (  # noqa: E402
    RecoveryDecisionInputs,
    decision_inputs_to_dict,
    eviction_aware_recovery_decision,
    original_simple_threshold_decision,
    two_step_lookahead_decision,
)


METHOD_ORDER = (
    "original_simple_threshold",
    "eviction_aware_recovery",
    "two_step_lookahead",
    "offline_reference",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    if config["methods"]["eviction_aware_recovery"]["explicit_opt_in"] is not True:
        raise ValueError("eviction-aware recovery must remain explicit opt-in")
    points = config["design_points"]
    if len(points) > int(config["maximum_design_points"]):
        raise ValueError("design-point budget exceeded")
    if len(points) != int(config["actual_design_points"]):
        raise ValueError("actual_design_points does not match design_points")
    return config


def _network_seconds(byte_count: int, cost: dict[str, Any]) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (float(cost["effective_link_mbps"]) * 1_000_000.0) + float(
        cost["positive_transfer_fixed_latency_seconds"]
    )


def _instance_cost(config: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config["cost_model"])
    result.update(deepcopy(instance.get("realized_cost_overrides", {})))
    return result


def _branch_config(config: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["cost_model"] = _instance_cost(config, instance)
    return result


def _resummarize_branch(
    branch: dict[str, Any],
    *,
    cost: dict[str, Any],
) -> dict[str, Any]:
    summary = deepcopy(branch["summary"])
    transfer_by_type: dict[str, int] = {}
    positive_events = 0
    for step in branch["steps"]:
        step_bytes = 0
        for object_type, value in step["transfer_bytes_by_type"].items():
            transfer_by_type[object_type] = transfer_by_type.get(object_type, 0) + int(value)
            step_bytes += int(value)
        if step_bytes > 0:
            positive_events += 1
    state_bytes = max(
        (
            int(step["production_action4_state_transfer"].get("package_bytes", 0))
            for step in branch["steps"]
            if step["production_action4_state_transfer"].get("migration_success")
        ),
        default=0,
    )
    if state_bytes > 0:
        transfer_by_type["workflow_state"] = state_bytes
        positive_events += 1
    if int(summary["first_action_id"]) == 0 and int(summary["service_failure_count"]) > 0:
        transfer_by_type["rerun_input"] = int(cost["rerun_input_bytes"])
        positive_events += 1
    total_bytes = sum(transfer_by_type.values())
    transfer_seconds = total_bytes * 8.0 / (float(cost["effective_link_mbps"]) * 1_000_000.0)
    transfer_seconds += positive_events * float(cost["positive_transfer_fixed_latency_seconds"])
    state_overhead = 0.0
    if state_bytes > 0:
        state_overhead = float(cost["state_serialize_seconds_reference"]) + float(
            cost["state_restore_seconds_reference"]
        )
    recompute = int(summary["service_failure_count"]) * float(cost["prefix_recompute_seconds"])
    completion = (
        transfer_seconds
        + state_overhead
        + recompute
        + int(summary["successful_request_count"])
        * float(cost["successful_request_service_seconds"])
    )
    violation = int(
        not summary["native_completion"] or completion > float(cost["deadline_seconds"])
    )
    summary.update(
        {
            "transfer_bytes_by_type": transfer_by_type,
            "total_transfer_bytes": total_bytes,
            "modeled_transfer_seconds": transfer_seconds,
            "modeled_state_overhead_seconds": state_overhead,
            "modeled_recompute_seconds": recompute,
            "modeled_completion_seconds": completion,
            "deadline_seconds": float(cost["deadline_seconds"]),
            "deadline_violation_count": violation,
        }
    )
    summary["objective_key"] = list(_objective(summary))
    return summary


def _preview_inputs(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> tuple[RecoveryDecisionInputs, dict[str, Any]]:
    preview_env = build_env(instance, config, output_root / "preview_state_packages" / instance["instance_id"])
    _, reset = preview_env.reset(seed=int(config["seed"]))
    core = preview_env.core_env
    catalog = core.adapter_catalog
    target_rsu_id = str(config["native_contract"]["handoff_target_rsu"])
    residents = _resident_ids(reset["cache_trace_snapshot"], target_rsu_id)
    current_adapter = str(instance["workflow_adapters"][0])
    placement = catalog.resolve_typed_placement_plan(
        adapter_id=current_adapter,
        resident_object_ids=residents,
    )
    capacity = float(instance["capacity_mib"])
    used = sum(float(catalog.get_typed_object(object_id).resident_size_mb) for object_id in residents)
    required_free = max(used + float(placement.requested_bundle_mb) - capacity, 0.0)
    victim_ids: list[str] = []
    victim_plan: dict[str, Any] | None = None
    planning_trace: dict[str, Any] | None = None
    feasible = True
    if required_free > CACHE_CAPACITY_EPSILON:
        rsu = next(item for item in core.rsu_states if item.rsu_id == target_rsu_id)
        policy_before = core.export_cache_eviction_policy_state()
        plan, planning_trace = core._plan_sequential_typed_lru_evictions(
            rsu=rsu,
            residents=list(residents),
            required_free_capacity=required_free,
            protected_object_ids=set(placement.ordered_object_ids),
        )
        if core.export_cache_eviction_policy_state() != policy_before:
            raise RuntimeError("native victim preview mutated policy state")
        victim_plan = plan.to_dict()
        feasible = bool(plan.sufficient)
        victim_ids = list(plan.ordered_victim_ids)
    post_recovery = [item for item in residents if item not in set(victim_ids)]
    for object_id in placement.missing_object_ids:
        if object_id not in post_recovery:
            post_recovery.append(object_id)

    near_term_nodes = list(instance["workflow_adapters"])[1 : 1 + int(config["near_term_horizon_nodes"])]
    near_dependencies: list[str] = []
    for adapter_id in near_term_nodes:
        plan = catalog.resolve_typed_placement_plan(
            adapter_id=str(adapter_id),
            resident_object_ids=[],
        )
        for object_id in plan.ordered_object_ids:
            if object_id not in near_dependencies:
                near_dependencies.append(object_id)
    missing_before = [item for item in near_dependencies if item not in residents]
    missing_after = [item for item in near_dependencies if item not in post_recovery]
    transfer_bytes = {
        item.object_id: int(round(float(item.transfer_size_mb) * MIB))
        for item in catalog.typed_cache_objects
        if item.counts_toward_capacity
    }
    cost = _instance_cost(config, instance)
    estimate = deepcopy(instance.get("decision_estimate", {}))
    link = estimate.get("effective_link_mbps", cost["effective_link_mbps"])
    recompute = float(cost["prefix_recompute_seconds"]) * float(
        estimate.get("prefix_recompute_scale", 1.0)
    )
    restore = (
        float(cost["state_serialize_seconds_reference"])
        + float(cost["state_restore_seconds_reference"])
    ) * float(estimate.get("state_restore_overhead_scale", 1.0))
    inputs = RecoveryDecisionInputs(
        action4_legal=bool(reset["action_mask"][4]),
        recovery_preview_feasible=feasible,
        current_missing_object_ids=tuple(placement.missing_object_ids),
        legal_victim_ids=tuple(victim_ids),
        near_term_dependency_ids=tuple(near_dependencies),
        near_term_missing_before_ids=tuple(missing_before),
        near_term_missing_after_ids=tuple(missing_after),
        transfer_bytes_by_object=transfer_bytes,
        state_package_bytes=int(cost["state_package_bytes_reference"]),
        rerun_input_bytes=int(cost["rerun_input_bytes"]),
        effective_link_mbps=None if link is None else float(link),
        positive_transfer_fixed_latency_seconds=float(
            cost["positive_transfer_fixed_latency_seconds"]
        ),
        state_restore_overhead_seconds=restore,
        prefix_recompute_seconds=recompute,
        future_reload_cost_scale=float(estimate.get("future_reload_cost_scale", 1.0)),
    )
    preview = {
        "target_rsu_id": target_rsu_id,
        "initial_resident_object_ids": residents,
        "capacity_mib": capacity,
        "used_mib": used,
        "current_placement": placement.to_dict(),
        "required_free_mib": required_free,
        "native_legal_victim_plan": victim_plan,
        "native_planning_trace": planning_trace,
        "post_recovery_resident_object_ids": post_recovery,
        "near_term_declared_adapters": near_term_nodes,
        "near_term_dependency_ids": near_dependencies,
        "near_term_missing_before_ids": missing_before,
        "near_term_missing_after_ids": missing_after,
        "read_only_preview": True,
        "future_realization_used": False,
    }
    return inputs, preview


def _timed_decision(
    function: Callable[[], dict[str, Any]],
    *,
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = function()
    durations: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        candidate = function()
        durations.append(time.perf_counter_ns() - started)
        if candidate["selected_first_action"] != result["selected_first_action"]:
            raise RuntimeError("non-deterministic online decision")
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return result, {
        "repetitions": repetitions,
        "mean_nanoseconds": statistics.fmean(durations),
        "median_nanoseconds": statistics.median(durations),
        "p95_nanoseconds": p95,
        "minimum_nanoseconds": min(durations),
        "maximum_nanoseconds": max(durations),
        "scope": "pure deterministic decision function; excludes native transition execution",
    }


def _method_result(
    method: str,
    decision: dict[str, Any],
    overhead: dict[str, Any],
    branch: dict[str, Any],
    *,
    oracle: bool,
) -> dict[str, Any]:
    return {
        "method": method,
        "online": not oracle,
        "oracle": oracle,
        "decision_record": decision,
        "decision_overhead": overhead,
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
    }


def _run_instance(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    local_config = _branch_config(config, instance)
    branches: dict[int, dict[str, Any]] = {}
    branch_wall_seconds: dict[str, float] = {}
    for action in (0, 4):
        started = time.perf_counter()
        branch = run_branch(instance, local_config, output_root, action)
        branch_wall_seconds[str(action)] = time.perf_counter() - started
        if not branch["feasible"]:
            raise AssertionError(f"infeasible frozen branch: {instance['instance_id']} action={action}")
        branch["summary"] = _resummarize_branch(branch, cost=local_config["cost_model"])
        branches[action] = branch

    inputs, preview = _preview_inputs(instance, local_config, output_root)
    repetitions = int(config["decision_overhead_repetitions"])
    original, original_overhead = _timed_decision(
        lambda: original_simple_threshold_decision(inputs), repetitions=repetitions
    )
    proposed, proposed_overhead = _timed_decision(
        lambda: eviction_aware_recovery_decision(inputs, opt_in_enabled=True),
        repetitions=repetitions,
    )
    lookahead, lookahead_overhead = _timed_decision(
        lambda: two_step_lookahead_decision(inputs), repetitions=repetitions
    )
    for decision in (original, proposed, lookahead):
        if int(decision["selected_first_action"]) not in branches:
            raise AssertionError("online method selected outside frozen feasible set")

    oracle_started = time.perf_counter_ns()
    oracle_action = min(branches, key=lambda action: _objective(branches[action]["summary"]))
    oracle_selection_ns = time.perf_counter_ns() - oracle_started
    oracle = {
        "method": "offline_reference",
        "selected_first_action": int(oracle_action),
        "decision": "recover" if oracle_action == 4 else "rerun",
        "candidate_first_actions": [0, 4],
        "candidate_objective_keys": {
            str(action): list(_objective(branches[action]["summary"])) for action in (0, 4)
        },
        "realized_future_used": True,
        "information_advantage": (
            "selects after both complete native branches and realized modeled outcomes are available"
        ),
    }
    methods = {
        "original_simple_threshold": _method_result(
            "original_simple_threshold",
            original,
            original_overhead,
            branches[int(original["selected_first_action"])],
            oracle=False,
        ),
        "eviction_aware_recovery": _method_result(
            "eviction_aware_recovery",
            proposed,
            proposed_overhead,
            branches[int(proposed["selected_first_action"])],
            oracle=False,
        ),
        "two_step_lookahead": _method_result(
            "two_step_lookahead",
            lookahead,
            lookahead_overhead,
            branches[int(lookahead["selected_first_action"])],
            oracle=False,
        ),
        "offline_reference": _method_result(
            "offline_reference",
            oracle,
            {
                "selection_nanoseconds": oracle_selection_ns,
                "branch_execution_seconds": branch_wall_seconds,
                "scope": "post-hoc selection after both realized branches; not online comparable",
            },
            branches[oracle_action],
            oracle=True,
        ),
    }
    return {
        "instance": deepcopy(instance),
        "realized_cost_model": deepcopy(local_config["cost_model"]),
        "decision_inputs": decision_inputs_to_dict(inputs),
        "native_transaction_preview": preview,
        "branches": {str(key): value for key, value in branches.items()},
        "methods": methods,
        "online_equivalence": {
            "eviction_aware_equals_two_step_action": (
                proposed["selected_first_action"] == lookahead["selected_first_action"]
            ),
            "eviction_aware_equals_two_step_costs": (
                proposed.get("restart_incremental_seconds")
                == lookahead.get("restart_incremental_seconds")
                and proposed.get("recovery_incremental_seconds")
                == lookahead.get("recovery_incremental_seconds")
            ),
        },
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
                    "completed_node_count": summary["completed_node_count"],
                    "service_failure_count": summary["service_failure_count"],
                    "total_transfer_bytes": summary["total_transfer_bytes"],
                    "base_model_transfer_bytes": summary["transfer_bytes_by_type"].get(
                        "base_model", 0
                    ),
                    "adapter_transfer_bytes": summary["transfer_bytes_by_type"].get(
                        "adapter", 0
                    ),
                    "workflow_state_transfer_bytes": summary["transfer_bytes_by_type"].get(
                        "workflow_state", 0
                    ),
                    "rerun_input_transfer_bytes": summary["transfer_bytes_by_type"].get(
                        "rerun_input", 0
                    ),
                    "modeled_recompute_seconds": summary["modeled_recompute_seconds"],
                    "modeled_completion_seconds": summary["modeled_completion_seconds"],
                    "deadline_violation_count": summary["deadline_violation_count"],
                    "decision_overhead_median_nanoseconds": item["decision_overhead"].get(
                        "median_nanoseconds"
                    ),
                    "conservative_fallback": item["decision_record"].get(
                        "conservative_fallback", False
                    ),
                    "induced_reload_object_ids": " ".join(
                        item["decision_record"].get("induced_reload_object_ids", [])
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        rows = [result["methods"][method] for result in results]
        transfer_types: dict[str, int] = {}
        for row in rows:
            for key, value in row["summary"]["transfer_bytes_by_type"].items():
                transfer_types[key] = transfer_types.get(key, 0) + int(value)
        online_overheads = [
            float(row["decision_overhead"]["median_nanoseconds"])
            for row in rows
            if "median_nanoseconds" in row["decision_overhead"]
        ]
        methods[method] = {
            "instance_count": len(rows),
            "completed_node_count": sum(
                int(row["summary"]["completed_node_count"]) for row in rows
            ),
            "service_failure_count": sum(
                int(row["summary"]["service_failure_count"]) for row in rows
            ),
            "deadline_violation_count": sum(
                int(row["summary"]["deadline_violation_count"]) for row in rows
            ),
            "total_transfer_bytes": sum(
                int(row["summary"]["total_transfer_bytes"]) for row in rows
            ),
            "transfer_bytes_by_type": transfer_types,
            "modeled_recompute_seconds": sum(
                float(row["summary"]["modeled_recompute_seconds"]) for row in rows
            ),
            "modeled_completion_seconds": sum(
                float(row["summary"]["modeled_completion_seconds"]) for row in rows
            ),
            "recover_decision_count": sum(
                int(row["summary"]["first_action_id"] == 4) for row in rows
            ),
            "decision_overhead_median_of_instance_medians_nanoseconds": (
                statistics.median(online_overheads) if online_overheads else None
            ),
            "oracle_information_advantage": method == "offline_reference",
        }
    return {
        "methods": methods,
        "all_methods_equal_completion": len(
            {row["completed_node_count"] for row in methods.values()}
        )
        == 1,
        "eviction_aware_two_step_action_agreement_count": sum(
            int(result["online_equivalence"]["eviction_aware_equals_two_step_action"])
            for result in results
        ),
        "eviction_aware_two_step_cost_agreement_count": sum(
            int(result["online_equivalence"]["eviction_aware_equals_two_step_costs"])
            for result in results
        ),
    }


def _event_explanations(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": "all frozen instances; native events preserved without filtering negative results",
        "instances": [
            {
                "instance_id": result["instance"]["instance_id"],
                "classification": result["instance"]["classification"],
                "frozen_reason": result["instance"]["frozen_reason"],
                "initial_residents": result["native_transaction_preview"][
                    "initial_resident_object_ids"
                ],
                "current_missing_objects": result["decision_inputs"][
                    "current_missing_object_ids"
                ],
                "legal_victim_ids": result["decision_inputs"]["legal_victim_ids"],
                "near_term_missing_before": result["decision_inputs"][
                    "near_term_missing_before_ids"
                ],
                "near_term_missing_after": result["decision_inputs"][
                    "near_term_missing_after_ids"
                ],
                "method_decisions": {
                    method: result["methods"][method]["decision_record"]
                    for method in METHOD_ORDER
                },
                "realized_branches": {
                    action: {
                        "summary": branch["summary"],
                        "events": [
                            {
                                "step_index": step["step_index"],
                                "action_id": step["action_id"],
                                "pre_target_residents": _resident_ids(
                                    step["pre_cache_snapshot"], "rsu_b"
                                ),
                                "evicted_object_ids": step["cache_event"].get(
                                    "evicted_object_ids", []
                                ),
                                "admitted_object_ids": step["cache_event"].get(
                                    "admitted_object_ids", []
                                ),
                                "post_target_residents": _resident_ids(
                                    step["post_cache_snapshot"], "rsu_b"
                                ),
                                "service_success": step["cache_event"].get("service_success"),
                                "completed_node_ids": step["completed_node_ids_after_step"],
                                "transfer_bytes_by_type": step["transfer_bytes_by_type"],
                                "state_package_bytes": step[
                                    "production_action4_state_transfer"
                                ].get("package_bytes", 0),
                            }
                            for step in branch["steps"]
                        ],
                    }
                    for action, branch in result["branches"].items()
                },
            }
            for result in results
        ],
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
    started = time.perf_counter()
    try:
        _write_json(
            output_root / "frozen_protocol.json",
            {
                "config": config,
                "config_path": str(config_path.relative_to(ROOT)),
                "config_sha256": _sha256_file(config_path),
                "execution_git_commit": actual_commit,
                "git_branch": _git("branch", "--show-current"),
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
            },
        )
        results = [
            _run_instance(instance, config, output_root)
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
        aggregate = _aggregate(results)
        _write_json(output_root / "aggregate_summary.json", aggregate)
        _write_json(output_root / "event_explanations.json", _event_explanations(results))
        completion = {
            "run_id": config["run_id"],
            "status": "COMPLETED",
            "single_execution": True,
            "rl_started": False,
            "model_calls": 0,
            "downloads": 0,
            "old_holdout_operations": 0,
            "design_point_count": len(results),
            "native_branch_execution_count": len(results) * 2,
            "all_actions_legal": all(
                legal["allowed"]
                for result in results
                for method in result["methods"].values()
                for legal in method["legal_actions"]
            ),
            "all_methods_equal_completion": aggregate["all_methods_equal_completion"],
            "eviction_aware_two_step_action_agreement_count": aggregate[
                "eviction_aware_two_step_action_agreement_count"
            ],
            "elapsed_wall_seconds": time.perf_counter() - started,
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
