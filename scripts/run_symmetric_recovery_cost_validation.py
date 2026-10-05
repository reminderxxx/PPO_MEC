"""Execute a frozen symmetric restart/recovery cost matrix without model calls."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shared_cache_recovery_coupling import MIB, _resident_ids, build_env  # noqa: E402
from src.envs.specs import ControlAction  # noqa: E402
from src.runtime.symmetric_recovery_cost import (  # noqa: E402
    LifecycleEventEstimate,
    PathLifecycleEstimate,
    SymmetricRecoveryDecisionInputs,
    decision_inputs_to_dict,
    eviction_aware_recovery_decision,
    original_simple_threshold_decision,
    score_estimated_path,
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    inherited = raw.get("inherits_frozen_design")
    if inherited:
        source = (ROOT / inherited["path"]).resolve()
        if _sha256_file(source) != inherited["sha256"]:
            raise ValueError("inherited frozen design hash mismatch")
        config = json.loads(source.read_text(encoding="utf-8"))
        config.update({key: value for key, value in raw.items() if key != "inherits_frozen_design"})
        config["inherits_frozen_design"] = inherited
    else:
        config = raw
    constraints = config["execution_constraints"]
    for field in (
        "rl_training_forbidden",
        "model_calls_forbidden",
        "downloads_forbidden",
        "old_holdout_operations_forbidden",
        "single_execution",
        "retuning_after_results_forbidden",
        "negative_result_deletion_forbidden",
    ):
        if constraints[field] is not True:
            raise ValueError(f"frozen constraint must be true: {field}")
    points = config["design_points"]
    if len(points) != int(config["actual_design_points"]):
        raise ValueError("actual_design_points mismatch")
    if len(points) > int(config["maximum_design_points"]):
        raise ValueError("design-point budget exceeded")
    if int(config["near_term_horizon_nodes"]) != 1:
        raise ValueError("this bounded runner requires one declared future node")
    return config


def _cache_control(target_rsu_id: str) -> ControlAction:
    return ControlAction(
        cache_action={
            "operation": "cache",
            "rsu_id": target_rsu_id,
            "strategy": "symmetric_lifecycle_validation",
        },
        offload_action={"mode": "rsu", "target_rsu_id": target_rsu_id},
        migration_action={"mode": "keep"},
    )


def _execute_cache_path(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    *,
    action_id: int,
    purpose: str,
) -> dict[str, Any]:
    branch_root = output_root / "isolated_paths" / purpose / instance["instance_id"] / str(action_id)
    env = build_env(instance, config, branch_root)
    _, reset = env.reset(seed=int(config["seed"]))
    core = env.core_env
    target_rsu = str(config["native_contract"]["handoff_target_rsu"])
    events: list[dict[str, Any]] = []
    for index, adapter_id in enumerate(instance["workflow_adapters"][:2]):
        core._episode_steps = index
        pre = core.export_cache_trace_snapshot()
        result = core._apply_typed_cache_action(
            control=_cache_control(target_rsu),
            primary_vehicle=None,
            current_node_id=f"{purpose}_{action_id}_n{index}",
            required_adapter=str(adapter_id),
        )
        post = core.export_cache_trace_snapshot()
        status = str(result.get("atomic_transaction_status"))
        admitted = tuple(str(row["object_id"]) for row in result.get("admitted_typed_objects", []))
        transfer = {
            object_id: int(
                round(float(core.adapter_catalog.get_typed_object(object_id).transfer_size_mb) * MIB)
            )
            for object_id in admitted
        }
        event = LifecycleEventEstimate(
            event_id=f"{purpose}:{action_id}:n{index}",
            required_adapter_id=str(adapter_id),
            pre_resident_object_ids=tuple(_resident_ids(pre, target_rsu)),
            victim_object_ids=tuple(str(value) for value in result.get("evicted_object_ids", [])),
            admitted_object_ids=admitted,
            post_resident_object_ids=tuple(_resident_ids(post, target_rsu)),
            transfer_bytes_by_object=transfer,
            transaction_status=status,
            dependency_safe=int(result.get("orphan_count", 0)) == 0,
        )
        events.append({"estimate": event, "native_result": deepcopy(result)})
        if status not in {"committed", "noop_all_resident"}:
            break
    feasible = len(events) == 2 and all(
        row["estimate"].transaction_status in {"committed", "noop_all_resident"}
        and row["estimate"].dependency_safe
        for row in events
    )
    return {
        "action_id": action_id,
        "purpose": purpose,
        "feasible": feasible,
        "initial_resident_object_ids": _resident_ids(reset["cache_trace_snapshot"], target_rsu),
        "events": events,
        "terminal_resident_object_ids": _resident_ids(core.export_cache_trace_snapshot(), target_rsu),
        "environment_identity": f"{purpose}:{action_id}:{id(core)}",
    }


def _path_estimate(
    branch: dict[str, Any],
    *,
    action_id: int,
    config: dict[str, Any],
    instance: dict[str, Any],
    estimated: bool,
) -> PathLifecycleEstimate:
    cost = deepcopy(config["cost_model"])
    cost.update(deepcopy(instance.get("realized_cost_overrides", {})))
    estimate = deepcopy(instance.get("decision_estimate", {})) if estimated else {}
    recompute = (
        float(cost["prefix_recompute_seconds"])
        * float(estimate.get("prefix_recompute_scale", 1.0))
        if action_id == 0
        else 0.0
    )
    restore = (
        (
            float(cost["state_serialize_seconds_reference"])
            + float(cost["state_restore_seconds_reference"])
        )
        * float(estimate.get("state_restore_overhead_scale", 1.0))
        if action_id == 4
        else 0.0
    )
    dynamic = int(cost["rerun_input_bytes"] if action_id == 0 else cost["state_package_bytes_reference"])
    return PathLifecycleEstimate(
        action_id=action_id,
        events=tuple(row["estimate"] for row in branch["events"]),
        dynamic_transfer_bytes=dynamic,
        state_overhead_seconds=restore,
        recompute_seconds=recompute,
        successful_service_seconds=2.0 * float(cost["successful_request_service_seconds"]),
    )


def _decision_inputs(
    previews: dict[int, dict[str, Any]],
    *,
    config: dict[str, Any],
    instance: dict[str, Any],
) -> SymmetricRecoveryDecisionInputs:
    cost = deepcopy(config["cost_model"])
    cost.update(deepcopy(instance.get("realized_cost_overrides", {})))
    estimate = deepcopy(instance.get("decision_estimate", {}))
    return SymmetricRecoveryDecisionInputs(
        action4_legal=True,
        restart_path=_path_estimate(
            previews[0], action_id=0, config=config, instance=instance, estimated=True
        ),
        recovery_path=_path_estimate(
            previews[4], action_id=4, config=config, instance=instance, estimated=True
        ),
        effective_link_mbps=float(
            estimate.get("effective_link_mbps", cost["effective_link_mbps"])
        ),
        positive_transfer_fixed_latency_seconds=float(
            cost["positive_transfer_fixed_latency_seconds"]
        ),
        future_reload_cost_scale=float(estimate.get("future_reload_cost_scale", 1.0)),
    )


def _timed(
    function: Callable[[], dict[str, Any]], repetitions: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = function()
    durations: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        candidate = function()
        durations.append(time.perf_counter_ns() - started)
        if candidate["selected_first_action"] != result["selected_first_action"]:
            raise RuntimeError("non-deterministic decision")
    return result, {
        "repetitions": repetitions,
        "median_nanoseconds": statistics.median(durations),
        "minimum_nanoseconds": min(durations),
        "maximum_nanoseconds": max(durations),
        "scope": "pure decision over detached previews; excludes native transitions",
    }


def _score_realized_branch(
    branch: dict[str, Any],
    *,
    action_id: int,
    config: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    cost = deepcopy(config["cost_model"])
    cost.update(deepcopy(instance.get("realized_cost_overrides", {})))
    path = _path_estimate(
        branch, action_id=action_id, config=config, instance=instance, estimated=False
    )
    score = score_estimated_path(
        path,
        effective_link_mbps=float(cost["effective_link_mbps"]),
        fixed_latency_seconds=float(cost["positive_transfer_fixed_latency_seconds"]),
        future_reload_cost_scale=1.0,
    )
    completed = 2 if branch["feasible"] else 0
    failures = 0 if branch["feasible"] else 1
    score.update(
        {
            "completed_node_count": completed,
            "service_failure_count": failures,
            "deadline_seconds": float(cost["deadline_seconds"]),
            "deadline_violation_count": int(
                completed != 2
                or float(score["estimated_total_seconds"]) > float(cost["deadline_seconds"])
            ),
            "total_transfer_bytes": int(score["model_transfer_bytes"])
            + int(score["dynamic_transfer_bytes"]),
            "modeled_completion_seconds": score.pop("estimated_total_seconds"),
            "branch_feasible": branch["feasible"],
            "wall_clock_network_measurement": False,
        }
    )
    return score


def _objective(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["completed_node_count"]),
        int(summary["deadline_violation_count"]),
        float(summary["modeled_completion_seconds"]),
        int(summary["total_transfer_bytes"]),
        float(summary["recompute_seconds"]),
        int(summary["service_failure_count"]),
        int(summary["action_id"]),
    )


def _serialize_branch(branch: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in branch.items()
        if key not in {"events", "environment_identity"}
    } | {
        "events": [
            {
                "lifecycle": deepcopy(row["estimate"].__dict__),
                "native_result": row["native_result"],
            }
            for row in branch["events"]
        ],
        "summary": summary,
    }


def _run_instance(
    instance: dict[str, Any], config: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    previews = {
        action: _execute_cache_path(
            instance, config, output_root, action_id=action, purpose="decision_preview"
        )
        for action in (0, 4)
    }
    branches = {
        action: _execute_cache_path(
            instance, config, output_root, action_id=action, purpose="realized_score"
        )
        for action in (0, 4)
    }
    environment_ids = [
        previews[action]["environment_identity"] for action in (0, 4)
    ] + [branches[action]["environment_identity"] for action in (0, 4)]
    if len(set(environment_ids)) != 4:
        raise AssertionError("branch environments are not isolated")
    if not all(path["feasible"] for path in (*previews.values(), *branches.values())):
        raise AssertionError(f"frozen path infeasible: {instance['instance_id']}")

    inputs = _decision_inputs(previews, config=config, instance=instance)
    repetitions = int(config["decision_overhead_repetitions"])
    original, original_timing = _timed(
        lambda: original_simple_threshold_decision(inputs), repetitions
    )
    proposed, proposed_timing = _timed(
        lambda: eviction_aware_recovery_decision(inputs), repetitions
    )
    lookahead, lookahead_timing = _timed(
        lambda: two_step_lookahead_decision(inputs), repetitions
    )

    summaries = {
        action: _score_realized_branch(
            branches[action], action_id=action, config=config, instance=instance
        )
        for action in (0, 4)
    }
    offline_action = min(summaries, key=lambda action: _objective(summaries[action]))
    offline = {
        "method": "offline_reference",
        "selected_first_action": offline_action,
        "decision": "recover" if offline_action == 4 else "rerun",
        "candidate_objective_keys": {
            str(action): list(_objective(summaries[action])) for action in (0, 4)
        },
        "future_truth_permission": True,
        "realized_score_used": True,
        "information_advantage": "selects after isolated legal branches are fully scored",
    }
    decisions = {
        "original_simple_threshold": (original, original_timing),
        "eviction_aware_recovery": (proposed, proposed_timing),
        "two_step_lookahead": (lookahead, lookahead_timing),
        "offline_reference": (offline, {"scope": "post-hoc micro reference"}),
    }
    methods = {}
    for method, (decision, timing) in decisions.items():
        action = int(decision["selected_first_action"])
        methods[method] = {
            "decision_record": decision,
            "decision_overhead": timing,
            "summary": deepcopy(summaries[action]),
            "relative_to_offline_reference_seconds": float(
                summaries[action]["modeled_completion_seconds"]
            )
            - float(summaries[offline_action]["modeled_completion_seconds"]),
        }
    return {
        "instance": deepcopy(instance),
        "decision_inputs": decision_inputs_to_dict(inputs),
        "preview_score_separation": {
            "separate_environment_count": 4,
            "online_realized_score_used": False,
            "offline_realized_score_used": True,
        },
        "branches": {
            str(action): _serialize_branch(branches[action], summaries[action])
            for action in (0, 4)
        },
        "methods": methods,
        "agreements": {
            "simple_equals_proposed": original["selected_first_action"]
            == proposed["selected_first_action"],
            "proposed_equals_two_step": proposed["selected_first_action"]
            == lookahead["selected_first_action"],
            "proposed_equals_offline": proposed["selected_first_action"] == offline_action,
            "restart_recovery_lifecycle_events_equal": [
                (
                    row["estimate"].victim_object_ids,
                    row["estimate"].admitted_object_ids,
                    row["estimate"].post_resident_object_ids,
                )
                for row in branches[0]["events"]
            ]
            == [
                (
                    row["estimate"].victim_object_ids,
                    row["estimate"].admitted_object_ids,
                    row["estimate"].post_resident_object_ids,
                )
                for row in branches[4]["events"]
            ],
        },
    }


def _rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for method in METHOD_ORDER:
            item = result["methods"][method]
            summary = item["summary"]
            branch = result["branches"][str(summary["action_id"])]
            rows.append(
                {
                    "instance_id": result["instance"]["instance_id"],
                    "classification": result["instance"]["classification"],
                    "method": method,
                    "first_action_id": summary["action_id"],
                    "victim_plan": " | ".join(
                        "+".join(event["lifecycle"]["victim_object_ids"])
                        for event in branch["events"]
                    ),
                    "resident_trace": " | ".join(
                        "+".join(event["lifecycle"]["post_resident_object_ids"])
                        for event in branch["events"]
                    ),
                    "model_transfer_bytes": summary["model_transfer_bytes"],
                    "dynamic_transfer_bytes": summary["dynamic_transfer_bytes"],
                    "total_transfer_bytes": summary["total_transfer_bytes"],
                    "modeled_completion_seconds": summary["modeled_completion_seconds"],
                    "completed_node_count": summary["completed_node_count"],
                    "service_failure_count": summary["service_failure_count"],
                    "recompute_seconds": summary["recompute_seconds"],
                    "relative_to_offline_reference_seconds": item[
                        "relative_to_offline_reference_seconds"
                    ],
                }
            )
    return rows


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        rows = [result["methods"][method] for result in results]
        methods[method] = {
            "instance_count": len(rows),
            "recover_decision_count": sum(
                int(row["summary"]["action_id"] == 4) for row in rows
            ),
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
            "model_transfer_bytes": sum(
                int(row["summary"]["model_transfer_bytes"]) for row in rows
            ),
            "dynamic_transfer_bytes": sum(
                int(row["summary"]["dynamic_transfer_bytes"]) for row in rows
            ),
            "modeled_completion_seconds": sum(
                float(row["summary"]["modeled_completion_seconds"]) for row in rows
            ),
            "recompute_seconds": sum(
                float(row["summary"]["recompute_seconds"]) for row in rows
            ),
            "match_offline_count": sum(
                int(row["relative_to_offline_reference_seconds"] == 0.0) for row in rows
            ),
        }
    return {
        "methods": methods,
        "simple_proposed_action_agreement_count": sum(
            int(result["agreements"]["simple_equals_proposed"]) for result in results
        ),
        "proposed_two_step_action_agreement_count": sum(
            int(result["agreements"]["proposed_equals_two_step"]) for result in results
        ),
        "proposed_offline_action_agreement_count": sum(
            int(result["agreements"]["proposed_equals_offline"]) for result in results
        ),
        "equal_lifecycle_event_count": sum(
            int(result["agreements"]["restart_recovery_lifecycle_events_equal"])
            for result in results
        ),
    }


def _old_to_new(
    results: list[dict[str, Any]], historical_path: Path
) -> list[dict[str, Any]]:
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    old = {row["instance"]["instance_id"]: row for row in historical["results"]}
    rows: list[dict[str, Any]] = []
    for result in results:
        instance_id = result["instance"]["instance_id"]
        previous = old[instance_id]
        for method in METHOD_ORDER:
            old_item = previous["methods"][method]
            old_summary = old_item["summary"]
            old_branch = previous["branches"][str(old_summary["first_action_id"])]
            new_item = result["methods"][method]
            new_summary = new_item["summary"]
            new_branch = result["branches"][str(new_summary["action_id"])]
            rows.append(
                {
                    "instance_id": instance_id,
                    "method": method,
                    "old_action_id": old_summary["first_action_id"],
                    "new_action_id": new_summary["action_id"],
                    "old_victims": " | ".join(
                        "+".join(step["cache_event"].get("evicted_object_ids", []))
                        for step in old_branch["steps"]
                    ),
                    "new_victims": " | ".join(
                        "+".join(event["lifecycle"]["victim_object_ids"])
                        for event in new_branch["events"]
                    ),
                    "old_terminal_residents": "+".join(
                        old_branch["final_state"]["target_resident_object_ids"]
                    ),
                    "new_terminal_residents": "+".join(
                        new_branch["terminal_resident_object_ids"]
                    ),
                    "old_total_transfer_bytes": old_summary["total_transfer_bytes"],
                    "new_total_transfer_bytes": new_summary["total_transfer_bytes"],
                    "old_completion_seconds": old_summary["modeled_completion_seconds"],
                    "new_completion_seconds": new_summary["modeled_completion_seconds"],
                    "old_completed_nodes": old_summary["completed_node_count"],
                    "new_completed_nodes": new_summary["completed_node_count"],
                    "old_service_failures": old_summary["service_failure_count"],
                    "new_service_failures": new_summary["service_failure_count"],
                    "old_recompute_seconds": old_summary["modeled_recompute_seconds"],
                    "new_recompute_seconds": new_summary["recompute_seconds"],
                    "old_relative_to_reference_seconds": float(
                        old_summary["modeled_completion_seconds"]
                    )
                    - float(
                        previous["methods"]["offline_reference"]["summary"][
                            "modeled_completion_seconds"
                        ]
                    ),
                    "new_relative_to_reference_seconds": new_item[
                        "relative_to_offline_reference_seconds"
                    ],
                }
            )
    return rows


def _integrity_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "integrity_manifest.json"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--historical-results", type=Path)
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
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
            },
        )
        results = [
            _run_instance(instance, config, output_root)
            for instance in config["design_points"]
        ]
        _write_json(
            output_root / "all_method_results.json",
            {
                "run_id": config["run_id"],
                "execution_git_commit": actual_commit,
                "results": results,
            },
        )
        _write_csv(output_root / "all_method_results.csv", _rows(results))
        aggregate = _aggregate(results)
        _write_json(output_root / "aggregate_summary.json", aggregate)
        if args.historical_results:
            comparison = _old_to_new(results, args.historical_results.resolve())
            _write_json(output_root / "old_to_new_point_comparison.json", comparison)
            _write_csv(output_root / "old_to_new_point_comparison.csv", comparison)
        _write_json(
            output_root / "completion_receipt.json",
            {
                "run_id": config["run_id"],
                "status": "COMPLETED",
                "single_execution": True,
                "rl_started": False,
                "model_calls": 0,
                "downloads": 0,
                "old_holdout_operations": 0,
                "design_point_count": len(results),
                "isolated_native_path_count": len(results) * 4,
                "all_paths_feasible": all(
                    branch["summary"]["branch_feasible"]
                    for result in results
                    for branch in result["branches"].values()
                ),
                "elapsed_wall_seconds": time.perf_counter() - started,
            },
        )
        _write_json(
            output_root / "integrity_manifest.json",
            {
                "run_id": config["run_id"],
                "hash_algorithm": "sha256",
                "files": _integrity_rows(output_root),
            },
        )
    except Exception as exc:
        _write_json(
            output_root / "failure_receipt.json",
            {
                "run_id": config.get("run_id"),
                "status": "FAILED_PRESERVED_NO_RETRY",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "elapsed_wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
