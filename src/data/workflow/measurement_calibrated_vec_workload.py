"""Generator and bounded evaluator for workload v0.1."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import random
import time
from typing import Any


MIB = 1024 * 1024


def _value(config: dict[str, Any], section: str, key: str) -> float:
    return float(config["calibration"][section][key]["value"])


def generate_workloads(config: dict[str, Any]) -> list[dict[str, Any]]:
    assumptions = config["synthetic_assumptions"]
    base_bytes = int(_value(config, "model", "base_network_directory_bytes"))
    adapter_bytes = int(_value(config, "model", "adapter_network_directory_bytes"))
    rows: list[dict[str, Any]] = []
    for point in config["design"]["points"]:
        for seed in config["seeds"]:
            rng = random.Random(int(seed))
            arrivals = sorted(
                rng.uniform(*assumptions["arrival_jitter_seconds"]) + 3.0 * index
                for index in range(3)
            )
            base_ids = (
                ["abstract_base_shared"] * 3
                if point["sharing"]
                else ["abstract_base_a", "abstract_base_b", "abstract_base_a"]
            )
            adapters = list(assumptions["request_order"])
            workflows = []
            for index, (base_id, adapter_id) in enumerate(zip(base_ids, adapters)):
                workflow_id = f"{point['design_id']}__seed_{seed}__workflow_{index}"
                workflows.append(
                    {
                        "workflow_id": workflow_id,
                        "arrival_seconds": arrivals[index],
                        "request_order_index": index,
                        "sharing_relation": {
                            "base_model_id": base_id,
                            "shares_base_with": [
                                other
                                for other, candidate in enumerate(base_ids)
                                if other != index and candidate == base_id
                            ],
                            "source_type": "synthetic",
                        },
                        "nodes": [
                            {
                                "node_id": "n0",
                                "predecessors": [],
                                "successors": ["n1"],
                                "model_family": "abstract_typed_vision_family",
                                "base_model_id": base_id,
                                "adapter_id": adapter_id,
                                "base_file_bytes": int(_value(config, "model", "base_file_bytes")),
                                "base_network_bytes": base_bytes,
                                "adapter_file_bytes": int(_value(config, "model", "adapter_file_bytes")),
                                "adapter_network_bytes": adapter_bytes,
                                "compute_seconds": _value(config, "time_seconds", "prefix_compute"),
                            },
                            {
                                "node_id": "n1",
                                "predecessors": ["n0"],
                                "successors": [],
                                "model_family": "abstract_typed_vision_family",
                                "base_model_id": base_id,
                                "adapter_id": adapter_id,
                                "base_file_bytes": int(_value(config, "model", "base_file_bytes")),
                                "base_network_bytes": base_bytes,
                                "adapter_file_bytes": int(_value(config, "model", "adapter_file_bytes")),
                                "adapter_network_bytes": adapter_bytes,
                                "compute_seconds": _value(config, "time_seconds", "suffix_compute"),
                            },
                        ],
                        "edges": [["n0", "n1"]],
                        "input_bytes": int(_value(config, "dynamic", "input_bytes")),
                        "intermediate_state_bytes": int(
                            _value(config, "dynamic", "state_package_bytes")
                        ),
                        "intermediate_record_bytes": int(
                            _value(config, "dynamic", "intermediate_record_bytes")
                        ),
                        "deadline_seconds_from_arrival": float(
                            assumptions["deadline_seconds_from_arrival"]
                        ),
                        "mobility": {
                            "access_handoff": assumptions["access_handoff_event"],
                            "compute_migration": assumptions["compute_migration_event"],
                            "source_rsu_id": "rsu_source",
                            "target_rsu_id": "rsu_target",
                            "source_type": "synthetic",
                        },
                    }
                )
            rows.append(
                {
                    "schema_version": config["schema_version"],
                    "workload_version": config["workload_version"],
                    "design_id": point["design_id"],
                    "seed": int(seed),
                    "factors": deepcopy(point),
                    "rsu_resources": {
                        "target_model_cache_capacity_bytes": int(
                            assumptions[f"{point['capacity']}_capacity_bytes"]
                        ),
                        "initial_target_model": point["initial_target_model"],
                        "single_process_peak_runtime_memory_bytes": int(
                            _value(config, "model", "single_process_peak_runtime_memory_bytes")
                        ),
                    },
                    "link": {
                        "mbps": float(assumptions["link_mbps"]),
                        "fixed_one_way_seconds": float(assumptions["fixed_one_way_seconds"]),
                        "source_type": "synthetic",
                        "claim_boundary": "formula input; not measured wireless migration",
                    },
                    "workflows": workflows,
                    "provenance": {
                        "calibration_artifact_run_id": config["calibration"][
                            "source_artifact_run_id"
                        ],
                        "development_calibration_only": True,
                        "old_holdout_used": False,
                        "typed_model_objects_are_abstract": True,
                        "adapter_ids_are_not_claimed_as_distinct_real_task_models": True,
                    },
                    "integrity": {
                        "workflow_count": len(workflows),
                        "node_count": sum(len(item["nodes"]) for item in workflows),
                        "seed": int(seed),
                    },
                }
            )
    return rows


def _network_seconds(byte_count: int, link: dict[str, Any]) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (float(link["mbps"]) * 1_000_000.0) + float(
        link["fixed_one_way_seconds"]
    )


def _objective(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["completed_workflows"]),
        int(summary["deadline_violations"]),
        float(summary["total_completion_seconds"]),
        int(summary["total_transfer_bytes"]),
        float(summary["repeated_compute_seconds"]),
    )


def _simulate(
    workload: dict[str, Any],
    config: dict[str, Any],
    decisions: list[str],
) -> dict[str, Any]:
    base_bytes = int(_value(config, "model", "base_network_directory_bytes"))
    adapter_bytes = int(_value(config, "model", "adapter_network_directory_bytes"))
    input_bytes = int(_value(config, "dynamic", "input_bytes"))
    state_bytes = int(_value(config, "dynamic", "state_package_bytes"))
    prefix = _value(config, "time_seconds", "prefix_compute")
    suffix = _value(config, "time_seconds", "suffix_compute")
    model_load = _value(config, "time_seconds", "target_model_load")
    low_restore = (
        _value(config, "time_seconds", "state_serialize_save")
        + _value(config, "time_seconds", "state_restore_validation")
        + _value(config, "time_seconds", "suffix_input_rebuild")
    )
    restore = (
        low_restore
        if workload["factors"]["restore_cost"] == "low"
        else _value(config, "time_seconds", "high_restore_stress")
    )
    capacity = int(workload["rsu_resources"]["target_model_cache_capacity_bytes"])
    residents: dict[str, tuple[int, int]] = {}
    lru: list[str] = []
    first = workload["workflows"][0]["nodes"][0]
    if workload["factors"]["initial_target_model"] == "ready":
        residents[first["base_model_id"]] = (base_bytes, 0)
        residents[first["adapter_id"]] = (adapter_bytes, 0)
        lru.extend([first["base_model_id"], first["adapter_id"]])

    transfers = {"base": 0, "adapter": 0, "input": 0, "state": 0}
    results = []
    repeated_compute = 0.0
    clock = 0.0
    for index, (workflow, decision) in enumerate(zip(workload["workflows"], decisions)):
        node = workflow["nodes"][1]
        required = [node["base_model_id"], node["adapter_id"]]
        missing: list[tuple[str, int, str]] = []
        if required[0] not in residents:
            missing.append((required[0], base_bytes, "base"))
        if required[1] not in residents:
            missing.append((required[1], adapter_bytes, "adapter"))
        for object_id, size, kind in missing:
            while sum(item[0] for item in residents.values()) + size > capacity and lru:
                victim = lru.pop(0)
                if victim in required:
                    continue
                residents.pop(victim, None)
            residents[object_id] = (size, index)
            lru.append(object_id)
            transfers[kind] += size
        for object_id in required:
            if object_id in lru:
                lru.remove(object_id)
            lru.append(object_id)
        model_prepare_bytes = sum(size for _, size, _ in missing)
        model_prepare_seconds = _network_seconds(model_prepare_bytes, workload["link"])
        if missing:
            model_prepare_seconds += model_load
        if decision == "recover":
            dynamic_bytes = state_bytes
            incremental = restore + _network_seconds(state_bytes, workload["link"])
            transfers["state"] += state_bytes
            restart_prefix = 0.0
        else:
            dynamic_bytes = input_bytes
            incremental = prefix + _network_seconds(input_bytes, workload["link"])
            transfers["input"] += input_bytes
            restart_prefix = prefix
            repeated_compute += prefix
        arrival = float(workflow["arrival_seconds"])
        queue_wait = max(0.0, clock - arrival)
        clock = max(clock, arrival)
        elapsed = prefix + model_prepare_seconds + incremental + suffix
        clock += elapsed
        end_to_end = clock - float(workflow["arrival_seconds"])
        deadline = float(workflow["arrival_seconds"]) + float(
            workflow["deadline_seconds_from_arrival"]
        )
        results.append(
            {
                "workflow_id": workflow["workflow_id"],
                "decision": decision,
                "completed": True,
                "completion_seconds": clock,
                "end_to_end_seconds": end_to_end,
                "queue_wait_seconds": queue_wait,
                "deadline_seconds": deadline,
                "deadline_violation": clock > deadline,
                "model_was_ready": not missing,
                "missing_model_prepare_bytes": model_prepare_bytes,
                "missing_model_prepare_seconds": model_prepare_seconds,
                "state_transfer_bytes": state_bytes if decision == "recover" else 0,
                "input_transfer_bytes": input_bytes if decision == "restart" else 0,
                "restore_seconds": restore if decision == "recover" else 0.0,
                "restart_prefix_compute_seconds": restart_prefix,
                "suffix_compute_seconds": suffix,
                "dynamic_transfer_bytes": dynamic_bytes,
            }
        )
    summary = {
        "completed_nodes": len(results) * 2,
        "completed_workflows": len(results),
        "deadline_violations": sum(item["deadline_violation"] for item in results),
        "total_completion_seconds": sum(item["end_to_end_seconds"] for item in results),
        "makespan_seconds": clock,
        "total_queue_wait_seconds": sum(item["queue_wait_seconds"] for item in results),
        "transfer_bytes_by_type": transfers,
        "total_transfer_bytes": sum(transfers.values()),
        "repeated_compute_seconds": repeated_compute,
        "service_failure_count": 0,
        "cache_prepare_count": sum(bool(item["missing_model_prepare_bytes"]) for item in results),
    }
    return {"decisions": decisions, "workflow_results": results, "summary": summary}


def _local_decisions(workload: dict[str, Any], config: dict[str, Any]) -> list[str]:
    link = workload["link"]
    restart = _value(config, "time_seconds", "prefix_compute") + _network_seconds(
        int(_value(config, "dynamic", "input_bytes")), link
    )
    restore = (
        _value(config, "time_seconds", "state_serialize_save")
        + _value(config, "time_seconds", "state_restore_validation")
        + _value(config, "time_seconds", "suffix_input_rebuild")
        if workload["factors"]["restore_cost"] == "low"
        else _value(config, "time_seconds", "high_restore_stress")
    ) + _network_seconds(int(_value(config, "dynamic", "state_package_bytes")), link)
    return ["recover" if restore < restart else "restart"] * len(workload["workflows"])


def compare_methods(workload: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    count = len(workload["workflows"])
    decision_overheads: dict[str, float] = {}
    started = time.perf_counter()
    current_decisions = ["restart"] * count
    decision_overheads["current_restart_rule"] = time.perf_counter() - started
    started = time.perf_counter()
    local_decisions = _local_decisions(workload, config)
    decision_overheads["mechanism_aware_local_incremental_cost"] = (
        time.perf_counter() - started
    )
    method_decisions = {
        "current_restart_rule": current_decisions,
        "mechanism_aware_local_incremental_cost": local_decisions,
    }
    started = time.perf_counter()
    candidates = [
        _simulate(workload, config, list(decisions))
        for decisions in product(("restart", "recover"), repeat=count)
    ]
    method_decisions["offline_exact_enumeration"] = min(
        candidates, key=lambda item: (_objective(item["summary"]), tuple(item["decisions"]))
    )["decisions"]
    decision_overheads["offline_exact_enumeration"] = time.perf_counter() - started
    rows = []
    for method in config["methods"]:
        result = _simulate(workload, config, list(method_decisions[method]))
        rows.append(
            {
                "design_id": workload["design_id"],
                "seed": workload["seed"],
                "sharing": workload["factors"]["sharing"],
                "capacity": workload["factors"]["capacity"],
                "initial_target_model": workload["factors"]["initial_target_model"],
                "restore_cost": workload["factors"]["restore_cost"],
                "method": method,
                "decisions": list(result["decisions"]),
                "decision_overhead_seconds": decision_overheads[method],
                **result["summary"],
                "workflow_results": result["workflow_results"],
                "offline_future_information": method == "offline_exact_enumeration",
            }
        )
    return rows
