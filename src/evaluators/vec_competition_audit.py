"""Independent causal and resource audit of a competition event ledger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.envs.core.vec_competition_contract import FrozenScenario


def audit_competition_run(scenario: FrozenScenario, ledger: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    capacities = {"cloud_port": scenario.cloud_port_bytes_per_s}
    for rsu in scenario.rsus:
        capacities[f"radio:{rsu.rsu_id}"] = rsu.radio_bytes_per_s
        capacities[f"port:{rsu.rsu_id}"] = rsu.backhaul_port_bytes_per_s
    for link in scenario.backhaul_links:
        capacities[link.key] = link.bytes_per_s
    flow_resources: dict[str, tuple[str, ...]] = {}
    flow_sizes: dict[str, int] = {}
    flow_owners: dict[str, tuple[str, str]] = {}
    flow_progress: dict[str, float] = defaultdict(float)
    flow_terminal: set[str] = set()
    sent_by_kind: dict[str, float] = defaultdict(float)
    sent_by_vehicle: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    completed: dict[str, set[str]] = defaultdict(set)
    resident = {rsu.rsu_id: set(scenario.catalog.get_initial_typed_residents(rsu.rsu_id)) for rsu in scenario.rsus}
    arrived: set[str] = set()
    delivered: dict[str, float] = {}
    resource_by_tick: dict[tuple[float, str], float] = defaultdict(float)
    compute_by_tick: dict[tuple[float, str], float] = defaultdict(float)
    transfer_bytes: dict[str, int] = defaultdict(int)
    queue_wait_s = 0.0
    previous_time = -1.0
    for index, event in enumerate(ledger, 1):
        time_s = float(event["time_s"])
        kind = event["event"]
        if event["event_id"] != index or time_s + 1e-9 < previous_time:
            failures.append(f"event order violation at {index}")
        previous_time = time_s
        if kind == "request_arrived":
            if event["vehicle_id"] in arrived:
                failures.append("duplicate request arrival")
            arrived.add(event["vehicle_id"])
        elif kind == "flow_requested":
            if event["flow_id"] in flow_resources:
                failures.append(f"duplicate flow request: {event['flow_id']}")
            flow_resources[event["flow_id"]] = tuple(event["resources"])
            flow_sizes[event["flow_id"]] = int(event["size_bytes"])
            flow_owners[event["flow_id"]] = (event["vehicle_id"], event["kind"])
            if any(resource not in capacities for resource in event["resources"]):
                failures.append(f"unknown flow resource: {event['flow_id']}")
        elif kind == "flow_progress":
            flow_id = event["flow_id"]
            if flow_id not in flow_sizes or flow_id in flow_terminal:
                failures.append(f"progress outside active flow: {flow_id}")
            flow_progress[flow_id] += float(event["sent_bytes"])
            if flow_id in flow_owners:
                vehicle_id, flow_kind = flow_owners[flow_id]
                sent_by_kind[flow_kind] += float(event["sent_bytes"])
                sent_by_vehicle[vehicle_id][flow_kind] += float(event["sent_bytes"])
            if float(event["sent_bytes"]) <= 0 or flow_progress[flow_id] > flow_sizes.get(flow_id, 0) + 1e-3:
                failures.append(f"invalid flow byte progress: {flow_id}")
            for resource in flow_resources.get(event["flow_id"], ()):
                resource_by_tick[(time_s, resource)] += float(event["sent_bytes"])
        elif kind == "flow_completed":
            flow_id = event["flow_id"]
            if flow_id not in flow_sizes or flow_id in flow_terminal or abs(flow_progress[flow_id] - flow_sizes.get(flow_id, 0)) > 1e-3 or int(event["size_bytes"]) != flow_sizes.get(flow_id):
                failures.append(f"completed flow byte mismatch: {flow_id}")
            flow_terminal.add(flow_id)
            transfer_bytes[event["kind"]] += int(event["size_bytes"])
            if event["kind"] == "state_migration" and event["object_id"] not in completed[event["vehicle_id"]]:
                failures.append(f"state migrated before predecessor completed: {event['vehicle_id']}")
        elif kind == "flow_cancelled":
            flow_id = event["flow_id"]
            if flow_id not in flow_sizes or flow_id in flow_terminal or abs(flow_progress[flow_id] - float(event["sent_bytes"])) > 1e-3:
                failures.append(f"cancelled flow byte mismatch: {flow_id}")
            flow_terminal.add(flow_id)
        elif kind == "model_resident":
            object_id = event["object_id"]
            rid = event["rsu_id"]
            if any(dep not in resident[rid] for dep in scenario.catalog.get_typed_object(object_id).dependency_ids):
                failures.append(f"resident dependency missing: {rid}/{object_id}")
            resident[rid].add(object_id)
        elif kind == "cache_evicted":
            resident[event["rsu_id"]].discard(event["object_id"])
        elif kind in {"cache_reserved", "cache_evicted", "model_resident"}:
            pass
        if kind in {"cache_reserved", "cache_evicted", "model_resident"}:
            rid = event["rsu_id"]
            capacity = next(r.cache_capacity_bytes for r in scenario.rsus if r.rsu_id == rid)
            if int(event["used_bytes_after"]) > capacity:
                failures.append(f"cache capacity exceeded: {rid}")
        if kind == "compute_started":
            vehicle_id = event["vehicle_id"]
            node_id = event["node_id"]
            if vehicle_id not in arrived:
                failures.append(f"compute before request arrival: {vehicle_id}")
            node = scenario.workflows[vehicle_id].node_map()[node_id]
            if not set(node.predecessors).issubset(completed[vehicle_id]):
                failures.append(f"DAG predecessor incomplete: {vehicle_id}/{node_id}")
            base = scenario.catalog.get_typed_base(node.required_base_model).object_id
            adapter = scenario.catalog.get_typed_adapter(node.required_adapter).object_id
            if base not in resident[event["rsu_id"]] or adapter not in resident[event["rsu_id"]]:
                failures.append(f"compute before model ready: {vehicle_id}/{node_id}")
            queue_wait_s += float(event["queue_wait_s"])
        elif kind == "compute_progress":
            compute_by_tick[(time_s, event["rsu_id"])] += float(event["work_bytes"])
        elif kind == "compute_completed":
            completed[event["vehicle_id"]].add(event["node_id"])
        elif kind == "workflow_delivered":
            vehicle_id = event["vehicle_id"]
            if len(completed[vehicle_id]) != len(scenario.workflows[vehicle_id].nodes):
                failures.append(f"delivered before DAG completion: {vehicle_id}")
            delivered[vehicle_id] = time_s
    for (time_s, resource), bytes_sent in resource_by_tick.items():
        if resource not in capacities or bytes_sent > capacities.get(resource, 0) * scenario.tick_s + 1e-3:
            failures.append(f"resource oversubscribed: {resource} at {time_s}")
    for (time_s, rid), work_bytes in compute_by_tick.items():
        capacity = next(r.compute_work_bytes_per_s for r in scenario.rsus if r.rsu_id == rid)
        slots = next(r.compute_slots for r in scenario.rsus if r.rsu_id == rid)
        if work_bytes > capacity * scenario.tick_s * slots + 1e-3:
            failures.append(f"compute oversubscribed: {rid} at {time_s}")
    if arrived != set(scenario.requests):
        failures.append("request arrival denominator mismatch")
    if len(delivered) != summary["delivered_on_time"] + summary["delivered_late"]:
        failures.append("delivery count mismatch")
    if len(arrived) != summary["generated_requests"]:
        failures.append("generated request count mismatch")
    if sum(len(nodes) for nodes in completed.values()) != summary["completed_nodes"]:
        failures.append("completed node count mismatch")
    if dict(sorted(transfer_bytes.items())) != summary["completed_transfer_bytes_by_kind"]:
        failures.append("completed transfer bytes mismatch")
    if {kind: round(value, 6) for kind, value in sorted(sent_by_kind.items())} != summary["sent_transfer_bytes_by_kind"]:
        failures.append("sent transfer bytes mismatch")
    for vehicle_id in scenario.requests:
        expected = {kind: round(value, 6) for kind, value in sorted(sent_by_vehicle[vehicle_id].items())}
        if expected != summary["per_vehicle"][vehicle_id]["sent_transfer_bytes_by_kind"]:
            failures.append(f"vehicle sent bytes mismatch: {vehicle_id}")
    if abs(queue_wait_s - summary["compute_queue_wait_s"]) > 1e-6:
        failures.append("compute queue wait mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "audited_events": len(ledger),
        "audited_requests": len(arrived),
        "audited_deliveries": len(delivered),
        "audited_resource_tick_cells": len(resource_by_tick),
        "audited_compute_tick_cells": len(compute_by_tick),
        "evidence_scope": "engineering event conservation and causality; controlled service parameters; not model correctness or formal statistical inference",
    }
