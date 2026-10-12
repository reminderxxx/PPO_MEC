"""Concurrent, trace-driven VEC execution kernel (experimental contract v1).

All time, network and compute resources advance on the same 100 ms NGSIM clock.
The kernel executes DAG *metadata* and output receipts; it does not run an AI model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.specs import WorkflowGraphState


CONTRACT_VERSION = "vec_competition_v1"


@dataclass(frozen=True)
class TracePoint:
    time_ms: int
    x_m: float
    y_m: float
    speed_mps: float
    road_key: str = ""
    direction: str = ""


@dataclass(frozen=True)
class RoadsideUnit:
    rsu_id: str
    x_m: float
    y_m: float
    radius_m: float
    cache_capacity_mb: float = 512.0
    compute_slots: int = 1
    compute_mb_per_s: float = 40.0
    radio_mb_per_s: float = 20.0
    backhaul_mb_per_s: float = 40.0


@dataclass(frozen=True)
class CompetitionScenario:
    source_segment_id: str
    traces: dict[str, tuple[TracePoint, ...]]
    workflows: dict[str, WorkflowGraphState]
    rsus: tuple[RoadsideUnit, ...]
    catalog: AdapterCatalog
    alibaba_size_unit_mb: float = 0.001
    tick_ms: int = 100

    def validate(self) -> None:
        if len(self.rsus) != 4 or len(self.traces) != 8 or set(self.traces) != set(self.workflows):
            raise ValueError("competition contract requires exactly four RSUs and eight distinct vehicle workflows")
        if len({r.rsu_id for r in self.rsus}) != 4:
            raise ValueError("RSU IDs must be unique")
        if self.tick_ms <= 0 or self.alibaba_size_unit_mb <= 0:
            raise ValueError("time step and Alibaba size scale must be positive")
        lengths = {len(points) for points in self.traces.values()}
        if len(lengths) != 1 or min(lengths) < 2:
            raise ValueError("all eight vehicles must have one shared window with at least two frames")
        times = None
        for vehicle_id, points in self.traces.items():
            current = tuple(p.time_ms for p in points)
            if times is None:
                times = current
            if current != times or any(b - a != self.tick_ms for a, b in zip(current, current[1:])):
                raise ValueError(f"non-contiguous or asynchronous trace: {vehicle_id}")
            if any(not all(math.isfinite(v) for v in (p.x_m, p.y_m, p.speed_mps)) for p in points):
                raise ValueError(f"non-finite trace coordinate: {vehicle_id}")
        for rsu in self.rsus:
            if min(rsu.radius_m, rsu.cache_capacity_mb, rsu.compute_mb_per_s, rsu.radio_mb_per_s, rsu.backhaul_mb_per_s) <= 0 or rsu.compute_slots < 1:
                raise ValueError(f"invalid RSU resource: {rsu.rsu_id}")
        for vehicle_id, wf in self.workflows.items():
            nodes = wf.node_map()
            if not nodes or len(nodes) != len(wf.nodes) or len(wf.execution_order) != len(nodes) or set(wf.execution_order) != set(nodes):
                raise ValueError(f"invalid workflow nodes/order: {vehicle_id}")
            if any(p not in nodes for node in nodes.values() for p in node.predecessors):
                raise ValueError(f"unknown predecessor: {vehicle_id}")
            rank = {node_id: index for index, node_id in enumerate(wf.execution_order)}
            if any(rank[p] >= rank[node.node_id] or node.node_id not in nodes[p].successors for node in nodes.values() for p in node.predecessors):
                raise ValueError(f"inconsistent or cyclic workflow edges: {vehicle_id}")
            if any(node.required_adapter not in self.catalog.compatibility_map.get(node.required_base_model, []) for node in nodes.values()):
                raise ValueError(f"adapter/base incompatibility: {vehicle_id}")


@dataclass
class Transfer:
    transfer_id: str
    kind: str
    channel: str
    rsu_id: str
    vehicle_id: str
    node_id: str | None
    object_id: str | None
    source_rsu_id: str | None
    size_mb: float
    remaining_mb: float
    requested_at_ms: int


@dataclass
class NodeRun:
    vehicle_id: str
    node_id: str
    rsu_id: str
    remaining_mb: float
    started_at_ms: int


class VECCompetitionEnv:
    """Four-RSU/eight-vehicle shared-resource DAG replay.

    `prepare` is an explicit slow cache action. `step` places ready nodes at the
    currently attached RSU; cache loading, state transfer and compute all consume
    finite resources. One event ledger records causal state transitions.
    """

    def __init__(self, scenario: CompetitionScenario) -> None:
        scenario.validate()
        self.scenario = scenario
        self.rsus = {r.rsu_id: r for r in scenario.rsus}
        self.vehicle_ids = tuple(sorted(scenario.traces))
        self.frame_index = 0
        self.ledger: list[dict[str, Any]] = []
        self.residents = {rid: set(scenario.catalog.get_initial_typed_residents(rid)) for rid in self.rsus}
        self.last_used: dict[tuple[str, str], int] = {}
        self.reserved: dict[tuple[str, str], str] = {}
        self.transfers: dict[str, Transfer] = {}
        self.running: dict[str, NodeRun] = {}
        self.placement: dict[tuple[str, str], str] = {}
        self.completed: dict[str, set[str]] = {v: set() for v in self.vehicle_ids}
        self.output_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self.arrived: set[tuple[str, str, str, str]] = set()
        self.delivered: set[str] = set()
        self.handoffs: dict[str, int] = {v: 0 for v in self.vehicle_ids}
        self.attached: dict[str, str | None] = {v: None for v in self.vehicle_ids}
        self._sequence = 0
        self._validate_initial_cache()
        self._associate()

    @property
    def now_ms(self) -> int:
        return self.scenario.traces[self.vehicle_ids[0]][self.frame_index].time_ms

    def _event(self, event_type: str, **fields: Any) -> None:
        self.ledger.append({"event_id": len(self.ledger) + 1, "time_ms": self.now_ms, "event": event_type, **fields})

    def _validate_initial_cache(self) -> None:
        catalog = self.scenario.catalog
        for rid, objects in self.residents.items():
            for object_id in objects:
                catalog.get_typed_object(object_id)
            if self._used_mb(rid) > self.rsus[rid].cache_capacity_mb + 1e-9:
                raise ValueError(f"initial residents exceed cache capacity at {rid}")

    def _used_mb(self, rid: str) -> float:
        catalog = self.scenario.catalog
        objects = self.residents[rid] | {oid for (target, oid) in self.reserved if target == rid}
        return sum(catalog.get_typed_object(oid).resident_size_mb for oid in objects if catalog.get_typed_object(oid).counts_toward_capacity)

    def _associate(self) -> None:
        for vehicle_id in self.vehicle_ids:
            point = self.scenario.traces[vehicle_id][self.frame_index]
            candidates = [(math.hypot(point.x_m - r.x_m, point.y_m - r.y_m), r.rsu_id) for r in self.rsus.values()]
            distance, nearest = min(candidates)
            target = nearest if distance <= self.rsus[nearest].radius_m else None
            previous = self.attached[vehicle_id]
            if target != previous:
                for tid, transfer in list(self.transfers.items()):
                    if transfer.vehicle_id == vehicle_id and transfer.channel == "radio" and transfer.rsu_id != target:
                        del self.transfers[tid]
                        self._event("transfer_cancelled", transfer_id=tid, reason="radio_handoff")
                for key, placed_rsu in list(self.placement.items()):
                    if key[0] == vehicle_id and key[1] not in self.completed[vehicle_id] and vehicle_id not in self.running:
                        del self.placement[key]
                if previous is not None and target is not None:
                    self.handoffs[vehicle_id] += 1
                    self._event("handoff", vehicle_id=vehicle_id, from_rsu=previous, to_rsu=target)
                elif target is None:
                    self._event("coverage_lost", vehicle_id=vehicle_id, from_rsu=previous)
                else:
                    self._event("coverage_attached", vehicle_id=vehicle_id, to_rsu=target)
            self.attached[vehicle_id] = target

    def _new_transfer(self, kind: str, channel: str, rid: str, vehicle_id: str, node_id: str | None, object_id: str | None, source_rsu_id: str | None, size_mb: float) -> None:
        self._sequence += 1
        transfer_id = f"transfer_{self._sequence}"
        self.transfers[transfer_id] = Transfer(transfer_id, kind, channel, rid, vehicle_id, node_id, object_id, source_rsu_id, size_mb, size_mb, self.now_ms)
        self._event("transfer_requested", transfer_id=transfer_id, kind=kind, vehicle_id=vehicle_id, node_id=node_id, rsu_id=rid, source_rsu_id=source_rsu_id, object_id=object_id, size_mb=size_mb)

    def _admit(self, rid: str, object_id: str, vehicle_id: str, node_id: str | None) -> bool:
        key = (rid, object_id)
        if object_id in self.residents[rid] or key in self.reserved:
            return True
        catalog = self.scenario.catalog
        item = catalog.get_typed_object(object_id)
        if item.object_type == "workflow_state":
            raise ValueError("workflow state is a transfer payload, not a cache resident")
        if item.resident_size_mb > self.rsus[rid].cache_capacity_mb:
            self._event("cache_rejected", rsu_id=rid, object_id=object_id, reason="object_exceeds_capacity")
            return False
        while self._used_mb(rid) + item.resident_size_mb > self.rsus[rid].cache_capacity_mb + 1e-9:
            candidates = [oid for oid in self.residents[rid] if catalog.get_typed_object(oid).evictability == "evictable" and not any(run.rsu_id == rid and oid in self._required(run.vehicle_id, run.node_id) for run in self.running.values()) and not any(oid in catalog.get_typed_object(other).dependency_ids for other in self.residents[rid])]
            if not candidates:
                self._event("cache_rejected", rsu_id=rid, object_id=object_id, reason="no_evictable_capacity")
                return False
            victim = min(candidates, key=lambda oid: (self.last_used.get((rid, oid), -1), oid))
            self.residents[rid].remove(victim)
            self._event("cache_evicted", rsu_id=rid, object_id=victim)
        self.reserved[key] = vehicle_id
        self._new_transfer("model_load", "backhaul", rid, vehicle_id, node_id, object_id, None, item.transfer_size_mb)
        return True

    def _required(self, vehicle_id: str, node_id: str) -> tuple[str, str]:
        node = self.scenario.workflows[vehicle_id].node_map()[node_id]
        base = self.scenario.catalog.get_typed_base(node.required_base_model).object_id
        adapter = self.scenario.catalog.get_typed_adapter(node.required_adapter).object_id
        return base, adapter

    def prepare(self, vehicle_id: str, target_rsu_id: str, node_id: str) -> bool:
        if vehicle_id not in self.scenario.workflows or target_rsu_id not in self.rsus:
            raise ValueError("unknown vehicle or RSU")
        if node_id not in self.scenario.workflows[vehicle_id].node_map():
            raise ValueError("unknown workflow node")
        if node_id in self.completed[vehicle_id]:
            return False
        results = [self._admit(target_rsu_id, oid, vehicle_id, node_id) for oid in self._required(vehicle_id, node_id)]
        self._event("prepare_action", vehicle_id=vehicle_id, node_id=node_id, rsu_id=target_rsu_id, accepted=all(results))
        return all(results)

    def _schedule(self) -> None:
        for vehicle_id in self.vehicle_ids:
            rid = self.attached[vehicle_id]
            if rid is None or vehicle_id in self.delivered:
                continue
            wf = self.scenario.workflows[vehicle_id]
            for node_id in wf.execution_order:
                if node_id in self.completed[vehicle_id] or vehicle_id in self.running:
                    continue
                node = wf.node_map()[node_id]
                if not set(node.predecessors).issubset(self.completed[vehicle_id]):
                    continue
                key = (vehicle_id, node_id)
                if key not in self.placement:
                    self.placement[key] = rid
                    self._event("node_placed", vehicle_id=vehicle_id, node_id=node_id, rsu_id=rid)
                target = self.placement[key]
                for oid in self._required(vehicle_id, node_id):
                    self._admit(target, oid, vehicle_id, node_id)
                if not node.predecessors:
                    receipt = (vehicle_id, node_id, "input", target)
                    if receipt not in self.arrived and not self._inflight("uplink", vehicle_id, node_id, target, "input"):
                        self._new_transfer("uplink", "radio", target, vehicle_id, node_id, "input", None, max(0.001, node.input_size * self.scenario.alibaba_size_unit_mb))
                for pred in node.predecessors:
                    receipt = self.output_receipts[(vehicle_id, pred)]
                    object_id = receipt["object_id"]
                    if receipt["rsu_id"] == target:
                        self.arrived.add((vehicle_id, node_id, object_id, target))
                    elif (vehicle_id, node_id, object_id, target) not in self.arrived and not self._inflight("state_migration", vehicle_id, node_id, target, object_id):
                        self._new_transfer("state_migration", "backhaul", target, vehicle_id, node_id, object_id, receipt["rsu_id"], receipt["size_mb"])
                ready_inputs = all((vehicle_id, node_id, self.output_receipts[(vehicle_id, pred)]["object_id"], target) in self.arrived for pred in node.predecessors) if node.predecessors else (vehicle_id, node_id, "input", target) in self.arrived
                ready_model = all(oid in self.residents[target] for oid in self._required(vehicle_id, node_id))
                if not ready_inputs or not ready_model:
                    continue
                if sum(run.rsu_id == target for run in self.running.values()) >= self.rsus[target].compute_slots:
                    continue
                for oid in self._required(vehicle_id, node_id):
                    self.last_used[(target, oid)] = self.now_ms
                work_mb = max(0.001, (node.input_size + node.output_size) * self.scenario.alibaba_size_unit_mb)
                self.running[vehicle_id] = NodeRun(vehicle_id, node_id, target, work_mb, self.now_ms)
                self._event("compute_started", vehicle_id=vehicle_id, node_id=node_id, rsu_id=target, work_mb=work_mb)
                break

    def _inflight(self, kind: str, vehicle_id: str, node_id: str, rid: str, object_id: str) -> bool:
        return any(t.kind == kind and t.vehicle_id == vehicle_id and t.node_id == node_id and t.rsu_id == rid and t.object_id == object_id for t in self.transfers.values())

    def _advance_transfers(self, dt_s: float) -> None:
        groups: dict[tuple[str, str], list[Transfer]] = {}
        for transfer in self.transfers.values():
            groups.setdefault((transfer.channel, transfer.rsu_id), []).append(transfer)
            if transfer.channel == "backhaul" and transfer.source_rsu_id in self.rsus:
                groups.setdefault(("backhaul_source", transfer.source_rsu_id), []).append(transfer)
        finished: list[str] = []
        for transfer in self.transfers.values():
            group = groups[(transfer.channel, transfer.rsu_id)]
            rate = (self.rsus[transfer.rsu_id].radio_mb_per_s if transfer.channel == "radio" else self.rsus[transfer.rsu_id].backhaul_mb_per_s) / len(group)
            if transfer.channel == "backhaul" and transfer.source_rsu_id in self.rsus:
                source_group = groups[("backhaul_source", transfer.source_rsu_id)]
                rate = min(rate, self.rsus[transfer.source_rsu_id].backhaul_mb_per_s / len(source_group))
            transfer.remaining_mb -= rate * dt_s
            if transfer.remaining_mb <= 1e-9:
                finished.append(transfer.transfer_id)
        for tid in finished:
            t = self.transfers.pop(tid)
            if t.kind == "model_load":
                self.reserved.pop((t.rsu_id, str(t.object_id)))
                self.residents[t.rsu_id].add(str(t.object_id))
            elif t.kind in {"uplink", "state_migration", "final_backhaul"}:
                self.arrived.add((t.vehicle_id, str(t.node_id), str(t.object_id), t.rsu_id))
            elif t.kind == "downlink":
                self.delivered.add(t.vehicle_id)
                self._event("workflow_delivered", vehicle_id=t.vehicle_id, workflow_id=self.scenario.workflows[t.vehicle_id].workflow_id, rsu_id=t.rsu_id)
            self._event("transfer_completed", transfer_id=tid, kind=t.kind, vehicle_id=t.vehicle_id, node_id=t.node_id, rsu_id=t.rsu_id, object_id=t.object_id, size_mb=t.size_mb, duration_ms=self.now_ms - t.requested_at_ms)

    def _advance_compute(self, dt_s: float) -> None:
        finished: list[str] = []
        for vehicle_id, run in self.running.items():
            run.remaining_mb -= self.rsus[run.rsu_id].compute_mb_per_s * dt_s
            if run.remaining_mb <= 1e-9:
                finished.append(vehicle_id)
        for vehicle_id in finished:
            run = self.running.pop(vehicle_id)
            node = self.scenario.workflows[vehicle_id].node_map()[run.node_id]
            receipt_id = hashlib.sha256(f"{vehicle_id}|{run.node_id}|{run.rsu_id}|{self.now_ms}".encode()).hexdigest()
            self.completed[vehicle_id].add(run.node_id)
            self.output_receipts[(vehicle_id, run.node_id)] = {"object_id": receipt_id, "rsu_id": run.rsu_id, "size_mb": max(0.001, node.output_size * self.scenario.alibaba_size_unit_mb), "completed_at_ms": self.now_ms}
            self._event("compute_completed", vehicle_id=vehicle_id, node_id=run.node_id, rsu_id=run.rsu_id, output_receipt=receipt_id, duration_ms=self.now_ms - run.started_at_ms)

    def _schedule_delivery(self) -> None:
        for vehicle_id in self.vehicle_ids:
            if vehicle_id in self.delivered or len(self.completed[vehicle_id]) != len(self.scenario.workflows[vehicle_id].nodes):
                continue
            rid = self.attached[vehicle_id]
            if rid is None:
                continue
            sinks = [n for n in self.scenario.workflows[vehicle_id].nodes if not n.successors]
            for node in sinks:
                receipt = self.output_receipts[(vehicle_id, node.node_id)]
                object_id = receipt["object_id"]
                key = (vehicle_id, node.node_id, object_id, rid)
                if receipt["rsu_id"] != rid and key not in self.arrived and not self._inflight("final_backhaul", vehicle_id, node.node_id, rid, object_id):
                    self._new_transfer("final_backhaul", "backhaul", rid, vehicle_id, node.node_id, object_id, receipt["rsu_id"], receipt["size_mb"])
                    continue
                if receipt["rsu_id"] == rid:
                    self.arrived.add(key)
            if all((vehicle_id, node.node_id, self.output_receipts[(vehicle_id, node.node_id)]["object_id"], rid) in self.arrived for node in sinks):
                if not any(t.kind == "downlink" and t.vehicle_id == vehicle_id for t in self.transfers.values()):
                    size = sum(self.output_receipts[(vehicle_id, node.node_id)]["size_mb"] for node in sinks)
                    self._new_transfer("downlink", "radio", rid, vehicle_id, "sink", "result", None, size)

    def step(self) -> bool:
        """Advance one measured frame. Returns false after the last frame."""
        if self.frame_index >= len(self.scenario.traces[self.vehicle_ids[0]]) - 1:
            return False
        self.frame_index += 1
        self._associate()
        dt_s = self.scenario.tick_ms / 1000.0
        self._advance_transfers(dt_s)
        self._advance_compute(dt_s)
        self._schedule()
        self._schedule_delivery()
        return True

    def run(self) -> dict[str, Any]:
        while self.step():
            pass
        return self.summary()

    def summary(self) -> dict[str, Any]:
        all_nodes = sum(len(w.nodes) for w in self.scenario.workflows.values())
        completed_nodes = sum(len(nodes) for nodes in self.completed.values())
        transfer_mb: dict[str, float] = {}
        delivery_at: dict[str, int] = {}
        for event in self.ledger:
            if event["event"] == "transfer_completed":
                kind = str(event["kind"])
                transfer_mb[kind] = transfer_mb.get(kind, 0.0) + float(event["size_mb"])
            elif event["event"] == "workflow_delivered":
                delivery_at[str(event["vehicle_id"])] = int(event["time_ms"])
        start_ms = self.scenario.traces[self.vehicle_ids[0]][0].time_ms
        return {
            "contract_version": CONTRACT_VERSION,
            "source_segment_id": self.scenario.source_segment_id,
            "vehicle_count": len(self.vehicle_ids),
            "rsu_count": len(self.rsus),
            "frame_count": self.frame_index + 1,
            "window_start_ms": self.scenario.traces[self.vehicle_ids[0]][0].time_ms,
            "window_end_ms": self.now_ms,
            "completed_nodes": completed_nodes,
            "total_nodes": all_nodes,
            "delivered_workflows": len(self.delivered),
            "handoffs": sum(self.handoffs.values()),
            "model_loads": sum(e["event"] == "transfer_completed" and e.get("kind") == "model_load" for e in self.ledger),
            "state_migrations": sum(e["event"] == "transfer_completed" and e.get("kind") == "state_migration" for e in self.ledger),
            "ledger_event_count": len(self.ledger),
            "transfer_mb_by_kind": {kind: round(value, 6) for kind, value in sorted(transfer_mb.items())},
            "per_vehicle": {vid: {"workflow_id": self.scenario.workflows[vid].workflow_id, "completed_nodes": len(self.completed[vid]), "total_nodes": len(self.scenario.workflows[vid].nodes), "delivered": vid in self.delivered, "delivery_latency_ms": delivery_at[vid] - start_ms if vid in delivery_at else None, "handoffs": self.handoffs[vid]} for vid in self.vehicle_ids},
            "cache_used_mb": {rid: round(self._used_mb(rid), 6) for rid in self.rsus},
            "resource_profile": {rid: vars(r) for rid, r in self.rsus.items()},
            "alibaba_size_unit_mb": self.scenario.alibaba_size_unit_mb,
            "measurement_status": "measured_NGSIM_coordinates_and_times; Alibaba_DAG_structure; controlled_model_network_compute_and_payload_sizes",
        }

    def ledger_digest(self) -> str:
        blob = json.dumps(self.ledger, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode()).hexdigest()
