"""Deterministic 4-RSU/8-request event runtime with shared finite resources."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from src.envs.core.vec_competition_contract import (
    DECIMAL_MB, AccessPolicy, AttachedFifoPlacement, ControllerAction,
    ControllerPolicy, FrozenScenario, MetadataReceiptCodec, NearestCoveredAccess,
    PlacementPolicy, StateCodec,
)


@dataclass
class Flow:
    flow_id: str
    kind: str
    vehicle_id: str
    node_id: str | None
    object_id: str
    source_rsu_id: str | None
    target_rsu_id: str
    size_bytes: int
    sent_bytes: float
    resources: tuple[str, ...]
    requested_s: float
    ready_s: float


@dataclass
class ComputeRun:
    vehicle_id: str
    node_id: str
    rsu_id: str
    work_bytes: int
    completed_work_bytes: float
    started_s: float
    queued_s: float


class VECCompetitionRuntime:
    """Frame-discrete execution; all internal time is seconds and all sizes bytes.

    Access, placement and state serialization are injected plugins. The default
    codec serializes metadata only; the declared task bytes remain controlled.
    """

    contract_version = "vec_competition_v2"

    def __init__(
        self,
        scenario: FrozenScenario,
        *,
        access_policy: AccessPolicy | None = None,
        placement_policy: PlacementPolicy | None = None,
        state_codec: StateCodec | None = None,
    ) -> None:
        scenario.validate()
        self.scenario = scenario
        self.access_policy = access_policy or NearestCoveredAccess()
        self.placement_policy = placement_policy or AttachedFifoPlacement()
        self.state_codec = state_codec or MetadataReceiptCodec()
        self.rsu_by_id = {rsu.rsu_id: rsu for rsu in scenario.rsus}
        self.vehicle_ids = tuple(sorted(scenario.traces))
        self.frame_index = 0
        self.ledger: list[dict[str, Any]] = []
        self.attached: dict[str, str | None] = {vehicle_id: None for vehicle_id in self.vehicle_ids}
        self.association_timeline: list[dict[str, Any]] = []
        self.arrived_requests: set[str] = set()
        self.deadline_missed: set[str] = set()
        self.delivered_at: dict[str, float] = {}
        self.residents: dict[str, set[str]] = {rid: set(scenario.catalog.get_initial_typed_residents(rid)) for rid in self.rsu_by_id}
        self.reserved: dict[tuple[str, str], bool] = {}  # bool means transfer finished, dependency promotion pending
        self.last_used: dict[tuple[str, str], float] = {}
        self.flows: dict[str, Flow] = {}
        self.running: dict[tuple[str, str], ComputeRun] = {}
        self.queue: dict[tuple[str, str], float] = {}
        self.placement: dict[tuple[str, str], str] = {}
        self.completed: dict[str, set[str]] = {vehicle_id: set() for vehicle_id in self.vehicle_ids}
        self.outputs: dict[tuple[str, str], dict[str, Any]] = {}
        self.input_at: set[tuple[str, str, str, str]] = set()
        self.handoffs: dict[str, int] = {vehicle_id: 0 for vehicle_id in self.vehicle_ids}
        self._flow_counter = 0
        self._finalized = False
        self._predictions: dict[str, list[tuple[str, float, float, str, tuple[str, ...]]]] = defaultdict(list)
        self._capacity = self._resource_capacities()
        self._validate_initial_residents()
        self._associate()
        self._arrivals_and_deadlines()
        self._schedule()

    @property
    def now_s(self) -> float:
        return self.scenario.traces[self.vehicle_ids[0]][self.frame_index].time_s

    def _emit(self, event: str, **fields: Any) -> None:
        self.ledger.append({"event_id": len(self.ledger) + 1, "time_s": round(self.now_s, 9), "event": event, **fields})

    def _resource_capacities(self) -> dict[str, float]:
        resources = {"cloud_port": self.scenario.cloud_port_bytes_per_s}
        for rsu in self.scenario.rsus:
            resources[f"radio:{rsu.rsu_id}"] = rsu.radio_bytes_per_s
            resources[f"port:{rsu.rsu_id}"] = rsu.backhaul_port_bytes_per_s
        for link in self.scenario.backhaul_links:
            resources[link.key] = link.bytes_per_s
        return resources

    def _path(self, source: str, target: str) -> tuple[str, ...]:
        if source == target:
            return (source,)
        adjacency: dict[str, list[str]] = defaultdict(list)
        for link in self.scenario.backhaul_links:
            adjacency[link.source].append(link.target)
            adjacency[link.target].append(link.source)
        pending: deque[tuple[str, ...]] = deque([(source,)])
        visited = {source}
        while pending:
            path = pending.popleft()
            for neighbor in sorted(adjacency[path[-1]]):
                if neighbor in visited:
                    continue
                candidate = (*path, neighbor)
                if neighbor == target:
                    return candidate
                visited.add(neighbor)
                pending.append(candidate)
        raise ValueError(f"no backhaul route: {source} -> {target}")

    def _route_resources(self, source: str | None, target: str, channel: str) -> tuple[tuple[str, ...], float]:
        if channel == "radio":
            return (f"radio:{target}",), 0.0
        if source is None:
            return ("cloud_port", f"port:{target}"), 0.005
        path = self._path(source, target)
        resources = [f"port:{rid}" for rid in path]
        latency = 0.0
        for a, b in zip(path, path[1:]):
            link = next(link for link in self.scenario.backhaul_links if {link.source, link.target} == {a, b})
            resources.append(link.key)
            latency += link.latency_s
        return tuple(resources), latency

    def _new_flow(self, kind: str, vehicle_id: str, node_id: str | None, object_id: str, target: str, size_bytes: int, *, source: str | None = None, channel: str = "backhaul") -> str:
        if size_bytes <= 0:
            raise ValueError("transfers require positive bytes")
        resources, latency = self._route_resources(source, target, channel)
        self._flow_counter += 1
        flow_id = f"flow_{self._flow_counter}"
        self.flows[flow_id] = Flow(flow_id, kind, vehicle_id, node_id, object_id, source, target, size_bytes, 0.0, resources, self.now_s, self.now_s + latency)
        self._emit("flow_requested", flow_id=flow_id, kind=kind, vehicle_id=vehicle_id, node_id=node_id, object_id=object_id, source_rsu_id=source, target_rsu_id=target, size_bytes=size_bytes, resources=list(resources), propagation_s=round(latency, 6))
        return flow_id

    def _cancel_flow(self, flow_id: str, reason: str) -> None:
        flow = self.flows.pop(flow_id)
        if flow.kind == "model_load":
            self.reserved.pop((flow.target_rsu_id, flow.object_id), None)
            dependent = [oid for (rid, oid) in self.reserved if rid == flow.target_rsu_id and flow.object_id in self.scenario.catalog.get_typed_object(oid).dependency_ids]
            for object_id in dependent:
                other = next((fid for fid, pending in self.flows.items() if pending.kind == "model_load" and pending.target_rsu_id == flow.target_rsu_id and pending.object_id == object_id), None)
                if other is not None:
                    self._cancel_flow(other, "dependency_load_cancelled")
                else:
                    self.reserved.pop((flow.target_rsu_id, object_id), None)
                    self._emit("cache_reservation_released", rsu_id=flow.target_rsu_id, object_id=object_id, reason="dependency_load_cancelled")
        self._emit("flow_cancelled", flow_id=flow_id, kind=flow.kind, vehicle_id=flow.vehicle_id, node_id=flow.node_id, target_rsu_id=flow.target_rsu_id, sent_bytes=round(flow.sent_bytes, 6), reason=reason)

    def cancel_flow(self, flow_id: str, *, reason: str = "injected_failure") -> None:
        """Deterministic failure-injection hook; never fabricates completed bytes."""
        if flow_id not in self.flows:
            raise KeyError(flow_id)
        self._cancel_flow(flow_id, reason)

    def _used_bytes(self, rid: str) -> int:
        objects = self.residents[rid] | {object_id for target, object_id in self.reserved if target == rid}
        return sum(round(self.scenario.catalog.get_typed_object(oid).resident_size_mb * DECIMAL_MB) for oid in objects if self.scenario.catalog.get_typed_object(oid).counts_toward_capacity)

    def _validate_initial_residents(self) -> None:
        for rid, objects in self.residents.items():
            for object_id in objects:
                obj = self.scenario.catalog.get_typed_object(object_id)
                if any(dep not in objects for dep in obj.dependency_ids):
                    raise ValueError(f"orphan initial resident at {rid}: {object_id}")
            if self._used_bytes(rid) > self.rsu_by_id[rid].cache_capacity_bytes:
                raise ValueError(f"initial cache exceeds capacity: {rid}")

    def _required_objects(self, vehicle_id: str, node_id: str) -> tuple[str, str]:
        node = self.scenario.workflows[vehicle_id].node_map()[node_id]
        return (self.scenario.catalog.get_typed_base(node.required_base_model).object_id, self.scenario.catalog.get_typed_adapter(node.required_adapter).object_id)

    def _ensure_models(self, rid: str, vehicle_id: str, node_id: str) -> bool:
        required = self._required_objects(vehicle_id, node_id)
        missing = [oid for oid in required if oid not in self.residents[rid] and (rid, oid) not in self.reserved]
        if not missing:
            return True
        catalog = self.scenario.catalog
        shadow = set(self.residents[rid])
        protected = {oid for run in self.running.values() if run.rsu_id == rid for oid in self._required_objects(run.vehicle_id, run.node_id)}
        protected.update(oid for key in self.queue if self.placement.get(key) == rid for oid in self._required_objects(*key))
        protected.update(oid for target, oid in self.reserved if target == rid)
        protected.update(dep for target, oid in self.reserved if target == rid for dep in catalog.get_typed_object(oid).dependency_ids)
        required_capacity = sum(round(catalog.get_typed_object(oid).resident_size_mb * DECIMAL_MB) for oid in missing)
        used = self._used_bytes(rid)
        victims: list[str] = []
        while used + required_capacity > self.rsu_by_id[rid].cache_capacity_bytes:
            candidates = [oid for oid in shadow if oid not in protected and catalog.get_typed_object(oid).evictability == "evictable" and not any(oid in catalog.get_typed_object(other).dependency_ids for other in shadow)]
            if not candidates:
                self._emit("cache_rejected", rsu_id=rid, vehicle_id=vehicle_id, node_id=node_id, missing_objects=missing, reason="insufficient_dependency_safe_capacity")
                return False
            victim = min(candidates, key=lambda oid: (self.last_used.get((rid, oid), -1.0), oid))
            shadow.remove(victim)
            victims.append(victim)
            used -= round(catalog.get_typed_object(victim).resident_size_mb * DECIMAL_MB)
        for victim in victims:
            self.residents[rid].remove(victim)
            self._emit("cache_evicted", rsu_id=rid, object_id=victim, used_bytes_after=self._used_bytes(rid))
        for object_id in missing:
            obj = catalog.get_typed_object(object_id)
            self.reserved[(rid, object_id)] = False
            self._emit("cache_reserved", rsu_id=rid, object_id=object_id, resident_bytes=round(obj.resident_size_mb * DECIMAL_MB), used_bytes_after=self._used_bytes(rid))
            self._new_flow("model_load", vehicle_id, node_id, object_id, rid, round(obj.transfer_size_mb * DECIMAL_MB))
        return True

    def _promote_loaded(self, rid: str) -> None:
        changed = True
        while changed:
            changed = False
            for (target, oid), loaded in list(self.reserved.items()):
                if target != rid or not loaded:
                    continue
                obj = self.scenario.catalog.get_typed_object(oid)
                if all(dep in self.residents[rid] for dep in obj.dependency_ids):
                    self.residents[rid].add(oid)
                    del self.reserved[(rid, oid)]
                    self._emit("model_resident", rsu_id=rid, object_id=oid, object_type=obj.object_type, used_bytes_after=self._used_bytes(rid))
                    changed = True

    def _associate(self) -> None:
        snapshot: dict[str, str | None] = {}
        for vehicle_id in self.vehicle_ids:
            point = self.scenario.traces[vehicle_id][self.frame_index]
            target = self.access_policy.associate(point, self.attached[vehicle_id], self.scenario.rsus)
            if target is not None and target not in self.rsu_by_id:
                raise ValueError("access plugin returned unknown RSU")
            previous = self.attached[vehicle_id]
            if target != previous:
                for flow_id, flow in list(self.flows.items()):
                    if flow.vehicle_id == vehicle_id and flow.kind in {"uplink", "downlink"} and flow.target_rsu_id != target:
                        self._cancel_flow(flow_id, "radio_handoff_or_outage")
                for key, rid in list(self.placement.items()):
                    if key[0] == vehicle_id and key not in self.running and key[1] not in self.completed[vehicle_id]:
                        del self.placement[key]
                        self.queue.pop(key, None)
                        for flow_id, flow in list(self.flows.items()):
                            if flow.vehicle_id == vehicle_id and flow.node_id == key[1] and flow.target_rsu_id == rid and flow.kind in {"state_migration", "final_forward"}:
                                self._cancel_flow(flow_id, "placement_changed")
                if previous is not None and target is not None:
                    self.handoffs[vehicle_id] += 1
                    self._emit("handoff", vehicle_id=vehicle_id, from_rsu_id=previous, to_rsu_id=target)
                    for predicted, made_at, eta_s, version, flow_ids in self._predictions.pop(vehicle_id, []):
                        exposed = 0.0
                        for flow_id in flow_ids:
                            if flow_id in self.flows:
                                exposed += self.flows[flow_id].sent_bytes
                            else:
                                exposed += next((float(event.get("size_bytes", event.get("sent_bytes", 0))) for event in reversed(self.ledger) if event.get("flow_id") == flow_id and event["event"] in {"flow_completed", "flow_cancelled"}), 0.0)
                        self._emit("prediction_resolved", vehicle_id=vehicle_id, predicted_rsu_id=predicted, actual_rsu_id=target, predicted_eta_s=eta_s, actual_eta_s=round(self.now_s - made_at, 9), eta_error_s=round(self.now_s - made_at - eta_s, 9), target_correct=predicted == target, wrong_target_prefetch_bytes_sent_at_handoff=round(exposed, 6) if predicted != target else 0.0, predictor_version=version)
                elif target is None:
                    self._emit("coverage_lost", vehicle_id=vehicle_id, from_rsu_id=previous)
                else:
                    self._emit("coverage_attached", vehicle_id=vehicle_id, to_rsu_id=target)
            self.attached[vehicle_id] = target
            snapshot[vehicle_id] = target
        self.association_timeline.append({"time_s": round(self.now_s, 9), "attached": snapshot})

    def _arrivals_and_deadlines(self) -> None:
        for vehicle_id in self.vehicle_ids:
            request = self.scenario.requests[vehicle_id]
            if vehicle_id not in self.arrived_requests and self.now_s + 1e-9 >= request.arrival_s:
                self.arrived_requests.add(vehicle_id)
                self._emit("request_arrived", vehicle_id=vehicle_id, workflow_id=self.scenario.workflows[vehicle_id].workflow_id, deadline_s=request.deadline_s)
            if vehicle_id in self.arrived_requests and vehicle_id not in self.delivered_at and vehicle_id not in self.deadline_missed and self.now_s > request.deadline_s + 1e-9:
                self.deadline_missed.add(vehicle_id)
                self._emit("deadline_missed", vehicle_id=vehicle_id, deadline_s=request.deadline_s, completed_nodes=len(self.completed[vehicle_id]))

    def _inflight(self, kind: str, vehicle_id: str, node_id: str, target: str, object_id: str) -> bool:
        return any(flow.kind == kind and flow.vehicle_id == vehicle_id and flow.node_id == node_id and flow.target_rsu_id == target and flow.object_id == object_id for flow in self.flows.values())

    def _advance_flows(self, previous_s: float) -> None:
        active = [flow for flow in self.flows.values() if flow.ready_s <= self.now_s + 1e-9]
        users: dict[str, int] = defaultdict(int)
        for flow in active:
            for resource in flow.resources:
                users[resource] += 1
        finished: list[str] = []
        for flow in active:
            share = min(self._capacity[resource] / users[resource] for resource in flow.resources)
            duration = self.now_s - max(previous_s, flow.ready_s)
            sent = min(flow.size_bytes - flow.sent_bytes, share * duration)
            if sent <= 0:
                continue
            flow.sent_bytes += sent
            self._emit("flow_progress", flow_id=flow.flow_id, kind=flow.kind, sent_bytes=round(sent, 6), cumulative_sent_bytes=round(flow.sent_bytes, 6))
            if flow.sent_bytes + 1e-6 >= flow.size_bytes:
                finished.append(flow.flow_id)
        for flow_id in finished:
            flow = self.flows.pop(flow_id)
            if flow.kind == "model_load":
                self.reserved[(flow.target_rsu_id, flow.object_id)] = True
                self._promote_loaded(flow.target_rsu_id)
            elif flow.kind in {"uplink", "state_migration", "final_forward"}:
                if flow.kind in {"state_migration", "final_forward"}:
                    receipt = self.outputs[(flow.vehicle_id, flow.object_id)]
                    if not self.state_codec.verify_receipt(receipt["payload"], receipt["digest"]):
                        raise ValueError("state receipt failed integrity verification")
                self.input_at.add((flow.vehicle_id, str(flow.node_id), flow.object_id, flow.target_rsu_id))
            elif flow.kind == "downlink":
                self.delivered_at[flow.vehicle_id] = self.now_s
                self._emit("workflow_delivered", vehicle_id=flow.vehicle_id, rsu_id=flow.target_rsu_id, on_time=self.now_s <= self.scenario.requests[flow.vehicle_id].deadline_s + 1e-9)
            self._emit("flow_completed", flow_id=flow_id, kind=flow.kind, vehicle_id=flow.vehicle_id, node_id=flow.node_id, object_id=flow.object_id, source_rsu_id=flow.source_rsu_id, target_rsu_id=flow.target_rsu_id, size_bytes=flow.size_bytes, duration_s=round(self.now_s - flow.requested_s, 9))

    def _advance_compute(self, previous_s: float) -> None:
        finished: list[tuple[str, str]] = []
        for key, run in self.running.items():
            amount = min(run.work_bytes - run.completed_work_bytes, self.rsu_by_id[run.rsu_id].compute_work_bytes_per_s * (self.now_s - previous_s))
            run.completed_work_bytes += amount
            self._emit("compute_progress", vehicle_id=run.vehicle_id, node_id=run.node_id, rsu_id=run.rsu_id, work_bytes=round(amount, 6))
            if run.completed_work_bytes + 1e-6 >= run.work_bytes:
                finished.append(key)
        for key in finished:
            run = self.running.pop(key)
            vehicle_id, node_id = key
            wf = self.scenario.workflows[vehicle_id]
            node = wf.node_map()[node_id]
            payload, digest = self.state_codec.export_receipt(vehicle_id, wf.workflow_id, node_id, run.rsu_id, self.now_s)
            size_bytes = max(1, round(node.output_size * self.scenario.alibaba_unit_bytes))
            self.outputs[key] = {"source_rsu_id": run.rsu_id, "size_bytes": size_bytes, "payload": payload, "digest": digest, "completed_s": self.now_s}
            self.completed[vehicle_id].add(node_id)
            self._emit("compute_completed", vehicle_id=vehicle_id, node_id=node_id, rsu_id=run.rsu_id, output_digest=digest, declared_output_bytes=size_bytes, metadata_payload_bytes=len(payload), duration_s=round(self.now_s - run.started_s, 9))

    def observe(self) -> dict[str, Any]:
        """Only current/past state is exposed; no future trace samples."""
        vehicles: dict[str, dict[str, Any]] = {}
        for vehicle_id in self.vehicle_ids:
            point = self.scenario.traces[vehicle_id][self.frame_index]
            previous = self.scenario.traces[vehicle_id][max(0, self.frame_index - 1)]
            vehicles[vehicle_id] = {
                "x_m": point.x_m,
                "y_m": point.y_m,
                "speed_mps": point.speed_mps,
                "velocity_x_mps": (point.x_m - previous.x_m) / self.scenario.tick_s if self.frame_index else 0.0,
                "velocity_y_mps": (point.y_m - previous.y_m) / self.scenario.tick_s if self.frame_index else 0.0,
                "road_key": point.road_key,
                "attached_rsu_id": self.attached[vehicle_id],
                "request_arrived": vehicle_id in self.arrived_requests,
                "deadline_s": self.scenario.requests[vehicle_id].deadline_s,
                "completed_node_ids": sorted(self.completed[vehicle_id]),
                "node_order": list(self.scenario.workflows[vehicle_id].execution_order),
            }
        return {
            "time_s": self.now_s,
            "vehicles": vehicles,
            "rsus": {rid: {"resident_object_ids": sorted(self.residents[rid]), "reserved_object_ids": sorted(oid for target, oid in self.reserved if target == rid), "cache_used_bytes": self._used_bytes(rid), "compute_busy_slots": sum(run.rsu_id == rid for run in self.running.values()), "compute_queue_length": sum(self.placement.get(key) == rid for key in self.queue)} for rid in self.rsu_by_id},
        }

    def action_mask(self, vehicle_id: str, node_id: str) -> dict[str, list[str]]:
        if vehicle_id not in self.arrived_requests or node_id not in self.scenario.workflows[vehicle_id].node_map() or node_id in self.completed[vehicle_id]:
            return {"prepare": [], "place": []}
        attached = self.attached[vehicle_id]
        return {"prepare": sorted(self.rsu_by_id), "place": [attached] if attached is not None and (vehicle_id, node_id) not in self.running else []}

    def apply_action(self, action: ControllerAction) -> bool:
        mask = self.action_mask(action.vehicle_id, action.node_id)
        if action.kind not in mask or action.target_rsu_id not in mask[action.kind]:
            self._emit("action_rejected", kind=action.kind, vehicle_id=action.vehicle_id, node_id=action.node_id, target_rsu_id=action.target_rsu_id, reason="illegal_or_unavailable")
            return False
        if action.kind == "prepare":
            before = set(self.flows)
            accepted = self._ensure_models(action.target_rsu_id, action.vehicle_id, action.node_id)
            new_flow_ids = tuple(sorted(set(self.flows) - before))
            if action.prediction_eta_s is not None and action.prediction_version is not None:
                if action.prediction_eta_s < 0:
                    raise ValueError("prediction ETA must be non-negative")
                self._predictions[action.vehicle_id].append((action.target_rsu_id, self.now_s, action.prediction_eta_s, action.prediction_version, new_flow_ids))
                self._emit("prediction_made", vehicle_id=action.vehicle_id, predicted_rsu_id=action.target_rsu_id, eta_s=action.prediction_eta_s, predictor_version=action.prediction_version, new_model_flow_ids=list(new_flow_ids), new_model_transfer_bytes_requested=sum(self.flows[flow_id].size_bytes for flow_id in new_flow_ids))
        else:
            self.placement[(action.vehicle_id, action.node_id)] = action.target_rsu_id
            accepted = True
        self._emit("action_applied", kind=action.kind, vehicle_id=action.vehicle_id, node_id=action.node_id, target_rsu_id=action.target_rsu_id, accepted=accepted)
        return accepted

    def _schedule(self) -> None:
        observation = self.observe()
        for vehicle_id in self.vehicle_ids:
            if vehicle_id not in self.arrived_requests or vehicle_id in self.delivered_at:
                continue
            wf = self.scenario.workflows[vehicle_id]
            for node_id in wf.execution_order:
                key = (vehicle_id, node_id)
                if node_id in self.completed[vehicle_id] or key in self.running:
                    continue
                node = wf.node_map()[node_id]
                if not all(predecessor in self.completed[vehicle_id] for predecessor in node.predecessors):
                    continue
                rid = self.placement.get(key)
                if rid is None:
                    rid = self.placement_policy.place(observation, vehicle_id, node_id)
                    if rid is None:
                        continue
                    if rid != self.attached[vehicle_id]:
                        raise ValueError("placement plugin selected RSU without current radio attachment")
                    self.placement[key] = rid
                    self._emit("node_placed", vehicle_id=vehicle_id, node_id=node_id, rsu_id=rid)
                self._ensure_models(rid, vehicle_id, node_id)
                if not node.predecessors:
                    input_key = (vehicle_id, node_id, "request_input", rid)
                    if input_key not in self.input_at and not self._inflight("uplink", vehicle_id, node_id, rid, "request_input"):
                        self._new_flow("uplink", vehicle_id, node_id, "request_input", rid, max(1, round(node.input_size * self.scenario.alibaba_unit_bytes)), channel="radio")
                else:
                    for predecessor in node.predecessors:
                        output = self.outputs[(vehicle_id, predecessor)]
                        input_key = (vehicle_id, node_id, predecessor, rid)
                        if output["source_rsu_id"] == rid:
                            self.input_at.add(input_key)
                        elif input_key not in self.input_at and not self._inflight("state_migration", vehicle_id, node_id, rid, predecessor):
                            self._new_flow("state_migration", vehicle_id, node_id, predecessor, rid, output["size_bytes"], source=output["source_rsu_id"])
                ready_data = all((vehicle_id, node_id, predecessor, rid) in self.input_at for predecessor in node.predecessors) if node.predecessors else (vehicle_id, node_id, "request_input", rid) in self.input_at
                ready_models = all(oid in self.residents[rid] for oid in self._required_objects(vehicle_id, node_id))
                if ready_data and ready_models and key not in self.queue:
                    self.queue[key] = self.now_s
                    self._emit("node_queued", vehicle_id=vehicle_id, node_id=node_id, rsu_id=rid)
        for rid, rsu in self.rsu_by_id.items():
            free_slots = rsu.compute_slots - sum(run.rsu_id == rid for run in self.running.values())
            candidates = sorted((queued_s, vehicle_id, node_id) for (vehicle_id, node_id), queued_s in self.queue.items() if self.placement.get((vehicle_id, node_id)) == rid)
            for queued_s, vehicle_id, node_id in candidates[:free_slots]:
                key = (vehicle_id, node_id)
                self.queue.pop(key)
                node = self.scenario.workflows[vehicle_id].node_map()[node_id]
                work_bytes = max(1, round((node.input_size + node.output_size) * self.scenario.alibaba_unit_bytes))
                self.running[key] = ComputeRun(vehicle_id, node_id, rid, work_bytes, 0.0, self.now_s, queued_s)
                for object_id in self._required_objects(vehicle_id, node_id):
                    self.last_used[(rid, object_id)] = self.now_s
                self._emit("compute_started", vehicle_id=vehicle_id, node_id=node_id, rsu_id=rid, work_bytes=work_bytes, queue_wait_s=round(self.now_s - queued_s, 9))
        self._schedule_delivery()

    def _schedule_delivery(self) -> None:
        for vehicle_id in self.vehicle_ids:
            if vehicle_id in self.delivered_at or len(self.completed[vehicle_id]) != len(self.scenario.workflows[vehicle_id].nodes):
                continue
            rid = self.attached[vehicle_id]
            if rid is None:
                continue
            sinks = [node for node in self.scenario.workflows[vehicle_id].nodes if not node.successors]
            for node in sinks:
                output = self.outputs[(vehicle_id, node.node_id)]
                key = (vehicle_id, "final", node.node_id, rid)
                if output["source_rsu_id"] == rid:
                    self.input_at.add(key)
                elif key not in self.input_at and not self._inflight("final_forward", vehicle_id, "final", rid, node.node_id):
                    self._new_flow("final_forward", vehicle_id, "final", node.node_id, rid, output["size_bytes"], source=output["source_rsu_id"])
            if all((vehicle_id, "final", node.node_id, rid) in self.input_at for node in sinks):
                if not any(flow.kind == "downlink" and flow.vehicle_id == vehicle_id for flow in self.flows.values()):
                    size_bytes = sum(self.outputs[(vehicle_id, node.node_id)]["size_bytes"] for node in sinks)
                    self._new_flow("downlink", vehicle_id, "final", "result", rid, size_bytes, channel="radio")

    def _check_invariants(self) -> None:
        for rid, rsu in self.rsu_by_id.items():
            if self._used_bytes(rid) > rsu.cache_capacity_bytes:
                raise AssertionError(f"cache capacity violated at {rid}")
            if sum(run.rsu_id == rid for run in self.running.values()) > rsu.compute_slots:
                raise AssertionError(f"compute slots violated at {rid}")
            for oid in self.residents[rid]:
                if any(dep not in self.residents[rid] for dep in self.scenario.catalog.get_typed_object(oid).dependency_ids):
                    raise AssertionError(f"orphan resident at {rid}: {oid}")

    def step(self, actions: tuple[ControllerAction, ...] = ()) -> bool:
        if self.frame_index >= len(self.scenario.traces[self.vehicle_ids[0]]) - 1:
            return False
        previous_s = self.now_s
        self.frame_index += 1
        # Flows/compute consume the interval (previous_s, now_s] under the
        # previous association. A handoff at now_s may cancel only residual
        # bytes; it cannot retroactively erase work done in that interval.
        self._advance_flows(previous_s)
        self._advance_compute(previous_s)
        self._associate()
        self._arrivals_and_deadlines()
        for action in actions:
            self.apply_action(action)
        self._schedule()
        self._check_invariants()
        return True

    def run(self, controller: ControllerPolicy | None = None) -> dict[str, Any]:
        final_frame = len(self.scenario.traces[self.vehicle_ids[0]]) - 1
        while self.frame_index < final_frame:
            if controller is not None:
                for action in controller.actions(self.observe()):
                    self.apply_action(action)
            self.step()
        if not self._finalized:
            for vehicle_id in self.vehicle_ids:
                if vehicle_id in self.arrived_requests and vehicle_id not in self.delivered_at:
                    self._emit("request_unfinished", vehicle_id=vehicle_id, reason="horizon_exhausted", completed_nodes=len(self.completed[vehicle_id]))
            self._finalized = True
        return self.summary()

    def summary(self) -> dict[str, Any]:
        completed_flows = [event for event in self.ledger if event["event"] == "flow_completed"]
        flow_owners = {event["flow_id"]: (event["vehicle_id"], event["kind"]) for event in self.ledger if event["event"] == "flow_requested"}
        sent_bytes_by_vehicle: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        sent_bytes_by_kind: dict[str, float] = defaultdict(float)
        for event in self.ledger:
            if event["event"] == "flow_progress":
                vehicle_id, kind = flow_owners[event["flow_id"]]
                sent_bytes_by_vehicle[vehicle_id][kind] += float(event["sent_bytes"])
                sent_bytes_by_kind[kind] += float(event["sent_bytes"])
        completed_bytes: dict[str, int] = defaultdict(int)
        bytes_by_vehicle: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event in completed_flows:
            completed_bytes[event["kind"]] += int(event["size_bytes"])
            bytes_by_vehicle[event["vehicle_id"]][event["kind"]] += int(event["size_bytes"])
        per_vehicle: dict[str, dict[str, Any]] = {}
        for vehicle_id in self.vehicle_ids:
            delivered_s = self.delivered_at.get(vehicle_id)
            deadline = self.scenario.requests[vehicle_id].deadline_s
            status = "on_time" if delivered_s is not None and delivered_s <= deadline + 1e-9 else "late" if delivered_s is not None else "unfinished"
            per_vehicle[vehicle_id] = {"workflow_id": self.scenario.workflows[vehicle_id].workflow_id, "status": status, "arrival_s": self.scenario.requests[vehicle_id].arrival_s, "deadline_s": deadline, "delivered_s": delivered_s, "end_to_end_latency_s": round(delivered_s - self.scenario.requests[vehicle_id].arrival_s, 9) if delivered_s is not None else None, "completed_nodes": len(self.completed[vehicle_id]), "total_nodes": len(self.scenario.workflows[vehicle_id].nodes), "handoffs": self.handoffs[vehicle_id], "sent_transfer_bytes_by_kind": {kind: round(value, 6) for kind, value in sorted(sent_bytes_by_vehicle[vehicle_id].items())}, "completed_transfer_bytes_by_kind": dict(sorted(bytes_by_vehicle[vehicle_id].items())), "failure_reason": "deadline_missed" if status == "late" else "horizon_exhausted" if status == "unfinished" else None}
        return {
            "contract_version": self.contract_version,
            "scenario_id": self.scenario.scenario_id,
            "generated_requests": len(self.arrived_requests),
            "simultaneous_requests_at_start": sum(event["event"] == "request_arrived" and event["time_s"] == 0.0 for event in self.ledger),
            "rsu_count": len(self.rsu_by_id),
            "vehicle_count": len(self.vehicle_ids),
            "frame_count": self.frame_index + 1,
            "horizon_s": self.now_s,
            "completed_nodes": sum(len(nodes) for nodes in self.completed.values()),
            "total_nodes": sum(len(wf.nodes) for wf in self.scenario.workflows.values()),
            "delivered_on_time": sum(x["status"] == "on_time" for x in per_vehicle.values()),
            "delivered_late": sum(x["status"] == "late" for x in per_vehicle.values()),
            "unfinished": sum(x["status"] == "unfinished" for x in per_vehicle.values()),
            "deadline_missed": len(self.deadline_missed),
            "handoffs": sum(self.handoffs.values()),
            "state_migrations": sum(event["kind"] == "state_migration" for event in completed_flows),
            "resolved_predictions": sum(event["event"] == "prediction_resolved" for event in self.ledger),
            "wrong_predictions": sum(event["event"] == "prediction_resolved" and not event["target_correct"] for event in self.ledger),
            "wrong_target_prefetch_bytes_sent_at_handoff": round(sum(float(event["wrong_target_prefetch_bytes_sent_at_handoff"]) for event in self.ledger if event["event"] == "prediction_resolved"), 6),
            "completed_transfer_bytes_by_kind": dict(sorted(completed_bytes.items())),
            "sent_transfer_bytes_by_kind": {kind: round(value, 6) for kind, value in sorted(sent_bytes_by_kind.items())},
            "model_load_attribution": "triggering_vehicle_only; shared resident may serve other vehicles",
            "cancelled_transfer_bytes": round(sum(float(event["sent_bytes"]) for event in self.ledger if event["event"] == "flow_cancelled"), 6),
            "compute_queue_wait_s": round(sum(float(event["queue_wait_s"]) for event in self.ledger if event["event"] == "compute_started"), 9),
            "cache_used_bytes": {rid: self._used_bytes(rid) for rid in self.rsu_by_id},
            "per_vehicle": per_vehicle,
            "ledger_event_count": len(self.ledger),
            "plugin_versions": {"access": getattr(self.access_policy, "version", "custom"), "placement": getattr(self.placement_policy, "version", "custom"), "state_codec": getattr(self.state_codec, "version", "custom")},
            "model_state_real": bool(getattr(self.state_codec, "real_model_state", False)),
        }

    def ledger_digest(self) -> str:
        blob = json.dumps(self.ledger, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode()).hexdigest()
