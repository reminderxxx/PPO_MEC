"""Versioned, pluggable physical-unit contract for the 4-RSU/8-vehicle replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.specs import WorkflowGraphState


CONTRACT_VERSION = "vec_competition_v2"
DECIMAL_MB = 1_000_000


@dataclass(frozen=True)
class TraceSample:
    time_s: float
    source_time_ms: int
    x_m: float
    y_m: float
    speed_mps: float
    road_key: str
    direction: str


@dataclass(frozen=True)
class RSUConfig:
    rsu_id: str
    x_m: float
    y_m: float
    radius_m: float
    cache_capacity_bytes: int
    compute_slots: int
    compute_work_bytes_per_s: float
    radio_bytes_per_s: float
    backhaul_port_bytes_per_s: float


@dataclass(frozen=True)
class BackhaulLink:
    source: str
    target: str
    bytes_per_s: float
    latency_s: float

    @property
    def key(self) -> str:
        return "link:" + ":".join(sorted((self.source, self.target)))


@dataclass(frozen=True)
class RequestSpec:
    vehicle_id: str
    arrival_s: float
    deadline_s: float


@dataclass(frozen=True)
class ControllerAction:
    kind: str  # prepare | place
    vehicle_id: str
    node_id: str
    target_rsu_id: str
    prediction_eta_s: float | None = None
    prediction_version: str | None = None


@dataclass(frozen=True)
class FrozenScenario:
    scenario_id: str
    source_segment_id: str
    traces: dict[str, tuple[TraceSample, ...]]
    workflows: dict[str, WorkflowGraphState]
    requests: dict[str, RequestSpec]
    rsus: tuple[RSUConfig, ...]
    backhaul_links: tuple[BackhaulLink, ...]
    cloud_port_bytes_per_s: float
    allowed_road_edges: frozenset[tuple[str, str]]
    catalog: AdapterCatalog
    alibaba_unit_bytes: int
    tick_s: float

    def validate(self) -> None:
        vehicle_ids = set(self.traces)
        if len(vehicle_ids) != 8 or len(self.rsus) != 4 or vehicle_ids != set(self.workflows) or vehicle_ids != set(self.requests):
            raise ValueError("contract requires exactly four RSUs and eight distinct vehicle requests/workflows")
        rsu_ids = {r.rsu_id for r in self.rsus}
        if len(rsu_ids) != 4 or not self.source_segment_id or self.alibaba_unit_bytes <= 0 or self.tick_s <= 0:
            raise ValueError("invalid scenario identity or unit scale")
        for rsu in self.rsus:
            if min(rsu.radius_m, rsu.cache_capacity_bytes, rsu.compute_slots, rsu.compute_work_bytes_per_s, rsu.radio_bytes_per_s, rsu.backhaul_port_bytes_per_s) <= 0:
                raise ValueError(f"invalid RSU resource: {rsu.rsu_id}")
        if self.cloud_port_bytes_per_s <= 0:
            raise ValueError("cloud port capacity must be positive")
        link_keys = set()
        for link in self.backhaul_links:
            if link.source not in rsu_ids or link.target not in rsu_ids or link.source == link.target or link.key in link_keys or link.bytes_per_s <= 0 or link.latency_s < 0:
                raise ValueError("invalid backhaul topology")
            link_keys.add(link.key)
        reached = {next(iter(rsu_ids))}
        while True:
            extended = reached | {link.target for link in self.backhaul_links if link.source in reached} | {link.source for link in self.backhaul_links if link.target in reached}
            if extended == reached:
                break
            reached = extended
        if reached != rsu_ids:
            raise ValueError("backhaul graph must connect all four RSUs")
        signatures: set[tuple[tuple[int, float, float], ...]] = set()
        common_times: tuple[float, ...] | None = None
        for vehicle_id, trace in self.traces.items():
            if not vehicle_id.startswith(self.source_segment_id + ":") or len(trace) < 2:
                raise ValueError(f"invalid source identity or short trace: {vehicle_id}")
            times = tuple(sample.time_s for sample in trace)
            if common_times is None:
                common_times = times
            if times != common_times or not math.isclose(times[0], 0.0, abs_tol=1e-9) or any(not math.isclose(b - a, self.tick_s, abs_tol=1e-8) for a, b in zip(times, times[1:])):
                raise ValueError(f"asynchronous/non-contiguous trace: {vehicle_id}")
            signature = tuple((p.source_time_ms, p.x_m, p.y_m) for p in trace)
            if signature in signatures:
                raise ValueError("cloned vehicle trajectory")
            signatures.add(signature)
            if len({p.direction for p in trace}) != 1:
                raise ValueError(f"direction switch in frozen corridor trace: {vehicle_id}")
            for a, b in zip(trace, trace[1:]):
                if b.source_time_ms - a.source_time_ms != round(self.tick_s * 1000):
                    raise ValueError(f"raw source time gap: {vehicle_id}")
                if not all(math.isfinite(v) for v in (a.x_m, a.y_m, a.speed_mps, b.x_m, b.y_m, b.speed_mps)):
                    raise ValueError(f"non-finite trace: {vehicle_id}")
                distance = math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
                if distance > max(8.0, 2.5 * max(a.speed_mps, b.speed_mps) * self.tick_s):
                    raise ValueError(f"teleportation in trace: {vehicle_id}")
                if a.road_key != b.road_key and (a.road_key, b.road_key) not in self.allowed_road_edges:
                    raise ValueError(f"illegal road transition: {vehicle_id} {a.road_key} -> {b.road_key}")
            request = self.requests[vehicle_id]
            if request.vehicle_id != vehicle_id or request.arrival_s < 0 or request.deadline_s < request.arrival_s or request.deadline_s > trace[-1].time_s:
                raise ValueError(f"invalid request timing: {vehicle_id}")
            workflow = self.workflows[vehicle_id]
            nodes = workflow.node_map()
            if len(nodes) != len(workflow.nodes) or len(nodes) != len(workflow.execution_order) or set(nodes) != set(workflow.execution_order):
                raise ValueError(f"invalid DAG identity: {vehicle_id}")
            rank = {node_id: i for i, node_id in enumerate(workflow.execution_order)}
            for node in nodes.values():
                if node.required_adapter not in self.catalog.compatibility_map.get(node.required_base_model, []):
                    raise ValueError(f"incompatible model bundle: {vehicle_id}/{node.node_id}")
                for predecessor in node.predecessors:
                    if predecessor not in nodes or rank[predecessor] >= rank[node.node_id] or node.node_id not in nodes[predecessor].successors:
                        raise ValueError(f"invalid DAG dependency: {vehicle_id}/{node.node_id}")


class AccessPolicy(Protocol):
    def associate(self, point: TraceSample, current_rsu_id: str | None, rsus: tuple[RSUConfig, ...]) -> str | None: ...


class PlacementPolicy(Protocol):
    def place(self, observation: dict[str, Any], vehicle_id: str, node_id: str) -> str | None: ...


class StateCodec(Protocol):
    def export_receipt(self, vehicle_id: str, workflow_id: str, node_id: str, rsu_id: str, completed_s: float) -> tuple[bytes, str]: ...

    def verify_receipt(self, payload: bytes, expected_digest: str) -> bool: ...


class ControllerPolicy(Protocol):
    version: str

    def actions(self, observation: dict[str, Any]) -> tuple[ControllerAction, ...]: ...


class NearestCoveredAccess:
    version = "nearest_covered_v1"

    def associate(self, point: TraceSample, current_rsu_id: str | None, rsus: tuple[RSUConfig, ...]) -> str | None:
        covered = [(distance, rsu.rsu_id) for rsu in rsus if (distance := math.hypot(point.x_m - rsu.x_m, point.y_m - rsu.y_m)) <= rsu.radius_m]
        return min(covered)[1] if covered else None


class AttachedFifoPlacement:
    version = "attached_fifo_v1"

    def place(self, observation: dict[str, Any], vehicle_id: str, node_id: str) -> str | None:
        return observation["vehicles"][vehicle_id]["attached_rsu_id"]


class MetadataReceiptCodec:
    """Serializable metadata witness; never represents a real model/KV state."""

    version = "metadata_receipt_v1"
    real_model_state = False

    def export_receipt(self, vehicle_id: str, workflow_id: str, node_id: str, rsu_id: str, completed_s: float) -> tuple[bytes, str]:
        payload = json.dumps({"schema": self.version, "vehicle_id": vehicle_id, "workflow_id": workflow_id, "node_id": node_id, "source_rsu_id": rsu_id, "completed_s": completed_s}, sort_keys=True, separators=(",", ":")).encode()
        return payload, hashlib.sha256(payload).hexdigest()

    def verify_receipt(self, payload: bytes, expected_digest: str) -> bool:
        return hashlib.sha256(payload).hexdigest() == expected_digest
