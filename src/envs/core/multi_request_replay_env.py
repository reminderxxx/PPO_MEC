"""Trace-driven, shared-resource multi-request VEC diagnostic environment.

This contract intentionally models service in trace frames. It does not execute
models, emulate a radio, or claim a measured vehicle-to-workflow joint trace.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from math import dist
from typing import Any

from src.data.mobility.rsu_mapper import RSUMapper
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState


@dataclass(frozen=True)
class ReplayCosts:
    compute_slots_per_rsu: int = 1
    adapter_slots_per_rsu: int = 2
    link_slots_per_rsu: int = 1
    adapter_fetch_frames: int = 1
    handoff_state_frames: int = 1

    def __post_init__(self) -> None:
        if min(
            self.compute_slots_per_rsu,
            self.adapter_slots_per_rsu,
            self.link_slots_per_rsu,
        ) < 1:
            raise ValueError("RSU slot capacities must be positive")
        if min(self.adapter_fetch_frames, self.handoff_state_frames) < 0:
            raise ValueError("transfer costs must be nonnegative")


def select_continuous_window(
    frames: list[dict[str, Any]], *, vehicle_count: int = 8, window_length: int = 120
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Return the first same-segment window with persistent, distinct vehicles."""
    if vehicle_count < 1 or window_length < 2:
        raise ValueError("vehicle_count must be positive and window_length >= 2")
    for start in range(len(frames) - window_length + 1):
        window = frames[start : start + window_length]
        segment = window[0].get("source_segment_id")
        if any(frame.get("source_segment_id") != segment for frame in window):
            continue
        common = {vehicle_id for vehicle_id, count in Counter(_vehicle_ids(window[0])).items() if count == 1}
        for frame in window[1:]:
            common.intersection_update(
                vehicle_id for vehicle_id, count in Counter(_vehicle_ids(frame)).items() if count == 1
            )
            if len(common) < vehicle_count:
                break
        if len(common) >= vehicle_count:
            selected_ids = sorted(common)[:vehicle_count]
            selected = [
                {**frame, "vehicles": [
                    vehicle for vehicle in frame["vehicles"]
                    if vehicle.vehicle_id in selected_ids
                ]}
                for frame in window
            ]
            return selected, selected_ids, start
    raise RuntimeError(
        f"No {window_length}-frame same-segment window has {vehicle_count} persistent vehicles"
    )


def _vehicle_ids(frame: dict[str, Any]) -> list[str]:
    return [str(vehicle.vehicle_id) for vehicle in frame["vehicles"]]


class MultiRequestReplayEnv:
    """Advance one DAG per vehicle against shared RSU compute, link and cache."""

    def __init__(
        self,
        *,
        frames: list[dict[str, Any]],
        vehicle_workflows: dict[str, WorkflowGraphState],
        rsu_states: list[RSUState],
        costs: ReplayCosts = ReplayCosts(),
    ) -> None:
        if len(frames) < 2 or not vehicle_workflows or not rsu_states:
            raise ValueError("frames, vehicle_workflows and rsu_states must be nonempty")
        if len({rsu.rsu_id for rsu in rsu_states}) != len(rsu_states):
            raise ValueError("RSU IDs must be unique")
        if len({workflow.workflow_id for workflow in vehicle_workflows.values()}) != len(vehicle_workflows):
            raise ValueError("each vehicle must have a distinct workflow")
        assigned = set(vehicle_workflows)
        for frame in frames:
            frame_ids = _vehicle_ids(frame)
            if len(frame_ids) != len(set(frame_ids)) or set(frame_ids) != assigned:
                raise ValueError("every frame must contain exactly the assigned vehicles once")
        if len({frame.get("source_segment_id") for frame in frames}) != 1:
            raise ValueError("frames must stay within one source segment")
        self.frames = frames
        self.workflow_templates = deepcopy(vehicle_workflows)
        self.rsus = deepcopy(rsu_states)
        self.costs = costs
        self.mapper = RSUMapper(self.rsus)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.frame_index = 0
        self.workflows = deepcopy(self.workflow_templates)
        self.cache: dict[str, OrderedDict[str, None]] = {
            rsu.rsu_id: OrderedDict() for rsu in self.rsus
        }
        self.loading: dict[str, dict[str, int]] = {rsu.rsu_id: {} for rsu in self.rsus}
        self.last_service_rsu: dict[str, str | None] = {vehicle_id: None for vehicle_id in self.workflows}
        self.handoff_wait: dict[str, int] = {vehicle_id: 0 for vehicle_id in self.workflows}
        self.handoff_target: dict[str, str | None] = {vehicle_id: None for vehicle_id in self.workflows}
        self.events: list[dict[str, Any]] = []
        return self.observe()

    def observe(self) -> dict[str, Any]:
        vehicles = self.frames[self.frame_index]["vehicles"]
        associations = self.mapper.associate(vehicles)
        associated_counts = Counter(item for item in associations.values() if item is not None)
        return {
            "frame_index": self.frame_index,
            "time_index": int(self.frames[self.frame_index]["time_index"]),
            "vehicles": {vehicle.vehicle_id: {
                "associated_rsu_id": associations[vehicle.vehicle_id],
                "covered_rsu_ids": self._covered_rsus(vehicle),
                "workflow_id": self.workflows[vehicle.vehicle_id].workflow_id,
                "current_node_id": self.workflows[vehicle.vehicle_id].current_node_id,
                "completed": self.workflows[vehicle.vehicle_id].is_completed,
                "position_x": float(vehicle.position_x),
                "position_y": float(vehicle.position_y),
                "speed": float(vehicle.speed),
                "required_adapter": (
                    self.workflows[vehicle.vehicle_id].current_node().required_adapter
                    if self.workflows[vehicle.vehicle_id].current_node() is not None else None
                ),
                "remaining_nodes": len(self.workflows[vehicle.vehicle_id].nodes)
                - len(self.workflows[vehicle.vehicle_id].completed_node_ids),
                "last_service_rsu_id": self.last_service_rsu[vehicle.vehicle_id],
            } for vehicle in vehicles},
            "rsus": {rsu.rsu_id: {
                "cached_adapter_ids": list(self.cache[rsu.rsu_id]),
                "loading_adapter_ids": list(self.loading[rsu.rsu_id]),
                "compute_slots": self.costs.compute_slots_per_rsu,
                "link_slots": self.costs.link_slots_per_rsu,
                "adapter_slots": self.costs.adapter_slots_per_rsu,
                "associated_vehicle_count": associated_counts[rsu.rsu_id],
                "position_x": float(rsu.position_x),
                "position_y": float(rsu.position_y),
                "coverage_radius": float(rsu.coverage_radius),
            } for rsu in self.rsus},
        }

    def step(self, targets: dict[str, str] | None = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.frame_index >= len(self.frames):
            raise RuntimeError("episode has ended; call reset")
        targets = targets or {}
        if set(targets) - set(self.workflows):
            raise ValueError("target action contains an unknown vehicle")
        frame = self.frames[self.frame_index]
        vehicles: dict[str, VehicleState] = {item.vehicle_id: item for item in frame["vehicles"]}
        associations = self.mapper.associate(list(vehicles.values()))
        for vehicle_id, target in targets.items():
            if target not in self._covered_rsus(vehicles[vehicle_id]):
                raise ValueError(f"{vehicle_id} targets uncovered RSU {target}")
        compute_left = {rsu.rsu_id: self.costs.compute_slots_per_rsu for rsu in self.rsus}
        link_left = {
            rsu.rsu_id: self.costs.link_slots_per_rsu - len(self.loading[rsu.rsu_id])
            for rsu in self.rsus
        }
        completed_now = 0
        frame_events = []

        # Rotating priority avoids a permanent advantage for the first vehicle ID.
        vehicle_ids = sorted(self.workflows)
        shift = self.frame_index % len(vehicle_ids)
        for vehicle_id in vehicle_ids[shift:] + vehicle_ids[:shift]:
            workflow = self.workflows[vehicle_id]
            if workflow.is_completed:
                continue
            vehicle = vehicles[vehicle_id]
            associated = associations[vehicle_id]
            target = targets.get(vehicle_id, associated)
            node = workflow.current_node()
            if node is None:
                raise RuntimeError(f"active workflow {workflow.workflow_id} has no current node")
            status = "no_coverage"
            if target is not None:
                if vehicle.base_model_id != node.required_base_model:
                    status = "base_model_mismatch"
                elif self.last_service_rsu[vehicle_id] not in (None, target):
                    if self.handoff_target[vehicle_id] != target:
                        self.handoff_target[vehicle_id] = target
                        self.handoff_wait[vehicle_id] = self.costs.handoff_state_frames
                    if self.handoff_wait[vehicle_id] and link_left[target] > 0:
                        link_left[target] -= 1
                        self.handoff_wait[vehicle_id] -= 1
                        status = "handoff_state_transfer"
                    elif self.handoff_wait[vehicle_id]:
                        status = "link_queue"
                    else:
                        status = self._serve(vehicle_id, target, node.required_adapter, compute_left, link_left)
                else:
                    status = self._serve(vehicle_id, target, node.required_adapter, compute_left, link_left)
            if status == "node_completed":
                workflow.mark_current_completed()
                self.last_service_rsu[vehicle_id] = target
                self.handoff_target[vehicle_id] = None
                completed_now += 1
            event = {
                "frame_index": self.frame_index,
                "time_index": int(frame["time_index"]),
                "vehicle_id": vehicle_id,
                "request_id": f"{vehicle_id}/{workflow.workflow_id}",
                "workflow_id": workflow.workflow_id,
                "node_id": node.node_id,
                "associated_rsu_id": associated,
                "target_rsu_id": target,
                "adapter_id": node.required_adapter,
                "status": status,
                "workflow_completed": workflow.is_completed,
            }
            frame_events.append(event)
        self.events.extend(frame_events)
        self._finish_loading()
        self.frame_index += 1
        done = self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values())
        observation = self.observe() if not done else {
            "frame_index": self.frame_index,
            "time_index": int(frame["time_index"]),
            "vehicles": {},
            "rsus": {},
        }
        reward = float(completed_now) - 0.01 * sum(not w.is_completed for w in self.workflows.values())
        return observation, reward, done, {"events": frame_events, "completed_nodes": completed_now}

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.frame_index,
            "request_count": len(self.workflows),
            "completed_requests": sum(w.is_completed for w in self.workflows.values()),
            "completed_nodes": sum(len(w.completed_node_ids) for w in self.workflows.values()),
            "per_request": [{
                "vehicle_id": vehicle_id,
                "request_id": f"{vehicle_id}/{workflow.workflow_id}",
                "workflow_id": workflow.workflow_id,
                "completed_nodes": len(workflow.completed_node_ids),
                "total_nodes": len(workflow.nodes),
                "completed": workflow.is_completed,
                "completion_frame_index": next((
                    event["frame_index"] for event in self.events
                    if event["vehicle_id"] == vehicle_id and event["workflow_completed"]
                ), None),
            } for vehicle_id, workflow in sorted(self.workflows.items())],
            "status_counts": {
                status: sum(event["status"] == status for event in self.events)
                for status in sorted({event["status"] for event in self.events})
            },
        }

    def _covered_rsus(self, vehicle: VehicleState) -> list[str]:
        return [rsu.rsu_id for rsu in self.rsus if dist(
            (vehicle.position_x, vehicle.position_y),
            (rsu.position_x, rsu.position_y),
        ) <= rsu.coverage_radius]

    def _serve(
        self,
        vehicle_id: str,
        target: str,
        adapter_id: str,
        compute_left: dict[str, int],
        link_left: dict[str, int],
    ) -> str:
        cache = self.cache[target]
        loading = self.loading[target]
        if adapter_id in loading:
            return "adapter_loading"
        if adapter_id not in cache:
            if len(cache) + len(loading) >= self.costs.adapter_slots_per_rsu and not cache:
                return "cache_queue"
            if self.costs.adapter_fetch_frames > 0:
                if link_left[target] < 1:
                    return "link_queue"
                link_left[target] -= 1
                while len(cache) + len(loading) >= self.costs.adapter_slots_per_rsu:
                    cache.popitem(last=False)
                loading[adapter_id] = self.costs.adapter_fetch_frames
                return "adapter_fetch"
            while len(cache) + len(loading) >= self.costs.adapter_slots_per_rsu:
                cache.popitem(last=False)
            cache[adapter_id] = None
        if compute_left[target] < 1:
            return "compute_queue"
        compute_left[target] -= 1
        cache.move_to_end(adapter_id)
        return "node_completed"

    def _finish_loading(self) -> None:
        for rsu_id, loading in self.loading.items():
            for adapter_id in list(loading):
                loading[adapter_id] -= 1
                if loading[adapter_id] <= 0:
                    del loading[adapter_id]
                    self.cache[rsu_id][adapter_id] = None
