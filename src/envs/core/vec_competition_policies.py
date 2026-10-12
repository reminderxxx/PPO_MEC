"""Causal controller plugins for the frozen multi-vehicle action contract."""

from __future__ import annotations

import math
from typing import Any

from src.envs.core.vec_competition_contract import ControllerAction, RSUConfig


class NoPrefetchController:
    version = "no_prefetch_v1"

    def actions(self, observation: dict[str, Any]) -> tuple[ControllerAction, ...]:
        return ()


class ConstantVelocityPrefetchController:
    """Predict first future covered RSU from current/past positions only."""

    version = "constant_velocity_prefetch_v1"

    def __init__(self, rsus: tuple[RSUConfig, ...], *, lookahead_s: float = 2.0, projection_step_s: float = 0.1) -> None:
        if lookahead_s <= 0 or projection_step_s <= 0:
            raise ValueError("positive prediction horizon/step required")
        self.rsus = rsus
        self.lookahead_s = lookahead_s
        self.projection_step_s = projection_step_s
        self._predicted_for_attachment: set[tuple[str, str, str]] = set()
        self._last_attachment: dict[str, str | None] = {}

    def _predicted_next(self, vehicle: dict[str, Any]) -> tuple[str, float] | None:
        if vehicle["attached_rsu_id"] is None:
            return None
        vx = float(vehicle["velocity_x_mps"])
        vy = float(vehicle["velocity_y_mps"])
        if math.hypot(vx, vy) < 1.0:
            return None
        steps = int(self.lookahead_s / self.projection_step_s)
        for index in range(1, steps + 1):
            eta_s = round(index * self.projection_step_s, 9)
            x = float(vehicle["x_m"]) + vx * eta_s
            y = float(vehicle["y_m"]) + vy * eta_s
            distance, rid = min((math.hypot(x - r.x_m, y - r.y_m), r.rsu_id) for r in self.rsus)
            radius = next(r.radius_m for r in self.rsus if r.rsu_id == rid)
            if distance <= radius and rid != vehicle["attached_rsu_id"]:
                return rid, eta_s
        return None

    def actions(self, observation: dict[str, Any]) -> tuple[ControllerAction, ...]:
        actions: list[ControllerAction] = []
        for vehicle_id, vehicle in sorted(observation["vehicles"].items()):
            current = vehicle["attached_rsu_id"]
            if self._last_attachment.get(vehicle_id) != current:
                self._predicted_for_attachment = {key for key in self._predicted_for_attachment if key[0] != vehicle_id}
                self._last_attachment[vehicle_id] = current
            if not vehicle["request_arrived"]:
                continue
            incomplete = [node_id for node_id in vehicle["node_order"] if node_id not in vehicle["completed_node_ids"]]
            if not incomplete:
                continue
            prediction = self._predicted_next(vehicle)
            if prediction is None:
                continue
            target, eta_s = prediction
            node_id = incomplete[0]
            identity = (vehicle_id, str(current), target)
            if identity in self._predicted_for_attachment:
                continue
            self._predicted_for_attachment.add(identity)
            actions.append(ControllerAction("prepare", vehicle_id, node_id, target, eta_s, self.version))
        return tuple(actions)
