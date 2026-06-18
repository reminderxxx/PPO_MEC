"""toy 轨迹回放提供器。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.data.mobility.base_provider import MobilityProvider
from src.envs.specs import VehicleState


class ReplayProvider(MobilityProvider):
    """基于离散时间帧的最小回放器。"""

    def __init__(self, trajectory_frames: list[dict[str, Any]] | None = None) -> None:
        self._trajectory_frames = trajectory_frames or self._build_toy_trajectory()
        self._frame_index = 0
        self._active_vehicles: list[VehicleState] = []
        self._reached_end = False

    def reset(self) -> list[VehicleState]:
        self._frame_index = 0
        self._reached_end = False
        self._active_vehicles = self._frame_to_vehicle_states(self._trajectory_frames[0])
        return self.get_active_vehicles()

    def step(self) -> list[VehicleState]:
        if self._frame_index < len(self._trajectory_frames) - 1:
            self._frame_index += 1
        else:
            self._reached_end = True
        self._active_vehicles = self._frame_to_vehicle_states(
            self._trajectory_frames[self._frame_index]
        )
        return self.get_active_vehicles()

    def get_active_vehicles(self) -> list[VehicleState]:
        return deepcopy(self._active_vehicles)

    def get_time(self) -> int:
        return int(self._trajectory_frames[self._frame_index]["time_index"])

    def reached_end(self) -> bool:
        """返回是否已到轨迹尾部。"""
        return self._reached_end

    def peek_next_vehicles(self) -> list[VehicleState]:
        """返回下一时刻车辆，用于构造简易预测快照。"""
        next_index = min(self._frame_index + 1, len(self._trajectory_frames) - 1)
        return self._frame_to_vehicle_states(self._trajectory_frames[next_index])

    def _frame_to_vehicle_states(self, frame: dict[str, Any]) -> list[VehicleState]:
        vehicles: list[VehicleState] = []
        for item in frame["vehicles"]:
            if isinstance(item, VehicleState):
                vehicles.append(
                    VehicleState(
                        vehicle_id=item.vehicle_id,
                        position_x=float(item.position_x),
                        position_y=float(item.position_y),
                        speed=float(item.speed),
                        base_model_id=item.base_model_id,
                        active_workflow_id=item.active_workflow_id,
                    )
                )
                continue
            vehicles.append(
                VehicleState(
                    vehicle_id=item["vehicle_id"],
                    position_x=float(item["position_x"]),
                    position_y=float(item["position_y"]),
                    speed=float(item["speed"]),
                    base_model_id=item["base_model_id"],
                    active_workflow_id=item.get("active_workflow_id"),
                )
            )
        return vehicles

    def _build_toy_trajectory(self) -> list[dict[str, Any]]:
        return [
            {
                "time_index": 0,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 5.0,
                        "position_y": 0.0,
                        "speed": 12.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 58.0,
                        "position_y": 4.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
            {
                "time_index": 1,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 22.0,
                        "position_y": 0.0,
                        "speed": 12.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 62.0,
                        "position_y": 4.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
            {
                "time_index": 2,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 38.0,
                        "position_y": 0.0,
                        "speed": 12.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 67.0,
                        "position_y": 4.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
            {
                "time_index": 3,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 52.0,
                        "position_y": 0.0,
                        "speed": 11.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 71.0,
                        "position_y": 4.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
            {
                "time_index": 4,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 74.0,
                        "position_y": 0.0,
                        "speed": 11.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 78.0,
                        "position_y": 4.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
            {
                "time_index": 5,
                "vehicles": [
                    {
                        "vehicle_id": "veh_1",
                        "position_x": 95.0,
                        "position_y": 0.0,
                        "speed": 10.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                    {
                        "vehicle_id": "veh_2",
                        "position_x": 84.0,
                        "position_y": 4.0,
                        "speed": 9.0,
                        "base_model_id": "veh_base_v1",
                        "active_workflow_id": "wf_toy_1",
                    },
                ],
            },
        ]
