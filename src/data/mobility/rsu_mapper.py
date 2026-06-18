"""RSU 关联器。"""

from __future__ import annotations

from math import dist

from src.envs.specs import RSUState, VehicleState


class RSUMapper:
    """依据车辆坐标与覆盖半径完成关联。"""

    def __init__(self, rsu_states: list[RSUState]) -> None:
        self._rsu_states = rsu_states

    def update_rsus(self, rsu_states: list[RSUState]) -> None:
        """更新 RSU 拓扑。"""
        self._rsu_states = rsu_states

    def associate(self, vehicles: list[VehicleState]) -> dict[str, str | None]:
        associations: dict[str, str | None] = {}
        for vehicle in vehicles:
            associations[vehicle.vehicle_id] = self._find_best_rsu(vehicle)
        return associations

    def _find_best_rsu(self, vehicle: VehicleState) -> str | None:
        best_rsu_id: str | None = None
        best_distance: float | None = None
        for rsu in self._rsu_states:
            current_distance = dist(
                (vehicle.position_x, vehicle.position_y),
                (rsu.position_x, rsu.position_y),
            )
            if current_distance > rsu.coverage_radius:
                continue
            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_rsu_id = rsu.rsu_id
        return best_rsu_id
