"""移动性数据提供器抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.envs.specs import VehicleState


class MobilityProvider(ABC):
    """移动性提供器接口。"""

    @abstractmethod
    def reset(self) -> list[VehicleState]:
        """重置到初始时刻并返回活跃车辆。"""

    @abstractmethod
    def step(self) -> list[VehicleState]:
        """推进一个时间步并返回活跃车辆。"""

    @abstractmethod
    def get_active_vehicles(self) -> list[VehicleState]:
        """返回当前活跃车辆。"""

    @abstractmethod
    def get_time(self) -> int:
        """返回当前时间索引。"""
