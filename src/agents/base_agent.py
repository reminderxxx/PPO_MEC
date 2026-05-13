"""统一智能体接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """所有单智能体算法的统一基类。"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    @abstractmethod
    def act(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """根据观测与辅助信息产生动作。"""

    @abstractmethod
    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        """基于 rollout 更新内部状态。"""

    @abstractmethod
    def save(self, path: str) -> None:
        """保存智能体状态。"""

    @abstractmethod
    def load(self, path: str) -> None:
        """加载智能体状态。"""
