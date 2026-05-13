"""训练器统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """训练器抽象基类。"""

    @abstractmethod
    def run_episode(self, run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """运行一个完整 episode 并返回 summary。"""
