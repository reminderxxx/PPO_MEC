"""轻量 observation normalization。"""

from __future__ import annotations

import math
from typing import Any


class ObservationNormalizer:
    """对 wrapper 输出的连续特征做稳定化处理。"""

    def __init__(
        self,
        time_scale: float = 16.0,
        count_scale: float = 8.0,
        event_scale: float = 4.0,
        cache_scale: float = 8.0,
        load_scale: float = 6.0,
    ) -> None:
        self._time_scale = float(time_scale)
        self._count_scale = float(count_scale)
        self._event_scale = float(event_scale)
        self._cache_scale = float(cache_scale)
        self._load_scale = float(load_scale)
        self._episode_start_time_index = 0.0

    def reset(self, state: dict[str, Any]) -> None:
        self._episode_start_time_index = float(state.get("time_index", 0.0))

    def normalize(
        self,
        raw_observation: list[float],
        state: dict[str, Any],
        episode_step_index: int,
        max_steps: int,
    ) -> list[float]:
        time_index = float(state.get("time_index", 0.0))
        relative_time = max(0.0, time_index - self._episode_start_time_index)
        safe_horizon = max(float(max_steps), 1.0)
        return [
            math.tanh(relative_time / max(self._time_scale, safe_horizon)),
            min(raw_observation[1] / self._count_scale, 2.0),
            min(raw_observation[2] / self._count_scale, 2.0),
            min(max(raw_observation[3], 0.0), 1.0),
            math.tanh(raw_observation[4] / self._event_scale),
            math.tanh(raw_observation[5] / self._event_scale),
            min(max(raw_observation[6], 0.0), 1.0),
            math.tanh(raw_observation[7] / self._cache_scale),
            math.tanh(raw_observation[8] / self._load_scale),
        ]
