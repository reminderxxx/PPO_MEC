"""LuST mobility 数据接入骨架。"""

from __future__ import annotations

import csv
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.data.mobility.base_provider import MobilityProvider
from src.envs.specs import VehicleState


class LuSTProvider(MobilityProvider):
    """面向 LuST 的 mobility provider 骨架。

    当前版本约定两类输入：
    1. `scenario_root` 指向 LuST SUMO 场景目录，至少应包含 `lust.net.xml` 与 `*.sumocfg`
    2. `trace_csv_path` 指向已经从 SUMO FCD 或轨迹导出得到的逐时刻 CSV

    逐时刻 CSV 预期字段：
    - `time_index`
    - `vehicle_id`
    - `position_x`
    - `position_y`
    - `speed`
    - `base_model_id` 可选，缺省时回退为 `veh_base_v1`
    """

    REQUIRED_TRACE_COLUMNS = ["time_index", "vehicle_id", "position_x", "position_y", "speed"]

    def __init__(
        self,
        scenario_root: str | Path,
        trace_csv_path: str | Path,
        max_rows: int = 0,
        default_base_model_id: str = "veh_base_v1",
    ) -> None:
        self._scenario_root = Path(scenario_root)
        self._trace_csv_path = Path(trace_csv_path)
        self._default_base_model_id = default_base_model_id
        self._frame_index = 0
        self._active_vehicles: list[VehicleState] = []
        self._trajectory_frames: list[dict[str, Any]] = []
        self._validate_source_paths()
        self._validate_trace_schema()
        if max_rows > 0:
            self._trajectory_frames = self._load_sample_frames(max_rows=max_rows)

    @property
    def scenario_root(self) -> Path:
        return self._scenario_root

    @property
    def trace_csv_path(self) -> Path:
        return self._trace_csv_path

    def reset(self) -> list[VehicleState]:
        self._ensure_frames_loaded()
        self._frame_index = 0
        self._active_vehicles = deepcopy(self._trajectory_frames[0]["vehicles"])
        return self.get_active_vehicles()

    def step(self) -> list[VehicleState]:
        self._ensure_frames_loaded()
        if self._frame_index < len(self._trajectory_frames) - 1:
            self._frame_index += 1
        self._active_vehicles = deepcopy(self._trajectory_frames[self._frame_index]["vehicles"])
        return self.get_active_vehicles()

    def get_active_vehicles(self) -> list[VehicleState]:
        return deepcopy(self._active_vehicles)

    def get_time(self) -> int:
        self._ensure_frames_loaded()
        return int(self._trajectory_frames[self._frame_index]["time_index"])

    def get_loaded_frames(self) -> list[dict[str, Any]]:
        self._ensure_frames_loaded()
        return deepcopy(self._trajectory_frames)

    def get_loaded_frame_count(self) -> int:
        self._ensure_frames_loaded()
        return len(self._trajectory_frames)

    def get_loaded_vehicle_record_count(self) -> int:
        self._ensure_frames_loaded()
        return sum(len(frame["vehicles"]) for frame in self._trajectory_frames)

    def _validate_source_paths(self) -> None:
        if not self._scenario_root.exists():
            raise FileNotFoundError(
                f"LuST 场景目录不存在: {self._scenario_root}。请将 LuSTScenario-master/scenario 放在该路径下。"
            )
        net_file = self._scenario_root / "lust.net.xml"
        if not net_file.exists():
            raise FileNotFoundError(
                f"LuST 场景目录缺少 lust.net.xml: {net_file}。"
            )
        sumo_cfg_files = list(self._scenario_root.glob("*.sumocfg"))
        if not sumo_cfg_files:
            raise FileNotFoundError(
                f"LuST 场景目录缺少 *.sumocfg: {self._scenario_root}。"
            )
        if not self._trace_csv_path.exists():
            raise FileNotFoundError(
                "LuST 轨迹 CSV 不存在: "
                f"{self._trace_csv_path}。当前骨架要求先通过 SUMO FCD 导出或预转换生成逐时刻 CSV。"
            )

    def _validate_trace_schema(self) -> None:
        with self._trace_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames or []
        missing_columns = [column for column in self.REQUIRED_TRACE_COLUMNS if column not in header]
        if missing_columns:
            raise ValueError(
                "LuST 轨迹 CSV 字段不完整，缺少: "
                f"{missing_columns}。预期字段至少包含 {self.REQUIRED_TRACE_COLUMNS}。"
            )

    def _load_sample_frames(self, max_rows: int) -> list[dict[str, Any]]:
        grouped_rows: dict[int, list[VehicleState]] = defaultdict(list)
        loaded_rows = 0
        with self._trace_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_index = int(float(row["time_index"]))
                grouped_rows[time_index].append(
                    VehicleState(
                        vehicle_id=str(row["vehicle_id"]),
                        position_x=float(row["position_x"]),
                        position_y=float(row["position_y"]),
                        speed=float(row["speed"]),
                        base_model_id=str(row.get("base_model_id") or self._default_base_model_id),
                    )
                )
                loaded_rows += 1
                if loaded_rows >= max_rows:
                    break
        if not grouped_rows:
            raise RuntimeError("LuST 轨迹 CSV 已找到，但 sample 读取结果为空。")
        return [
            {"time_index": time_index, "vehicles": grouped_rows[time_index]}
            for time_index in sorted(grouped_rows.keys())
        ]

    def _ensure_frames_loaded(self) -> None:
        if not self._trajectory_frames:
            raise RuntimeError("LuSTProvider 尚未加载任何轨迹帧。请在初始化时传入 max_rows>0。")
