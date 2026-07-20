"""NGSIM mobility 数据接入骨架。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.mobility.base_provider import MobilityProvider
from src.envs.specs import VehicleState


class NGSIMProvider(MobilityProvider):
    """面向 NGSIM CSV 的 mobility provider 骨架。

    当前约定输入为官方车辆轨迹 CSV，至少需要以下字段：
    - `Vehicle_ID`
    - `Frame_ID`
    - `Local_X`
    - `Local_Y`
    - `v_Vel`

    该骨架已经支持：
    - 文件存在性检查
    - 表头字段检查
    - `max_rows` 限制下的 sample 级逐帧解析
    """

    REQUIRED_COLUMNS = ["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel"]

    def __init__(
        self,
        csv_path: str | Path,
        max_rows: int = 0,
        default_base_model_id: str = "veh_base_v1",
    ) -> None:
        self._csv_path = Path(csv_path)
        self._default_base_model_id = default_base_model_id
        self._frame_index = 0
        self._active_vehicles: list[VehicleState] = []
        self._trajectory_frames: list[dict[str, Any]] = []
        self._validate_source()
        if max_rows > 0:
            self._trajectory_frames = self._load_sample_frames(max_rows=max_rows)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def reset(self) -> list[VehicleState]:
        self._ensure_frames_loaded()
        self._frame_index = 0
        self._active_vehicles = self._copy_vehicles(self._trajectory_frames[0]["vehicles"])
        return self.get_active_vehicles()

    def step(self) -> list[VehicleState]:
        self._ensure_frames_loaded()
        if self._frame_index < len(self._trajectory_frames) - 1:
            self._frame_index += 1
        self._active_vehicles = self._copy_vehicles(self._trajectory_frames[self._frame_index]["vehicles"])
        return self.get_active_vehicles()

    def get_active_vehicles(self) -> list[VehicleState]:
        return self._copy_vehicles(self._active_vehicles)

    def get_time(self) -> int:
        self._ensure_frames_loaded()
        return int(self._trajectory_frames[self._frame_index]["time_index"])

    def get_loaded_frames(self) -> list[dict[str, Any]]:
        self._ensure_frames_loaded()
        return [self._copy_frame(frame) for frame in self._trajectory_frames]

    def get_loaded_frame_count(self) -> int:
        self._ensure_frames_loaded()
        return len(self._trajectory_frames)

    def get_loaded_vehicle_record_count(self) -> int:
        self._ensure_frames_loaded()
        return sum(len(frame["vehicles"]) for frame in self._trajectory_frames)

    def _validate_source(self) -> None:
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"NGSIM 轨迹文件不存在: {self._csv_path}。请把官方车辆轨迹 CSV 放到 data/raw/mobility/ngsim/ 下。"
            )
        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames or []
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in header]
        if missing_columns:
            raise ValueError(
                f"NGSIM CSV 缺少必要字段: {missing_columns}，预期至少包含 {self.REQUIRED_COLUMNS}。"
            )

    def _load_sample_frames(self, max_rows: int) -> list[dict[str, Any]]:
        grouped_rows: dict[tuple[str, int], dict[str, Any]] = {}
        loaded_rows = 0
        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame_id = int(self._to_float(row["Frame_ID"]))
                location = str(row.get("Location") or "unknown").strip() or "unknown"
                global_time_raw = row.get("Global_Time")
                global_time = int(self._to_float(global_time_raw)) if global_time_raw not in {None, ""} else frame_id
                segment_id = self._segment_id(location)
                frame_key = (segment_id, global_time)
                frame_record = grouped_rows.setdefault(
                    frame_key,
                    {
                        "time_index": global_time,
                        "ngsim_frame_id": frame_id,
                        "global_time": global_time,
                        "source_location": location,
                        "source_segment_id": segment_id,
                        "vehicles": [],
                    },
                )
                frame_record["vehicles"].append(
                    VehicleState(
                        vehicle_id=f"{segment_id}:{row['Vehicle_ID']}",
                        position_x=float(self._to_float(row["Local_X"])),
                        position_y=float(self._to_float(row["Local_Y"])),
                        speed=abs(float(self._to_float(row["v_Vel"]))),
                        base_model_id=self._default_base_model_id,
                    )
                )
                loaded_rows += 1
                if loaded_rows >= max_rows:
                    break
        if not grouped_rows:
            raise RuntimeError("NGSIM CSV 已找到，但 sample 读取结果为空。")
        ordered_keys = sorted(grouped_rows.keys(), key=lambda item: (item[0], item[1]))
        segment_indices: dict[str, int] = defaultdict(int)
        frames: list[dict[str, Any]] = []
        for key in ordered_keys:
            frame = grouped_rows[key]
            segment_id = str(frame["source_segment_id"])
            frame["segment_frame_index"] = segment_indices[segment_id]
            segment_indices[segment_id] += 1
            frames.append(frame)
        return frames

    def _ensure_frames_loaded(self) -> None:
        if not self._trajectory_frames:
            raise RuntimeError(
                "NGSIMProvider 当前只完成了源文件校验。"
                "如需 sample 级回放，请在初始化时传入 max_rows>0。"
            )

    def _copy_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        copied = {key: value for key, value in frame.items() if key != "vehicles"}
        copied["vehicles"] = self._copy_vehicles(frame.get("vehicles", []))
        return copied

    def _copy_vehicles(self, vehicles: list[VehicleState]) -> list[VehicleState]:
        return [
            VehicleState(
                vehicle_id=vehicle.vehicle_id,
                position_x=vehicle.position_x,
                position_y=vehicle.position_y,
                speed=vehicle.speed,
                base_model_id=vehicle.base_model_id,
                associated_rsu_id=vehicle.associated_rsu_id,
                active_workflow_id=vehicle.active_workflow_id,
            )
            for vehicle in vehicles
        ]

    def _to_float(self, raw_value: Any) -> float:
        return float(str(raw_value).replace(",", ""))

    def _segment_id(self, location: str) -> str:
        normalized = "".join(
            char.lower() if char.isalnum() else "_"
            for char in str(location or "unknown")
        ).strip("_")
        return normalized or "unknown"
