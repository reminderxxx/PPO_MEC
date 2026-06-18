"""highD mobility 数据接入骨架。"""

from __future__ import annotations

import csv
from collections import defaultdict
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any

from src.data.mobility.base_provider import MobilityProvider
from src.envs.specs import VehicleState


class HighDProvider(MobilityProvider):
    """面向 highD 的 mobility provider 骨架。

    当前约定使用官方拆分后的三类文件：
    - `*_tracks.csv`
    - `*_tracksMeta.csv`
    - `*_recordingMeta.csv`

    其中 sample 级逐帧解析主要读取 `*_tracks.csv`，但初始化会同时检查三类文件是否存在。
    """

    REQUIRED_TRACK_COLUMNS = ["id", "frame", "x", "y", "xVelocity", "yVelocity"]

    def __init__(
        self,
        tracks_csv_path: str | Path,
        tracks_meta_csv_path: str | Path | None = None,
        recording_meta_csv_path: str | Path | None = None,
        max_rows: int = 0,
        default_base_model_id: str = "veh_base_v1",
    ) -> None:
        self._tracks_csv_path = Path(tracks_csv_path)
        self._tracks_meta_csv_path = self._infer_sibling_path(
            explicit_path=tracks_meta_csv_path,
            suffix_from="_tracks.csv",
            suffix_to="_tracksMeta.csv",
        )
        self._recording_meta_csv_path = self._infer_sibling_path(
            explicit_path=recording_meta_csv_path,
            suffix_from="_tracks.csv",
            suffix_to="_recordingMeta.csv",
        )
        self._default_base_model_id = default_base_model_id
        self._frame_index = 0
        self._active_vehicles: list[VehicleState] = []
        self._trajectory_frames: list[dict[str, Any]] = []
        self._validate_source()
        if max_rows > 0:
            self._trajectory_frames = self._load_sample_frames(max_rows=max_rows)

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

    def _infer_sibling_path(
        self,
        explicit_path: str | Path | None,
        suffix_from: str,
        suffix_to: str,
    ) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)
        file_name = self._tracks_csv_path.name
        if file_name.endswith(suffix_from):
            return self._tracks_csv_path.with_name(file_name.replace(suffix_from, suffix_to))
        return self._tracks_csv_path.with_name(file_name + suffix_to)

    def _validate_source(self) -> None:
        if not self._tracks_csv_path.exists():
            raise FileNotFoundError(
                f"highD tracks 文件不存在: {self._tracks_csv_path}。请把 *_tracks.csv 放到 data/raw/mobility/highD/ 下。"
            )
        if not self._tracks_meta_csv_path.exists():
            raise FileNotFoundError(
                f"highD tracksMeta 文件不存在: {self._tracks_meta_csv_path}。"
            )
        if not self._recording_meta_csv_path.exists():
            raise FileNotFoundError(
                f"highD recordingMeta 文件不存在: {self._recording_meta_csv_path}。"
            )
        with self._tracks_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames or []
        missing_columns = [column for column in self.REQUIRED_TRACK_COLUMNS if column not in header]
        if missing_columns:
            raise ValueError(
                f"highD tracks CSV 缺少必要字段: {missing_columns}，预期至少包含 {self.REQUIRED_TRACK_COLUMNS}。"
            )

    def _load_sample_frames(self, max_rows: int) -> list[dict[str, Any]]:
        grouped_rows: dict[int, list[VehicleState]] = defaultdict(list)
        loaded_rows = 0
        with self._tracks_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame_id = int(float(row["frame"]))
                x_velocity = float(row["xVelocity"])
                y_velocity = float(row["yVelocity"])
                grouped_rows[frame_id].append(
                    VehicleState(
                        vehicle_id=str(row["id"]),
                        position_x=float(row["x"]),
                        position_y=float(row["y"]),
                        speed=round(sqrt(x_velocity * x_velocity + y_velocity * y_velocity), 6),
                        base_model_id=self._default_base_model_id,
                    )
                )
                loaded_rows += 1
                if loaded_rows >= max_rows:
                    break
        if not grouped_rows:
            raise RuntimeError("highD tracks CSV 已找到，但 sample 读取结果为空。")
        return [
            {"time_index": frame_id, "vehicles": grouped_rows[frame_id]}
            for frame_id in sorted(grouped_rows.keys())
        ]

    def _ensure_frames_loaded(self) -> None:
        if not self._trajectory_frames:
            raise RuntimeError(
                "HighDProvider 当前只完成了源文件校验。"
                "如需 sample 级回放，请在初始化时传入 max_rows>0。"
            )
