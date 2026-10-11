"""Read only exact, previously exposed NGSIM development intervals."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace


FROZEN_DESIGN_IDS = ("dev_01", "regression_00", "regression_05")
FEET_TO_METRES = 0.3048


def _segment(location: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(location)).strip("_") or "unknown"


def _number(raw: str) -> float:
    return float(str(raw).replace(",", ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_traces(path: Path, instances: list[dict[str, Any]]) -> tuple[dict[str, RawVehicleTrace], dict[str, Any]]:
    """Scan source once; fail closed per frozen interval without replacements."""
    chosen = {row["design_id"]: row for row in instances if row["design_id"] in FROZEN_DESIGN_IDS}
    if tuple(design_id for design_id in FROZEN_DESIGN_IDS if design_id in chosen) != FROZEN_DESIGN_IDS:
        raise ValueError("frozen development instance list is incomplete")
    selectors = {design_id: row["source_interval"] for design_id, row in chosen.items()}
    rows: dict[str, list[dict[str, str]]] = {design_id: [] for design_id in FROZEN_DESIGN_IDS}
    scanned = 0
    for chunk in pd.read_csv(path, usecols=["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "Location"],
                             dtype=str, keep_default_na=False, chunksize=500_000):
        scanned += len(chunk)
        times = pd.to_numeric(chunk["Global_Time"], errors="coerce")
        for design_id, interval in selectors.items():
            subset = chunk[times.between(int(interval["time_index_start"]), int(interval["time_index_end"]))]
            if len(subset):
                subset = subset[subset["Location"].map(_segment) == interval["source_segment_id"]]
                rows[design_id].extend(subset.to_dict("records"))
    traces: dict[str, RawVehicleTrace] = {}
    metadata: dict[str, Any] = {"raw_file_size_bytes": path.stat().st_size,
                                "raw_file_sha256": _sha256(path), "scanned_rows": scanned,
                                "selection_rule": "smallest_numeric_vehicle_id_in_first_frame_no_replacement",
                                "units": {"Global_Time": "milliseconds", "Local_X_Local_Y": "feet", "sim_time": "seconds", "sim_position": "metres"},
                                "instances": {}}
    for design_id in FROZEN_DESIGN_IDS:
        interval = selectors[design_id]
        matched = rows[design_id]
        entry: dict[str, Any] = {"split": chosen[design_id]["split"], "window_id": chosen[design_id]["window_id"],
                                 "source_interval": interval, "matched_source_rows": len(matched)}
        metadata["instances"][design_id] = entry
        try:
            lo, hi = int(interval["time_index_start"]), int(interval["time_index_end"])
            expected = [lo + 100 * i for i in range(int(interval["window_length"]))]
            actual = sorted({_number(row["Global_Time"]) for row in matched})
            if expected[-1] != hi or actual != expected:
                raise ValueError("source timestamp bounds, count or 100 ms cadence mismatch")
            first_ids = sorted({int(_number(row["Vehicle_ID"])) for row in matched if int(_number(row["Global_Time"])) == lo})
            if not first_ids:
                raise ValueError("first source timestamp has no vehicle")
            vehicle_id = first_ids[0]
            vehicle = sorted((row for row in matched if int(_number(row["Vehicle_ID"])) == vehicle_id),
                             key=lambda row: int(_number(row["Global_Time"])))
            if len(vehicle) != len(expected) or [int(_number(row["Global_Time"])) for row in vehicle] != expected:
                raise ValueError("selected first-frame vehicle is not uniquely present throughout")
            frames = [int(_number(row["Frame_ID"])) for row in vehicle]
            if any(b != a + 1 for a, b in zip(frames, frames[1:])):
                raise ValueError("selected vehicle Frame_ID is not contiguous and increasing")
            xy = tuple((_number(row["Local_X"]) * FEET_TO_METRES,
                        _number(row["Local_Y"]) * FEET_TO_METRES) for row in vehicle)
            if any(not math.isfinite(value) for point in xy for value in point):
                raise ValueError("nonfinite selected vehicle coordinate")
            trace = RawVehicleTrace(tuple((timestamp - lo) / 1000.0 for timestamp in expected), xy, str(vehicle_id))
            traces[design_id] = trace
            entry.update(status="validated", selected_vehicle_id=str(vehicle_id),
                         matched_timestamp_count=len(actual), selected_vehicle_frame_start=frames[0],
                         selected_vehicle_frame_end=frames[-1], trace_duration_seconds=trace.times_seconds[-1],
                         decision_start_seconds=trace.times_seconds[1])
        except (ValueError, KeyError, IndexError) as exc:
            entry.update(status="excluded", reason=str(exc))
    return traces, metadata
