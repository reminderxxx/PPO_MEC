"""真实 sample 主线的共享辅助函数。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any

from src.data.mobility.handoff_builder import HandoffBuilder
from src.data.mobility.lust_provider import LuSTProvider
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.envs.specs import RSUState, VehicleState


@dataclass
class RealMobilityBundle:
    """真实 mobility sample 的已加载上下文。"""

    provider: Any
    frames: list[dict[str, Any]]
    rsu_states: list[RSUState]
    rsu_metadata: dict[str, Any]
    source_path: str


def discover_ngsim_csv(root_dir: Path, explicit_path: str = "") -> Path:
    if explicit_path:
        csv_path = Path(explicit_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"指定的 NGSIM CSV 不存在: {csv_path}")
        return csv_path
    ngsim_root = root_dir / "data" / "raw" / "mobility" / "ngsim"
    csv_files = sorted(ngsim_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"未在 {ngsim_root} 找到 NGSIM CSV。请把官方轨迹文件放到该目录下。"
        )
    return csv_files[0]


def discover_lust_csv(root_dir: Path, explicit_path: str = "") -> Path:
    if explicit_path:
        csv_path = Path(explicit_path)
    else:
        csv_path = root_dir / "data" / "processed" / "mobility" / "lust" / "lust_fcd.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LuST 逐时刻 CSV 不存在: {csv_path}。"
            "请先运行 scripts/export_lust_fcd.py 导出 SUMO FCD，再回接 benchmark/training。"
        )
    return csv_path


def resolve_lust_scenario_root(root_dir: Path, explicit_path: str = "") -> Path:
    if explicit_path:
        return Path(explicit_path)
    return root_dir / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"


def load_real_source_frames(
    root_dir: Path,
    mobility_source: str,
    mobility_csv_path: str,
    lust_scenario_root: str,
    max_mobility_rows: int,
) -> tuple[list[dict[str, Any]], str]:
    return _load_real_source_frames_cached(
        str(root_dir.resolve()),
        str(mobility_source),
        str(mobility_csv_path or ""),
        str(lust_scenario_root or ""),
        int(max_mobility_rows),
    )


@lru_cache(maxsize=4)
def _load_real_source_frames_cached(
    root_dir_raw: str,
    mobility_source: str,
    mobility_csv_path: str,
    lust_scenario_root: str,
    max_mobility_rows: int,
) -> tuple[list[dict[str, Any]], str]:
    root_dir = Path(root_dir_raw)
    if mobility_source == "ngsim":
        csv_path = discover_ngsim_csv(root_dir=root_dir, explicit_path=mobility_csv_path)
        provider = NGSIMProvider(csv_path=csv_path, max_rows=max_mobility_rows)
        return provider.get_loaded_frames(), str(csv_path)

    if mobility_source == "lust":
        scenario_root = resolve_lust_scenario_root(root_dir=root_dir, explicit_path=lust_scenario_root)
        trace_csv_path = discover_lust_csv(root_dir=root_dir, explicit_path=mobility_csv_path)
        provider = LuSTProvider(
            scenario_root=scenario_root,
            trace_csv_path=trace_csv_path,
            max_rows=max_mobility_rows,
        )
        return provider.get_loaded_frames(), f"scenario={scenario_root}; trace_csv={trace_csv_path}"

    raise ValueError(f"不支持的 mobility_source: {mobility_source}")


def load_real_mobility_bundle(
    root_dir: Path,
    mobility_source: str,
    mobility_csv_path: str,
    lust_scenario_root: str,
    max_mobility_rows: int,
    rsu_layout: str,
    frame_offset: int = 0,
    window_length: int = 24,
    window_selector: str = "max_handoff_candidate",
    random_seed: int = 7,
) -> RealMobilityBundle:
    raw_frames, source_path = load_real_source_frames(
        root_dir=root_dir,
        mobility_source=mobility_source,
        mobility_csv_path=mobility_csv_path,
        lust_scenario_root=lust_scenario_root,
        max_mobility_rows=max_mobility_rows,
    )
    selection = select_mobility_window(
        frames=raw_frames,
        requested_rsu_layout=rsu_layout,
        frame_offset=frame_offset,
        window_length=window_length,
        window_selector=window_selector,
        random_seed=random_seed,
    )
    selected_frames = selection.pop("selected_frames")
    effective_layout = str(selection.get("effective_rsu_layout") or rsu_layout)
    rsu_states, rsu_metadata = build_sample_rsus(frames=selected_frames, rsu_layout=effective_layout)
    merged_metadata = {
        **selection,
        **rsu_metadata,
        "source_frame_count": len(raw_frames),
        "selected_frame_count": len(selected_frames),
    }
    return RealMobilityBundle(
        provider=ReplayProvider(trajectory_frames=selected_frames),
        frames=selected_frames,
        rsu_states=rsu_states,
        rsu_metadata=merged_metadata,
        source_path=source_path,
    )


def select_mobility_window(
    frames: list[dict[str, Any]],
    requested_rsu_layout: str,
    frame_offset: int = 0,
    window_length: int = 24,
    window_selector: str = "ordered",
    random_seed: int = 7,
) -> dict[str, Any]:
    normalized_length = _normalize_window_length(
        frame_count=len(frames),
        frame_offset=frame_offset,
        window_length=window_length,
    )
    selector = (window_selector or "ordered").strip().lower()

    if selector == "ordered":
        start_index = _normalize_window_start(
            frame_count=len(frames),
            frame_offset=frame_offset,
            window_length=normalized_length,
        )
    elif selector == "random":
        start_index = _sample_random_window_start(
            frame_count=len(frames),
            frame_offset=frame_offset,
            window_length=normalized_length,
            random_seed=random_seed,
        )
    elif selector in {"max_handoff_candidate", "max_axis_crossing"}:
        scan_results = scan_mobility_windows(
            frames=frames,
            layout_candidates=[requested_rsu_layout],
            frame_offset=frame_offset,
            window_length=normalized_length,
            stride=1,
            ranking_mode=selector,
        )
        if not scan_results:
            raise RuntimeError("未找到可用于真实 mobility dry-run 的窗口。")
        start_index = int(scan_results[0]["frame_offset"])
    else:
        raise ValueError(
            "未知 window_selector。支持: ordered, random, max_handoff_candidate, max_axis_crossing"
        )

    selected_frames = frames[start_index : start_index + normalized_length]
    if not selected_frames:
        raise RuntimeError("真实 mobility 窗口选择结果为空。")
    if not _window_within_single_segment(selected_frames):
        raise RuntimeError(
            "真实 mobility 窗口跨越了不同 source_segment_id；"
            "请使用 max_handoff_candidate/random 扫描或调整 frame_offset。"
        )
    window_evaluation = evaluate_window_with_layout(
        window_frames=selected_frames,
        rsu_layout=requested_rsu_layout,
    )
    window_id = _build_window_id(selected_frames=selected_frames, frame_offset=start_index)
    return {
        "selected_frames": selected_frames,
        "window_id": window_id,
        "frame_offset": start_index,
        "window_length": len(selected_frames),
        "window_selector": selector,
        "time_index_start": int(selected_frames[0]["time_index"]),
        "time_index_end": int(selected_frames[-1]["time_index"]),
        "source_segment_id": selected_frames[0].get("source_segment_id", ""),
        "source_location": selected_frames[0].get("source_location", ""),
        "segment_frame_start": selected_frames[0].get("segment_frame_index"),
        "segment_frame_end": selected_frames[-1].get("segment_frame_index"),
        "effective_rsu_layout": window_evaluation["effective_rsu_layout"],
        "requested_rsu_layout": requested_rsu_layout,
        "dominant_axis": window_evaluation["dominant_axis"],
        "chosen_rsu_axis": window_evaluation["chosen_rsu_axis"],
        "coverage_radius": window_evaluation["coverage_radius"],
        "spacing": window_evaluation["spacing"],
        "estimated_association_change_count": window_evaluation["estimated_association_change_count"],
        "estimated_handoff_count": window_evaluation["estimated_handoff_count"],
        "axis_crossing_score": window_evaluation["axis_crossing_score"],
        "active_vehicle_count_mean": window_evaluation["active_vehicle_count_mean"],
        "active_vehicle_count_max": window_evaluation["active_vehicle_count_max"],
        "unique_vehicle_count": window_evaluation["unique_vehicle_count"],
    }


def scan_mobility_windows(
    frames: list[dict[str, Any]],
    layout_candidates: list[str],
    frame_offset: int = 0,
    window_length: int = 24,
    stride: int = 1,
    ranking_mode: str = "max_handoff_candidate",
) -> list[dict[str, Any]]:
    normalized_length = _normalize_window_length(
        frame_count=len(frames),
        frame_offset=frame_offset,
        window_length=window_length,
    )
    if stride <= 0:
        raise ValueError("stride 必须大于 0。")
    if not layout_candidates:
        raise ValueError("layout_candidates 不能为空。")

    results: list[dict[str, Any]] = []
    max_start = len(frames) - normalized_length
    for start_index in range(frame_offset, max_start + 1, stride):
        window_frames = frames[start_index : start_index + normalized_length]
        if not _window_within_single_segment(window_frames):
            continue
        layout_evaluations = [
            evaluate_window_with_layout(window_frames=window_frames, rsu_layout=layout)
            for layout in layout_candidates
        ]
        recommended = max(
            layout_evaluations,
            key=lambda item: (
                item["estimated_association_change_count"],
                item["estimated_handoff_count"],
                item["axis_crossing_score"],
                item["active_vehicle_count_mean"],
            ),
        )
        results.append(
            {
                "window_id": _build_window_id(selected_frames=window_frames, frame_offset=start_index),
                "frame_offset": start_index,
                "window_length": len(window_frames),
                "time_index_start": int(window_frames[0]["time_index"]),
                "time_index_end": int(window_frames[-1]["time_index"]),
                "source_segment_id": window_frames[0].get("source_segment_id", ""),
                "source_location": window_frames[0].get("source_location", ""),
                "segment_frame_start": window_frames[0].get("segment_frame_index"),
                "segment_frame_end": window_frames[-1].get("segment_frame_index"),
                "dominant_axis": recommended["dominant_axis"],
                "recommended_rsu_layout": recommended["effective_rsu_layout"],
                "chosen_rsu_axis": recommended["chosen_rsu_axis"],
                "coverage_radius": recommended["coverage_radius"],
                "spacing": recommended["spacing"],
                "estimated_association_change_count": recommended["estimated_association_change_count"],
                "estimated_handoff_count": recommended["estimated_handoff_count"],
                "axis_crossing_score": recommended["axis_crossing_score"],
                "active_vehicle_count_mean": recommended["active_vehicle_count_mean"],
                "active_vehicle_count_max": recommended["active_vehicle_count_max"],
                "unique_vehicle_count": recommended["unique_vehicle_count"],
                "layout_evaluations": layout_evaluations,
            }
        )

    if ranking_mode == "max_axis_crossing":
        results.sort(
            key=lambda item: (
                item["axis_crossing_score"],
                item["estimated_association_change_count"],
                item["active_vehicle_count_mean"],
            ),
            reverse=True,
        )
    else:
        results.sort(
            key=lambda item: (
                item["estimated_association_change_count"],
                item["estimated_handoff_count"],
                item["axis_crossing_score"],
                item["active_vehicle_count_mean"],
            ),
            reverse=True,
        )
    return results


def evaluate_window_with_layout(
    window_frames: list[dict[str, Any]],
    rsu_layout: str,
) -> dict[str, Any]:
    motion_summary = summarize_mobility_frames(window_frames)
    rsu_states, rsu_metadata = build_sample_rsus(frames=window_frames, rsu_layout=rsu_layout)
    association_stats = estimate_association_change_stats(window_frames=window_frames, rsu_states=rsu_states)
    axis_crossing_score = estimate_axis_crossing_score(
        window_frames=window_frames,
        axis=str(rsu_metadata["chosen_rsu_axis"]),
        spacing=float(rsu_metadata["spacing"]),
    )
    return {
        "effective_rsu_layout": rsu_metadata["effective_rsu_layout"],
        "dominant_axis": motion_summary["dominant_axis"],
        "chosen_rsu_axis": rsu_metadata["chosen_rsu_axis"],
        "coverage_radius": rsu_metadata["coverage_radius"],
        "spacing": rsu_metadata["spacing"],
        "estimated_association_change_count": association_stats["estimated_association_change_count"],
        "estimated_handoff_count": association_stats["estimated_handoff_count"],
        "axis_crossing_score": axis_crossing_score,
        "active_vehicle_count_mean": motion_summary["active_vehicle_count_mean"],
        "active_vehicle_count_max": motion_summary["active_vehicle_count_max"],
        "unique_vehicle_count": motion_summary["unique_vehicle_count"],
    }


def build_sample_rsus(
    frames: list[dict[str, Any]],
    rsu_layout: str = "auto_dominant_tight",
) -> tuple[list[RSUState], dict[str, Any]]:
    motion_summary = summarize_mobility_frames(frames)
    xs = [vehicle.position_x for frame in frames for vehicle in frame["vehicles"]]
    ys = [vehicle.position_y for frame in frames for vehicle in frame["vehicles"]]
    if not xs or not ys:
        raise RuntimeError("真实 mobility sample 中没有可用于构造 RSU 的车辆坐标。")

    dominant_axis = str(motion_summary["dominant_axis"])
    center_x = float(motion_summary["center_x"])
    center_y = float(motion_summary["center_y"])
    x_min = float(motion_summary["x_min"])
    y_min = float(motion_summary["y_min"])
    x_range = float(motion_summary["x_range"])
    y_range = float(motion_summary["y_range"])

    layout_config = _resolve_rsu_layout(
        rsu_layout=rsu_layout,
        dominant_axis=dominant_axis,
        x_range=x_range,
        y_range=y_range,
    )
    axis = str(layout_config["axis"])
    if axis == "grid":
        x_count = int(layout_config["x_count"])
        y_count = int(layout_config["y_count"])
        x_spacing = x_range / max(x_count - 1, 1)
        y_spacing = y_range / max(y_count - 1, 1)
        spacing = max(x_spacing, y_spacing)
        coverage = max(
            ((x_spacing / 2.0) ** 2 + (y_spacing / 2.0) ** 2) ** 0.5
            * float(layout_config["coverage_factor"]),
            float(layout_config.get("min_coverage", 8.0)),
        )
        rsu_states = []
        for x_index in range(x_count):
            for y_index in range(y_count):
                index = x_index * y_count + y_index
                rsu_states.append(
                    RSUState(
                        rsu_id=f"rsu_{chr(ord('a') + index)}",
                        position_x=round(x_min + x_spacing * x_index, 3),
                        position_y=round(y_min + y_spacing * y_index, 3),
                        coverage_radius=round(coverage, 3),
                    )
                )
        rsu_count = len(rsu_states)
    elif axis == "x":
        rsu_count = int(layout_config["count"])
        span = max(x_range, float(layout_config.get("min_span", 60.0)))
        start_value = x_min
        fixed_value = center_y
    else:
        rsu_count = int(layout_config["count"])
        span = max(y_range, float(layout_config.get("min_span", 60.0)))
        start_value = y_min
        fixed_value = center_x

    if axis != "grid":
        spacing = float(layout_config["spacing"]) if layout_config.get("spacing") is not None else span / max(rsu_count - 1, 1)
        coverage = float(layout_config["coverage"]) if layout_config.get("coverage") is not None else max(
            spacing * float(layout_config["coverage_factor"]),
            float(layout_config.get("min_coverage", 8.0)),
        )

        rsu_states = []
        for index in range(rsu_count):
            along_value = start_value + spacing * index
            rsu_id = f"rsu_{chr(ord('a') + index)}"
            if axis == "x":
                rsu_states.append(
                    RSUState(
                        rsu_id=rsu_id,
                        position_x=round(along_value, 3),
                        position_y=round(fixed_value, 3),
                        coverage_radius=round(coverage, 3),
                    )
                )
            else:
                rsu_states.append(
                    RSUState(
                        rsu_id=rsu_id,
                        position_x=round(fixed_value, 3),
                        position_y=round(along_value, 3),
                        coverage_radius=round(coverage, 3),
                    )
                )

    metadata = {
        "requested_rsu_layout": rsu_layout,
        "effective_rsu_layout": layout_config["layout_name"],
        "dominant_axis": dominant_axis,
        "chosen_rsu_axis": axis,
        "axis": axis,
        "rsu_count": rsu_count,
        "spacing": round(spacing, 6),
        "coverage_radius": round(coverage, 6),
        "frame_count": len(frames),
        "vehicle_record_count": sum(len(frame["vehicles"]) for frame in frames),
        "total_abs_dx": motion_summary["total_abs_dx"],
        "total_abs_dy": motion_summary["total_abs_dy"],
        "x_range": motion_summary["x_range"],
        "y_range": motion_summary["y_range"],
    }
    return rsu_states, metadata


def summarize_mobility_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("没有可用于分析的 mobility frames。")
    positions_by_vehicle: dict[str, list[tuple[float, float]]] = {}
    xs: list[float] = []
    ys: list[float] = []
    active_counts: list[int] = []
    for frame in frames:
        vehicles = frame.get("vehicles", [])
        active_counts.append(len(vehicles))
        for vehicle in vehicles:
            xs.append(float(vehicle.position_x))
            ys.append(float(vehicle.position_y))
            positions_by_vehicle.setdefault(vehicle.vehicle_id, []).append(
                (float(vehicle.position_x), float(vehicle.position_y))
            )
    if not xs or not ys:
        raise RuntimeError("mobility frames 中没有车辆坐标。")

    total_abs_dx = 0.0
    total_abs_dy = 0.0
    for samples in positions_by_vehicle.values():
        for index in range(1, len(samples)):
            total_abs_dx += abs(samples[index][0] - samples[index - 1][0])
            total_abs_dy += abs(samples[index][1] - samples[index - 1][1])

    dominant_axis = "x"
    if total_abs_dy > total_abs_dx:
        dominant_axis = "y"
    elif total_abs_dx == total_abs_dy:
        dominant_axis = "y" if (max(ys) - min(ys)) >= (max(xs) - min(xs)) else "x"

    return {
        "dominant_axis": dominant_axis,
        "total_abs_dx": round(total_abs_dx, 6),
        "total_abs_dy": round(total_abs_dy, 6),
        "x_min": round(min(xs), 6),
        "x_max": round(max(xs), 6),
        "y_min": round(min(ys), 6),
        "y_max": round(max(ys), 6),
        "x_range": round(max(xs) - min(xs), 6),
        "y_range": round(max(ys) - min(ys), 6),
        "center_x": round(fmean(xs), 6),
        "center_y": round(fmean(ys), 6),
        "active_vehicle_count_mean": round(sum(active_counts) / max(len(active_counts), 1), 6),
        "active_vehicle_count_max": max(active_counts),
        "unique_vehicle_count": len(positions_by_vehicle),
    }


def estimate_association_change_stats(
    window_frames: list[dict[str, Any]],
    rsu_states: list[RSUState],
) -> dict[str, Any]:
    if len(window_frames) < 2:
        return {
            "estimated_association_change_count": 0,
            "estimated_handoff_count": 0,
        }
    mapper = RSUMapper(rsu_states)
    handoff_builder = HandoffBuilder()
    previous_associations = mapper.associate(window_frames[0]["vehicles"])
    estimated_change_count = 0
    estimated_handoff_count = 0
    for frame in window_frames[1:]:
        current_associations = mapper.associate(frame["vehicles"])
        events = handoff_builder.build_events(
            previous_associations=previous_associations,
            current_associations=current_associations,
            time_index=int(frame["time_index"]),
        )
        estimated_change_count += len(events)
        estimated_handoff_count += sum(1 for event in events if event.event_type == "handoff")
        previous_associations = current_associations
    return {
        "estimated_association_change_count": int(estimated_change_count),
        "estimated_handoff_count": int(estimated_handoff_count),
    }


def estimate_axis_crossing_score(
    window_frames: list[dict[str, Any]],
    axis: str,
    spacing: float,
) -> float:
    axis = axis.strip().lower()
    denominator = max(float(spacing), 1.0)
    samples_by_vehicle: dict[str, list[float]] = {}
    for frame in window_frames:
        for vehicle in frame["vehicles"]:
            axis_value = float(vehicle.position_x) if axis == "x" else float(vehicle.position_y)
            samples_by_vehicle.setdefault(vehicle.vehicle_id, []).append(axis_value)
    total_span = 0.0
    for values in samples_by_vehicle.values():
        if len(values) < 2:
            continue
        total_span += max(values) - min(values)
    return round(total_span / denominator, 6)


def _resolve_rsu_layout(
    rsu_layout: str,
    dominant_axis: str,
    x_range: float,
    y_range: float,
) -> dict[str, Any]:
    layout = (rsu_layout or "auto_dominant_tight").strip().lower()
    axis_range = float(x_range if dominant_axis == "x" else y_range)
    lust_micro_count = max(6, min(16, int(axis_range // 55.0) + 3))
    presets = {
        "auto": {
            "layout_name": "auto",
            "axis": dominant_axis,
            "count": 3,
            "coverage_factor": 0.85,
            "min_span": 60.0,
            "min_coverage": 20.0,
            "spacing": None,
            "coverage": None,
        },
        "auto_dominant_tight": {
            "layout_name": "auto_dominant_tight",
            "axis": dominant_axis,
            "count": 4,
            "coverage_factor": 0.45,
            "min_span": 24.0,
            "min_coverage": 8.0,
            "spacing": None,
            "coverage": None,
        },
        "auto_dominant_wide": {
            "layout_name": "auto_dominant_wide",
            "axis": dominant_axis,
            "count": 3,
            "coverage_factor": 1.2,
            "min_span": 60.0,
            "min_coverage": 25.0,
            "spacing": None,
            "coverage": None,
        },
        "auto_grid_tight": {
            "layout_name": "auto_grid_tight",
            "axis": "grid",
            "x_count": 4,
            "y_count": 4,
            "coverage_factor": 1.05,
            "min_coverage": 8.0,
        },
        "lust_micro": {
            "layout_name": "lust_micro",
            "axis": dominant_axis,
            "count": lust_micro_count,
            "coverage_factor": 1.0,
            "min_span": 220.0,
            "min_coverage": 50.0,
            "spacing": 55.0,
            "coverage": 60.0,
        },
        "tight_x": {
            "layout_name": "tight_x",
            "axis": "x",
            "count": 4,
            "coverage_factor": 0.45,
            "min_span": 24.0,
            "min_coverage": 8.0,
            "spacing": None,
            "coverage": None,
        },
        "tight_y": {
            "layout_name": "tight_y",
            "axis": "y",
            "count": 4,
            "coverage_factor": 0.45,
            "min_span": 24.0,
            "min_coverage": 8.0,
            "spacing": None,
            "coverage": None,
        },
        "wide_x": {
            "layout_name": "wide_x",
            "axis": "x",
            "count": 3,
            "coverage_factor": 1.2,
            "min_span": 60.0,
            "min_coverage": 25.0,
            "spacing": None,
            "coverage": None,
        },
        "wide_y": {
            "layout_name": "wide_y",
            "axis": "y",
            "count": 3,
            "coverage_factor": 1.2,
            "min_span": 60.0,
            "min_coverage": 25.0,
            "spacing": None,
            "coverage": None,
        },
    }
    if layout in presets:
        return presets[layout]

    if layout.startswith("custom:"):
        raw_items = layout.removeprefix("custom:").split(",")
        config = {
            "layout_name": layout,
            "axis": dominant_axis,
            "count": 3,
            "coverage_factor": 0.85,
            "min_span": 60.0,
            "min_coverage": 8.0,
            "spacing": None,
            "coverage": None,
        }
        for item in raw_items:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "axis":
                if value not in {"x", "y"}:
                    raise ValueError(f"rsu_layout 的 axis 只支持 x/y，收到: {value}")
                config["axis"] = value
            elif key == "count":
                config["count"] = max(1, int(value))
            elif key == "coverage":
                config["coverage"] = float(value)
            elif key == "spacing":
                config["spacing"] = float(value)
            elif key == "coverage_factor":
                config["coverage_factor"] = float(value)
            elif key == "min_span":
                config["min_span"] = float(value)
        return config

    raise ValueError(
        "未知 rsu_layout。支持: auto, auto_dominant_tight, auto_dominant_wide, auto_grid_tight, lust_micro, "
        "tight_x, tight_y, wide_x, wide_y, custom:axis=x,count=4,coverage=10,spacing=8"
    )


def _normalize_window_length(
    frame_count: int,
    frame_offset: int,
    window_length: int,
) -> int:
    if frame_count <= 0:
        raise RuntimeError("mobility frame 为空，无法选择窗口。")
    if frame_offset < 0 or frame_offset >= frame_count:
        raise ValueError(
            f"frame_offset 超出范围: {frame_offset}，当前 frame_count={frame_count}。"
        )
    remaining = frame_count - frame_offset
    if remaining <= 0:
        raise RuntimeError("frame_offset 已越过最后一帧，无法选择窗口。")
    if window_length <= 0 or window_length > remaining:
        return remaining
    return max(2, window_length)


def _normalize_window_start(
    frame_count: int,
    frame_offset: int,
    window_length: int,
) -> int:
    max_start = frame_count - window_length
    return min(frame_offset, max_start)


def _sample_random_window_start(
    frame_count: int,
    frame_offset: int,
    window_length: int,
    random_seed: int,
) -> int:
    max_start = frame_count - window_length
    valid_starts = list(range(frame_offset, max_start + 1))
    if not valid_starts:
        raise RuntimeError("没有可用于 random window_selector 的窗口起点。")
    rng = random.Random(random_seed)
    return rng.choice(valid_starts)


def _build_window_id(selected_frames: list[dict[str, Any]], frame_offset: int) -> str:
    time_start = int(selected_frames[0]["time_index"])
    time_end = int(selected_frames[-1]["time_index"])
    segment_id = str(selected_frames[0].get("source_segment_id") or "").strip()
    segment_prefix = f"{segment_id}_" if segment_id else ""
    return f"window_{segment_prefix}off{frame_offset}_len{len(selected_frames)}_t{time_start}_{time_end}"


def _window_within_single_segment(frames: list[dict[str, Any]]) -> bool:
    if not frames:
        return False
    segment_ids = [frame.get("source_segment_id") for frame in frames if frame.get("source_segment_id")]
    if not segment_ids:
        return True
    return len(set(segment_ids)) == 1 and len(segment_ids) == len(frames)
