"""运行 NGSIM sample 检查。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.handoff_builder import HandoffBuilder
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.envs.specs import RSUState, VehicleState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 NGSIM sample 并验证 association / handoff")
    parser.add_argument("--csv_path", type=str, default="", help="NGSIM 官方轨迹 CSV 路径")
    parser.add_argument("--max_rows", type=int, default=1200, help="读入的最大原始行数")
    parser.add_argument("--preview_frames", type=int, default=3, help="打印前几帧")
    parser.add_argument("--preview_vehicles", type=int, default=3, help="每帧打印前几个车辆")
    parser.add_argument("--rsu_count", type=int, default=3, help="用于 association 的临时 RSU 数量")
    return parser.parse_args()


def discover_ngsim_csv(explicit_path: str) -> Path:
    if explicit_path:
        csv_path = Path(explicit_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"指定的 NGSIM CSV 不存在: {csv_path}")
        return csv_path
    ngsim_root = ROOT_DIR / "data" / "raw" / "mobility" / "ngsim"
    csv_files = sorted(ngsim_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"未在 {ngsim_root} 找到 NGSIM CSV。请把官方轨迹文件放到该目录下。"
        )
    return csv_files[0]


def build_sample_rsus(frames: list[dict[str, object]], rsu_count: int) -> tuple[list[RSUState], str, float]:
    vehicles: list[VehicleState] = [vehicle for frame in frames for vehicle in frame["vehicles"]]  # type: ignore[index]
    xs = [vehicle.position_x for vehicle in vehicles]
    ys = [vehicle.position_y for vehicle in vehicles]
    if not xs or not ys:
        raise RuntimeError("NGSIM sample 中没有可用于构造 RSU 的车辆坐标。")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min

    rsus: list[RSUState] = []
    if y_range >= x_range:
        axis = "y"
        center_x = fmean(xs)
        span = max(y_range, 30.0)
        step = span / max(rsu_count - 1, 1)
        coverage = max(step * 0.8, 20.0)
        start_y = y_min
        for index in range(rsu_count):
            rsus.append(
                RSUState(
                    rsu_id=f"rsu_{chr(ord('a') + index)}",
                    position_x=round(center_x, 3),
                    position_y=round(start_y + step * index, 3),
                    coverage_radius=round(coverage, 3),
                )
            )
    else:
        axis = "x"
        center_y = fmean(ys)
        span = max(x_range, 30.0)
        step = span / max(rsu_count - 1, 1)
        coverage = max(step * 0.8, 20.0)
        start_x = x_min
        for index in range(rsu_count):
            rsus.append(
                RSUState(
                    rsu_id=f"rsu_{chr(ord('a') + index)}",
                    position_x=round(start_x + step * index, 3),
                    position_y=round(center_y, 3),
                    coverage_radius=round(coverage, 3),
                )
            )
    return rsus, axis, coverage


def main() -> None:
    args = parse_args()
    csv_path = discover_ngsim_csv(args.csv_path)
    provider = NGSIMProvider(csv_path=csv_path, max_rows=args.max_rows)
    frames = provider.get_loaded_frames()
    unique_vehicle_ids = sorted({vehicle.vehicle_id for frame in frames for vehicle in frame["vehicles"]})

    print("NGSIM sample 读取完成")
    print(f"csv_path: {csv_path}")
    print(f"frame_count: {provider.get_loaded_frame_count()}")
    print(f"vehicle_record_count: {provider.get_loaded_vehicle_record_count()}")
    print(f"unique_vehicle_count: {len(unique_vehicle_ids)}")

    for frame in frames[: args.preview_frames]:
        print(f"frame={frame['time_index']} vehicle_count={len(frame['vehicles'])}")
        for vehicle in frame["vehicles"][: args.preview_vehicles]:
            print(
                f"  vehicle_id={vehicle.vehicle_id} x={vehicle.position_x:.3f} "
                f"y={vehicle.position_y:.3f} speed={vehicle.speed:.3f}"
            )

    rsu_states, axis, coverage = build_sample_rsus(frames, args.rsu_count)
    mapper = RSUMapper(rsu_states)
    handoff_builder = HandoffBuilder()
    previous_associations: dict[str, str | None] = {}
    total_handoff_events = 0

    print("RSU 拓扑摘要")
    print(f"dominant_axis: {axis}")
    print(f"coverage_radius: {coverage:.3f}")
    for rsu in rsu_states:
        print(
            f"  rsu_id={rsu.rsu_id} x={rsu.position_x:.3f} "
            f"y={rsu.position_y:.3f} coverage={rsu.coverage_radius:.3f}"
        )

    for frame in frames[: args.preview_frames]:
        associations = mapper.associate(frame["vehicles"])
        handoff_events = handoff_builder.build_events(
            previous_associations=previous_associations,
            current_associations=associations,
            time_index=int(frame["time_index"]),
        )
        total_handoff_events += sum(1 for event in handoff_events if event.event_type == "handoff")
        association_items = list(associations.items())[: args.preview_vehicles]
        print(f"association_frame={frame['time_index']} sample={association_items}")
        print(f"handoff_events_in_frame={len(handoff_events)}")
        previous_associations = associations

    print(f"sample_total_handoff_events_in_preview: {total_handoff_events}")


if __name__ == "__main__":
    main()
