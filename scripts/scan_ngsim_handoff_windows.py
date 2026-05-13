"""扫描 NGSIM 中更容易出现 handoff 的窗口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.real_sample_support import discover_ngsim_csv, load_real_source_frames, scan_mobility_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 NGSIM mobility sample 中更容易出现 handoff 的窗口")
    parser.add_argument("--mobility_csv_path", type=str, default="", help="可选，显式指定 NGSIM CSV")
    parser.add_argument("--max_mobility_rows", type=int, default=2500, help="读取的最大轨迹记录数")
    parser.add_argument("--frame_offset", type=int, default=0, help="扫描起始偏移")
    parser.add_argument("--window_length", type=int, default=24, help="每个窗口的长度")
    parser.add_argument("--stride", type=int, default=1, help="窗口扫描步长")
    parser.add_argument("--top_k", type=int, default=5, help="输出 top-k 推荐窗口")
    parser.add_argument(
        "--layout_candidates",
        type=str,
        default="auto_dominant_tight,auto_dominant_wide,tight_x,tight_y",
        help="逗号分隔的 RSU 布局候选",
    )
    parser.add_argument("--output_json", type=str, default="", help="可选，导出扫描结果 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout_candidates = [item.strip() for item in args.layout_candidates.split(",") if item.strip()]
    csv_path = discover_ngsim_csv(ROOT_DIR, explicit_path=args.mobility_csv_path)
    frames, source_path = load_real_source_frames(
        root_dir=ROOT_DIR,
        mobility_source="ngsim",
        mobility_csv_path=str(csv_path),
        lust_scenario_root="",
        max_mobility_rows=args.max_mobility_rows,
    )
    scan_results = scan_mobility_windows(
        frames=frames,
        layout_candidates=layout_candidates,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        stride=args.stride,
        ranking_mode="max_handoff_candidate",
    )
    top_results = scan_results[: max(args.top_k, 1)]

    print("NGSIM handoff window 扫描完成")
    print(f"source_path: {source_path}")
    print(f"loaded_frame_count: {len(frames)}")
    print(f"layout_candidates: {layout_candidates}")
    print(f"top_k: {len(top_results)}")
    for index, result in enumerate(top_results, start=1):
        print(
            f"top{index}: window_id={result['window_id']} frame_offset={result['frame_offset']} "
            f"time_range={result['time_index_start']}->{result['time_index_end']} "
            f"estimated_association_change_count={result['estimated_association_change_count']} "
            f"estimated_handoff_count={result['estimated_handoff_count']} "
            f"active_vehicle_count_mean={result['active_vehicle_count_mean']:.3f} "
            f"dominant_axis={result['dominant_axis']} recommended_rsu_layout={result['recommended_rsu_layout']} "
            f"chosen_rsu_axis={result['chosen_rsu_axis']} coverage={result['coverage_radius']} spacing={result['spacing']}"
        )

    payload = {
        "source_path": source_path,
        "loaded_frame_count": len(frames),
        "layout_candidates": layout_candidates,
        "frame_offset": args.frame_offset,
        "window_length": args.window_length,
        "stride": args.stride,
        "top_k": args.top_k,
        "top_results": top_results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
