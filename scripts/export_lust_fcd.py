"""导出 LuST FCD 为 VEC 可读 CSV。"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as etree
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 LuST SUMO 场景导出为逐时刻 mobility CSV")
    parser.add_argument(
        "--scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
        help="LuST scenario 目录，至少应包含 lust.net.xml 与 *.sumocfg",
    )
    parser.add_argument(
        "--sumocfg",
        type=str,
        default="",
        help="可选，指定要运行的 sumocfg 文件；默认自动选择 due.static.sumocfg 等",
    )
    parser.add_argument(
        "--sumo_binary",
        type=str,
        default="",
        help="可选，指定 sumo 可执行文件；默认从 PATH 中查找 sumo",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "mobility" / "lust" / "lust_fcd.csv"),
        help="导出的逐时刻 CSV 路径",
    )
    parser.add_argument("--begin", type=float, default=0.0, help="SUMO 仿真起始时间")
    parser.add_argument("--end", type=float, default=30.0, help="SUMO 仿真结束时间")
    parser.add_argument("--step_length", type=float, default=1.0, help="SUMO 步长")
    parser.add_argument(
        "--default_base_model_id",
        type=str,
        default="veh_base_v1",
        help="导出 CSV 时默认填充的 base_model_id",
    )
    parser.add_argument(
        "--keep_fcd_xml",
        action="store_true",
        help="保留中间 FCD XML 文件，默认导出后删除",
    )
    parser.add_argument(
        "--temp_fcd_xml_path",
        type=str,
        default="",
        help="可选，指定中间 FCD XML 路径，便于后台导出时监控文件大小增长",
    )
    return parser.parse_args()


def resolve_sumo_binary(explicit_path: str) -> Path:
    if explicit_path:
        binary_path = Path(explicit_path)
        if not binary_path.exists():
            raise FileNotFoundError(f"指定的 SUMO 可执行文件不存在: {binary_path}")
        return binary_path
    discovered = shutil.which("sumo")
    if discovered is None:
        raise FileNotFoundError(
            "未找到 sumo 可执行文件。请安装 SUMO，或通过 --sumo_binary 指定 sumo.exe 的完整路径。"
        )
    return Path(discovered)


def resolve_sumocfg(scenario_root: Path, explicit_sumocfg: str) -> Path:
    if explicit_sumocfg:
        sumocfg_path = Path(explicit_sumocfg)
        if not sumocfg_path.is_absolute():
            sumocfg_path = scenario_root / sumocfg_path
        if not sumocfg_path.exists():
            raise FileNotFoundError(f"指定的 sumocfg 不存在: {sumocfg_path}")
        return sumocfg_path

    preferred_names = [
        "due.static.sumocfg",
        "due.actuated.sumocfg",
        "dua.static.sumocfg",
        "dua.actuated.sumocfg",
    ]
    for name in preferred_names:
        candidate = scenario_root / name
        if candidate.exists():
            return candidate
    all_cfgs = sorted(scenario_root.glob("*.sumocfg"))
    if not all_cfgs:
        raise FileNotFoundError(
            f"LuST scenario 目录缺少 *.sumocfg: {scenario_root}"
        )
    return all_cfgs[0]


def validate_scenario_root(scenario_root: Path) -> None:
    if not scenario_root.exists():
        raise FileNotFoundError(
            f"LuST scenario 目录不存在: {scenario_root}。请检查 data/raw/mobility/LuSTScenario/.../scenario 是否已就绪。"
        )
    net_file = scenario_root / "lust.net.xml"
    if not net_file.exists():
        raise FileNotFoundError(
            f"LuST scenario 缺少必要文件 lust.net.xml: {net_file}"
        )


def validate_sumocfg_inputs(scenario_root: Path, sumocfg_path: Path) -> dict[str, list[str]]:
    tree = etree.parse(sumocfg_path)
    root = tree.getroot()
    missing_files: list[str] = []
    discovered_files: dict[str, list[str]] = {"net": [], "route": [], "additional": []}

    input_node = root.find("input")
    if input_node is None:
        raise ValueError(f"sumocfg 中缺少 <input> 节点: {sumocfg_path}")

    field_map = {
        "net-file": "net",
        "route-files": "route",
        "additional-files": "additional",
    }
    for xml_key, bucket in field_map.items():
        node = input_node.find(xml_key)
        if node is None:
            continue
        raw_value = node.attrib.get("value", "")
        for item in [part.strip() for part in raw_value.split(",") if part.strip()]:
            resolved_path = scenario_root / item
            discovered_files[bucket].append(str(resolved_path))
            if not resolved_path.exists():
                missing_files.append(str(resolved_path))

    if missing_files:
        raise FileNotFoundError(
            "sumocfg 引用了不存在的输入文件:\n"
            + "\n".join(missing_files)
        )
    return discovered_files


def export_fcd_to_csv(
    fcd_xml_path: Path,
    output_csv_path: Path,
    default_base_model_id: str,
) -> tuple[int, int]:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    unique_vehicle_ids: set[str] = set()
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["time_index", "vehicle_id", "position_x", "position_y", "speed", "base_model_id"],
        )
        writer.writeheader()
        context = etree.iterparse(fcd_xml_path, events=("start", "end"))
        current_time_index = 0
        for event, elem in context:
            if event == "start" and elem.tag == "timestep":
                current_time_index = int(float(elem.attrib.get("time", 0.0)))
            if event == "end" and elem.tag == "vehicle":
                vehicle_id = elem.attrib["id"]
                writer.writerow(
                    {
                        "time_index": current_time_index,
                        "vehicle_id": vehicle_id,
                        "position_x": elem.attrib.get("x", "0.0"),
                        "position_y": elem.attrib.get("y", "0.0"),
                        "speed": elem.attrib.get("speed", "0.0"),
                        "base_model_id": default_base_model_id,
                    }
                )
                row_count += 1
                unique_vehicle_ids.add(vehicle_id)
                elem.clear()
    return row_count, len(unique_vehicle_ids)


def main() -> None:
    args = parse_args()
    scenario_root = Path(args.scenario_root)
    output_csv_path = Path(args.output_csv)
    validate_scenario_root(scenario_root)
    sumo_binary = resolve_sumo_binary(args.sumo_binary)
    sumocfg_path = resolve_sumocfg(scenario_root, args.sumocfg)
    sumocfg_inputs = validate_sumocfg_inputs(scenario_root, sumocfg_path)

    if args.temp_fcd_xml_path:
        temp_xml_path = Path(args.temp_fcd_xml_path)
        temp_xml_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_xml_path.exists():
            temp_xml_path.unlink()
    else:
        with tempfile.NamedTemporaryFile(prefix="lust_fcd_", suffix=".xml", delete=False) as temp_file:
            temp_xml_path = Path(temp_file.name)

    command = [
        str(sumo_binary),
        "-c",
        sumocfg_path.name,
        "--begin",
        str(args.begin),
        "--end",
        str(args.end),
        "--step-length",
        str(args.step_length),
        "--fcd-output",
        str(temp_xml_path),
        "--start",
    ]

    print("LuST FCD 导出启动", flush=True)
    print(f"scenario_root: {scenario_root}", flush=True)
    print(f"sumocfg: {sumocfg_path}", flush=True)
    print(f"sumo_binary: {sumo_binary}", flush=True)
    print(f"output_csv: {output_csv_path}", flush=True)
    print(f"temp_fcd_xml_path: {temp_xml_path}", flush=True)
    print(f"time_window: [{args.begin}, {args.end}]", flush=True)

    try:
        completed = subprocess.run(
            command,
            cwd=scenario_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_text = exc.stderr.strip() or exc.stdout.strip() or "SUMO 未返回详细错误。"
        raise RuntimeError(
            "LuST FCD 导出失败。\n"
            f"scenario_root: {scenario_root}\n"
            f"sumocfg: {sumocfg_path}\n"
            f"sumo_binary: {sumo_binary}\n"
            f"command: {' '.join(command)}\n"
            f"net_file: {sumocfg_inputs['net']}\n"
            f"route_files: {sumocfg_inputs['route']}\n"
            f"additional_files: {sumocfg_inputs['additional']}\n"
            f"SUMO 输出: {stderr_text}\n"
            "如果输出仍是 unknown error，请优先检查：SUMO 版本兼容性、sumocfg 对应 route-files、以及 scenario 目录是否能在原生命令行直接跑通。"
        ) from exc

    if not temp_xml_path.exists():
        raise RuntimeError(
            f"SUMO 运行已结束，但未生成 FCD XML: {temp_xml_path}。请检查 sumocfg={sumocfg_path} 是否支持 fcd-output。"
        )

    row_count, unique_vehicle_count = export_fcd_to_csv(
        fcd_xml_path=temp_xml_path,
        output_csv_path=output_csv_path,
        default_base_model_id=args.default_base_model_id,
    )

    if not args.keep_fcd_xml and temp_xml_path.exists():
        temp_xml_path.unlink()

    print("LuST FCD 导出完成")
    print(f"scenario_root: {scenario_root}")
    print(f"sumocfg: {sumocfg_path}")
    print(f"sumo_binary: {sumo_binary}")
    print(f"output_csv: {output_csv_path}")
    print(f"time_window: [{args.begin}, {args.end}]")
    print(f"row_count: {row_count}")
    print(f"unique_vehicle_count: {unique_vehicle_count}")
    if completed.stderr.strip():
        print(f"sumo_stderr_tail: {completed.stderr.strip().splitlines()[-1]}")


if __name__ == "__main__":
    main()
