"""检查真实数据是否就绪。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _check_lust() -> dict[str, Any]:
    scenario_root = ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"
    converted_trace = ROOT_DIR / "data" / "processed" / "mobility" / "lust" / "lust_fcd.csv"
    net_file = scenario_root / "lust.net.xml"
    sumocfg_files = sorted(path.name for path in scenario_root.glob("*.sumocfg")) if scenario_root.exists() else []
    missing_items: list[str] = []
    if not scenario_root.exists():
        missing_items.append("LuST scenario 目录")
    if not net_file.exists():
        missing_items.append("lust.net.xml")
    if not sumocfg_files:
        missing_items.append("*.sumocfg")
    if not converted_trace.exists():
        missing_items.append("已导出的 FCD/CSV 轨迹")
    return {
        "dataset_name": "LuST",
        "ready": len(missing_items) == 0,
        "scenario_root": str(scenario_root),
        "converted_trace": str(converted_trace),
        "found_sumocfg_files": sumocfg_files,
        "missing_items": missing_items,
    }


def _check_ngsim() -> dict[str, Any]:
    ngsim_root = ROOT_DIR / "data" / "raw" / "mobility" / "ngsim"
    csv_files = sorted(path.name for path in ngsim_root.glob("*.csv")) if ngsim_root.exists() else []
    missing_items: list[str] = []
    if not ngsim_root.exists():
        missing_items.append("NGSIM 根目录")
    if not csv_files:
        missing_items.append("官方车辆轨迹 CSV")
    return {
        "dataset_name": "NGSIM",
        "ready": len(missing_items) == 0,
        "root": str(ngsim_root),
        "csv_files": csv_files,
        "missing_items": missing_items,
    }


def _check_highd() -> dict[str, Any]:
    highd_root = ROOT_DIR / "data" / "raw" / "mobility" / "highD"
    track_files = sorted(path.name for path in highd_root.glob("*_tracks.csv")) if highd_root.exists() else []
    track_meta_files = sorted(path.name for path in highd_root.glob("*_tracksMeta.csv")) if highd_root.exists() else []
    recording_meta_files = sorted(path.name for path in highd_root.glob("*_recordingMeta.csv")) if highd_root.exists() else []
    missing_items: list[str] = []
    if not highd_root.exists():
        missing_items.append("highD 根目录")
    if not track_files:
        missing_items.append("*_tracks.csv")
    if not track_meta_files:
        missing_items.append("*_tracksMeta.csv")
    if not recording_meta_files:
        missing_items.append("*_recordingMeta.csv")
    return {
        "dataset_name": "highD",
        "ready": len(missing_items) == 0,
        "root": str(highd_root),
        "tracks_files": track_files,
        "tracks_meta_files": track_meta_files,
        "recording_meta_files": recording_meta_files,
        "missing_items": missing_items,
    }


def _check_alibaba_batch_task() -> dict[str, Any]:
    workflow_root = ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018"
    csv_path = workflow_root / "batch_task.csv"
    archive_path = workflow_root / "batch_task.tar.gz"
    missing_items: list[str] = []
    if not workflow_root.exists():
        missing_items.append("Alibaba workflow 根目录")
    if not csv_path.exists():
        missing_items.append("batch_task.csv")
    return {
        "dataset_name": "Alibaba batch task",
        "ready": len(missing_items) == 0,
        "workflow_root": str(workflow_root),
        "batch_task_csv": str(csv_path),
        "batch_task_archive": str(archive_path),
        "archive_exists": archive_path.exists(),
        "missing_items": missing_items,
    }


def _check_huggingface_model_cache_sources() -> dict[str, Any]:
    model_cache_manifest = ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"
    dataset_sources_manifest = ROOT_DIR / "configs" / "data" / "dataset_sources.json"
    missing_items: list[str] = []
    source_count = 0
    declared_dataset_ids: list[str] = []

    if not model_cache_manifest.exists():
        missing_items.append("Hugging Face model-cache audit manifest")
    else:
        payload = json.loads(model_cache_manifest.read_text(encoding="utf-8-sig"))
        sources = list(payload.get("sources", []))
        source_count = len(sources)
        declared_dataset_ids = [
            str(item.get("dataset_id"))
            for item in sources
            if item.get("dataset_id")
        ]
        for item in sources:
            if not item.get("dataset_name") or not item.get("download_page_url"):
                missing_items.append("model-cache source dataset_name/download_page_url")
                break

    if not dataset_sources_manifest.exists():
        missing_items.append("统一数据源声明 manifest")

    return {
        "dataset_name": "Hugging Face model-cache audit metadata",
        "ready": len(missing_items) == 0,
        "manifest": str(model_cache_manifest),
        "dataset_sources_manifest": str(dataset_sources_manifest),
        "source_count": source_count,
        "declared_dataset_ids": declared_dataset_ids,
        "missing_items": missing_items,
    }


def main() -> None:
    results = [
        _check_lust(),
        _check_ngsim(),
        _check_highd(),
        _check_alibaba_batch_task(),
        _check_huggingface_model_cache_sources(),
    ]
    ready_count = sum(1 for item in results if item["ready"])
    print("数据就绪检查完成")
    print(f"ready_count: {ready_count}/{len(results)}")
    for item in results:
        status_text = "已就绪" if item["ready"] else "缺失"
        print(f"[{item['dataset_name']}] {status_text}")
        if item["ready"]:
            print(f"[{item['dataset_name']}] missing_items: []")
        else:
            print(f"[{item['dataset_name']}] missing_items: {item['missing_items']}")
    print("JSON 摘要:")
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
