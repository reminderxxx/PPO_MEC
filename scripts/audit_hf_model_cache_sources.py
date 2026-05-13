"""Summarize audited Hugging Face model-cache source suitability."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"
PLAN_PATH = ROOT_DIR / "configs" / "data" / "hf_model_cache_integration_plan.json"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "analysis" / "hf_model_cache_dataset_audit_round14"
REPORT_PATH = ROOT_DIR / "docs" / "agent" / "hf_model_cache_dataset_audit_round14_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _audit_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        viewer = source.get("viewer", {})
        file_summary = source.get("file_summary", {})
        fit = source.get("fit_assessment", {})
        rows.append(
            {
                "dataset_id": source.get("dataset_id", ""),
                "dataset_name": source.get("dataset_name", ""),
                "download_page_url": source.get("download_page_url", ""),
                "observed_downloads": source.get("observed_downloads", ""),
                "license": source.get("license"),
                "viewer_enabled": viewer.get("viewer", False),
                "viewer_rows": viewer.get("viewer_rows", ""),
                "file_count": file_summary.get("file_count", 0),
                "lfs_file_count": file_summary.get("lfs_file_count", 0),
                "total_size_mb": file_summary.get("total_size_mb", 0),
                "direct_benchmark_fit": fit.get("direct_benchmark_fit", "missing"),
                "safe_integration_scope": fit.get("safe_integration_scope", "missing"),
                "integration_status": source.get("integration_status", "missing"),
                "usage_scope": source.get("usage_scope", "missing"),
                "reason": fit.get("reason", ""),
            }
        )
    return rows


def _build_diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_ready = [
        row["dataset_id"]
        for row in rows
        if row["direct_benchmark_fit"] == "ready"
    ]
    file_size_candidates = [
        row["dataset_id"]
        for row in rows
        if row["safe_integration_scope"] not in {"audit_record_only", "missing"}
    ]
    rejected = [
        row["dataset_id"]
        for row in rows
        if row["direct_benchmark_fit"] == "not_suitable"
    ]
    return {
        "task_name": "hf_model_cache_dataset_audit_round14",
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT_DIR)),
        "plan_path": str(PLAN_PATH.relative_to(ROOT_DIR)),
        "candidate_count": len(rows),
        "direct_benchmark_ready_count": len(direct_ready),
        "direct_benchmark_ready_dataset_ids": direct_ready,
        "file_size_profile_candidate_count": len(file_size_candidates),
        "file_size_profile_candidate_dataset_ids": file_size_candidates,
        "rejected_dataset_ids": rejected,
        "benchmark_consumption_allowed_now": False,
        "claim_boundary": (
            "HF candidates can support real file-size/cache-volume references only; "
            "they do not provide real VEC cache event traces."
        ),
        "generated_artifacts": {
            "audit_csv": str(OUTPUT_DIR / "hf_model_cache_dataset_audit.csv"),
            "diagnosis_summary": str(OUTPUT_DIR / "diagnosis_summary.json"),
            "report": str(REPORT_PATH),
        },
    }


def _write_report(rows: list[dict[str, Any]], diagnosis: dict[str, Any]) -> None:
    lines = [
        "# hf_model_cache_dataset_audit_round14",
        "",
        "## 结论",
        "",
        "本轮没有把 Hugging Face 数据集作为正式 benchmark 输入。审计后也不建议直接接入正式对比实验：当前候选只能支撑真实模型文件大小、分块大小或大规模 cache-like 体量参考，不能支撑真实 VEC cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace。",
        "",
        "因此当前安全接入边界是 metadata + file-size profile；benchmark consumption 需要先实现显式 importer、adapter_id 映射和单独结果标签。",
        "",
        "## 候选审计",
        "",
        "| Dataset | 可用文件/规模 | Viewer | 当前判定 | 下载页 |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        viewer_text = "yes" if row["viewer_enabled"] else "no"
        lines.append(
            "| {dataset} | {size} MB / {files} files | {viewer} | {fit} | {url} |".format(
                dataset=row["dataset_id"],
                size=row["total_size_mb"],
                files=row["file_count"],
                viewer=viewer_text,
                fit=row["direct_benchmark_fit"],
                url=row["download_page_url"],
            )
        )
    lines.extend(
        [
            "",
            "## 如何接入",
            "",
            "1. 保留 `data/raw/model_cache/huggingface_model_cache_sources.json` 作为 HF 候选全集审计 manifest，不自动下载原始大文件。",
            "2. 只把具备真实模型/cache 文件的候选投影成单独的 `hf_file_size_profile`，从 Hub metadata 的 file size 生成 `CacheObject.size_mb`。",
            "3. `adapter_id` 不能从 HF 文件名自动推断，必须增加显式映射表，例如 `hf_file -> adapter_tracking`，并在报告里标注这是 size-profile projection。",
            "4. benchmark 入口需要显式选择新的 catalog/profile，输出目录必须独立命名为 `hf_model_cache_*`，不能覆盖当前 `NGSIM + Alibaba` 主线结论。",
            "5. 论文或报告中只能声明“使用 HF 真实模型文件大小/缓存体量 profile”，不能声明“使用 HF 真实 cache request trace”。",
            "",
            "## 产物",
            "",
            f"- audit_csv: `{diagnosis['generated_artifacts']['audit_csv']}`",
            f"- diagnosis_summary: `{diagnosis['generated_artifacts']['diagnosis_summary']}`",
            f"- integration_plan: `{diagnosis['plan_path']}`",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = _load_json(MANIFEST_PATH)
    rows = _audit_rows(manifest)
    diagnosis = _build_diagnosis(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "hf_model_cache_dataset_audit.csv", rows)
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(rows, diagnosis)
    print("HF model-cache dataset audit complete")
    print(f"candidate_count: {diagnosis['candidate_count']}")
    print(f"direct_benchmark_ready_count: {diagnosis['direct_benchmark_ready_count']}")
    print(f"file_size_profile_candidate_count: {diagnosis['file_size_profile_candidate_count']}")
    for key, path in diagnosis["generated_artifacts"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
