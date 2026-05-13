"""Validate dataset/source declarations used in reports."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.adapter_catalog import AdapterCatalog


DATASET_SOURCES_PATH = ROOT_DIR / "configs" / "data" / "dataset_sources.json"
MODEL_CACHE_SOURCES_PATH = ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"
SAMPLE_CATALOG_PATH = ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "analysis" / "model_cache_dataset_integration_round13"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_reachable_page_shape(url: str) -> bool:
    parsed = urlparse(str(url))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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


def _dataset_rows() -> list[dict[str, Any]]:
    payload = _load_json(DATASET_SOURCES_PATH)
    rows: list[dict[str, Any]] = []
    for item in payload.get("datasets", []):
        dataset_name = str(item.get("dataset_name", "")).strip()
        download_page_url = str(item.get("download_page_url", "")).strip()
        rows.append(
            {
                "source_file": str(DATASET_SOURCES_PATH.relative_to(ROOT_DIR)),
                "dataset_key": item.get("dataset_key", "missing"),
                "dataset_name": dataset_name or "missing",
                "provider": item.get("provider", "missing"),
                "source_role": item.get("source_role", "missing"),
                "local_path": item.get("local_path", "missing"),
                "download_page_url": download_page_url or "missing",
                "has_dataset_name": bool(dataset_name),
                "has_download_page_url": bool(download_page_url),
                "download_page_url_shape_valid": _is_reachable_page_shape(download_page_url),
                "current_status": item.get("current_status", "missing"),
            }
        )
    return rows


def _model_cache_rows() -> list[dict[str, Any]]:
    payload = _load_json(MODEL_CACHE_SOURCES_PATH)
    rows: list[dict[str, Any]] = []
    for item in payload.get("sources", []):
        download_page_url = str(item.get("download_page_url", "")).strip()
        rows.append(
            {
                "source_file": str(MODEL_CACHE_SOURCES_PATH.relative_to(ROOT_DIR)),
                "dataset_id": item.get("dataset_id", "missing"),
                "dataset_name": item.get("dataset_name", "missing"),
                "provider": item.get("provider", "missing"),
                "download_page_url": download_page_url or "missing",
                "download_page_url_shape_valid": _is_reachable_page_shape(download_page_url),
                "integration_status": item.get("integration_status", "missing"),
                "usage_scope": item.get("usage_scope", "missing"),
            }
        )
    return rows


def _catalog_rows() -> list[dict[str, Any]]:
    catalog = AdapterCatalog.from_json(SAMPLE_CATALOG_PATH)
    rows: list[dict[str, Any]] = []
    for item in catalog.model_cache_datasets:
        rows.append(
            {
                "source_file": str(SAMPLE_CATALOG_PATH.relative_to(ROOT_DIR)),
                "dataset_id": item.dataset_id,
                "dataset_name": item.dataset_name,
                "provider": item.provider,
                "download_page_url": item.download_page_url,
                "download_page_url_shape_valid": _is_reachable_page_shape(item.download_page_url),
                "local_status": item.local_status,
                "usage_scope": item.usage_scope,
            }
        )
    return rows


def main() -> None:
    dataset_rows = _dataset_rows()
    model_cache_rows = _model_cache_rows()
    catalog_rows = _catalog_rows()
    all_declaration_rows = dataset_rows + model_cache_rows + catalog_rows

    missing_required = [
        row
        for row in all_declaration_rows
        if row.get("dataset_name") in {"", "missing"}
        or row.get("download_page_url") in {"", "missing"}
        or not row.get("download_page_url_shape_valid")
    ]
    model_cache_dataset_ids = {
        str(row.get("dataset_id"))
        for row in model_cache_rows
        if row.get("dataset_id") not in {None, "", "missing"}
    }
    catalog_dataset_ids = {
        str(row.get("dataset_id"))
        for row in catalog_rows
        if row.get("dataset_id") not in {None, "", "missing"}
    }
    diagnosis = {
        "task_name": "model_cache_dataset_integration_round13",
        "dataset_sources_file": str(DATASET_SOURCES_PATH.relative_to(ROOT_DIR)),
        "model_cache_sources_file": str(MODEL_CACHE_SOURCES_PATH.relative_to(ROOT_DIR)),
        "sample_catalog_file": str(SAMPLE_CATALOG_PATH.relative_to(ROOT_DIR)),
        "dataset_declaration_count": len(dataset_rows),
        "model_cache_source_count": len(model_cache_rows),
        "catalog_model_cache_dataset_count": len(catalog_rows),
        "all_dataset_declarations_have_name_and_download_page": not missing_required,
        "model_cache_dataset_declared_in_catalog": bool(model_cache_dataset_ids & catalog_dataset_ids),
        "all_model_cache_sources_declared_in_catalog": model_cache_dataset_ids <= catalog_dataset_ids,
        "missing_model_cache_sources_in_catalog": sorted(model_cache_dataset_ids - catalog_dataset_ids),
        "metadata_only_no_automatic_download": True,
        "missing_required_rows": missing_required,
        "generated_artifacts": {
            "dataset_source_declarations": str(OUTPUT_DIR / "dataset_source_declarations.csv"),
            "model_cache_source_declarations": str(OUTPUT_DIR / "model_cache_source_declarations.csv"),
            "catalog_model_cache_sources": str(OUTPUT_DIR / "catalog_model_cache_sources.csv"),
            "diagnosis_summary": str(OUTPUT_DIR / "diagnosis_summary.json"),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "dataset_source_declarations.csv", dataset_rows)
    _write_csv(OUTPUT_DIR / "model_cache_source_declarations.csv", model_cache_rows)
    _write_csv(OUTPUT_DIR / "catalog_model_cache_sources.csv", catalog_rows)
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if missing_required or not diagnosis["all_model_cache_sources_declared_in_catalog"]:
        raise SystemExit(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print("dataset source declaration validation complete")
    print(
        "all_dataset_declarations_have_name_and_download_page: "
        f"{diagnosis['all_dataset_declarations_have_name_and_download_page']}"
    )
    print(
        "model_cache_dataset_declared_in_catalog: "
        f"{diagnosis['model_cache_dataset_declared_in_catalog']}"
    )
    print(
        "all_model_cache_sources_declared_in_catalog: "
        f"{diagnosis['all_model_cache_sources_declared_in_catalog']}"
    )
    for key, path in diagnosis["generated_artifacts"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
