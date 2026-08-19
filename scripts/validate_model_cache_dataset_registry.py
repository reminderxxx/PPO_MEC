"""Validate the G11 model-cache dataset registry and rebuild its audit artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.model_cache_dataset_registry import (
    assert_valid,
    build_artifact_payloads,
    deterministic_fingerprint,
    load_json,
    validate_all,
    write_artifacts,
)


REGISTRY_PATH = ROOT_DIR / "configs" / "data" / "model_cache_dataset_registry.json"
DATASET_SOURCES_PATH = ROOT_DIR / "configs" / "data" / "dataset_sources.json"
HF_MANIFEST_PATH = ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"
CATALOG_PATH = ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "analysis" / "model_cache_dataset_discovery_20260819_g11_v1"


def main() -> None:
    registry = load_json(REGISTRY_PATH)
    report = validate_all(
        registry,
        dataset_sources=load_json(DATASET_SOURCES_PATH),
        hf_manifest=load_json(HF_MANIFEST_PATH),
        sample_catalog=load_json(CATALOG_PATH),
    )
    report["registry_fingerprint"] = deterministic_fingerprint(registry)
    assert_valid(report)
    payloads = build_artifact_payloads(registry, report)
    manifest = write_artifacts(OUTPUT_DIR, payloads, metadata=registry["audit_metadata"])
    print("model-cache dataset registry validation complete")
    print(f"source_count: {report['registry']['source_count']}")
    print(f"valid: {report['valid']}")
    print(f"registry_fingerprint: {report['registry_fingerprint']}")
    print(f"artifact_file_count: {len(manifest['files']) + 1}")
    print(f"output_dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
