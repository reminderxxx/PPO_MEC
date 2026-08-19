"""Frozen G11 registry, qualification-gate, and compatibility tests."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.check_data_ready import _check_model_cache_dataset_registry
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.model_catalog.model_cache_dataset_registry import (
    build_artifact_payloads,
    deterministic_fingerprint,
    load_json,
    validate_all,
    validate_compatibility,
    validate_registry_payload,
    write_artifacts,
)


REGISTRY_PATH = ROOT_DIR / "configs" / "data" / "model_cache_dataset_registry.json"
DATASET_SOURCES_PATH = ROOT_DIR / "configs" / "data" / "dataset_sources.json"
HF_MANIFEST_PATH = ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"
CATALOG_PATH = ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"


class ModelCacheDatasetRegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_json(REGISTRY_PATH)
        self.dataset_sources = load_json(DATASET_SOURCES_PATH)
        self.hf_manifest = load_json(HF_MANIFEST_PATH)
        self.catalog = load_json(CATALOG_PATH)

    @staticmethod
    def _codes(report: dict) -> set[str]:
        return {str(item["code"]) for item in report["errors"]}

    def test_frozen_registry_is_valid(self) -> None:
        report = validate_all(
            self.registry,
            dataset_sources=self.dataset_sources,
            hf_manifest=self.hf_manifest,
            sample_catalog=self.catalog,
        )
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["registry"]["source_count"], 19)

    def test_duplicate_source_key_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][1]["identity"]["source_key"] = payload["sources"][0]["identity"]["source_key"]
        self.assertIn("duplicate_source_key", self._codes(validate_registry_payload(payload)))

    def test_duplicate_canonical_url_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][1]["identity"]["canonical_landing_page"] = payload["sources"][0]["identity"]["canonical_landing_page"]
        self.assertIn("duplicate_canonical_url", self._codes(validate_registry_payload(payload)))

    def test_unknown_taxonomy_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["identity"]["primary_class"] = "marketing_cache_dataset"
        self.assertIn("unknown_taxonomy", self._codes(validate_registry_payload(payload)))

    def test_invalid_field_availability_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["fields"]["timestamp"] = "maybe"
        self.assertIn("invalid_field_availability", self._codes(validate_registry_payload(payload)))

    def test_score_total_mismatch_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["qualification_score"]["total"] += 1
        self.assertIn("score_total_mismatch", self._codes(validate_registry_payload(payload)))

    def test_class_a_requires_mobility_and_rsu(self) -> None:
        payload = deepcopy(self.registry)
        source = payload["sources"][8]
        source["identity"]["primary_class"] = "joint_vec_ai_cache_trace"
        source["fields"]["mobility_handoff"] = "absent"
        source["fields"]["location_rsu"] = "absent"
        self.assertIn("joint_vec_hard_gate_failed", self._codes(validate_registry_payload(payload)))

    def test_class_b_requires_time_and_model_identity(self) -> None:
        payload = deepcopy(self.registry)
        source = payload["sources"][8]
        source["fields"]["model_id"] = "absent"
        self.assertIn("request_trace_hard_gate_failed", self._codes(validate_registry_payload(payload)))

    def test_class_c_requires_time_and_adapter_identity(self) -> None:
        payload = deepcopy(self.registry)
        source = payload["sources"][8]
        source["identity"]["primary_class"] = "adapter_or_lora_workload_trace"
        source["fields"]["adapter_lora_id"] = "absent"
        self.assertIn("adapter_trace_hard_gate_failed", self._codes(validate_registry_payload(payload)))

    def test_class_d_requires_time_and_prefix_identity(self) -> None:
        payload = deepcopy(self.registry)
        source = payload["sources"][9]
        source["fields"]["reuse_prefix_identity"] = "absent"
        self.assertIn("kv_trace_hard_gate_failed", self._codes(validate_registry_payload(payload)))

    def test_class_e_cannot_be_request_ready(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["fitness"]["request_trace_ready"] = True
        self.assertIn("size_metadata_marked_request_ready", self._codes(validate_registry_payload(payload)))

    def test_unknown_license_cannot_be_formal_ready(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["fitness"]["formal_ready"] = True
        self.assertIn("unknown_license_marked_formal_ready", self._codes(validate_registry_payload(payload)))

    def test_rejected_source_cannot_be_live(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][2]["fitness"]["live_catalog_projection"] = True
        self.assertIn("rejected_source_live_projection", self._codes(validate_registry_payload(payload)))

    def test_dataset_sources_unknown_registry_key_is_rejected(self) -> None:
        declarations = deepcopy(self.dataset_sources)
        declarations["datasets"][4]["model_cache_registry_source_key"] = "missing_key"
        report = validate_compatibility(
            self.registry,
            dataset_sources=declarations,
            hf_manifest=self.hf_manifest,
            sample_catalog=self.catalog,
        )
        self.assertIn("dataset_sources_missing_registry_key", self._codes(report))

    def test_hf_manifest_url_mismatch_is_rejected(self) -> None:
        manifest = deepcopy(self.hf_manifest)
        manifest["sources"][0]["download_page_url"] = "https://example.com/wrong"
        report = validate_compatibility(
            self.registry,
            dataset_sources=self.dataset_sources,
            hf_manifest=manifest,
            sample_catalog=self.catalog,
        )
        self.assertIn("hf_manifest_url_mismatch", self._codes(report))

    def test_invalid_url_and_date_are_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["identity"]["canonical_landing_page"] = "not-a-url"
        payload["sources"][0]["evidence"]["last_verified_date"] = "2026-99-99"
        codes = self._codes(validate_registry_payload(payload))
        self.assertIn("invalid_url", codes)
        self.assertIn("invalid_date", codes)

    def test_non_finite_number_is_rejected(self) -> None:
        payload = deepcopy(self.registry)
        payload["sources"][0]["qualification_score"]["total"] = math.nan
        self.assertIn("non_finite_number", self._codes(validate_registry_payload(payload)))

    def test_json_round_trip_and_fingerprint_are_stable(self) -> None:
        round_trip = json.loads(json.dumps(self.registry, ensure_ascii=False, allow_nan=False))
        self.assertEqual(self.registry, round_trip)
        self.assertEqual(deterministic_fingerprint(self.registry), deterministic_fingerprint(round_trip))

    def test_artifact_generation_is_deterministic(self) -> None:
        report = validate_all(
            self.registry,
            dataset_sources=self.dataset_sources,
            hf_manifest=self.hf_manifest,
            sample_catalog=self.catalog,
        )
        payloads = build_artifact_payloads(self.registry, report)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest_a = write_artifacts(first, payloads, metadata=self.registry["audit_metadata"])
            manifest_b = write_artifacts(second, payloads, metadata=self.registry["audit_metadata"])
            self.assertEqual(manifest_a, manifest_b)

    def test_legacy_adapter_catalog_remains_compatible(self) -> None:
        catalog = AdapterCatalog.from_json(CATALOG_PATH)
        self.assertEqual(
            set(catalog.get_model_cache_dataset_ids()),
            {"ClemSummer/qwen-model-cache", "ClemSummer/cbow-model-cache", "Kuperberg/bert-model-cache"},
        )
        self.assertGreater(len(catalog.cache_objects), 0)

    def test_readiness_check_never_downloads_raw_payloads(self) -> None:
        result = _check_model_cache_dataset_registry()
        self.assertTrue(result["ready"], result)
        self.assertFalse(result["raw_download_performed"])
        self.assertEqual(result["source_count"], 19)


if __name__ == "__main__":
    unittest.main()
