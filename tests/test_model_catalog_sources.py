"""Model catalog and dataset source declaration tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.adapter_catalog import AdapterCatalog


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class ModelCatalogSourcesTestCase(unittest.TestCase):
    def test_sample_catalog_declares_real_hf_model_cache_dataset(self) -> None:
        catalog = AdapterCatalog.from_json(
            ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"
        )
        self.assertIn(
            "ClemSummer/qwen-model-cache",
            catalog.get_model_cache_dataset_ids(),
        )
        download_pages = catalog.get_model_cache_download_pages()
        self.assertEqual(
            download_pages["ClemSummer/qwen-model-cache"],
            "https://huggingface.co/datasets/ClemSummer/qwen-model-cache",
        )
        self.assertTrue(_is_http_url(download_pages["ClemSummer/qwen-model-cache"]))
        self.assertIn(
            "Kuperberg/bert-model-cache",
            catalog.get_model_cache_dataset_ids(),
        )

    def test_all_hf_model_cache_sources_are_declared_in_catalog(self) -> None:
        manifest = json.loads(
            (
                ROOT_DIR
                / "data"
                / "raw"
                / "model_cache"
                / "huggingface_model_cache_sources.json"
            ).read_text(encoding="utf-8-sig")
        )
        catalog = AdapterCatalog.from_json(
            ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"
        )
        manifest_ids = {
            str(item["dataset_id"])
            for item in manifest.get("sources", [])
            if item.get("dataset_id")
        }
        self.assertLessEqual(
            manifest_ids,
            set(catalog.get_model_cache_dataset_ids()),
        )

    def test_dataset_sources_have_names_and_download_pages(self) -> None:
        payload = json.loads(
            (ROOT_DIR / "configs" / "data" / "dataset_sources.json").read_text(
                encoding="utf-8-sig"
            )
        )
        datasets = payload.get("datasets", [])
        self.assertGreaterEqual(len(datasets), 5)
        for item in datasets:
            self.assertTrue(item.get("dataset_name"), item)
            self.assertTrue(item.get("download_page_url"), item)
            self.assertTrue(_is_http_url(str(item["download_page_url"])), item)


if __name__ == "__main__":
    unittest.main()
