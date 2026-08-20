from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark" / "typed_model_cache_controlled_lru.yaml"
CATALOG_FINGERPRINT = "89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6"
EXPECTED_RUNTIME_HASHES = {
    "constrained": "82ef8858c101aed0f702fa118a11c03db9f5689c3c2776214ef95c70265a689f",
    "medium": "8e3de284b1e839fc05076826308b506c81c419607d5b673d22ba09871d63de85",
    "relaxed": "e314785d4fe14a106322687ee7a4e0e1e1d6739e8fd74f11a08a2ff1c71be3e0",
}


def resolve_formal_runtime(stratum: str, capacity_mb: float) -> dict:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8-sig"))
    config = deepcopy(raw)
    config["cache_capacity_profile"]["capacity_mb"] = capacity_mb
    config["profile_id"] = f"typed_model_cache_formal_{stratum}_v1"
    config["claim_boundary"] = "G14B pre-run frozen protocol; no formal outcome"
    return resolve_model_cache_runtime(config, root=ROOT)


@pytest.mark.parametrize(
    ("stratum", "capacity_mb"),
    [("constrained", 288.0), ("medium", 576.0), ("relaxed", 864.0)],
)
def test_frozen_formal_capacity_runtime_hashes(stratum: str, capacity_mb: float) -> None:
    runtime = resolve_formal_runtime(stratum, capacity_mb)
    assert runtime["runtime_contract_version"] == "typed_model_cache_runtime_contract_v1.0.0"
    assert runtime["cache_capacity_profile"]["unit"] == "mb"
    assert runtime["cache_capacity_profile"]["capacity_mb"] == capacity_mb
    assert runtime["typed_catalog_fingerprint"] == CATALOG_FINGERPRINT
    assert runtime["cache_event_schema_version"] == "1.3.0"
    assert runtime["cache_efficiency_metrics_contract_version"] == "1.2.0"
    assert runtime["runtime_contract_sha256"] == EXPECTED_RUNTIME_HASHES[stratum]


def test_formal_runtime_keeps_atomic_dependency_and_kv_boundary() -> None:
    runtime = resolve_formal_runtime("medium", 576.0)
    transaction = runtime["transaction_contract"]
    assert transaction["atomic_rollback"] is True
    assert transaction["partial_admission"] is False
    assert transaction["max_dependency_bundle_objects"] == 2
    assert not any(
        item["object_type"] == "workflow_state" and item["counts_toward_capacity"]
        for item in runtime["resident_objects"]
    )
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8-sig"))
    assert raw["kv_prefix"]["enabled"] is False
