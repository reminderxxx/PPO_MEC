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
    "constrained": "84d88d58c71f0775858e2e38ea544bec4fa657d08ed918567d2c2dff48f418a0",
    "medium": "e811a16576a21b499848d9325d02d7bd94c04e754854b22f5c453c80b66104d8",
    "relaxed": "2ef6f7f98ad4007fc16bd5cf6d30a6ee026ae8340ce321b3aebe6272a50420ba",
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
    assert runtime["cache_efficiency_metrics_contract_version"] == "1.1.0"
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
