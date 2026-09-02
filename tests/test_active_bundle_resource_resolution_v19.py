from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

import src.runtime.active_formal_bundle as active
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    reject_permanently_invalid_run_references,
)
from scripts.manage_typed_model_cache_formal_artifacts import INVALID_FORMAL_RUN_ROOTS


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / active.DEFAULT_ACTIVE_INDEX_RELATIVE
V8_RUN = "typed_model_cache_formal_20260828_101804_g14c_v8"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture()
def bundle(monkeypatch: pytest.MonkeyPatch) -> dict:
    def fake_git(_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        return "a" * 40

    monkeypatch.setattr(active, "_git", fake_git)
    ready = load(INDEX).get("status") == active.READY_STATUS
    return active.validate_active_formal_bundle(
        repository_root=ROOT, require_ready=ready
    )


def test_v18_schema_failure_is_reproduced_and_not_reintroduced() -> None:
    index = load(
        ROOT
        / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827"
        / "protocol_index.json"
    )
    with pytest.raises(KeyError):
        _ = index["runtime_configs"]
    assert "runtime_configs" not in load(INDEX)
    assert "dev_fairness_manifests" not in load(INDEX)


def test_active_dev_consumer_has_no_raw_legacy_index_access() -> None:
    source = (
        ROOT / "scripts/run_typed_model_cache_formal_dev_selection.py"
    ).read_text(encoding="utf-8")
    assert 'index["runtime_configs"]' not in source
    assert 'index["dev_fairness_manifests"]' not in source
    assert "resolve_capacity_resource_pairs(" in source


def test_all_active_version_gates_include_v19() -> None:
    consumers = (
        "scripts/validate_typed_model_cache_formal_restart.py",
        "scripts/run_typed_model_cache_formal_statistics.py",
        "scripts/train_algo_pool_real_sample.py",
        "src/runtime/formal_training_contract.py",
    )
    for relative in consumers:
        assert '"1.9.0"' in (ROOT / relative).read_text(encoding="utf-8")


def test_resolver_requires_validated_bundle() -> None:
    with pytest.raises(active.ActiveFormalBundleError, match="validate_active_formal_bundle"):
        active.resolve_active_bundle_resource(load(INDEX), "protocol_manifest")


def test_capacity_pairing_is_exact_and_ordered(bundle: dict) -> None:
    pairs = active.resolve_capacity_resource_pairs(
        bundle, fairness_group="dev_fairness_manifests"
    )
    assert [row["capacity_label"] for row in pairs] == list(active.CAPACITY_ORDER)
    for row in pairs:
        label = row["capacity_label"]
        assert row["runtime"]["logical_id"] == f"runtime_configs.{label}"
        assert row["fairness"]["logical_id"] == f"dev_fairness_manifests.{label}"


def test_resource_inventory_reordering_is_not_identity(bundle: dict) -> None:
    expected = active.resolve_capacity_resource_pairs(
        bundle, fairness_group="dev_fairness_manifests"
    )
    random.Random(17).shuffle(bundle["index"]["active_bundle_resources"])
    assert active.resolve_capacity_resource_pairs(
        bundle, fairness_group="dev_fairness_manifests"
    ) == expected


@pytest.mark.parametrize("group", ["runtime_configs", "dev_fairness_manifests"])
def test_missing_capacity_is_rejected(bundle: dict, group: str) -> None:
    logical_id = f"{group}.medium_576mb"
    bundle["index"]["active_bundle_resources"] = [
        row
        for row in bundle["index"]["active_bundle_resources"]
        if row.get("logical_id") != logical_id
    ]
    with pytest.raises(active.ActiveFormalBundleError, match="exactly the frozen three"):
        active.resolve_capacity_resource_pairs(
            bundle, fairness_group="dev_fairness_manifests"
        )


def test_extra_or_mismatched_capacity_is_rejected(bundle: dict) -> None:
    row = deepcopy(
        next(
            item
            for item in bundle["index"]["active_bundle_resources"]
            if item.get("logical_id") == "dev_fairness_manifests.medium_576mb"
        )
    )
    row["logical_id"] = "dev_fairness_manifests.unfrozen_999mb"
    bundle["index"]["active_bundle_resources"].append(row)
    with pytest.raises(active.ActiveFormalBundleError, match="exactly the frozen three"):
        active.resolve_capacity_resource_pairs(
            bundle, fairness_group="dev_fairness_manifests"
        )


def test_duplicate_logical_id_is_rejected(bundle: dict) -> None:
    row = deepcopy(bundle["index"]["active_bundle_resources"][0])
    bundle["index"]["active_bundle_resources"].append(row)
    with pytest.raises(active.ActiveFormalBundleError, match="exactly once"):
        active.resolve_active_bundle_resource(bundle, row["logical_id"])


def test_wrong_role_is_rejected(bundle: dict) -> None:
    row = next(
        item
        for item in bundle["index"]["active_bundle_resources"]
        if item.get("logical_id") == "runtime_configs.medium_576mb"
    )
    row["role"] = "dev fairness manifest"
    with pytest.raises(active.ActiveFormalBundleError, match="role mismatch"):
        active.resolve_capacity_resource_pairs(
            bundle, fairness_group="dev_fairness_manifests"
        )


def test_registered_cli_path_hash_and_size_are_checked(bundle: dict, tmp_path: Path) -> None:
    resource = active.resolve_active_bundle_resource(
        bundle, "runtime_configs.medium_576mb", expected_role="typed runtime config"
    )
    active.validate_registered_resource_path(
        resource, resource["resolved_absolute_path"]
    )
    other = tmp_path / Path(resource["resolved_absolute_path"]).name
    other.write_bytes(Path(resource["resolved_absolute_path"]).read_bytes())
    with pytest.raises(active.ActiveFormalBundleError, match="differs"):
        active.validate_registered_resource_path(resource, other)
    drift = dict(resource)
    drift["content_sha256"] = "0" * 64
    with pytest.raises(active.ActiveFormalBundleError, match="content drift"):
        active.validate_registered_resource_path(
            drift, resource["resolved_absolute_path"]
        )
    drift = dict(resource)
    drift["size_bytes"] += 1
    with pytest.raises(active.ActiveFormalBundleError, match="size drift"):
        active.validate_registered_resource_path(
            drift, resource["resolved_absolute_path"]
        )


def test_symlink_and_relative_cli_paths_are_rejected(bundle: dict, tmp_path: Path) -> None:
    resource = active.resolve_active_bundle_resource(bundle, "protocol_manifest")
    alias = tmp_path / "protocol.json"
    alias.symlink_to(resource["resolved_absolute_path"])
    with pytest.raises(active.ActiveFormalBundleError, match="differs"):
        active.validate_registered_resource_path(resource, alias)
    with pytest.raises(active.ActiveFormalBundleError, match="absolute"):
        active.validate_registered_resource_path(resource, Path("protocol.json"))


def test_version_scope_and_same_name_different_hash_are_rejected(bundle: dict) -> None:
    protocol = active.resolve_active_bundle_resource(bundle, "protocol_manifest")
    assert protocol["version_scope"] == "current_protocol_version"
    assert "v2_2" in protocol["logical_path"]
    shared = active.resolve_active_bundle_resource(bundle, "portable_resource_registry")
    assert shared["version_scope"] == "shared_historical_stable"
    altered = dict(protocol)
    altered["content_sha256"] = "f" * 64
    with pytest.raises(active.ActiveFormalBundleError, match="content drift"):
        active.validate_registered_resource_path(
            altered, protocol["resolved_absolute_path"]
        )


def test_support_setting_id_mismatch_is_rejected(bundle: dict) -> None:
    active.resolve_support_resource(
        bundle, "prediction_boundary-efa09ee2409d3d87"
    )
    with pytest.raises(active.ActiveFormalBundleError, match="exactly once"):
        active.resolve_support_resource(bundle, "prediction_boundary-wrong")


def test_resource_audit_binds_bundle_and_all_groups(bundle: dict) -> None:
    audit = active.build_active_bundle_resource_resolution_audit(bundle)
    assert audit["active_bundle_sha256"] == bundle["active_formal_bundle_sha256"]
    assert audit["audit_sha256"] == active.canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )
    assert len(audit["formal_capacity_pairs"]) == 3
    assert len(audit["dev_capacity_pairs"]) == 3
    assert len(audit["nonformal_rehearsal_capacity_pairs"]) == 3


def test_v8_run_is_permanently_rejected_without_order_hash_change(bundle: dict) -> None:
    contract = load(
        ROOT
        / "configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829"
        / "formal_agent_order_contract.json"
    )
    assert contract["semantic_sha256"] == (
        "82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0"
    )
    with pytest.raises(FormalAgentOrderError, match="permanently invalid"):
        reject_permanently_invalid_run_references(
            [ROOT / "artifacts/experiments/typed_model_cache_formal" / V8_RUN],
            contract=contract,
        )
    assert any(V8_RUN in str(path) for path in INVALID_FORMAL_RUN_ROOTS)
    assert any("typed_model_cache_formal_20260830_113339_g14c_v9" in str(path) for path in INVALID_FORMAL_RUN_ROOTS)


def test_historical_index_cannot_start_active_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    old = (
        ROOT
        / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827"
        / "protocol_index.json"
    )
    with pytest.raises(active.ActiveFormalBundleError, match="only the unique"):
        active.validate_active_formal_bundle(
            repository_root=ROOT,
            index_path=old,
            require_clean_git=False,
            require_origin_main_match=False,
        )


def test_holdout_capability_remains_false(bundle: dict) -> None:
    assert bundle["holdout_capability"] is False
    assert bundle["index"]["holdout_seal"]["sealed"] is True
    assert bundle["index"]["holdout_seal"]["opened"] is False
