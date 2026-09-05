from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

import src.runtime.active_formal_bundle as active
from scripts.run_typed_model_cache_formal_protocol import reject_invalid_run_root
from src.evaluators.typed_model_cache_formal_execution import FormalExecutionError


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905"
INDEX = ACTIVE_ROOT / "protocol_index.json"
V17_INDEX = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
    / "protocol_index.json"
)

pytestmark = pytest.mark.skipif(
    json.loads(INDEX.read_text(encoding="utf-8-sig")).get("status")
    != active.READY_STATUS,
    reason="negative active-bundle suite runs after evidence-gated finalization",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.fixture()
def bundle_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    index = load(INDEX)
    paths = [row["logical_path"] for row in index["active_bundle_resources"]]
    readiness = load(ROOT / index["readiness_companion"]["logical_path"])
    paths.append(readiness["evidence_manifest_path"])
    evidence = load(ROOT / readiness["evidence_manifest_path"])
    paths.extend(
        [
            evidence["real_downstream_consumer_rehearsal_path"],
            evidence["formal_training_entrypoint_acceptance_path"],
        ]
    )
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(tmp_path / active.DEFAULT_ACTIVE_INDEX_RELATIVE, index)

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        return "a" * 40

    monkeypatch.setattr(active, "_git", fake_git)
    return tmp_path


def validate(root: Path, **kwargs):
    return active.validate_active_formal_bundle(
        repository_root=root,
        require_clean_git=True,
        require_origin_main_match=True,
        **kwargs,
    )


def refresh_ready_hash(root: Path, index: dict) -> None:
    index["active_formal_bundle_sha256"] = active.canonical_sha256(
        active.ready_index_projection(index)
    )
    dump(root / active.DEFAULT_ACTIVE_INDEX_RELATIVE, index)


def test_ready_active_bundle_validates_and_holdout_capability_is_false(
    bundle_root: Path,
) -> None:
    report = validate(bundle_root)
    assert report["status"] == "pass"
    assert report["holdout_capability"] is False
    assert report["index"]["holdout_seal"] == {
        "sealed": True,
        "opened": False,
        "consumed_permanently": False,
        "performance_gate_forbidden": True,
        "seal_semantic_sha256": "3d9bcbe36d0e8749941067d3c134fee4733bc4c8614762af0ddc0f70bdb9de5f",
    }


def test_v17_pending_index_and_ready_readiness_pending_index_are_rejected(
    bundle_root: Path,
) -> None:
    index = load(V17_INDEX)
    dump(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE, index)
    with pytest.raises(active.ActiveFormalBundleError, match="contract version|index version"):
        validate(bundle_root)
    ready = load(INDEX)
    ready["status"] = "PENDING_G14R7A_CLEAN_ACCEPTANCE"
    dump(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE, ready)
    with pytest.raises(active.ActiveFormalBundleError, match="not ready"):
        validate(bundle_root)


@pytest.mark.parametrize("mode", ["pending", "missing"])
def test_ready_index_with_pending_or_missing_readiness_is_rejected(
    bundle_root: Path, mode: str
) -> None:
    index = load(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    row = next(
        row for row in index["active_bundle_resources"] if row["logical_id"] == "readiness_companion"
    )
    readiness_path = bundle_root / row["logical_path"]
    if mode == "pending":
        readiness = load(readiness_path)
        readiness["status"] = "pending"
        dump(readiness_path, readiness)
    else:
        readiness_path.unlink()
    with pytest.raises(active.ActiveFormalBundleError, match="content drift|missing"):
        validate(bundle_root)


def test_old_environment_path_rejected_even_with_correct_cli_environment(
    bundle_root: Path,
) -> None:
    index = load(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    row = next(
        row
        for row in index["active_bundle_resources"]
        if row["logical_id"] == "execution_environment_manifest"
    )
    old_relative = (
        "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/"
        "execution_environment_manifest.json"
    )
    target = bundle_root / old_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / old_relative,
        target,
    )
    row.update(
        logical_path=old_relative,
        content_sha256=active.sha256_file(target),
        size_bytes=target.stat().st_size,
    )
    index["active_bundle_core_sha256"] = active.canonical_sha256(
        active.active_bundle_core_projection(index)
    )
    refresh_ready_hash(bundle_root, index)
    correct_cli = bundle_root / (
        "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905/"
        "execution_environment_manifest.json"
    )
    with pytest.raises(active.ActiveFormalBundleError, match="outside the active Protocol|does not equal"):
        validate(
            bundle_root,
            execution_environment_manifest_path=correct_cli,
        )


@pytest.mark.parametrize(
    ("logical_id", "expected"),
    [
        ("execution_environment_manifest", "content drift"),
        ("protocol_manifest", "content drift"),
        ("agent_training_scientific_config", "content drift"),
        ("formal_agent_order_contract", "content drift"),
    ],
)
def test_active_resource_content_or_identity_drift_is_rejected(
    bundle_root: Path, logical_id: str, expected: str
) -> None:
    index = load(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    row = next(
        row for row in index["active_bundle_resources"] if row["logical_id"] == logical_id
    )
    path = bundle_root / row["logical_path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(active.ActiveFormalBundleError, match=expected):
        validate(bundle_root)


def test_protocol_path_hash_drift_and_same_name_different_hash_are_rejected(
    bundle_root: Path,
) -> None:
    different = bundle_root / "elsewhere/protocol_v2_7_manifest.json"
    different.parent.mkdir(parents=True)
    different.write_bytes((ACTIVE_ROOT / "protocol_v2_7_manifest.json").read_bytes() + b"\n")
    with pytest.raises(active.ActiveFormalBundleError, match="does not equal"):
        validate(bundle_root, protocol_path=different)


def test_execution_commit_dirty_or_origin_drift_is_rejected(
    bundle_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dirty(_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return " M drift"
        return "a" * 40

    monkeypatch.setattr(active, "_git", dirty)
    with pytest.raises(active.ActiveFormalBundleError, match="clean Git"):
        validate(bundle_root)

    def origin_drift(_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "origin/main"):
            return "b" * 40
        return "a" * 40

    monkeypatch.setattr(active, "_git", origin_drift)
    with pytest.raises(active.ActiveFormalBundleError, match="HEAD == origin/main"):
        validate(bundle_root)


def test_cross_version_resource_requires_explicit_shared_declaration(
    bundle_root: Path,
) -> None:
    index = load(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    row = next(
        row
        for row in index["active_bundle_resources"]
        if row["logical_id"] == "portable_resource_registry"
    )
    row.pop("shared_reason")
    index["active_bundle_core_sha256"] = active.canonical_sha256(
        active.active_bundle_core_projection(index)
    )
    refresh_ready_hash(bundle_root, index)
    with pytest.raises(active.ActiveFormalBundleError, match="not explicitly allowlisted"):
        validate(bundle_root)


def test_missing_readiness_evidence_and_status_without_evidence_hash_are_rejected(
    bundle_root: Path,
) -> None:
    index = load(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    readiness = load(bundle_root / index["readiness_companion"]["logical_path"])
    (bundle_root / readiness["evidence_manifest_path"]).unlink()
    with pytest.raises(active.ActiveFormalBundleError, match="missing"):
        validate(bundle_root)
    index["status"] = active.READY_STATUS
    index.pop("active_formal_bundle_sha256")
    dump(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE, index)
    with pytest.raises(active.ActiveFormalBundleError, match="SHA-256|missing"):
        validate(bundle_root)


def test_symlink_cwd_guessing_and_alternate_index_are_rejected(
    bundle_root: Path,
) -> None:
    alternate = bundle_root / "protocol_index.json"
    alternate.symlink_to(bundle_root / active.DEFAULT_ACTIVE_INDEX_RELATIVE)
    with pytest.raises(active.ActiveFormalBundleError, match="only the unique"):
        validate(bundle_root, index_path=alternate)
    alias = bundle_root / "protocol_alias"
    alias.symlink_to(
        bundle_root
        / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905",
        target_is_directory=True,
    )
    with pytest.raises(active.ActiveFormalBundleError, match="symlink"):
        validate(bundle_root, protocol_path=alias / "protocol_v2_7_manifest.json")


def test_outer_runner_source_gates_dry_run_before_output_writes() -> None:
    source = (ROOT / "scripts/run_typed_model_cache_formal_protocol.py").read_text()
    gate = source.index("validate_active_formal_bundle(")
    dry = source.index("if args.dry_run:")
    binding_write = source.index("atomic_create_execution_binding(", dry)
    assert gate < dry < binding_write
    assert "require_live_execution_protocol(protocol_version)" in source
    assert "--training-entrypoint-acceptance is restricted to a fresh --preflight" in source


def test_all_registered_invalid_roots_including_v11_remain_rejected() -> None:
    protocol = load(ACTIVE_ROOT / "protocol_v2_7_manifest.json")
    assert any(item["run_id"].endswith("g14c_v11") for item in protocol["supersession"]["invalid_execution_runs"])
    for item in protocol["supersession"]["invalid_execution_runs"]:
        root = ROOT / "artifacts/experiments/typed_model_cache_formal" / item["run_id"]
        with pytest.raises(FormalExecutionError, match="permanently rejected"):
            reject_invalid_run_root(protocol, root)
