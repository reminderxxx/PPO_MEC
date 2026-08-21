from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.manage_typed_model_cache_formal_artifacts import (
    INVALID_G14C_V3_RUN_ROOT,
    checkpoint_freeze,
    scientific_candidate_projection,
)
from scripts.run_typed_model_cache_formal_dev_selection import (
    build_parser as dev_parser,
)
from scripts.run_typed_model_cache_formal_support import (
    benchmark_flags,
    build_parser as support_parser,
)
from src.evaluators.cache_baseline_fairness import semantic_projection
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    READY_V5_VERDICT,
    readiness_v5,
    validate_protocol_v1_1,
)
from src.runtime.portable_resource_identity import (
    ALLOWED_RESOLVERS,
    PortableResourceError,
    add_portable_resource_arguments,
    build_registry,
    build_resource_identity,
    canonical_json_bytes,
    load_registry,
    resolve_argument_resources,
    resolve_resource,
    scientific_identity_fingerprint,
    scientific_identity_projection,
    validate_registry,
    validate_resource_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
)
ARTIFACT_ROOT = (
    ROOT / "artifacts/analysis/typed_model_cache_formal_path_repair_20260821_g14r3_v1"
)


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def identity(path: Path, **overrides) -> dict:
    values = {
        "logical_resource_id": "resource.test",
        "resource_role": "window_plan",
        "schema_version": "test/v1",
        "revision": "fixture",
        "expected_logical_relative_path": "inputs/value.txt",
        "allowed_resolvers": ALLOWED_RESOLVERS,
        "provenance": {"fixture": True},
    }
    values.update(overrides)
    return build_resource_identity(path, **values)


def registry(item: dict) -> dict:
    return build_registry(
        [item], registry_id="test-registry", created_at="2026-08-21T00:00:00Z"
    )


def protocol_v13() -> dict:
    return json.loads(
        (CONFIG_ROOT / "protocol_v1_3_manifest.json").read_text(encoding="utf-8-sig")
    )


def test_contract_versions_and_protocol_v13() -> None:
    report = validate_protocol_v1_1(protocol_v13())
    assert report["status"] == "pass"
    assert report["protocol_version"] == "1.3.0"


def test_protocol_supersession_and_failure_binding() -> None:
    protocol = protocol_v13()
    assert protocol["supersession"]["supersedes_version"] == "1.2.0"
    assert (
        protocol["supersession"]["old_protocol_status"]
        == "invalid_before_dev_performance_execution"
    )
    assert (
        protocol["supersession"]["failure_audit_sha256"]
        == "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af"
    )


def test_protocol_semantic_hash_changed_without_split_or_window_change() -> None:
    diff = json.loads((ARTIFACT_ROOT / "protocol_restart_diff.json").read_text())
    split = json.loads((ARTIFACT_ROOT / "split_revalidation.json").read_text())
    assert diff["semantic_hash_changed"] is True
    assert diff["old_semantic_sha256"] != diff["new_semantic_sha256"]
    assert split["observed"] == "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a"
    assert split["window_contract_semantic_sha256"] == "ec475799b3fba4a3af3e4372e7c25781c6565a88ec814322b4cd4d447fef2771"


def test_canonical_json_rejects_non_finite_and_non_string_keys() -> None:
    for value in ({"x": math.nan}, {"x": math.inf}, {1: "bad"}):
        with pytest.raises(PortableResourceError):
            canonical_json_bytes(value)


def test_scientific_identity_excludes_artifact_location(tmp_path: Path) -> None:
    item = identity(write(tmp_path / "value.txt", "same"))
    first = {**item, "artifact_location": "/machine/a/value.txt"}
    second = {**item, "artifact_location": "/machine/b/value.txt"}
    assert scientific_identity_projection(first) == scientific_identity_projection(second)
    assert scientific_identity_fingerprint(first) == scientific_identity_fingerprint(second)


def test_build_and_validate_identity(tmp_path: Path) -> None:
    item = identity(write(tmp_path / "value.txt", "same"))
    report = validate_resource_identity(item)
    assert report["status"] == "pass"
    assert report["logical_resource_id"] == "resource.test"


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("content_sha256", "bad", "content_sha256"),
        ("size_bytes", -1, "size_bytes"),
        ("allowed_resolvers", ["guess_from_cwd"], "unknown allowed"),
        ("semantic_identity_fingerprint", "0" * 64, "fingerprint mismatch"),
    ],
)
def test_identity_drift_rejected(
    tmp_path: Path, field: str, value: object, pattern: str
) -> None:
    item = identity(write(tmp_path / "value.txt", "same"))
    item[field] = value
    with pytest.raises(PortableResourceError, match=pattern):
        validate_resource_identity(item)


def test_registry_hash_is_order_invariant(tmp_path: Path) -> None:
    first = identity(write(tmp_path / "a.txt", "a"), logical_resource_id="a")
    second = identity(write(tmp_path / "b.txt", "b"), logical_resource_id="b")
    left = build_registry([first, second], registry_id="left", created_at="one")
    right = build_registry([second, first], registry_id="right", created_at="two")
    assert left["hashes"]["semantic_sha256"] == right["hashes"]["semantic_sha256"]


def test_duplicate_registry_id_rejected(tmp_path: Path) -> None:
    item = identity(write(tmp_path / "value.txt", "same"))
    with pytest.raises(PortableResourceError, match="duplicate"):
        build_registry([item, item], registry_id="duplicate")


def test_registry_round_trip(tmp_path: Path) -> None:
    payload = registry(identity(write(tmp_path / "value.txt", "same")))
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_registry(load_registry(path))["status"] == "pass"


@pytest.mark.parametrize(
    ("method", "root_key"),
    [
        ("data_root", "data_root"),
        ("worktree_root", "worktree_root"),
        ("protocol_artifact_root", "protocol_artifact_root"),
        ("checkpoint_root", "checkpoint_root"),
    ],
)
def test_root_resolvers(
    tmp_path: Path, method: str, root_key: str
) -> None:
    root = tmp_path / method
    target = write(root / "inputs/value.txt", "same")
    item = identity(target, allowed_resolvers=(method,))
    report = resolve_resource(registry(item), "resource.test", roots={root_key: root})
    assert report["status"] == "compatible"
    assert report["resolution_method"] == method


def test_manifest_relative_resolver(tmp_path: Path) -> None:
    manifest = write(tmp_path / "manifest/registry.json", "placeholder")
    target = write(tmp_path / "manifest/inputs/value.txt", "same")
    item = identity(target, allowed_resolvers=("manifest_relative",))
    report = resolve_resource(
        registry(item), "resource.test", manifest_path=manifest
    )
    assert report["resolution_method"] == "manifest_relative"


def test_content_identical_explicit_relocation(tmp_path: Path) -> None:
    original = write(tmp_path / "one/value.txt", "same")
    relocated = write(tmp_path / "two/value.txt", "same")
    report = resolve_resource(
        registry(identity(original, allowed_resolvers=("explicit_path",))),
        "resource.test",
        explicit_paths=[relocated],
    )
    assert report["status"] == "compatible"
    assert report["resolved_path"] == str(relocated.resolve())


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        ("content", "content_sha256_mismatch"),
        ("size", "size_mismatch"),
        ("missing", "missing"),
    ],
)
def test_resolution_failure_classes(
    tmp_path: Path, mutation: str, pattern: str
) -> None:
    original = write(tmp_path / "original.txt", "same")
    candidate = tmp_path / "candidate.txt"
    if mutation == "content":
        write(candidate, "diff")
    elif mutation == "size":
        write(candidate, "different-size")
    item = identity(original, allowed_resolvers=("explicit_path",))
    if mutation == "content":
        candidate.write_text("xxxx", encoding="utf-8")
        item["size_bytes"] = candidate.stat().st_size
        item["semantic_identity_fingerprint"] = scientific_identity_fingerprint(item)
    with pytest.raises(PortableResourceError, match=pattern):
        resolve_resource(registry(item), "resource.test", explicit_paths=[candidate])


def test_role_and_schema_swaps_rejected(tmp_path: Path) -> None:
    item = identity(write(tmp_path / "value.txt", "same"))
    with pytest.raises(PortableResourceError, match="role mismatch"):
        resolve_resource(registry(item), "resource.test", expected_role="workflow_dataset")
    with pytest.raises(PortableResourceError, match="schema version"):
        resolve_resource(
            registry(item),
            "resource.test",
            observed_schema_version="wrong/v2",
        )


def test_unknown_logical_id_rejected(tmp_path: Path) -> None:
    payload = registry(identity(write(tmp_path / "value.txt", "same")))
    with pytest.raises(PortableResourceError, match="unknown or duplicate"):
        resolve_resource(payload, "resource.unknown")


def test_conflicting_candidates_rejected(tmp_path: Path) -> None:
    good_root = tmp_path / "good"
    bad = write(tmp_path / "bad.txt", "diff")
    good = write(good_root / "inputs/value.txt", "same")
    item = identity(good, allowed_resolvers=("explicit_path", "data_root"))
    with pytest.raises(PortableResourceError, match="conflicting"):
        resolve_resource(
            registry(item),
            "resource.test",
            explicit_paths=[bad],
            roots={"data_root": good_root},
        )


def test_symlink_resolution_is_audited(tmp_path: Path) -> None:
    original = write(tmp_path / "original.txt", "same")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("symlinks unavailable")
    report = resolve_resource(
        registry(identity(original, allowed_resolvers=("explicit_path",))),
        "resource.test",
        explicit_paths=[link],
    )
    assert report["symlink_audit"]["is_symlink"] is True
    assert report["symlink_audit"]["resolved_path"] == str(original.resolve())


def test_argument_resolver_binds_in_place(tmp_path: Path) -> None:
    target = write(tmp_path / "data/inputs/value.txt", "same")
    payload = registry(
        identity(target, allowed_resolvers=("data_root",), resource_role="window_plan")
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    parser = argparse.ArgumentParser()
    add_portable_resource_arguments(parser)
    parser.add_argument("--window-plan-path", default="")
    args = parser.parse_args(
        [
            "--resource-registry-path", str(registry_path),
            "--data-root", str(tmp_path / "data"),
            "--window-plan-resource-id", "resource.test",
        ]
    )
    report = resolve_argument_resources(
        args,
        bindings=(("window_plan_resource_id", "window_plan_path", "window_plan"),),
    )
    assert report["status"] == "pass"
    assert args.window_plan_path == str(target.resolve())


def test_fairness_semantic_projection_ignores_runtime_locations() -> None:
    manifest = {
        "identity": {"manifest_id": "x", "created_at": "now"},
        "dataset_provenance": {
            "inputs": [
                {
                    "logical_dataset_id": "d",
                    "sha256": "a" * 64,
                    "normalized_absolute_path": "/host/one/data.csv",
                    "runtime_resolution": {"resolved_path": "/host/one/data.csv"},
                }
            ]
        },
    }
    relocated = deepcopy(manifest)
    relocated["dataset_provenance"]["inputs"][0]["normalized_absolute_path"] = "/host/two/data.csv"
    relocated["dataset_provenance"]["inputs"][0]["runtime_resolution"] = {
        "resolved_path": "/host/two/data.csv"
    }
    assert semantic_projection(manifest) == semantic_projection(relocated)


def test_dev_and_support_parsers_require_explicit_workflow() -> None:
    dev_required = {action.dest for action in dev_parser()._actions if action.required}
    support_required = {action.dest for action in support_parser()._actions if action.required}
    assert "workflow_csv_path" in dev_required
    assert "workflow_csv_path" in support_required


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ({"parameter": "typed_semantics", "value": "no_prediction"}, ["--prediction_confidence_scale", "0.0", "--drop_handoff_prediction_prob", "1.0"]),
        ({"parameter": "prediction_condition", "value": "noise_0.2"}, ["--prediction_noise_std", "0.2"]),
        ({"parameter": "prediction_condition", "value": "delay_2"}, ["--prediction_delay_steps", "2"]),
    ],
)
def test_support_setting_flags_are_executed(setting: dict, expected: list[str]) -> None:
    assert benchmark_flags(setting) == expected


def test_support_unknown_setting_binding_fails() -> None:
    with pytest.raises(FormalExecutionError, match="no safe benchmark"):
        benchmark_flags({"parameter": "unbound", "value": 1})


def test_dev_selection_identity_is_path_invariant() -> None:
    row = {"agent_name": "ppo", "checkpoint_path": "/one/a.pt", "artifact_location": {"x": 1}, "value": 2}
    relocated = {**row, "checkpoint_path": "/two/a.pt", "artifact_location": {"x": 2}}
    assert scientific_candidate_projection(row) == scientific_candidate_projection(relocated)


def test_invalid_g14c_v3_checkpoint_is_rejected(tmp_path: Path) -> None:
    protocol = protocol_v13()
    selection = {
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_sha256": "selection",
        "selected": [
            {
                "agent_name": "ppo",
                "seed": 7,
                "capacity_label": "medium_576mb",
                "update_index": 4,
                "checkpoint_path": str(INVALID_G14C_V3_RUN_ROOT / "candidate.pt"),
                "checkpoint_sha256": "0" * 64,
            }
        ],
    }
    (tmp_path / "dev_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid G14C v3"):
        checkpoint_freeze(tmp_path, protocol)


def test_command_resource_matrix_and_negative_cases() -> None:
    commands = json.loads((ARTIFACT_ROOT / "command_resource_validation.json").read_text())
    negatives = json.loads((ARTIFACT_ROOT / "path_negative_cases.json").read_text())
    assert commands["status"] == "pass"
    assert commands["training_command_count"] == 150
    assert commands["formal_support_command_count"] == 24
    assert commands["command_count"] == 186
    assert negatives["status"] == "pass"
    assert len(negatives["cases"]) == 12
    assert all(case["status"] == "pass" for case in negatives["cases"])


def test_readiness_v5_exact_contract() -> None:
    names = {
        "external_resource_matrix_complete",
        "all_resources_content_addressed",
        "no_cwd_path_guessing",
        "main_clean_scientific_identity_parity",
        "training_commands_150_of_150",
        "dev_selector_complete",
        "checkpoint_freeze_complete",
        "formal_support_resolution_complete",
        "exact_non_formal_phase_chain_complete",
        "invalid_g14c_v3_checkpoints_not_reused",
        "holdout_sealed",
        "no_formal_performance_results",
    }
    assert readiness_v5({name: True for name in names}) == READY_V5_VERDICT
    blocked = {name: True for name in names}
    blocked["exact_non_formal_phase_chain_complete"] = False
    assert readiness_v5(blocked) == "BLOCKED_G14R3_READINESS_V5"


def test_readiness_artifact_is_blocked_only_until_rehearsal() -> None:
    readiness = json.loads((ARTIFACT_ROOT / "readiness_review_v5.json").read_text())
    false_checks = {name for name, value in readiness["checks"].items() if not value}
    assert false_checks <= {
        "dev_selector_complete",
        "checkpoint_freeze_complete",
        "exact_non_formal_phase_chain_complete",
    }
    assert readiness["formal_performance_count"] == 0
    assert readiness["holdout_opened"] is False
