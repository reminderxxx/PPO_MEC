from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.runtime.generated_checkpoint_resources import (
    CAPACITY_MB,
    GeneratedCheckpointResourceError,
    atomic_create_registry,
    audit_forwarded_resource_arguments,
    build_generated_checkpoint_registry,
    canonical_sha256,
    resolve_generated_checkpoint_resource,
    validate_generated_checkpoint_registry,
)
from src.evaluators.typed_model_cache_formal_execution import validate_protocol_v1_1
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities
from scripts.manage_typed_model_cache_formal_artifacts import formal_gate


ROOT = Path(__file__).resolve().parents[1]
V25 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_5_20260905/protocol_v2_5_manifest.json"
V26 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905/protocol_v2_6_manifest.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict, dict, dict]:
    run = tmp_path / "g14r14_rehearsal_run"
    run.mkdir()
    protocol = {
        "hashes": {"semantic_sha256": "1" * 64, "full_sha256": "2" * 64}
    }
    static = {"hashes": {"semantic_sha256": "3" * 64}, "resources": []}
    context = {
        "context_sha256": "4" * 64,
        "scientific_identity": {
            "active_formal_bundle_sha256": "5" * 64,
            "execution_commit": "6" * 40,
        },
    }
    binding = {
        "binding_full_sha256": "7" * 64,
        "agent_scientific_config_semantic_sha256": "8" * 64,
    }
    write_json(run / "phase_state.jsonl", {})
    (run / "phase_state.jsonl").write_text(
        json.dumps({"phase": "checkpoint_freeze", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    frozen = []
    for index, capacity in enumerate(CAPACITY_MB):
        checkpoint = run / "training" / capacity / "agent" / "seed7" / "checkpoints" / "latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{capacity}".encode())
        import hashlib

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        identity = {
            "checkpoint_sha256": digest,
            "capacity": capacity,
            "semantic_identity_fingerprint": f"{index + 9:x}" * 64,
        }
        entry = {"agent": "ppo", "seed": 7, "checkpoint_identity": identity}
        seed = {
            "_portable_checkpoint_manifest": {
                "checkpoint_manifest_version": "1.1.0",
                "checkpoint_location_contract_version": "1.0.0",
                "entries": [entry],
            },
            "ppo": {"7": str(checkpoint)},
        }
        provenance = {"ppo": {"7": {"checkpoint_sha256": digest}}}
        root = run / "checkpoint_manifests" / capacity
        write_json(root / "seed_checkpoint_manifest.json", seed)
        write_json(root / "checkpoint_provenance_manifest.json", provenance)
        frozen.append({"capacity_label": capacity})
    write_json(
        run / "checkpoint_freeze.json",
        {
            "selection_sha256": "a" * 64,
            "freeze_sha256": "b" * 64,
            "frozen_checkpoints": frozen,
        },
    )
    registry = build_generated_checkpoint_registry(
        run_root=run,
        protocol=protocol,
        static_registry=static,
        resolved_execution_context=context,
        execution_binding=binding,
    )
    registry_path = run / "generated_checkpoint_resource_registry.json"
    atomic_create_registry(registry_path, registry)
    return run, protocol, static, context, binding, registry


def validate(run: Path, protocol: dict, static: dict, context: dict, binding: dict, registry: dict) -> dict:
    return validate_generated_checkpoint_registry(
        registry,
        registry_path=run / "generated_checkpoint_resource_registry.json",
        run_root=run,
        expected_run_id=run.name,
        static_registry_semantic_sha256=static["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        protocol_full_sha256=protocol["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=context["scientific_identity"]["execution_commit"],
        resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"],
    )


def rehash(registry: dict) -> dict:
    registry["registry_canonical_sha256"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_canonical_sha256"}
    )
    return registry


def test_generated_registry_positive_and_create_only(tmp_path: Path) -> None:
    run, protocol, static, context, binding, registry = fixture(tmp_path)
    assert validate(run, protocol, static, context, binding, registry)["resource_count"] == 6
    with pytest.raises(FileExistsError, match="create-only"):
        atomic_create_registry(run / "generated_checkpoint_resource_registry.json", registry)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["resources"].pop(), "six resources"),
        (lambda r: r["resources"].__setitem__(1, deepcopy(r["resources"][0])), "duplicate"),
        (lambda r: r["resources"][0].__setitem__("resource_role", "wrong"), "role"),
        (lambda r: r["resources"][0].__setitem__("schema_version", "9"), "schema"),
        (lambda r: r["resources"][0].__setitem__("capacity_mb", 999), "capacity"),
        (lambda r: r["resources"][0].__setitem__("size_bytes", 999), "size"),
        (lambda r: r["resources"][0].__setitem__("content_sha256", "0" * 64), "content"),
        (lambda r: r.__setitem__("protocol_semantic_sha256", "0" * 64), "identity drift"),
        (lambda r: r.__setitem__("current_run_id", "other"), "cross-run"),
    ],
)
def test_generated_registry_negative_mutations(tmp_path: Path, mutation, message: str) -> None:
    run, protocol, static, context, binding, registry = fixture(tmp_path)
    mutated = deepcopy(registry)
    mutation(mutated)
    rehash(mutated)
    with pytest.raises(GeneratedCheckpointResourceError, match=message):
        validate(run, protocol, static, context, binding, mutated)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("protocol_full_sha256", "identity drift"),
        ("active_formal_bundle_sha256", "identity drift"),
        ("execution_commit", "identity drift"),
        ("resolved_execution_context_sha256", "identity drift"),
        ("formal_training_execution_binding_sha256", "identity drift"),
        ("dev_selection_sha256", "stale dev selection"),
        ("checkpoint_freeze_sha256", "stale checkpoint freeze"),
    ],
)
def test_stale_protocol_bundle_context_binding_selection_and_freeze_rejected(
    tmp_path: Path, field: str, message: str
) -> None:
    run, protocol, static, context, binding, registry = fixture(tmp_path)
    changed = deepcopy(registry)
    changed[field] = "f" * len(str(changed[field]))
    rehash(changed)
    with pytest.raises(GeneratedCheckpointResourceError, match=message):
        validate(run, protocol, static, context, binding, changed)


def test_wrong_path_capacity_symlink_and_escape_rejected(tmp_path: Path) -> None:
    run, protocol, static, context, binding, registry = fixture(tmp_path)
    validate(run, protocol, static, context, binding, registry)
    row = next(r for r in registry["resources"] if r["logical_resource_id"] == "checkpoint_manifest.constrained_288mb")
    wrong = run / "checkpoint_manifests" / "medium_576mb" / "seed_checkpoint_manifest.json"
    with pytest.raises(GeneratedCheckpointResourceError, match="explicit path"):
        resolve_generated_checkpoint_resource(
            registry, run_root=run,
            logical_resource_id=row["logical_resource_id"],
            expected_role=row["resource_role"], explicit_path=wrong,
            expected_capacity_label="constrained_288mb",
        )
    changed = deepcopy(registry)
    changed["resources"][0]["durable_run_root_relative_path"] = "../escape.json"
    rehash(changed)
    with pytest.raises(GeneratedCheckpointResourceError, match="normalization|escapes"):
        validate(run, protocol, static, context, binding, changed)
    link = run / "checkpoint_manifests" / "constrained_288mb" / "linked.json"
    link.symlink_to(wrong)
    changed = deepcopy(registry)
    changed["resources"][0]["durable_run_root_relative_path"] = link.relative_to(run).as_posix()
    rehash(changed)
    with pytest.raises(GeneratedCheckpointResourceError, match="symlink"):
        validate(run, protocol, static, context, binding, changed)


def test_static_collision_and_uncommitted_freeze_rejected(tmp_path: Path) -> None:
    run, protocol, static, context, binding, _ = fixture(tmp_path)
    (run / "generated_checkpoint_resource_registry.json").unlink()
    static["resources"] = [{"logical_resource_id": "checkpoint_manifest.medium_576mb"}]
    with pytest.raises(GeneratedCheckpointResourceError, match="collision"):
        build_generated_checkpoint_registry(
            run_root=run, protocol=protocol, static_registry=static,
            resolved_execution_context=context, execution_binding=binding,
        )


def test_historical_g14c_v1_v13_checkpoint_reference_rejected(tmp_path: Path) -> None:
    run, protocol, static, context, binding, _ = fixture(tmp_path)
    (run / "generated_checkpoint_resource_registry.json").unlink()
    capacity = "constrained_288mb"
    manifest_path = run / "checkpoint_manifests" / capacity / "seed_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_path = run / "training/g14c_v13/ppo/seed7/checkpoints/latest.pt"
    old_path.parent.mkdir(parents=True)
    source = Path(manifest["ppo"]["7"])
    old_path.write_bytes(source.read_bytes())
    manifest["ppo"]["7"] = str(old_path)
    write_json(manifest_path, manifest)
    with pytest.raises(GeneratedCheckpointResourceError, match="G14C v1-v13"):
        build_generated_checkpoint_registry(
            run_root=run, protocol=protocol, static_registry=static,
            resolved_execution_context=context, execution_binding=binding,
        )


def test_generated_registry_post_publication_rewrite_is_detected(tmp_path: Path) -> None:
    run, protocol, static, context, binding, registry = fixture(tmp_path)
    path = run / "generated_checkpoint_resource_registry.json"
    rewritten = deepcopy(registry)
    rewritten["registry_id"] = "rewritten"
    write_json(path, rewritten)
    with pytest.raises(GeneratedCheckpointResourceError, match="canonical hash"):
        validate(run, protocol, static, context, binding, rewritten)
    (run / "phase_state.jsonl").write_text(
        json.dumps({"phase": "checkpoint_freeze", "status": "failed"}) + "\n",
        encoding="utf-8",
    )
    static["resources"] = []
    with pytest.raises(GeneratedCheckpointResourceError, match="committed terminal"):
        build_generated_checkpoint_registry(
            run_root=run, protocol=protocol, static_registry=static,
            resolved_execution_context=context, execution_binding=binding,
        )


def test_outer_nested_flags_are_exact_and_not_accepted_unused() -> None:
    values = {
        "resource_registry_path": "/r/static.json",
        "repository_root": "/r",
        "data_root": "/r/data",
        "protocol_artifact_root": "/r/config",
        "checkpoint_root": "/run",
        "mobility_resource_id": "mobility",
        "workflow_resource_id": "workflow",
        "window_plan_resource_id": "window",
        "runtime_config_resource_id": "runtime_config.medium_576mb",
        "fairness_manifest_resource_id": "fairness",
        "generated_checkpoint_registry_path": "/run/generated.json",
        "checkpoint_manifest_id": "checkpoint_manifest.medium_576mb",
        "checkpoint_provenance_id": "checkpoint_provenance.medium_576mb",
    }
    args = argparse.Namespace(**values)
    command = ["python", "child.py"]
    for name, value in values.items():
        command.extend(["--" + name.replace("_", "-"), value])
    assert audit_forwarded_resource_arguments(command, args)["status"] == "pass"
    command.remove("--checkpoint-provenance-id")
    with pytest.raises(GeneratedCheckpointResourceError, match="exactly one"):
        audit_forwarded_resource_arguments(command, args)


def test_protocol_v25_capacity_generator_and_consumer_closure() -> None:
    protocol = json.loads(V26.read_text(encoding="utf-8-sig"))
    assert validate_protocol_v1_1(protocol)["status"] == "pass"
    assert get_protocol_capabilities("2.6.0").generated_checkpoint_resource_required
    templates = protocol["execution_contract"]["command_templates"]
    capacity_rows = [
        row for row in templates["formal_support"]["matrix_contexts"]
        if row["support_setting_id"].startswith("capacity-")
    ]
    assert [row["capacity_label"] for row in capacity_rows] == list(CAPACITY_MB)
    for row in capacity_rows:
        capacity = row["capacity_label"]
        assert row["runtime_config_resource_id"] == f"runtime_config.{capacity}"
        assert row["checkpoint_manifest_id"] == f"checkpoint_manifest.{capacity}"
        assert row["checkpoint_provenance_id"] == f"checkpoint_provenance.{capacity}"
        assert capacity in row["runtime_config_path"]
        assert capacity in row["fairness_manifest_path"]
        assert capacity in row["seed_checkpoint_manifest_path"]
        assert capacity in row["checkpoint_provenance_manifest_path"]
    cache_argv = templates["formal_cache_policy"]["argv"]
    marker = cache_argv.index("--command")
    outer, child = cache_argv[:marker], cache_argv[marker + 1:]
    for flag in (
        "--resource-registry-path", "--generated-checkpoint-registry-path",
        "--checkpoint-manifest-id", "--checkpoint-provenance-id",
    ):
        assert flag in outer
        assert flag in child


def test_static_registry_does_not_predeclare_run_generated_checkpoint_hashes() -> None:
    protocol = json.loads(V26.read_text(encoding="utf-8-sig"))
    registry_path = ROOT / protocol["execution_contract"]["default_expansion_context"]["resource_registry_path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    ids = {row["logical_resource_id"] for row in registry["resources"]}
    assert not any(item.startswith("checkpoint_manifest.") for item in ids)
    assert not any(item.startswith("checkpoint_provenance.") for item in ids)


def test_exact_gate_rehearsal_counts_and_claim_states(tmp_path: Path) -> None:
    root = tmp_path / "rehearsal"
    root.mkdir()
    phases = [
        "train", "formal_cache_policy", "formal_controller", "formal_ablation",
        "formal_support", "formal_scalability",
    ]
    (root / "cell_state.jsonl").write_text(
        "".join(json.dumps({"phase": phase, "status": "committed"}) + "\n" for phase in phases),
        encoding="utf-8",
    )
    write_json(root / "checkpoint_candidates.json", [{"candidate": 1}])
    write_json(root / "dev_selection.json", {"selected": [{"selected": 1}]})
    write_json(
        root / "checkpoint_freeze.json",
        {
            "frozen_checkpoint_count": 3,
            "frozen_checkpoints": [
                {"capacity_label": capacity} for capacity in CAPACITY_MB
            ],
        },
    )
    (root / "training/a/checkpoints").mkdir(parents=True)
    (root / "training/a/checkpoints/latest.pt").write_bytes(b"x")
    for relative in (
        "formal_cache_policy/a/aggregate_summary.json",
        "formal_controller/a/aggregate_summary.json",
        "formal_ablation/a/support_provenance.json",
        "formal_support/a/support_provenance.json",
        "formal_scalability/a/support_provenance.json",
        "artifact_integrity_manifest.json",
    ):
        write_json(root / relative, {})
    rows_path = root / "formal_controller/a/benchmark_rows.csv"
    rows_path.write_text("window_id,agent\nw1,ppo\n", encoding="utf-8")
    write_json(
        root / "statistics/paired_statistics.json",
        {
            "rows": [
                {
                    "candidate_agent": "sa_ghmappo", "baseline_agent": "ppo",
                    "metric": "workflow_continuity_rate", "available_paired_count": 1,
                    "ci95_low": -0.1, "ci95_high": 0.2,
                }
            ]
        },
    )
    expected = {
        "committed_training_cells": 1,
        "candidate_checkpoints": 1,
        "latest_checkpoints": 1,
        "dev_candidate_evaluations": 1,
        "selections": 1,
        "frozen_checkpoints": 3,
        "frozen_checkpoints_by_capacity": {capacity: 1 for capacity in CAPACITY_MB},
        "cache_policy_cells": 1,
        "controller_cells": 1,
        "ablation_settings": 1,
        "support_settings": 1,
        "scalability_settings": 1,
        "primary_comparison_rows": 1,
        "formal_outer_window_clusters": 1,
    }
    write_json(
        root / "non_formal_rehearsal.json",
        {"formal": False, "performance_evidence": False, "expected_counts": expected},
    )
    gate = formal_gate(
        root,
        {"hashes": {"semantic_sha256": "1" * 64}},
        generated_registry_audit={"status": "pass", "registry_canonical_sha256": "2" * 64},
    )
    assert gate["passed"] is False
    assert gate["cell_ledger_validation_status"] == "fail"
    assert gate["claim_evidence_map"][0]["status"] == "mixed"
    assert gate["paper_claims_permitted"] is False
    write_json(root / "checkpoint_candidates.json", [])
    failed = formal_gate(
        root,
        {"hashes": {"semantic_sha256": "1" * 64}},
        generated_registry_audit={"status": "pass"},
    )
    assert failed["passed"] is False
    assert "candidate_checkpoints" in failed["exact_count_mismatches"]
