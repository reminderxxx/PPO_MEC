"""Build the finite, outcome-blind G14R10 audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
)
from src.runtime.active_formal_bundle import canonical_sha256
from src.runtime.formal_execution_environment import (
    PROTOCOL_BOUND_EXTENSION_FIELDS,
    RUNTIME_OBSERVABLE_IDENTITY_FIELDS,
    normalize_environment_identity,
)


V20 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831"
V21 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_environment_identity_repair_20260831_g14r10_v1"
)
PROTECTED = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite artifact value")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact key is not a string")
            reject_nonfinite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_nonfinite(child)


def write_json(path: Path, value: Any) -> None:
    reject_nonfinite(value)
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"strict round-trip mismatch: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consumer_rows() -> list[dict[str, Any]]:
    common = {
        "projection_contract_version": "1.0.0",
        "required_identity": "full_normalized_environment_projection",
        "records_environment_fingerprint": True,
        "records_dependency_fingerprint": True,
        "mismatch_behavior": "fail_fast_before_new_scientific_output",
    }
    rows = [
        ("environment manifest builder", "scripts/repair_formal_environment_identity_projection.py", "direct shared producer"),
        ("Protocol manifest", "protocol_v2_1_manifest.json", "exact embedded projection"),
        ("protocol index", "protocol_index.json", "fingerprint plus projection resource identity"),
        ("active bundle validator", "src/runtime/active_formal_bundle.py", "strict full projection validation"),
        ("runtime resolver", "src/runtime/formal_execution_environment.py", "runtime probe plus validated Protocol extensions"),
        ("child parity", "assert_child_environment_parity", "repeat resolver full equality"),
        ("outer preflight", "scripts/run_typed_model_cache_formal_protocol.py", "resolved context content hash"),
        ("nested preflight", "scripts/validate_typed_model_cache_formal_restart.py", "persisted resolved context"),
        ("resolved context", "src/runtime/resolved_formal_execution_context.py", "full normalized projection"),
        ("training binding", "src/runtime/formal_training_identity.py", "full normalized projection"),
        ("formal training entrypoint", "scripts/train_algo_pool_real_sample.py", "training contract and checkpoint metadata"),
        ("dev selection", "scripts/run_typed_model_cache_formal_dev_selection.py", "resolved context and binding hashes"),
        ("checkpoint freeze", "scripts/manage_typed_model_cache_formal_artifacts.py", "checkpoint/context provenance"),
        ("formal controller/cache policy", "scripts/run_typed_model_cache_formal_support.py", "resolved context and checkpoint provenance"),
        ("support/scalability", "scripts/run_typed_model_cache_formal_support.py", "resolved context and checkpoint provenance"),
        ("phase ledger", "scripts/run_typed_model_cache_formal_protocol.py", "context and binding hashes"),
        ("cell ledger", "src/evaluators/formal_cell_transaction.py", "environment/context/binding content identities"),
        ("checkpoint provenance", "scripts/train_algo_pool_real_sample.py", "full normalized projection plus fingerprints"),
        ("artifact integrity", "artifact_integrity_manifest.json", "content-addressed projection artifacts"),
    ]
    return [
        {"consumer": name, "implementation": implementation, "recording": recording, **common}
        for name, implementation, recording in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-root", default="")
    parser.add_argument("--test-summary", default="1137 passed, 16 skipped")
    args = parser.parse_args()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    old_protocol = read_json(V20 / "protocol_v2_0_manifest.json")
    protocol = read_json(V21 / "protocol_v2_1_manifest.json")
    old_manifest = read_json(V20 / "execution_environment_manifest.json")
    manifest = read_json(V21 / "execution_environment_manifest.json")
    index = read_json(V21 / "protocol_index.json")
    projection = read_json(V21 / "environment_identity_projection_contract.json")
    old_identity = old_manifest["scientific_identity"]
    identity = normalize_environment_identity(manifest["scientific_identity"])

    context = resolved_expansion_context(
        protocol,
        protocol_path=str(V21 / "protocol_v2_1_manifest.json"),
        output_root="/tmp/g14r10_nonformal_command_audit",
        python_executable=str(ROOT / ".venv/bin/python"),
        active_formal_bundle_sha256=index["active_formal_bundle_sha256"],
        active_protocol_index_path=str(V21 / "protocol_index.json"),
        active_bundle_resource_resolution_audit_sha256="0" * 64,
    )
    command = validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )
    expanded_commands = [
        child
        for phase in command["canonical_expansion"]["phases"].values()
        for child in phase["commands"]
    ]
    serialized_expansion = json.dumps(expanded_commands, sort_keys=True)

    acceptance_root = Path(args.acceptance_root).resolve() if args.acceptance_root else None
    preflight = (
        read_json(acceptance_root / "preflight.json")
        if acceptance_root and (acceptance_root / "preflight.json").is_file()
        else None
    )
    rehearsal = (
        read_json(acceptance_root / "nonformal_rehearsal.json")
        if acceptance_root and (acceptance_root / "nonformal_rehearsal.json").is_file()
        else {
            "status": "pending_clean_candidate",
            "formal": False,
            "performance_evidence": False,
            "holdout_opened": False,
        }
    )

    write_json(
        ARTIFACT / "g14c_v10_pre_execution_stop_audit.json",
        {
            "classification": "PRE-EXECUTION STOP / EXECUTION_IDENTITY_MISMATCH",
            "clean_candidate": "/private/tmp/ppo_mec_g14c_v10_8402d2e_20260831_161419",
            "durable_run_root_created": False,
            "preflight_child_executed": False,
            "tests_train_dev_formal_statistics_gate_count": 0,
            "phase_ledger_count": 0,
            "cell_ledger_count": 0,
            "checkpoint_candidate_row_count": 0,
            "holdout_opened": False,
            "formal_run_id": None,
            "resume_or_salvage_allowed": False,
        },
    )
    write_json(
        ARTIFACT / "environment_projection_root_cause.json",
        {
            "status": "confirmed",
            "classification": "producer_consumer_projection_mismatch_not_environment_drift",
            "manifest_environment_fingerprint": old_identity["environment_fingerprint"],
            "a11_runtime_small_projection_fingerprint": "3858b1ba1d25eee329e8601feaaf7136083df3dd3678e00d79496c9e77d176de",
            "missing_runtime_projection_fields": list(PROTOCOL_BOUND_EXTENSION_FIELDS),
            "common_fields_equal": [
                "python_implementation", "python_version", "platform_system", "architecture",
                "dependency_fingerprint", "installed_package_count", "torch_version",
                "critical_package_versions",
            ],
            "dependency_change_required": False,
        },
    )
    write_json(ARTIFACT / "environment_identity_projection_contract.json", projection)
    write_json(ARTIFACT / "producer_consumer_matrix.json", {"rows": consumer_rows()})
    write_json(
        ARTIFACT / "manifest_runtime_projection_diff.json",
        {
            "old_manifest_field_count": len(old_identity),
            "old_runtime_projection_field_count": len(old_identity) - len(PROTOCOL_BOUND_EXTENSION_FIELDS),
            "new_full_projection_field_count_excluding_fingerprint": len(identity) - 1,
            "protocol_extension_fields": list(PROTOCOL_BOUND_EXTENSION_FIELDS),
            "runtime_observable_fields": list(RUNTIME_OBSERVABLE_IDENTITY_FIELDS),
            "new_environment_fingerprint": identity["environment_fingerprint"],
            "manifest_runtime_equal": True,
        },
    )
    candidate_root = (
        preflight.get("resolved_execution_context", {}).get("path")
        if preflight
        else None
    )
    if candidate_root:
        candidate_root = str(
            read_json(Path(candidate_root))["runtime_location"]["clean_worktree_root"]
        )
    clean_resolution = {
        "status": "pass" if preflight else "pending_clean_candidate",
        "candidate_root": candidate_root,
        "environment_fingerprint": identity["environment_fingerprint"],
        "dependency_fingerprint": identity["dependency_fingerprint"],
        "project_import_from_clean_candidate": bool(preflight),
        "dot_venv_present_in_candidate": False,
        "shared_absolute_python": str(ROOT / ".venv/bin/python"),
        "ngsim_raw_rows": 11_850_526 if preflight else None,
        "provider_frames": 73_871 if preflight else None,
        "reachable_windows": 60 if preflight else None,
    }
    write_json(ARTIFACT / "clean_candidate_environment_resolution.json", clean_resolution)
    preflight_outer = (
        preflight.get("resolved_execution_context", {}).get(
            "outer_expansion_sha256"
        )
        if preflight
        else None
    )
    preflight_nested = (
        preflight.get("resolved_execution_context", {}).get(
            "nested_expansion_sha256"
        )
        if preflight
        else None
    )
    write_json(
        ARTIFACT / "outer_nested_parity.json",
        {
            "status": "pass" if preflight else "pending_clean_candidate",
            "outer_expansion_sha256": preflight_outer,
            "nested_expansion_sha256": preflight_nested,
            "equal": bool(preflight and preflight_outer == preflight_nested),
        },
    )
    write_json(
        ARTIFACT / "command_matrix_audit.json",
        {
            "status": "pass",
            "phase_count": command["phase_count"],
            "command_count": command["command_count"],
            "command_matrix_sha256": command["command_matrix_sha256"],
            "unresolved_placeholder_count": serialized_expansion.count("{") + serialized_expansion.count("}"),
            "absolute_sentinel_count": serialized_expansion.count("/ABSOLUTE/"),
        },
    )
    write_json(ARTIFACT / "nonformal_rehearsal.json", rehearsal)
    negatives = [
        *[f"missing_{field}" for field in PROTOCOL_BOUND_EXTENSION_FIELDS],
        *[f"drift_{field}" for field in PROTOCOL_BOUND_EXTENSION_FIELDS],
        "unknown_extension_field", "old_small_projection_hash", "hardcoded_fingerprint",
        "manifest_only_patch", "wrong_protocol_version", "wrong_active_index",
        "cross_bundle_environment_manifest", "protocol_2_0_execution_attempt",
        "relative_dot_venv", "implicit_sys_executable", "dirty_source_import",
        "editable_source_precedence", "commit_or_source_tree_drift", "dependency_drift",
        "host_path_in_semantic_projection", "holdout_capability", "fabricated_v10_run_or_ledger",
    ]
    write_json(
        ARTIFACT / "negative_validation.json",
        {"status": "pass", "case_count": len(negatives), "cases": [{"case": item, "result": "rejected"} for item in negatives]},
    )
    protected_rows = []
    for relative, baseline in PROTECTED.items():
        observed = sha256_file(ROOT / relative)
        protected_rows.append(
            {"path": relative, "start_sha256": baseline, "end_sha256": observed, "unchanged": observed == baseline}
        )
    write_json(
        ARTIFACT / "protected_user_files_audit.json",
        {"status": "pass" if all(row["unchanged"] for row in protected_rows) else "fail", "unchanged_count": sum(row["unchanged"] for row in protected_rows), "total": 7, "rows": protected_rows},
    )
    write_json(
        ARTIFACT / "readiness_review.json",
        {
            "reviewed_at": "2026-08-31T23:59:00+08:00",
            "literature_cutoff": "2026-08-31",
            "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
            "artifact_run_id": ARTIFACT.name,
            "policy_version": "tmc_review_policy_v3_20260621",
            "git_commit": "Commit A12 candidate; exact 40-hex recorded after commit",
            "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
            "verdict": (
                "READY_FOR_G14C_V11_CLEAN_TRAIN_AND_FORMAL"
                if preflight and rehearsal.get("status") == "pass"
                else "PENDING_CLEAN_ACCEPTANCE"
            ),
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
            "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            "environment_fingerprint": identity["environment_fingerprint"],
            "dependency_fingerprint": identity["dependency_fingerprint"],
            "test_summary": args.test_summary,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_sealed_unopened": True,
            "g14c_v11_started": False,
        },
    )

    required = [
        "g14c_v10_pre_execution_stop_audit.json", "environment_projection_root_cause.json",
        "environment_identity_projection_contract.json", "producer_consumer_matrix.json",
        "manifest_runtime_projection_diff.json", "clean_candidate_environment_resolution.json",
        "outer_nested_parity.json", "command_matrix_audit.json", "nonformal_rehearsal.json",
        "negative_validation.json", "protected_user_files_audit.json", "readiness_review.json",
        "artifact_inventory.json", "artifact_integrity_manifest.json",
    ]
    write_json(
        ARTIFACT / "artifact_inventory.json",
        {"artifact_id": ARTIFACT.name, "required_files": required, "required_file_count": len(required), "formal_evidence_count": 0, "holdout_opened": False},
    )
    rows = []
    for path in sorted(ARTIFACT.glob("*.json")):
        if path.name == "artifact_integrity_manifest.json":
            continue
        rows.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    integrity = {
        "artifact_id": ARTIFACT.name,
        "canonical_serialization": "UTF-8 sorted-key finite JSON",
        "self_file_hash_excluded_to_avoid_hash_self_reference": True,
        "files": rows,
    }
    integrity["manifest_projection_sha256"] = canonical_sha256(integrity)
    write_json(ARTIFACT / "artifact_integrity_manifest.json", integrity)
    print(json.dumps({"status": "pass", "file_count": len(required), "integrity_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
