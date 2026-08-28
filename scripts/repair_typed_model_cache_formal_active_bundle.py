"""Generate the pending G14R7A Protocol v1.8 active-bundle candidate.

This generator never writes a ready index.  If a ready v1.8 index already
exists it refuses to overwrite or downgrade it; only the evidence-gated
finalizer may transition the candidate from pending to ready.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.runtime.active_formal_bundle import (
    ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
    ACTIVE_PROTOCOL_ID,
    ACTIVE_PROTOCOL_VERSION,
    READY_STATUS,
    active_bundle_core_projection,
    build_resource_row,
    canonical_sha256,
)
from src.runtime.formal_execution_environment import (
    probe_python_environment,
    scientific_environment_identity,
)


RUN_ID = "typed_model_cache_formal_active_bundle_closure_20260827_g14r7a_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
V17_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
V18_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827"
V17_PROTOCOL = V17_ROOT / "protocol_v1_7_manifest.json"
V18_PROTOCOL = V18_ROOT / "protocol_v1_8_manifest.json"
INDEX_PATH = V18_ROOT / "protocol_index.json"
SCIENTIFIC_SHA256 = "f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8"
ORDER_SHA256 = "82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0"
DEPENDENCY_FINGERPRINT = "88963f6107e2042298da7c6920a5d0a2d50429c92634f3873a03d0ad8f4e2d00"
PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def current_resource(
    logical_id: str, role: str, filename: str, semantic_sha256: str | None = None
) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(V18_ROOT / filename).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic_sha256,
    )


def shared_resource(
    logical_id: str, role: str, relative_path: str, reason: str
) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=relative_path,
        version_scope="shared_historical_stable",
        shared_reason=reason,
    )


def build_protocol(shared_python: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = deepcopy(read_json(V17_PROTOCOL))
    probe = probe_python_environment(shared_python, clean_worktree_root=ROOT)
    environment_identity = scientific_environment_identity(
        probe,
        execution_commit=(
            "Commit A9; exact 40-hex clean HEAD == origin/main is observed and bound "
            "before every execution"
        ),
        source_tree_sha256=(
            "Commit A9 Git tree; runtime active-bundle gate verifies the exact clean HEAD "
            "without embedding a self-referential commit hash"
        ),
    )
    if environment_identity["dependency_fingerprint"] != DEPENDENCY_FINGERPRINT:
        raise ValueError("dependency fingerprint changed")
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        created_at=now(),
        status="frozen_pre_execution_active_bundle_gate_no_performance",
    )
    protocol["supersession"] = {
        **deepcopy(protocol["supersession"]),
        "supersedes_version": "1.7.0",
        "old_protocol_status": "audit_only_active_index_readiness_inconsistent",
        "old_protocol_semantic_sha256": read_json(V17_PROTOCOL)["hashes"]["semantic_sha256"],
        "formal_performance_observed": False,
        "scientific_fields_changed": False,
        "repair_scope": [
            "bind the unique active protocol index to every executable resource",
            "make Readiness and index status an atomic evidence-gated identity",
            "validate the complete active bundle before any run-root write",
            "propagate the active bundle hash into execution provenance",
        ],
    }
    protocol["identity"]["execution_git_commit_binding"] = (
        "exact observed clean 40-hex Git HEAD must equal origin/main at execution; "
        "the runtime binding records it"
    )
    protocol["formal_execution_environment_contract"]["scientific_identity"] = (
        environment_identity
    )
    protocol["formal_agent_order_contract"]["active_protocol_versions"] = ["1.8.0"]
    protocol["formal_agent_order_contract"]["historical_protocol_versions_audit_only"] = [
        "1.0.0",
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
    ]
    protocol["active_formal_bundle_contract"] = {
        "version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "unique_active_index": INDEX_PATH.relative_to(ROOT).as_posix(),
        "validator": "src.runtime.active_formal_bundle.validate_active_formal_bundle",
        "outer_runner_gate_before_any_run_root_write": True,
        "manual_protocol_or_environment_override_allowed": False,
        "pending_ready_coexistence_allowed": False,
        "ordinary_generator_can_write_ready": False,
        "ready_downgrade_allowed": False,
        "hash_graph": [
            "resource content hashes -> active_bundle_core_sha256",
            "core plus acceptance evidence -> Readiness v10 content hash",
            "ready index plus Readiness content hash -> active_formal_bundle_sha256",
        ],
        "hash_self_reference_allowed": False,
    }
    context = protocol["execution_contract"]["default_expansion_context"]
    context.update(
        protocol_path=V18_PROTOCOL.relative_to(ROOT).as_posix(),
        agent_scientific_config_path=(V18_ROOT / "agent_training_scientific_config.json")
        .relative_to(ROOT)
        .as_posix(),
        formal_agent_order_contract_path=(V18_ROOT / "formal_agent_order_contract.json")
        .relative_to(ROOT)
        .as_posix(),
        active_protocol_index_path=INDEX_PATH.relative_to(ROOT).as_posix(),
    )
    protocol["formal_training_execution_binding_contract"][
        "active_formal_bundle_hash_enters_binding_context_commands_and_checkpoint"
    ] = True
    protocol["resolved_formal_execution_context_contract"][
        "active_formal_bundle_hash_in_context"
    ] = True
    bindings = protocol["execution_contract"]["same_run_resume"]["bindings"]
    if "active_formal_bundle_sha256" not in bindings:
        bindings.append("active_formal_bundle_sha256")
    protocol["paper_claim_boundary"] = (
        "G14R7A closes the pre-execution active-bundle identity only. It performs no "
        "formal training, checkpoint production, performance evaluation, holdout access, "
        "G14C v8, G14D, or G15."
    )
    return attach_hashes(protocol), probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    shared_python = Path(args.python_executable).absolute()
    if not shared_python.is_file() or not os.access(shared_python, os.X_OK):
        raise FileNotFoundError(shared_python)
    if INDEX_PATH.is_file():
        existing = read_json(INDEX_PATH)
        if existing.get("status") == READY_STATUS:
            raise ValueError("ready active formal bundle is frozen; generator downgrade refused")

    V18_ROOT.mkdir(parents=True, exist_ok=True)
    protocol, probe = build_protocol(shared_python)
    scientific = read_json(V17_ROOT / "agent_training_scientific_config.json")
    order = read_json(V17_ROOT / "formal_agent_order_contract.json")
    if scientific.get("config_semantic_sha256") != SCIENTIFIC_SHA256:
        raise ValueError("Scientific Config identity drift")
    if order.get("semantic_sha256") != ORDER_SHA256:
        raise ValueError("Agent Order Contract identity drift")
    write_json(V18_PROTOCOL, protocol)
    write_json(V18_ROOT / "agent_training_scientific_config.json", scientific)
    write_json(V18_ROOT / "formal_agent_order_contract.json", order)
    environment = {
        "formal_execution_environment_contract_version": "1.0.0",
        "scientific_identity": protocol["formal_execution_environment_contract"][
            "scientific_identity"
        ],
        "runtime_location": {
            "resolved_python_absolute_path": str(shared_python),
            "virtual_environment_root": probe["sys_prefix"],
            "site_packages_paths": probe["site_packages"],
        },
        "runtime_location_is_scientific_identity": False,
    }
    write_json(V18_ROOT / "execution_environment_manifest.json", environment)
    write_json(
        V18_ROOT / "formal_training_execution_binding_contract.json",
        protocol["formal_training_execution_binding_contract"],
    )
    write_json(
        V18_ROOT / "resolved_execution_context_contract.json",
        protocol["resolved_formal_execution_context_contract"],
    )
    current = [
        current_resource(
            "protocol_manifest",
            "active Protocol manifest",
            "protocol_v1_8_manifest.json",
            protocol["hashes"]["semantic_sha256"],
        ),
        current_resource(
            "execution_environment_manifest",
            "active execution environment identity",
            "execution_environment_manifest.json",
        ),
        current_resource(
            "agent_training_scientific_config",
            "Scientific Config 2.0.0",
            "agent_training_scientific_config.json",
            SCIENTIFIC_SHA256,
        ),
        current_resource(
            "formal_agent_order_contract",
            "Formal Agent Order Contract 1.0.0",
            "formal_agent_order_contract.json",
            ORDER_SHA256,
        ),
        current_resource(
            "formal_training_execution_binding_schema",
            "formal execution binding schema 1.0.0",
            "formal_training_execution_binding_contract.json",
        ),
        current_resource(
            "resolved_execution_context_schema",
            "resolved context schema 2.0.0",
            "resolved_execution_context_contract.json",
        ),
    ]
    old_index = read_json(V17_ROOT / "protocol_index.json")
    shared: list[dict[str, Any]] = []
    reason = (
        "content-addressed resource is unchanged by G14R7A; its role and exact content hash "
        "are frozen in the v1.8 active index"
    )
    for logical_id, role, field in (
        ("portable_resource_registry", "portable resource registry", "portable_resource_registry"),
        ("split_companion", "formal split identity", "split_companion"),
        ("window_consumption_contract", "formal window consumption contract", "window_consumption_contract"),
        ("fairness_portable_identity_companion", "portable fairness identity", "fairness_portable_identity_companion"),
    ):
        shared.append(shared_resource(logical_id, role, old_index[field], reason))
    for group, role in (
        ("runtime_configs", "typed runtime config"),
        ("fairness_manifests", "formal fairness manifest"),
        ("dev_fairness_manifests", "dev fairness manifest"),
        ("support_fairness_manifests", "support fairness manifest"),
    ):
        for label, path in sorted(old_index[group].items()):
            shared.append(shared_resource(f"{group}.{label}", role, path, reason))
    resources = [*current, *shared]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "PENDING_G14R7A_CLEAN_ACCEPTANCE",
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": {
            "mode": "observed_clean_head_equal_origin_main",
            "exact_40_hex_recorded_in_execution_binding": True,
            "index_embeds_own_commit_hash": False,
            "self_reference_avoided": True,
        },
        "environment_identity": {
            "environment_fingerprint": environment["scientific_identity"][
                "environment_fingerprint"
            ],
            "dependency_fingerprint": DEPENDENCY_FINGERPRINT,
        },
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(
                protocol["execution_contract"]["command_templates"]
            ),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": resources,
        # Compatibility fields remain read-only aliases; the validator consumes
        # only the resource inventory above.
        "protocol_manifest": V18_PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "execution_environment_manifest": (
            V18_ROOT / "execution_environment_manifest.json"
        ).relative_to(ROOT).as_posix(),
        "agent_scientific_config": (
            V18_ROOT / "agent_training_scientific_config.json"
        ).relative_to(ROOT).as_posix(),
        "agent_scientific_config_semantic_sha256": SCIENTIFIC_SHA256,
        "formal_agent_order_contract": (
            V18_ROOT / "formal_agent_order_contract.json"
        ).relative_to(ROOT).as_posix(),
        "formal_agent_order_contract_version": "1.0.0",
        "formal_agent_order_contract_semantic_sha256": ORDER_SHA256,
        "formal_training_execution_binding_version": "1.0.0",
        "resolved_execution_context_contract_version": "2.0.0",
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )
    write_json(INDEX_PATH, index)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    from src.runtime.active_formal_bundle import sha256_file

    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes_start.json",
        {"files": {name: sha256_file(ROOT / name) for name in PROTECTED_FILES}},
    )
    write_json(
        ARTIFACT_ROOT / "v1_7_inconsistency_audit.json",
        {
            "status": "PRE_EXECUTION_BLOCKED_ACTIVE_BUNDLE_INCONSISTENT",
            "audited_at": now(),
            "v1_7_index_status": old_index["status"],
            "v1_7_index_environment_path": old_index["execution_environment_manifest"],
            "v1_7_actual_environment_path": (
                V17_ROOT / "execution_environment_manifest.json"
            ).relative_to(ROOT).as_posix(),
            "indexed_environment_fingerprint": read_json(
                ROOT / old_index["execution_environment_manifest"]
            )["scientific_identity"]["environment_fingerprint"],
            "actual_environment_fingerprint": read_json(
                V17_ROOT / "execution_environment_manifest.json"
            )["scientific_identity"]["environment_fingerprint"],
            "readiness_v9_verdict": READY_STATUS,
            "consistent": False,
        },
    )
    print(
        json.dumps(
            {
                "status": index["status"],
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "environment_fingerprint": environment["scientific_identity"][
                    "environment_fingerprint"
                ],
                "dependency_fingerprint": DEPENDENCY_FINGERPRINT,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
