"""Audit the native typed-cache transaction path against the v0.2 probe package.

This is a bounded, non-training audit adapter.  It constructs an in-memory
typed catalog from the package's declared object graph, then delegates every
cache request to ``VecWorkflowCoreEnv._apply_typed_cache_action``.  It does not
implement an alternative cache, eviction, or admission policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.model_catalog.adapter_catalog import (  # noqa: E402
    AdapterCatalog,
    TYPED_MODEL_CACHE_PROFILE_ID,
)
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import (  # noqa: E402
    ControlAction,
    RSUState,
    WorkflowGraphState,
    WorkflowNode,
)


EXPECTED_PROBE_SHA256 = "c0b3325874e7427629af8ed36720ed75db5b292c9658d7f3defd39a9deb44bbe"
AUDITED_NATIVE_COMMIT = "73051ab264aa868e83f2e011b5ced26968eef74b"
MIB_BYTES = 1_048_576
CAPACITY_BYTES = 136 * MIB_BYTES
CAPACITY_NATIVE_MB = CAPACITY_BYTES / MIB_BYTES
AUDIT_RUN_ID = "native_typed_cache_request_audit_20260930_v1"
POLICY_VERSION = "tmc_review_policy_v3_20260621"
TARGET_VENUE = "IEEE Transactions on Mobile Computing (TMC)"

BASE_SIZE_BYTES = {"b0": 96 * MIB_BYTES, "b1": 128 * MIB_BYTES}
ADAPTER_IDS = [f"b{base_index}.a{adapter_index}" for base_index in range(2) for adapter_index in range(3)]
ADAPTER_SIZE_BYTES = 8 * MIB_BYTES

WORKLOADS = {
    "blocked": ["b0.a0", "b0.a1", "b0.a2", "b1.a0", "b1.a1", "b1.a2"] * 12,
    "interleaved": ["b0.a0", "b1.a0", "b0.a1", "b1.a1", "b0.a2", "b1.a2"] * 12,
}
REFERENCE_WORKLOAD_IDS = {
    "blocked": "two_families_blocked",
    "interleaved": "two_families_interleaved",
}
# The capture requested by the audit is intentionally executed first.
CONFIG_EXECUTION_ORDER = [
    ("interleaved", True),
    ("blocked", True),
    ("blocked", False),
    ("interleaved", False),
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _zip_member_name(names: Iterable[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ZIP member ending with {suffix!r}, got {matches}")
    return matches[0]


def verify_and_load_probe(zip_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_sha256 = _sha256_file(zip_path)
    if actual_sha256 != EXPECTED_PROBE_SHA256:
        raise ValueError(f"probe ZIP SHA-256 mismatch: {actual_sha256} != {EXPECTED_PROBE_SHA256}")

    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"probe ZIP CRC failure: {bad_member}")
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        manifest_name = _zip_member_name(names, "/PACKAGE_MANIFEST.json")
        manifest = json.loads(archive.read(manifest_name))
        manifest_root = manifest_name.rsplit("/", 1)[0]
        checked_files = []
        for relative_name, expected_hash in sorted(manifest["files"].items()):
            member_name = f"{manifest_root}/{relative_name}"
            payload = archive.read(member_name)
            actual_hash = _sha256_bytes(payload)
            if actual_hash != expected_hash:
                raise ValueError(f"package member SHA-256 mismatch: {relative_name}")
            checked_files.append({"path": relative_name, "sha256": actual_hash, "size_bytes": len(payload)})

        results_name = _zip_member_name(names, "/results/run_002/cache_results.jsonl")
        ledger_name = _zip_member_name(names, "/results/run_002/cache_event_ledger.jsonl")
        workloads_name = _zip_member_name(names, "/results/run_002/cache_workloads.jsonl")
        catalog_name = _zip_member_name(names, "/results/run_002/catalog.json")
        reference_results = [json.loads(line) for line in archive.read(results_name).decode("utf-8").splitlines()]
        reference_ledger = [json.loads(line) for line in archive.read(ledger_name).decode("utf-8").splitlines()]
        reference_workloads = [json.loads(line) for line in archive.read(workloads_name).decode("utf-8").splitlines()]
        source_catalog = json.loads(archive.read(catalog_name))

    expected_catalog = {
        "b0": BASE_SIZE_BYTES["b0"],
        "b1": BASE_SIZE_BYTES["b1"],
        **{adapter_id: ADAPTER_SIZE_BYTES for adapter_id in ADAPTER_IDS},
    }
    observed_catalog = {item["id"]: int(item["size_bytes"]) for item in source_catalog["objects"]}
    if observed_catalog != expected_catalog:
        raise ValueError("probe catalog does not match the frozen two-base/six-adapter design")

    workload_by_id = {row["workload_id"]: row for row in reference_workloads}
    for label, expected_requests in WORKLOADS.items():
        observed = workload_by_id[REFERENCE_WORKLOAD_IDS[label]]["requests"]
        if observed != expected_requests:
            raise ValueError(f"probe request order mismatch for {label}")

    references: dict[str, Any] = {}
    for workload_label, sharing_enabled in CONFIG_EXECUTION_ORDER:
        workload_id = REFERENCE_WORKLOAD_IDS[workload_label]
        summary = next(
            row
            for row in reference_results
            if row["workload_id"] == workload_id
            and int(row["capacity_bytes"]) == CAPACITY_BYTES
            and row["eviction_policy"] == "lru"
            and bool(row["sharing_enabled"]) is sharing_enabled
        )
        events = [row for row in reference_ledger if row["run_id"] == summary["run_id"]]
        if len(events) != 72 or [row["adapter_id"] for row in events] != WORKLOADS[workload_label]:
            raise ValueError(f"reference ledger mismatch for {workload_label}/sharing={sharing_enabled}")
        references[_config_id(workload_label, sharing_enabled)] = {
            "summary": summary,
            "events": events,
        }

    integrity = {
        "input_zip_path": str(zip_path.resolve()),
        "input_zip_size_bytes": zip_path.stat().st_size,
        "input_zip_sha256": actual_sha256,
        "expected_zip_sha256": EXPECTED_PROBE_SHA256,
        "zip_crc_status": "pass",
        "zip_file_count": len(names),
        "package_manifest_code_version": manifest.get("code_version"),
        "package_manifest_checked_file_count": len(checked_files),
        "package_manifest_member_hash_status": "pass",
        "checked_files": checked_files,
        "partial_audit_zip": None,
        "partial_audit_zip_required": False,
    }
    return references, integrity


def _config_id(workload_label: str, sharing_enabled: bool) -> str:
    return f"{workload_label}__sharing_{'on' if sharing_enabled else 'off'}"


def _base_identity_for_adapter(adapter_id: str, sharing_enabled: bool) -> str:
    family = adapter_id.split(".", 1)[0]
    return family if sharing_enabled else f"{family}::private::{adapter_id}"


def _native_base_object_id(base_identity: str) -> str:
    return f"base:{base_identity}"


def _native_adapter_object_id(adapter_id: str) -> str:
    return f"adapter:{adapter_id}"


def _reference_identity(native_object_id: str) -> str:
    if native_object_id.startswith("base:"):
        return native_object_id[len("base:") :]
    if native_object_id.startswith("adapter:"):
        return native_object_id[len("adapter:") :]
    return native_object_id


def _typed_object(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(payload)
    return payload


def build_audit_catalog(*, sharing_enabled: bool, input_sha256: str = EXPECTED_PROBE_SHA256) -> AdapterCatalog:
    base_identities = sorted({_base_identity_for_adapter(adapter_id, sharing_enabled) for adapter_id in ADAPTER_IDS})
    vehicle_base_models = []
    typed_objects = []
    compatibility_map: dict[str, list[str]] = {}

    for base_identity in base_identities:
        source_family = base_identity.split("::", 1)[0]
        family = f"probe_family:{base_identity}"
        size_mb = BASE_SIZE_BYTES[source_family] / MIB_BYTES
        vehicle_base_models.append(
            {"base_model_id": base_identity, "family": family, "memory_mb": size_mb}
        )
        compatibility_map[base_identity] = []
        typed_objects.append(
            _typed_object(
                {
                    "object_id": _native_base_object_id(base_identity),
                    "object_type": "base_model",
                    "version": "probe_v0_2",
                    "resident_size_mb": size_mb,
                    "transfer_size_mb": size_mb,
                    "source": "vec_mechanism_probe_v0_2",
                    "provenance": {
                        "input_zip_sha256": input_sha256,
                        "source_identity": source_family,
                        "source_size_bytes": BASE_SIZE_BYTES[source_family],
                        "bytes_per_native_mb_mapping": MIB_BYTES,
                    },
                    "base_model_family": family,
                    "base_model_id": base_identity,
                    "required_base_model_id": None,
                    "adapter_id": None,
                    "workflow_identity": None,
                    "shareability_scope": (
                        "all_compatible_adapters_at_rsu" if sharing_enabled else "private_adapter_copy"
                    ),
                    "mutability": "immutable",
                    "persistence": "audit_episode_resident",
                    "evictability": "evictable",
                    "migration_semantics": "independent_model_weight_transfer",
                    "dependency_ids": [],
                    "dataset_profile_source": "synthetic_probe_package",
                    "license_status": "synthetic_not_applicable",
                    "formal_use_status": "non_formal_native_audit_only",
                    "availability": "available",
                    "counts_toward_capacity": True,
                }
            )
        )

    for adapter_id in ADAPTER_IDS:
        base_identity = _base_identity_for_adapter(adapter_id, sharing_enabled)
        family = f"probe_family:{base_identity}"
        compatibility_map[base_identity].append(adapter_id)
        typed_objects.append(
            _typed_object(
                {
                    "object_id": _native_adapter_object_id(adapter_id),
                    "object_type": "adapter",
                    "version": "probe_v0_2",
                    "resident_size_mb": ADAPTER_SIZE_BYTES / MIB_BYTES,
                    "transfer_size_mb": ADAPTER_SIZE_BYTES / MIB_BYTES,
                    "source": "vec_mechanism_probe_v0_2",
                    "provenance": {
                        "input_zip_sha256": input_sha256,
                        "source_identity": adapter_id,
                        "source_size_bytes": ADAPTER_SIZE_BYTES,
                        "bytes_per_native_mb_mapping": MIB_BYTES,
                    },
                    "base_model_family": family,
                    "base_model_id": None,
                    "required_base_model_id": base_identity,
                    "adapter_id": adapter_id,
                    "workflow_identity": None,
                    "shareability_scope": "rsu_workflows",
                    "mutability": "immutable",
                    "persistence": "audit_episode_resident",
                    "evictability": "evictable",
                    "migration_semantics": "independent_adapter_transfer",
                    "dependency_ids": [_native_base_object_id(base_identity)],
                    "dataset_profile_source": "synthetic_probe_package",
                    "license_status": "synthetic_not_applicable",
                    "formal_use_status": "non_formal_native_audit_only",
                    "availability": "available",
                    "counts_toward_capacity": True,
                }
            )
        )

    raw = {
        "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
        "typed_model_cache_contract_version": "1.0.0",
        "vehicle_base_models": vehicle_base_models,
        "rsu_adapter_caches": [{"rsu_id": "rsu_a", "cached_adapter_ids": []}],
        "adapter_state_bundles": [],
        "cache_objects": [
            {
                "object_id": f"legacy:{adapter_id}",
                "adapter_id": adapter_id,
                "size_mb": ADAPTER_SIZE_BYTES / MIB_BYTES,
                "source": "probe_audit_legacy_view",
            }
            for adapter_id in ADAPTER_IDS
        ],
        "typed_cache_objects": typed_objects,
        "rsu_typed_cache_profiles": [{"rsu_id": "rsu_a", "resident_object_ids": []}],
        "compatibility_map": compatibility_map,
        "kv_prefix_enabled": False,
        "vehicle_adapter_residency_enabled": False,
        "model_cache_datasets": [],
    }
    return AdapterCatalog.from_dict(raw)


def _build_env(catalog: AdapterCatalog) -> VecWorkflowCoreEnv:
    bootstrap_base_id = str(catalog.get_typed_adapter("b0.a0").required_base_model_id)
    bootstrap_node = WorkflowNode(
        node_id="bootstrap",
        node_name="native typed-cache audit bootstrap",
        required_base_model=bootstrap_base_id,
        required_adapter="b0.a0",
        input_size=0,
        output_size=0,
    )
    workflow = WorkflowGraphState(
        workflow_id="native_typed_cache_audit",
        nodes=[bootstrap_node],
        edges=[],
        execution_order=["bootstrap"],
        current_node_id="bootstrap",
    )
    mobility = ReplayProvider(
        trajectory_frames=[
            {
                "time_index": step,
                "vehicles": [
                    {
                        "vehicle_id": "audit_vehicle",
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "speed": 0.0,
                        "base_model_id": bootstrap_base_id,
                        "active_workflow_id": workflow.workflow_id,
                    }
                ],
            }
            for step in range(2)
        ]
    )
    env = VecWorkflowCoreEnv(
        mobility_provider=mobility,
        workflow_state=workflow,
        adapter_catalog=catalog,
        rsu_states=[
            RSUState(
                rsu_id="rsu_a",
                position_x=0.0,
                position_y=0.0,
                coverage_radius=1_000_000.0,
            )
        ],
        max_steps=72,
        cache_capacity_profile={
            "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
            "enabled": True,
            "unit": "mb",
            "capacity_mb": CAPACITY_NATIVE_MB,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "telemetry_enabled": True,
        },
    )
    env.reset()
    return env


def _control() -> ControlAction:
    return ControlAction(
        cache_action={"operation": "cache", "rsu_id": "rsu_a", "strategy": "native_probe_replay"},
        offload_action={"mode": "rsu", "target_rsu_id": "rsu_a"},
        migration_action={"mode": "keep"},
        metadata={"action_id": 1, "action_name": "cache_current"},
    )


def _readiness(env: VecWorkflowCoreEnv, adapter_id: str, base_identity: str) -> dict[str, Any]:
    node = WorkflowNode(
        node_id="audit_request",
        node_name="native typed-cache audit request",
        required_base_model=base_identity,
        required_adapter=adapter_id,
        input_size=0,
        output_size=0,
    )
    return env._typed_service_readiness(
        current_node=node,
        primary_vehicle=None,
        offload_mode="rsu",
        service_rsu_id="rsu_a",
        state_required=False,
        state_ready=True,
    )


def _resident_snapshot(env: VecWorkflowCoreEnv) -> dict[str, Any]:
    object_ids = list(env._typed_resident_object_ids["rsu_a"])
    rows = [env._typed_object_row(object_id) for object_id in object_ids]
    native_mb = sum(float(row["resident_size_mb"]) for row in rows)
    return {
        "native_object_ids": object_ids,
        "reference_object_ids": [_reference_identity(object_id) for object_id in object_ids],
        "objects": rows,
        "used_native_mb": native_mb,
        "used_bytes_under_explicit_mapping": int(round(native_mb * MIB_BYTES)),
    }


def _bytes_by_type(native_mb_by_type: dict[str, Any]) -> dict[str, int]:
    return {
        object_type: int(round(float(value) * MIB_BYTES))
        for object_type, value in sorted(native_mb_by_type.items())
    }


def _policy_rsu_state(policy_state: dict[str, Any]) -> dict[str, Any]:
    return dict(policy_state.get("rsus", {}).get("rsu_a", {}))


def _diff(reference: Any, native: Any) -> bool:
    return reference != native


def execute_config(
    *,
    workload_label: str,
    sharing_enabled: bool,
    reference: dict[str, Any],
    request_limit: int | None = None,
) -> list[dict[str, Any]]:
    catalog = build_audit_catalog(sharing_enabled=sharing_enabled)
    env = _build_env(catalog)
    output = []
    requests = WORKLOADS[workload_label]
    reference_events = reference["events"]
    if request_limit is not None:
        requests = requests[:request_limit]
        reference_events = reference_events[:request_limit]

    for request_index, (adapter_id, reference_event) in enumerate(zip(requests, reference_events)):
        env._episode_steps = request_index
        base_identity = _base_identity_for_adapter(adapter_id, sharing_enabled)
        residents_before = _resident_snapshot(env)
        policy_before = env.export_cache_eviction_policy_state()
        pre_readiness = _readiness(env, adapter_id, base_identity)
        placement = env.adapter_catalog.resolve_typed_placement_plan(
            adapter_id=adapter_id,
            resident_object_ids=list(env._typed_resident_object_ids["rsu_a"]),
        )
        eligible_victims = list(env._typed_evictable_residents("rsu_a"))
        required_free_native_mb = max(
            residents_before["used_native_mb"] + float(placement.requested_bundle_mb) - CAPACITY_NATIVE_MB,
            0.0,
        )

        native_result = env._apply_typed_cache_action(
            control=_control(),
            primary_vehicle=None,
            current_node_id=f"request_{request_index:03d}",
            required_adapter=adapter_id,
        )
        post_readiness = _readiness(env, adapter_id, base_identity)
        native_result["service_readiness"] = post_readiness
        request_node = WorkflowNode(
            node_id=f"request_{request_index:03d}",
            node_name="native typed-cache audit request",
            required_base_model=base_identity,
            required_adapter=adapter_id,
            input_size=0,
            output_size=0,
        )
        native_cache_event = env._build_cache_event(
            current_node=request_node,
            primary_vehicle=None,
            request_rsu_id="rsu_a",
            selected_target_rsu_id="rsu_a",
            predicted_next_rsu_id=None,
            predicted_handoff_target_rsu_id=None,
            cache_hit=bool(post_readiness["full_service_ready"]),
            stall_occurred=not bool(post_readiness["full_service_ready"]),
            control=_control(),
            cache_result=native_result,
            handoff_count=0,
            migration_prepare_requested=False,
            migration_prepare_realized=False,
        ).to_dict()
        residents_after = _resident_snapshot(env)
        policy_after = env.export_cache_eviction_policy_state()

        # Invoke the native invariant checks after every real transaction attempt.
        env._validate_typed_resident_invariants("rsu_a")
        if residents_after["used_native_mb"] > CAPACITY_NATIVE_MB + 1.0e-9:
            raise RuntimeError("native request audit exceeded capacity")

        admitted_rows = list(native_result.get("admitted_typed_objects") or [])
        evicted_rows = list(native_result.get("evicted_typed_objects") or [])
        resident_conservation_lhs = (
            residents_before["used_native_mb"]
            - sum(float(row["resident_size_mb"]) for row in evicted_rows)
            + sum(float(row["resident_size_mb"]) for row in admitted_rows)
        )
        resident_conservation_ok = math.isclose(
            resident_conservation_lhs,
            residents_after["used_native_mb"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        if not resident_conservation_ok:
            raise RuntimeError("native resident-byte conservation failure")

        transaction_status = str(native_result.get("atomic_transaction_status"))
        admission_rejected = transaction_status.startswith("rejected") or transaction_status == "rolled_back_no_mutation"
        rejected_state_unchanged = None
        if admission_rejected:
            rejected_state_unchanged = (
                residents_before == residents_after and policy_before == policy_after
            )
            if not rejected_state_unchanged:
                raise RuntimeError("native rejection mutated resident or policy state")

        native_evicted_reference_ids = [
            _reference_identity(object_id) for object_id in native_result.get("evicted_object_ids") or []
        ]
        transfer_bytes_by_type = _bytes_by_type(dict(native_result.get("transfer_mb_by_type") or {}))
        native_loaded_bytes = sum(transfer_bytes_by_type.values())
        comparison_fields = {
            "pre_base_hit": {
                "reference": bool(reference_event["base_hit"]),
                "native": bool(pre_readiness["base_ready"]),
            },
            "pre_adapter_hit": {
                "reference": bool(reference_event["adapter_hit"]),
                "native": bool(pre_readiness["adapter_ready"]),
            },
            "pre_full_hit": {
                "reference": bool(reference_event["full_hit"]),
                "native": bool(pre_readiness["full_service_ready"]),
            },
            "resident_before": {
                "reference": list(reference_event["resident_before"]),
                "native": residents_before["reference_object_ids"],
            },
            "resident_after": {
                "reference": list(reference_event["resident_after"]),
                "native": residents_after["reference_object_ids"],
            },
            "evicted_objects": {
                "reference": list(reference_event["evicted_ids"]),
                "native": native_evicted_reference_ids,
            },
            "loaded_base_bytes": {
                "reference": int(reference_event["loaded_base_bytes"]),
                "native": int(transfer_bytes_by_type.get("base_model", 0)),
            },
            "loaded_adapter_bytes": {
                "reference": int(reference_event["loaded_adapter_bytes"]),
                "native": int(transfer_bytes_by_type.get("adapter", 0)),
            },
            "loaded_bytes": {
                "reference": int(reference_event["loaded_bytes"]),
                "native": native_loaded_bytes,
            },
            "resident_bytes_after": {
                "reference": int(reference_event["resident_bytes_after"]),
                "native": residents_after["used_bytes_under_explicit_mapping"],
            },
        }
        differing_fields = [
            name for name, values in comparison_fields.items() if _diff(values["reference"], values["native"])
        ]

        output.append(
            {
                "audit_sequence_index": None,
                "config_id": _config_id(workload_label, sharing_enabled),
                "workload": workload_label,
                "sharing_enabled": sharing_enabled,
                "request_index": request_index,
                "adapter_id": adapter_id,
                "reference": {
                    **reference_event,
                    "admission_status": None,
                    "admission_rejection_reason": None,
                    "post_action_service_success": None,
                    "origin_execution": None,
                },
                "native": {
                    "required_base_identity": base_identity,
                    "dependency_bundle": asdict(placement),
                    "missing_native_object_ids": list(placement.missing_object_ids),
                    "resident_before": residents_before,
                    "resident_after": residents_after,
                    "capacity": {
                        "source_capacity_bytes": CAPACITY_BYTES,
                        "source_capacity_mib": CAPACITY_BYTES / MIB_BYTES,
                        "native_capacity_field": "capacity_mb",
                        "native_capacity_value": CAPACITY_NATIVE_MB,
                        "bytes_per_native_mb_mapping": MIB_BYTES,
                        "required_free_native_mb": required_free_native_mb,
                        "required_free_bytes_under_explicit_mapping": int(round(required_free_native_mb * MIB_BYTES)),
                    },
                    "eligible_victim_native_object_ids": eligible_victims,
                    "eligible_victim_reference_object_ids": [_reference_identity(item) for item in eligible_victims],
                    "actual_eviction_plan": native_result.get("eviction_plan"),
                    "transaction_status": transaction_status,
                    "admission_requested": bool(native_result.get("requested")),
                    "admission_added_adapter": bool(native_result.get("added_new_adapter")),
                    "admission_rejected": admission_rejected,
                    "admission_rejection_reason": native_result.get("capacity_rejection_reason"),
                    "admitted_typed_objects": admitted_rows,
                    "evicted_typed_objects": evicted_rows,
                    "transfer_native_mb_by_type": dict(native_result.get("transfer_mb_by_type") or {}),
                    "transfer_bytes_by_type_under_explicit_mapping": transfer_bytes_by_type,
                    "pre_action_readiness": pre_readiness,
                    "post_action_readiness": post_readiness,
                    "cache_event": native_cache_event,
                    "service_success": bool(native_cache_event["service_success"]),
                    "service_failure": bool(native_cache_event["stall_occurred"]),
                    "origin_execution": False,
                    "origin_execution_reason": "not_configured_in_bounded_rsu_native_transaction_audit",
                    "policy_state_before": _policy_rsu_state(policy_before),
                    "policy_state_after": _policy_rsu_state(policy_after),
                    "rejected_state_unchanged": rejected_state_unchanged,
                    "resident_byte_conservation_ok": resident_conservation_ok,
                    "orphan_count": int(native_result.get("orphan_count", 0)),
                },
                "comparison": {
                    "aligned_fields": comparison_fields,
                    "differing_fields": differing_fields,
                    "matches_all_aligned_fields": not differing_fields,
                    "reference_unobserved_fields_kept_null": [
                        "admission_status",
                        "admission_rejection_reason",
                        "post_action_service_success",
                        "origin_execution",
                    ],
                },
            }
        )
    return output


def _summarize_config(rows: list[dict[str, Any]], reference_summary: dict[str, Any]) -> dict[str, Any]:
    native_rows = [row["native"] for row in rows]
    status_counts: dict[str, int] = {}
    for native in native_rows:
        status = native["transaction_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    transfer_bytes_by_type: dict[str, int] = {}
    for native in native_rows:
        for object_type, value in native["transfer_bytes_by_type_under_explicit_mapping"].items():
            transfer_bytes_by_type[object_type] = transfer_bytes_by_type.get(object_type, 0) + int(value)
    return {
        "config_id": rows[0]["config_id"],
        "workload": rows[0]["workload"],
        "sharing_enabled": rows[0]["sharing_enabled"],
        "request_count": len(rows),
        "reference": {
            "request_count": int(reference_summary["n_requests"]),
            "pre_full_hit_count": int(reference_summary["n_full_hits"]),
            "pre_base_hit_count": int(reference_summary["n_base_hits"]),
            "pre_base_hit_adapter_miss_count": int(reference_summary["n_base_hit_adapter_miss"]),
            "loaded_bytes": int(reference_summary["loaded_bytes"]),
            "loaded_base_bytes": int(reference_summary["loaded_base_bytes"]),
            "loaded_adapter_bytes": int(reference_summary["loaded_adapter_bytes"]),
            "evicted_object_count": int(reference_summary["evicted_objects"]),
            "admission_rejection_count": None,
            "post_action_service_success_count": None,
            "service_failure_count": None,
            "origin_execution_count": None,
        },
        "native": {
            "request_count": len(rows),
            "admission_requested_count": sum(row["admission_requested"] for row in native_rows),
            "transaction_status_counts": dict(sorted(status_counts.items())),
            "admission_rejection_count": sum(row["admission_rejected"] for row in native_rows),
            "pre_full_hit_count": sum(row["pre_action_readiness"]["full_service_ready"] for row in native_rows),
            "pre_base_hit_count": sum(row["pre_action_readiness"]["base_ready"] for row in native_rows),
            "pre_base_hit_adapter_miss_count": sum(
                row["pre_action_readiness"]["base_ready"] and not row["pre_action_readiness"]["adapter_ready"]
                for row in native_rows
            ),
            "post_action_service_success_count": sum(row["service_success"] for row in native_rows),
            "service_failure_count": sum(row["service_failure"] for row in native_rows),
            "origin_execution_count": sum(row["origin_execution"] for row in native_rows),
            "transfer_bytes_by_type_under_explicit_mapping": dict(sorted(transfer_bytes_by_type.items())),
            "total_transfer_bytes_under_explicit_mapping": sum(transfer_bytes_by_type.values()),
            "evicted_object_count": sum(len(row["evicted_typed_objects"]) for row in native_rows),
            "capacity_invariant_all_requests": all(
                row["resident_after"]["used_native_mb"] <= CAPACITY_NATIVE_MB + 1.0e-9 for row in native_rows
            ),
            "dependency_invariant_all_requests": all(row["orphan_count"] == 0 for row in native_rows),
            "resident_byte_conservation_all_requests": all(row["resident_byte_conservation_ok"] for row in native_rows),
            "rejection_state_unchanged_all_rejections": all(
                row["rejected_state_unchanged"] is True for row in native_rows if row["admission_rejected"]
            ),
        },
        "request_rows_with_any_aligned_difference": sum(
            not row["comparison"]["matches_all_aligned_fields"] for row in rows
        ),
    }


def _mapping_contract() -> dict[str, Any]:
    shared_map = {
        adapter_id: {
            "adapter_object_id": _native_adapter_object_id(adapter_id),
            "base_identity": _base_identity_for_adapter(adapter_id, True),
            "base_object_id": _native_base_object_id(_base_identity_for_adapter(adapter_id, True)),
        }
        for adapter_id in ADAPTER_IDS
    }
    unshared_map = {
        adapter_id: {
            "adapter_object_id": _native_adapter_object_id(adapter_id),
            "independent_base_copy_identity": _base_identity_for_adapter(adapter_id, False),
            "independent_base_copy_object_id": _native_base_object_id(_base_identity_for_adapter(adapter_id, False)),
        }
        for adapter_id in ADAPTER_IDS
    }
    return {
        "contract_id": "native_mapping_contract_v1_20260930",
        "audited_native_commit": AUDITED_NATIVE_COMMIT,
        "input_probe_sha256": EXPECTED_PROBE_SHA256,
        "native_consumer": "src.envs.core.vec_workflow_core_env.VecWorkflowCoreEnv._apply_typed_cache_action",
        "native_eviction_policy": "src.envs.core.cache_eviction.LRUEvictionPolicy/1.0.0",
        "native_readiness_consumer": "VecWorkflowCoreEnv._typed_service_readiness",
        "simulation_reimplementation": False,
        "capacity": {
            "source_value": 136,
            "source_unit": "MiB",
            "source_bytes": CAPACITY_BYTES,
            "native_field": "capacity_mb",
            "native_value": CAPACITY_NATIVE_MB,
            "observed_native_semantics": "floating-point numeric values are summed and compared directly; no byte conversion occurs in the native runtime",
            "mapping": "source bytes / 1,048,576 -> native numeric mb value",
            "bytes_per_native_mb_mapping": MIB_BYTES,
            "semantic_warning": "This is an explicit audit mapping. The native field name alone does not establish SI MB or MiB semantics.",
        },
        "object_sizes": {
            "b0": {"source_bytes": BASE_SIZE_BYTES["b0"], "source_mib": 96, "native_resident_size_mb": 96.0},
            "b1": {"source_bytes": BASE_SIZE_BYTES["b1"], "source_mib": 128, "native_resident_size_mb": 128.0},
            "each_adapter": {"source_bytes": ADAPTER_SIZE_BYTES, "source_mib": 8, "native_resident_size_mb": 8.0},
        },
        "initial_cache": {"rsu_a": [], "empty": True},
        "sharing_on_identity_map": shared_map,
        "sharing_off_identity_map": unshared_map,
        "sharing_off_semantics": "Each adapter has one distinct base-model identity and one distinct base object; no base object is shared across adapters.",
        "request_timing": "serial audit order only; request_index is assigned to native episode-step LRU time",
        "service_scope": "single rsu_a; no origin/cloud fallback is configured",
        "fixed_configs": [
            {"workload": workload, "sharing_enabled": sharing, "policy": "lru", "capacity_mib": 136, "request_count": 72}
            for workload, sharing in [("blocked", True), ("blocked", False), ("interleaved", True), ("interleaved", False)]
        ],
    }


def _first_difference(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row["comparison"]["differing_fields"]:
            return {
                "observation_order": "sharing-on/interleaved was executed first by frozen audit order",
                "audit_sequence_index": row["audit_sequence_index"],
                "config_id": row["config_id"],
                "request_index": row["request_index"],
                "adapter_id": row["adapter_id"],
                "differing_fields": row["comparison"]["differing_fields"],
                "reference_observed": {
                    "resident_before": row["reference"]["resident_before"],
                    "evicted_ids": row["reference"]["evicted_ids"],
                    "loaded_ids": row["reference"]["loaded_ids"],
                    "loaded_bytes": row["reference"]["loaded_bytes"],
                    "resident_after": row["reference"]["resident_after"],
                },
                "native_observed": {
                    "resident_before": row["native"]["resident_before"],
                    "dependency_bundle": row["native"]["dependency_bundle"],
                    "missing_native_object_ids": row["native"]["missing_native_object_ids"],
                    "eligible_victim_native_object_ids": row["native"]["eligible_victim_native_object_ids"],
                    "actual_eviction_plan": row["native"]["actual_eviction_plan"],
                    "required_free_bytes_under_explicit_mapping": row["native"]["capacity"]["required_free_bytes_under_explicit_mapping"],
                    "transaction_status": row["native"]["transaction_status"],
                    "admission_rejection_reason": row["native"]["admission_rejection_reason"],
                    "resident_after": row["native"]["resident_after"],
                    "policy_state_before": row["native"]["policy_state_before"],
                    "policy_state_after": row["native"]["policy_state_after"],
                    "post_action_service_success": row["native"]["service_success"],
                    "origin_execution": row["native"]["origin_execution"],
                },
                "interpretation": "The reference evicts an adapter plus its base dependency closure. The native one-shot eligible set excludes a base while a resident adapter depends on it, so the 8 MiB adapter-only plan cannot release the required 104 MiB and the atomic admission is rejected without state mutation.",
                "classification": "independent_design_semantics_issue_no_mechanism_fix_in_this_audit",
            }
    return None


def _eviction_comparison(first_difference: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "comparison_id": "reference_vs_native_dependency_safe_lru_v1",
        "reference_feasible_set": {
            "observed_source": "vec_mechanism_probe_v0_2/results/run_002/cache_event_ledger.jsonl",
            "semantics": "A base candidate expands to the base plus all resident dependent adapters; an adapter and its now-unneeded base may both be removed within the request loop.",
            "lru_key": "(last_touch_clock, adapter_before_base_tie_class, object_id)",
            "base_touch": "The reference refreshes the base whenever any dependent adapter is requested.",
            "first_difference_evictions": (
                first_difference["reference_observed"]["evicted_ids"] if first_difference else None
            ),
        },
        "native_feasible_set": {
            "observed_source": "VecWorkflowCoreEnv._typed_evictable_residents + LRUEvictionPolicy.plan_victims",
            "semantics": "A base is excluded from the single static candidate set while any resident adapter depends on it. The planner selects one minimum LRU prefix from that set and does not recompute eligibility after selecting an adapter.",
            "lru_key": "(last_used_step, native_object_id)",
            "initial_order": "-N..-1; runtime admission and hit use request_index/episode_step",
            "first_difference_eligible_victims": (
                first_difference["native_observed"]["eligible_victim_native_object_ids"] if first_difference else None
            ),
            "first_difference_actual_plan": (
                first_difference["native_observed"]["actual_eviction_plan"] if first_difference else None
            ),
        },
        "same_lru_label_same_decision_space": False,
        "design_issue": {
            "status": "OPEN_DESIGN_SEMANTICS_DIFFERENCE",
            "implemented_fix": False,
            "scope": "Whether dependency-closure eviction or sequential eligibility recomputation belongs in the native typed-cache action space.",
            "audit_boundary": "Recorded only; production eviction/admission rules were not changed.",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "audit_sequence_index",
        "config_id",
        "request_index",
        "adapter_id",
        "reference_loaded_bytes",
        "native_loaded_bytes",
        "reference_evicted_ids",
        "native_evicted_ids",
        "native_transaction_status",
        "native_admission_rejection_reason",
        "native_pre_full_hit",
        "native_post_service_success",
        "native_origin_execution",
        "differing_fields",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            native = row["native"]
            writer.writerow(
                {
                    "audit_sequence_index": row["audit_sequence_index"],
                    "config_id": row["config_id"],
                    "request_index": row["request_index"],
                    "adapter_id": row["adapter_id"],
                    "reference_loaded_bytes": row["reference"]["loaded_bytes"],
                    "native_loaded_bytes": sum(native["transfer_bytes_by_type_under_explicit_mapping"].values()),
                    "reference_evicted_ids": _canonical_json(row["reference"]["evicted_ids"]),
                    "native_evicted_ids": _canonical_json([item["object_id"] for item in native["evicted_typed_objects"]]),
                    "native_transaction_status": native["transaction_status"],
                    "native_admission_rejection_reason": native["admission_rejection_reason"],
                    "native_pre_full_hit": native["pre_action_readiness"]["full_service_ready"],
                    "native_post_service_success": native["service_success"],
                    "native_origin_execution": native["origin_execution"],
                    "differing_fields": _canonical_json(row["comparison"]["differing_fields"]),
                }
            )


def run_audit(*, zip_path: Path, output_root: Path, phase: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    references, input_integrity = verify_and_load_probe(zip_path)
    if _git("rev-parse", "HEAD") != AUDITED_NATIVE_COMMIT:
        raise RuntimeError("native audit must start from the frozen audited commit")

    output_root.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    config_summaries = []
    for workload_label, sharing_enabled in CONFIG_EXECUTION_ORDER:
        if phase == "first-two" and (workload_label, sharing_enabled) != ("interleaved", True):
            continue
        config_id = _config_id(workload_label, sharing_enabled)
        rows = execute_config(
            workload_label=workload_label,
            sharing_enabled=sharing_enabled,
            reference=references[config_id],
            request_limit=2 if phase == "first-two" else None,
        )
        for row in rows:
            row["audit_sequence_index"] = len(all_rows)
            all_rows.append(row)
        config_summaries.append(_summarize_config(rows, references[config_id]["summary"]))

    expected_count = 2 if phase == "first-two" else 288
    if len(all_rows) != expected_count:
        raise RuntimeError(f"unexpected audited request count: {len(all_rows)} != {expected_count}")
    first_difference = _first_difference(all_rows)
    if phase == "full" and (
        first_difference is None
        or first_difference["config_id"] != "interleaved__sharing_on"
        or first_difference["request_index"] != 1
        or first_difference["adapter_id"] != "b1.a0"
    ):
        raise RuntimeError("frozen first observed native difference changed")

    now = datetime.now(timezone.utc).isoformat()
    mapping_contract = _mapping_contract()
    summary = {
        "audit_run_id": AUDIT_RUN_ID,
        "phase": phase,
        "request_count": len(all_rows),
        "configuration_count": len(config_summaries),
        "configuration_summaries": config_summaries,
        "first_observed_native_difference": first_difference,
        "safety_stop_triggered": False,
        "state_invariant_failures": 0,
        "training_runs": 0,
        "policy_changes": 0,
        "matrix_expansion": 0,
    }
    claims = {
        "reviewed_at": now,
        "literature_cutoff": "2026-09-30 (no new literature search; bounded mechanism audit)",
        "target_venue": TARGET_VENUE,
        "artifact_run_id": AUDIT_RUN_ID,
        "policy_version": POLICY_VERSION,
        "git_commit": AUDITED_NATIVE_COMMIT,
        "evidence_level": "E2_ARTIFACT_AUDITED_FOR_BOUNDED_NATIVE_MICROTRACE_ONLY",
        "verdict": "Unverifiable for paper readiness; verified bounded native transaction semantics",
        "safe_claims": [
            "All four frozen LRU/136-MiB/empty-cache configurations were consumed by the native typed-cache transaction path for 72 requests each." if phase == "full" else "The first two sharing-on/interleaved requests were consumed by the native typed-cache transaction path.",
            "The native one-shot dependency-safe victim set rejects the observed b1.a0 family swap at request index 1 under the explicit 136-MiB-to-native-mb mapping.",
            "The observed rejection leaves resident and LRU policy state unchanged and preserves capacity, dependency, and resident-byte conservation invariants.",
        ],
        "unsupported_claims": [
            "Native and reference cache semantics are equivalent.",
            "The native mb field is intrinsically SI MB or MiB without the explicit mapping contract.",
            "Admission rejection is the same as service failure or origin execution.",
            "The audit establishes latency, throughput, algorithm advantage, training behavior, formal/holdout validity, or paper readiness.",
            "The synthetic b0/b1 adapters are real model artifacts or measured deployment sizes.",
        ],
    }
    environment = {
        "executed_at_utc": now,
        "command": " ".join([sys.executable, str(Path(__file__).resolve()), "--input-zip", str(zip_path.resolve()), "--output-root", str(output_root.resolve()), "--phase", phase]),
        "cwd": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_head_at_execution": _git("rev-parse", "HEAD"),
        "git_branch_at_execution": _git("branch", "--show-current"),
        "git_status_porcelain_at_execution": _git("status", "--porcelain").splitlines(),
        "audited_native_commit": AUDITED_NATIVE_COMMIT,
        "native_request_execution_count": len(all_rows),
        "training_execution_count": 0,
    }

    _write_json(output_root / "native_mapping_contract.json", mapping_contract)
    _write_jsonl(output_root / "request_comparison.jsonl", all_rows)
    _write_csv(output_root / "request_comparison.csv", all_rows)
    _write_json(output_root / "first_observed_native_difference.json", first_difference)
    _write_json(output_root / "eviction_feasible_set_comparison.json", _eviction_comparison(first_difference))
    _write_json(output_root / "request_admission_hit_transfer_failure_summary.json", summary)
    _write_json(output_root / "research_claim_boundary.json", claims)
    _write_json(output_root / "commands_and_environment.json", environment)
    _write_json(output_root / "input_integrity_manifest.json", input_integrity)

    artifact_files = []
    for path in sorted(output_root.iterdir()):
        if path.name == "artifact_integrity_manifest.json" or not path.is_file():
            continue
        artifact_files.append(
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    integrity = {
        "artifact_run_id": AUDIT_RUN_ID,
        "phase": phase,
        "generated_at_utc": now,
        "files": artifact_files,
        "file_count_excluding_manifest": len(artifact_files),
        "status": "pass",
    }
    _write_json(output_root / "artifact_integrity_manifest.json", integrity)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("first-two", "full"), default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(zip_path=args.input_zip, output_root=args.output_root, phase=args.phase)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
