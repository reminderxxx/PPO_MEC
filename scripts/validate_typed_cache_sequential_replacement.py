"""Generate bounded evidence for the opt-in typed-cache sequential LRU candidate.

This diagnostic uses the native environment, native LRU, normal ControlAction/Gym
action paths, and the frozen v0.2 probe sizes.  It does not train, tune, consume a
holdout, or claim algorithm advantage.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_native_typed_cache_probe import (  # noqa: E402
    ADAPTER_SIZE_BYTES,
    AUDIT_RUN_ID,
    BASE_SIZE_BYTES,
    CAPACITY_NATIVE_MB,
    CONFIG_EXECUTION_ORDER,
    EXPECTED_PROBE_SHA256,
    MIB_BYTES,
    WORKLOADS,
    _base_identity_for_adapter,
    _bytes_by_type,
    _control,
    _policy_rsu_state,
    _readiness,
    _resident_snapshot,
    build_audit_catalog,
    verify_and_load_probe,
)
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.data.model_catalog.adapter_catalog import (  # noqa: E402
    AdapterCatalog,
    TYPED_MODEL_CACHE_PROFILE_ID,
)
from src.envs.core.cache_eviction import (  # noqa: E402
    TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
    TYPED_EVICTION_SEMANTICS_STATIC,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode  # noqa: E402
from src.envs.wrappers.gym_vec_env import GymVecEnv  # noqa: E402


DELIVERED_AUDIT_COMMIT = "371159dabbacdc0d82acbc63efd588508338d289"
AUDITED_NATIVE_COMMIT = "73051ab264aa868e83f2e011b5ced26968eef74b"
RUN_ID = "native_typed_cache_replacement_20260930_v1"
POLICY_VERSION = "tmc_review_policy_v3_20260621"
TARGET_VENUE = "IEEE Transactions on Mobile Computing (TMC)"
PROTECTED_MAIN_ROOT = Path("/Users/howen/Projects/PPO_MEC")
PROTECTED_FILES = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}
PROTECTED_DIFF_SHA256 = "1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _replace_object(catalog: AdapterCatalog, object_id: str, **changes: object) -> None:
    index = next(
        index
        for index, item in enumerate(catalog.typed_cache_objects)
        if item.object_id == object_id
    )
    payload = catalog.typed_cache_objects[index].to_dict()
    payload.update(changes)
    payload["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(payload)
    catalog.typed_cache_objects[index] = type(catalog.typed_cache_objects[index])(**payload)


def _workflow(*, two_nodes: bool, first_base: str = "b0") -> WorkflowGraphState:
    nodes = [
        WorkflowNode(
            node_id="n0",
            node_name="synthetic diagnostic b0 request",
            required_base_model=first_base,
            required_adapter="b0.a0",
            input_size=1,
            output_size=1,
            successors=["n1"] if two_nodes else [],
        )
    ]
    if two_nodes:
        nodes.append(
            WorkflowNode(
                node_id="n1",
                node_name="synthetic diagnostic b1 request",
                required_base_model="b1",
                required_adapter="b1.a0",
                input_size=1,
                output_size=1,
                predecessors=["n0"],
            )
        )
    return WorkflowGraphState(
        workflow_id="synthetic_typed_cache_replacement_diagnostic",
        nodes=nodes,
        edges=[("n0", "n1")] if two_nodes else [],
        execution_order=[item.node_id for item in nodes],
        current_node_id="n0",
    )


def _build_env(
    catalog: AdapterCatalog,
    *,
    semantics: str,
    capacity_mb: float = CAPACITY_NATIVE_MB,
    two_node_workflow: bool = False,
    max_steps: int = 72,
) -> VecWorkflowCoreEnv:
    first_base = str(catalog.get_typed_adapter("b0.a0").required_base_model_id)
    frames = [
        {
            "time_index": step,
            "vehicles": [
                {
                    "vehicle_id": "diagnostic_vehicle",
                    "position_x": 0.0,
                    "position_y": 0.0,
                    "speed": 0.0,
                    "base_model_id": first_base,
                    "active_workflow_id": "synthetic_typed_cache_replacement_diagnostic",
                }
            ],
        }
        for step in range(max_steps + 2)
    ]
    env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=_workflow(two_nodes=two_node_workflow, first_base=first_base),
        adapter_catalog=catalog,
        rsu_states=[
            RSUState(
                rsu_id="rsu_a",
                position_x=0.0,
                position_y=0.0,
                coverage_radius=1_000_000.0,
            )
        ],
        max_steps=max_steps,
        cache_capacity_profile={
            "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
            "enabled": True,
            "unit": "mb",
            "capacity_mb": capacity_mb,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": semantics,
            "telemetry_enabled": True,
        },
    )
    env.reset()
    return env


def _execute_requests(
    *, workload: str, sharing_enabled: bool, semantics: str
) -> list[dict[str, Any]]:
    env = _build_env(
        build_audit_catalog(sharing_enabled=sharing_enabled), semantics=semantics
    )
    rows: list[dict[str, Any]] = []
    for request_index, adapter_id in enumerate(WORKLOADS[workload]):
        env._episode_steps = request_index
        base_identity = _base_identity_for_adapter(adapter_id, sharing_enabled)
        resident_before = _resident_snapshot(env)
        policy_before = env.export_cache_eviction_policy_state()
        readiness_before = _readiness(env, adapter_id, base_identity)
        placement = env.adapter_catalog.resolve_typed_placement_plan(
            adapter_id=adapter_id,
            resident_object_ids=list(env._typed_resident_object_ids["rsu_a"]),
        )
        required_free = max(
            resident_before["used_native_mb"]
            + float(placement.requested_bundle_mb)
            - CAPACITY_NATIVE_MB,
            0.0,
        )
        eligible_before = env._typed_evictable_residents("rsu_a")
        result = env._apply_typed_cache_action(
            control=_control(),
            primary_vehicle=None,
            current_node_id=f"request-{request_index:03d}",
            required_adapter=adapter_id,
        )
        readiness_after = _readiness(env, adapter_id, base_identity)
        result["service_readiness"] = readiness_after
        request_node = WorkflowNode(
            node_id=f"request-{request_index:03d}",
            node_name="fixed request replay",
            required_base_model=base_identity,
            required_adapter=adapter_id,
            input_size=0,
            output_size=0,
        )
        event = env._build_cache_event(
            current_node=request_node,
            primary_vehicle=None,
            request_rsu_id="rsu_a",
            selected_target_rsu_id="rsu_a",
            predicted_next_rsu_id=None,
            predicted_handoff_target_rsu_id=None,
            cache_hit=bool(readiness_after["full_service_ready"]),
            stall_occurred=not bool(readiness_after["full_service_ready"]),
            control=_control(),
            cache_result=result,
            handoff_count=0,
            migration_prepare_requested=False,
            migration_prepare_realized=False,
        ).to_dict()
        resident_after = _resident_snapshot(env)
        policy_after = env.export_cache_eviction_policy_state()
        env._validate_typed_resident_invariants("rsu_a")
        if resident_after["used_native_mb"] > CAPACITY_NATIVE_MB + 1.0e-9:
            raise RuntimeError("capacity invariant failed")
        admitted = list(result.get("admitted_typed_objects") or [])
        evicted = list(result.get("evicted_typed_objects") or [])
        expected_after = (
            resident_before["used_native_mb"]
            - sum(float(item["resident_size_mb"]) for item in evicted)
            + sum(float(item["resident_size_mb"]) for item in admitted)
        )
        if not math.isclose(
            expected_after, resident_after["used_native_mb"], abs_tol=1e-9
        ):
            raise RuntimeError("resident byte conservation failed")
        rejected = result.get("atomic_transaction_status") == "rolled_back_no_mutation"
        rejection_unchanged = None
        if rejected:
            rejection_unchanged = (
                resident_before == resident_after and policy_before == policy_after
            )
            if not rejection_unchanged:
                raise RuntimeError("rejection mutated real resident or policy state")
        rows.append(
            {
                "config_id": f"{workload}__sharing_{'on' if sharing_enabled else 'off'}",
                "workload": workload,
                "sharing_enabled": sharing_enabled,
                "request_index": request_index,
                "adapter_id": adapter_id,
                "required_base_identity": base_identity,
                "typed_eviction_semantics": semantics,
                "transaction_contract_version": result.get(
                    "typed_cache_transaction_contract_version"
                ),
                "resident_before": resident_before,
                "policy_state_before": _policy_rsu_state(policy_before),
                "readiness_before": readiness_before,
                "dependency_bundle": placement.to_dict(),
                "initial_eligible_victim_ids": eligible_before,
                "required_free_native_mb": required_free,
                "required_free_bytes_under_explicit_mapping": int(
                    round(required_free * MIB_BYTES)
                ),
                "eviction_plan": result.get("eviction_plan"),
                "eviction_planning_trace": result.get("eviction_planning_trace"),
                "atomic_transaction_status": result.get("atomic_transaction_status"),
                "admission_rejection_reason": result.get(
                    "capacity_rejection_reason"
                ),
                "admitted_typed_objects": admitted,
                "evicted_typed_objects": evicted,
                "actual_freed_native_mb": sum(
                    float(item["resident_size_mb"]) for item in evicted
                ),
                "transfer_native_mb_by_type": dict(
                    result.get("transfer_mb_by_type") or {}
                ),
                "transfer_bytes_by_type_under_explicit_mapping": _bytes_by_type(
                    dict(result.get("transfer_mb_by_type") or {})
                ),
                "readiness_after": readiness_after,
                "cache_event": event,
                "service_success": bool(event["service_success"]),
                "service_failure": bool(event["stall_occurred"]),
                "origin_execution": False,
                "resident_after": resident_after,
                "policy_state_after": _policy_rsu_state(policy_after),
                "rejection_state_unchanged": rejection_unchanged,
                "resident_byte_conservation_ok": True,
                "orphan_count": int(result.get("orphan_count", 0)),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    transfer: dict[str, int] = {}
    for row in rows:
        status = str(row["atomic_transaction_status"])
        statuses[status] = statuses.get(status, 0) + 1
        for object_type, value in row[
            "transfer_bytes_by_type_under_explicit_mapping"
        ].items():
            transfer[object_type] = transfer.get(object_type, 0) + int(value)
    return {
        "config_id": rows[0]["config_id"],
        "typed_eviction_semantics": rows[0]["typed_eviction_semantics"],
        "request_count": len(rows),
        "admission_requested_count": len(rows),
        "transaction_status_counts": dict(sorted(statuses.items())),
        "admission_rejection_count": sum(
            row["atomic_transaction_status"] == "rolled_back_no_mutation"
            for row in rows
        ),
        "actual_evicted_object_count": sum(
            len(row["evicted_typed_objects"]) for row in rows
        ),
        "service_success_count": sum(row["service_success"] for row in rows),
        "service_failure_count": sum(row["service_failure"] for row in rows),
        "origin_execution_count": sum(row["origin_execution"] for row in rows),
        "transfer_bytes_by_type_under_explicit_mapping": dict(sorted(transfer.items())),
        "total_transfer_bytes_under_explicit_mapping": sum(transfer.values()),
        "capacity_invariant_all_requests": all(
            row["resident_after"]["used_native_mb"] <= CAPACITY_NATIVE_MB + 1e-9
            for row in rows
        ),
        "dependency_invariant_all_requests": all(
            row["orphan_count"] == 0 for row in rows
        ),
        "rejection_state_unchanged_all_rejections": all(
            row["rejection_state_unchanged"] is True
            for row in rows
            if row["atomic_transaction_status"] == "rolled_back_no_mutation"
        ),
    }


def _case(
    name: str,
    *,
    initial: list[str],
    request: str,
    changes: dict[str, dict[str, object]] | None = None,
    capacity_mb: float = CAPACITY_NATIVE_MB,
) -> dict[str, Any]:
    catalog = build_audit_catalog(sharing_enabled=True)
    for object_id, values in (changes or {}).items():
        _replace_object(catalog, object_id, **values)
    catalog.rsu_typed_cache_profiles[0].resident_object_ids = list(initial)
    env = _build_env(
        catalog,
        semantics=TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
        capacity_mb=capacity_mb,
    )
    before_resident = _resident_snapshot(env)
    before_policy = env.export_cache_eviction_policy_state()
    result = env._apply_typed_cache_action(
        control=_control(),
        primary_vehicle=None,
        current_node_id=name,
        required_adapter=request,
    )
    after_resident = _resident_snapshot(env)
    after_policy = env.export_cache_eviction_policy_state()
    env._validate_typed_resident_invariants("rsu_a")
    rejected = result.get("atomic_transaction_status") == "rolled_back_no_mutation"
    unchanged = before_resident == after_resident and before_policy == after_policy
    if rejected and not unchanged:
        raise RuntimeError(f"{name}: rejection was not atomic")
    return {
        "case": name,
        "initial_resident_ids": initial,
        "requested_adapter_id": request,
        "catalog_changes": changes or {},
        "capacity_native_mb": capacity_mb,
        "resident_before": before_resident,
        "policy_state_before": _policy_rsu_state(before_policy),
        "transaction_status": result.get("atomic_transaction_status"),
        "rejection_reason": result.get("capacity_rejection_reason"),
        "dependency_bundle": result.get("dependency_bundle"),
        "eviction_plan": result.get("eviction_plan"),
        "eviction_planning_trace": result.get("eviction_planning_trace"),
        "evicted_object_ids": list(result.get("evicted_object_ids") or []),
        "required_free_native_mb": (
            (result.get("eviction_plan") or {}).get("required_free_capacity")
        ),
        "actual_freed_native_mb": float(result.get("evicted_size_mb_sum") or 0.0),
        "admitted_typed_objects": list(result.get("admitted_typed_objects") or []),
        "transfer_native_mb_by_type": dict(result.get("transfer_mb_by_type") or {}),
        "resident_after": after_resident,
        "policy_state_after": _policy_rsu_state(after_policy),
        "rejected_state_unchanged": unchanged if rejected else None,
        "capacity_ok": after_resident["used_native_mb"] <= capacity_mb + 1e-9,
        "orphan_count": int(result.get("orphan_count", 0)),
    }


def _boundary_cases() -> dict[str, Any]:
    cases = [
        _case(
            "two_request_second_swap",
            initial=["base:b0", "adapter:b0.a0"],
            request="b1.a0",
        ),
        _case(
            "multiple_adapter_dependency_release_positive",
            initial=["base:b0", "adapter:b0.a0", "adapter:b0.a1"],
            request="b1.a0",
        ),
        _case(
            "pinned_adapter_blocks_base_release",
            initial=["base:b0", "adapter:b0.a0"],
            request="b1.a0",
            changes={"adapter:b0.a0": {"evictability": "pinned"}},
        ),
        _case(
            "pinned_base_blocks_full_release",
            initial=["base:b0", "adapter:b0.a0"],
            request="b1.a0",
            changes={"base:b0": {"evictability": "pinned"}},
        ),
        _case(
            "non_evictable_object_blocks_release",
            initial=["base:b0", "adapter:b0.a0"],
            request="b1.a0",
            changes={"adapter:b0.a0": {"evictability": "non_evictable"}},
        ),
        _case(
            "other_retained_adapter_keeps_base_protected",
            initial=["base:b0", "adapter:b0.a0", "adapter:b0.a1"],
            request="b1.a0",
            changes={"adapter:b0.a1": {"evictability": "non_evictable"}},
        ),
        _case(
            "bundle_exceeds_total_capacity",
            initial=[],
            request="b1.a0",
            capacity_mb=135.0,
        ),
    ]
    return {
        "capacity_mapping": {
            "bytes_per_native_mb": MIB_BYTES,
            "capacity_bytes": int(CAPACITY_NATIVE_MB * MIB_BYTES),
            "capacity_native_mb": CAPACITY_NATIVE_MB,
            "b0_bytes": BASE_SIZE_BYTES["b0"],
            "b0_native_mb": BASE_SIZE_BYTES["b0"] / MIB_BYTES,
            "b1_bytes": BASE_SIZE_BYTES["b1"],
            "b1_native_mb": BASE_SIZE_BYTES["b1"] / MIB_BYTES,
            "adapter_bytes": ADAPTER_SIZE_BYTES,
            "adapter_native_mb": ADAPTER_SIZE_BYTES / MIB_BYTES,
            "warning": "native mb is a numeric field; MiB meaning comes only from the explicit bytes/1,048,576 mapping",
        },
        "cases": cases,
    }


def _env_step_witness(semantics: str) -> dict[str, Any]:
    core = _build_env(
        build_audit_catalog(sharing_enabled=True),
        semantics=semantics,
        two_node_workflow=True,
        max_steps=2,
    )
    env = GymVecEnv(core)
    _, reset_info = env.reset(seed=7)
    rows = []
    terminated = False
    truncated = False
    for step_index in range(2):
        workflow_before = deepcopy(env.last_semantic_state["workflow"])
        resident_before = deepcopy(
            reset_info["cache_trace_snapshot"]
            if step_index == 0
            else rows[-1]["cache_trace_snapshot_after"]
        )
        _, reward, terminated, truncated, info = env.step(0)
        workflow_after = deepcopy(info["semantic_state"]["workflow"])
        event = deepcopy(info["cache_event"])
        completed_now = [
            node_id
            for node_id in workflow_after["completed_node_ids"]
            if node_id not in workflow_before["completed_node_ids"]
        ]
        rows.append(
            {
                "step_index": step_index + 1,
                "action_id": 0,
                "action_name": info["action_name"],
                "action_mask_allowed_before": True,
                "control_action": info["control_action"],
                "resident_before": resident_before,
                "cache_admission_status": event["atomic_transaction_status"],
                "cache_rejection_reason": event["capacity_rejection_reason"],
                "cache_service_available": event["full_service_ready"],
                "node_execution_service_success": event["service_success"],
                "node_completed_this_step": completed_now,
                "workflow_before": workflow_before,
                "workflow_after": workflow_after,
                "workflow_completed": workflow_after["is_completed"],
                "hit_source": event["hit_source"],
                "origin_execution_evidence": False,
                "cache_event": event,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "cache_trace_snapshot_after": deepcopy(info["cache_trace_snapshot"]),
            }
        )
        if terminated or truncated:
            break
    return {
        "label": "synthetic diagnostic",
        "typed_eviction_semantics": semantics,
        "fixed_step_cap": 2,
        "same_workload_resource_action_rule_id": "one_vehicle_one_rsu_two_ordered_nodes_action0_v1",
        "reset_action_zero_allowed": bool(reset_info["action_mask"][0]),
        "steps": rows,
        "stop_reason": (
            "workflow_completed"
            if terminated
            else "fixed_step_cap_reached_before_workflow_completion"
            if truncated or len(rows) >= 2
            else "unexpected_early_stop"
        ),
        "workflow_completed": bool(core.workflow_state.is_completed),
    }


def _protected_snapshot() -> dict[str, Any]:
    actual = {
        path: _sha256_file(PROTECTED_MAIN_ROOT / path) for path in PROTECTED_FILES
    }
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *PROTECTED_FILES],
        cwd=PROTECTED_MAIN_ROOT,
    )
    status = _git("status", "--short", cwd=PROTECTED_MAIN_ROOT).splitlines()
    return {
        "protected_workspace": str(PROTECTED_MAIN_ROOT),
        "expected_file_sha256": PROTECTED_FILES,
        "actual_file_sha256": actual,
        "file_hashes_match_start": actual == PROTECTED_FILES,
        "expected_combined_binary_diff_sha256": PROTECTED_DIFF_SHA256,
        "actual_combined_binary_diff_sha256": _sha256_bytes(diff),
        "combined_diff_matches_start": _sha256_bytes(diff) == PROTECTED_DIFF_SHA256,
        "git_status_short": status,
        "exactly_seven_protected_modified_files": sorted(status)
        == sorted(f" M {path}" for path in PROTECTED_FILES),
        "protection_pass": actual == PROTECTED_FILES
        and _sha256_bytes(diff) == PROTECTED_DIFF_SHA256,
    }


def _verify_delivered_audit() -> dict[str, Any]:
    audit_root = ROOT / "artifacts/analysis" / AUDIT_RUN_ID
    manifest = json.loads((audit_root / "artifact_integrity_manifest.json").read_text())
    checks = []
    for row in manifest["files"]:
        path = audit_root / row["path"]
        checks.append(
            {
                "path": row["path"],
                "expected_sha256": row["sha256"],
                "actual_sha256": _sha256_file(path),
                "expected_size_bytes": row["size_bytes"],
                "actual_size_bytes": path.stat().st_size,
            }
        )
    if not all(
        row["expected_sha256"] == row["actual_sha256"]
        and row["expected_size_bytes"] == row["actual_size_bytes"]
        for row in checks
    ):
        raise RuntimeError("delivered audit artifact integrity mismatch")
    parent = _git("rev-parse", f"{DELIVERED_AUDIT_COMMIT}^")
    return {
        "delivered_audit_commit": DELIVERED_AUDIT_COMMIT,
        "delivered_audit_parent": parent,
        "audited_native_commit": AUDITED_NATIVE_COMMIT,
        "parent_matches_audited_native_commit": parent == AUDITED_NATIVE_COMMIT,
        "artifact_manifest_status": manifest["status"],
        "artifact_files_verified": checks,
    }


def run(*, input_zip: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    _, input_integrity = verify_and_load_probe(input_zip)
    delivered_audit = _verify_delivered_audit()
    protection_start = _protected_snapshot()
    if not protection_start["protection_pass"]:
        raise RuntimeError("protected main-workspace snapshot differs from task start")

    old_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for workload, sharing in CONFIG_EXECUTION_ORDER:
        old_rows.extend(
            _execute_requests(
                workload=workload,
                sharing_enabled=sharing,
                semantics=TYPED_EVICTION_SEMANTICS_STATIC,
            )
        )
        candidate_rows.extend(
            _execute_requests(
                workload=workload,
                sharing_enabled=sharing,
                semantics=TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            )
        )
    old_summaries = [
        _summarize([row for row in old_rows if row["config_id"] == config_id])
        for config_id in dict.fromkeys(row["config_id"] for row in old_rows)
    ]
    candidate_summaries = [
        _summarize([row for row in candidate_rows if row["config_id"] == config_id])
        for config_id in dict.fromkeys(row["config_id"] for row in candidate_rows)
    ]
    delivered_summary = json.loads(
        (
            ROOT
            / "artifacts/analysis"
            / AUDIT_RUN_ID
            / "request_admission_hit_transfer_failure_summary.json"
        ).read_text()
    )["configuration_summaries"]
    delivered_native = {
        row["config_id"]: row["native"] for row in delivered_summary
    }
    old_matches = {}
    for summary in old_summaries:
        expected = delivered_native[summary["config_id"]]
        old_matches[summary["config_id"]] = {
            "request_count": summary["request_count"] == expected["request_count"],
            "transaction_status_counts": summary["transaction_status_counts"]
            == expected["transaction_status_counts"],
            "admission_rejection_count": summary["admission_rejection_count"]
            == expected["admission_rejection_count"],
            "actual_evicted_object_count": summary["actual_evicted_object_count"]
            == expected["evicted_object_count"],
            "service_success_count": summary["service_success_count"]
            == expected["post_action_service_success_count"],
            "service_failure_count": summary["service_failure_count"]
            == expected["service_failure_count"],
            "transfer_bytes_by_type": summary[
                "transfer_bytes_by_type_under_explicit_mapping"
            ]
            == expected["transfer_bytes_by_type_under_explicit_mapping"],
        }
    if not all(all(fields.values()) for fields in old_matches.values()):
        raise RuntimeError("old semantics no longer match the delivered four-config audit")

    boundaries = _boundary_cases()
    env_witness = {
        "old": _env_step_witness(TYPED_EVICTION_SEMANTICS_STATIC),
        "candidate": _env_step_witness(
            TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU
        ),
    }
    protection_end = _protected_snapshot()
    if protection_end != protection_start:
        raise RuntimeError("protected main-workspace snapshot changed during validation")

    output_root.mkdir(parents=True)
    _write_jsonl(output_root / "four_config_old_request_log.jsonl", old_rows)
    _write_jsonl(
        output_root / "four_config_candidate_request_log.jsonl", candidate_rows
    )
    _write_json(
        output_root / "four_config_old_candidate_summary.json",
        {
            "request_count_per_semantics": 288,
            "configuration_count_per_semantics": 4,
            "old_semantics": old_summaries,
            "candidate_semantics": candidate_summaries,
            "old_matches_delivered_audit": old_matches,
            "interpretation_guard": "Lower transfer caused by rejected service is not a benefit; service success/failure and rejection counts must be read with transfer.",
        },
    )
    _write_json(output_root / "two_request_and_boundary_cases.json", boundaries)
    _write_json(output_root / "env_step_workflow_witness.json", env_witness)
    _write_json(
        output_root / "main_workspace_seven_file_protection.json",
        {"start": protection_start, "end": protection_end, "unchanged": True},
    )
    now = datetime.now(timezone.utc).isoformat()
    git_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    _write_json(
        output_root / "commands_environment_input_integrity.json",
        {
            "executed_at_utc": now,
            "command": " ".join(sys.argv),
            "cwd": str(ROOT),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "execution_git_commit": _git("rev-parse", "HEAD"),
            "execution_git_branch": _git("branch", "--show-current"),
            "execution_git_status_porcelain": _git("status", "--porcelain").splitlines(),
            "execution_worktree_diff_sha256": _sha256_bytes(git_diff),
            "input_integrity": input_integrity,
            "delivered_audit_identity": delivered_audit,
            "matrix_expansion": 0,
            "training_runs": 0,
            "formal_holdout_runs": 0,
        },
    )
    _write_json(
        output_root / "research_claim_boundary.json",
        {
            "reviewed_at": now,
            "literature_cutoff": "2026-09-30 (no literature search; bounded implementation diagnostic)",
            "target_venue": TARGET_VENUE,
            "artifact_run_id": RUN_ID,
            "policy_version": POLICY_VERSION,
            "git_commit": _git("rev-parse", "HEAD"),
            "evidence_level": "E2_BOUNDED_NATIVE_IMPLEMENTATION_DIAGNOSTIC",
            "verdict": "Candidate mechanism feasible on frozen synthetic microtraces; paper readiness and algorithm advantage unverifiable",
            "proved": [
                "The normal typed cache-fill action reaches the opt-in candidate for every agent using the shared five-action codec.",
                "The frozen b0.a0 to b1.a0 swap can atomically evict the old adapter and then its newly unneeded base under native LRU ordering.",
                "Pinned, non-evictable, oversized, and retained-dependency cases reject without real resident or policy-state mutation.",
                "A two-node synthetic workflow progresses and completes through reset plus two legal Gym env.step(action=0) calls under the candidate.",
            ],
            "not_proved": [
                "Algorithm advantage, latency improvement, throughput improvement, formal or holdout validity, canonical promotion, or paper readiness.",
                "Equivalence to the external reference ordering or numerical results.",
                "Support for sequential recomputation under FIFO, LFU, Aging-LFU, or Random eviction.",
                "Real model loading or real joint VEC adapter traces.",
            ],
        },
    )
    files = []
    for path in sorted(output_root.iterdir()):
        if path.name == "artifact_integrity_manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output_root / "artifact_integrity_manifest.json",
        {
            "artifact_run_id": RUN_ID,
            "generated_at_utc": now,
            "file_count_excluding_manifest": len(files),
            "files": files,
            "status": "pass",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(input_zip=args.input_zip, output_root=args.output_root)
    print(json.dumps({"status": "pass", "artifact_run_id": RUN_ID}, sort_keys=True))


if __name__ == "__main__":
    main()
