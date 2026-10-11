"""Versioned state identities and durable per-branch evidence for CSCWD v3."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

from scripts.confirm_cscwd_phase_v2_repeated_action4 import canonical_sha, public_projection


STATE_IDENTITY_VERSION = "cscwd_public_full_state_v3_1"
BRANCH_RECORD_VERSION = "cscwd_region_branch_v3_1"


def full_state_projection(env: Any) -> dict[str, Any]:
    """Exact simulator/exogenous snapshot; never passed to an online policy."""
    return {
        "version": STATE_IDENTITY_VERSION,
        "config": env.config,
        "instance": env.instance,
        "object_catalog": env._object_catalog,
        "adapter_to_bundle": env._adapter_to_bundle,
        "trace": asdict(env.trace),
        "geometry": env.geometry,
        "phase_contract": env.phase_contract,
        "phase_condition": env.phase_condition,
        "nodes": env.nodes,
        "node_map": env.node_map,
        "execution_order": env.execution_order,
        "completed": env.completed,
        "node_index": env.node_index,
        "step_index": env.step_index,
        "clock_seconds": env.clock_seconds,
        "last_execution_rsu": env.last_execution_rsu,
        "prepared_state": env.prepared_state,
        "pending_delivery": env.pending_delivery,
        "execution_owner_rsu_id": env.execution_owner_rsu_id,
        "caches": {key: asdict(cache) for key, cache in sorted(env.caches.items())},
        "metrics": env.metrics,
        "reward_totals_by_profile": env._reward_totals_by_profile,
        "deadline_miss_recorded": env._deadline_miss_recorded,
        "decision_model_mode": env._decision_model_mode,
        "rng_state": env._rng.bit_generator.state,
    }


def state_identities(env: Any, info: dict[str, Any]) -> dict[str, str]:
    return {
        "identity_version": STATE_IDENTITY_VERSION,
        "public_observation_sha256": canonical_sha(public_projection(info)),
        "full_simulation_sha256": canonical_sha(full_state_projection(env)),
    }


def classify_identities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every selected cell; only exact full-state repeats become aliases."""
    first_full: dict[str, dict[str, Any]] = {}
    public_full: dict[str, set[str]] = {}
    result = []
    for row in rows:
        item = dict(row)
        public = str(item["public_observation_sha256"])
        full = str(item["full_simulation_sha256"])
        if full in first_full and first_full[full]["public_observation_sha256"] != public:
            raise RuntimeError("same full simulation state has different public observation")
        item["full_duplicate_of"] = first_full[full]["start_id"] if full in first_full else None
        item["public_observation_group_id"] = public
        if full not in first_full:
            first_full[full] = item
        public_full.setdefault(public, set()).add(full)
        result.append(item)
    for item in result:
        item["public_group_distinct_full_state_count"] = len(public_full[item["public_observation_group_id"]])
        item["same_public_different_full"] = item["public_group_distinct_full_state_count"] > 1
    return result


def atomic_record(path: Path, payload: dict[str, Any], *, version: str) -> str:
    """Publish one self-checking JSON file without exposing a partial write."""
    if path.exists():
        raise FileExistsError(f"create-only evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": version, "payload_sha256": canonical_sha(payload), "payload": payload}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return record["payload_sha256"]


def read_verified_record(path: Path, *, version: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != version or record.get("payload_sha256") != canonical_sha(record["payload"]):
        raise RuntimeError(f"evidence record identity mismatch: {path}")
    return record["payload"]


def execute_and_persist(
    tasks: Iterable[dict[str, Any]],
    execute_one: Callable[[dict[str, Any]], dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """A failed task leaves all earlier completed branch files readable."""
    completed = []
    for task in tasks:
        payload = execute_one(task)
        path = output_dir / f"{task['start_id_safe']}_action{task['action']}.json"
        atomic_record(path, payload, version=BRANCH_RECORD_VERSION)
        completed.append(path)
    return completed


def partial_evidence_status(output_dir: Path, *, current_branch: str | None,
                            environment_step_count: int) -> dict[str, Any]:
    """Count only durable branches whose embedded checksum reads back."""
    paths = sorted(output_dir.glob("*.json"))
    verified = []
    corrupt = []
    for path in paths:
        try:
            read_verified_record(path, version=BRANCH_RECORD_VERSION)
            verified.append(path.name)
        except Exception:
            corrupt.append(path.name)
    return {
        "partial_verdict": "PARTIAL_UNVERIFIABLE",
        "environment_step_count": environment_step_count,
        "completed_branch_count": len(verified),
        "completed_branch_files": verified,
        "corrupt_or_unverifiable_branch_files": corrupt,
        "incomplete_branch": current_branch,
    }
