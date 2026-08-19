"""Single source of truth for legacy and typed model-cache runtime plumbing.

The resolver deliberately owns only runtime identity and validation.  Cache
semantics remain implemented by :class:`VecWorkflowCoreEnv`; callers pass the
resolved catalog and capacity profile to that environment unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.agents.registry import get_algo_spec
from src.data.model_catalog.adapter_catalog import (
    LEGACY_MODEL_CACHE_PROFILE_ID,
    TYPED_MODEL_CACHE_CONTRACT_VERSION,
    TYPED_MODEL_CACHE_PROFILE_ID,
    AdapterCatalog,
)


RUNTIME_CONTRACT_VERSION = "typed_model_cache_runtime_contract_v1.0.0"
CHECKPOINT_PROVENANCE_VERSION = "typed_checkpoint_provenance_v1.0.0"
TYPED_CACHE_TRANSACTION_CONTRACT_VERSION = "typed_cache_transaction_contract_v1.0.0"
CACHE_EVENT_SCHEMA_VERSION = "1.3.0"
CACHE_EFFICIENCY_METRICS_CONTRACT_VERSION = "1.1.0"
CACHE_TRACE_CONTEXT_VERSION = "1.0.0"
REQUEST_REPLAY_TYPED_CONTRACT_VERSION = "typed_cache_request_replay_v1.0.0"
ENVIRONMENT_CONTRACT = "VecWorkflowCoreEnv+GymVecEnv/v1"
REWARD_CONTRACT = "vec_workflow_reward_breakdown_v1"
OBSERVATION_SHAPE = [9]
ACTION_SHAPE = [5]


class RuntimeContractError(ValueError):
    """Raised when configuration or provenance would permit semantic drift."""


def _canonical_bytes(value: Any) -> bytes:
    def reject(item: Any, path: str = "$") -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise RuntimeContractError(f"non-finite runtime value at {path}")
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                reject(child, f"{path}[{index}]")

    reject(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _load_config(config: Mapping[str, Any] | str | Path | None) -> tuple[dict[str, Any], str]:
    if config is None:
        return {}, "implicit_legacy_defaults"
    if isinstance(config, Mapping):
        return deepcopy(dict(config)), "inline_mapping"
    path = Path(config).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeContractError(f"runtime config must be a mapping: {path}")
    return payload, path.as_posix()


def _path_identity(path: Path, root: Path) -> tuple[str, str]:
    resolved = path.resolve()
    try:
        logical = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        logical = resolved.as_posix()
    return logical, resolved.as_posix()


def _typed_catalog_binding(catalog: AdapterCatalog, capacity_mb: float) -> dict[str, Any]:
    objects = [
        {
            "object_id": item.object_id,
            "object_type": item.object_type,
            "version": item.version,
            "resident_size_mb": float(item.resident_size_mb),
            "transfer_size_mb": float(item.transfer_size_mb),
            "dependency_ids": list(item.dependency_ids),
            "evictability": item.evictability,
            "counts_toward_capacity": bool(item.counts_toward_capacity),
            "stable_fingerprint": item.stable_fingerprint,
        }
        for item in sorted(catalog.typed_cache_objects, key=lambda row: row.object_id)
    ]
    dependency_map = [
        {
            "object_id": item["object_id"],
            "object_type": item["object_type"],
            "dependency_ids": item["dependency_ids"],
        }
        for item in objects
    ]
    pinned = [
        {
            "object_id": item["object_id"],
            "evictability": item["evictability"],
            "counts_toward_capacity": item["counts_toward_capacity"],
        }
        for item in objects
    ]
    initial = [
        {
            "rsu_id": profile.rsu_id,
            "resident_object_ids": list(profile.resident_object_ids),
            "resident_mb": round(
                sum(
                    float(catalog.get_typed_object(object_id).resident_size_mb)
                    for object_id in profile.resident_object_ids
                ),
                6,
            ),
        }
        for profile in sorted(catalog.rsu_typed_cache_profiles, key=lambda row: row.rsu_id)
    ]
    oversized = [row for row in initial if row["resident_mb"] > capacity_mb + 1e-9]
    if oversized:
        raise RuntimeContractError(
            "typed initial state exceeds frozen MB capacity: "
            + ", ".join(f"{row['rsu_id']}={row['resident_mb']}" for row in oversized)
        )
    compatibility = {
        key: list(values) for key, values in sorted(catalog.compatibility_map.items())
    }
    taxonomy = sorted({item["object_type"] for item in objects})
    return {
        "object_taxonomy": taxonomy,
        "resident_objects": objects,
        "dependency_map": dependency_map,
        "dependency_fingerprint": sha256_value(dependency_map),
        "compatibility_map": compatibility,
        "compatibility_map_fingerprint": sha256_value(compatibility),
        "pinned_evictability_metadata": pinned,
        "pinned_evictability_fingerprint": sha256_value(pinned),
        "initial_per_rsu_typed_state": initial,
        "typed_initial_state_fingerprint": sha256_value(initial),
    }


def resolve_model_cache_runtime(
    config: Mapping[str, Any] | str | Path | None,
    *,
    root: str | Path,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve, validate, and hash one runtime contract.

    Missing profile fields intentionally select the legacy adapter-only profile.
    Typed mode never infers a catalog fingerprint or converts slots to MB.
    """

    root_path = Path(root).resolve()
    raw, config_source = _load_config(config)
    capacity = dict(raw.get("cache_capacity_profile") or {})
    profile = str(
        raw.get("model_cache_profile")
        or raw.get("model_cache_profile_id")
        or capacity.get("model_cache_profile_id")
        or LEGACY_MODEL_CACHE_PROFILE_ID
    )
    if profile not in {LEGACY_MODEL_CACHE_PROFILE_ID, TYPED_MODEL_CACHE_PROFILE_ID}:
        raise RuntimeContractError(f"unsupported model_cache_profile: {profile}")
    if capacity.get("model_cache_profile_id") not in {None, profile}:
        raise RuntimeContractError("cache capacity profile identity differs from model_cache_profile")
    declared_versions = {
        "typed_model_cache_contract_version": TYPED_MODEL_CACHE_CONTRACT_VERSION,
        "typed_cache_transaction_contract_version": TYPED_CACHE_TRANSACTION_CONTRACT_VERSION,
        "cache_event_schema_version": CACHE_EVENT_SCHEMA_VERSION,
        "cache_efficiency_metrics_contract_version": CACHE_EFFICIENCY_METRICS_CONTRACT_VERSION,
    }
    for field, expected_version in declared_versions.items():
        declared_version = raw.get(field)
        if declared_version is not None and str(declared_version) != expected_version:
            raise RuntimeContractError(
                f"{field} mismatch: {declared_version!r} != {expected_version!r}"
            )
    declared_transaction = raw.get("transaction_contract") or {}
    expected_transaction_fields = {
        "max_logical_cache_actions_per_step": 1,
        "max_dependency_bundle_objects": 2,
        "admission_order": ["base_model", "adapter"],
        "partial_admission": False,
        "atomic_rollback": True,
        "dependency_safe_base_eviction": "prohibit_while_resident_adapter_depends",
    }
    for field, expected_value in expected_transaction_fields.items():
        if field in declared_transaction and declared_transaction[field] != expected_value:
            raise RuntimeContractError(f"typed transaction config mismatch for {field}")

    default_catalog = root_path / "src/data/model_catalog/sample_model_catalog.json"
    catalog_value = raw.get("typed_catalog_path") or raw.get("catalog_path")
    catalog_path = Path(str(catalog_value)) if catalog_value else default_catalog
    if not catalog_path.is_absolute():
        catalog_path = root_path / catalog_path
    if not catalog_path.is_file():
        raise RuntimeContractError(f"model-cache catalog does not exist: {catalog_path}")
    catalog = AdapterCatalog.from_json(catalog_path)
    logical_catalog_path, absolute_catalog_path = _path_identity(catalog_path, root_path)
    catalog_fingerprint = catalog.canonical_fingerprint()

    if profile == TYPED_MODEL_CACHE_PROFILE_ID:
        if catalog.model_cache_profile_id != TYPED_MODEL_CACHE_PROFILE_ID:
            raise RuntimeContractError("typed runtime cannot consume a legacy adapter-only catalog")
        if capacity.get("enabled") is not True:
            raise RuntimeContractError("typed runtime requires cache_capacity_profile.enabled=true")
        if capacity.get("unit") != "mb":
            raise RuntimeContractError("typed runtime requires MB capacity; adapter slots are forbidden")
        try:
            capacity_mb = float(capacity.get("capacity_mb"))
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError("typed runtime requires finite positive capacity_mb") from exc
        if not math.isfinite(capacity_mb) or capacity_mb <= 0.0:
            raise RuntimeContractError("typed runtime requires finite positive capacity_mb")
        declared_fingerprint = str(raw.get("typed_catalog_fingerprint") or "")
        if not declared_fingerprint:
            raise RuntimeContractError("typed_catalog_fingerprint must be explicitly frozen")
        if declared_fingerprint != catalog_fingerprint:
            raise RuntimeContractError(
                f"typed catalog fingerprint mismatch: {declared_fingerprint} != {catalog_fingerprint}"
            )
        typed_binding = _typed_catalog_binding(catalog, capacity_mb)
        declared_initial = raw.get("typed_initial_state_fingerprint")
        if declared_initial and declared_initial != typed_binding["typed_initial_state_fingerprint"]:
            raise RuntimeContractError("typed initial-state fingerprint mismatch")
        declared_dependency = raw.get("typed_dependency_fingerprint")
        if declared_dependency and declared_dependency != typed_binding["dependency_fingerprint"]:
            raise RuntimeContractError("typed dependency fingerprint mismatch")
        declared_pinned = raw.get("typed_pinned_evictability_fingerprint")
        if declared_pinned and declared_pinned != typed_binding["pinned_evictability_fingerprint"]:
            raise RuntimeContractError("typed pinned/evictability fingerprint mismatch")
        normalized_capacity = {
            "model_cache_profile_id": profile,
            "enabled": True,
            "unit": "mb",
            "capacity_mb": capacity_mb,
            "count_base_model_separately": True,
            "eviction_policy": str(capacity.get("eviction_policy") or "lru"),
            "eviction_policy_seed": capacity.get("eviction_policy_seed"),
            "telemetry_enabled": True,
        }
    else:
        if catalog.model_cache_profile_id == TYPED_MODEL_CACHE_PROFILE_ID:
            raise RuntimeContractError("typed catalog cannot be consumed by a legacy runtime")
        unit = str(capacity.get("unit") or "adapter_slots")
        if unit not in {"adapter_slots", "mb"}:
            raise RuntimeContractError("legacy capacity unit must be adapter_slots or mb")
        enabled = bool(capacity.get("enabled", False))
        normalized_capacity = {
            "model_cache_profile_id": profile,
            "enabled": enabled,
            "unit": unit,
            "rsu_adapter_slots": None,
            "capacity_mb": None,
            "count_base_model_separately": False,
            "eviction_policy": str(capacity.get("eviction_policy") or "lru"),
            "eviction_policy_seed": capacity.get("eviction_policy_seed"),
            "telemetry_enabled": bool(capacity.get("telemetry_enabled", True)),
        }
        if enabled:
            key = "rsu_adapter_slots" if unit == "adapter_slots" else "capacity_mb"
            supplied = capacity.get(key)
            try:
                number = float(supplied)
            except (TypeError, ValueError) as exc:
                raise RuntimeContractError(f"legacy enabled capacity requires positive {key}") from exc
            if not math.isfinite(number) or number <= 0:
                raise RuntimeContractError(f"legacy enabled capacity requires positive {key}")
            normalized_capacity[key] = int(number) if unit == "adapter_slots" else number
        typed_binding = {
            "object_taxonomy": ["adapter"],
            "resident_objects": [],
            "dependency_map": [],
            "dependency_fingerprint": "unavailable_legacy_profile",
            "compatibility_map": {},
            "compatibility_map_fingerprint": "unavailable_legacy_profile",
            "pinned_evictability_metadata": [],
            "pinned_evictability_fingerprint": "unavailable_legacy_profile",
            "initial_per_rsu_typed_state": [],
            "typed_initial_state_fingerprint": "unavailable_legacy_profile",
        }

    contract = {
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "model_cache_profile": profile,
        "typed_model_cache_contract_version": (
            TYPED_MODEL_CACHE_CONTRACT_VERSION
            if profile == TYPED_MODEL_CACHE_PROFILE_ID
            else "unavailable_legacy_profile"
        ),
        "typed_cache_transaction_contract_version": (
            TYPED_CACHE_TRANSACTION_CONTRACT_VERSION
            if profile == TYPED_MODEL_CACHE_PROFILE_ID
            else "unavailable_legacy_profile"
        ),
        "typed_catalog_path": logical_catalog_path,
        "typed_catalog_absolute_path": absolute_catalog_path,
        "typed_catalog_file_sha256": sha256_file(catalog_path),
        "typed_catalog_fingerprint": catalog_fingerprint,
        "cache_capacity_profile": normalized_capacity,
        **typed_binding,
        "cache_event_schema_version": CACHE_EVENT_SCHEMA_VERSION,
        "cache_efficiency_metrics_contract_version": CACHE_EFFICIENCY_METRICS_CONTRACT_VERSION,
        "cache_trace_context_version": CACHE_TRACE_CONTEXT_VERSION,
        "request_replay_typed_contract_version": (
            REQUEST_REPLAY_TYPED_CONTRACT_VERSION
            if profile == TYPED_MODEL_CACHE_PROFILE_ID
            else "unavailable_legacy_profile"
        ),
        "transaction_contract": {
            "max_logical_cache_actions_per_step": 1,
            "max_dependency_bundle_objects": 2,
            "action_before_lookup": True,
            "admission_order": ["base_model", "adapter"],
            "partial_admission": False,
            "atomic_rollback": True,
            "dependency_safe_base_eviction": "prohibit_while_resident_adapter_depends",
        },
        "config_source": config_source,
        "execution_git_commit": _git_commit(root_path),
        "runtime_contract_hash_excludes": [
            "typed_catalog_absolute_path",
            "config_source",
            "execution_git_commit",
            "cache_capacity_profile.eviction_policy",
            "cache_capacity_profile.eviction_policy_seed",
            "cache_capacity_profile.eviction_policy_config",
        ],
    }
    hash_projection = deepcopy(contract)
    hash_projection.pop("typed_catalog_absolute_path", None)
    hash_projection.pop("config_source", None)
    hash_projection.pop("execution_git_commit", None)
    hash_capacity = hash_projection.get("cache_capacity_profile") or {}
    for field in ("eviction_policy", "eviction_policy_seed", "eviction_policy_config"):
        hash_capacity.pop(field, None)
    contract["runtime_contract_sha256"] = sha256_value(hash_projection)
    if expected_contract is not None:
        validate_runtime_compatibility(contract, expected_contract)
    return contract


def load_runtime_catalog(contract: Mapping[str, Any], *, root: str | Path) -> AdapterCatalog:
    path = Path(str(contract.get("typed_catalog_path") or ""))
    if not path.is_absolute():
        path = Path(root).resolve() / path
    catalog = AdapterCatalog.from_json(path)
    if catalog.canonical_fingerprint() != contract.get("typed_catalog_fingerprint"):
        raise RuntimeContractError("runtime catalog fingerprint drift detected at consumption")
    if catalog.model_cache_profile_id != contract.get("model_cache_profile"):
        raise RuntimeContractError("runtime profile/catalog mismatch at consumption")
    return catalog


def validate_runtime_compatibility(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "runtime_contract_version",
        "model_cache_profile",
        "typed_model_cache_contract_version",
        "typed_cache_transaction_contract_version",
        "typed_catalog_fingerprint",
        "typed_initial_state_fingerprint",
        "dependency_fingerprint",
        "compatibility_map_fingerprint",
        "pinned_evictability_fingerprint",
        "cache_event_schema_version",
        "cache_efficiency_metrics_contract_version",
        "cache_trace_context_version",
    )
    mismatches = [
        {"field": field, "actual": actual.get(field), "expected": expected.get(field)}
        for field in fields
        if actual.get(field) != expected.get(field)
    ]
    actual_capacity = actual.get("cache_capacity_profile") or {}
    expected_capacity = expected.get("cache_capacity_profile") or {}
    for field in ("enabled", "unit", "rsu_adapter_slots", "capacity_mb"):
        if actual_capacity.get(field) != expected_capacity.get(field):
            mismatches.append(
                {
                    "field": f"cache_capacity_profile.{field}",
                    "actual": actual_capacity.get(field),
                    "expected": expected_capacity.get(field),
                }
            )
    if mismatches:
        raise RuntimeContractError(f"runtime contract mismatch: {mismatches}")
    return {"status": "compatible", "mismatches": []}


def build_checkpoint_provenance(
    *,
    root: str | Path,
    agent_name: str,
    training_seed: int,
    runtime_contract: Mapping[str, Any],
    reward_positive_offset: float,
    train_window_plan_identity: Mapping[str, Any],
) -> dict[str, Any]:
    spec = get_algo_spec(agent_name)
    capacity = runtime_contract["cache_capacity_profile"]
    return {
        "checkpoint_provenance_version": CHECKPOINT_PROVENANCE_VERSION,
        "execution_git_commit": _git_commit(Path(root).resolve()),
        "agent_identity": agent_name,
        "training_seed": int(training_seed),
        "model_cache_profile": runtime_contract["model_cache_profile"],
        "typed_model_cache_contract_version": runtime_contract[
            "typed_model_cache_contract_version"
        ],
        "typed_catalog_fingerprint": runtime_contract["typed_catalog_fingerprint"],
        "typed_initial_state_fingerprint": runtime_contract[
            "typed_initial_state_fingerprint"
        ],
        "runtime_contract_sha256": runtime_contract["runtime_contract_sha256"],
        "capacity_unit": capacity["unit"],
        "capacity_value": (
            capacity.get("capacity_mb")
            if capacity["unit"] == "mb"
            else capacity.get("rsu_adapter_slots")
        ),
        "eviction_policy": capacity.get("eviction_policy"),
        "observation_contract": spec["observation_contract"],
        "action_contract": spec["action_contract"],
        "observation_shape": list(OBSERVATION_SHAPE),
        "action_shape": list(ACTION_SHAPE),
        "reward_contract": {
            "version": REWARD_CONTRACT,
            "reward_positive_offset": float(reward_positive_offset),
        },
        "environment_contract": ENVIRONMENT_CONTRACT,
        "cache_event_schema_version": runtime_contract["cache_event_schema_version"],
        "cache_efficiency_metrics_contract_version": runtime_contract[
            "cache_efficiency_metrics_contract_version"
        ],
        "train_window_plan_identity": deepcopy(dict(train_window_plan_identity)),
        "checkpoint_sha256_binding": "external_manifest_after_serialization",
    }


def _extract_training_metadata(checkpoint_path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - exact torch error varies by version
        raise RuntimeContractError(f"unable to load checkpoint metadata: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata") or {}
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def validate_checkpoint_provenance(
    checkpoint_path: str | Path,
    *,
    expected_agent_name: str,
    expected_seed: int,
    expected_runtime_contract: Mapping[str, Any],
    expected_reward_positive_offset: float,
    expected_window_plan_identity: Mapping[str, Any],
    expected_checkpoint_sha256: str | None = None,
    require_git_commit: str | None = None,
) -> dict[str, Any]:
    """Return the three-state typed checkpoint provenance gate result."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    metadata = _extract_training_metadata(path)
    provenance = metadata.get("typed_runtime_provenance")
    file_hash = sha256_file(path)
    if not isinstance(provenance, Mapping):
        return {
            "status": "unavailable_legacy_metadata",
            "checkpoint_path": path.as_posix(),
            "checkpoint_sha256": file_hash,
            "errors": ["typed_runtime_provenance missing"],
        }
    expected_capacity = expected_runtime_contract["cache_capacity_profile"]
    expected_value = (
        expected_capacity.get("capacity_mb")
        if expected_capacity.get("unit") == "mb"
        else expected_capacity.get("rsu_adapter_slots")
    )
    spec = get_algo_spec(expected_agent_name)
    checks = {
        "checkpoint_provenance_version": CHECKPOINT_PROVENANCE_VERSION,
        "agent_identity": expected_agent_name,
        "training_seed": int(expected_seed),
        "model_cache_profile": expected_runtime_contract["model_cache_profile"],
        "typed_model_cache_contract_version": expected_runtime_contract[
            "typed_model_cache_contract_version"
        ],
        "typed_catalog_fingerprint": expected_runtime_contract["typed_catalog_fingerprint"],
        "typed_initial_state_fingerprint": expected_runtime_contract[
            "typed_initial_state_fingerprint"
        ],
        "runtime_contract_sha256": expected_runtime_contract["runtime_contract_sha256"],
        "capacity_unit": expected_capacity["unit"],
        "capacity_value": expected_value,
        "eviction_policy": expected_capacity.get("eviction_policy"),
        "observation_contract": spec["observation_contract"],
        "action_contract": spec["action_contract"],
        "observation_shape": OBSERVATION_SHAPE,
        "action_shape": ACTION_SHAPE,
        "environment_contract": ENVIRONMENT_CONTRACT,
        "cache_event_schema_version": CACHE_EVENT_SCHEMA_VERSION,
        "cache_efficiency_metrics_contract_version": CACHE_EFFICIENCY_METRICS_CONTRACT_VERSION,
        "train_window_plan_identity": dict(expected_window_plan_identity),
    }
    errors = [
        f"{field} mismatch: {provenance.get(field)!r} != {expected!r}"
        for field, expected in checks.items()
        if provenance.get(field) != expected
    ]
    reward = provenance.get("reward_contract") or {}
    if reward.get("version") != REWARD_CONTRACT or reward.get(
        "reward_positive_offset"
    ) != float(expected_reward_positive_offset):
        errors.append("reward_contract mismatch")
    if require_git_commit is not None and provenance.get("execution_git_commit") != require_git_commit:
        errors.append("execution_git_commit mismatch")
    if not str(provenance.get("execution_git_commit") or ""):
        errors.append("execution_git_commit missing")
    if expected_checkpoint_sha256 is not None and file_hash != expected_checkpoint_sha256:
        errors.append("checkpoint SHA-256 mismatch")
    return {
        "status": "compatible" if not errors else "incompatible",
        "checkpoint_path": path.as_posix(),
        "checkpoint_sha256": file_hash,
        "metadata": dict(provenance),
        "errors": errors,
    }
