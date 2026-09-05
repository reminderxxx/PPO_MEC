"""Fail-closed identity gate for the one active formal Protocol bundle.

The hash graph is intentionally acyclic:

1. ``active_bundle_core_sha256`` binds immutable resource identities.
2. Readiness binds that core plus the acceptance-evidence manifest.
3. ``active_formal_bundle_sha256`` binds the ready index, including the
   Readiness file content hash, while excluding only its own hash field.

The outer formal runner is the execution consumer.  It must call this module
before it creates a run root, execution binding, resolved context, or ledger.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from src.runtime.formal_execution_environment import (
    FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION,
    normalize_environment_identity,
    protocol_bound_extensions_from_protocol,
)
from src.runtime.formal_protocol_capabilities import (
    FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
    protocol_capability_matrix,
    require_live_execution_protocol,
)


ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION = "1.1.0"
ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION = "1.0.0"
ACTIVE_PROTOCOL_VERSION = "2.7.0"
ACTIVE_PROTOCOL_ID = "typed_model_cache_formal_protocol_v2_7"
READY_STATUS = "READY_FOR_G14C_V15_CLEAN_TRAIN_AND_FORMAL"
READINESS_VERSION = "19.0.0"
DEFAULT_ACTIVE_INDEX_RELATIVE = (
    "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905/"
    "protocol_index.json"
)
CAPACITY_ORDER = (
    "constrained_288mb",
    "medium_576mb",
    "relaxed_864mb",
)
_VALIDATED_BUNDLE_TOKEN = object()


class ActiveFormalBundleError(ValueError):
    """Raised before any formal run artifact can be created."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActiveFormalBundleError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ActiveFormalBundleError(f"non-finite JSON value in {label}: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveFormalBundleError(f"unable to load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ActiveFormalBundleError(f"{label} must be a JSON object")
    return payload


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ActiveFormalBundleError(
            f"Git identity check failed: git {' '.join(args)}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _resolve_registered_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ActiveFormalBundleError(f"{label} must be one repository-relative path")
    relative = Path(value)
    if any(part in {".", ".."} for part in relative.parts):
        raise ActiveFormalBundleError(
            f"{label} must not depend on cwd or path normalization"
        )
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ActiveFormalBundleError(
                f"symlink is forbidden for active bundle resource: {label}"
            )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ActiveFormalBundleError(f"active bundle resource escapes repository: {label}") from exc
    if not resolved.is_file():
        raise ActiveFormalBundleError(f"active bundle resource is missing: {label}")
    return resolved


def active_bundle_core_projection(index: Mapping[str, Any]) -> dict[str, Any]:
    resources = [
        deepcopy(row)
        for row in index.get("active_bundle_resources", [])
        if isinstance(row, Mapping) and row.get("logical_id") != "readiness_companion"
    ]
    return {
        "active_formal_bundle_contract_version": index.get(
            "active_formal_bundle_contract_version"
        ),
        "active_bundle_resource_resolution_contract_version": index.get(
            "active_bundle_resource_resolution_contract_version"
        ),
        "protocol_index_version": index.get("protocol_index_version"),
        "protocol_identity": deepcopy(index.get("protocol_identity")),
        "execution_commit_binding": deepcopy(index.get("execution_commit_binding")),
        "resources": resources,
        "command_matrix_identity": deepcopy(index.get("command_matrix_identity")),
        "holdout_seal": deepcopy(index.get("holdout_seal")),
    }


def ready_index_projection(index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in index.items()
        if key != "active_formal_bundle_sha256"
    }


def build_resource_row(
    *,
    root: Path,
    logical_id: str,
    role: str,
    relative_path: str,
    version_scope: str,
    shared_reason: str | None = None,
    semantic_sha256: str | None = None,
) -> dict[str, Any]:
    path = _resolve_registered_path(root, relative_path, logical_id)
    row: dict[str, Any] = {
        "logical_id": logical_id,
        "role": role,
        "logical_path": relative_path,
        "content_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "version_scope": version_scope,
    }
    if semantic_sha256:
        row["semantic_sha256"] = semantic_sha256
    if version_scope == "shared_historical_stable":
        if not shared_reason:
            raise ActiveFormalBundleError(
                f"shared resource lacks an allowlisted reason: {logical_id}"
            )
        row["shared_reason"] = shared_reason
    elif shared_reason is not None:
        raise ActiveFormalBundleError(
            f"current-version resource cannot declare a shared reason: {logical_id}"
        )
    return row


def _validate_resource_rows(
    root: Path, rows: Any, *, require_readiness: bool = True
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not isinstance(rows, list) or not rows:
        raise ActiveFormalBundleError("active bundle resource inventory is missing")
    observed: dict[str, dict[str, Any]] = {}
    current_root = Path(DEFAULT_ACTIVE_INDEX_RELATIVE).parent.as_posix() + "/"
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ActiveFormalBundleError("active bundle resource row is not an object")
        row = dict(raw)
        logical_id = row.get("logical_id")
        if not isinstance(logical_id, str) or not logical_id or logical_id in observed:
            raise ActiveFormalBundleError("active bundle resource logical ID is invalid or duplicated")
        path = _resolve_registered_path(root, row.get("logical_path"), logical_id)
        if sha256_file(path) != row.get("content_sha256"):
            raise ActiveFormalBundleError(f"active bundle resource content drift: {logical_id}")
        if path.stat().st_size != row.get("size_bytes"):
            raise ActiveFormalBundleError(f"active bundle resource size drift: {logical_id}")
        scope = row.get("version_scope")
        logical_path = str(row.get("logical_path"))
        if scope == "current_protocol_version":
            if not logical_path.startswith(current_root):
                raise ActiveFormalBundleError(
                    f"current-version resource points outside the active Protocol: {logical_id}"
                )
            if "shared_reason" in row:
                raise ActiveFormalBundleError(
                    f"current-version resource has a shared exception: {logical_id}"
                )
        elif scope == "shared_historical_stable":
            if not isinstance(row.get("shared_reason"), str) or not row["shared_reason"]:
                raise ActiveFormalBundleError(
                    f"cross-version resource is not explicitly allowlisted: {logical_id}"
                )
        else:
            raise ActiveFormalBundleError(f"invalid resource version scope: {logical_id}")
        observed[logical_id] = row
    required = {
        "protocol_manifest",
        "execution_environment_manifest",
        "agent_training_scientific_config",
        "formal_agent_order_contract",
        "formal_training_execution_binding_schema",
        "resolved_execution_context_schema",
        "active_bundle_resource_resolution_contract",
        "portable_resource_registry",
        "split_companion",
        "window_consumption_contract",
        "formal_exogenous_request_execution_contract",
        "formal_request_exposure_schema",
        "formal_request_subject_lifecycle_contract",
        "environment_identity_projection_contract",
        "formal_nullable_metric_aggregation_contract",
        "formal_protocol_capability_routing_contract",
        "formal_generated_checkpoint_resource_identity_contract",
    }
    if require_readiness:
        required.add("readiness_companion")
    missing = sorted(required - set(observed))
    if missing:
        raise ActiveFormalBundleError(f"active bundle resources are incomplete: {missing}")
    return observed, sorted(observed)


def validate_active_formal_bundle(
    *,
    repository_root: str | Path,
    index_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    execution_environment_manifest_path: str | Path | None = None,
    require_ready: bool = True,
    require_clean_git: bool = True,
    require_origin_main_match: bool = True,
) -> dict[str, Any]:
    """Validate the unique active bundle without creating any run output."""

    root = Path(repository_root).resolve()
    selected_index = Path(index_path or root / DEFAULT_ACTIVE_INDEX_RELATIVE)
    if not selected_index.is_absolute():
        selected_index = root / selected_index
    expected_index = root / DEFAULT_ACTIVE_INDEX_RELATIVE
    try:
        selected_relative = selected_index.relative_to(root)
    except ValueError as exc:
        raise ActiveFormalBundleError(
            "only the unique active protocol index is accepted"
        ) from exc
    if any(part in {".", ".."} for part in selected_relative.parts):
        raise ActiveFormalBundleError(
            "active protocol index must not depend on cwd or path normalization"
        )
    current = root
    for part in selected_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ActiveFormalBundleError(
                "only the unique non-symlinked active protocol index is accepted"
            )
    if selected_index.resolve() != expected_index.resolve():
        raise ActiveFormalBundleError("only the unique active protocol index is accepted")
    index = _strict_object(selected_index.resolve(), "active protocol index")
    if index.get("active_formal_bundle_contract_version") != (
        ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION
    ):
        raise ActiveFormalBundleError("active formal bundle contract version mismatch")
    if index.get("active_bundle_resource_resolution_contract_version") != (
        ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION
    ):
        raise ActiveFormalBundleError(
            "active bundle resource resolution contract version mismatch"
        )
    if index.get("protocol_index_version") != ACTIVE_PROTOCOL_VERSION:
        raise ActiveFormalBundleError("active protocol index version mismatch")
    if require_ready and index.get("status") != READY_STATUS:
        raise ActiveFormalBundleError("active protocol index is not ready")

    expected_core = canonical_sha256(active_bundle_core_projection(index))
    if index.get("active_bundle_core_sha256") != expected_core:
        raise ActiveFormalBundleError("active bundle core SHA-256 mismatch")
    resources, resource_ids = _validate_resource_rows(
        root,
        index.get("active_bundle_resources"),
        require_readiness=require_ready,
    )
    protocol_row = resources["protocol_manifest"]
    environment_row = resources["execution_environment_manifest"]
    protocol_file = _resolve_registered_path(root, protocol_row["logical_path"], "protocol")
    environment_file = _resolve_registered_path(
        root, environment_row["logical_path"], "execution environment"
    )

    def same_file(user_value: str | Path | None, registered: Path, label: str) -> None:
        if user_value is None:
            return
        user_path = Path(user_value)
        if not user_path.is_absolute():
            user_path = root / user_path
        try:
            user_relative = user_path.relative_to(root)
        except ValueError as exc:
            raise ActiveFormalBundleError(
                f"CLI {label} does not equal the indexed resource"
            ) from exc
        if any(part in {".", ".."} for part in user_relative.parts):
            raise ActiveFormalBundleError(
                f"CLI {label} must not depend on cwd or path normalization"
            )
        current = root
        for part in user_relative.parts:
            current = current / part
            if current.is_symlink():
                raise ActiveFormalBundleError(
                    f"CLI {label} must not traverse a symlink"
                )
        if user_path.resolve() != registered.resolve():
            raise ActiveFormalBundleError(f"CLI {label} does not equal the indexed resource")
        if sha256_file(user_path) != sha256_file(registered):
            raise ActiveFormalBundleError(f"CLI {label} content differs from the indexed resource")

    same_file(protocol_path, protocol_file, "Protocol")
    same_file(
        execution_environment_manifest_path,
        environment_file,
        "execution environment manifest",
    )
    protocol = _strict_object(protocol_file, "active Protocol manifest")
    environment = _strict_object(environment_file, "active execution environment manifest")
    identity = index.get("protocol_identity")
    if not isinstance(identity, Mapping):
        raise ActiveFormalBundleError("active index protocol identity is missing")
    comparisons = {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("typed_model_cache_formal_protocol_version"),
        "protocol_semantic_sha256": protocol.get("hashes", {}).get("semantic_sha256"),
        "protocol_full_sha256": protocol.get("hashes", {}).get("full_sha256"),
    }
    for field, observed in comparisons.items():
        if identity.get(field) != observed:
            raise ActiveFormalBundleError(f"index/Protocol identity drift: {field}")
    if comparisons["protocol_id"] != ACTIVE_PROTOCOL_ID or comparisons[
        "protocol_version"
    ] != ACTIVE_PROTOCOL_VERSION:
        raise ActiveFormalBundleError("old or unexpected Protocol is audit-only")
    if protocol_row.get("semantic_sha256") != comparisons["protocol_semantic_sha256"]:
        raise ActiveFormalBundleError("Protocol resource semantic identity drift")
    try:
        capabilities = require_live_execution_protocol(comparisons["protocol_version"])
    except ValueError as exc:
        raise ActiveFormalBundleError(str(exc)) from exc
    if (
        not capabilities.persisted_resolved_execution_context_required
        or not capabilities.nullable_metric_contract_required
        or not capabilities.generated_checkpoint_resource_required
        or capabilities.holdout_capability
    ):
        raise ActiveFormalBundleError("active Protocol capability route is unsafe")
    routing = _strict_object(
        _resolve_registered_path(
            root,
            resources["formal_protocol_capability_routing_contract"]["logical_path"],
            "formal protocol capability routing contract",
        ),
        "formal protocol capability routing contract",
    )
    if (
        routing.get("version")
        != FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION
        or routing.get("capability_matrix") != protocol_capability_matrix()
        or routing.get("semantic_sha256")
        != canonical_sha256(
            {key: value for key, value in routing.items() if key != "semantic_sha256"}
        )
        or routing.get("semantic_sha256")
        != protocol.get("formal_protocol_capability_routing_contract", {}).get(
            "semantic_sha256"
        )
    ):
        raise ActiveFormalBundleError("formal Protocol capability routing drift")

    scientific = _strict_object(
        _resolve_registered_path(
            root,
            resources["agent_training_scientific_config"]["logical_path"],
            "scientific config",
        ),
        "agent training scientific config",
    )
    order = _strict_object(
        _resolve_registered_path(
            root,
            resources["formal_agent_order_contract"]["logical_path"],
            "agent order contract",
        ),
        "formal agent order contract",
    )
    if scientific.get("config_semantic_sha256") != protocol.get(
        "agent_training_scientific_config_contract", {}
    ).get("config_semantic_sha256"):
        raise ActiveFormalBundleError("Scientific Config identity drift")
    if order.get("semantic_sha256") != protocol.get("formal_agent_order_contract", {}).get(
        "semantic_sha256"
    ):
        raise ActiveFormalBundleError("Agent Order Contract identity drift")
    environment_identity = environment.get("scientific_identity")
    protocol_environment = protocol.get("formal_execution_environment_contract", {}).get(
        "scientific_identity"
    )
    if environment_identity != protocol_environment:
        raise ActiveFormalBundleError("environment manifest/Protocol identity drift")
    try:
        normalized_environment = normalize_environment_identity(environment_identity)
        protocol_extensions = protocol_bound_extensions_from_protocol(protocol)
    except ValueError as exc:
        raise ActiveFormalBundleError(
            f"environment identity projection validation failed: {exc}"
        ) from exc
    if normalized_environment != environment_identity:
        raise ActiveFormalBundleError("environment identity is not canonical")
    if {
        field: environment_identity[field] for field in protocol_extensions
    } != protocol_extensions:
        raise ActiveFormalBundleError("Protocol/environment extension identity drift")
    projection_contract = _strict_object(
        _resolve_registered_path(
            root,
            resources["environment_identity_projection_contract"]["logical_path"],
            "environment identity projection contract",
        ),
        "environment identity projection contract",
    )
    if projection_contract.get("version") != (
        FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    ):
        raise ActiveFormalBundleError(
            "environment identity projection contract version mismatch"
        )
    if not isinstance(environment_identity, Mapping) or environment_identity.get(
        "environment_fingerprint"
    ) != index.get("environment_identity", {}).get("environment_fingerprint"):
        raise ActiveFormalBundleError("environment fingerprint drift")
    if environment_identity.get("dependency_fingerprint") != index.get(
        "environment_identity", {}
    ).get("dependency_fingerprint"):
        raise ActiveFormalBundleError("dependency fingerprint drift")

    readiness: dict[str, Any] | None = None
    if require_ready:
        readiness_row = resources["readiness_companion"]
        readiness = _strict_object(
            _resolve_registered_path(root, readiness_row["logical_path"], "readiness"),
            "Readiness v19 companion",
        )
        if readiness.get("readiness_review_version") != READINESS_VERSION:
            raise ActiveFormalBundleError("Readiness companion version mismatch")
        if readiness.get("verdict") != READY_STATUS or readiness.get("status") != "ready":
            raise ActiveFormalBundleError("Readiness companion is not ready")
        if readiness.get("active_bundle_core_sha256") != expected_core:
            raise ActiveFormalBundleError("Readiness does not bind the active bundle core")
        evidence_path = _resolve_registered_path(
            root, readiness.get("evidence_manifest_path"), "readiness evidence"
        )
        if sha256_file(evidence_path) != readiness.get("evidence_manifest_sha256"):
            raise ActiveFormalBundleError("Readiness acceptance evidence drift")
        evidence = _strict_object(evidence_path, "Readiness acceptance evidence")
        if evidence.get("status") != "pass" or not evidence.get("clean_candidate"):
            raise ActiveFormalBundleError("Readiness acceptance evidence is missing or incomplete")
        if evidence.get("active_bundle_core_sha256") != expected_core:
            raise ActiveFormalBundleError("acceptance evidence bundle identity drift")
        rehearsal_path = _resolve_registered_path(
            root,
            evidence.get("real_downstream_consumer_rehearsal_path"),
            "real downstream consumer rehearsal",
        )
        if sha256_file(rehearsal_path) != evidence.get(
            "real_downstream_consumer_rehearsal_sha256"
        ):
            raise ActiveFormalBundleError("real downstream consumer rehearsal drift")
        rehearsal = _strict_object(
            rehearsal_path, "real downstream consumer rehearsal"
        )
        if not all(
            (
                evidence.get("real_downstream_consumer_rehearsal_status") == "pass",
                rehearsal.get("status") == "pass",
                rehearsal.get("clean_detached_candidate") is True,
                rehearsal.get("completed_phase_terminal_count") == 13,
                rehearsal.get("formal") is False,
                rehearsal.get("performance_evidence") is False,
                rehearsal.get("holdout_capability") is False,
            )
        ):
            raise ActiveFormalBundleError(
                "real downstream consumer rehearsal is missing or incomplete"
            )
        entrypoint_path = _resolve_registered_path(
            root,
            evidence.get("formal_training_entrypoint_acceptance_path"),
            "formal training entrypoint acceptance",
        )
        if sha256_file(entrypoint_path) != evidence.get(
            "formal_training_entrypoint_acceptance_sha256"
        ):
            raise ActiveFormalBundleError(
                "formal training entrypoint acceptance drift"
            )
        entrypoint = _strict_object(
            entrypoint_path, "formal training entrypoint acceptance"
        )
        nullable_hash = protocol[
            "formal_nullable_metric_aggregation_contract"
        ]["semantic_sha256"]
        if not all(
            (
                evidence.get("formal_training_entrypoint_acceptance_status") == "pass",
                entrypoint.get("status") == "pass",
                entrypoint.get("clean_detached_candidate") is True,
                entrypoint.get("formal") is False,
                entrypoint.get("training") is False,
                entrypoint.get("performance_evidence") is False,
                entrypoint.get("training_command_count") == 150,
                entrypoint.get("passed_command_count") == 150,
                entrypoint.get("episode_count") == 0,
                entrypoint.get("environment_interaction_count") == 0,
                entrypoint.get("update_count") == 0,
                entrypoint.get("checkpoint_file_count") == 0,
                entrypoint.get("performance_result_count") == 0,
                entrypoint.get("nullable_metric_contract_required") is True,
                entrypoint.get("nullable_metric_contract_semantic_sha256")
                == nullable_hash,
                entrypoint.get("active_bundle_core_sha256") == expected_core,
            )
        ):
            raise ActiveFormalBundleError(
                "formal training entrypoint acceptance is missing or incomplete"
            )

    if require_ready:
        readiness_row = resources["readiness_companion"]
        observed_ready_hash = canonical_sha256(ready_index_projection(index))
        if index.get("active_formal_bundle_sha256") != observed_ready_hash:
            raise ActiveFormalBundleError("ready active formal bundle SHA-256 mismatch")
        if index.get("readiness_companion", {}).get("content_sha256") != readiness_row.get(
            "content_sha256"
        ):
            raise ActiveFormalBundleError("index Readiness evidence hash drift")

    head = _git(root, "rev-parse", "HEAD")
    if require_clean_git and _git(root, "status", "--porcelain"):
        raise ActiveFormalBundleError("active formal execution requires a clean Git worktree")
    if require_origin_main_match:
        origin_main = _git(root, "rev-parse", "origin/main")
        if head != origin_main:
            raise ActiveFormalBundleError("active formal execution requires HEAD == origin/main")
    commit_binding = index.get("execution_commit_binding")
    if not isinstance(commit_binding, Mapping) or commit_binding.get("mode") != (
        "observed_clean_head_equal_origin_main"
    ):
        raise ActiveFormalBundleError("active bundle execution commit binding is invalid")

    return {
        "status": "pass",
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_core_sha256": expected_core,
        "active_formal_bundle_sha256": index.get("active_formal_bundle_sha256"),
        "protocol_path": str(protocol_file),
        "execution_environment_manifest_path": str(environment_file),
        "protocol": protocol,
        "environment_manifest": environment,
        "index": index,
        "readiness": readiness,
        "resource_ids": resource_ids,
        "execution_commit": head,
        "holdout_capability": False,
        "repository_root": str(root),
        "validation_status": "validated_active_formal_bundle",
        "_validation_token": _VALIDATED_BUNDLE_TOKEN,
    }


def _require_validated_bundle(bundle: Mapping[str, Any]) -> None:
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("_validation_token") is not _VALIDATED_BUNDLE_TOKEN
        or bundle.get("validation_status") != "validated_active_formal_bundle"
        or bundle.get("status") != "pass"
    ):
        raise ActiveFormalBundleError(
            "resource resolver requires validate_active_formal_bundle() output"
        )


def resolve_active_bundle_resource(
    bundle: Mapping[str, Any], logical_id: str, *, expected_role: str | None = None
) -> dict[str, Any]:
    """Resolve one content-addressed resource from a validated active bundle."""

    _require_validated_bundle(bundle)
    rows = bundle["index"].get("active_bundle_resources")
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("logical_id") == logical_id
    ]
    if len(matches) != 1:
        raise ActiveFormalBundleError(
            f"active bundle resource must resolve exactly once: {logical_id}"
        )
    row = matches[0]
    if expected_role is not None and row.get("role") != expected_role:
        raise ActiveFormalBundleError(
            f"active bundle resource role mismatch: {logical_id}"
        )
    root = Path(str(bundle["repository_root"])).resolve()
    resolved = _resolve_registered_path(root, row["logical_path"], logical_id)
    return {
        "logical_id": logical_id,
        "role": row["role"],
        "logical_path": row["logical_path"],
        "resolved_absolute_path": str(resolved),
        "version_scope": row["version_scope"],
        "content_sha256": row["content_sha256"],
        "size_bytes": row["size_bytes"],
        "semantic_sha256": row.get("semantic_sha256"),
        "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
        "validation_status": "validated",
    }


def resolve_active_bundle_group(
    bundle: Mapping[str, Any], prefix: str, *, expected_role: str
) -> list[dict[str, Any]]:
    """Resolve a logical group without depending on inventory order."""

    _require_validated_bundle(bundle)
    normalized = prefix.rstrip(".") + "."
    logical_ids = [
        str(row.get("logical_id"))
        for row in bundle["index"].get("active_bundle_resources", [])
        if isinstance(row, Mapping)
        and str(row.get("logical_id", "")).startswith(normalized)
    ]
    if not logical_ids:
        raise ActiveFormalBundleError(f"active bundle resource group is missing: {prefix}")
    return [
        resolve_active_bundle_resource(bundle, logical_id, expected_role=expected_role)
        for logical_id in sorted(logical_ids)
    ]


def resolve_capacity_resource_pairs(
    bundle: Mapping[str, Any], *, fairness_group: str
) -> list[dict[str, Any]]:
    """Resolve runtime/fairness pairs in the frozen capacity order."""

    fairness_roles = {
        "fairness_manifests": "formal fairness manifest",
        "dev_fairness_manifests": "dev fairness manifest",
        "rehearsal_fairness_manifests": "nonformal rehearsal fairness manifest",
    }
    if fairness_group not in fairness_roles:
        raise ActiveFormalBundleError(f"unsupported capacity fairness group: {fairness_group}")
    runtime = {
        row["logical_id"].split(".", 1)[1]: row
        for row in resolve_active_bundle_group(
            bundle, "runtime_configs", expected_role="typed runtime config"
        )
    }
    fairness = {
        row["logical_id"].split(".", 1)[1]: row
        for row in resolve_active_bundle_group(
            bundle, fairness_group, expected_role=fairness_roles[fairness_group]
        )
    }
    expected = set(CAPACITY_ORDER)
    if set(runtime) != expected or set(fairness) != expected:
        raise ActiveFormalBundleError(
            "capacity resources must contain exactly the frozen three labels"
        )
    return [
        {
            "capacity_label": label,
            "runtime": runtime[label],
            "fairness": fairness[label],
            "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
            "validation_status": "validated",
        }
        for label in CAPACITY_ORDER
    ]


def resolve_support_resource(
    bundle: Mapping[str, Any], setting_id: str
) -> dict[str, Any]:
    return resolve_active_bundle_resource(
        bundle,
        f"support_fairness_manifests.{setting_id}",
        expected_role="support fairness manifest",
    )


def validate_registered_resource_path(
    resource: Mapping[str, Any], supplied_path: str | Path
) -> dict[str, Any]:
    """Fail if an explicit CLI path is not the registered resolved resource."""

    expected = Path(str(resource.get("resolved_absolute_path"))).resolve()
    supplied = Path(supplied_path)
    if not supplied.is_absolute():
        raise ActiveFormalBundleError("active consumer resource path must be absolute")
    if supplied.is_symlink() or supplied.resolve() != expected:
        raise ActiveFormalBundleError(
            f"explicit path differs from registered resource: {resource.get('logical_id')}"
        )
    if sha256_file(supplied) != resource.get("content_sha256"):
        raise ActiveFormalBundleError(
            f"explicit resource content drift: {resource.get('logical_id')}"
        )
    if supplied.stat().st_size != resource.get("size_bytes"):
        raise ActiveFormalBundleError(
            f"explicit resource size drift: {resource.get('logical_id')}"
        )
    return dict(resource)


def build_active_bundle_resource_resolution_audit(
    bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the order-independent inventory audit bound into every active run."""

    _require_validated_bundle(bundle)
    resources = [
        resolve_active_bundle_resource(bundle, logical_id)
        for logical_id in sorted(bundle["resource_ids"])
    ]
    payload = {
        "active_bundle_resource_resolution_contract_version": (
            ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION
        ),
        "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
        "validation_status": "validated",
        "resources": resources,
        "formal_capacity_pairs": resolve_capacity_resource_pairs(
            bundle, fairness_group="fairness_manifests"
        ),
        "dev_capacity_pairs": resolve_capacity_resource_pairs(
            bundle, fairness_group="dev_fairness_manifests"
        ),
        "nonformal_rehearsal_capacity_pairs": resolve_capacity_resource_pairs(
            bundle, fairness_group="rehearsal_fairness_manifests"
        ),
        "support_resources": resolve_active_bundle_group(
            bundle,
            "support_fairness_manifests",
            expected_role="support fairness manifest",
        ),
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION",
    "ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION",
    "ACTIVE_PROTOCOL_ID",
    "ACTIVE_PROTOCOL_VERSION",
    "ActiveFormalBundleError",
    "DEFAULT_ACTIVE_INDEX_RELATIVE",
    "CAPACITY_ORDER",
    "READINESS_VERSION",
    "READY_STATUS",
    "active_bundle_core_projection",
    "build_resource_row",
    "canonical_sha256",
    "ready_index_projection",
    "sha256_file",
    "validate_active_formal_bundle",
    "resolve_active_bundle_resource",
    "resolve_active_bundle_group",
    "resolve_capacity_resource_pairs",
    "resolve_support_resource",
    "validate_registered_resource_path",
    "build_active_bundle_resource_resolution_audit",
]
