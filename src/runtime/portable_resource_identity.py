"""Portable scientific identity and runtime path resolution for formal resources.

Scientific identity is content-addressed.  Host absolute paths are recorded only
in the runtime audit and never participate in the semantic fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PORTABLE_RESOURCE_IDENTITY_CONTRACT_VERSION = "1.0.0"
RESOURCE_RESOLVER_VERSION = "1.0.0"
ALLOWED_RESOLVERS = (
    "explicit_path",
    "data_root",
    "worktree_root",
    "manifest_relative",
    "protocol_artifact_root",
    "checkpoint_root",
)
SEMANTIC_IDENTITY_FIELDS = (
    "logical_resource_id",
    "resource_role",
    "content_sha256",
    "size_bytes",
    "schema_version",
    "revision",
    "expected_logical_relative_path",
    "required",
    "allowed_resolvers",
    "provenance",
    "path_relocation_allowed",
)


class PortableResourceError(ValueError):
    """Raised when a resource cannot be resolved without semantic drift."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PortableResourceError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PortableResourceError(f"non-string JSON key at {path}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _reject_non_finite(value)
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


def scientific_identity_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(identity.get(field)) for field in SEMANTIC_IDENTITY_FIELDS}


def scientific_identity_fingerprint(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(scientific_identity_projection(identity))


def build_resource_identity(
    path: str | Path,
    *,
    logical_resource_id: str,
    resource_role: str,
    schema_version: str,
    revision: str,
    expected_logical_relative_path: str,
    required: bool = True,
    allowed_resolvers: Sequence[str] = ALLOWED_RESOLVERS,
    provenance: Mapping[str, Any] | str = "repository_or_explicit_data_root",
    path_relocation_allowed: bool = True,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    unknown = sorted(set(allowed_resolvers) - set(ALLOWED_RESOLVERS))
    if unknown:
        raise PortableResourceError(f"unknown allowed resolver(s): {unknown}")
    identity = {
        "logical_resource_id": str(logical_resource_id),
        "resource_role": str(resource_role),
        "content_sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "schema_version": str(schema_version),
        "revision": str(revision),
        "expected_logical_relative_path": Path(
            expected_logical_relative_path
        ).as_posix(),
        "required": bool(required),
        "allowed_resolvers": list(allowed_resolvers),
        "provenance": deepcopy(provenance),
        "path_relocation_allowed": bool(path_relocation_allowed),
    }
    identity["semantic_identity_fingerprint"] = scientific_identity_fingerprint(
        identity
    )
    return identity


def validate_resource_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(identity)
    missing = [field for field in SEMANTIC_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise PortableResourceError(f"resource identity missing fields: {missing}")
    logical_id = str(identity.get("logical_resource_id") or "")
    role = str(identity.get("resource_role") or "")
    digest = str(identity.get("content_sha256") or "")
    if not logical_id or not role:
        raise PortableResourceError("logical resource ID and role are required")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PortableResourceError("resource content_sha256 is invalid")
    size = identity.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise PortableResourceError("resource size_bytes is invalid")
    allowed = identity.get("allowed_resolvers")
    if not isinstance(allowed, list) or not allowed:
        raise PortableResourceError("resource allowed_resolvers must be non-empty")
    unknown = sorted(set(allowed) - set(ALLOWED_RESOLVERS))
    if unknown:
        raise PortableResourceError(f"unknown allowed resolver(s): {unknown}")
    observed = identity.get("semantic_identity_fingerprint")
    expected = scientific_identity_fingerprint(identity)
    if observed not in {None, expected}:
        raise PortableResourceError("resource semantic identity fingerprint mismatch")
    return {
        "status": "pass",
        "logical_resource_id": logical_id,
        "resource_role": role,
        "semantic_identity_fingerprint": expected,
    }


def build_registry(
    resources: Iterable[Mapping[str, Any]],
    *,
    registry_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    rows = []
    seen: set[str] = set()
    for raw in resources:
        identity = dict(raw)
        report = validate_resource_identity(identity)
        logical_id = report["logical_resource_id"]
        if logical_id in seen:
            raise PortableResourceError(f"duplicate logical resource ID: {logical_id}")
        seen.add(logical_id)
        identity["semantic_identity_fingerprint"] = report[
            "semantic_identity_fingerprint"
        ]
        rows.append(identity)
    rows.sort(key=lambda item: item["logical_resource_id"])
    semantic = {
        "portable_resource_identity_contract_version": PORTABLE_RESOURCE_IDENTITY_CONTRACT_VERSION,
        "resource_resolver_version": RESOURCE_RESOLVER_VERSION,
        "scientific_identity_rule": "scientific identity != host absolute path",
        "resources": rows,
    }
    return {
        **semantic,
        "registry_id": str(registry_id),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hashes": {
            "semantic_sha256": canonical_sha256(semantic),
            "semantic_exclusions": ["created_at", "registry_id", "runtime_resolution", "hashes"],
        },
    }


def validate_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(registry)
    if (
        registry.get("portable_resource_identity_contract_version")
        != PORTABLE_RESOURCE_IDENTITY_CONTRACT_VERSION
    ):
        raise PortableResourceError("unsupported portable resource identity contract")
    if registry.get("resource_resolver_version") != RESOURCE_RESOLVER_VERSION:
        raise PortableResourceError("unsupported resource resolver version")
    resources = registry.get("resources")
    if not isinstance(resources, list) or not resources:
        raise PortableResourceError("resource registry must contain resources")
    rebuilt = build_registry(
        resources,
        registry_id=str(registry.get("registry_id") or "validation"),
        created_at=str(registry.get("created_at") or "validation"),
    )
    expected = rebuilt["hashes"]["semantic_sha256"]
    if registry.get("hashes", {}).get("semantic_sha256") != expected:
        raise PortableResourceError("resource registry semantic hash mismatch")
    return {
        "status": "pass",
        "resource_count": len(resources),
        "semantic_sha256": expected,
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableResourceError(f"unable to load resource registry: {target}") from exc
    if not isinstance(payload, dict):
        raise PortableResourceError("resource registry must be a JSON object")
    validate_registry(payload)
    return payload


def _registry_entry(
    registry: Mapping[str, Any], logical_resource_id: str
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in registry.get("resources", [])
        if item.get("logical_resource_id") == logical_resource_id
    ]
    if len(matches) != 1:
        raise PortableResourceError(
            f"unknown or duplicate logical resource ID: {logical_resource_id}"
        )
    return matches[0]


def _candidate_rows(
    identity: Mapping[str, Any],
    *,
    explicit_paths: Iterable[str | Path] = (),
    roots: Mapping[str, str | Path | None] | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    relative = Path(str(identity["expected_logical_relative_path"]))

    def add(path: str | Path | None, method: str, root: str | Path | None) -> None:
        if path in {None, ""}:
            return
        candidate = Path(path).expanduser()
        key = candidate.absolute().as_posix()
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "candidate_path": key,
                "resolution_method": method,
                "resolution_root": str(Path(root).absolute()) if root not in {None, ""} else None,
            }
        )

    for path in explicit_paths:
        add(path, "explicit_path", None)
    allowed = set(identity.get("allowed_resolvers") or [])
    for method, root in (roots or {}).items():
        if method in allowed and root not in {None, ""}:
            add(Path(root) / relative, method, root)
    if manifest_path and "manifest_relative" in allowed:
        add(Path(manifest_path).resolve().parent / relative, "manifest_relative", Path(manifest_path).resolve().parent)
    return rows


def resolve_resource(
    registry: Mapping[str, Any],
    logical_resource_id: str,
    *,
    expected_role: str | None = None,
    explicit_paths: Iterable[str | Path] = (),
    roots: Mapping[str, str | Path | None] | None = None,
    manifest_path: str | Path | None = None,
    observed_schema_version: str | None = None,
) -> dict[str, Any]:
    """Resolve one content-addressed resource and return a JSON-safe audit."""

    validate_registry(registry)
    identity = _registry_entry(registry, logical_resource_id)
    validate_resource_identity(identity)
    if expected_role is not None and identity["resource_role"] != expected_role:
        raise PortableResourceError(
            f"resource role mismatch: {identity['resource_role']} != {expected_role}"
        )
    if (
        observed_schema_version is not None
        and observed_schema_version != identity["schema_version"]
    ):
        raise PortableResourceError("resource schema version mismatch")
    candidates = _candidate_rows(
        identity,
        explicit_paths=explicit_paths,
        roots=roots,
        manifest_path=manifest_path,
    )
    observations: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    existing_digests: set[tuple[str, int]] = set()
    for row in candidates:
        lexical = Path(row["candidate_path"])
        exists = lexical.is_file()
        resolved = lexical.resolve(strict=False)
        observation = {
            **row,
            "exists": exists,
            "is_symlink": lexical.is_symlink(),
            "resolved_absolute_path": resolved.as_posix(),
            "observed_sha256": None,
            "observed_size_bytes": None,
            "compatible": False,
            "mismatch_reason": "missing" if not exists else None,
        }
        if exists:
            size = resolved.stat().st_size
            digest = sha256_file(resolved)
            observation["observed_size_bytes"] = size
            observation["observed_sha256"] = digest
            existing_digests.add((digest, size))
            if size != identity["size_bytes"]:
                observation["mismatch_reason"] = "size_mismatch"
            elif digest != identity["content_sha256"]:
                observation["mismatch_reason"] = "content_sha256_mismatch"
            else:
                observation["compatible"] = True
                observation["mismatch_reason"] = None
                compatible.append(observation)
        observations.append(observation)
    if len(existing_digests) > 1:
        raise PortableResourceError(
            f"conflicting resource candidates: {logical_resource_id}"
        )
    if not compatible:
        required = bool(identity.get("required", True))
        if required:
            reasons = sorted(
                {
                    str(item.get("mismatch_reason"))
                    for item in observations
                    if item.get("mismatch_reason")
                }
            )
            raise PortableResourceError(
                f"resource resolution failed: {logical_resource_id}: {reasons or ['no_candidates']}"
            )
        return {
            "portable_resource_resolution_version": RESOURCE_RESOLVER_VERSION,
            "logical_resource_id": logical_resource_id,
            "resource_role": identity["resource_role"],
            "semantic_identity_fingerprint": identity[
                "semantic_identity_fingerprint"
            ],
            "status": "optional_unavailable",
            "resolved_path": None,
            "observations": observations,
        }
    chosen = compatible[0]
    return {
        "portable_resource_resolution_version": RESOURCE_RESOLVER_VERSION,
        "logical_resource_id": logical_resource_id,
        "resource_role": identity["resource_role"],
        "semantic_identity_fingerprint": identity["semantic_identity_fingerprint"],
        "status": "compatible",
        "portability_status": (
            "allowed_content_identical_path_relocation"
            if len({item["resolved_absolute_path"] for item in compatible}) > 1
            or chosen["candidate_path"] != chosen["resolved_absolute_path"]
            else "exact_or_single_content_identity_match"
        ),
        "resolved_path": chosen["resolved_absolute_path"],
        "resolution_root": chosen["resolution_root"],
        "resolution_method": chosen["resolution_method"],
        "observed_sha256": chosen["observed_sha256"],
        "observed_size_bytes": chosen["observed_size_bytes"],
        "symlink_audit": {
            "lexical_path": chosen["candidate_path"],
            "is_symlink": chosen["is_symlink"],
            "resolved_path": chosen["resolved_absolute_path"],
        },
        "observations": observations,
        "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def add_portable_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared path-resolution flags without changing legacy defaults."""

    parser.add_argument("--resource-registry-path", "--resource_registry_path", default="")
    parser.add_argument("--repository-root", "--repository_root", default="")
    parser.add_argument("--data-root", "--data_root", default="")
    parser.add_argument("--protocol-artifact-root", "--protocol_artifact_root", default="")
    parser.add_argument("--checkpoint-root", "--checkpoint_root", default="")
    parser.add_argument("--mobility-resource-id", "--mobility_resource_id", default="")
    parser.add_argument("--workflow-resource-id", "--workflow_resource_id", default="")
    parser.add_argument("--catalog-resource-id", "--catalog_resource_id", default="")
    parser.add_argument("--window-plan-resource-id", "--window_plan_resource_id", default="")
    parser.add_argument("--window-contract-resource-id", "--window_contract_resource_id", default="")
    parser.add_argument("--protocol-resource-id", "--protocol_resource_id", default="")
    parser.add_argument("--fairness-manifest-resource-id", "--fairness_manifest_resource_id", default="")
    parser.add_argument("--runtime-config-resource-id", "--runtime_config_resource_id", default="")
    parser.add_argument("--checkpoint-manifest-id", "--checkpoint_manifest_id", default="")


def resolve_argument_resources(
    args: argparse.Namespace,
    *,
    bindings: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Resolve ``(id attribute, path attribute, role)`` bindings in-place."""

    registry_path = str(getattr(args, "resource_registry_path", "") or "")
    if not registry_path:
        return {"status": "legacy_no_resource_registry", "resolutions": []}
    registry = load_registry(registry_path)
    roots = {
        "worktree_root": getattr(args, "repository_root", "") or None,
        "data_root": getattr(args, "data_root", "") or None,
        "protocol_artifact_root": getattr(args, "protocol_artifact_root", "") or None,
        "checkpoint_root": getattr(args, "checkpoint_root", "") or None,
    }
    audits = []
    for id_attribute, path_attribute, expected_role in bindings:
        logical_id = str(getattr(args, id_attribute, "") or "")
        if not logical_id:
            raise PortableResourceError(
                f"portable resource binding requires --{id_attribute.replace('_', '-')}"
            )
        explicit = str(getattr(args, path_attribute, "") or "")
        audit = resolve_resource(
            registry,
            logical_id,
            expected_role=expected_role,
            explicit_paths=[explicit] if explicit else [],
            roots=roots,
            manifest_path=registry_path,
        )
        setattr(args, path_attribute, audit["resolved_path"])
        audits.append(audit)
    result = {
        "status": "pass",
        "resource_registry_path": str(Path(registry_path).resolve()),
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "resolutions": audits,
    }
    setattr(args, "_portable_resource_resolution", result)
    return result


__all__ = [
    "ALLOWED_RESOLVERS",
    "PORTABLE_RESOURCE_IDENTITY_CONTRACT_VERSION",
    "RESOURCE_RESOLVER_VERSION",
    "PortableResourceError",
    "add_portable_resource_arguments",
    "build_registry",
    "build_resource_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_registry",
    "resolve_argument_resources",
    "resolve_resource",
    "scientific_identity_fingerprint",
    "scientific_identity_projection",
    "sha256_file",
    "validate_registry",
    "validate_resource_identity",
]
