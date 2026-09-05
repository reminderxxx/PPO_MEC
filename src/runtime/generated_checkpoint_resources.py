"""Run-local checkpoint resource registry and fail-closed resolver.

The static portable registry describes immutable inputs that exist before a run.
This module deliberately owns a separate, create-only registry that is published
only after the current run's ``checkpoint_freeze`` phase has committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION = "1.0.0"
GENERATED_CHECKPOINT_REGISTRY_SCHEMA_VERSION = "1.0.0"
CAPACITY_MB = {
    "constrained_288mb": 288,
    "medium_576mb": 576,
    "relaxed_864mb": 864,
}
RESOURCE_ROLES = {
    "checkpoint_manifest": "generated_seed_checkpoint_manifest",
    "checkpoint_provenance": "generated_checkpoint_provenance_manifest",
}


class GeneratedCheckpointResourceError(ValueError):
    """Raised when a generated checkpoint resource is not current and exact."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GeneratedCheckpointResourceError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise GeneratedCheckpointResourceError(f"non-string JSON key at {path}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _reject_non_finite(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
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


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GeneratedCheckpointResourceError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GeneratedCheckpointResourceError(
                    f"non-finite JSON value in {label}: {token}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneratedCheckpointResourceError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GeneratedCheckpointResourceError(f"{label} must be a JSON object")
    return value


def _registry_projection(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in registry.items()
        if key != "registry_canonical_sha256"
    }


def _safe_run_relative(run_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise GeneratedCheckpointResourceError(f"{label} path must be run-root-relative")
    lexical = Path(relative)
    if any(part in {"", ".", ".."} for part in lexical.parts):
        raise GeneratedCheckpointResourceError(f"{label} path normalization is forbidden")
    current = run_root
    for part in lexical.parts:
        current = current / part
        if current.is_symlink():
            raise GeneratedCheckpointResourceError(f"symlink is forbidden: {label}")
    resolved = (run_root / lexical).resolve(strict=False)
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as exc:
        raise GeneratedCheckpointResourceError(f"{label} escapes current run root") from exc
    return resolved


def _terminal_checkpoint_freeze_record(run_root: Path) -> tuple[dict[str, Any], str]:
    ledger_path = run_root / "phase_state.jsonl"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise GeneratedCheckpointResourceError(
            "generated registry requires committed phase_state.jsonl"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GeneratedCheckpointResourceError(
                f"invalid phase ledger line {line_number}"
            ) from exc
        if isinstance(row, dict):
            records.append(row)
    matches = [
        row for row in records
        if row.get("phase") == "checkpoint_freeze" and row.get("status") == "completed"
    ]
    if len(matches) != 1:
        raise GeneratedCheckpointResourceError(
            "checkpoint_freeze must have exactly one committed terminal record"
        )
    record = matches[0]
    record_hash = record.get("current_hash") or record.get("record_sha256")
    if not isinstance(record_hash, str) or len(record_hash) != 64:
        record_hash = canonical_sha256(record)
    return record, sha256_file(ledger_path)


def _checkpoint_coverage(
    run_root: Path,
    seed_manifest: Mapping[str, Any],
    provenance_manifest: Mapping[str, Any],
    *,
    capacity_label: str,
) -> list[dict[str, Any]]:
    portable = seed_manifest.get("_portable_checkpoint_manifest")
    if not isinstance(portable, Mapping):
        raise GeneratedCheckpointResourceError("seed manifest lacks portable identity")
    entries = portable.get("entries")
    if not isinstance(entries, list) or not entries:
        raise GeneratedCheckpointResourceError("seed manifest has no frozen checkpoint entries")
    entry_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            raise GeneratedCheckpointResourceError("checkpoint entry must be an object")
        key = (str(item.get("agent")), str(item.get("seed")))
        if key in entry_map:
            raise GeneratedCheckpointResourceError("duplicate frozen checkpoint entry")
        entry_map[key] = item
    coverage: list[dict[str, Any]] = []
    manifest_keys: set[tuple[str, str]] = set()
    for agent, seed_map in seed_manifest.items():
        if str(agent).startswith("_"):
            continue
        if not isinstance(seed_map, Mapping):
            raise GeneratedCheckpointResourceError("seed manifest agent row is invalid")
        for seed, raw_path in seed_map.items():
            key = (str(agent), str(seed))
            if key in manifest_keys:
                raise GeneratedCheckpointResourceError("duplicate seed manifest checkpoint")
            manifest_keys.add(key)
            entry = entry_map.get(key)
            provenance = provenance_manifest.get(str(agent), {})
            binding = provenance.get(str(seed)) if isinstance(provenance, Mapping) else None
            if entry is None or not isinstance(binding, Mapping):
                raise GeneratedCheckpointResourceError("checkpoint provenance coverage is incomplete")
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = run_root / path
            if path.is_symlink():
                raise GeneratedCheckpointResourceError("checkpoint symlink is forbidden")
            resolved = path.resolve(strict=False)
            try:
                relative = resolved.relative_to(run_root.resolve()).as_posix()
            except ValueError as exc:
                raise GeneratedCheckpointResourceError("checkpoint escapes current run root") from exc
            historical_pattern = r"g14c_v(?:[1-9]|1[0-3])(?:\D|$)"
            if re.search(
                historical_pattern,
                f"{run_root.name}/{relative}",
                re.IGNORECASE,
            ):
                raise GeneratedCheckpointResourceError(
                    "G14C v1-v13 checkpoint reference is permanently forbidden"
                )
            if not resolved.is_file():
                raise GeneratedCheckpointResourceError("frozen checkpoint is missing")
            identity = entry.get("checkpoint_identity")
            if not isinstance(identity, Mapping):
                raise GeneratedCheckpointResourceError("checkpoint identity is missing")
            digest = sha256_file(resolved)
            if digest != identity.get("checkpoint_sha256") or digest != binding.get("checkpoint_sha256"):
                raise GeneratedCheckpointResourceError("checkpoint content/provenance hash drift")
            if identity.get("capacity") != capacity_label:
                raise GeneratedCheckpointResourceError("checkpoint capacity identity drift")
            coverage.append(
                {
                    "agent": str(agent),
                    "seed": int(seed),
                    "run_root_relative_path": relative,
                    "size_bytes": resolved.stat().st_size,
                    "content_sha256": digest,
                    "checkpoint_identity_fingerprint": identity.get(
                        "semantic_identity_fingerprint"
                    ),
                }
            )
    if manifest_keys != set(entry_map):
        raise GeneratedCheckpointResourceError("portable checkpoint entries are missing or extra")
    provenance_keys = {
        (str(agent), str(seed))
        for agent, seed_map in provenance_manifest.items()
        if isinstance(seed_map, Mapping)
        for seed, binding in seed_map.items()
        if isinstance(binding, Mapping)
    }
    if provenance_keys != manifest_keys:
        raise GeneratedCheckpointResourceError("checkpoint provenance has missing or extra entries")
    return sorted(coverage, key=lambda row: (row["agent"], row["seed"]))


def build_generated_checkpoint_registry(
    *,
    run_root: str | Path,
    protocol: Mapping[str, Any],
    static_registry: Mapping[str, Any],
    resolved_execution_context: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    freeze = _strict_json_object(root / "checkpoint_freeze.json", "checkpoint freeze")
    terminal, ledger_sha256 = _terminal_checkpoint_freeze_record(root)
    run_id = root.name
    static_ids = {
        str(row.get("logical_resource_id"))
        for row in static_registry.get("resources", [])
        if isinstance(row, Mapping)
    }
    entries: list[dict[str, Any]] = []
    for capacity_label, capacity_mb in CAPACITY_MB.items():
        capacity_root = root / "checkpoint_manifests" / capacity_label
        seed_path = capacity_root / "seed_checkpoint_manifest.json"
        provenance_path = capacity_root / "checkpoint_provenance_manifest.json"
        seed_manifest = _strict_json_object(seed_path, "seed checkpoint manifest")
        provenance_manifest = _strict_json_object(
            provenance_path, "checkpoint provenance manifest"
        )
        coverage = _checkpoint_coverage(
            root, seed_manifest, provenance_manifest, capacity_label=capacity_label
        )
        for resource_type, path, schema_version in (
            ("checkpoint_manifest", seed_path, "1.1.0"),
            ("checkpoint_provenance", provenance_path, "1.0.0"),
        ):
            logical_id = f"{resource_type}.{capacity_label}"
            if logical_id in static_ids:
                raise GeneratedCheckpointResourceError(
                    f"static/generated logical resource collision: {logical_id}"
                )
            entries.append(
                {
                    "logical_resource_id": logical_id,
                    "resource_role": RESOURCE_ROLES[resource_type],
                    "schema_version": schema_version,
                    "capacity_label": capacity_label,
                    "capacity_mb": capacity_mb,
                    "durable_run_root_relative_path": path.resolve().relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "content_sha256": sha256_file(path),
                    "checkpoint_coverage": coverage,
                }
            )
    entries.sort(key=lambda row: row["logical_resource_id"])
    if len({row["logical_resource_id"] for row in entries}) != len(entries):
        raise GeneratedCheckpointResourceError("duplicate generated logical resource ID")
    scientific = resolved_execution_context.get("scientific_identity") or {}
    registry: dict[str, Any] = {
        "formal_generated_checkpoint_resource_identity_contract_version": (
            FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION
        ),
        "generated_checkpoint_registry_schema_version": (
            GENERATED_CHECKPOINT_REGISTRY_SCHEMA_VERSION
        ),
        "registry_id": f"generated-checkpoints-{run_id}",
        "current_run_id": run_id,
        "durable_run_root_name": run_id,
        "create_only": True,
        "atomic_publication": True,
        "static_registry_semantic_sha256": static_registry.get("hashes", {}).get(
            "semantic_sha256"
        ),
        "protocol_semantic_sha256": protocol.get("hashes", {}).get("semantic_sha256"),
        "protocol_full_sha256": protocol.get("hashes", {}).get("full_sha256"),
        "active_formal_bundle_sha256": scientific.get("active_formal_bundle_sha256"),
        "execution_commit": scientific.get("execution_commit"),
        "resolved_execution_context_sha256": resolved_execution_context.get("context_sha256"),
        "agent_scientific_config_semantic_sha256": execution_binding.get(
            "agent_scientific_config_semantic_sha256"
        ),
        "formal_training_execution_binding_sha256": execution_binding.get(
            "binding_full_sha256"
        ),
        "dev_selection_sha256": freeze.get("selection_sha256"),
        "checkpoint_freeze_sha256": freeze.get("freeze_sha256"),
        "source_phase": "checkpoint_freeze",
        "source_phase_committed_ledger_identity": {
            "terminal_record_sha256": terminal.get("current_hash")
            or terminal.get("record_sha256")
            or canonical_sha256(terminal),
            "phase_ledger_file_sha256_at_publication": ledger_sha256,
            "terminal_status": terminal.get("status"),
        },
        "resources": entries,
    }
    required_strings = (
        "static_registry_semantic_sha256",
        "protocol_semantic_sha256",
        "protocol_full_sha256",
        "active_formal_bundle_sha256",
        "execution_commit",
        "resolved_execution_context_sha256",
        "agent_scientific_config_semantic_sha256",
        "formal_training_execution_binding_sha256",
        "dev_selection_sha256",
        "checkpoint_freeze_sha256",
    )
    for field in required_strings:
        value = registry.get(field)
        if not isinstance(value, str) or not value:
            raise GeneratedCheckpointResourceError(
                f"generated checkpoint registry lacks identity: {field}"
            )
    registry["registry_canonical_sha256"] = canonical_sha256(
        _registry_projection(registry)
    )
    return registry


def atomic_create_registry(path: str | Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"generated checkpoint registry is create-only: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        registry, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "published_create_only",
        "path": str(target.resolve()),
        "file_sha256": sha256_file(target),
        "registry_canonical_sha256": registry["registry_canonical_sha256"],
    }


def validate_generated_checkpoint_registry(
    registry: Mapping[str, Any],
    *,
    registry_path: str | Path,
    run_root: str | Path,
    expected_run_id: str,
    static_registry_semantic_sha256: str,
    protocol_semantic_sha256: str | None = None,
    protocol_full_sha256: str | None = None,
    active_formal_bundle_sha256: str | None = None,
    execution_commit: str | None = None,
    resolved_execution_context_sha256: str | None = None,
    formal_training_execution_binding_sha256: str | None = None,
    require_committed_phase: bool = True,
) -> dict[str, Any]:
    _reject_non_finite(registry)
    if registry.get("formal_generated_checkpoint_resource_identity_contract_version") != (
        FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION
    ) or registry.get("generated_checkpoint_registry_schema_version") != (
        GENERATED_CHECKPOINT_REGISTRY_SCHEMA_VERSION
    ):
        raise GeneratedCheckpointResourceError("unsupported generated registry contract/schema")
    expected_hash = canonical_sha256(_registry_projection(registry))
    if registry.get("registry_canonical_sha256") != expected_hash:
        raise GeneratedCheckpointResourceError("generated registry canonical hash mismatch")
    root = Path(run_root).resolve()
    registry_file = Path(registry_path)
    if registry_file.is_symlink():
        raise GeneratedCheckpointResourceError("generated registry symlink is forbidden")
    if registry_file.resolve().parent != root:
        raise GeneratedCheckpointResourceError("generated registry must be in current run root")
    if registry.get("current_run_id") != expected_run_id or root.name != expected_run_id:
        raise GeneratedCheckpointResourceError("cross-run generated registry rejected")
    expected_identity = {
        "static_registry_semantic_sha256": static_registry_semantic_sha256,
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "protocol_full_sha256": protocol_full_sha256,
        "active_formal_bundle_sha256": active_formal_bundle_sha256,
        "execution_commit": execution_commit,
        "resolved_execution_context_sha256": resolved_execution_context_sha256,
        "formal_training_execution_binding_sha256": formal_training_execution_binding_sha256,
    }
    for field, expected in expected_identity.items():
        if expected is not None and registry.get(field) != expected:
            raise GeneratedCheckpointResourceError(f"generated registry identity drift: {field}")
    freeze = _strict_json_object(root / "checkpoint_freeze.json", "checkpoint freeze")
    if registry.get("dev_selection_sha256") != freeze.get("selection_sha256"):
        raise GeneratedCheckpointResourceError(
            "generated registry stale dev selection identity"
        )
    if registry.get("checkpoint_freeze_sha256") != freeze.get("freeze_sha256"):
        raise GeneratedCheckpointResourceError(
            "generated registry stale checkpoint freeze identity"
        )
    resources = registry.get("resources")
    if not isinstance(resources, list) or len(resources) != 6:
        raise GeneratedCheckpointResourceError("generated registry must contain six resources")
    ids: set[str] = set()
    for raw in resources:
        if not isinstance(raw, Mapping):
            raise GeneratedCheckpointResourceError("generated resource row is invalid")
        logical_id = str(raw.get("logical_resource_id") or "")
        if not logical_id or logical_id in ids:
            raise GeneratedCheckpointResourceError("duplicate generated logical resource ID")
        ids.add(logical_id)
        prefix, _, capacity_label = logical_id.partition(".")
        if prefix not in RESOURCE_ROLES or capacity_label not in CAPACITY_MB:
            raise GeneratedCheckpointResourceError("generated logical ID is unknown")
        if raw.get("resource_role") != RESOURCE_ROLES[prefix]:
            raise GeneratedCheckpointResourceError("generated resource role mismatch")
        if raw.get("capacity_label") != capacity_label or raw.get("capacity_mb") != CAPACITY_MB[capacity_label]:
            raise GeneratedCheckpointResourceError("generated resource capacity mismatch")
        target = _safe_run_relative(
            root, raw.get("durable_run_root_relative_path"), logical_id
        )
        if not target.is_file():
            raise GeneratedCheckpointResourceError("generated resource file is missing")
        if target.stat().st_size != raw.get("size_bytes"):
            raise GeneratedCheckpointResourceError("generated resource size drift")
        if sha256_file(target) != raw.get("content_sha256"):
            raise GeneratedCheckpointResourceError("generated resource content hash drift")
        if prefix == "checkpoint_manifest" and raw.get("schema_version") != "1.1.0":
            raise GeneratedCheckpointResourceError("checkpoint manifest schema mismatch")
        if prefix == "checkpoint_provenance" and raw.get("schema_version") != "1.0.0":
            raise GeneratedCheckpointResourceError("checkpoint provenance schema mismatch")
    expected_ids = {
        f"{prefix}.{capacity}"
        for prefix in RESOURCE_ROLES
        for capacity in CAPACITY_MB
    }
    if ids != expected_ids:
        raise GeneratedCheckpointResourceError("generated registry resource membership drift")
    if require_committed_phase:
        terminal, _ = _terminal_checkpoint_freeze_record(root)
        observed = registry.get("source_phase_committed_ledger_identity", {}).get(
            "terminal_record_sha256"
        )
        actual = terminal.get("current_hash") or terminal.get("record_sha256") or canonical_sha256(terminal)
        if observed != actual:
            raise GeneratedCheckpointResourceError("checkpoint-freeze ledger identity drift")
    return {
        "status": "pass",
        "registry_canonical_sha256": expected_hash,
        "current_run_id": expected_run_id,
        "resource_count": len(resources),
    }


def load_generated_checkpoint_registry(
    path: str | Path, **validation: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    registry = _strict_json_object(target, "generated checkpoint registry")
    return registry, validate_generated_checkpoint_registry(
        registry, registry_path=target, **validation
    )


def resolve_generated_checkpoint_resource(
    registry: Mapping[str, Any],
    *,
    run_root: str | Path,
    logical_resource_id: str,
    expected_role: str,
    explicit_path: str | Path,
    expected_capacity_label: str | None = None,
) -> dict[str, Any]:
    matches = [
        dict(row) for row in registry.get("resources", [])
        if isinstance(row, Mapping)
        and row.get("logical_resource_id") == logical_resource_id
    ]
    if len(matches) != 1:
        raise GeneratedCheckpointResourceError(
            f"unknown or duplicate generated logical resource ID: {logical_resource_id}"
        )
    row = matches[0]
    if row.get("resource_role") != expected_role:
        raise GeneratedCheckpointResourceError("generated resource role mismatch")
    if expected_capacity_label is not None and row.get("capacity_label") != expected_capacity_label:
        raise GeneratedCheckpointResourceError("generated resource capacity mismatch")
    resolved = _safe_run_relative(
        Path(run_root).resolve(), row["durable_run_root_relative_path"], logical_resource_id
    )
    supplied = Path(explicit_path)
    if not supplied.is_absolute() or supplied.is_symlink() or supplied.resolve() != resolved:
        raise GeneratedCheckpointResourceError("explicit path/generated logical resource conflict")
    if sha256_file(supplied) != row.get("content_sha256") or supplied.stat().st_size != row.get("size_bytes"):
        raise GeneratedCheckpointResourceError("generated resource content identity drift")
    return {
        "status": "compatible",
        "logical_resource_id": logical_resource_id,
        "resource_role": expected_role,
        "capacity_label": row["capacity_label"],
        "capacity_mb": row["capacity_mb"],
        "resolved_path": str(resolved),
        "content_sha256": row["content_sha256"],
        "size_bytes": row["size_bytes"],
        "checkpoint_count": len(row.get("checkpoint_coverage", [])),
    }


def add_generated_checkpoint_resource_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generated-checkpoint-registry-path", default="")
    parser.add_argument("--checkpoint-provenance-id", default="")


def resolve_generated_checkpoint_arguments(
    args: argparse.Namespace,
    *,
    expected_capacity_label: str | None = None,
    protocol: Mapping[str, Any] | None = None,
    resolved_execution_context: Mapping[str, Any] | None = None,
    execution_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_path = str(getattr(args, "generated_checkpoint_registry_path", "") or "")
    if not registry_path:
        raise GeneratedCheckpointResourceError(
            "generated checkpoint registry is required before checkpoint access"
        )
    run_root = Path(registry_path).resolve().parent
    static_registry_path = str(getattr(args, "resource_registry_path", "") or "")
    if not static_registry_path:
        raise GeneratedCheckpointResourceError("static resource registry is required")
    static = _strict_json_object(Path(static_registry_path), "static resource registry")
    registry, validation = load_generated_checkpoint_registry(
        registry_path,
        run_root=run_root,
        expected_run_id=run_root.name,
        static_registry_semantic_sha256=str(static.get("hashes", {}).get("semantic_sha256") or ""),
        protocol_semantic_sha256=(protocol or {}).get("hashes", {}).get("semantic_sha256"),
        protocol_full_sha256=(protocol or {}).get("hashes", {}).get("full_sha256"),
        active_formal_bundle_sha256=(resolved_execution_context or {}).get("scientific_identity", {}).get("active_formal_bundle_sha256"),
        execution_commit=(resolved_execution_context or {}).get("scientific_identity", {}).get("execution_commit"),
        resolved_execution_context_sha256=(resolved_execution_context or {}).get("context_sha256"),
        formal_training_execution_binding_sha256=(execution_binding or {}).get("binding_full_sha256"),
    )
    manifest_id = str(getattr(args, "checkpoint_manifest_id", "") or "")
    provenance_id = str(getattr(args, "checkpoint_provenance_id", "") or "")
    if not manifest_id or not provenance_id:
        raise GeneratedCheckpointResourceError(
            "checkpoint manifest and provenance logical IDs are required"
        )
    manifest = resolve_generated_checkpoint_resource(
        registry,
        run_root=run_root,
        logical_resource_id=manifest_id,
        expected_role=RESOURCE_ROLES["checkpoint_manifest"],
        explicit_path=getattr(args, "seed_checkpoint_manifest_path"),
        expected_capacity_label=expected_capacity_label,
    )
    provenance = resolve_generated_checkpoint_resource(
        registry,
        run_root=run_root,
        logical_resource_id=provenance_id,
        expected_role=RESOURCE_ROLES["checkpoint_provenance"],
        explicit_path=getattr(args, "checkpoint_provenance_manifest_path"),
        expected_capacity_label=expected_capacity_label,
    )
    if manifest["capacity_label"] != provenance["capacity_label"]:
        raise GeneratedCheckpointResourceError("manifest/provenance capacity mismatch")
    audit = {
        **validation,
        "generated_checkpoint_registry_path": str(Path(registry_path).resolve()),
        "manifest": manifest,
        "provenance": provenance,
        "capacity_label": manifest["capacity_label"],
    }
    setattr(args, "_generated_checkpoint_resource_resolution", audit)
    return audit


def required_forwarded_flags() -> tuple[str, ...]:
    return (
        "--resource-registry-path",
        "--repository-root",
        "--data-root",
        "--protocol-artifact-root",
        "--checkpoint-root",
        "--mobility-resource-id",
        "--workflow-resource-id",
        "--window-plan-resource-id",
        "--runtime-config-resource-id",
        "--fairness-manifest-resource-id",
        "--generated-checkpoint-registry-path",
        "--checkpoint-manifest-id",
        "--checkpoint-provenance-id",
    )


def audit_forwarded_resource_arguments(
    command: Sequence[str], args: argparse.Namespace
) -> dict[str, Any]:
    command_list = list(command)
    values: dict[str, str] = {}
    for flag in required_forwarded_flags():
        if command_list.count(flag) != 1:
            raise GeneratedCheckpointResourceError(
                f"nested consumer must consume exactly one {flag}"
            )
        index = command_list.index(flag)
        if index + 1 >= len(command_list):
            raise GeneratedCheckpointResourceError(f"nested resource flag lacks value: {flag}")
        child = command_list[index + 1]
        attribute = flag[2:].replace("-", "_")
        outer = str(getattr(args, attribute, "") or "")
        if child != outer:
            raise GeneratedCheckpointResourceError(
                f"outer/nested resource argument mismatch: {flag}"
            )
        values[flag] = child
    return {"status": "pass", "forwarded_flag_count": len(values), "values": values}


__all__ = [
    "CAPACITY_MB",
    "FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION",
    "GENERATED_CHECKPOINT_REGISTRY_SCHEMA_VERSION",
    "GeneratedCheckpointResourceError",
    "add_generated_checkpoint_resource_arguments",
    "atomic_create_registry",
    "audit_forwarded_resource_arguments",
    "build_generated_checkpoint_registry",
    "canonical_sha256",
    "load_generated_checkpoint_registry",
    "resolve_generated_checkpoint_arguments",
    "resolve_generated_checkpoint_resource",
    "sha256_file",
    "validate_generated_checkpoint_registry",
]
