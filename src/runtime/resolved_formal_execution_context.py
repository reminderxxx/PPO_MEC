"""Immutable resolved execution context for formal Protocol v1.5 runs.

The outer protocol runner is the only producer.  Nested consumers load this
artifact instead of reconstructing host paths or selecting an interpreter.
Scientific identities remain separate from runtime locations, while the full
context hash deliberately binds the latter to one durable run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION = "1.0.0"
RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION = "2.0.0"
RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME = "resolved_execution_context.json"


class ResolvedFormalExecutionContextError(ValueError):
    """Raised when a resolved formal execution context is absent or drifts."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ResolvedFormalExecutionContextError(
            f"non-finite resolved execution context value at {path}"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResolvedFormalExecutionContextError(
                    f"non-string resolved execution context key at {path}"
                )
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


def _context_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "context_sha256"
    }


def _require_absolute_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResolvedFormalExecutionContextError(
            f"resolved execution context lacks {field}"
        )
    path = Path(value)
    if not path.is_absolute():
        raise ResolvedFormalExecutionContextError(
            f"resolved execution context path is not absolute: {field}"
        )
    return str(path)


def _git_identity(root: Path) -> tuple[str, str]:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if top.returncode != 0 or head.returncode != 0:
        raise ResolvedFormalExecutionContextError(
            "resolved clean worktree is not a Git worktree"
        )
    return str(Path(top.stdout.strip()).resolve()), head.stdout.strip()


def build_resolved_formal_execution_context(
    *,
    protocol: Mapping[str, Any],
    expansion_context: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    runtime_audit: Mapping[str, Any],
    environment_manifest_path: str | Path,
    outer_expansion_sha256: str,
    phase_count: int,
    command_count: int,
    execution_binding: Mapping[str, Any] | None = None,
    active_formal_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one full host-bound context after environment resolution."""

    protocol_path = Path(str(expansion_context["protocol_path"])).resolve()
    clean_root = Path(str(expansion_context["clean_worktree_root"])).resolve()
    output_root = Path(str(expansion_context["output_root"])).resolve()
    context_path = Path(
        str(expansion_context["resolved_execution_context_path"])
    ).resolve()
    manifest_path = Path(environment_manifest_path).resolve()
    python = Path(str(expansion_context["python_executable"])).absolute()
    registry_sha = protocol["portable_resource_identity_contract"][
        "resource_registry_semantic_sha256"
    ]
    window_sha = protocol["execution_contract"]["window_consumption_contract"][
        "semantic_sha256"
    ]
    execution_commit = str(runtime_audit.get("observed_execution_commit") or "")
    context_contract_version = str(
        protocol.get("resolved_formal_execution_context_contract", {}).get(
            "version", RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION
        )
    )
    if context_contract_version == RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION:
        if not isinstance(execution_binding, Mapping):
            raise ResolvedFormalExecutionContextError(
                "resolved context v2 requires formal training execution binding"
            )
        binding_sha256 = str(execution_binding.get("binding_full_sha256") or "")
        scientific_config_sha256 = str(
            execution_binding.get("agent_scientific_config_semantic_sha256") or ""
        )
        if not binding_sha256 or not scientific_config_sha256:
            raise ResolvedFormalExecutionContextError(
                "formal training execution binding identity is incomplete"
            )
        if protocol.get("typed_model_cache_formal_protocol_version") in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
            if active_formal_bundle_sha256 != execution_binding.get(
                "active_formal_bundle_sha256"
            ):
                raise ResolvedFormalExecutionContextError(
                    "resolved context active formal bundle identity drift"
                )
    else:
        binding_sha256 = None
        scientific_config_sha256 = None
    created_for_run_identity = canonical_sha256(
        {
            "execution_commit": execution_commit,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "portable_registry_semantic_sha256": registry_sha,
            "environment_fingerprint": environment_identity[
                "environment_fingerprint"
            ],
            "clean_worktree_root": str(clean_root),
            "durable_run_root": str(output_root),
        }
    )
    payload: dict[str, Any] = {
        "resolved_formal_execution_context_version": context_contract_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_for_run_identity": created_for_run_identity,
        "scientific_identity": {
            "execution_commit": execution_commit,
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol[
                "typed_model_cache_formal_protocol_version"
            ],
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "portable_registry_semantic_sha256": registry_sha,
            "split_semantic_sha256": protocol["identity"][
                "split_semantic_sha256"
            ],
            "window_contract_semantic_sha256": window_sha,
            "catalog_fingerprint": protocol["identity"]["catalog_fingerprint"],
            "typed_runtime_identities": protocol["identity"][
                "typed_runtime_contract_hashes_by_capacity"
            ],
            "environment_fingerprint": environment_identity[
                "environment_fingerprint"
            ],
            "dependency_fingerprint": environment_identity[
                "dependency_fingerprint"
            ],
            "agent_scientific_config_semantic_sha256": scientific_config_sha256,
            "formal_training_execution_binding_sha256": binding_sha256,
            "host_paths_are_scientific_identity": False,
        },
        "runtime_location": {
            "resolved_python_absolute_path": str(python),
            "python_resolution_source": runtime_audit.get("resolution_source"),
            "clean_worktree_root": str(clean_root),
            "durable_run_root": str(output_root),
            "protocol_path": str(protocol_path),
            "repository_root": str(
                Path(str(expansion_context["repository_root"])).resolve()
            ),
            "data_root": str(Path(str(expansion_context["data_root"])).resolve()),
            "checkpoint_root": str(
                Path(str(expansion_context["checkpoint_root"])).resolve()
            ),
            "protocol_artifact_root": str(
                Path(str(expansion_context["protocol_artifact_root"])).resolve()
            ),
            "resource_registry_path": str(
                Path(str(expansion_context["resource_registry_path"])).resolve()
            ),
            "execution_environment_manifest_path": str(manifest_path),
            "execution_environment_manifest_sha256": sha256_file(manifest_path),
            "resolved_execution_context_path": str(context_path),
            "formal_training_execution_binding_path": str(
                Path(str(expansion_context["formal_training_execution_binding_path"])).resolve()
            ) if context_contract_version == RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION else None,
            "cwd_guessing_allowed": False,
            "implicit_sys_executable_fallback_allowed": False,
            "relative_venv_fallback_allowed": False,
        },
        "command_expansion": {
            "outer_expansion_sha256": str(outer_expansion_sha256),
            "resolved_command_matrix_sha256": str(outer_expansion_sha256),
            "phase_count": int(phase_count),
            "command_count": int(command_count),
            "canonical_serialization": "UTF-8 sorted-key compact JSON; NaN/Infinity rejected",
        },
        "resolved_expansion_context": dict(expansion_context),
        "portable_resources": {
            "registry_semantic_sha256": registry_sha,
            "resolver_roots_are_explicit": True,
            "network_or_cwd_discovery_allowed": False,
        },
    }
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        payload["scientific_identity"][
            "formal_agent_order_contract_semantic_sha256"
        ] = protocol["formal_agent_order_contract"]["semantic_sha256"]
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        payload["scientific_identity"]["formal_exogenous_request_execution"] = deepcopy(
            protocol["formal_exogenous_request_execution_contract"]
        )
    if protocol.get("typed_model_cache_formal_protocol_version") == "2.3.0":
        payload["scientific_identity"][
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.1.0", "2.2.0", "2.3.0"}:
        payload["scientific_identity"][
            "environment_identity_projection_contract_version"
        ] = protocol["formal_execution_environment_contract"][
            "identity_projection_contract_version"
        ]
        payload["scientific_identity"][
            "full_normalized_environment_projection"
        ] = deepcopy(dict(environment_identity))
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        if not isinstance(active_formal_bundle_sha256, str) or len(
            active_formal_bundle_sha256
        ) != 64:
            raise ResolvedFormalExecutionContextError(
                "active resolved context requires active formal bundle SHA-256"
            )
        payload["scientific_identity"]["active_formal_bundle_sha256"] = (
            active_formal_bundle_sha256
        )
    payload["context_sha256"] = canonical_sha256(_context_projection(payload))
    validate_resolved_formal_execution_context(payload)
    return payload


def validate_resolved_formal_execution_context(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    clean_worktree_root: str | Path | None = None,
    durable_run_root: str | Path | None = None,
    environment_identity: Mapping[str, Any] | None = None,
    runtime_audit: Mapping[str, Any] | None = None,
    check_git: bool = False,
) -> dict[str, Any]:
    """Validate hash, identities, host paths, and optional live bindings."""

    _reject_non_finite(payload)
    context_version = payload.get("resolved_formal_execution_context_version")
    if context_version not in {
        RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION,
        RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION,
    }:
        raise ResolvedFormalExecutionContextError(
            "resolved formal execution context version mismatch"
        )
    expected_hash = canonical_sha256(_context_projection(payload))
    if payload.get("context_sha256") != expected_hash:
        raise ResolvedFormalExecutionContextError(
            "resolved formal execution context SHA-256 mismatch"
        )
    runtime = payload.get("runtime_location")
    scientific = payload.get("scientific_identity")
    expansion = payload.get("resolved_expansion_context")
    command = payload.get("command_expansion")
    if not all(isinstance(item, Mapping) for item in (runtime, scientific, expansion, command)):
        raise ResolvedFormalExecutionContextError(
            "resolved formal execution context sections are incomplete"
        )
    required_runtime_paths = (
        "resolved_python_absolute_path",
        "clean_worktree_root",
        "durable_run_root",
        "protocol_path",
        "repository_root",
        "data_root",
        "checkpoint_root",
        "protocol_artifact_root",
        "resource_registry_path",
        "execution_environment_manifest_path",
        "resolved_execution_context_path",
    )
    if context_version == RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION:
        required_runtime_paths = (*required_runtime_paths, "formal_training_execution_binding_path")
        if not scientific.get("agent_scientific_config_semantic_sha256"):
            raise ResolvedFormalExecutionContextError(
                "resolved context lacks scientific config identity"
            )
        if not scientific.get("formal_training_execution_binding_sha256"):
            raise ResolvedFormalExecutionContextError(
                "resolved context lacks execution binding identity"
            )
    for field in required_runtime_paths:
        _require_absolute_path(runtime.get(field), field)
    python = Path(str(runtime["resolved_python_absolute_path"]))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ResolvedFormalExecutionContextError(
            "resolved Python is missing or not executable"
        )
    if str(runtime["resolved_python_absolute_path"]).startswith(".venv/"):
        raise ResolvedFormalExecutionContextError(
            "relative .venv Python fallback is forbidden"
        )
    serialized = canonical_json_bytes(payload).decode("utf-8")
    if "/ABSOLUTE/" in serialized:
        raise ResolvedFormalExecutionContextError(
            "resolved formal execution context contains /ABSOLUTE/ sentinel"
        )
    for field, value in expansion.items():
        if isinstance(value, str) and ("{" in value or "}" in value):
            raise ResolvedFormalExecutionContextError(
                "resolved formal execution context contains an unresolved "
                f"placeholder: {field}"
            )
    if "{" in str(command.get("outer_expansion_sha256", "")):
        raise ResolvedFormalExecutionContextError(
            "resolved command expansion contains an unresolved placeholder"
        )
    context_path = Path(str(runtime["resolved_execution_context_path"])).resolve()
    expected_context_path = (
        Path(str(runtime["durable_run_root"])).resolve()
        / RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME
    )
    if context_path != expected_context_path:
        raise ResolvedFormalExecutionContextError(
            "resolved context path is not bound to the durable run root"
        )
    if protocol is not None:
        comparisons = {
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol[
                "typed_model_cache_formal_protocol_version"
            ],
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "portable_registry_semantic_sha256": protocol[
                "portable_resource_identity_contract"
            ]["resource_registry_semantic_sha256"],
            "split_semantic_sha256": protocol["identity"][
                "split_semantic_sha256"
            ],
            "window_contract_semantic_sha256": protocol["execution_contract"][
                "window_consumption_contract"
            ]["semantic_sha256"],
            "catalog_fingerprint": protocol["identity"]["catalog_fingerprint"],
            "typed_runtime_identities": protocol["identity"][
                "typed_runtime_contract_hashes_by_capacity"
            ],
        }
        if protocol.get("typed_model_cache_formal_protocol_version") in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
            comparisons["formal_agent_order_contract_semantic_sha256"] = protocol[
                "formal_agent_order_contract"
            ]["semantic_sha256"]
        if protocol.get("typed_model_cache_formal_protocol_version") in {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
            comparisons["formal_exogenous_request_execution"] = protocol[
                "formal_exogenous_request_execution_contract"
            ]
        if protocol.get("typed_model_cache_formal_protocol_version") == "2.3.0":
            comparisons[
                "formal_nullable_metric_aggregation_contract_semantic_sha256"
            ] = protocol["formal_nullable_metric_aggregation_contract"][
                "semantic_sha256"
            ]
        if protocol.get("typed_model_cache_formal_protocol_version") in {"2.1.0", "2.2.0", "2.3.0"}:
            comparisons["environment_identity_projection_contract_version"] = protocol[
                "formal_execution_environment_contract"
            ]["identity_projection_contract_version"]
            if environment_identity is not None:
                comparisons["full_normalized_environment_projection"] = dict(
                    environment_identity
                )
        if protocol.get("typed_model_cache_formal_protocol_version") in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
            expected_bundle = payload.get("resolved_expansion_context", {}).get(
                "active_formal_bundle_sha256"
            )
            if not isinstance(expected_bundle, str) or len(expected_bundle) != 64:
                raise ResolvedFormalExecutionContextError(
                    "resolved context lacks active formal bundle identity"
                )
            comparisons["active_formal_bundle_sha256"] = expected_bundle
        for field, expected in comparisons.items():
            if scientific.get(field) != expected:
                raise ResolvedFormalExecutionContextError(
                    f"resolved context protocol identity drift: {field}"
                )
    if clean_worktree_root is not None:
        expected_root = str(Path(clean_worktree_root).resolve())
        if runtime.get("clean_worktree_root") != expected_root:
            raise ResolvedFormalExecutionContextError(
                "resolved context clean worktree root drift"
            )
        if runtime.get("repository_root") != expected_root:
            raise ResolvedFormalExecutionContextError(
                "resolved context repository root drift"
            )
        if check_git:
            git_root, git_head = _git_identity(Path(expected_root))
            if git_root != expected_root:
                raise ResolvedFormalExecutionContextError(
                    "clean worktree root does not match actual Git root"
                )
            if scientific.get("execution_commit") != git_head:
                raise ResolvedFormalExecutionContextError(
                    "resolved context execution commit drift"
                )
    if durable_run_root is not None and runtime.get("durable_run_root") != str(
        Path(durable_run_root).resolve()
    ):
        raise ResolvedFormalExecutionContextError(
            "resolved context durable run root drift"
        )
    if environment_identity is not None:
        for field in ("environment_fingerprint", "dependency_fingerprint"):
            if scientific.get(field) != environment_identity.get(field):
                raise ResolvedFormalExecutionContextError(
                    f"resolved context environment identity drift: {field}"
                )
        if scientific.get("protocol_version") in {"2.1.0", "2.2.0", "2.3.0"} and scientific.get(
            "full_normalized_environment_projection"
        ) != dict(environment_identity):
            raise ResolvedFormalExecutionContextError(
                "resolved context full environment projection drift"
            )
    if runtime_audit is not None:
        if scientific.get("execution_commit") != runtime_audit.get(
            "observed_execution_commit"
        ):
            raise ResolvedFormalExecutionContextError(
                "resolved context observed execution commit drift"
            )
        if runtime.get("resolved_python_absolute_path") != runtime_audit.get(
            "resolved_python_absolute_path"
        ):
            raise ResolvedFormalExecutionContextError(
                "resolved context Python path drift"
            )
    return {
        "status": "pass",
        "context_sha256": expected_hash,
        "outer_expansion_sha256": command["outer_expansion_sha256"],
        "created_for_run_identity": payload["created_for_run_identity"],
    }


def atomic_create_resolved_formal_execution_context(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically publish a create-only context artifact."""

    validate_resolved_formal_execution_context(payload)
    target = Path(path)
    if target.exists():
        raise ResolvedFormalExecutionContextError(
            f"resolved execution context already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        f".{target.name}.staging-{os.getpid()}-{time.monotonic_ns()}"
    )
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ResolvedFormalExecutionContextError(
                f"resolved execution context already exists: {target}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target.resolve()),
        "context_sha256": payload["context_sha256"],
        "file_sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def load_resolved_formal_execution_context(
    path: str | Path,
    **validation_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise ResolvedFormalExecutionContextError(
            "resolved execution context artifact is required"
        )
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8-sig"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ResolvedFormalExecutionContextError(
                    f"non-finite JSON constant in resolved context: {value}"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise ResolvedFormalExecutionContextError(
            "resolved execution context is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ResolvedFormalExecutionContextError(
            "resolved execution context must be a JSON object"
        )
    report = validate_resolved_formal_execution_context(
        payload, **validation_kwargs
    )
    report.update(
        {
            "path": str(target.resolve()),
            "file_sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        }
    )
    return payload, report


def resolved_python_for_nested_consumer(
    payload: Mapping[str, Any], *, observed_sys_executable: str
) -> str:
    """Return the outer-selected Python after verifying the current child."""

    resolved = str(payload["runtime_location"]["resolved_python_absolute_path"])
    if str(Path(observed_sys_executable).absolute()) != resolved:
        raise ResolvedFormalExecutionContextError(
            "nested consumer Python differs from resolved outer context"
        )
    return resolved


__all__ = [
    "RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME",
    "RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION",
    "RESOLVED_FORMAL_EXECUTION_CONTEXT_V2_VERSION",
    "ResolvedFormalExecutionContextError",
    "atomic_create_resolved_formal_execution_context",
    "build_resolved_formal_execution_context",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_resolved_formal_execution_context",
    "resolved_python_for_nested_consumer",
    "sha256_file",
    "validate_resolved_formal_execution_context",
]
