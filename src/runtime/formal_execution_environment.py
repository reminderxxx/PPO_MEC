"""Portable Python execution environment contract for formal runs.

The scientific identity deliberately excludes host-specific paths. Runtime
resolution records those paths separately and proves that project imports come
from the requested clean worktree while third-party packages may come from a
shared virtual environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION = "1.2.0"
LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION = "1.0.0"
FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION = "1.1.0"
EXECUTION_ENVIRONMENT_RESOLVER_VERSION = "1.2.0"
CRITICAL_PACKAGES = ("torch", "numpy", "pandas", "PyYAML", "pytest")
DEFAULT_IMPORT_MODULES = (
    "src",
    "src.evaluators.typed_model_cache_formal_execution",
)
RUNTIME_OBSERVABLE_IDENTITY_FIELDS = (
    "formal_execution_environment_contract_version",
    "python_implementation",
    "python_version",
    "platform_system",
    "architecture",
    "dependency_fingerprint",
    "installed_package_count",
    "torch_version",
    "critical_package_versions",
    "execution_commit",
    "source_root_identity",
    "identity_rule",
)
PROTOCOL_BOUND_EXTENSION_FIELDS = (
    "formal_endpoint_metrics_contract_version",
    "formal_exogenous_request_execution_contract_version",
    "formal_request_exposure_trace_version",
    "formal_request_subject_lifecycle_contract_version",
)
ENVIRONMENT_FINGERPRINT_FIELD = "environment_fingerprint"
EXECUTION_COMMIT_IDENTITY_RULE = (
    "observed_clean_40_hex_HEAD_equal_main_equal_origin_main_bound_at_execution"
)
SOURCE_ROOT_IDENTITY_RULE = {
    "project_package": "src",
    "source_tree_identity_rule": (
        "observed_tracked_src_and_scripts_tree_sha256_bound_at_execution"
    ),
}
ENVIRONMENT_IDENTITY_RULE = (
    "full_projection_v1_excludes_host_paths_and_binds_observed_commit_and_tree_out_of_band"
)


class ExecutionEnvironmentError(ValueError):
    """Raised when a Python runtime cannot satisfy the frozen environment."""


def _canonical_sha256(value: Any) -> str:
    def reject(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ExecutionEnvironmentError("non-finite environment identity value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ExecutionEnvironmentError("environment identity key must be a string")
                reject(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject(child)

    reject(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ExecutionEnvironmentError(
            f"{label} has missing or unknown fields: missing={missing}, unknown={unknown}"
        )


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionEnvironmentError(f"environment identity field has wrong type: {field}")
    return value


def _validate_contract_version(value: Any, *, field: str, expected: str) -> str:
    version = _require_nonempty_string(value, field=field)
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ExecutionEnvironmentError(
            f"environment identity version is invalid: {field}"
        ) from exc
    if major != int(expected.split(".", 1)[0]):
        raise ExecutionEnvironmentError(
            f"unsupported environment identity contract major: {field}={version}"
        )
    if version != expected:
        raise ExecutionEnvironmentError(
            f"environment identity contract version mismatch: {field}"
        )
    return version


def normalize_protocol_bound_extensions(
    extensions: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the only Protocol fields allowed to extend environment identity."""

    if not isinstance(extensions, Mapping):
        raise ExecutionEnvironmentError("protocol-bound environment extensions must be an object")
    _require_exact_fields(
        extensions, set(PROTOCOL_BOUND_EXTENSION_FIELDS), label="protocol-bound extensions"
    )
    normalized = {
        field: _require_nonempty_string(extensions.get(field), field=field)
        for field in PROTOCOL_BOUND_EXTENSION_FIELDS
    }
    expected_versions = {
        "formal_endpoint_metrics_contract_version": "2.0.0",
        "formal_exogenous_request_execution_contract_version": "1.1.0",
        "formal_request_exposure_trace_version": "2.0.0",
        "formal_request_subject_lifecycle_contract_version": "1.0.0",
    }
    for field, expected in expected_versions.items():
        _validate_contract_version(normalized[field], field=field, expected=expected)
    return normalized


def protocol_bound_extensions_from_protocol(
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    """Project extension versions from a validated Protocol-shaped object."""

    request_contract = protocol.get("formal_exogenous_request_execution_contract")
    endpoint = protocol.get("endpoint_schema")
    if not isinstance(request_contract, Mapping) or not isinstance(endpoint, Mapping):
        raise ExecutionEnvironmentError(
            "Protocol lacks environment identity extension contracts"
        )
    return normalize_protocol_bound_extensions(
        {
            "formal_endpoint_metrics_contract_version": endpoint.get(
                "formal_endpoint_metrics_contract_version"
            ),
            "formal_exogenous_request_execution_contract_version": request_contract.get(
                "version"
            ),
            "formal_request_exposure_trace_version": request_contract.get(
                "request_exposure_trace_version"
            ),
            "formal_request_subject_lifecycle_contract_version": request_contract.get(
                "request_subject_lifecycle_contract_version"
            ),
        }
    )


def normalize_environment_identity(
    identity: Mapping[str, Any], *, require_fingerprint: bool = True
) -> dict[str, Any]:
    """Return the canonical full v1.1 projection and reject every schema ambiguity."""

    if not isinstance(identity, Mapping):
        raise ExecutionEnvironmentError("environment scientific identity must be an object")
    expected = set(RUNTIME_OBSERVABLE_IDENTITY_FIELDS) | set(
        PROTOCOL_BOUND_EXTENSION_FIELDS
    )
    if require_fingerprint:
        expected.add(ENVIRONMENT_FINGERPRINT_FIELD)
    _require_exact_fields(identity, expected, label="environment scientific identity")
    _validate_contract_version(
        identity.get("formal_execution_environment_contract_version"),
        field="formal_execution_environment_contract_version",
        expected=FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
    )
    normalized: dict[str, Any] = {}
    for field in RUNTIME_OBSERVABLE_IDENTITY_FIELDS:
        value = identity.get(field)
        if field == "installed_package_count":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionEnvironmentError(
                    "environment identity field has wrong type: installed_package_count"
                )
        elif field == "critical_package_versions":
            if not isinstance(value, Mapping):
                raise ExecutionEnvironmentError(
                    "environment identity field has wrong type: critical_package_versions"
                )
            _require_exact_fields(
                value, set(CRITICAL_PACKAGES), label="critical package versions"
            )
            for package, version in value.items():
                _require_nonempty_string(version, field=f"critical_package_versions.{package}")
            value = {str(key): value[key] for key in sorted(value)}
        elif field == "source_root_identity":
            if not isinstance(value, Mapping):
                raise ExecutionEnvironmentError(
                    "environment identity field has wrong type: source_root_identity"
                )
            _require_exact_fields(
                value, set(SOURCE_ROOT_IDENTITY_RULE), label="source root identity"
            )
            if dict(value) != SOURCE_ROOT_IDENTITY_RULE:
                raise ExecutionEnvironmentError("source root identity rule drift")
            value = dict(SOURCE_ROOT_IDENTITY_RULE)
        else:
            value = _require_nonempty_string(value, field=field)
        normalized[field] = value
    normalized.update(
        normalize_protocol_bound_extensions(
            {field: identity.get(field) for field in PROTOCOL_BOUND_EXTENSION_FIELDS}
        )
    )
    _canonical_sha256(normalized)
    if require_fingerprint:
        fingerprint = _require_nonempty_string(
            identity.get(ENVIRONMENT_FINGERPRINT_FIELD),
            field=ENVIRONMENT_FINGERPRINT_FIELD,
        )
        if len(fingerprint) != 64 or fingerprint != _canonical_sha256(normalized):
            raise ExecutionEnvironmentError("environment fingerprint mismatch")
        normalized[ENVIRONMENT_FINGERPRINT_FIELD] = fingerprint
    return normalized


def build_environment_identity_projection(
    runtime_observable: Mapping[str, Any],
    protocol_bound_extensions: Mapping[str, Any],
) -> dict[str, Any]:
    """The sole producer for the normalized Protocol 2.x scientific identity."""

    if not isinstance(runtime_observable, Mapping):
        raise ExecutionEnvironmentError("runtime-observable environment identity must be an object")
    _require_exact_fields(
        runtime_observable,
        set(RUNTIME_OBSERVABLE_IDENTITY_FIELDS),
        label="runtime-observable environment identity",
    )
    candidate = {**dict(runtime_observable), **normalize_protocol_bound_extensions(protocol_bound_extensions)}
    normalized = normalize_environment_identity(candidate, require_fingerprint=False)
    normalized[ENVIRONMENT_FINGERPRINT_FIELD] = _canonical_sha256(normalized)
    return normalize_environment_identity(normalized)


def source_tree_fingerprint(root: str | Path) -> str:
    """Hash tracked project Python files without including the host root path."""

    worktree = Path(root).resolve()
    completed = subprocess.run(
        ["git", "ls-files", "-z", "src", "scripts"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExecutionEnvironmentError("cannot enumerate clean worktree source files")
    digest = hashlib.sha256()
    for raw_relative in completed.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8")
        path = worktree / relative
        if not path.is_file() or path.suffix not in {".py", ".json", ".yaml", ".yml"}:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


_PROBE = r'''
import hashlib, importlib, importlib.metadata as md, json, os, platform, site, sys
distributions = list(md.distributions())
packages = sorted(
    (dist.metadata.get("Name", "").strip().lower(), dist.version)
    for dist in distributions if dist.metadata.get("Name")
)
editable_installs = []
for dist in distributions:
    direct = dist.read_text("direct_url.json")
    if not direct:
        continue
    try:
        direct_payload = json.loads(direct)
    except json.JSONDecodeError:
        continue
    if direct_payload.get("dir_info", {}).get("editable"):
        editable_installs.append({
            "name": dist.metadata.get("Name", ""),
            "url": direct_payload.get("url"),
        })
package_text = "\n".join(f"{name}=={version}" for name, version in packages)
requested = json.loads(os.environ["PPO_MEC_IMPORT_MODULES_JSON"])
imports = {}
for name in requested:
    module = importlib.import_module(name)
    imports[name] = {
        "file": str(getattr(module, "__file__", "") or ""),
        "version": str(getattr(module, "__version__", "") or ""),
    }
critical = {}
for distribution in json.loads(os.environ["PPO_MEC_CRITICAL_PACKAGES_JSON"]):
    try:
        critical[distribution] = md.version(distribution)
    except md.PackageNotFoundError:
        critical[distribution] = None
print(json.dumps({
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "platform_system": platform.system(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "site_packages": site.getsitepackages(),
    "cwd": os.getcwd(),
    "sys_path": sys.path,
    "pythonpath": os.environ.get("PYTHONPATH"),
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    "execution_commit": os.environ.get("PPO_MEC_EXECUTION_COMMIT"),
    "dependency_fingerprint": hashlib.sha256(package_text.encode()).hexdigest(),
    "installed_package_count": len(packages),
    "critical_packages": critical,
    "editable_installs": editable_installs,
    "imports": imports,
}, sort_keys=True))
'''


def child_environment(clean_worktree_root: str | Path) -> dict[str, str]:
    root = str(Path(clean_worktree_root).resolve())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = root
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PPO_MEC_CLEAN_WORKTREE_ROOT"] = root
    return environment


def probe_python_environment(
    executable: str | Path,
    *,
    clean_worktree_root: str | Path,
    import_modules: Sequence[str] = DEFAULT_IMPORT_MODULES,
) -> dict[str, Any]:
    python = Path(executable).absolute()
    if not python.exists():
        raise ExecutionEnvironmentError(f"Python interpreter does not exist: {python}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ExecutionEnvironmentError(f"Python interpreter is not executable: {python}")
    root = Path(clean_worktree_root).resolve()
    if not (root / "src/__init__.py").is_file():
        raise ExecutionEnvironmentError("clean worktree root lacks src/__init__.py")
    environment = child_environment(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode == 0:
        environment["PPO_MEC_EXECUTION_COMMIT"] = commit.stdout.strip()
    environment["PPO_MEC_IMPORT_MODULES_JSON"] = json.dumps(list(import_modules))
    environment["PPO_MEC_CRITICAL_PACKAGES_JSON"] = json.dumps(list(CRITICAL_PACKAGES))
    completed = subprocess.run(
        [str(python), "-c", _PROBE],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ExecutionEnvironmentError(f"Python environment probe failed: {message}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionEnvironmentError("Python environment probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ExecutionEnvironmentError("Python environment probe must return an object")
    return payload


def scientific_environment_identity(
    probe: Mapping[str, Any],
    *,
    execution_commit: str,
    source_tree_sha256: str,
    protocol_bound_extensions: Mapping[str, Any] | None = None,
    contract_version: str = LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
) -> dict[str, Any]:
    critical = dict(probe.get("critical_packages", {}))
    identity = {
        "formal_execution_environment_contract_version": contract_version,
        "python_implementation": probe.get("implementation"),
        "python_version": probe.get("python_version"),
        "platform_system": probe.get("platform_system"),
        "architecture": probe.get("architecture"),
        "dependency_fingerprint": probe.get("dependency_fingerprint"),
        "installed_package_count": probe.get("installed_package_count"),
        "torch_version": critical.get("torch"),
        "critical_package_versions": critical,
        "execution_commit": execution_commit,
        "source_root_identity": {
            "project_package": "src",
            "source_tree_sha256": source_tree_sha256,
        },
        "identity_rule": "environment identity != host-specific Python absolute path",
    }
    if contract_version == FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION:
        if protocol_bound_extensions is None:
            raise ExecutionEnvironmentError(
                "formal environment 1.2 requires Protocol-bound extensions"
            )
        identity["execution_commit"] = EXECUTION_COMMIT_IDENTITY_RULE
        identity["source_root_identity"] = dict(SOURCE_ROOT_IDENTITY_RULE)
        identity["identity_rule"] = ENVIRONMENT_IDENTITY_RULE
        return build_environment_identity_projection(
            identity, protocol_bound_extensions
        )
    if contract_version != LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION:
        raise ExecutionEnvironmentError(
            "unsupported formal execution environment contract version"
        )
    identity["environment_fingerprint"] = _canonical_sha256(identity)
    return identity


def validate_import_origins(
    probe: Mapping[str, Any],
    *,
    clean_worktree_root: str | Path,
    forbidden_source_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(clean_worktree_root).resolve()
    imports = probe.get("imports")
    if not isinstance(imports, Mapping) or not imports:
        raise ExecutionEnvironmentError("environment probe lacks project import origins")
    validated: dict[str, str] = {}
    for module, record in imports.items():
        origin = Path(str(record.get("file", ""))).resolve()
        try:
            origin.relative_to(root)
        except ValueError as exc:
            raise ExecutionEnvironmentError(
                f"project import does not originate from clean worktree: {module}={origin}"
            ) from exc
        for forbidden in forbidden_source_roots:
            forbidden_root = Path(forbidden).resolve()
            if forbidden_root == root:
                continue
            try:
                origin.relative_to(forbidden_root)
            except ValueError:
                continue
            raise ExecutionEnvironmentError(
                f"project import originates from forbidden dirty worktree: {module}={origin}"
            )
        validated[str(module)] = str(origin)
    return {"status": "pass", "import_origins": validated}


def assert_clean_git_worktree(root: str | Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(root).resolve(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExecutionEnvironmentError("clean source root is not a Git worktree")
    if completed.stdout.strip():
        raise ExecutionEnvironmentError("formal execution source root is not clean")


@dataclass(frozen=True)
class ResolvedExecutionEnvironment:
    python_executable: str
    environment_identity: dict[str, Any]
    runtime_audit: dict[str, Any]
    child_environment: dict[str, str]


def resolve_execution_environment(
    *,
    clean_worktree_root: str | Path,
    execution_commit: str,
    python_executable: str | Path | None = None,
    environment_manifest: Mapping[str, Any] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    protocol_bound_extensions: Mapping[str, Any] | None = None,
    forbidden_source_roots: Sequence[str | Path] = (),
    import_modules: Sequence[str] = DEFAULT_IMPORT_MODULES,
    require_clean_git_worktree: bool = False,
) -> ResolvedExecutionEnvironment:
    """Resolve and validate one interpreter for every child command.

    Priority is explicit ``python_executable``, manifest runtime location,
    current runner ``sys.executable``, then protocol-approved candidates.
    """

    manifest = dict(environment_manifest or {})
    runtime = manifest.get("runtime_location", {})
    candidates: list[tuple[str, str]] = []
    if python_executable is not None:
        candidates.append(("explicit_python_executable", str(python_executable)))
    elif isinstance(runtime, Mapping) and runtime.get("resolved_python_absolute_path"):
        candidates.append(
            (
                "execution_environment_manifest",
                str(runtime["resolved_python_absolute_path"]),
            )
        )
    else:
        candidates.append(("current_runner_sys_executable", sys.executable))
        for item in manifest.get("allowed_python_candidates", []):
            candidates.append(("protocol_allowed_candidate", str(item)))
    if not candidates:
        raise ExecutionEnvironmentError("no Python interpreter candidate is available")
    source, selected = candidates[0]
    python = Path(selected).absolute()
    probe = probe_python_environment(
        python,
        clean_worktree_root=clean_worktree_root,
        import_modules=import_modules,
    )
    if require_clean_git_worktree:
        assert_clean_git_worktree(clean_worktree_root)
    clean_root = Path(clean_worktree_root).resolve()
    effective_forbidden = [
        item
        for item in forbidden_source_roots
        if Path(item).resolve() != clean_root
    ]
    venv_root = Path(str(probe.get("sys_prefix", ""))).resolve()
    inferred_project_root = venv_root.parent
    if (
        venv_root.name == ".venv"
        and inferred_project_root != Path(clean_worktree_root).resolve()
        and (inferred_project_root / "src/__init__.py").is_file()
    ):
        effective_forbidden.append(inferred_project_root)
    origin_report = validate_import_origins(
        probe,
        clean_worktree_root=clean_worktree_root,
        forbidden_source_roots=effective_forbidden,
    )
    tree_hash = source_tree_fingerprint(clean_worktree_root)
    expected = dict(expected_identity or manifest.get("scientific_identity", {}) or {})
    contract_version = str(
        expected.get(
            "formal_execution_environment_contract_version",
            LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
        )
    )
    extensions = protocol_bound_extensions
    if contract_version == FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION:
        expected = normalize_environment_identity(expected)
        expected_extensions = {
            field: expected[field] for field in PROTOCOL_BOUND_EXTENSION_FIELDS
        }
        if extensions is None:
            extensions = expected_extensions
        elif normalize_protocol_bound_extensions(extensions) != expected_extensions:
            raise ExecutionEnvironmentError(
                "Protocol/environment extension version mismatch"
            )
    identity = scientific_environment_identity(
        probe,
        execution_commit=execution_commit,
        source_tree_sha256=tree_hash,
        protocol_bound_extensions=extensions,
        contract_version=contract_version,
    )
    if expected:
        # Commit/tree bindings may be logical out-of-band identities because a
        # commit cannot contain its own hash. Runtime locations remain audited,
        # while the immutable protocol provides the scientific binding.
        if contract_version == LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION and expected.get("execution_commit"):
            identity["execution_commit"] = expected["execution_commit"]
        if contract_version == LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION and expected.get("source_root_identity"):
            identity["source_root_identity"] = expected["source_root_identity"]
        if contract_version == LEGACY_FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION:
            identity.pop("environment_fingerprint", None)
            identity["environment_fingerprint"] = _canonical_sha256(identity)
            comparison_fields = (
                "python_implementation",
                "python_version",
                "platform_system",
                "architecture",
                "dependency_fingerprint",
                "torch_version",
                "critical_package_versions",
                "execution_commit",
                "source_root_identity",
                "environment_fingerprint",
            )
        else:
            comparison_fields = (*RUNTIME_OBSERVABLE_IDENTITY_FIELDS, *PROTOCOL_BOUND_EXTENSION_FIELDS, ENVIRONMENT_FINGERPRINT_FIELD)
        for key in comparison_fields:
            if expected.get(key) != identity.get(key):
                raise ExecutionEnvironmentError(
                    f"execution environment identity mismatch: {key}"
                )
    root = Path(clean_worktree_root).resolve()
    commit_probe = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    observed_execution_commit = (
        commit_probe.stdout.strip() if commit_probe.returncode == 0 else None
    )
    if require_clean_git_worktree and not observed_execution_commit:
        raise ExecutionEnvironmentError("clean source root lacks an execution commit")
    runtime_audit = {
        "execution_environment_resolver_version": EXECUTION_ENVIRONMENT_RESOLVER_VERSION,
        "formal_environment_identity_projection_contract_version": (
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        "full_normalized_environment_projection": identity,
        "resolution_source": source,
        "resolved_python_absolute_path": str(python),
        "virtual_environment_root": probe.get("sys_prefix"),
        "site_packages_paths": probe.get("site_packages", []),
        "clean_worktree_root": str(root),
        "cwd": probe.get("cwd"),
        "sys_path": probe.get("sys_path", []),
        "import_origin": origin_report["import_origins"],
        "environment_variables": {
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "PPO_MEC_CLEAN_WORKTREE_ROOT": str(root),
            "PPO_MEC_EXECUTION_COMMIT": observed_execution_commit,
        },
        "editable_install_detected": bool(probe.get("editable_installs")),
        "editable_installs": probe.get("editable_installs", []),
        "observed_execution_commit": observed_execution_commit,
        "host_paths_excluded_from_scientific_identity": True,
        "observed_source_tree_sha256": tree_hash,
        "forbidden_project_source_roots": [
            str(Path(item).resolve()) for item in effective_forbidden
        ],
    }
    resolved_child_environment = child_environment(root)
    if observed_execution_commit:
        resolved_child_environment["PPO_MEC_EXECUTION_COMMIT"] = observed_execution_commit
    return ResolvedExecutionEnvironment(
        python_executable=str(python),
        environment_identity=identity,
        runtime_audit=runtime_audit,
        child_environment=resolved_child_environment,
    )


def assert_child_environment_parity(
    resolution: ResolvedExecutionEnvironment,
    *,
    clean_worktree_root: str | Path,
    execution_commit: str,
    forbidden_source_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    repeated = resolve_execution_environment(
        clean_worktree_root=clean_worktree_root,
        execution_commit=execution_commit,
        python_executable=resolution.python_executable,
        expected_identity=resolution.environment_identity,
        protocol_bound_extensions={
            field: resolution.environment_identity[field]
            for field in PROTOCOL_BOUND_EXTENSION_FIELDS
        }
        if resolution.environment_identity.get(
            "formal_execution_environment_contract_version"
        )
        == FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
        else None,
        forbidden_source_roots=forbidden_source_roots,
    )
    return {
        "status": "pass",
        "same_python": repeated.python_executable == resolution.python_executable,
        "same_environment_fingerprint": (
            repeated.environment_identity["environment_fingerprint"]
            == resolution.environment_identity["environment_fingerprint"]
        ),
    }


__all__ = [
    "CRITICAL_PACKAGES",
    "DEFAULT_IMPORT_MODULES",
    "EXECUTION_ENVIRONMENT_RESOLVER_VERSION",
    "ExecutionEnvironmentError",
    "FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION",
    "FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION",
    "PROTOCOL_BOUND_EXTENSION_FIELDS",
    "RUNTIME_OBSERVABLE_IDENTITY_FIELDS",
    "ResolvedExecutionEnvironment",
    "assert_child_environment_parity",
    "assert_clean_git_worktree",
    "build_environment_identity_projection",
    "child_environment",
    "normalize_environment_identity",
    "normalize_protocol_bound_extensions",
    "probe_python_environment",
    "protocol_bound_extensions_from_protocol",
    "resolve_execution_environment",
    "scientific_environment_identity",
    "source_tree_fingerprint",
    "validate_import_origins",
]
