"""Two-layer scientific-config and execution-binding identities for formal training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime.formal_protocol_capabilities import get_protocol_capabilities


AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION = "2.0.0"
FORMAL_TRAINING_EXECUTION_BINDING_VERSION = "1.0.0"
SCIENTIFIC_FIELDS = (
    "learning_rate",
    "entropy_coef",
    "value_coef",
    "auxiliary_coef",
)

BASE_CHECKPOINT_TRAINING_IDENTITY_FIELDS = (
    "agent_scientific_config_semantic_sha256",
    "formal_training_execution_binding_sha256",
    "formal_protocol_semantic_sha256",
    "execution_commit",
    "resolved_execution_context_sha256",
)


class FormalTrainingIdentityError(ValueError):
    """Raised when either layer of the formal training identity drifts."""


def checkpoint_training_identity_fields(protocol_version: str) -> tuple[str, ...]:
    """Return the exact checkpoint identity fields required by one Protocol capability set."""

    capabilities = get_protocol_capabilities(protocol_version)
    fields = BASE_CHECKPOINT_TRAINING_IDENTITY_FIELDS
    if capabilities.agent_order_contract_required:
        fields = (*fields, "formal_agent_order_contract_semantic_sha256")
    if capabilities.active_bundle_required:
        fields = (*fields, "active_formal_bundle_sha256")
    if capabilities.nullable_metric_contract_required:
        fields = (
            *fields,
            "formal_nullable_metric_aggregation_contract_semantic_sha256",
        )
    return fields


def checkpoint_training_identity_projection(
    metadata: Mapping[str, Any],
    *,
    protocol_version: str,
    require_nested_contract: bool,
) -> dict[str, str]:
    """Project the identity actually serialized in checkpoint metadata.

    Active consumers require both top-level fields and the nested resolved training
    contract. Missing legacy top-level fields are intentionally not backfilled.
    """

    projected: dict[str, str] = {}
    for field in checkpoint_training_identity_fields(protocol_version):
        if field not in metadata:
            raise FormalTrainingIdentityError(
                f"checkpoint provenance identity missing: {field}"
            )
        value = metadata[field]
        if not isinstance(value, str) or not value:
            raise FormalTrainingIdentityError(
                f"checkpoint provenance identity invalid: {field}"
            )
        projected[field] = value
    if require_nested_contract:
        nested = metadata.get("formal_training_contract")
        if not isinstance(nested, Mapping):
            raise FormalTrainingIdentityError(
                "checkpoint formal_training_contract is missing"
            )
        if nested.get("formal_protocol_version") != protocol_version:
            raise FormalTrainingIdentityError(
                "checkpoint formal_training_contract Protocol version mismatch"
            )
        for field, value in projected.items():
            if field not in nested:
                raise FormalTrainingIdentityError(
                    f"checkpoint nested training identity missing: {field}"
                )
            if nested[field] != value:
                raise FormalTrainingIdentityError(
                    f"checkpoint top-level/nested training identity mismatch: {field}"
                )
    return projected


def build_checkpoint_training_identity(
    resolved_training_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Build producer metadata only from the already resolved training contract."""

    protocol_version = resolved_training_contract.get("formal_protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version:
        return {}
    envelope = {
        "formal_training_contract": deepcopy(dict(resolved_training_contract)),
        **{
            field: resolved_training_contract.get(field)
            for field in checkpoint_training_identity_fields(protocol_version)
        },
    }
    return checkpoint_training_identity_projection(
        envelope,
        protocol_version=protocol_version,
        require_nested_contract=True,
    )


def expected_checkpoint_training_identity(
    *,
    protocol: Mapping[str, Any],
    resolved_execution_context: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the trusted active-run identity from validated Protocol/context inputs."""

    protocol_version = str(protocol.get("typed_model_cache_formal_protocol_version") or "")
    scientific = resolved_execution_context.get("scientific_identity")
    if not isinstance(scientific, Mapping):
        raise FormalTrainingIdentityError(
            "resolved execution context scientific identity is missing"
        )
    expected: dict[str, Any] = {
        "agent_scientific_config_semantic_sha256": scientific.get(
            "agent_scientific_config_semantic_sha256"
        ),
        "formal_training_execution_binding_sha256": scientific.get(
            "formal_training_execution_binding_sha256"
        ),
        "formal_protocol_semantic_sha256": protocol.get("hashes", {}).get(
            "semantic_sha256"
        ),
        "execution_commit": scientific.get("execution_commit"),
        "resolved_execution_context_sha256": resolved_execution_context.get(
            "context_sha256"
        ),
        "formal_agent_order_contract_semantic_sha256": protocol.get(
            "formal_agent_order_contract", {}
        ).get("semantic_sha256"),
        "active_formal_bundle_sha256": scientific.get(
            "active_formal_bundle_sha256"
        ),
        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol.get(
            "formal_nullable_metric_aggregation_contract", {}
        ).get("semantic_sha256"),
    }
    fields = checkpoint_training_identity_fields(protocol_version)
    result: dict[str, str] = {}
    for field in fields:
        value = expected.get(field)
        if not isinstance(value, str) or not value:
            raise FormalTrainingIdentityError(
                f"trusted expected checkpoint identity is incomplete: {field}"
            )
        result[field] = value
    return result


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalTrainingIdentityError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FormalTrainingIdentityError(f"non-string key at {path}")
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


def load_strict_json_mapping(path: str | Path, field_name: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalTrainingIdentityError(
                    f"duplicate JSON key in {field_name}: {key}"
                )
            result[key] = value
        return result

    target = Path(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8-sig"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FormalTrainingIdentityError(
                    f"non-finite JSON constant in {field_name}: {value}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalTrainingIdentityError(
            f"unable to load {field_name}: {target}"
        ) from exc
    if not isinstance(payload, dict):
        raise FormalTrainingIdentityError(f"{field_name} must be a JSON object")
    return payload


def scientific_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in config.items()
        if key != "config_semantic_sha256"
    }


def learned_agent_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = protocol.get("agent_matrix", {}).get("controller_table", [])
    learned = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("training_requirement") == "clean_typed_checkpoint_per_seed_and_capacity"
    ]
    if len(learned) != 10:
        raise FormalTrainingIdentityError("formal protocol learned-agent matrix must contain 10 agents")
    names = [str(row.get("agent")) for row in learned]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise FormalTrainingIdentityError("formal protocol learned-agent matrix is duplicated or invalid")
    return learned


def agent_matrix_identity(protocol: Mapping[str, Any]) -> str:
    return canonical_sha256(learned_agent_rows(protocol))


def training_budget_identity(protocol: Mapping[str, Any]) -> str:
    budget = protocol.get("training_budget")
    if not isinstance(budget, Mapping):
        raise FormalTrainingIdentityError("formal protocol lacks training_budget")
    return canonical_sha256(budget)


def validate_scientific_config(
    config: Mapping[str, Any], *, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    _reject_non_finite(config)
    expected_top = {
        "agent_training_scientific_config_contract_version",
        "learned_agent_order",
        "scientific_fields",
        "not_applicable_semantics",
        "canonical_serialization",
        "agents",
        "config_semantic_sha256",
    }
    if set(config) != expected_top:
        raise FormalTrainingIdentityError("scientific config has missing or unknown top-level fields")
    if config.get("agent_training_scientific_config_contract_version") != (
        AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION
    ):
        raise FormalTrainingIdentityError("unsupported scientific config contract")
    if tuple(config.get("scientific_fields", ())) != SCIENTIFIC_FIELDS:
        raise FormalTrainingIdentityError("scientific config field schema drift")
    if config.get("not_applicable_semantics") != (
        "field absent from hyperparameters and marked not_applicable; null/default inference forbidden"
    ):
        raise FormalTrainingIdentityError("scientific config not-applicable semantics drift")
    if config.get("canonical_serialization") != (
        "UTF-8 sorted-key compact JSON; NaN/Infinity and duplicate/unknown fields rejected"
    ):
        raise FormalTrainingIdentityError("scientific config canonical serialization drift")
    agents = config.get("agents")
    order = config.get("learned_agent_order")
    if not isinstance(agents, Mapping) or not isinstance(order, list):
        raise FormalTrainingIdentityError("scientific config agent matrix is invalid")
    if len(order) != 10 or len(set(order)) != 10 or set(order) != set(agents):
        raise FormalTrainingIdentityError("scientific config has missing, duplicate, or unknown agent")
    for name in order:
        row = agents.get(name)
        if not isinstance(row, Mapping) or set(row) != {
            "agent_identity",
            "hyperparameters",
            "field_applicability",
        }:
            raise FormalTrainingIdentityError(f"scientific config entry schema drift: {name}")
        if row.get("agent_identity") != name:
            raise FormalTrainingIdentityError(f"scientific config agent identity drift: {name}")
        hyper = row.get("hyperparameters")
        applicability = row.get("field_applicability")
        if not isinstance(hyper, Mapping) or not isinstance(applicability, Mapping):
            raise FormalTrainingIdentityError(f"scientific config entry is invalid: {name}")
        if set(applicability) != set(SCIENTIFIC_FIELDS):
            raise FormalTrainingIdentityError(f"scientific config applicability schema drift: {name}")
        expected_applicable = {field for field, state in applicability.items() if state == "applicable"}
        if any(state not in {"applicable", "not_applicable"} for state in applicability.values()):
            raise FormalTrainingIdentityError(f"scientific config applicability value drift: {name}")
        if set(hyper) != expected_applicable or not set(hyper).issubset(SCIENTIFIC_FIELDS):
            raise FormalTrainingIdentityError(f"scientific config not-applicable encoding drift: {name}")
        if "learning_rate" not in hyper:
            raise FormalTrainingIdentityError(f"scientific config lacks learning_rate: {name}")
        for field, value in hyper.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FormalTrainingIdentityError(f"scientific config value is not numeric: {name}.{field}")
            if not math.isfinite(float(value)):
                raise FormalTrainingIdentityError(f"scientific config value is non-finite: {name}.{field}")
        if (name == "sa_ghmappo") != ("auxiliary_coef" in hyper):
            raise FormalTrainingIdentityError("auxiliary_coef applicability drift")
        if name == "sa_ghmappo" and float(hyper["auxiliary_coef"]) != 0.06:
            raise FormalTrainingIdentityError("SA-GHMAPPO auxiliary_coef drift")
    expected_hash = canonical_sha256(scientific_config_projection(config))
    if config.get("config_semantic_sha256") != expected_hash:
        raise FormalTrainingIdentityError("scientific config semantic SHA-256 mismatch")
    if protocol is not None:
        capabilities = get_protocol_capabilities(
            protocol.get("typed_model_cache_formal_protocol_version")
        )
        protocol_names = [str(row["agent"]) for row in learned_agent_rows(protocol)]
        if list(order) != protocol_names:
            raise FormalTrainingIdentityError("scientific config/protocol agent order mismatch")
        protocol_configs = protocol.get("training_budget", {}).get("agent_configs")
        if not isinstance(protocol_configs, Mapping) or set(protocol_configs) != set(order):
            raise FormalTrainingIdentityError("protocol agent config matrix drift")
        for name in order:
            if dict(agents[name]["hyperparameters"]) != dict(protocol_configs[name]):
                raise FormalTrainingIdentityError(
                    f"scientific config/protocol hyperparameter mismatch: {name}"
                )
        contract = protocol.get("agent_training_scientific_config_contract", {})
        if contract.get("version") != AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION:
            raise FormalTrainingIdentityError("protocol scientific config contract version mismatch")
        if contract.get("config_semantic_sha256") != expected_hash:
            raise FormalTrainingIdentityError("protocol scientific config hash mismatch")
        if capabilities.agent_order_contract_required:
            from src.runtime.formal_agent_order import (
                FormalAgentOrderError,
                resolve_formal_agent_order,
            )

            try:
                resolve_formal_agent_order(
                    protocol=protocol,
                    scientific_config=config,
                )
            except FormalAgentOrderError as exc:
                raise FormalTrainingIdentityError(str(exc)) from exc
    return {
        "status": "pass",
        "config_semantic_sha256": expected_hash,
        "agent_count": len(order),
        "agent_order": list(order),
    }


def resolved_agent_hyperparameters(config: Mapping[str, Any], agent_name: str) -> dict[str, Any]:
    validate_scientific_config(config)
    agents = config["agents"]
    if agent_name not in agents:
        raise FormalTrainingIdentityError(f"scientific config lacks {agent_name}")
    return dict(agents[agent_name]["hyperparameters"])


def binding_projection(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in binding.items()
        if key != "binding_full_sha256"
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_create_execution_binding(
    path: str | Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FormalTrainingIdentityError(
            f"formal training execution binding already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.staging-{os.getpid()}-{time.monotonic_ns()}"
    encoded = json.dumps(
        binding, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FormalTrainingIdentityError(
                f"formal training execution binding already exists: {target}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target.resolve()),
        "binding_full_sha256": binding["binding_full_sha256"],
        "file_sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def build_execution_binding(
    *,
    protocol: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    execution_commit: str,
    environment_identity: Mapping[str, Any],
    command_matrix_sha256: str,
    active_formal_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    capabilities = get_protocol_capabilities(
        protocol.get("typed_model_cache_formal_protocol_version")
    )
    scientific = validate_scientific_config(scientific_config, protocol=protocol)
    if not isinstance(execution_commit, str) or len(execution_commit) != 40:
        raise FormalTrainingIdentityError("execution binding requires exact 40-hex commit")
    try:
        int(execution_commit, 16)
    except ValueError as exc:
        raise FormalTrainingIdentityError("execution binding commit is not hexadecimal") from exc
    data_and_runtime_identity = {
        "split_semantic_sha256": protocol["identity"]["split_semantic_sha256"],
        "window_contract_semantic_sha256": protocol["execution_contract"]
        ["window_consumption_contract"]["semantic_sha256"],
        "catalog_fingerprint": protocol["identity"]["catalog_fingerprint"],
        "typed_runtime_identities": protocol["identity"]
        ["typed_runtime_contract_hashes_by_capacity"],
    }
    if capabilities.agent_order_contract_required:
        data_and_runtime_identity["formal_agent_order_contract_semantic_sha256"] = (
            protocol["formal_agent_order_contract"]["semantic_sha256"]
        )
    if capabilities.exogenous_request_execution_required:
        data_and_runtime_identity["formal_exogenous_request_execution"] = deepcopy(
            protocol["formal_exogenous_request_execution_contract"]
        )
    if capabilities.nullable_metric_contract_required:
        data_and_runtime_identity[
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]
    payload: dict[str, Any] = {
        "formal_training_execution_binding_version": FORMAL_TRAINING_EXECUTION_BINDING_VERSION,
        "protocol_identity": {
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["typed_model_cache_formal_protocol_version"],
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        },
        "execution_commit": execution_commit,
        "agent_scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
        "agent_matrix_identity": agent_matrix_identity(protocol),
        "training_budget_identity": training_budget_identity(protocol),
        "resolved_execution_context_contract_version": protocol[
            "resolved_formal_execution_context_contract"
        ]["version"],
        "environment_identity": {
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        },
        "data_and_runtime_identity": data_and_runtime_identity,
        "command_matrix_sha256": command_matrix_sha256,
        "portable_resource_identity": {
            "resource_registry_semantic_sha256": protocol[
                "portable_resource_identity_contract"
            ]["resource_registry_semantic_sha256"],
            "content_identical_path_relocation_allowed": True,
            "host_path_is_scientific_identity": False,
        },
        "canonical_serialization": "UTF-8 sorted-key compact JSON; NaN/Infinity rejected",
    }
    if capabilities.full_environment_projection_required:
        payload["environment_identity"] = {
            "projection_contract_version": protocol[
                "formal_execution_environment_contract"
            ]["identity_projection_contract_version"],
            "full_normalized_projection": deepcopy(dict(environment_identity)),
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        }
    if capabilities.active_bundle_required:
        if not isinstance(active_formal_bundle_sha256, str) or len(
            active_formal_bundle_sha256
        ) != 64:
            raise FormalTrainingIdentityError(
                "active execution binding requires active formal bundle SHA-256"
            )
        payload["active_formal_bundle_sha256"] = active_formal_bundle_sha256
    payload["binding_full_sha256"] = canonical_sha256(binding_projection(payload))
    validate_execution_binding(
        payload,
        protocol=protocol,
        scientific_config=scientific_config,
        execution_commit=execution_commit,
        environment_identity=environment_identity,
        command_matrix_sha256=command_matrix_sha256,
        active_formal_bundle_sha256=active_formal_bundle_sha256,
    )
    return payload


def validate_execution_binding(
    binding: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    execution_commit: str,
    environment_identity: Mapping[str, Any],
    command_matrix_sha256: str,
    active_formal_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    capabilities = get_protocol_capabilities(
        protocol.get("typed_model_cache_formal_protocol_version")
    )
    _reject_non_finite(binding)
    if binding.get("formal_training_execution_binding_version") != (
        FORMAL_TRAINING_EXECUTION_BINDING_VERSION
    ):
        raise FormalTrainingIdentityError("execution binding version mismatch")
    observed_hash = canonical_sha256(binding_projection(binding))
    if binding.get("binding_full_sha256") != observed_hash:
        raise FormalTrainingIdentityError("execution binding full SHA-256 mismatch")
    scientific = validate_scientific_config(scientific_config, protocol=protocol)
    data_and_runtime_identity = {
        "split_semantic_sha256": protocol["identity"]["split_semantic_sha256"],
        "window_contract_semantic_sha256": protocol["execution_contract"]
        ["window_consumption_contract"]["semantic_sha256"],
        "catalog_fingerprint": protocol["identity"]["catalog_fingerprint"],
        "typed_runtime_identities": protocol["identity"]
        ["typed_runtime_contract_hashes_by_capacity"],
    }
    if capabilities.agent_order_contract_required:
        data_and_runtime_identity["formal_agent_order_contract_semantic_sha256"] = (
            protocol["formal_agent_order_contract"]["semantic_sha256"]
        )
    if capabilities.exogenous_request_execution_required:
        data_and_runtime_identity["formal_exogenous_request_execution"] = deepcopy(
            protocol["formal_exogenous_request_execution_contract"]
        )
    if capabilities.nullable_metric_contract_required:
        data_and_runtime_identity[
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]
    comparisons = {
        "protocol_identity": {
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["typed_model_cache_formal_protocol_version"],
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        },
        "execution_commit": execution_commit,
        "agent_scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
        "agent_matrix_identity": agent_matrix_identity(protocol),
        "training_budget_identity": training_budget_identity(protocol),
        "resolved_execution_context_contract_version": protocol[
            "resolved_formal_execution_context_contract"
        ]["version"],
        "environment_identity": {
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        },
        "data_and_runtime_identity": data_and_runtime_identity,
        "command_matrix_sha256": command_matrix_sha256,
        "portable_resource_identity": {
            "resource_registry_semantic_sha256": protocol[
                "portable_resource_identity_contract"
            ]["resource_registry_semantic_sha256"],
            "content_identical_path_relocation_allowed": True,
            "host_path_is_scientific_identity": False,
        },
        "canonical_serialization": "UTF-8 sorted-key compact JSON; NaN/Infinity rejected",
    }
    if capabilities.full_environment_projection_required:
        comparisons["environment_identity"] = {
            "projection_contract_version": protocol[
                "formal_execution_environment_contract"
            ]["identity_projection_contract_version"],
            "full_normalized_projection": deepcopy(dict(environment_identity)),
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        }
    if capabilities.active_bundle_required:
        if not isinstance(active_formal_bundle_sha256, str) or len(
            active_formal_bundle_sha256
        ) != 64:
            raise FormalTrainingIdentityError(
                "active execution binding requires active formal bundle SHA-256"
            )
        comparisons["active_formal_bundle_sha256"] = active_formal_bundle_sha256
    allowed = {
        "formal_training_execution_binding_version",
        *comparisons.keys(),
        "binding_full_sha256",
    }
    if set(binding) != allowed:
        raise FormalTrainingIdentityError("execution binding has missing or unknown fields")
    for field, value in comparisons.items():
        if binding.get(field) != value:
            raise FormalTrainingIdentityError(f"execution binding drift: {field}")
    return {"status": "pass", "binding_full_sha256": observed_hash}


def validate_checkpoint_training_identity(
    metadata: Mapping[str, Any], *, scientific_config_sha256: str, binding_sha256: str,
    protocol_semantic_sha256: str, execution_commit: str, resolved_context_sha256: str,
    formal_agent_order_contract_semantic_sha256: str | None = None,
    active_formal_bundle_sha256: str | None = None,
    formal_nullable_metric_aggregation_contract_semantic_sha256: str | None = None,
    protocol_version: str | None = None,
    require_nested_contract: bool = False,
    expected_agent_name: str | None = None,
    expected_seed: int | None = None,
    expected_runtime_contract_sha256: str | None = None,
) -> dict[str, Any]:
    expected = {
        "agent_scientific_config_semantic_sha256": scientific_config_sha256,
        "formal_training_execution_binding_sha256": binding_sha256,
        "formal_protocol_semantic_sha256": protocol_semantic_sha256,
        "execution_commit": execution_commit,
        "resolved_execution_context_sha256": resolved_context_sha256,
    }
    if formal_agent_order_contract_semantic_sha256 is not None:
        expected["formal_agent_order_contract_semantic_sha256"] = (
            formal_agent_order_contract_semantic_sha256
        )
    if active_formal_bundle_sha256 is not None:
        expected["active_formal_bundle_sha256"] = active_formal_bundle_sha256
    if formal_nullable_metric_aggregation_contract_semantic_sha256 is not None:
        expected[
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = formal_nullable_metric_aggregation_contract_semantic_sha256
    if protocol_version is not None:
        observed = checkpoint_training_identity_projection(
            metadata,
            protocol_version=protocol_version,
            require_nested_contract=require_nested_contract,
        )
        if set(observed) != set(expected):
            raise FormalTrainingIdentityError(
                "checkpoint expected identity fields do not match Protocol capabilities"
            )
    for field, value in expected.items():
        if field not in metadata or metadata[field] != value:
            raise FormalTrainingIdentityError(f"checkpoint provenance identity mismatch: {field}")
    provenance = metadata.get("typed_runtime_provenance")
    if any(
        value is not None
        for value in (expected_agent_name, expected_seed, expected_runtime_contract_sha256)
    ):
        if not isinstance(provenance, Mapping):
            raise FormalTrainingIdentityError("checkpoint typed runtime provenance is missing")
        per_cell_expected = {
            "agent_identity": expected_agent_name,
            "training_seed": int(expected_seed) if expected_seed is not None else None,
            "runtime_contract_sha256": expected_runtime_contract_sha256,
        }
        for field, value in per_cell_expected.items():
            if value is not None and provenance.get(field) != value:
                raise FormalTrainingIdentityError(
                    f"checkpoint per-cell identity mismatch: {field}"
                )
    return {"status": "pass", **expected}


__all__ = [
    "AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION",
    "BASE_CHECKPOINT_TRAINING_IDENTITY_FIELDS",
    "FORMAL_TRAINING_EXECUTION_BINDING_VERSION",
    "FormalTrainingIdentityError",
    "agent_matrix_identity",
    "atomic_create_execution_binding",
    "binding_projection",
    "build_checkpoint_training_identity",
    "build_execution_binding",
    "canonical_sha256",
    "checkpoint_training_identity_fields",
    "checkpoint_training_identity_projection",
    "expected_checkpoint_training_identity",
    "learned_agent_rows",
    "load_strict_json_mapping",
    "resolved_agent_hyperparameters",
    "scientific_config_projection",
    "sha256_file",
    "training_budget_identity",
    "validate_checkpoint_training_identity",
    "validate_execution_binding",
    "validate_scientific_config",
]
