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


AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION = "2.0.0"
FORMAL_TRAINING_EXECUTION_BINDING_VERSION = "1.0.0"
SCIENTIFIC_FIELDS = (
    "learning_rate",
    "entropy_coef",
    "value_coef",
    "auxiliary_coef",
)


class FormalTrainingIdentityError(ValueError):
    """Raised when either layer of the formal training identity drifts."""


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
        if protocol.get("typed_model_cache_formal_protocol_version") in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
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
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        data_and_runtime_identity["formal_agent_order_contract_semantic_sha256"] = (
            protocol["formal_agent_order_contract"]["semantic_sha256"]
        )
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        data_and_runtime_identity["formal_exogenous_request_execution"] = deepcopy(
            protocol["formal_exogenous_request_execution_contract"]
        )
    if protocol.get("typed_model_cache_formal_protocol_version") == "2.3.0":
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
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.1.0", "2.2.0", "2.3.0"}:
        payload["environment_identity"] = {
            "projection_contract_version": protocol[
                "formal_execution_environment_contract"
            ]["identity_projection_contract_version"],
            "full_normalized_projection": deepcopy(dict(environment_identity)),
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        }
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
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
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        data_and_runtime_identity["formal_agent_order_contract_semantic_sha256"] = (
            protocol["formal_agent_order_contract"]["semantic_sha256"]
        )
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        data_and_runtime_identity["formal_exogenous_request_execution"] = deepcopy(
            protocol["formal_exogenous_request_execution_contract"]
        )
    if protocol.get("typed_model_cache_formal_protocol_version") == "2.3.0":
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
    if protocol.get("typed_model_cache_formal_protocol_version") in {"2.1.0", "2.2.0", "2.3.0"}:
        comparisons["environment_identity"] = {
            "projection_contract_version": protocol[
                "formal_execution_environment_contract"
            ]["identity_projection_contract_version"],
            "full_normalized_projection": deepcopy(dict(environment_identity)),
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        }
    if protocol.get("typed_model_cache_formal_protocol_version") in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
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
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise FormalTrainingIdentityError(f"checkpoint provenance identity mismatch: {field}")
    return {"status": "pass", **expected}


__all__ = [
    "AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION",
    "FORMAL_TRAINING_EXECUTION_BINDING_VERSION",
    "FormalTrainingIdentityError",
    "agent_matrix_identity",
    "atomic_create_execution_binding",
    "binding_projection",
    "build_execution_binding",
    "canonical_sha256",
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
