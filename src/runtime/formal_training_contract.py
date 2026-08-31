"""Executable training bindings for the typed model-cache formal protocol.

The helpers in this module deliberately contain no training loop.  They turn a
versioned manifest and an optional agent-config companion into the small set of
values that the shared training entrypoint is allowed to consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from src.runtime.formal_training_identity import (
    FormalTrainingIdentityError,
    resolved_agent_hyperparameters,
    validate_execution_binding,
    validate_scientific_config,
)


FORMAL_TRAINING_CONTRACT_VERSION = "1.0.0"
FORMAL_TRAINING_CONTRACT_V2_VERSION = "2.0.0"
LEGACY_CHECKPOINT_EVERY_UPDATES = 1


class FormalTrainingContractError(ValueError):
    """Raised when a frozen training value is missing, invalid, or overridden."""


def positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise FormalTrainingContractError(f"{field_name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalTrainingContractError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0 or float(numeric) != float(value):
        raise FormalTrainingContractError(f"{field_name} must be a positive integer")
    return numeric


def finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise FormalTrainingContractError(f"{field_name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalTrainingContractError(f"{field_name} must be finite") from exc
    if not isfinite(numeric):
        raise FormalTrainingContractError(f"{field_name} must be finite")
    return numeric


def load_json_mapping(path: str | Path, field_name: str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalTrainingContractError(f"unable to load {field_name}: {target}") from exc
    if not isinstance(payload, dict):
        raise FormalTrainingContractError(f"{field_name} must be a JSON object")
    return payload


def checkpoint_snapshot_indices(expected_updates: int, checkpoint_every_updates: int) -> list[int]:
    expected = positive_int(expected_updates, "expected_updates")
    cadence = positive_int(checkpoint_every_updates, "checkpoint_every_updates")
    return [index for index in range(1, expected + 1) if index % cadence == 0]


def should_save_checkpoint(update_index: int, checkpoint_every_updates: int) -> bool:
    index = positive_int(update_index, "update_index")
    cadence = positive_int(checkpoint_every_updates, "checkpoint_every_updates")
    return index % cadence == 0


def checkpoint_schedule_metadata(
    *, checkpoint_every_updates: int, expected_update_count: int | None
) -> dict[str, Any]:
    cadence = positive_int(checkpoint_every_updates, "checkpoint_every_updates")
    expected = (
        positive_int(expected_update_count, "expected_update_count")
        if expected_update_count is not None
        else None
    )
    return {
        "checkpoint_schedule_version": "1.0.0",
        "checkpoint_every_updates": cadence,
        "candidate_update_indices": (
            checkpoint_snapshot_indices(expected, cadence) if expected is not None else None
        ),
        "resume_latest_saved_each_update": True,
        "resume_latest_selection_eligible": False,
    }


def validate_resume_checkpoint_schedule(
    metadata: Mapping[str, Any], *, checkpoint_every_updates: int
) -> int:
    cadence = positive_int(checkpoint_every_updates, "checkpoint_every_updates")
    schedule = metadata.get("checkpoint_schedule")
    if not isinstance(schedule, Mapping):
        if cadence == LEGACY_CHECKPOINT_EVERY_UPDATES:
            return cadence
        raise FormalTrainingContractError("resume checkpoint lacks checkpoint_schedule")
    observed = positive_int(
        schedule.get("checkpoint_every_updates"),
        "checkpoint_schedule.checkpoint_every_updates",
    )
    if observed != cadence:
        raise FormalTrainingContractError(
            "resume checkpoint cadence mismatch: "
            f"checkpoint={observed}, resolved={cadence}"
        )
    return observed


def _formal_values(protocol: Mapping[str, Any], agent_name: str) -> dict[str, Any]:
    if protocol.get("typed_model_cache_formal_protocol_version") not in {
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
        "2.0.0",
    }:
        raise FormalTrainingContractError("formal training requires protocol version 1.1, 1.2, or 1.3")
    budget = protocol.get("training_budget")
    if not isinstance(budget, Mapping):
        raise FormalTrainingContractError("formal protocol lacks training_budget")
    agent_configs = budget.get("agent_configs")
    if not isinstance(agent_configs, Mapping) or agent_name not in agent_configs:
        raise FormalTrainingContractError(f"formal protocol lacks agent config for {agent_name}")
    raw_agent = agent_configs[agent_name]
    if not isinstance(raw_agent, Mapping):
        raise FormalTrainingContractError(f"agent config for {agent_name} must be an object")
    return {
        "episodes": positive_int(
            budget.get("episodes_per_learned_agent_seed_capacity"), "training_budget.episodes"
        ),
        "update_every": positive_int(
            budget.get("update_interval_episodes"), "training_budget.update_interval_episodes"
        ),
        "batch_size": positive_int(budget.get("batch_size"), "training_budget.batch_size"),
        "max_steps": positive_int(
            budget.get("max_steps_per_episode"), "training_budget.max_steps_per_episode"
        ),
        "checkpoint_every_updates": positive_int(
            budget.get("checkpoint_frequency_updates"),
            "training_budget.checkpoint_frequency_updates",
        ),
        "expected_update_count": positive_int(
            budget.get("expected_update_count"), "training_budget.expected_update_count"
        ),
        "agent_config": dict(raw_agent),
    }


def _validate_agent_config_companion(
    companion: Mapping[str, Any],
    *, protocol: Mapping[str, Any] | None,
    agent_name: str,
) -> dict[str, Any]:
    if companion.get("agent_training_config_contract_version") != "1.0.0":
        raise FormalTrainingContractError("unsupported agent training config contract")
    configs = companion.get("agents")
    if not isinstance(configs, Mapping) or agent_name not in configs:
        raise FormalTrainingContractError(f"agent config companion lacks {agent_name}")
    config = configs[agent_name]
    if not isinstance(config, Mapping):
        raise FormalTrainingContractError(f"agent config companion entry for {agent_name} is invalid")
    resolved = dict(config)
    if protocol is not None:
        expected = _formal_values(protocol, agent_name)["agent_config"]
        if resolved != expected:
            raise FormalTrainingContractError(
                f"agent config companion mismatch for {agent_name}"
            )
        expected_hash = protocol.get("hashes", {}).get("semantic_sha256")
        bound_hash = companion.get("protocol_semantic_sha256")
        if expected_hash and bound_hash != expected_hash:
            raise FormalTrainingContractError("agent config companion protocol hash mismatch")
    return resolved


def _cli_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


@dataclass(frozen=True)
class ResolvedTrainingContract:
    contract_version: str
    formal_protocol_version: str | None
    formal_protocol_semantic_sha256: str | None
    episodes: int
    update_every: int
    batch_size: int
    max_steps: int
    checkpoint_every_updates: int
    expected_update_count: int
    agent_config: dict[str, Any]
    agent_scientific_config_semantic_sha256: str | None = None
    formal_training_execution_binding_sha256: str | None = None
    execution_commit: str | None = None
    resolved_execution_context_sha256: str | None = None
    environment_fingerprint: str | None = None
    dependency_fingerprint: str | None = None
    formal_agent_order_contract_semantic_sha256: str | None = None
    active_formal_bundle_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formal_training_contract_version": self.contract_version,
            "formal_protocol_version": self.formal_protocol_version,
            "formal_protocol_semantic_sha256": self.formal_protocol_semantic_sha256,
            "episodes": self.episodes,
            "update_every": self.update_every,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "expected_update_count": self.expected_update_count,
            "agent_config": dict(self.agent_config),
            "agent_scientific_config_semantic_sha256": (
                self.agent_scientific_config_semantic_sha256
            ),
            "formal_training_execution_binding_sha256": (
                self.formal_training_execution_binding_sha256
            ),
            "execution_commit": self.execution_commit,
            "resolved_execution_context_sha256": (
                self.resolved_execution_context_sha256
            ),
            "environment_fingerprint": self.environment_fingerprint,
            "dependency_fingerprint": self.dependency_fingerprint,
            "formal_agent_order_contract_semantic_sha256": (
                self.formal_agent_order_contract_semantic_sha256
            ),
            "active_formal_bundle_sha256": self.active_formal_bundle_sha256,
        }


def resolve_training_contract(
    *,
    agent_name: str,
    profile_defaults: Mapping[str, Any],
    cli_values: Mapping[str, Any],
    formal_protocol: Mapping[str, Any] | None = None,
    agent_config_companion: Mapping[str, Any] | None = None,
    scientific_config: Mapping[str, Any] | None = None,
    execution_binding: Mapping[str, Any] | None = None,
    resolved_execution_context: Mapping[str, Any] | None = None,
) -> ResolvedTrainingContract:
    """Resolve legacy defaults or enforce an exact manifest-bound formal contract."""

    if formal_protocol is None:
        episodes = positive_int(
            _cli_or_default(cli_values.get("episodes"), profile_defaults["episodes"]), "episodes"
        )
        update_every = positive_int(
            _cli_or_default(cli_values.get("update_every"), profile_defaults["update_every"]),
            "update_every",
        )
        batch_size = positive_int(
            _cli_or_default(cli_values.get("batch_size"), profile_defaults["batch_size"]),
            "batch_size",
        )
        max_steps = positive_int(
            _cli_or_default(cli_values.get("max_steps"), profile_defaults["max_steps"]),
            "max_steps",
        )
        cadence = positive_int(
            _cli_or_default(
                cli_values.get("checkpoint_every_updates"),
                LEGACY_CHECKPOINT_EVERY_UPDATES,
            ),
            "checkpoint_every_updates",
        )
        config = (
            _validate_agent_config_companion(
                agent_config_companion, protocol=None, agent_name=agent_name
            )
            if agent_config_companion is not None
            else {}
        )
        return ResolvedTrainingContract(
            contract_version=FORMAL_TRAINING_CONTRACT_VERSION,
            formal_protocol_version=None,
            formal_protocol_semantic_sha256=None,
            episodes=episodes,
            update_every=update_every,
            batch_size=batch_size,
            max_steps=max_steps,
            checkpoint_every_updates=cadence,
            expected_update_count=(episodes + update_every - 1) // update_every,
            agent_config=config,
        )

    frozen = _formal_values(formal_protocol, agent_name)
    for field_name in ("episodes", "update_every", "batch_size", "max_steps"):
        supplied = cli_values.get(field_name)
        if supplied is not None and positive_int(supplied, field_name) != frozen[field_name]:
            raise FormalTrainingContractError(
                f"formal CLI override rejected for {field_name}: "
                f"supplied={supplied}, frozen={frozen[field_name]}"
            )
    supplied_cadence = cli_values.get("checkpoint_every_updates")
    if supplied_cadence is not None and positive_int(
        supplied_cadence, "checkpoint_every_updates"
    ) != frozen["checkpoint_every_updates"]:
        raise FormalTrainingContractError("formal CLI override rejected for checkpoint_every_updates")
    protocol_version = str(
        formal_protocol.get("typed_model_cache_formal_protocol_version")
    )
    identity_values: dict[str, str | None] = {
        "agent_scientific_config_semantic_sha256": None,
        "formal_training_execution_binding_sha256": None,
        "execution_commit": None,
        "resolved_execution_context_sha256": None,
        "environment_fingerprint": None,
        "dependency_fingerprint": None,
        "formal_agent_order_contract_semantic_sha256": None,
        "active_formal_bundle_sha256": None,
    }
    contract_version = FORMAL_TRAINING_CONTRACT_VERSION
    if protocol_version in {"1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0"}:
        if agent_config_companion is not None:
            raise FormalTrainingContractError(
                "Protocol v1.6 rejects legacy --agent_config_path companion"
            )
        if scientific_config is None:
            raise FormalTrainingContractError(
                "formal training requires --agent_scientific_config_path"
            )
        if execution_binding is None:
            raise FormalTrainingContractError(
                "formal training requires --formal_training_execution_binding_path"
            )
        if resolved_execution_context is None:
            raise FormalTrainingContractError(
                "formal training requires --resolved_execution_context_path"
            )
        scientific_identity = resolved_execution_context.get("scientific_identity")
        command = resolved_execution_context.get("command_expansion")
        if not isinstance(scientific_identity, Mapping) or not isinstance(command, Mapping):
            raise FormalTrainingContractError("resolved execution context identity is incomplete")
        try:
            scientific_report = validate_scientific_config(
                scientific_config, protocol=formal_protocol
            )
            binding_report = validate_execution_binding(
                execution_binding,
                protocol=formal_protocol,
                scientific_config=scientific_config,
                execution_commit=str(scientific_identity.get("execution_commit") or ""),
                environment_identity=scientific_identity,
                command_matrix_sha256=str(
                    command.get("resolved_command_matrix_sha256") or ""
                ),
                active_formal_bundle_sha256=(
                    str(scientific_identity.get("active_formal_bundle_sha256") or "")
                    if protocol_version in {"1.8.0", "1.9.0", "2.0.0"}
                    else None
                ),
            )
            agent_config = resolved_agent_hyperparameters(
                scientific_config, agent_name
            )
        except FormalTrainingIdentityError as exc:
            raise FormalTrainingContractError(str(exc)) from exc
        if agent_config != frozen["agent_config"]:
            raise FormalTrainingContractError(
                f"scientific config/protocol mismatch for {agent_name}"
            )
        identity_values = {
            "agent_scientific_config_semantic_sha256": scientific_report[
                "config_semantic_sha256"
            ],
            "formal_training_execution_binding_sha256": binding_report[
                "binding_full_sha256"
            ],
            "execution_commit": str(scientific_identity["execution_commit"]),
            "resolved_execution_context_sha256": str(
                resolved_execution_context.get("context_sha256") or ""
            ),
            "environment_fingerprint": str(
                scientific_identity.get("environment_fingerprint") or ""
            ),
            "dependency_fingerprint": str(
                scientific_identity.get("dependency_fingerprint") or ""
            ),
            "formal_agent_order_contract_semantic_sha256": (
                str(
                    scientific_identity.get(
                        "formal_agent_order_contract_semantic_sha256"
                    )
                    or ""
                )
                if protocol_version in {"1.7.0", "1.8.0", "1.9.0", "2.0.0"}
                else None
            ),
            "active_formal_bundle_sha256": (
                str(scientific_identity.get("active_formal_bundle_sha256") or "")
                if protocol_version in {"1.8.0", "1.9.0", "2.0.0"}
                else None
            ),
        }
        if scientific_identity.get("agent_scientific_config_semantic_sha256") != (
            identity_values["agent_scientific_config_semantic_sha256"]
        ):
            raise FormalTrainingContractError(
                "resolved context scientific config identity mismatch"
            )
        if scientific_identity.get("formal_training_execution_binding_sha256") != (
            identity_values["formal_training_execution_binding_sha256"]
        ):
            raise FormalTrainingContractError(
                "resolved context execution binding identity mismatch"
            )
        if protocol_version in {"1.7.0", "1.8.0", "1.9.0", "2.0.0"} and identity_values[
            "formal_agent_order_contract_semantic_sha256"
        ] != formal_protocol["formal_agent_order_contract"]["semantic_sha256"]:
            raise FormalTrainingContractError(
                "resolved context formal agent order contract identity mismatch"
            )
        if protocol_version in {"1.8.0", "1.9.0", "2.0.0"} and identity_values[
            "active_formal_bundle_sha256"
        ] != execution_binding.get("active_formal_bundle_sha256"):
            raise FormalTrainingContractError(
                "resolved context active formal bundle identity mismatch"
            )
        contract_version = FORMAL_TRAINING_CONTRACT_V2_VERSION
    else:
        if agent_config_companion is None:
            raise FormalTrainingContractError("formal training requires --agent_config_path")
        agent_config = _validate_agent_config_companion(
            agent_config_companion, protocol=formal_protocol, agent_name=agent_name
        )
    return ResolvedTrainingContract(
        contract_version=contract_version,
        formal_protocol_version=protocol_version,
        formal_protocol_semantic_sha256=formal_protocol.get("hashes", {}).get(
            "semantic_sha256"
        ),
        episodes=frozen["episodes"],
        update_every=frozen["update_every"],
        batch_size=frozen["batch_size"],
        max_steps=frozen["max_steps"],
        checkpoint_every_updates=frozen["checkpoint_every_updates"],
        expected_update_count=frozen["expected_update_count"],
        agent_config=agent_config,
        **identity_values,
    )


def audited_agent_config(agent: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Read the instantiated agent's serialized config and match every requested field."""

    exporter = getattr(agent, "_checkpoint_config", None)
    if not callable(exporter):
        if requested:
            raise FormalTrainingContractError("agent cannot export resolved checkpoint config")
        return {}
    actual = exporter()
    if not isinstance(actual, Mapping):
        raise FormalTrainingContractError("agent checkpoint config exporter returned a non-object")
    audit: dict[str, Any] = {}
    for field_name, expected in requested.items():
        if field_name not in actual:
            raise FormalTrainingContractError(
                f"instantiated agent does not expose requested field: {field_name}"
            )
        observed = actual[field_name]
        if isinstance(expected, float):
            if abs(finite_float(observed, field_name) - expected) > 1e-12:
                raise FormalTrainingContractError(
                    f"instantiated agent config mismatch for {field_name}"
                )
        elif observed != expected:
            raise FormalTrainingContractError(
                f"instantiated agent config mismatch for {field_name}"
            )
        audit[field_name] = observed
    return audit


__all__ = [
    "FORMAL_TRAINING_CONTRACT_VERSION",
    "FORMAL_TRAINING_CONTRACT_V2_VERSION",
    "FormalTrainingContractError",
    "ResolvedTrainingContract",
    "audited_agent_config",
    "checkpoint_schedule_metadata",
    "checkpoint_snapshot_indices",
    "load_json_mapping",
    "positive_int",
    "resolve_training_contract",
    "should_save_checkpoint",
    "validate_resume_checkpoint_schedule",
]
