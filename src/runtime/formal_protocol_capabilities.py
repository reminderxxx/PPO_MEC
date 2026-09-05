"""Authoritative fail-closed capability routing for Formal Protocol versions.

Every known version is registered explicitly.  Callers must not infer execution
capabilities from a version prefix, lexical comparison, or an open-ended range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION: Final = "1.0.0"
ACTIVE_EXECUTION_PROTOCOL_VERSION: Final = "2.7.0"


class FormalProtocolCapabilityError(ValueError):
    """Raised when a Protocol version or requested capability is not registered."""


@dataclass(frozen=True)
class FormalProtocolCapabilities:
    version: str
    execution_status: str
    persisted_resolved_execution_context_required: bool
    explicit_python_and_environment_required: bool
    execution_binding_required: bool
    agent_order_contract_required: bool
    active_bundle_required: bool
    active_bundle_resource_resolution_required: bool
    exogenous_request_execution_required: bool
    full_environment_projection_required: bool
    request_subject_lifecycle_required: bool
    nullable_metric_contract_required: bool
    generated_checkpoint_resource_required: bool
    cell_artifact_publication_required: bool
    holdout_capability: bool = False

    @property
    def live_execution_allowed(self) -> bool:
        return self.execution_status == "active_execution"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"live_execution_allowed": self.live_execution_allowed}


def _capabilities(
    version: str,
    *,
    status: str = "historical_audit_only",
    context: bool = False,
    environment: bool = False,
    binding: bool = False,
    order: bool = False,
    bundle: bool = False,
    bundle_resources: bool = False,
    exogenous: bool = False,
    projection: bool = False,
    lifecycle: bool = False,
    nullable: bool = False,
    generated_checkpoints: bool = False,
    cell_publication: bool = False,
) -> FormalProtocolCapabilities:
    return FormalProtocolCapabilities(
        version=version,
        execution_status=status,
        persisted_resolved_execution_context_required=context,
        explicit_python_and_environment_required=environment,
        execution_binding_required=binding,
        agent_order_contract_required=order,
        active_bundle_required=bundle,
        active_bundle_resource_resolution_required=bundle_resources,
        exogenous_request_execution_required=exogenous,
        full_environment_projection_required=projection,
        request_subject_lifecycle_required=lifecycle,
        nullable_metric_contract_required=nullable,
        generated_checkpoint_resource_required=generated_checkpoints,
        cell_artifact_publication_required=cell_publication,
    )


# This table is deliberately verbose.  A new Protocol is unknown until an
# explicit row is reviewed and added; it never inherits the active permission.
_REGISTRY: Final[dict[str, FormalProtocolCapabilities]] = {
    "1.0.0": _capabilities("1.0.0"),
    "1.1.0": _capabilities("1.1.0"),
    "1.2.0": _capabilities("1.2.0"),
    "1.3.0": _capabilities("1.3.0"),
    "1.4.0": _capabilities("1.4.0"),
    "1.5.0": _capabilities("1.5.0", context=True, environment=True),
    "1.6.0": _capabilities(
        "1.6.0", context=True, environment=True, binding=True
    ),
    "1.7.0": _capabilities(
        "1.7.0", context=True, environment=True, binding=True, order=True
    ),
    "1.8.0": _capabilities(
        "1.8.0", context=True, environment=True, binding=True, order=True, bundle=True
    ),
    "1.9.0": _capabilities(
        "1.9.0", context=True, environment=True, binding=True, order=True,
        bundle=True, bundle_resources=True,
    ),
    "2.0.0": _capabilities(
        "2.0.0", context=True, environment=True, binding=True, order=True,
        bundle=True, bundle_resources=True, exogenous=True,
    ),
    "2.1.0": _capabilities(
        "2.1.0", context=True, environment=True, binding=True, order=True,
        bundle=True, bundle_resources=True, exogenous=True, projection=True,
    ),
    "2.2.0": _capabilities(
        "2.2.0", context=True, environment=True, binding=True, order=True,
        bundle=True, bundle_resources=True, exogenous=True, projection=True, lifecycle=True,
    ),
    "2.3.0": _capabilities(
        "2.3.0", context=True, environment=True, binding=True, order=True,
        bundle=True, bundle_resources=True, exogenous=True, projection=True,
        lifecycle=True, nullable=True,
    ),
    "2.4.0": _capabilities(
        "2.4.0", context=True, environment=True,
        binding=True, order=True, bundle=True, bundle_resources=True,
        exogenous=True, projection=True, lifecycle=True, nullable=True,
    ),
    "2.5.0": _capabilities(
        "2.5.0", context=True, environment=True,
        binding=True, order=True, bundle=True, bundle_resources=True,
        exogenous=True, projection=True, lifecycle=True, nullable=True,
        generated_checkpoints=True,
    ),
    "2.6.0": _capabilities(
        "2.6.0", context=True, environment=True,
        binding=True, order=True, bundle=True, bundle_resources=True,
        exogenous=True, projection=True, lifecycle=True, nullable=True,
        generated_checkpoints=True, cell_publication=True,
    ),
    "2.7.0": _capabilities(
        "2.7.0", status="active_execution", context=True, environment=True,
        binding=True, order=True, bundle=True, bundle_resources=True,
        exogenous=True, projection=True, lifecycle=True, nullable=True,
        generated_checkpoints=True, cell_publication=True,
    ),
}


def get_protocol_capabilities(version: object) -> FormalProtocolCapabilities:
    if not isinstance(version, str) or version not in _REGISTRY:
        raise FormalProtocolCapabilityError(
            f"unregistered Formal Protocol version: {version!r}"
        )
    return _REGISTRY[version]


def require_live_execution_protocol(version: object) -> FormalProtocolCapabilities:
    capabilities = get_protocol_capabilities(version)
    if not capabilities.live_execution_allowed:
        raise FormalProtocolCapabilityError(
            f"Formal Protocol {capabilities.version} is historical audit-only"
        )
    if capabilities.version != ACTIVE_EXECUTION_PROTOCOL_VERSION:
        raise FormalProtocolCapabilityError("active Formal Protocol registry mismatch")
    if capabilities.holdout_capability:
        raise FormalProtocolCapabilityError("active Formal Protocol exposes holdout capability")
    return capabilities


def protocol_capability_matrix() -> dict[str, object]:
    return {
        "formal_protocol_capability_routing_contract_version": (
            FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION
        ),
        "active_execution_protocol_version": ACTIVE_EXECUTION_PROTOCOL_VERSION,
        "unknown_versions_fail_closed": True,
        "versions": {
            version: _REGISTRY[version].to_dict() for version in _REGISTRY
        },
    }


__all__ = [
    "ACTIVE_EXECUTION_PROTOCOL_VERSION",
    "FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION",
    "FormalProtocolCapabilities",
    "FormalProtocolCapabilityError",
    "get_protocol_capabilities",
    "protocol_capability_matrix",
    "require_live_execution_protocol",
]
