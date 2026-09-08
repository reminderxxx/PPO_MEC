"""Continuation-only orchestration identities, preserving original projections.

The eight-phase adapter uses old expand_command_plan/stable_cell_id and does not
call the old public main. Formula equivalence is tested against the frozen source.
"""
from __future__ import annotations

from . import PHASES
from .identity import ContinuationError, digest


def phase_plan(phase, protocol, context, binding, context_file_sha256, registry_sha256,
               expand_command_plan):
    if phase not in PHASES:
        raise ContinuationError("unauthorized phase")
    if phase == "complete_without_holdout":
        commands, outputs, coordinates, retries = [], [], [], 0
    else:
        spec = protocol["execution_contract"]["command_templates"][phase]
        plan = expand_command_plan(spec, context["resolved_expansion_context"])
        commands, outputs, coordinates = plan["commands"], plan["expected_outputs"], plan["matrix_contexts"]
        retries = int(spec.get("infrastructure_retries", 1))
    base = digest({
        "protocol": protocol["hashes"]["semantic_sha256"], "phase": phase,
        "commands": commands, "resolved_execution_context_sha256": context["context_sha256"],
        "resolved_execution_context_file_sha256": context_file_sha256,
        "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
        "active_formal_bundle_sha256": context["scientific_identity"]["active_formal_bundle_sha256"],
        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
            "formal_nullable_metric_aggregation_contract"]["semantic_sha256"],
        "generated_checkpoint_registry_canonical_sha256": None,
    })
    input_hash = digest({"base_phase_input_hash": base,
                         "generated_checkpoint_registry_canonical_sha256": registry_sha256})
    hashes = [digest(list(argv)) for argv in commands]
    if len(set(hashes)) != len(hashes):
        raise ContinuationError("duplicate command identity")
    return {"phase": phase, "commands": commands, "expected_outputs": outputs,
            "matrix_contexts": coordinates, "infrastructure_retries": retries,
            "input_hash": input_hash, "command_hashes": hashes}


def run_identity(protocol, context, binding, context_file_sha256, run_root, execution_commit,
                 environment_fingerprint):
    return digest({
        "output_root": str(run_root), "protocol": protocol["hashes"]["semantic_sha256"],
        "resource_registry": protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"],
        "environment": environment_fingerprint, "execution_commit": execution_commit,
        "resolved_execution_context_sha256": context["context_sha256"],
        "resolved_execution_context_file_sha256": context_file_sha256,
        "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
        "active_formal_bundle_sha256": context["scientific_identity"]["active_formal_bundle_sha256"],
    })


def cell_identity_fields(protocol, context, binding, run_id, execution_commit, environment_fingerprint):
    return dict(
        run_id=run_id, execution_commit=execution_commit,
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        resource_registry_semantic_sha256=protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"],
        environment_fingerprint=environment_fingerprint,
        split_semantic_sha256=protocol["identity"]["split_semantic_sha256"],
        window_contract_semantic_sha256=protocol["execution_contract"]["window_consumption_contract"]["semantic_sha256"],
        catalog_fingerprint=protocol["identity"]["catalog_fingerprint"],
        runtime_identity=digest(protocol["identity"]["typed_runtime_contract_hashes_by_capacity"]),
        command_matrix_sha256=digest({
            "command_templates": protocol["execution_contract"]["command_templates"],
            "resolved_execution_context_sha256": context["context_sha256"],
            "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
            "active_formal_bundle_sha256": context["scientific_identity"]["active_formal_bundle_sha256"],
            "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                "formal_nullable_metric_aggregation_contract"]["semantic_sha256"],
        }),
    )


def cell_input_hash(phase, coordinates, command, protocol, context, registry_sha256):
    if phase not in PHASES[:5]:
        raise ContinuationError("not a continuation cell phase")
    return digest({
        "protocol": protocol["hashes"]["semantic_sha256"], "phase": phase,
        "coordinates": coordinates, "command": command,
        "active_formal_bundle_sha256": context["scientific_identity"]["active_formal_bundle_sha256"],
        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
            "formal_nullable_metric_aggregation_contract"]["semantic_sha256"],
        "generated_checkpoint_registry_canonical_sha256": registry_sha256,
    })
