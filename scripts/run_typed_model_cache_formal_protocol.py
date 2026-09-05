"""Run the append-only typed model-cache formal protocol v1.1.

This ordinary runner deliberately exposes no holdout or hidden-data option.
Use ``--dry-run`` during G14R to validate expansion without creating results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    FormalExecutionError,
    PHASE_ORDER,
    expand_command_plan,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    TransactionalPhaseRunner,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    execute_cell_artifact_transaction,
    resolve_child_output_descriptor,
    single_child_directory,
    stable_cell_id,
)
from src.runtime.formal_execution_environment import (
    protocol_bound_extensions_from_protocol,
    resolve_execution_environment,
)
from src.runtime.active_formal_bundle import (
    ACTIVE_PROTOCOL_VERSION,
    ActiveFormalBundleError,
    DEFAULT_ACTIVE_INDEX_RELATIVE,
    build_active_bundle_resource_resolution_audit,
    validate_active_formal_bundle,
)
from src.runtime.portable_resource_identity import load_registry
from src.runtime.generated_checkpoint_resources import (
    build_generated_checkpoint_registry,
    load_generated_checkpoint_registry,
    publish_or_validate_generated_checkpoint_registry,
)
from src.runtime.formal_training_identity import (
    atomic_create_execution_binding,
    build_execution_binding,
    load_strict_json_mapping,
    validate_execution_binding,
)
from src.runtime.resolved_formal_execution_context import (
    RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME,
    atomic_create_resolved_formal_execution_context,
    build_resolved_formal_execution_context,
    load_resolved_formal_execution_context,
)
from src.runtime.formal_invalid_run_registry import (
    PermanentlyInvalidFormalReferenceError,
    reject_permanently_invalid_formal_references,
)
from src.runtime.formal_protocol_capabilities import (
    FormalProtocolCapabilityError,
    get_protocol_capabilities,
    require_live_execution_protocol,
)


def _absolute_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def resolved_expansion_context(
    protocol: dict,
    *,
    protocol_path: str,
    output_root: str,
    python_executable: str,
    active_formal_bundle_sha256: str | None = None,
    active_protocol_index_path: str | None = None,
    active_bundle_resource_resolution_audit_sha256: str | None = None,
) -> dict:
    """Resolve every host/run location once at the outermost runner."""

    context = dict(protocol["execution_contract"]["default_expansion_context"])
    requested_output_root = str(Path(output_root).resolve())
    for key, value in list(context.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            context[key] = requested_output_root + value[
                len("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            ]
    context.update(
        protocol_path=str(Path(protocol_path).resolve()),
        output_root=requested_output_root,
        python_executable=str(Path(python_executable).absolute()),
        clean_worktree_root=str(ROOT),
        repository_root=str(ROOT),
        data_root=str((ROOT / "data").resolve()),
        checkpoint_root=requested_output_root,
        resolved_execution_context_path=str(
            Path(requested_output_root)
            / RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME
        ),
        formal_training_execution_binding_path=str(
            Path(requested_output_root)
            / "formal_training_execution_binding.json"
        ),
        resolve_relative_paths_against_repository_root=True,
    )
    if active_formal_bundle_sha256 is not None:
        context["active_formal_bundle_sha256"] = active_formal_bundle_sha256
    if active_protocol_index_path is not None:
        context["active_protocol_index_path"] = str(
            Path(active_protocol_index_path).resolve()
        )
    if active_bundle_resource_resolution_audit_sha256 is not None:
        context["active_bundle_resource_resolution_audit_sha256"] = (
            active_bundle_resource_resolution_audit_sha256
        )
    for _ in range(4):
        changed = False
        for key, value in list(context.items()):
            if not isinstance(value, str):
                continue
            rendered = value
            for name in re.findall(r"\{([a-z][a-z0-9_]*)\}", value):
                replacement = context.get(name)
                if replacement is not None and not isinstance(
                    replacement, (dict, list, tuple)
                ):
                    rendered = rendered.replace("{" + name + "}", str(replacement))
            if rendered != value:
                context[key] = rendered
                changed = True
        if not changed:
            break
    for key in (
        "protocol_artifact_root",
        "resource_registry_path",
        "agent_config_path",
        "agent_scientific_config_path",
        "formal_agent_order_contract_path",
        "dev_window_plan_path",
        "formal_window_plan_path",
        "train_window_plan_path",
        "window_consumption_contract_path",
    ):
        value = context.get(key)
        if isinstance(value, str) and value and "{" not in value:
            context[key] = _absolute_project_path(value)
    unresolved = {
        key: value
        for key, value in context.items()
        if isinstance(value, str) and ("{" in value or "}" in value)
    }
    if unresolved:
        raise FormalExecutionError(
            f"resolved execution context contains placeholders: {sorted(unresolved)}"
        )
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-path",
        default="",
        help="optional v1.8 consistency assertion; omitted path is resolved from the active index",
    )
    parser.add_argument(
        "--active-protocol-index",
        default=DEFAULT_ACTIVE_INDEX_RELATIVE,
        help="unique active Protocol index; alternate indexes are rejected",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--phase", choices=PHASE_ORDER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-cell-ledger", action="store_true")
    parser.add_argument("--finalize-phase-only", action="store_true")
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--execution-environment-manifest", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--training-entrypoint-acceptance",
        action="store_true",
        help=(
            "Pre-readiness active-bundle acceptance; restricted to --preflight and "
            "does not authorize training or formal performance."
        ),
    )
    parser.add_argument("--non-formal-rehearsal-profile", default="")
    args = parser.parse_args()
    if args.preflight == bool(args.phase):
        parser.error("select exactly one of --preflight or --phase")
    if args.resume_from_cell_ledger and not args.resume:
        parser.error("--resume-from-cell-ledger requires --resume")
    if args.training_entrypoint_acceptance and (
        not args.preflight
        or args.resume
        or args.finalize_phase_only
        or args.non_formal_rehearsal_profile
    ):
        parser.error(
            "--training-entrypoint-acceptance is restricted to a fresh --preflight"
        )
    return args


def reject_invalid_run_root(protocol: dict, output_root: str | Path) -> None:
    version = protocol.get("typed_model_cache_formal_protocol_version")
    capabilities = get_protocol_capabilities(version)
    supersession = protocol.get("supersession", {})
    references = (
        supersession.get("invalid_execution_runs", [])
        if capabilities.persisted_resolved_execution_context_required
        else supersession.get("invalid_g14c_v4_runs", [])
        if version == "1.4.0"
        else []
    )
    invalid_run_ids = {
        str(item.get("run_id"))
        for item in references
        if isinstance(item, dict) and item.get("run_id")
    }
    if any(part in invalid_run_ids for part in Path(output_root).resolve().parts):
        raise FormalExecutionError(
            "legacy invalid G14C run root is permanently rejected"
        )


def load_protocol(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FormalExecutionError("protocol manifest must be an object")
    validate_protocol_v1_1(payload)
    return payload


def validate_complete_without_holdout_gate(
    run_root: str | Path, protocol: dict
) -> dict:
    root = Path(run_root)
    gate_path = root / "formal_gate.json"
    if gate_path.is_symlink() or not gate_path.is_file():
        raise FormalExecutionError(
            "complete_without_holdout requires a legal formal_gate.json"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    required = {
        "passed": True,
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "completeness_only": True,
        "performance_threshold_used": False,
        "holdout_opened": False,
        "exact_count_status": "pass",
    }
    if not isinstance(gate, dict) or any(
        gate.get(key) != value for key, value in required.items()
    ):
        raise FormalExecutionError(
            "complete_without_holdout rejected an invalid or failed formal gate"
        )
    if gate.get("missing_outputs") or gate.get("exact_count_mismatches"):
        raise FormalExecutionError("complete_without_holdout gate is incomplete")
    registry = gate.get("generated_checkpoint_registry_audit")
    if not isinstance(registry, dict) or registry.get("status") != "pass":
        raise FormalExecutionError(
            "complete_without_holdout gate lacks generated registry validation"
        )
    ledger_path = root / "phase_state.jsonl"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise FormalExecutionError(
            "complete_without_holdout requires the phase ledger"
        )
    terminals = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("phase") == "formal_gate" and row.get("status") == "completed":
                terminals.append(row)
    if len(terminals) != 1:
        raise FormalExecutionError(
            "complete_without_holdout requires one completed formal_gate terminal"
        )
    return gate


def main() -> None:
    args = parse_args()
    try:
        reject_permanently_invalid_formal_references(
            [args.output_root, getattr(args, "resume_run_root", "")]
        )
    except PermanentlyInvalidFormalReferenceError as exc:
        raise FormalExecutionError(str(exc)) from exc
    active_bundle = None
    if not args.protocol_path:
        try:
            active_bundle = validate_active_formal_bundle(
                repository_root=ROOT,
                index_path=args.active_protocol_index,
                execution_environment_manifest_path=(
                    args.execution_environment_manifest or None
                ),
                require_ready=not bool(
                    args.non_formal_rehearsal_profile
                    or args.training_entrypoint_acceptance
                ),
                require_origin_main_match=not bool(args.non_formal_rehearsal_profile),
            )
        except ActiveFormalBundleError as exc:
            raise FormalExecutionError(f"active formal bundle gate failed: {exc}") from exc
        args.protocol_path = active_bundle["protocol_path"]
        args.execution_environment_manifest = active_bundle[
            "execution_environment_manifest_path"
        ]
    protocol = load_protocol(args.protocol_path)
    protocol_version = protocol["typed_model_cache_formal_protocol_version"]
    try:
        capabilities = get_protocol_capabilities(protocol_version)
    except FormalProtocolCapabilityError as exc:
        raise FormalExecutionError(str(exc)) from exc
    if args.non_formal_rehearsal_profile:
        profile_path = Path(args.non_formal_rehearsal_profile)
        expected_profile = ROOT / protocol.get("execution_contract", {}).get(
            "nonformal_rehearsal_profile", {}
        ).get("path", "")
        if (
            not args.preflight
            or not profile_path.is_absolute()
            or profile_path.is_symlink()
            or profile_path.resolve() != expected_profile.resolve()
        ):
            raise FormalExecutionError("non-formal rehearsal profile identity mismatch")
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if (
            profile.get("semantic_sha256")
            != protocol["execution_contract"]["nonformal_rehearsal_profile"][
                "semantic_sha256"
            ]
            or canonical_sha256(
                {key: value for key, value in profile.items() if key != "semantic_sha256"}
            )
            != profile.get("semantic_sha256")
            or profile.get("formal") is not False
            or profile.get("performance_evidence") is not False
            or profile.get("holdout_capability") is not False
        ):
            raise FormalExecutionError("non-formal rehearsal profile contract mismatch")
    if capabilities.live_execution_allowed and active_bundle is None:
        try:
            active_bundle = validate_active_formal_bundle(
                repository_root=ROOT,
                index_path=args.active_protocol_index,
                protocol_path=args.protocol_path,
                execution_environment_manifest_path=(
                    args.execution_environment_manifest or None
                ),
                require_ready=not bool(
                    args.non_formal_rehearsal_profile
                    or args.training_entrypoint_acceptance
                ),
                require_origin_main_match=not bool(args.non_formal_rehearsal_profile),
            )
        except ActiveFormalBundleError as exc:
            raise FormalExecutionError(f"active formal bundle gate failed: {exc}") from exc
        if not args.execution_environment_manifest:
            args.execution_environment_manifest = active_bundle[
                "execution_environment_manifest_path"
            ]
    phase = "preflight" if args.preflight else args.phase
    resource_resolution_audit = (
        build_active_bundle_resource_resolution_audit(active_bundle)
        if active_bundle is not None
        else None
    )
    templates = protocol["execution_contract"]["command_templates"]
    requested_output_root = str(Path(args.output_root).resolve())
    reject_invalid_run_root(protocol, requested_output_root)
    environment_resolution = None
    environment_manifest = None
    if capabilities.explicit_python_and_environment_required:
        if not args.python_executable or not args.execution_environment_manifest:
            raise FormalExecutionError(
                "active formal protocol requires explicit Python and execution environment manifest"
            )
        if not Path(args.python_executable).is_absolute():
            raise FormalExecutionError(
                "active formal protocol forbids relative Python or .venv fallback"
            )
    if protocol_version == "1.4.0" or capabilities.explicit_python_and_environment_required:
        if args.execution_environment_manifest:
            environment_manifest = json.loads(
                Path(args.execution_environment_manifest).read_text(encoding="utf-8-sig")
            )
        environment_contract = protocol["formal_execution_environment_contract"]
        environment_resolution = resolve_execution_environment(
            clean_worktree_root=ROOT,
            execution_commit=environment_contract["scientific_identity"][
                "execution_commit"
            ],
            python_executable=args.python_executable or None,
            environment_manifest=environment_manifest,
            expected_identity=environment_contract["scientific_identity"],
            protocol_bound_extensions=(
                protocol_bound_extensions_from_protocol(protocol)
                if capabilities.full_environment_projection_required
                else None
            ),
            forbidden_source_roots=environment_contract.get(
                "forbidden_project_source_roots", []
            ),
            require_clean_git_worktree=True,
        )
        if capabilities.explicit_python_and_environment_required and environment_resolution.runtime_audit.get(
            "resolution_source"
        ) != "explicit_python_executable":
            raise FormalExecutionError(
                "active formal protocol Python must come from explicit outer resolution"
            )
    resolved_python = (
        environment_resolution.python_executable
        if environment_resolution is not None
        else args.python_executable or sys.executable
    )
    context = resolved_expansion_context(
        protocol,
        protocol_path=args.protocol_path,
        output_root=requested_output_root,
        python_executable=resolved_python,
        active_formal_bundle_sha256=(
            active_bundle["active_formal_bundle_sha256"]
            if active_bundle is not None
            else None
        ),
        active_protocol_index_path=(
            args.active_protocol_index if active_bundle is not None else None
        ),
        active_bundle_resource_resolution_audit_sha256=(
            resource_resolution_audit["audit_sha256"]
            if resource_resolution_audit is not None
            else None
        ),
    )
    outer_validation = validate_command_templates(templates, context)
    scientific_config = None
    execution_binding = None
    if capabilities.execution_binding_required:
        if environment_resolution is None:
            raise FormalExecutionError("Protocol v1.6 environment was not resolved")
        scientific_config = load_strict_json_mapping(
            context["agent_scientific_config_path"], "agent scientific config"
        )
        binding_path = Path(context["formal_training_execution_binding_path"])
        observed_commit = str(
            environment_resolution.runtime_audit.get("observed_execution_commit") or ""
        )
        if args.resume or args.finalize_phase_only:
            execution_binding = load_strict_json_mapping(
                binding_path, "formal training execution binding"
            )
            validate_execution_binding(
                execution_binding,
                protocol=protocol,
                scientific_config=scientific_config,
                execution_commit=observed_commit,
                environment_identity=environment_resolution.environment_identity,
                command_matrix_sha256=outer_validation["command_matrix_sha256"],
                active_formal_bundle_sha256=(
                    active_bundle["active_formal_bundle_sha256"]
                    if active_bundle is not None
                    else None
                ),
            )
        else:
            execution_binding = build_execution_binding(
                protocol=protocol,
                scientific_config=scientific_config,
                execution_commit=observed_commit,
                environment_identity=environment_resolution.environment_identity,
                command_matrix_sha256=outer_validation["command_matrix_sha256"],
                active_formal_bundle_sha256=(
                    active_bundle["active_formal_bundle_sha256"]
                    if active_bundle is not None
                    else None
                ),
            )
    resolved_context_payload = None
    resolved_context_report = None
    resolved_context_file_sha256 = None
    if capabilities.persisted_resolved_execution_context_required:
        if environment_resolution is None or not args.execution_environment_manifest:
            raise FormalExecutionError("protocol v1.5 environment was not resolved")
        context_path = Path(context["resolved_execution_context_path"])
        if args.resume or args.finalize_phase_only:
            resolved_context_payload, resolved_context_report = (
                load_resolved_formal_execution_context(
                    context_path,
                    protocol=protocol,
                    clean_worktree_root=ROOT,
                    durable_run_root=requested_output_root,
                    environment_identity=environment_resolution.environment_identity,
                    runtime_audit=environment_resolution.runtime_audit,
                    check_git=True,
                )
            )
            if resolved_context_payload["resolved_expansion_context"] != context:
                raise FormalExecutionError(
                    "same-run resolved expansion context drift"
                )
            if resolved_context_payload["command_expansion"][
                "outer_expansion_sha256"
            ] != outer_validation["command_matrix_sha256"]:
                raise FormalExecutionError(
                    "same-run resolved command matrix drift"
                )
            resolved_context_file_sha256 = resolved_context_report["file_sha256"]
        else:
            resolved_context_payload = build_resolved_formal_execution_context(
                protocol=protocol,
                expansion_context=context,
                environment_identity=environment_resolution.environment_identity,
                runtime_audit=environment_resolution.runtime_audit,
                environment_manifest_path=args.execution_environment_manifest,
                outer_expansion_sha256=outer_validation["command_matrix_sha256"],
                phase_count=outer_validation["phase_count"],
                command_count=outer_validation["command_count"],
                execution_binding=execution_binding,
                active_formal_bundle_sha256=(
                    active_bundle["active_formal_bundle_sha256"]
                    if active_bundle is not None
                    else None
                ),
            )
            encoded_context = (
                json.dumps(
                    resolved_context_payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            resolved_context_file_sha256 = hashlib.sha256(
                encoded_context
            ).hexdigest()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_pass",
                    "selected_phase": phase,
                    "writes_performed": False,
                    "holdout_capability": False,
                    "phase_order": list(PHASE_ORDER),
                    "command_expansion": outer_validation,
                    "resolved_execution_context": (
                        {
                            "context_sha256": resolved_context_payload[
                                "context_sha256"
                            ],
                            "file_sha256": resolved_context_file_sha256,
                            "persisted": False,
                            "outer_expansion_sha256": outer_validation[
                                "command_matrix_sha256"
                            ],
                        }
                        if resolved_context_payload is not None
                        else None
                    ),
                    "execution_environment": (
                        environment_resolution.runtime_audit
                        if environment_resolution is not None
                        else None
                    ),
                    "formal_training_execution_binding": (
                        {
                            "binding_full_sha256": execution_binding[
                                "binding_full_sha256"
                            ],
                            "persisted": False,
                        }
                        if execution_binding is not None
                        else None
                    ),
                    "active_formal_bundle": (
                        {
                            "contract_version": active_bundle[
                                "active_formal_bundle_contract_version"
                            ],
                            "active_formal_bundle_sha256": active_bundle[
                                "active_formal_bundle_sha256"
                            ],
                            "execution_commit": active_bundle["execution_commit"],
                        }
                        if active_bundle is not None
                        else None
                    ),
                    "active_bundle_resource_resolution": resource_resolution_audit,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    try:
        require_live_execution_protocol(protocol_version)
    except FormalProtocolCapabilityError as exc:
        raise FormalExecutionError(str(exc)) from exc

    if phase == "complete_without_holdout":
        validate_complete_without_holdout_gate(args.output_root, protocol)
        command: list[str] | list[list[str]] = []
        expected_outputs: list[str] = []
        retries = 0
        matrix_contexts: list[dict] = []
    else:
        spec = templates[phase]
        plan = expand_command_plan(spec, context)
        command = plan["commands"]
        expected_outputs = plan["expected_outputs"]
        matrix_contexts = plan["matrix_contexts"]
        retries = int(spec.get("infrastructure_retries", 1))
    input_hash = canonical_sha256(
        {
            "protocol": protocol["hashes"]["semantic_sha256"],
            "phase": phase,
            "commands": command,
            "resolved_execution_context_sha256": resolved_context_payload[
                "context_sha256"
            ],
            "resolved_execution_context_file_sha256": (
                resolved_context_file_sha256
            ),
            "formal_training_execution_binding_sha256": execution_binding[
                "binding_full_sha256"
            ],
            "active_formal_bundle_sha256": active_bundle[
                "active_formal_bundle_sha256"
            ],
            "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                "formal_nullable_metric_aggregation_contract"
            ]["semantic_sha256"],
            "generated_checkpoint_registry_canonical_sha256": None,
        }
    )
    generated_registry_audit = None
    generated_registry_path = Path(
        context.get(
            "generated_checkpoint_registry_path",
            Path(args.output_root) / "generated_checkpoint_resource_registry.json",
        )
    )
    if capabilities.generated_checkpoint_resource_required and phase in {
        "formal_cache_policy",
        "formal_controller",
        "formal_ablation",
        "formal_support",
        "formal_scalability",
        "formal_statistics",
        "formal_gate",
        "complete_without_holdout",
    }:
        _, generated_registry_audit = load_generated_checkpoint_registry(
            generated_registry_path,
            run_root=Path(args.output_root).resolve(),
            expected_run_id=Path(args.output_root).resolve().name,
            static_registry_semantic_sha256=protocol[
                "portable_resource_identity_contract"
            ]["resource_registry_semantic_sha256"],
            protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
            protocol_full_sha256=protocol["hashes"]["full_sha256"],
            active_formal_bundle_sha256=active_bundle["active_formal_bundle_sha256"],
            execution_commit=environment_resolution.runtime_audit[
                "observed_execution_commit"
            ],
            resolved_execution_context_sha256=resolved_context_payload["context_sha256"],
            formal_training_execution_binding_sha256=execution_binding[
                "binding_full_sha256"
            ],
        )
        input_hash = canonical_sha256(
            {
                "base_phase_input_hash": input_hash,
                "generated_checkpoint_registry_canonical_sha256": (
                    generated_registry_audit["registry_canonical_sha256"]
                ),
            }
        )
    if capabilities.live_execution_allowed:
        if environment_resolution is None or resolved_context_payload is None:
            raise FormalExecutionError("protocol v1.5 context was not resolved")
        ledger_resume_phases = {
            "train",
            "dev_select",
            "formal_cache_policy",
            "formal_controller",
            "formal_ablation",
            "formal_support",
            "formal_scalability",
        }
        if (
            args.resume
            and phase in ledger_resume_phases
            and (Path(args.output_root) / "cell_ledger_identity.json").is_file()
            and not args.resume_from_cell_ledger
            and not args.finalize_phase_only
        ):
            raise FormalExecutionError(
                f"transactional {phase} resume requires --resume-from-cell-ledger"
            )
        if args.resume_from_cell_ledger and phase not in ledger_resume_phases:
            raise FormalExecutionError(
                "--resume-from-cell-ledger is only valid for a cell-transaction phase"
            )
        run_identity = canonical_sha256(
            {
                "output_root": requested_output_root,
                "protocol": protocol["hashes"]["semantic_sha256"],
                "resource_registry": protocol["portable_resource_identity_contract"][
                    "resource_registry_semantic_sha256"
                ],
                "environment": environment_resolution.environment_identity[
                    "environment_fingerprint"
                ],
                "execution_commit": environment_resolution.runtime_audit[
                    "observed_execution_commit"
                ],
                "resolved_execution_context_sha256": resolved_context_payload[
                    "context_sha256"
                ],
                "resolved_execution_context_file_sha256": (
                    resolved_context_file_sha256
                ),
                "formal_training_execution_binding_sha256": execution_binding[
                    "binding_full_sha256"
                ],
                "active_formal_bundle_sha256": active_bundle[
                    "active_formal_bundle_sha256"
                ],
            }
        )
        runner_v3 = TransactionalPhaseRunner(
            output_root=args.output_root,
            run_identity_fingerprint=run_identity,
            phase_order=PHASE_ORDER,
            resume=args.resume or args.finalize_phase_only,
            resolved_execution_context_sha256=resolved_context_payload[
                "context_sha256"
            ],
            resolved_execution_context_file_sha256=(
                resolved_context_file_sha256
            ),
        )
        if not (args.resume or args.finalize_phase_only):
            atomic_create_execution_binding(
                context["formal_training_execution_binding_path"], execution_binding
            )
            persisted = atomic_create_resolved_formal_execution_context(
                context["resolved_execution_context_path"],
                resolved_context_payload,
            )
            if persisted["file_sha256"] != resolved_context_file_sha256:
                raise FormalExecutionError(
                    "resolved execution context artifact hash drift"
                )

        cell_ledger = None
        cell_transaction_phases = {
            "train",
            "formal_cache_policy",
            "formal_controller",
            "formal_ablation",
            "formal_support",
            "formal_scalability",
        }
        if phase in cell_transaction_phases:
            cell_identity = CellExecutionIdentity(
                run_id=Path(args.output_root).resolve().name,
                execution_commit=environment_resolution.runtime_audit[
                    "observed_execution_commit"
                ],
                protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
                resource_registry_semantic_sha256=protocol[
                    "portable_resource_identity_contract"
                ]["resource_registry_semantic_sha256"],
                environment_fingerprint=environment_resolution.environment_identity[
                    "environment_fingerprint"
                ],
                split_semantic_sha256=protocol["identity"]["split_semantic_sha256"],
                window_contract_semantic_sha256=protocol["execution_contract"][
                    "window_consumption_contract"
                ]["semantic_sha256"],
                catalog_fingerprint=protocol["identity"]["catalog_fingerprint"],
                runtime_identity=canonical_sha256(
                    protocol["identity"]["typed_runtime_contract_hashes_by_capacity"]
                ),
                command_matrix_sha256=canonical_sha256(
                    {
                        "command_templates": protocol["execution_contract"][
                            "command_templates"
                        ],
                        "resolved_execution_context_sha256": (
                            resolved_context_payload["context_sha256"]
                        ),
                        "formal_training_execution_binding_sha256": execution_binding[
                            "binding_full_sha256"
                        ],
                        "active_formal_bundle_sha256": active_bundle[
                            "active_formal_bundle_sha256"
                        ],
                        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                            "formal_nullable_metric_aggregation_contract"
                        ]["semantic_sha256"],
                    }
                ),
            )
            cell_identity_path = Path(args.output_root) / "cell_ledger_identity.json"
            if args.resume_from_cell_ledger and not cell_identity_path.is_file():
                raise FormalExecutionError(
                    "--resume-from-cell-ledger requires an existing cell ledger identity"
                )
            if phase != "train" and not cell_identity_path.is_file():
                raise FormalExecutionError(
                    f"transactional {phase} requires the training cell ledger identity"
                )
            cell_ledger = FormalCellLedger(
                run_root=args.output_root,
                identity=cell_identity,
                resume=cell_identity_path.is_file(),
            )

        command_rows = command if command and isinstance(command[0], list) else ([command] if command else [])
        coordinate_by_command_hash = {
            canonical_sha256(list(argv)): dict(matrix_contexts[index])
            for index, argv in enumerate(command_rows)
        }
        if len(coordinate_by_command_hash) != len(command_rows):
            raise FormalExecutionError("transactional command matrix contains duplicate commands")

        def execute(argv):
            if (
                capabilities.cell_artifact_publication_required
                and phase in cell_transaction_phases
                and cell_ledger is not None
            ):
                original = list(argv)
                coordinates = coordinate_by_command_hash[canonical_sha256(original)]
                cell_input_hash = canonical_sha256(
                    {
                        "protocol": protocol["hashes"]["semantic_sha256"],
                        "phase": phase,
                        "coordinates": coordinates,
                        "command": original,
                        "active_formal_bundle_sha256": active_bundle[
                            "active_formal_bundle_sha256"
                        ],
                        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                            "formal_nullable_metric_aggregation_contract"
                        ]["semantic_sha256"],
                        "generated_checkpoint_registry_canonical_sha256": (
                            generated_registry_audit["registry_canonical_sha256"]
                            if generated_registry_audit is not None else None
                        ),
                    }
                )
                output_flag = (
                    "--output_root" if "--output_root" in original else "--output-root"
                )
                if output_flag not in original:
                    raise FormalExecutionError(
                        f"transactional {phase} command lacks an output-root flag"
                    )
                final_output_root = Path(original[original.index(output_flag) + 1])
                if phase == "train":
                    if "--run_id" not in original or "--agent_name" not in original:
                        raise FormalExecutionError(
                            "transactional train command lacks identity flags"
                        )
                    run_id = original[original.index("--run_id") + 1]
                    agent = original[original.index("--agent_name") + 1]
                    final_path = final_output_root / agent / run_id

                    def build(staging, _cell_id):
                        staged = list(original)
                        staged[staged.index(output_flag) + 1] = str(staging)
                        return staged

                    def resolve(staging, _cell_id, _completed):
                        return (
                            staging / agent / run_id,
                            ["train_summary.json"],
                            staging,
                        )
                else:
                    cell_id = stable_cell_id(phase, coordinates)
                    if phase == "formal_cache_policy":
                        final_path = final_output_root / str(coordinates["capacity_label"])
                    elif phase == "formal_controller":
                        final_path = final_output_root / cell_id
                    else:
                        setting_key = (
                            "ablation_setting_id"
                            if phase == "formal_ablation"
                            else "support_setting_id"
                            if phase == "formal_support"
                            else "scalability_setting_id"
                        )
                        final_path = final_output_root / str(coordinates[setting_key])

                    def build(staging, actual_cell_id):
                        staged = list(original)
                        if phase == "formal_cache_policy":
                            artifact = staging / "artifact"
                            staged[staged.index(output_flag) + 1] = str(
                                artifact / "benchmark"
                            )
                            replay_flag = "--request-replay-path"
                            if replay_flag not in staged:
                                raise FormalExecutionError(
                                    "cache-policy transaction lacks request replay path"
                                )
                            staged[staged.index(replay_flag) + 1] = str(
                                artifact / "request_replay.json"
                            )
                        elif phase == "formal_controller":
                            staged[staged.index(output_flag) + 1] = str(
                                staging / "child_output"
                            )
                        else:
                            child = staging / "child_output"
                            staged[staged.index(output_flag) + 1] = str(child)
                            staged.extend(
                                [
                                    "--cell-id", actual_cell_id,
                                    "--cell-phase", phase,
                                    "--cell-output-descriptor-path",
                                    str(child / "cell_child_output.json"),
                                ]
                            )
                        return staged

                    def resolve(staging, actual_cell_id, _completed):
                        if phase == "formal_cache_policy":
                            artifact = staging / "artifact"
                            benchmark = single_child_directory(artifact / "benchmark")
                            if not (benchmark / "aggregate_summary.json").is_file():
                                raise CellTransactionError(
                                    "cache-policy benchmark aggregate is missing"
                                )
                            return artifact, ["request_replay.json"], artifact / "benchmark"
                        if phase == "formal_controller":
                            child = staging / "child_output"
                            return (
                                single_child_directory(child),
                                ["aggregate_summary.json", "benchmark_rows.csv"],
                                child,
                            )
                        child = staging / "child_output"
                        artifact, descriptor = resolve_child_output_descriptor(
                            child / "cell_child_output.json",
                            output_root=child,
                            expected_cell_id=actual_cell_id,
                            expected_phase=phase,
                            expected_setting_id=final_path.name,
                        )
                        provenance = json.loads(
                            (artifact / "support_provenance.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if provenance.get("setting_id") != final_path.name:
                            raise CellTransactionError(
                                "support provenance setting identity mismatch"
                            )
                        return artifact, descriptor["required_payload"], child

                transaction = execute_cell_artifact_transaction(
                    cell_ledger,
                    phase=phase,
                    coordinates=coordinates,
                    command=original,
                    input_hash=cell_input_hash,
                    committed_path=final_path,
                    command_builder=build,
                    artifact_resolver=resolve,
                    environment=environment_resolution.child_environment,
                    cwd=ROOT,
                    command_failure_classification=(
                        "training_cell_failure"
                        if phase == "train" else "formal_cell_failure"
                    ),
                )
                return PhaseCommandResult(
                    int(transaction.get("return_code", 0)),
                    str(transaction.get("stdout", "")),
                    str(transaction.get("stderr", "")),
                )
            if phase == "train" and cell_ledger is not None:
                original = list(argv)
                coordinates = coordinate_by_command_hash[canonical_sha256(original)]
                cell_input_hash = canonical_sha256(
                    {
                        "protocol": protocol["hashes"]["semantic_sha256"],
                        "phase": phase,
                        "coordinates": coordinates,
                        "command": original,
                        "active_formal_bundle_sha256": active_bundle[
                            "active_formal_bundle_sha256"
                        ],
                        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                            "formal_nullable_metric_aggregation_contract"
                        ]["semantic_sha256"],
                        "generated_checkpoint_registry_canonical_sha256": (
                            generated_registry_audit["registry_canonical_sha256"]
                            if generated_registry_audit is not None else None
                        ),
                    }
                )
                output_flag = "--output_root"
                if output_flag not in original or "--run_id" not in original or "--agent_name" not in original:
                    raise FormalExecutionError("transactional train command lacks output/identity flags")
                final_output_root = Path(original[original.index(output_flag) + 1])
                run_id = original[original.index("--run_id") + 1]
                agent = original[original.index("--agent_name") + 1]
                final_path = final_output_root / agent / run_id
                begun = cell_ledger.begin_cell(
                    phase=phase,
                    coordinates=coordinates,
                    command=original,
                    input_hash=cell_input_hash,
                    committed_path=final_path,
                )
                if begun["status"] == "skipped_committed":
                    return PhaseCommandResult(0, "committed cell hash-verified and skipped", "")
                staging = Path(begun["record"]["staging_path"])
                staged = list(original)
                staged[staged.index(output_flag) + 1] = str(staging)
                started_ns = time.monotonic_ns()
                completed = subprocess.run(
                    staged,
                    cwd=ROOT,
                    env=environment_resolution.child_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                artifact_root = staging / agent / run_id
                artifact_root.mkdir(parents=True, exist_ok=True)
                (artifact_root / "cell_stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (artifact_root / "cell_stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                if completed.returncode != 0:
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=completed.returncode,
                        classification=(
                            "infrastructure_retryable"
                            if completed.returncode == 75
                            else "training_cell_failure"
                        ),
                        retryable=completed.returncode == 75,
                    )
                    return PhaseCommandResult(
                        completed.returncode, completed.stdout, completed.stderr
                    )
                cell_ledger.commit_cell(
                    begun["cell_id"],
                    required_paths=[
                        "train_summary.json", "cell_stdout.log", "cell_stderr.log"
                    ],
                    monotonic_started_ns=started_ns,
                    artifact_subpath=Path(agent) / run_id,
                    child_output_path=staging,
                )
                return PhaseCommandResult(0, completed.stdout, completed.stderr)
            if phase in cell_transaction_phases and cell_ledger is not None:
                original = list(argv)
                coordinates = coordinate_by_command_hash[canonical_sha256(original)]
                cell_input_hash = canonical_sha256(
                    {
                        "protocol": protocol["hashes"]["semantic_sha256"],
                        "phase": phase,
                        "coordinates": coordinates,
                        "command": original,
                        "active_formal_bundle_sha256": active_bundle[
                            "active_formal_bundle_sha256"
                        ],
                        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                            "formal_nullable_metric_aggregation_contract"
                        ]["semantic_sha256"],
                        "generated_checkpoint_registry_canonical_sha256": (
                            generated_registry_audit["registry_canonical_sha256"]
                            if generated_registry_audit is not None else None
                        ),
                    }
                )
                output_flag = "--output_root" if "--output_root" in original else "--output-root"
                if output_flag not in original:
                    raise FormalExecutionError(
                        f"transactional {phase} command lacks an output-root flag"
                    )
                final_output_root = Path(original[original.index(output_flag) + 1])
                cell_id = stable_cell_id(phase, coordinates)
                if phase == "formal_cache_policy":
                    capacity = str(coordinates["capacity_label"])
                    final_path = final_output_root / capacity
                elif phase == "formal_controller":
                    final_path = final_output_root / cell_id
                else:
                    setting_key = (
                        "ablation_setting_id"
                        if phase == "formal_ablation"
                        else "support_setting_id"
                        if phase == "formal_support"
                        else "scalability_setting_id"
                    )
                    final_path = final_output_root / str(coordinates[setting_key])
                begun = cell_ledger.begin_cell(
                    phase=phase,
                    coordinates=coordinates,
                    command=original,
                    input_hash=cell_input_hash,
                    committed_path=final_path,
                )
                if begun["status"] == "skipped_committed":
                    return PhaseCommandResult(
                        0, "committed formal cell hash-verified and skipped", ""
                    )
                staging = Path(begun["record"]["staging_path"])
                staged = list(original)
                if phase == "formal_cache_policy":
                    artifact_root = staging / "artifact"
                    staged[staged.index(output_flag) + 1] = str(
                        artifact_root / "benchmark"
                    )
                    replay_flag = "--request-replay-path"
                    if replay_flag not in staged:
                        raise FormalExecutionError(
                            "cache-policy transaction lacks request replay path"
                        )
                    staged[staged.index(replay_flag) + 1] = str(
                        artifact_root / "request_replay.json"
                    )
                    artifact_subpath = Path("artifact")
                elif phase == "formal_controller":
                    child_output_root = staging / "child_output"
                    staged[staged.index(output_flag) + 1] = str(child_output_root)
                    artifact_root = None
                    artifact_subpath = Path(".")
                else:
                    child_output_root = staging / "child_output"
                    descriptor_path = child_output_root / "cell_child_output.json"
                    staged[staged.index(output_flag) + 1] = str(child_output_root)
                    staged.extend(
                        [
                            "--cell-id", cell_id,
                            "--cell-phase", phase,
                            "--cell-output-descriptor-path", str(descriptor_path),
                        ]
                    )
                    artifact_root = None
                    artifact_subpath = Path(".")
                started_ns = time.monotonic_ns()
                completed = subprocess.run(
                    staged,
                    cwd=ROOT,
                    env=environment_resolution.child_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode != 0:
                    (staging / "cell_stdout.log").write_text(
                        completed.stdout, encoding="utf-8"
                    )
                    (staging / "cell_stderr.log").write_text(
                        completed.stderr, encoding="utf-8"
                    )
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=completed.returncode,
                        classification=(
                            "infrastructure_retryable"
                            if completed.returncode == 75
                            else "formal_cell_failure"
                        ),
                        retryable=completed.returncode == 75,
                    )
                    return PhaseCommandResult(
                        completed.returncode, completed.stdout, completed.stderr
                    )
                try:
                    if phase == "formal_cache_policy":
                        if not (artifact_root / "request_replay.json").is_file():
                            raise CellTransactionError("cache-policy replay was not staged")
                        benchmark_root = single_child_directory(
                            artifact_root / "benchmark"
                        )
                        if not (benchmark_root / "aggregate_summary.json").is_file():
                            raise CellTransactionError(
                                "cache-policy benchmark aggregate is missing"
                            )
                        required_paths = ["request_replay.json"]
                        child_output_path = artifact_root / "benchmark"
                    elif phase == "formal_controller":
                        artifact_root = single_child_directory(child_output_root)
                        required_paths = ["aggregate_summary.json", "benchmark_rows.csv"]
                        child_output_path = child_output_root
                    else:
                        artifact_root, descriptor = resolve_child_output_descriptor(
                            descriptor_path,
                            output_root=child_output_root,
                            expected_cell_id=cell_id,
                            expected_phase=phase,
                            expected_setting_id=final_path.name,
                        )
                        provenance = json.loads(
                            (artifact_root / "support_provenance.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if provenance.get("setting_id") != final_path.name:
                            raise CellTransactionError(
                                "support provenance setting identity mismatch"
                            )
                        required_paths = list(descriptor["required_payload"])
                        child_output_path = child_output_root
                    artifact_root.mkdir(parents=True, exist_ok=True)
                    (artifact_root / "cell_stdout.log").write_text(
                        completed.stdout, encoding="utf-8"
                    )
                    (artifact_root / "cell_stderr.log").write_text(
                        completed.stderr, encoding="utf-8"
                    )
                    cell_ledger.commit_cell(
                        begun["cell_id"],
                        required_paths=[
                            *required_paths, "cell_stdout.log", "cell_stderr.log"
                        ],
                        monotonic_started_ns=started_ns,
                        artifact_subpath=artifact_subpath,
                        validated_artifact_root=artifact_root,
                        child_output_path=child_output_path,
                    )
                except Exception:
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=0,
                        classification="cell_artifact_publication_validation_failure",
                        retryable=False,
                    )
                    raise
                return PhaseCommandResult(0, completed.stdout, completed.stderr)
            completed = subprocess.run(
                list(argv),
                cwd=ROOT,
                env=environment_resolution.child_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            return PhaseCommandResult(
                completed.returncode, completed.stdout, completed.stderr
            )

        if args.finalize_phase_only:
            result = runner_v3.finalize_phase_only(
                phase,
                commands=command_rows,
                input_hash=input_hash,
                expected_outputs=expected_outputs,
            )
        else:
            result = runner_v3.run_phase(
                phase,
                commands=command_rows,
                input_hash=input_hash,
                expected_outputs=expected_outputs,
                executor=execute,
                infrastructure_retries=retries,
            )
        if cell_ledger is not None:
            cell_ledger.assert_complete_matrix(
                phase=phase,
                expected_cell_ids=[
                    stable_cell_id(phase, context_row)
                    for context_row in matrix_contexts
                ],
            )
        if (
            phase == "checkpoint_freeze"
            and capabilities.generated_checkpoint_resource_required
            and result.get("status") in {
                "completed",
                "skipped_completed",
                "skipped_completed_hash_match",
                "already_finalized",
            }
        ):
            static_registry = load_registry(context["resource_registry_path"])
            registry = build_generated_checkpoint_registry(
                run_root=args.output_root,
                protocol=protocol,
                static_registry=static_registry,
                resolved_execution_context=resolved_context_payload,
                execution_binding=execution_binding,
            )
            publication = publish_or_validate_generated_checkpoint_registry(
                generated_registry_path,
                registry,
                run_root=Path(args.output_root).resolve(),
                expected_run_id=Path(args.output_root).resolve().name,
                static_registry_semantic_sha256=static_registry["hashes"][
                    "semantic_sha256"
                ],
                protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
                protocol_full_sha256=protocol["hashes"]["full_sha256"],
                active_formal_bundle_sha256=active_bundle[
                    "active_formal_bundle_sha256"
                ],
                execution_commit=environment_resolution.runtime_audit[
                    "observed_execution_commit"
                ],
                resolved_execution_context_sha256=resolved_context_payload[
                    "context_sha256"
                ],
                formal_training_execution_binding_sha256=execution_binding[
                    "binding_full_sha256"
                ],
            )
            result = {**result, "generated_checkpoint_registry": publication}
    else:
        if args.finalize_phase_only or args.resume_from_cell_ledger:
            raise FormalExecutionError(
                "transactional resume/finalize is unavailable for legacy protocols"
            )
        runner = AppendOnlyPhaseRunner(
            protocol=protocol,
            output_root=args.output_root,
            resume=args.resume,
        )
        result = runner.run_phase(
            phase,
            command=command,
            input_hash=input_hash,
            expected_outputs=expected_outputs,
            infrastructure_retries=retries,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
