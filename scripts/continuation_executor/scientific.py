"""Read-only qualification against the exact old checkout, never public main."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from . import PHASES
from .identity import ContinuationError, absolute_path, file_hash, git, read_json, verify_file, digest
from .planning import cell_identity_fields, phase_plan, run_identity
from .reconciliation import reconcile


def loaded_origins(root):
    result = {}
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith(("src.", "scripts.")):
            origin = getattr(module, "__file__", None)
            if origin:
                path = Path(origin).resolve()
                if root not in path.parents:
                    raise ContinuationError("shadow scientific module: " + name + "=" + str(path))
                result[name] = {"path": str(path), "sha256": file_hash(path)}
    return result


def load_native(root, commit):
    root = absolute_path(root)
    if git(root, "rev-parse", "HEAD") != commit:
        raise ContinuationError("scientific fixed commit drift")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ContinuationError("scientific checkout must be clean")
    if Path.cwd() != root or os.environ.get("PYTHONPATH") != str(root):
        raise ContinuationError("scientific cwd/PYTHONPATH drift")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.dont_write_bytecode:
        raise ContinuationError("scientific loader requires no user-site and no bytecode writes")
    loaded_origins(root)
    for entry in sys.path:
        path = Path(entry or os.getcwd()).absolute()
        if (path / "src").is_dir() and path != root:
            raise ContinuationError("external/current-main src on sys.path")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from .process_environment import verify_process_environment
    process_environment=verify_process_environment(root)
    modules = {
        "cell": "src.evaluators.formal_cell_transaction",
        "phase": "src.evaluators.formal_phase_transaction",
        "execution": "src.evaluators.typed_model_cache_formal_execution",
        "bundle": "src.runtime.active_formal_bundle",
        "environment": "src.runtime.formal_execution_environment",
        "context": "src.runtime.resolved_formal_execution_context",
        "binding": "src.runtime.formal_training_identity",
        "registry": "src.runtime.generated_checkpoint_resources",
        "denylist": "src.runtime.formal_invalid_run_registry",
        "capability": "src.runtime.formal_protocol_capabilities",
        "public_checks": "scripts.run_typed_model_cache_formal_protocol",
    }
    result = {key: importlib.import_module(name) for key, name in modules.items()}
    loaded_origins(root)
    result["process_environment"]=process_environment
    return result


class QualifiedRun:
    """No constructor creates a run, context, registry, lock, staging or record."""
    def __init__(self, proposal, contract):
        self.proposal, self.contract = proposal, contract
        self.root = absolute_path(proposal["worktree_root"])
        self.run_root = absolute_path(proposal["run_root"])
        self.native = load_native(str(self.root), proposal["execution_commit"])
        n = self.native
        if git(self.root, "rev-parse", "HEAD^{tree}") != proposal["source_tree"]["git_tree"]:
            raise ContinuationError("scientific tree identity drift")
        if n["environment"].source_tree_fingerprint(self.root) != proposal["source_tree"]["tracked_sources_sha256"]:
            raise ContinuationError("scientific tracked source content drift")
        n["denylist"].reject_permanently_invalid_formal_references(
            [str(self.root), str(self.run_root), proposal["run_id"], proposal["evidence"]])
        for row in proposal["evidence"]:
            # A evidence consists of immutable source/inputs/committed artifacts.
            verify_file({key: row[key] for key in ("path", "sha256", "size_bytes")})
        self.context_path = self.run_root / "resolved_execution_context.json"
        self.context = read_json(self.context_path)
        runtime = self.context["runtime_location"]
        if os.path.abspath(sys.executable) != os.path.abspath(runtime["resolved_python_absolute_path"]):
            raise ContinuationError("actual interpreter differs from frozen context")
        self.protocol = read_json(runtime["protocol_path"])
        self.binding = read_json(self.run_root / "formal_training_execution_binding.json")
        n["execution"].validate_protocol_v1_1(self.protocol)
        n["capability"].require_live_execution_protocol(self.protocol["typed_model_cache_formal_protocol_version"])
        n["public_checks"].reject_invalid_run_root(self.protocol, self.run_root)
        # The only replaced rule. This read-only qualification is NOT approval.
        self.bundle = n["bundle"].validate_active_formal_bundle(
            repository_root=self.root, require_origin_main_match=False)
        resource_audit = n["bundle"].build_active_bundle_resource_resolution_audit(self.bundle)
        self.environment = n["environment"].resolve_execution_environment(
            clean_worktree_root=self.root, execution_commit=proposal["execution_commit"],
            python_executable=runtime["resolved_python_absolute_path"],
            environment_manifest=read_json(runtime["execution_environment_manifest_path"]),
            expected_identity=self.context["scientific_identity"]["full_normalized_environment_projection"],
            protocol_bound_extensions=n["environment"].protocol_bound_extensions_from_protocol(self.protocol),
            forbidden_source_roots=[str(Path(__file__).resolve().parents[2])],
            require_clean_git_worktree=True)
        _, report = n["context"].load_resolved_formal_execution_context(
            self.context_path, protocol=self.protocol, clean_worktree_root=self.root,
            durable_run_root=self.run_root, environment_identity=self.environment.environment_identity,
            runtime_audit=self.environment.runtime_audit, check_git=True)
        self.context_file_sha256 = report["file_sha256"]
        expansion = n["public_checks"].resolved_expansion_context(
            self.protocol, protocol_path=runtime["protocol_path"], output_root=str(self.run_root),
            python_executable=self.environment.python_executable,
            active_formal_bundle_sha256=self.bundle["active_formal_bundle_sha256"],
            active_protocol_index_path=self.context["resolved_expansion_context"]["active_protocol_index_path"],
            active_bundle_resource_resolution_audit_sha256=resource_audit["audit_sha256"])
        if expansion != self.context["resolved_expansion_context"]:
            raise ContinuationError("frozen outer expansion differs from original resolver")
        self.matrix = n["execution"].validate_command_templates(
            self.protocol["execution_contract"]["command_templates"], expansion)
        n["binding"].validate_execution_binding(
            self.binding, protocol=self.protocol, scientific_config=read_json(expansion["agent_scientific_config_path"]),
            execution_commit=proposal["execution_commit"], environment_identity=self.environment.environment_identity,
            command_matrix_sha256=self.matrix["command_matrix_sha256"],
            active_formal_bundle_sha256=self.bundle["active_formal_bundle_sha256"])
        _, self.registry_audit = n["registry"].load_generated_checkpoint_registry(
            expansion["generated_checkpoint_registry_path"], run_root=self.run_root, expected_run_id=proposal["run_id"],
            static_registry_semantic_sha256=self.protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"],
            protocol_semantic_sha256=self.protocol["hashes"]["semantic_sha256"],
            protocol_full_sha256=self.protocol["hashes"]["full_sha256"],
            active_formal_bundle_sha256=self.bundle["active_formal_bundle_sha256"], execution_commit=proposal["execution_commit"],
            resolved_execution_context_sha256=self.context["context_sha256"],
            formal_training_execution_binding_sha256=self.binding["binding_full_sha256"])
        self.registry_sha256 = self.registry_audit["registry_canonical_sha256"]
        self.plans = {phase: phase_plan(phase, self.protocol, self.context, self.binding,
                     self.context_file_sha256, self.registry_sha256, n["execution"].expand_command_plan) for phase in PHASES}
        if contract["command_plan_sha256"] != digest(self.plans):
            raise ContinuationError("approved command plan drift")
        fields = cell_identity_fields(self.protocol, self.context, self.binding, proposal["run_id"],
                                      proposal["execution_commit"], self.environment.environment_identity["environment_fingerprint"])
        self.cell_ledger = n["cell"].FormalCellLedger(run_root=self.run_root, identity=n["cell"].CellExecutionIdentity(**fields), resume=True)
        self.run_identity = run_identity(self.protocol, self.context, self.binding, self.context_file_sha256,
                         self.run_root, proposal["execution_commit"], self.environment.environment_identity["environment_fingerprint"])
        self.phase_runner = n["phase"].TransactionalPhaseRunner(
            output_root=self.run_root, run_identity_fingerprint=self.run_identity,
            phase_order=n["execution"].PHASE_ORDER, resume=True,
            resolved_execution_context_sha256=self.context["context_sha256"],
            resolved_execution_context_file_sha256=self.context_file_sha256)
        self.reconciliation = reconcile(contract, n["phase"], self.cell_ledger, self.plans, protocol=self.protocol, context=self.context, registry_sha256=self.registry_sha256)
        from .checkpoints import check_frozen_checkpoints
        self.checkpoints = check_frozen_checkpoints(self.root,self.run_root,self.protocol,self.context,self.binding)
        self.origins = loaded_origins(self.root)
