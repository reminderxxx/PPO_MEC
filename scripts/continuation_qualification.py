"""Read-only original-run qualification for the external continuation executor.

No writer is instantiated here. Source, bundle, environment, context and generated
resource checks are the original implementations. This is not an approval issuer.
"""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def external(name):
    path=ROOT/'scripts'/(name+'.py')
    spec=importlib.util.spec_from_file_location('external_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def qualify(proposal_path, contract):
    security=external('continuation_executor_security')
    adapter=external('continuation_legacy_adapter')
    proposal=security.strict_json(proposal_path)
    if security.file_hash(proposal_path)!=contract['proposal_file_sha256'] or security.digest(proposal)!=contract['proposal_sha256']:
        raise ValueError('A proposal identity drift')
    spec=importlib.util.spec_from_file_location('external_original_proposal_validator',
        ROOT/'src/runtime/fixed_commit_continuation.py')
    original_proposal=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_proposal)
    original_proposal.validate_structure(proposal)
    for original,new in [('run_root','run_root'),('run_id','run_id'),
                         ('worktree_root','scientific_worktree'),('execution_commit','scientific_commit')]:
        if proposal[original]!=contract[new]:raise ValueError('proposal/execution contract drift: '+new)
    root=Path(proposal['worktree_root']);run=Path(proposal['run_root'])
    if security.canonical(contract['ledger_anchors'])!=security.canonical(proposal['ledgers']):
        raise ValueError('approved ledger anchors differ from A proposal')
    protected=[row for row in proposal['evidence'] if row['role']=='committed_output']
    if len(protected)!=1 or protected[0]['sha256']!=contract['immutable_payload_inventory_sha256']:
        raise ValueError('immutable inventory does not match original proposal')
    if Path.cwd()!=root:
        raise ValueError('qualification cwd differs from frozen scientific worktree')
    if os.environ.get('PYTHONPATH')!=str(root):
        raise ValueError('qualification PYTHONPATH differs from frozen scientific worktree')
    context_path=run/'resolved_execution_context.json'
    context_rows=[row for row in proposal['evidence'] if row['role']=='context' and row['path']==str(context_path)]
    if len(context_rows)!=1 or security.file_hash(context_path)!=context_rows[0]['sha256']:
        raise ValueError('original context evidence drift')
    context=security.strict_json(run/'resolved_execution_context.json')
    runtime=context['runtime_location']
    if os.path.abspath(sys.executable)!=os.path.abspath(runtime['resolved_python_absolute_path']):
        raise ValueError('qualification interpreter differs from frozen scientific context')
    if not sys.dont_write_bytecode or os.environ.get('PYTHONNOUSERSITE')!='1':
        raise ValueError('qualification requires bytecode/user-site isolation')
    # Original evidence files remain immutable even when successor ledgers grow.
    for row in proposal['evidence']:
        path=Path(row['path'])
        security.contained(path,path,must_exist=True)
        if path.stat().st_size!=row['size_bytes'] or security.file_hash(path)!=row['sha256']:
            raise ValueError('original evidence drift: '+str(path))
    original_runner=root/'scripts/run_typed_model_cache_formal_protocol.py'
    runner,source=adapter.load_scientific_modules(root,security.file_hash(original_runner),
        expected_commit=proposal['execution_commit'])
    from src.runtime.active_formal_bundle import validate_active_formal_bundle
    from src.runtime.formal_execution_environment import (source_tree_fingerprint,
        resolve_execution_environment,protocol_bound_extensions_from_protocol)
    from src.runtime.resolved_formal_execution_context import load_resolved_formal_execution_context
    from src.runtime.formal_training_identity import validate_execution_binding
    from src.runtime.formal_invalid_run_registry import reject_permanently_invalid_formal_references
    from src.runtime.generated_checkpoint_resources import load_generated_checkpoint_registry
    from src.evaluators.formal_cell_transaction import validate_cell_ledger,artifact_inventory
    from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
    protocol=security.strict_json(runtime['protocol_path'])
    binding=security.strict_json(run/'formal_training_execution_binding.json')
    reject_permanently_invalid_formal_references([str(run),str(root),proposal['run_id'],context,binding])
    if source_tree_fingerprint(root)!=proposal['source_tree']['tracked_sources_sha256']:
        raise ValueError('original tracked source fingerprint drift')
    # Only the moving-main equality is disabled here. Independent approval remains
    # an outer admission requirement; this read-only function cannot grant it.
    bundle=validate_active_formal_bundle(repository_root=root,require_origin_main_match=False)
    environment=resolve_execution_environment(clean_worktree_root=root,
        execution_commit=proposal['execution_commit'],python_executable=runtime['resolved_python_absolute_path'],
        environment_manifest=security.strict_json(runtime['execution_environment_manifest_path']),
        expected_identity=context['scientific_identity']['full_normalized_environment_projection'],
        protocol_bound_extensions=protocol_bound_extensions_from_protocol(protocol),
        forbidden_source_roots=[str(ROOT)],require_clean_git_worktree=True)
    load_resolved_formal_execution_context(run/'resolved_execution_context.json',protocol=protocol,
        clean_worktree_root=root,durable_run_root=run,environment_identity=environment.environment_identity,
        runtime_audit=environment.runtime_audit,check_git=True)
    matrix=runner.validate_command_templates(protocol['execution_contract']['command_templates'],
        context['resolved_expansion_context'])
    if matrix['command_matrix_sha256']!=context['command_expansion']['resolved_command_matrix_sha256']:
        raise ValueError('frozen command matrix drift')
    validate_execution_binding(binding,protocol=protocol,
        scientific_config=security.strict_json(context['resolved_expansion_context']['agent_scientific_config_path']),
        execution_commit=proposal['execution_commit'],environment_identity=environment.environment_identity,
        command_matrix_sha256=matrix['command_matrix_sha256'],
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'])
    registry,audit=load_generated_checkpoint_registry(run/'generated_checkpoint_resource_registry.json',
        run_root=run,expected_run_id=proposal['run_id'],
        static_registry_semantic_sha256=protocol['portable_resource_identity_contract']['resource_registry_semantic_sha256'],
        protocol_semantic_sha256=protocol['hashes']['semantic_sha256'],protocol_full_sha256=protocol['hashes']['full_sha256'],
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'],execution_commit=proposal['execution_commit'],
        resolved_execution_context_sha256=context['context_sha256'],
        formal_training_execution_binding_sha256=binding['binding_full_sha256'])
    expected_cells={phase:{runner.stable_cell_id(phase,row) for row in
        protocol['execution_contract']['command_templates'][phase]['matrix_contexts']} for phase in adapter.PHASES[:5]}
    reconciliation=external('continuation_ledger_validation').validate_successors(contract['ledger_anchors'],
        root=run,phase_validator=validate_phase_ledger_v3,cell_validator=validate_cell_ledger,
        artifact_inventory=artifact_inventory,expected_cells_by_phase=expected_cells)
    preserved=external('continuation_ledger_validation').validate_preserved_inventory(
        protected[0]['path'],anchors=contract['ledger_anchors'],scientific_root=root)
    from scripts.benchmark_main_results import validate_benchmark_checkpoint_gate
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime
    runtime_paths={row['capacity_label']:root/row['runtime_config_path'] for row in
        protocol['execution_contract']['command_templates']['train']['matrix_contexts']}
    gates=[]
    frozen=security.strict_json(run/'checkpoint_freeze.json')
    for row in frozen['frozen_checkpoints']:
        capacity,agent,seed=row['capacity_label'],row['agent_name'],row['seed']
        companion=security.strict_json(run/'checkpoint_manifests'/capacity/'checkpoint_provenance_manifest.json')
        gate=validate_benchmark_checkpoint_gate(row['checkpoint_path'],expected_agent_name=agent,expected_seed=seed,
            expected_runtime_contract=resolve_model_cache_runtime(runtime_paths[capacity],root=root),
            expected_reward_positive_offset=0.0,provenance_envelope=companion[agent][str(seed)],
            protocol=protocol,resolved_execution_context=context,execution_binding=binding,expected_capacity_label=capacity)
        if gate['status']!='compatible':raise ValueError('frozen checkpoint gate rejected')
        gates.append({'agent':agent,'seed':seed,'capacity':capacity,'sha256':row['checkpoint_sha256']})
    plans={phase:runner.expand_command_plan(protocol['execution_contract']['command_templates'][phase],
        context['resolved_expansion_context']) for phase in adapter.PHASES[:-1]}
    plans['complete_without_holdout']={'commands':[],'expected_outputs':[],'matrix_contexts':[],
        'precondition':'original validate_complete_without_holdout_gate before any writer'}
    return {'proposal':proposal,'runner':runner,'source':source,'protocol':protocol,'context':context,'binding':binding,
        'bundle':bundle,'environment':environment,'registry_audit':audit,'plans':plans,
        'report':{'scope':'read-only qualification; no execution approval','checkpoint_gates':gates,
            'reconciliation':reconciliation,'preserved_inventory':preserved,'command_matrix':matrix,'source_modules':adapter.loaded_origins(root),
            'environment_audit':environment.runtime_audit,'generated_registry_audit':audit,
            'moving_origin_main_rule':'independent continuation admission required; original public runner unchanged'}}


def reconcile(qualified, contract):
    """Repeat current successor/payload validation while holding the writer lock."""
    from src.evaluators.formal_cell_transaction import validate_cell_ledger,artifact_inventory
    from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
    refresh(qualified,contract)
    runner=qualified['runner'];protocol=qualified['protocol']
    phases=external('continuation_legacy_adapter').PHASES
    expected={phase:{runner.stable_cell_id(phase,row) for row in
        protocol['execution_contract']['command_templates'][phase]['matrix_contexts']} for phase in phases[:5]}
    return external('continuation_ledger_validation').validate_successors(contract['ledger_anchors'],
        root=Path(contract['run_root']),phase_validator=validate_phase_ledger_v3,cell_validator=validate_cell_ledger,
        artifact_inventory=artifact_inventory,expected_cells_by_phase=expected)


def refresh(qualified, contract):
    """Recheck immutable identities and registry immediately before dispatch."""
    sec=external('continuation_executor_security');proposal=qualified['proposal']
    root=Path(contract['scientific_worktree']);run=Path(contract['run_root'])
    if sec.git(root,'rev-parse','HEAD')!=contract['scientific_commit']:
        raise ValueError('scientific commit changed after qualification')
    if sec.git(root,'status','--porcelain','--untracked-files=all'):
        raise ValueError('scientific worktree changed after qualification')
    from src.runtime.formal_execution_environment import source_tree_fingerprint
    if source_tree_fingerprint(root)!=proposal['source_tree']['tracked_sources_sha256']:
        raise ValueError('scientific source changed after qualification')
    for row in proposal['evidence']:
        path=Path(row['path']);sec.contained(path,path,must_exist=True)
        if path.stat().st_size!=row['size_bytes'] or sec.file_hash(path)!=row['sha256']:
            raise ValueError('original evidence changed after qualification: '+str(path))
    from src.runtime.active_formal_bundle import validate_active_formal_bundle
    bundle=validate_active_formal_bundle(repository_root=root,require_origin_main_match=False)
    if bundle['active_formal_bundle_sha256']!=qualified['bundle']['active_formal_bundle_sha256']:
        raise ValueError('active bundle changed after qualification')
    from src.runtime.generated_checkpoint_resources import load_generated_checkpoint_registry
    protocol=qualified['protocol'];context=qualified['context'];binding=qualified['binding']
    _,audit=load_generated_checkpoint_registry(run/'generated_checkpoint_resource_registry.json',
        run_root=run,expected_run_id=contract['run_id'],
        static_registry_semantic_sha256=protocol['portable_resource_identity_contract']['resource_registry_semantic_sha256'],
        protocol_semantic_sha256=protocol['hashes']['semantic_sha256'],protocol_full_sha256=protocol['hashes']['full_sha256'],
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'],execution_commit=contract['scientific_commit'],
        resolved_execution_context_sha256=context['context_sha256'],
        formal_training_execution_binding_sha256=binding['binding_full_sha256'])
    if audit['registry_canonical_sha256']!=qualified['registry_audit']['registry_canonical_sha256']:
        raise ValueError('generated registry changed after qualification')
    return {'status':'pass','evidence_files':len(proposal['evidence']),
            'registry_canonical_sha256':audit['registry_canonical_sha256']}
