"""Private read-only scientific validator adapter; invoked by the preflight CLI.

Never import or invoke the public runner. No write/dispatch callable is exposed.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys


def validate_loaded_origins(root):
    origins = {}
    for name, mod in list(sys.modules.items()):
        if name == 'src' or name.startswith(('src.', 'scripts.')):
            origin = getattr(mod, '__file__', None)
            if origin:
                path = Path(origin).resolve()
                if root not in path.parents:
                    raise ValueError('shadow/current main module: ' + name + '=' + str(path))
                origins[name] = str(path)
    return origins


def probe(proposal, implementation_root):
    root = Path(proposal['worktree_root'])
    if Path.cwd() != root or os.environ.get('PYTHONPATH') != str(root):
        raise ValueError('probe cwd/PYTHONPATH differs from pinned source')
    if not sys.dont_write_bytecode or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise ValueError('probe requires bytecode disabled and user-site disabled')
    if os.path.abspath(sys.executable) != os.path.abspath(json.loads(
            (Path(proposal['run_root']) / 'resolved_execution_context.json').read_text()
            )['runtime_location']['resolved_python_absolute_path']):
        raise ValueError('probe interpreter differs from original context')
    validate_loaded_origins(root)
    sys.path.insert(0, str(root))
    dispatch_counts = dict(train=0, dev=0, formal=0, holdout=0)
    observed_calls = {'scientific_function_calls': 0}
    def deny_execution(frame, event, arg):
        if event != 'call':
            return
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        if module.startswith(('src.', 'scripts.')):
            observed_calls['scientific_function_calls'] += 1
        forbidden = (module.startswith('src.envs.') and name in {'step', 'reset'}) or (
            module.startswith(('src.', 'scripts.')) and ('rollout' in name or name in {'train', 'learn'})) or (
            module.startswith('scripts.') and name == 'main')
        if forbidden:
            category = 'train' if 'train' in module or name in {'train', 'learn'} else 'formal'
            dispatch_counts[category] += 1
            raise RuntimeError('read-only probe blocked scientific execution: ' + module + '.' + name)
    sys.setprofile(deny_execution)
    from src.runtime.active_formal_bundle import validate_active_formal_bundle
    from src.runtime.formal_execution_environment import (
        source_tree_fingerprint, resolve_execution_environment,
        protocol_bound_extensions_from_protocol)
    from src.runtime.resolved_formal_execution_context import load_resolved_formal_execution_context
    from src.runtime.formal_training_identity import validate_execution_binding
    from src.runtime.generated_checkpoint_resources import load_generated_checkpoint_registry
    from src.runtime.formal_invalid_run_registry import reject_permanently_invalid_formal_references
    from src.evaluators.typed_model_cache_formal_execution import validate_command_templates
    from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
    from src.evaluators.formal_cell_transaction import validate_cell_ledger
    validate_loaded_origins(root)
    results = []
    def attempt(name, fn):
        try:
            result = fn()
            results.append({'name': name, 'status': 'pass', 'detail': result})
            return result
        except Exception as exc:
            results.append({'name': name, 'status': 'fail', 'detail': str(exc)})
            return None
    def read(path):
        return json.loads(Path(path).read_text())
    run = Path(proposal['run_root'])
    ctx = read(run / 'resolved_execution_context.json')
    binding = read(run / 'formal_training_execution_binding.json')
    runtime = ctx['runtime_location']
    original = attempt('original_publication_gate_current', lambda: validate_active_formal_bundle(
        repository_root=root))
    if original:
        results[-1]['detail'] = {'status': 'pass', 'meaning': 'current gate only; not launch authorization'}
    # This relaxation is ONLY read-only candidate analysis. The original gate is
    # independently reported above. It cannot authorize any public runner call.
    bundle = attempt('candidate_bundle_read_only', lambda: validate_active_formal_bundle(
        repository_root=root, require_origin_main_match=False))
    if bundle:
        results[-1]['detail'] = {'status': 'pass', 'resource_count': len(bundle['resource_ids']),
                                 'substitute_rule': 'fixed source + original launch/release evidence + independent approval (pending)'}
    protocol = read(runtime['protocol_path'])
    attempt('denylist', lambda: reject_permanently_invalid_formal_references(
        [str(run), str(root), proposal['run_id']]))
    def tree():
        value = source_tree_fingerprint(root)
        if value != proposal['source_tree']['tracked_sources_sha256']:
            raise ValueError('tracked source content drift')
        return value
    attempt('tracked_source_contents', tree)
    env = attempt('actual_python_environment', lambda: resolve_execution_environment(
        clean_worktree_root=root, execution_commit=proposal['execution_commit'],
        python_executable=runtime['resolved_python_absolute_path'],
        environment_manifest=read(runtime['execution_environment_manifest_path']),
        expected_identity=ctx['scientific_identity']['full_normalized_environment_projection'],
        protocol_bound_extensions=protocol_bound_extensions_from_protocol(protocol),
        forbidden_source_roots=[implementation_root], require_clean_git_worktree=True))
    if env:
        results[-1]['detail'] = env.runtime_audit
    attempt('context', lambda: load_resolved_formal_execution_context(
        run / 'resolved_execution_context.json', protocol=protocol,
        clean_worktree_root=root, durable_run_root=run,
        environment_identity=env.environment_identity if env else None,
        runtime_audit=env.runtime_audit if env else None, check_git=True)[1])
    matrix = attempt('original_command_matrix', lambda: validate_command_templates(
        protocol['execution_contract']['command_templates'], ctx['resolved_expansion_context']))
    if matrix:
        if matrix['command_matrix_sha256'] != ctx['command_expansion']['resolved_command_matrix_sha256']:
            results.append({'name': 'matrix_binding', 'status': 'fail', 'detail': 'command matrix drift'})
    attempt('binding', lambda: validate_execution_binding(
        binding, protocol=protocol,
        scientific_config=read(ctx['resolved_expansion_context']['agent_scientific_config_path']),
        execution_commit=proposal['execution_commit'],
        environment_identity=env.environment_identity if env else binding['environment_identity'],
        command_matrix_sha256=matrix['command_matrix_sha256'] if matrix else '',
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'] if bundle else ''))
    for kind, validator in [('phase', validate_phase_ledger_v3), ('cell', validate_cell_ledger)]:
        attempt('native_' + kind + '_ledger', lambda kind=kind, validator=validator: validator(
            [json.loads(line) for line in (run / (kind + '_state.jsonl')).read_text().splitlines()]))
    attempt('registry_freeze_terminal', lambda: load_generated_checkpoint_registry(
        run / 'generated_checkpoint_resource_registry.json', run_root=run,
        expected_run_id=proposal['run_id'],
        static_registry_semantic_sha256=protocol['portable_resource_identity_contract']['resource_registry_semantic_sha256'],
        protocol_semantic_sha256=protocol['hashes']['semantic_sha256'],
        protocol_full_sha256=protocol['hashes']['full_sha256'],
        active_formal_bundle_sha256=ctx['scientific_identity']['active_formal_bundle_sha256'],
        execution_commit=proposal['execution_commit'],
        resolved_execution_context_sha256=ctx['context_sha256'],
        formal_training_execution_binding_sha256=binding['binding_full_sha256'])[1])
    # Import the benchmark module only; never call main/evaluation/rollout.
    from scripts.benchmark_main_results import validate_benchmark_checkpoint_gate
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime
    runtime_paths = {row['capacity_label']: root / row['runtime_config_path'] for row in
                     protocol['execution_contract']['command_templates']['train']['matrix_contexts']}
    frozen = read(run / 'checkpoint_freeze.json')
    gates = []
    for row in frozen['frozen_checkpoints']:
        capacity, agent, seed = row['capacity_label'], row['agent_name'], row['seed']
        companion = read(run / 'checkpoint_manifests' / capacity / 'checkpoint_provenance_manifest.json')
        envelope = companion[agent][str(seed)]
        def gate():
            report = validate_benchmark_checkpoint_gate(
                row['checkpoint_path'], expected_agent_name=agent, expected_seed=seed,
                expected_runtime_contract=resolve_model_cache_runtime(runtime_paths[capacity], root=root),
                expected_reward_positive_offset=0.0, provenance_envelope=envelope,
                protocol=protocol, resolved_execution_context=ctx, execution_binding=binding,
                expected_capacity_label=capacity)
            if report['status'] != 'compatible':
                raise ValueError(str(report))
            return {'path': row['checkpoint_path'], 'sha256': row['checkpoint_sha256'],
                    'agent': agent, 'seed': seed, 'capacity': capacity, 'status': 'pass'}
        checked = attempt('checkpoint_gate', gate)
        if checked:
            gates.append(checked)
    origins = validate_loaded_origins(root)
    results.append({'name': 'all_loaded_project_module_origins', 'status': 'pass', 'detail': origins})
    return {'checks': results, 'checkpoint_gate_pass_count': len(gates),
            'checkpoint_gate_attempt_count': len(frozen['frozen_checkpoints']),
            'execution_authorized': False, 'dispatch_count': dispatch_counts,
            'call_monitor': observed_calls}


if __name__ == '__main__':
    p = json.loads(sys.stdin.read())
    try:
        result = probe(p, str(Path(__file__).resolve().parents[1]))
    except Exception as exc:
        result = {'checks': [{'name': 'isolated_probe', 'status': 'fail', 'detail': str(exc)}],
                  'execution_authorized': False, 'dispatch_count': dict(train=0, dev=0, formal=0, holdout=0)}
    print(json.dumps(result, allow_nan=False, default=str))
