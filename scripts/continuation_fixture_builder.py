"""Generate test-only inputs and already-completed prefix, without training/rollout.

All scientific builders, checkpoint annotation/read-back, selection, freeze and
registry publication are the original pinned functions. Fixture command templates
explicitly substitute payload producers; this is not the real v16 command plan.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build_fixture(fixture, science, implementation, runner, source, adapter, *, implementation_commit, scientific_commit, benchmark_inputs=False):
    from src.runtime.portable_resource_identity import build_registry, build_resource_identity
    from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
    from src.runtime.formal_training_identity import expected_checkpoint_training_identity
    from src.runtime.formal_training_contract import resolve_training_contract
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime, build_checkpoint_provenance
    from scripts.train_algo_pool_real_sample import (
        build_training_identity_metadata, annotate_checkpoint, validate_serialized_formal_checkpoint)
    from scripts.manage_typed_model_cache_formal_artifacts import dev_select, checkpoint_freeze, write_checkpoint_companions
    import torch
    science, fixture, implementation = Path(science), Path(fixture), Path(implementation)
    run = fixture / 'synthetic_run'
    run.mkdir()
    inputs = fixture / 'inputs'
    inputs.mkdir()
    original_dir = science / 'configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906'
    protocol = json.loads((original_dir / 'protocol_v2_9_manifest.json').read_text())
    source_protocol_sha256 = protocol['hashes']['full_sha256']
    order_path = inputs / 'formal_agent_order_contract.json'
    order_path.write_bytes((original_dir / 'formal_agent_order_contract.json').read_bytes())
    scientific_path = inputs / 'agent_training_scientific_config.json'
    scientific_path.write_bytes((original_dir / 'agent_training_scientific_config.json').read_bytes())
    environment_path = inputs / 'execution_environment_manifest.json'
    environment_path.write_bytes((original_dir / 'execution_environment_manifest.json').read_bytes())
    scientific = json.loads(scientific_path.read_text())
    agents = json.loads(order_path.read_text())['main_benchmark_agent_order']
    catalog = inputs / 'src/data/model_catalog/typed_model_cache_controlled.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes((science / 'src/data/model_catalog/typed_model_cache_controlled.json').read_bytes())
    runtime_paths = {}
    for row in protocol['execution_contract']['command_templates']['train']['matrix_contexts']:
        capacity = row['capacity_label']
        if capacity in runtime_paths:
            continue
        relative = Path(row['runtime_config_path'])
        destination = inputs / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((science / relative).read_bytes())
        runtime_paths[capacity] = destination
    table = inputs / 'synthetic_input.json'
    write(table, {'test_only': True, 'no_real_data': True})
    resources = [build_resource_identity(table, logical_resource_id='synthetic.input',
        resource_role='test_only_input', schema_version='1.0.0', revision='synthetic',
        expected_logical_relative_path='synthetic_input.json')]
    benchmark = None
    if benchmark_inputs:
        import importlib.util
        spec = importlib.util.spec_from_file_location('external_benchmark_inputs',
            implementation / 'scripts/continuation_benchmark_inputs.py')
        producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
        benchmark = producer.prepare(inputs, science, runtime_paths)
        protocol['identity']['typed_runtime_contract_hashes_by_capacity'] = {
            capacity.split('_',1)[0]:row['runtime']['runtime_contract_sha256']
            for capacity,row in benchmark['fairness'].items()}
        for coordinate in protocol['execution_contract']['command_templates']['train']['matrix_contexts']:
            coordinate['runtime_config_path'] = str(runtime_paths[coordinate['capacity_label']])
        def add(path, logical, role):
            resources.append(build_resource_identity(path,logical_resource_id=logical,
                resource_role=role,schema_version='1.0.0',revision='synthetic',
                expected_logical_relative_path=Path(path).relative_to(inputs).as_posix()))
        add(benchmark['mobility'],'mobility.synthetic','mobility_dataset')
        add(benchmark['workflow'],'workflow.synthetic','workflow_dataset')
        add(benchmark['window_plan'],'window.synthetic','window_plan')
        for capacity, runtime_path in runtime_paths.items():
            add(runtime_path,'runtime.'+capacity,'runtime_config')
            add(benchmark['fairness'][capacity]['path'],'fairness.'+capacity,'fairness_manifest')
    static = build_registry(resources, registry_id='synthetic_continuation_inputs')
    registry_path = inputs / 'static_registry.json'
    write(registry_path, static)
    protocol['portable_resource_identity_contract']['resource_registry_semantic_sha256'] = static['hashes']['semantic_sha256']
    protocol['continuation_fixture'] = {'test_only': True, 'source_protocol_full_sha256': source_protocol_sha256,
        'scientific_rollout_allowed': False, 'performance_evidence': False,
        'substitution': 'synthetic payload commands; real statistics/integrity/gate consumers'}
    child = str(implementation / 'scripts/continuation_fixture_child.py')
    consumer = str(implementation / 'scripts/continuation_fixture_consumer.py')
    phases = adapter.PHASES
    templates = protocol['execution_contract']['command_templates']
    # Keep all original coordinates. Prefix phases exist only as prepared terminals.
    for phase, spec in templates.items():
        if phase not in phases:
            spec['argv'] = ['{python_executable}', 'synthetic_prefix_never_dispatched', phase]
            spec['expected_outputs'] = ['synthetic_prefix/' + phase + '.json']
    for phase in phases[:5]:
        key = {'formal_cache_policy':'capacity_label', 'formal_controller':'capacity_label',
               'formal_ablation':'ablation_setting_id', 'formal_support':'support_setting_id',
               'formal_scalability':'scalability_setting_id'}[phase]
        templates[phase]['argv'] = ['{python_executable}', '-I', '-B', child,
            '--fixture-root', '{fixture_root}', '--scientific-root', '{clean_worktree_root}',
            '--output-root', '{output_root}/' + phase, '--cell-phase', phase,
            '--setting', '{' + key + '}', '--row-input', '{synthetic_rows_path}']
        if phase == 'formal_controller':
            templates[phase]['argv'] += ['--agents', *agents]
        if phase == 'formal_cache_policy':
            templates[phase]['argv'] += ['--request-replay-path', '{output_root}/synthetic_replay.json']
        templates[phase]['expected_outputs'] = [phase + '/**/aggregate_summary.json']
    common = ['--protocol-path', '{protocol_path}', '--input-root', '{output_root}',
        '--resource-registry-path', '{resource_registry_path}',
        '--generated-checkpoint-registry-path', '{generated_checkpoint_registry_path}']
    templates['formal_statistics']['argv'] = ['{python_executable}', '-I', '-B', consumer,
        '--fixture-root', '{fixture_root}', '--scientific-root', '{clean_worktree_root}',
        '--module', 'scripts.run_typed_model_cache_formal_statistics', '--', *common,
        '--output-root', '{output_root}/statistics', '--non-formal-rehearsal',
        '--rehearsal-baseline-agent', 'reactive_lru',
        '--resolved-execution-context-path', '{resolved_execution_context_path}',
        '--formal-agent-order-contract-path', '{formal_agent_order_contract_path}']
    templates['formal_statistics']['expected_outputs'] = ['statistics/paired_statistics.json']
    templates['formal_gate']['argv'] = ['{python_executable}', '-I', '-B', consumer,
        '--fixture-root', '{fixture_root}', '--scientific-root', '{clean_worktree_root}',
        '--module', 'scripts.manage_typed_model_cache_formal_artifacts', '--', *common,
        '--action', 'integrity_and_formal_gate', '--output-path', '{output_root}/formal_gate.json']
    templates['formal_gate']['expected_outputs'] = ['artifact_integrity_manifest.json', 'formal_gate.json']
    protocol = attach_hashes(protocol)
    protocol_path = inputs / 'synthetic_protocol.json'
    write(protocol_path, protocol)
    runner.validate_protocol_v1_1(protocol)
    bundle = {'active_formal_bundle_sha256': runner.canonical_sha256({'test_only_fixture': str(fixture),
        'protocol': protocol['hashes']['full_sha256'], 'implementation_commit': implementation_commit})}
    context = runner.resolved_expansion_context(protocol, protocol_path=str(protocol_path),
        output_root=str(run), python_executable=sys.executable,
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'])
    context.update(fixture_root=str(fixture), data_root=str(inputs), checkpoint_root=str(run),
        protocol_artifact_root=str(inputs), resource_registry_path=str(registry_path),
        agent_scientific_config_path=str(scientific_path), formal_agent_order_contract_path=str(order_path),
        generated_checkpoint_registry_path=str(run / 'generated_checkpoint_resource_registry.json'),
        synthetic_rows_path=str(inputs / 'synthetic_benchmark_rows.csv'))
    # Inactive historical template paths are not consumed. The active fixture argv
    # and all checkpoint/companion/output references remain inside the fixture.
    command_report = runner.validate_command_templates(templates, context)
    env_identity = deepcopy(protocol['formal_execution_environment_contract']['scientific_identity'])
    commit = scientific_commit
    binding = runner.build_execution_binding(protocol=protocol, scientific_config=scientific,
        execution_commit=commit, environment_identity=env_identity,
        command_matrix_sha256=command_report['command_matrix_sha256'],
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'])
    audit = {'observed_execution_commit': commit, 'resolution_source': 'explicit_python_executable',
             'resolved_python_absolute_path': sys.executable}
    payload = runner.build_resolved_formal_execution_context(protocol=protocol, expansion_context=context,
        environment_identity=env_identity, runtime_audit=audit, environment_manifest_path=environment_path,
        outer_expansion_sha256=command_report['command_matrix_sha256'], phase_count=command_report['phase_count'],
        command_count=command_report['command_count'], execution_binding=binding,
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'])
    context_bytes = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    environment = SimpleNamespace(environment_identity=env_identity, runtime_audit=audit,
        child_environment=dict(__import__('os').environ))
    namespace = dict(vars(runner))
    namespace.update(protocol=protocol, requested_output_root=str(run), args=SimpleNamespace(output_root=str(run)),
        resolved_context_payload=payload, resolved_context_file_sha256=hashlib.sha256(context_bytes).hexdigest(), execution_binding=binding,
        active_bundle=bundle, environment_resolution=environment)
    run_identity, _ = adapter.original_identity_expression(source, Path(runner.__file__), 'run_identity', namespace)
    cell_identity, _ = adapter.original_identity_expression(source, Path(runner.__file__), 'cell_identity', namespace)
    phase_runner = runner.TransactionalPhaseRunner(output_root=run, run_identity_fingerprint=run_identity,
        phase_order=runner.PHASE_ORDER, resolved_execution_context_sha256=payload['context_sha256'],
        resolved_execution_context_file_sha256=namespace['resolved_context_file_sha256'])
    write(run / 'formal_training_execution_binding.json', binding)
    (run / 'resolved_execution_context.json').write_bytes(context_bytes)
    ledger = runner.FormalCellLedger(run_root=run, identity=cell_identity)
    for phase in ('preflight', 'tests'):
        phase_runner.run_phase(phase, commands=[], input_hash='synthetic_prefix', expected_outputs=[])
    resolved = {agent: resolve_training_contract(agent_name=agent, profile_defaults={'episodes':1,'update_every':1,'batch_size':1,'max_steps':1}, cli_values={},
        formal_protocol=protocol, scientific_config=scientific, execution_binding=binding,
        resolved_execution_context=payload) for agent in protocol['training_budget']['learned_agent_order']}
    candidates = []
    runtimes = {cap: resolve_model_cache_runtime(path, root=science if benchmark_inputs else inputs) for cap, path in runtime_paths.items()}
    def create_prefix(argv):
        for coordinate in templates['train']['matrix_contexts']:
            agent, seed, capacity = coordinate['agent'], int(coordinate['seed']), coordinate['capacity_label']
            config, runtime = resolved[agent], runtimes[capacity]
            cell = ledger.begin_cell(phase='train', coordinates=coordinate,
                command=['synthetic_prepared_checkpoint', agent, str(seed), capacity],
                input_hash='synthetic_prefix', committed_path=run / 'training' / capacity / agent / ('seed_' + str(seed)))
            staging = Path(cell['record']['staging_path'])
            path = staging / 'checkpoints/update_0004.pt'
            path.parent.mkdir()
            metadata = {'test_only': True, 'run_id': 'synthetic_' + agent + '_' + str(seed), 'agent_name': agent,
                'episodes': config.episodes, 'update_count': 4,
                'checkpoint_schedule': {'checkpoint_every_updates': config.checkpoint_every_updates,
                    'expected_update_count': config.expected_update_count},
                **build_training_identity_metadata(config),
                'typed_runtime_provenance': build_checkpoint_provenance(root=science, agent_name=agent,
                    training_seed=seed, runtime_contract=runtime, reward_positive_offset=0.0,
                    train_window_plan_identity={'test_only': True, 'split': 'train'})}
            torch.save({'test_state': True}, path)
            annotate_checkpoint(path, metadata)
            validate_serialized_formal_checkpoint(path, resolved_training=config, agent_name=agent,
                seed=seed, runtime_contract_sha256=runtime['runtime_contract_sha256'])
            (path.parent / 'latest.pt').write_bytes(path.read_bytes())
            ledger.commit_cell(cell['cell_id'], required_paths=['checkpoints/update_0004.pt','checkpoints/latest.pt'],
                               monotonic_started_ns=time.monotonic_ns())
            final = Path(cell['record']['committed_path']) / 'checkpoints/update_0004.pt'
            metrics = {'full_service_ready_byte_hit_rate':0.5, 'workflow_continuity_rate':0.5,
                       'transfer_mb_per_request':1.0, 'end_to_end_workflow_delay':2.0}
            candidates.append({'agent_name':agent,'seed':seed,'capacity_label':capacity,'update_index':4,
                'checkpoint_path':str(final),'checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
                **metrics, 'selection_metric_availability':{metric:{'available_count':1,'unavailable_count':0,'total_count':1} for metric in metrics},
                'runtime_contract_sha256':runtime['runtime_contract_sha256'],'resolved_agent_config':config.agent_config,
                'checkpoint_schedule':metadata['checkpoint_schedule'],**build_training_identity_metadata(config),
                'non_formal_rehearsal':False,'typed_runtime_provenance':metadata['typed_runtime_provenance']})
        write(run / 'checkpoint_candidates.json', candidates)
        return runner.PhaseCommandResult(0)
    phase_runner.run_phase('train', commands=[['synthetic_prefix_builder_not_training']], input_hash='synthetic_prefix',
        expected_outputs=['training/**/checkpoints/update_0004.pt'], executor=create_prefix)
    expected = expected_checkpoint_training_identity(protocol=protocol, resolved_execution_context=payload)
    def select(argv):
        selection = dev_select(run, protocol, expected_training_identity=expected)
        write(run / 'dev_selection.json', selection)
        return runner.PhaseCommandResult(0)
    phase_runner.run_phase('dev_select', commands=[['synthetic_prefix_selection_no_evaluation']], input_hash='synthetic_prefix',
        expected_outputs=['checkpoint_candidates.json','dev_selection.json'], executor=select)
    def freeze(argv):
        frozen = checkpoint_freeze(run, protocol)
        frozen['checkpoint_companions'] = write_checkpoint_companions(run, frozen)
        write(run / 'checkpoint_freeze.json', frozen)
        return runner.PhaseCommandResult(0)
    phase_runner.run_phase('checkpoint_freeze', commands=[['synthetic_prefix_freeze']], input_hash='synthetic_prefix',
        expected_outputs=['checkpoint_freeze.json','checkpoint_manifests/**/*.json'], executor=freeze)
    registry = runner.build_generated_checkpoint_registry(run_root=run, protocol=protocol, static_registry=static,
        resolved_execution_context=payload, execution_binding=binding)
    generated = run / 'generated_checkpoint_resource_registry.json'
    publication = runner.publish_or_validate_generated_checkpoint_registry(generated, registry,
        run_root=run, expected_run_id=run.name, static_registry_semantic_sha256=static['hashes']['semantic_sha256'],
        protocol_semantic_sha256=protocol['hashes']['semantic_sha256'], protocol_full_sha256=protocol['hashes']['full_sha256'],
        active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'], execution_commit=commit,
        resolved_execution_context_sha256=payload['context_sha256'],
        formal_training_execution_binding_sha256=binding['binding_full_sha256'])
    counts = {'committed_training_cells':150,'candidate_checkpoints':150,'latest_checkpoints':150,
        'dev_candidate_evaluations':150,'selections':150,'frozen_checkpoints':150,
        'frozen_checkpoints_by_capacity':{cap:50 for cap in runtime_paths},
        'cache_policy_cells':3,'controller_cells':3,'ablation_settings':2,'support_settings':11,
        'scalability_settings':3,'primary_comparison_rows':6,'formal_outer_window_clusters':2}
    write(run / 'non_formal_rehearsal.json', {'test_only':True,'performance_evidence':False,
        'holdout_capability':False,'expected_counts':counts})
    import csv
    fields = ['agent_name','seed','workflow_id','source_segment_run_id','window_id', *protocol['endpoints']['primary']]
    with (inputs / 'synthetic_benchmark_rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for agent in agents:
            for window in ['synthetic_window_0','synthetic_window_1']:
                for seed in [7,13,29,43,71]:
                    writer.writerow({'agent_name':agent,'seed':seed,'workflow_id':'synthetic_workflow',
                        'source_segment_run_id':'synthetic_segment','window_id':window,
                        **{metric:0.5 for metric in protocol['endpoints']['primary']}})
    return dict(run_root=run, protocol=protocol, context_payload=payload, execution_binding=binding,
        bundle=bundle, environment=environment, publication=publication, command_report=command_report,
        runtime_paths=runtime_paths, synthetic_inputs_root=inputs, benchmark_inputs=benchmark)
