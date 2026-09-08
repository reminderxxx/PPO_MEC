"""Test-only integration fixture for real pinned writers and B reconciliation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')
COMMIT = 'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d'


def external(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor(path, kind):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    key = 'current_record_hash' if kind == 'phase' else 'current_ledger_hash'
    return {'path': str(path), 'kind': kind, 'byte_count': len(raw), 'record_count': len(rows),
            'prefix_sha256': hashlib.sha256(raw).hexdigest(), 'terminal_hash': rows[-1][key],
            'run_identity_fingerprint': rows[-1]['run_identity_fingerprint']}


def run(fixture, fault):
    security = external('continuation_executor_security')
    security.validate_fixture_root(fixture)
    if not fixture.name.startswith('synthetic_') or fixture.exists():
        raise ValueError('fixture requires fresh synthetic root')
    if security.git(SCIENCE, 'rev-parse', 'HEAD') != COMMIT:
        raise ValueError('scientific source commit mismatch')
    fixture.mkdir()
    kernel_boundary = external('continuation_fixture_sandbox').verify_kernel_boundary(fixture)
    adapter = external('continuation_legacy_adapter')
    science_runner, source = adapter.load_scientific_modules(
        SCIENCE, security.file_hash(SCIENCE / 'scripts/run_typed_model_cache_formal_protocol.py'), expected_commit=COMMIT)
    monitor = external('continuation_fixture_monitor').FixtureMonitor(
        fixture_root=fixture, scientific_root=SCIENCE, python_executable=sys.executable,
        implementation_root=ROOT, forbidden_roots=[
            '/Users/howen/Projects/PPO_MEC/artifacts/experiments',
            '/Users/howen/Projects/PPO_MEC/data', SCIENCE / 'data', SCIENCE / 'artifacts']).install()
    run_root = fixture / 'synthetic_run'
    runner = science_runner.TransactionalPhaseRunner(output_root=run_root,
        run_identity_fingerprint='a'*64, phase_order=science_runner.PHASE_ORDER)
    identity = science_runner.CellExecutionIdentity(
        run_id=run_root.name, execution_commit=COMMIT, protocol_semantic_sha256='b'*64,
        resource_registry_semantic_sha256='c'*64, environment_fingerprint='d'*64,
        split_semantic_sha256='e'*64, window_contract_semantic_sha256='f'*64,
        catalog_fingerprint='1'*64, runtime_identity='2'*64, command_matrix_sha256='3'*64)
    ledger = science_runner.FormalCellLedger(run_root=run_root, identity=identity)
    for phase in ('preflight', 'tests'):
        runner.run_phase(phase, commands=[], input_hash='synthetic', expected_outputs=[])
    def prepare_prefix(argv):
        begun = ledger.begin_cell(phase='train', coordinates={'fixture': 'already_complete'},
            command=argv, input_hash='synthetic_prefix', committed_path=run_root / 'training' / 'synthetic_prefix')
        staging = Path(begun['record']['staging_path'])
        (staging / 'test_only_checkpoint.json').write_text('{"test_only":true}\n')
        ledger.commit_cell(begun['cell_id'], required_paths=['test_only_checkpoint.json'],
                           monotonic_started_ns=time.monotonic_ns())
        return science_runner.PhaseCommandResult(0)
    runner.run_phase('train', commands=[['synthetic_prefix_setup_not_dispatch']],
        input_hash='synthetic', expected_outputs=['training/**/test_only_checkpoint.json'], executor=prepare_prefix)
    for phase in ('dev_select', 'checkpoint_freeze'):
        runner.run_phase(phase, commands=[], input_hash='synthetic', expected_outputs=[])
    anchors = [anchor(run_root / (kind + '_state.jsonl'), kind) for kind in ('phase', 'cell')]
    before = {row['kind']: Path(row['path']).read_bytes() for row in anchors}
    # Fill the preceding two non-cell phases as legitimate empty synthetic phases;
    # this fixture tests one support-style publication, not full eight-stage coverage.
    for phase in ('formal_cache_policy', 'formal_controller'):
        runner.run_phase(phase, commands=[], input_hash='synthetic', expected_outputs=[])
    phase = 'formal_ablation'
    coordinates = {'ablation_setting_id': 'synthetic_setting'}
    command = [sys.executable, '-I', '-B', str(ROOT / 'scripts/continuation_fixture_child.py'),
               '--fixture-root', str(fixture), '--scientific-root', str(SCIENCE),
               '--output-root', str(run_root / phase), '--setting', 'synthetic_setting', '--fault', fault if fault not in {'publication_crash', 'finalize_only'} else 'none']
    protocol = {'typed_model_cache_formal_protocol_version': '2.9.0', 'hashes': {'semantic_sha256': 'b'*64},
                'formal_nullable_metric_aggregation_contract': {'semantic_sha256': '4'*64}}
    environment = SimpleNamespace(child_environment=dict(os.environ, PYTHONPATH=str(SCIENCE),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0'))
    execute, receipt = adapter.make_cell_executor(science_runner, source, phase=phase,
        protocol=protocol, context_payload={}, execution_binding={},
        bundle={'active_formal_bundle_sha256': '5'*64}, environment=environment,
        cell_ledger=ledger, commands=[command], matrix_contexts=[coordinates],
        generated_registry_audit={'registry_canonical_sha256': '6'*64})
    external('continuation_fixture_authorization').issue(fixture,run_root,ROOT)
    error = None
    recovery = {}
    phase_args = dict(commands=[command], input_hash='synthetic',
        expected_outputs=[phase + '/**/support_provenance.json'])
    original_append = ledger._append
    try:
        if fault == 'publication_crash':
            # A real process exit bypasses exception handlers just like power loss.
            # The native phase runner correctly makes caught exceptions terminal;
            # those are not equivalent to a crash between publication and append.
            crash_receipt = fixture / 'publication_crash_monitor.json'
            pid = os.fork()
            if pid == 0:
                def interrupt_terminal(record):
                    if record.get('status') == 'committed' and record.get('phase') == phase:
                        assert Path(record['committed_path']).is_dir()
                        crash_receipt.write_text(json.dumps(monitor.report()))
                        os._exit(86)
                    return original_append(record)
                ledger._append = interrupt_terminal
                try:
                    runner.run_phase(phase, **phase_args, executor=execute, infrastructure_retries=1)
                finally:
                    os._exit(87)
            _, status = os.waitpid(pid, 0)
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 86
            crashed_monitor = json.loads(crash_receipt.read_text())
            monitor.counts.update(crashed_monitor['counts'])
            monitor.events[:] = crashed_monitor['events']
            monitor.scientific_calls.update(crashed_monitor['scientific_calls'])
            published = list(run_root.glob('formal_ablation/**/committed_marker.json'))
            assert len(published) == 1
            before_recovery = monitor.counts['synthetic_child_dispatch_count']
            result = runner.run_phase(phase, **phase_args, executor=execute, infrastructure_retries=1)
            recovery = {'case':'publication_before_append', 'native_recovery':True,
                'crash_process_exit_code':86,
                'extra_child_dispatches':monitor.counts['synthetic_child_dispatch_count'] - before_recovery,
                'publication_states':[row.get('publication_state') for row in ledger.committed_records(phase=phase)]}
            assert recovery['extra_child_dispatches'] == 0
        else:
            result = runner.run_phase(phase, **phase_args, executor=execute,
                infrastructure_retries=1, stop_after_completion_candidate=fault == 'finalize_only')
        if fault == 'finalize_only':
            assert result['status'] == 'completion_candidate_created'
            before_finalize = monitor.counts['synthetic_child_dispatch_count']
            try:
                runner.run_phase(phase, **phase_args, executor=execute)
            except ValueError as exc:
                assert 'finalize' in str(exc)
            else:
                raise AssertionError('candidate was rerun')
            result = runner.finalize_phase_only(phase, **phase_args)
            recovery = {'case':'completion_candidate_finalize_only', 'native_recovery':True,
                'extra_child_dispatches':monitor.counts['synthetic_child_dispatch_count'] - before_finalize}
            assert recovery['extra_child_dispatches'] == 0
        before_restart = monitor.counts['synthetic_child_dispatch_count']
        repeated = runner.run_phase(phase, **phase_args, executor=execute)
        assert repeated['status'] == 'skipped_completed'
        recovery['repeated_start_dispatches'] = monitor.counts['synthetic_child_dispatch_count'] - before_restart
        assert recovery['repeated_start_dispatches'] == 0
    except Exception as exc:
        error, result = str(exc), None
        before_restart = monitor.counts['synthetic_child_dispatch_count']
        try:
            runner.run_phase(phase, **phase_args, executor=execute)
        except ValueError as terminal:
            assert 'terminal' in str(terminal)
            recovery['failed_terminal_restart'] = 'rejected'
        else:
            raise AssertionError('failed terminal restarted')
        assert monitor.counts['synthetic_child_dispatch_count'] == before_restart
    finally:
        ledger._append = original_append
    expected = {name: set() for name in security.PHASES}
    expected[phase] = {science_runner.stable_cell_id(phase, coordinates)}
    validator = external('continuation_ledger_validation')
    from src.evaluators.formal_cell_transaction import artifact_inventory, validate_cell_ledger
    from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
    validation = None
    validation_error = None
    try:
        validation = validator.validate_successors(anchors, root=run_root,
            phase_validator=validate_phase_ledger_v3, cell_validator=validate_cell_ledger,
            artifact_inventory=artifact_inventory, expected_cells_by_phase=expected)
    except Exception as exc:
        validation_error = str(exc)
    assert all(Path(row['path']).read_bytes().startswith(before[row['kind']]) for row in anchors)

    negative_checks = []
    if fault == 'none' and validation is not None:
        phase_path, cell_path = run_root / 'phase_state.jsonl', run_root / 'cell_state.jsonl'
        original_phase, original_cell = phase_path.read_bytes(), cell_path.read_bytes()
        def rehash(rows, kind):
            current, previous = (('current_record_hash', 'previous_record_hash')
                if kind == 'phase' else ('current_ledger_hash', 'previous_ledger_hash'))
            tip = None
            for index, row in enumerate(rows, 1):
                row['sequence_number'], row[previous] = index, tip
                row[current] = security.digest({k:v for k,v in row.items() if k != current})
                tip = row[current]
            return b''.join(security.canonical(row) + b'\n' for row in rows)
        def assert_rejected(name, mutate):
            try:
                mutate()
                validator.validate_successors(anchors, root=run_root,
                    phase_validator=validate_phase_ledger_v3, cell_validator=validate_cell_ledger,
                    artifact_inventory=artifact_inventory, expected_cells_by_phase=expected)
            except ValueError as exc:
                negative_checks.append({'case': name, 'status': 'rejected', 'reason': str(exc)})
            else:
                raise AssertionError('invalid ledger accepted: ' + name)
            finally:
                phase_path.write_bytes(original_phase)
                cell_path.write_bytes(original_cell)
        assert_rejected('prefix_truncation', lambda: phase_path.write_bytes(original_phase[:20]))
        assert_rejected('cell_truncation', lambda: cell_path.write_bytes(original_cell[:-20]))
        assert_rejected('prefix_rewrite', lambda: phase_path.write_bytes(original_phase.replace(b'synthetic', b'changed__', 1)))
        def no_matching_phase():
            rows = [json.loads(line) for line in original_phase.splitlines()]
            rows = [row for row in rows if row['phase'] != phase]
            # Keep the approved prefix byte-for-byte; append only re-encoded successors.
            phase_path.write_bytes(original_phase[:anchors[0]['byte_count']])
            phase_path.write_bytes(original_phase[:anchors[0]['byte_count']] +
                rehash(rows, 'phase').split(b'\n', anchors[0]['record_count'])[-1])
        assert_rejected('cross_ledger_missing_phase', no_matching_phase)
        def append_duplicate():
            rows = [json.loads(line) for line in original_cell.splitlines()]
            row = dict(rows[-1]); row['sequence_number'] += 1
            row['previous_ledger_hash'] = rows[-1]['current_ledger_hash']
            row['current_ledger_hash'] = security.digest({k:v for k,v in row.items() if k != 'current_ledger_hash'})
            cell_path.write_bytes(original_cell + security.canonical(row) + b'\n')
        assert_rejected('duplicate_committed_id', append_duplicate)
        def change_last(key, value):
            rows = [json.loads(line) for line in original_cell.splitlines()]
            rows[-1][key] = value
            rows[-1]['current_ledger_hash'] = security.digest({k:v for k,v in rows[-1].items() if k != 'current_ledger_hash'})
            cell_path.write_bytes(b'\n'.join(original_cell.splitlines()[:-1]) + b'\n' + security.canonical(rows[-1]) + b'\n')
        assert_rejected('successor_fork', lambda: change_last('previous_ledger_hash', '0'*64))
        assert_rejected('cross_run_splice', lambda: change_last('run_identity_fingerprint', '0'*64))
        assert_rejected('cell_command_identity_drift', lambda: change_last('command_hash', '0'*64))
        assert_rejected('cell_input_identity_drift', lambda: change_last('input_hash', '0'*64))
        protected = run_root / 'training/synthetic_prefix/test_only_checkpoint.json'
        protected_bytes = protected.read_bytes()
        try:
            assert_rejected('immutable_prefix_payload_changed', lambda: protected.write_text('changed'))
        finally:
            protected.write_bytes(protected_bytes)

    completion_rejections = []
    if fault == 'none':
        gate_path = run_root / 'formal_gate.json'
        frozen_ledgers = {path:path.read_bytes() for path in (run_root/'phase_state.jsonl',run_root/'cell_state.jsonl')}
        for case in ('missing_gate', 'failed_gate'):
            if case == 'failed_gate':
                gate_path.write_text(json.dumps({'passed':False}))
            dispatch_before = monitor.counts['synthetic_child_dispatch_count']
            try:
                adapter.run_continuation_phase(science_runner, source, phase='complete_without_holdout',
                    protocol=protocol, context_payload={'resolved_expansion_context':{'output_root':str(run_root)}},
                    execution_binding={}, bundle={}, environment=environment,
                    generated_registry_audit={}, admission=lambda:None)
            except ValueError as exc:
                completion_rejections.append({'case':case,'status':'rejected','reason':str(exc),
                    'ledger_unchanged':all(path.read_bytes()==raw for path,raw in frozen_ledgers.items()),
                    'extra_dispatches':monitor.counts['synthetic_child_dispatch_count']-dispatch_before})
            else:
                raise AssertionError('completion allowed without legal gate')
        gate_path.unlink()

    child_reports = [json.loads(path.read_text()) for path in run_root.glob('formal_ablation/**/cell_stdout.log')]
    report = {'test_only': True, 'scope': 'one projected publication transaction; not full eight-stage acceptance',
              'kernel_boundary': kernel_boundary, 'fault': fault, 'result': result, 'error': error, 'validation': validation,
              'validation_error': validation_error, 'recovery': recovery, 'anchors': anchors, 'expected_cells': {k:sorted(v) for k,v in expected.items()},
              'monitor': monitor.report(), 'child_monitors': child_reports,
              'negative_checks': negative_checks, 'completion_rejections':completion_rejections, 'projection_receipt': receipt, 'scientific_modules': adapter.loaded_origins(SCIENCE)}
    (fixture / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'report': str(fixture / 'report.json'), 'error': error,
                      'validation_error': validation_error, 'counts': monitor.counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--fault', default='none')
    args = parser.parse_args()
    run(args.fixture_root, args.fault)
