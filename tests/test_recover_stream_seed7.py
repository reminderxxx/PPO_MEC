"""Pure mock files/schema fixtures; no scientific model, environment or training."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('seed7_recovery', ROOT / 'scripts/recover_stream_seed7.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
AUDIT_SPEC = importlib.util.spec_from_file_location('seed7_mock_schema', ROOT / 'scripts/sa_stage5c_credit_audit.py')
A = importlib.util.module_from_spec(AUDIT_SPEC); AUDIT_SPEC.loader.exec_module(A)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(parent, *, early_stop=False):
    parent = Path(parent); root = parent / 'failed'; root.mkdir()
    protocol = parent / 'manifest.json'
    cases = ['train_000', 'train_032', 'train_064', 'train_004']
    put(protocol, dict(protocol='joint_stream_learning_v2', action_contract='joint_stream_control_v2',
        smoke=False, episodes=64, seeds=[7, 17, 27], training_schedule={'7': cases + ['train_rest'] * 60}))
    original = parent / 'original_contract.json'
    put(original, dict(schema='stream_stability_diagnostic_v1', status='frozen', source_commit=M.SOURCE_COMMIT,
        output_root=str(root), protocol=dict(path=str(protocol), sha256=M.sha_file(protocol)),
        seeds=[7, 17, 27], sides=['off', 'on'], training_budget=64, sessions_per_seed=4, epochs=4,
        limits=dict(audit_bytes_per_seed=M.LIMITS['max_audit_bytes'], audit_rows_per_seed=100000)))
    put(root / 'run_started.json', dict(contract_sha256=M.sha_file(original), source_commit=M.SOURCE_COMMIT,
        no_retry=True, pid=123, started_at='2026-10-22T07:31:32+00:00'))
    off_counts = dict(training_sessions=4, training_updates=1, training_frames=8, models=0,
        environments=4, adam_calls=152, backward_calls=76, categorical_sample_calls=12, evaluate_calls=20)
    counts = {k: 2 * v for k, v in off_counts.items()}; counts['models'] = 2
    context = dict(seed=7, side='on', episode=4, case_id=cases[-1], phase='independent_consumer')
    put(root / 'failure.json', dict(status='failed', completed_seeds=0, no_retry=True,
        exception_type='RuntimeError', message='Frozen RSS limit exceeded; no automatic expansion/retry',
        context=context, counts=counts, possibly_partial_update=True))
    put(root / 'progress.json', dict(context=context, counts=counts))
    for side in ('off', 'on'):
        home = root / 'seed7' / side; home.mkdir(parents=True)
        for name in ('checkpoint.pt', 'final_evidence.pt'):
            (home / name).write_bytes(b'opaque mock bytes; deliberately not Torch ' + name.encode())
        for episode, case in enumerate(cases, 1):
            put(home / ('episode%d.json' % episode), dict(episode=episode, case_id=case,
                summary_sha256=str(episode) * 64, rolls_sha256='a' * 64,
                receipts_sha256='b' * 64, rng_after_sha256='c' * 64))
    off = root / 'seed7/off'
    put(off / 'result.json', dict(seed=7, side='off', case_ids=cases, episode_count=4,
        checkpoint_sha256=M.sha_file(off / 'checkpoint.pt'), final_evidence_sha256=M.sha_file(off / 'final_evidence.pt'),
        original_call_counts=off_counts, coverage=dict(decision_rows=4, active_heads=[4, 0, 4], forced_heads=[0, 4, 0],
            main_adam_calls=76, surrogate_adam_calls=76, idle_frames=1)))
    run_uid = 'qualification_seed7'
    header = dict(run_uid=run_uid, source_commit=M.SOURCE_COMMIT, evidence_kind='unverified_actual_chain',
        provenance=dict(contract_sha256=M.sha_file(original), protocol_sha256=M.sha_file(protocol),
            protocol_path=str(protocol), seed=7, side='on', source='fresh_original_stochastic_stream_sessions',
            source_run_id='MOCK_NO_SCIENTIFIC_EXECUTION'))
    sink = A.AuditSink(root / 'seed7/on/audit', enable=True, header=header,
        max_bytes=M.LIMITS['max_audit_bytes'], max_rows=100000, compact_joins=True)
    bids = []
    for j, case in enumerate(cases):
        bids.append(sink.emit('behavior', dict(seed=7, session_id='mock_' + str(j), frame=j, vehicle_id='v0',
            pass_name='two_pass_behavior', request_id='r0', node_id='n0', decision_uid='d' + str(j),
            candidates=[['a', 'b'], [None], ['p', 'q']], masks=[[True, True], [True], [True, True]],
            action_indices=[0, 0, 1], old_log_probs=[-.693147, 0., -.693147], causal_prefix=dict(behavior_uids=[]),
            metadata=dict(session=dict(case_id=case), stored_baseline=[.25, .5, .75]))))
    uid = A.update_uid(run_uid, 7, 1); cids = []
    for bid in bids:
        cids.append(sink.emit('credit', dict(seed=7, update_index=1, update_uid=uid, behavior_uid=bid,
            T=8., T_team=1., fleet=8, B=[.25, .5, .75], raw_adv=[.75, .5, .25], normalized_adv=[1., 0., -1.],
            active_heads=[True, False, True], global_value_target=1.)))
    delta = {group: dict(l2=.01, max_abs=.01, parameter_count=2, changed_parameter_count=1)
             for group in ('policy', 'value', 'surrogate')}
    gradient = {group: dict(l2=.1, present_tensors=1, elements=2) for group in delta}
    calls = [dict(optimizer=label, postclip_gradients=copy.deepcopy(gradient), parameter_delta=copy.deepcopy(delta))
             for label in ('main', 'surrogate')]
    steps = []
    for j in range(76 + int(early_stop)):
        stepped = j < 76; step = 'main_' + str(j) if stepped else None
        sink.emit('minibatch', dict(seed=7, update_index=1, update_uid=uid, epoch=j // 19, minibatch_index=j % 19,
            behavior_uids=bids, credit_uids=cids, rng_before_sha256='d' * 64, rng_after_sha256='e' * 64,
            kl=.001, step_observed=stepped, step_uid=step, kl_early_stop=not stepped,
            metadata=dict(scores=dict(ratio=[[1., 1., 1.]] * 4, active_heads=[[True, False, True]] * 4),
                optimizer_calls=calls if stepped else [], losses=dict(policy=.1, value=.2) if stepped else None)))
        if step: steps.append(step)
    sink.emit('update', dict(seed=7, update_index=1, update_uid=uid, completed_updates_before=0,
        completed_updates_after=1, observed_adam_steps=76, step_uids=steps, delta_observation='producer_supplied',
        adam_delta_summary=delta, metadata=dict(optimizer_step_counts=dict(main=76, surrogate=76),
            total_adam_calls=152, eligible_behavior_uids=bids, eligible_behavior_rows=4, losses=dict(samples=4, gradient_steps=76))))
    sink.finish(complete=True)
    contract = parent / 'recovery_contract.json'; output = parent / 'recovered'
    value = M.make_contract(root, original, output)
    M.write_json(contract, value)
    return root, contract, output


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def rewrite_audit(path, mutation):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    mutation(rows)
    prior = [A.canonical(row) for row in rows[:-1]]
    terminal = rows[-1]['payload']; terminal['bytes_before'] = sum(map(len, prior))
    import hashlib
    terminal['data_sha256'] = hashlib.sha256(b''.join(prior)).hexdigest()
    path.write_bytes(b''.join(prior) + A.canonical(rows[-1]))


class RecoveryTests(unittest.TestCase):
    def test_only_stdlib_imports_and_check_does_not_import_validator(self):
        imports = []
        for node in ast.walk(ast.parse(M.HERE.read_text())):
            if isinstance(node, ast.Import): imports.extend(item.name.split('.')[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append((node.module or '').split('.')[0])
        self.assertFalse(set(imports) & {'torch', 'numpy', 'src', 'scripts'})
        with tempfile.TemporaryDirectory() as folder:
            root, contract, output = fixture(folder); before = snapshot(root)
            with patch.object(M, 'load_validator', side_effect=AssertionError('imported')):
                M.check_contract(contract)
            self.assertEqual(snapshot(root), before); self.assertFalse(output.exists())

    def test_contract_binds_all_required_files_and_check_is_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            root, contract, output = fixture(folder); before = snapshot(root)
            value = M.check_contract(contract)
            self.assertEqual(len(value['input_bindings']), 19)
            for item in value['input_bindings'].values(): self.assertEqual(set(item), {'path', 'size_bytes', 'sha256'})
            with patch('sys.stdout', new_callable=io.StringIO) as stdout:
                self.assertEqual(M.main(['check', '--contract', str(contract)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'RECOVERY_CHECK_PASSED')
            self.assertEqual(snapshot(root), before); self.assertFalse(output.exists())

    def test_real_schema_mock_roundtrip_retains_partial_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root, contract, output = fixture(folder); before = snapshot(root)
            module = M.load_validator(); original = module.validate
            with patch.object(M, 'load_validator', return_value=module), patch.object(module, 'validate', wraps=original) as called:
                result = M.run(contract)
            self.assertTrue(called.call_args.kwargs['compact_joins'])
            self.assertEqual(result['status'], M.STATUS); self.assertTrue(result['original_run_failed'])
            self.assertTrue(result['full_matrix_incomplete']); self.assertEqual(result['not_run_seeds'], [17, 27])
            self.assertEqual(result['counts'], dict(training_sessions=0, training_updates=0))
            self.assertEqual(result['original_run_counts']['training_sessions'], 8)
            self.assertFalse(result['actual_training_verified'])
            self.assertEqual(result['convergence'], 'UNVERIFIED'); self.assertEqual(result['causal_root_cause'], 'UNIDENTIFIED')
            counts = result['seed7_summary']['counts']
            self.assertEqual(counts['sampled_unique_behavior_rows'], 4)
            self.assertEqual(counts['sample_occurrences'], 304)
            self.assertEqual(counts['main_adam_calls'], 76); self.assertEqual(counts['surrogate_adam_calls'], 76)
            self.assertEqual(snapshot(root), before); self.assertTrue((output / 'result.json').exists())
            with self.assertRaises(ValueError): M.run(contract)

    def test_valid_kl_stop_with_no_loss_observation_is_not_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            _, contract, _ = fixture(folder, early_stop=True)
            result = M.run(contract)
            self.assertEqual(result['seed7_summary']['counts']['early_stop_minibatches'], 1)
            self.assertEqual(result['seed7_summary']['counts']['main_adam_calls'], 76)

    def test_tampered_bound_bytes_fail_before_any_output(self):
        for relative in ('failure.json', 'progress.json', 'seed7/off/result.json', 'seed7/on/checkpoint.pt',
                         'seed7/on/final_evidence.pt', 'seed7/on/episode1.json', 'seed7/on/audit/events.jsonl'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as folder:
                root, contract, output = fixture(folder)
                with (root / relative).open('ab') as handle: handle.write(b' ')
                with self.assertRaises(ValueError): M.run(contract)
                self.assertFalse(output.exists())

    def test_rebound_payload_or_episode_parity_cannot_pass(self):
        for kind in ('payload', 'episode'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root, contract, output = fixture(folder)
                if kind == 'payload': (root / 'seed7/on/final_evidence.pt').write_bytes(b'different')
                else:
                    path = root / 'seed7/on/episode1.json'; receipt = M.read_json(path)
                    receipt['rng_after_sha256'] = 'f' * 64; put(path, receipt)
                with self.assertRaises(ValueError): M.make_contract(root, Path(folder) / 'original_contract.json', output)

    def test_missing_or_added_old_success_or_extra_seed_prevents_recovery(self):
        for mode in ('root_result', 'on_result', 'seed17', 'no_failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root, contract, output = fixture(folder)
                if mode == 'no_failure': (root / 'failure.json').unlink()
                elif mode == 'seed17': (root / 'seed17').mkdir()
                else: put(root / ('result.json' if mode == 'root_result' else 'seed7/on/result.json'), {})
                with self.assertRaises(ValueError): M.check_contract(contract)
                self.assertFalse(output.exists())

    def test_false_header_origin_or_credit_units_fail_in_derived_root_only(self):
        for mode in ('origin', 'units', 'terminal'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root, contract, output = fixture(folder); audit = root / 'seed7/on/audit/events.jsonl'
                def mutate(rows):
                    if mode == 'origin': rows[0]['payload']['actual_training_verified'] = True
                    elif mode == 'units': next(row for row in rows if row['event_type'] == 'credit')['payload']['T_team'] = 2.
                    else: rows[-1]['payload']['status'] = 'partial'
                rewrite_audit(audit, mutate)
                value = M.make_contract(root, Path(folder) / 'original_contract.json', output)
                contract.unlink(); M.write_json(contract, value); before = snapshot(root)
                with self.assertRaises(ValueError): M.run(contract)
                self.assertEqual(snapshot(root), before)
                self.assertTrue((output / 'failure.json').exists()); self.assertFalse((output / 'result.json').exists())

    def test_schema_join_validator_is_required_even_if_summary_numbers_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root, contract, output = fixture(folder); audit = root / 'seed7/on/audit/events.jsonl'
            def mutate(rows):
                next(row for row in rows if row['event_type'] == 'minibatch')['payload']['rng_before_sha256'] = 'invalid'
            rewrite_audit(audit, mutate)
            value = M.make_contract(root, Path(folder) / 'original_contract.json', output)
            contract.unlink(); M.write_json(contract, value)
            with self.assertRaisesRegex(ValueError, 'Compact schema validation failed'): M.run(contract)
            self.assertFalse((output / 'result.json').exists())

    def test_changed_recovery_contract_during_consumer_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as folder:
            _, contract, output = fixture(folder); module = M.load_validator(); validate = module.validate
            def changed(*args, **kwargs):
                value = validate(*args, **kwargs)
                with contract.open('a') as handle: handle.write(' ')
                return value
            with patch.object(M, 'load_validator', return_value=module), patch.object(module, 'validate', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'contract changed'): M.run(contract)
            self.assertFalse((output / 'result.json').exists())

    def test_contract_bytes_are_frozen_before_preflight_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            _, contract, output = fixture(folder); original = M.check_contract
            def changed(*args, **kwargs):
                result = original(*args, **kwargs)
                with contract.open('a') as handle: handle.write(' ')
                return result
            with patch.object(M, 'check_contract', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed during preflight'): M.run(contract)
            self.assertFalse(output.exists())

    def test_line_limit_rejects_before_json_materialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root, contract, _ = fixture(folder); before = snapshot(root)
            value = M.read_json(contract)
            receipts = {str(M._path(value, name)): M.read_json(M._path(value, name))
                        for name in ('original_contract', 'off_result')}
            with patch.dict(M.LIMITS, max_line_bytes=64), patch.object(M, 'parse', side_effect=AssertionError('row parsed')):
                # Supply already-decoded receipt reads; the audit row must be
                # rejected before the parser is called on its oversized bytes.
                with patch.object(M, 'read_json', side_effect=lambda p: receipts[str(p)]):
                    with self.assertRaisesRegex(ValueError, 'Oversize/truncated audit row'):
                        M.summarize_audit(value, M.Resources())
            self.assertEqual(snapshot(root), before)

    def test_recovery_sources_and_zero_budget_frozen(self):
        for kind in ('source', 'budget', 'limits', 'path'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root, contract, output = fixture(folder); value = M.read_json(contract)
                if kind == 'source': value['recovery_sources']['entry']['sha256'] = '0' * 64
                elif kind == 'budget': value['counts']['training_sessions'] = True
                elif kind == 'limits': value['limits']['max_rss_bytes'] *= 2
                else: value['input_bindings']['off_episode1']['path'] = str(root / 'seed7/on/episode1.json')
                contract.write_text(json.dumps(value))
                with self.assertRaises(ValueError): M.check_contract(contract)
                self.assertFalse(output.exists())

    def test_failed_root_is_never_a_writable_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _ = fixture(folder)
            with self.assertRaises(ValueError): M.make_contract(root, Path(folder) / 'original_contract.json', root / 'recovery')

    def test_resource_guards_have_independent_units_and_boundaries(self):
        wall = [0.]; cpu = [0.]; rss = [1]
        guard = M.Resources(wall=lambda: wall[0], cpu=lambda: cpu[0], rss=lambda: rss[0])
        guard.check()
        for key, sequence in (('max_wall_seconds', wall), ('max_cpu_seconds', cpu), ('max_rss_bytes', rss)):
            sequence[0] = M.LIMITS[key] + 1
            with self.subTest(key=key), self.assertRaises(ValueError): guard.check()
            sequence[0] = 0 if sequence is not rss else 1
        with patch.object(M.sys, 'platform', 'darwin'), patch.object(M.resource, 'getrusage', return_value=type('R', (), {'ru_maxrss': 123})()):
            self.assertEqual(M.Resources.current_rss(), 123)
        with patch.object(M.sys, 'platform', 'linux'), patch.object(M.resource, 'getrusage', return_value=type('R', (), {'ru_maxrss': 123})()):
            self.assertEqual(M.Resources.current_rss(), 123 * 1024)


if __name__ == '__main__': unittest.main()
