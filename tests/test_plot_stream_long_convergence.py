"""Opaque-checkpoint SYNTHETIC fixtures: zero model/env/evaluation calls."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/plot_stream_long_convergence.py'
SPEC = importlib.util.spec_from_file_location('plot_stream_long_test_module', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def jsonl(path, values):
    Path(path).write_text(''.join(json.dumps(value, sort_keys=True, allow_nan=False) + '\n' for value in values))


def fixture(root):
    """Build a complete explicitly SYNTHETIC twelve-cell metadata package.

    Checkpoints are opaque text bytes, not tensors. This helper is also usable
    for isolated visual QA; callers must label its figures SYNTHETIC.
    """
    root = Path(root).resolve()
    root.mkdir(exist_ok=False)
    p = M.producer()
    source = root / 'source_assets'
    source.mkdir()
    for name in ('protocol.pt', 'predictor.pt'):
        (source / name).write_bytes(b'SYNTHETIC opaque asset; never deserialize ' + name.encode())
    train_ids = ['train_%03d' % index for index in range(72)]
    dev_ids = ['dev_%03d' % index for index in range(48)]
    probes = dev_ids[::4]
    original = dict(protocol='joint_stream_learning_v2', action_contract='joint_stream_control_v2', smoke=False,
        episodes=64, seeds=M.SEEDS, train_case_count=72, dev_case_count=48, curve_probe_ids=probes,
        assets={name: sha(source / name) for name in ('protocol.pt', 'predictor.pt')})
    dump(source / 'manifest.json', original)
    plan = dict(schema=p.SCHEMA, status='frozen', methods=M.METHODS, seeds=M.SEEDS, episodes=504, training_budget=504,
        epochs=4, update_every=4, training_case_ids=train_ids, dev_case_ids=dev_ids, probe_case_ids=probes,
        training_schedule={str(seed): train_ids * 7 for seed in M.SEEDS}, full_points=M.FULL_POINTS, probe_points=M.PROBE_POINTS,
        checkpoint_points=M.PROBE_POINTS, primary_checkpoint=504, counts=p.budget_counts(), limits=p.LIMITS,
        output_byte_count_scope=p.OUTPUT_BYTE_COUNT_SCOPE, tail_stability=p.TAIL_STABILITY, predictor_fit_calls=0,
        action_contract='joint_stream_control_v2', formal_holdout='sealed_not_read', baseline_source_commit=p.BASELINE_COMMIT,
        repo_root=str(ROOT), source_commit='a' * 40, output_root=str(root), run_root=str(root / 'run'),
        source_fingerprints=p.source_bindings(), source_protocol=dict(path=str(source / 'manifest.json'), sha256=sha(source / 'manifest.json')),
        source_assets={str(source / name): sha(source / name) for name in ('protocol.pt', 'predictor.pt')},
        probe_scope='SYNTHETIC: families=1 and capacity=4096 only')
    plan['protocol_id'] = p.protocol_identity(plan)
    protocol = root / 'manifest.json'
    dump(protocol, plan)
    protocol_sha = sha(protocol)
    diagnostic = root / 'synthetic_diagnostic.json'
    dump(diagnostic, dict(status='complete_exact_observer_parity', synthetic_fixture=True))
    review = root / 'synthetic_review.json'
    dump(review, dict(synthetic_fixture=True, not_scientific_evidence=True))
    decision = root / 'synthetic_decision.json'
    dump(decision, dict(schema=p.DECISION_SCHEMA, execution_authorized=True, decision='authorize_frozen_long_run',
        reviewer_role='root_independent_diagnostic_recovery', protocol_sha256=protocol_sha,
        protocol_id=plan['protocol_id'], source_commit=plan['source_commit'], intervention_type='bounded_exposure_budget_test',
        root_cause_status='UNIDENTIFIED', rationale='SYNTHETIC software-only fixture', algorithm_stability_fix_verified=False,
        diagnostic_result=dict(path=str(diagnostic), sha256=sha(diagnostic)), independent_review=dict(path=str(review), sha256=sha(review)),
        reviewed_at='2000-01-01', diagnostic_run_id='SYNTHETIC', validation_evidence=[dict(path=str(review), sha256=sha(review))]))
    decision_spec = dict(path=str(decision), sha256=sha(decision))
    run = root / 'run'
    run.mkdir()
    dump(run / 'run_started.json', dict(schema=p.SCHEMA, protocol_sha256=protocol_sha, source_commit=plan['source_commit'],
        decision=decision_spec, no_resume=True, no_retry=True,
        command=[str(ROOT / 'scripts/train_stream_long.py'), 'run', '--protocol', str(protocol), '--decision_receipt', str(decision)]))
    cells = []
    for method in M.METHODS:
        for seed in M.SEEDS:
            home = run / method / ('seed' + str(seed))
            home.mkdir(parents=True)
            identities, isolation, dev = [], [], []
            for episode in M.PROBE_POINTS:
                checkpoint = home / ('ep%03d.pt' % episode)
                checkpoint.write_bytes(('SYNTHETIC opaque %s %s %s' % (method, seed, episode)).encode())
                identity = dict(protocol=p.SCHEMA, protocol_id=plan['protocol_id'], protocol_sha256=protocol_sha,
                    source_commit=plan['source_commit'], method=method, seed=seed, episode=episode,
                    checkpoint_sha256=sha(checkpoint), checkpoint_actual_bytes=checkpoint.stat().st_size,
                    checkpoint_write_bytes=checkpoint.stat().st_size, checkpoint_save_calls=1,
                    checkpoint_charged_bytes=p.LIMITS['checkpoint_bytes'], output_byte_count_scope=p.OUTPUT_BYTE_COUNT_SCOPE,
                    training_budget=504, completed_updates=episode // 4, resume_supported=False, primary=episode == 504)
                dump(checkpoint.with_suffix('.identity.json'), identity)
                identities.append(identity)
                isolation.append(dict(identity, learner_before='b' * 64, learner_after='b' * 64,
                    rng_before='c' * 64, rng_after='c' * 64, rng_and_learner_unchanged=True,
                    unique_evaluation_cases=48 if episode in M.FULL_POINTS else 12))
                for index, case in enumerate(dev_ids if episode in M.FULL_POINTS else probes):
                    panels = (['full48'] if episode in M.FULL_POINTS else []) + (['probe12'] if case in probes else [])
                    dev.append(dict(protocol_id=plan['protocol_id'], method=method, seed=seed, episode=episode, case_id=case,
                        panels=panels, checkpoint_sha256=identity['checkpoint_sha256'],
                        reward=-10 + min(episode, 360) / 100 + seed / 10 + index / 100,
                        request_count=2, completed_requests=1,
                        model_cache={str(rsu): dict(transferred_bytes=1024) for rsu in range(4)}, steps=1,
                        per_request=[{'SYNTHETIC_METADATA': 'not retained'}]))
            training = [dict(method=method, seed=seed, episode=episode, case_id=train_ids[(episode - 1) % 72],
                reward=-20 + episode / 100 + seed / 10, request_count=2, completed_requests=1,
                model_cache={str(rsu): dict(transferred_bytes=1024) for rsu in range(4)}, steps=1)
                for episode in range(1, 505)]
            updates = [dict(method=method, seed=seed, episode=episode, update=episode // 4, losses={'SYNTHETIC': 0}) for episode in range(4, 505, 4)]
            jsonl(home / 'training.jsonl', training)
            jsonl(home / 'dev.jsonl', dev)
            jsonl(home / 'updates.jsonl', updates)
            jsonl(home / 'rng_isolation.jsonl', isolation)
            for name in ('final_events.jsonl.gz', 'final_receipts.jsonl.gz'):
                (home / name).write_bytes(b'SYNTHETIC opaque compressed-log bytes; hash only')
            counts = {key: 0 for key in ('learner_models', 'evaluation_models', 'predictor_models', 'training_sessions',
                'evaluation_sessions', 'environments', 'training_frames', 'evaluation_frames', 'learner_updates',
                'backward_calls', 'main_optimizer_steps', 'surrogate_optimizer_steps', 'categorical_sample_calls',
                'evaluate_calls', 'checkpoints', 'output_bytes')}
            counts.update(learner_models=1, evaluation_models=22, predictor_models=1056, training_sessions=504,
                evaluation_sessions=552, environments=1056, training_frames=504, evaluation_frames=552,
                learner_updates=126, checkpoints=22, backward_calls=126, main_optimizer_steps=126,
                surrogate_optimizer_steps=126 if method == 'sa_ghmappo' else 0, categorical_sample_calls=1, evaluate_calls=126)
            receipt = dict(schema=p.SCHEMA, status='complete_fixed504', method=method, seed=seed, counts=counts,
                checkpoints=identities, output_sha256={path.name: sha(path) for path in home.iterdir()},
                primary_checkpoint='ep504.pt', predictor_fit_calls=0, formal_holdout='not_read', no_peak_selection=True)
            dump(home / 'result.json', receipt)
            cells.append(receipt)
    total = {key: sum(cell['counts'][key] for cell in cells) for key in cells[0]['counts']}
    dump(run / 'result.json', dict(schema=p.SCHEMA, status='complete_fixed504', protocol_sha256=protocol_sha,
        protocol_id=plan['protocol_id'], decision=decision_spec, counts=total, limits=p.LIMITS, cells=cells,
        predictor_fit_calls=0, tail_stability=dict(criteria=plan['tail_stability'], audit_status='pending_independent_consumer'),
        paper_claim_ready=False, formal_holdout='not_read', synthetic_fixture=True))
    return protocol, protocol_sha


class ConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.protocol, cls.protocol_sha = fixture(Path(cls.temporary.name) / 'SYNTHETIC')
        cls.run_root = cls.protocol.parent / 'run'
        cls.home = cls.run_root / M.METHODS[0] / 'seed7'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def verify(self, *args, **kwargs):
        return M.verify(*args, allow_synthetic=True, **kwargs)

    @contextmanager
    def mutate_json(self, path, change, *, refresh_output=False):
        path = Path(path)
        original = path.read_bytes()
        receipt_path, result_path = self.home / 'result.json', self.run_root / 'result.json'
        original_receipt, original_result = receipt_path.read_bytes(), result_path.read_bytes()
        value = json.loads(original)
        change(value)
        dump(path, value)
        if refresh_output:
            receipt = json.loads(original_receipt)
            receipt['output_sha256'][path.name] = sha(path)
            dump(receipt_path, receipt)
            result = json.loads(original_result)
            result['cells'][0] = receipt
            dump(result_path, result)
        try:
            yield
        finally:
            path.write_bytes(original)
            if refresh_output:
                receipt_path.write_bytes(original_receipt)
                result_path.write_bytes(original_result)

    @contextmanager
    def mutate_rows(self, name, change):
        path = self.home / name
        original = path.read_bytes()
        receipt_path, result_path = self.home / 'result.json', self.run_root / 'result.json'
        original_receipt, original_result = receipt_path.read_bytes(), result_path.read_bytes()
        values = [json.loads(line) for line in original.splitlines()]
        change(values)
        jsonl(path, values)
        receipt = json.loads(original_receipt)
        receipt['output_sha256'][name] = sha(path)
        dump(receipt_path, receipt)
        result = json.loads(original_result)
        result['cells'][0] = receipt
        dump(result_path, result)
        try:
            yield
        finally:
            path.write_bytes(original)
            receipt_path.write_bytes(original_receipt)
            result_path.write_bytes(original_result)

    def test_top_level_and_check_are_stdlib_no_tensor_deserialization(self):
        names = []
        tree = ast.parse(SCRIPT.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'matplotlib', 'src') for name in names))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in ('load', 'from_checkpoint', 'step_controls') for node in ast.walk(tree)))

    def test_complete_twelve_cell_fixture_is_streamed_and_opaque(self):
        plan, dev, train, report = self.verify(self.protocol, self.protocol_sha)
        self.assertEqual(len(dev), 12)
        self.assertEqual(sum(len(values) for values in dev.values()), 6624)
        self.assertEqual(sum(len(values) for values in train.values()), 6048)
        self.assertNotIn('per_request', next(iter(dev.values()))[(0, plan['dev_case_ids'][0])])
        point = report['panels']['full48']['sa_ghmappo']['overall'][0]
        self.assertAlmostEqual(point['seed_reward_sample_sd'], 1.)
        self.assertAlmostEqual(point['request_completion_percent'], 50.)
        self.assertAlmostEqual(point['model_kib_per_completed_request'], 4.)
        self.assertAlmostEqual(report['final_minus_initial']['sa_ghmappo']['overall']['final504_minus_initial0_reward'], 3.6)
        self.assertTrue(report['tail_stability']['all_methods_and_seeds_passed'])
        self.assertFalse(report['paper_claim_ready'])
        self.assertEqual(report['status'], 'SYNTHETIC_SOFTWARE_PREVIEW')
        self.assertEqual(report['evidence_level'], 'SOFTWARE_SYNTHETIC_NOT_SCIENTIFIC')
        self.assertTrue(report['synthetic_fixture'])

    def test_known_synthetic_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            M.verify(self.protocol, self.protocol_sha)

    def test_cli_check_uses_completed_result_sha_when_external_sha_omitted(self):
        with patch('builtins.print'):
            self.assertEqual(M.main(['check', '--protocol', str(self.protocol), '--allow_synthetic']), 0)

    def test_manifest_and_source_or_asset_sha_tampering_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(self.protocol, '0' * 64)
        source = self.protocol.parent / 'source_assets' / 'predictor.pt'
        original = source.read_bytes()
        try:
            source.write_bytes(original + b'tamper')
            with self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            source.write_bytes(original)

    def test_frozen_source_inventory_and_source_digest_are_verified(self):
        for remove in (True, False):
            def change(value):
                name = next(iter(value['source_fingerprints']))
                if remove:
                    value['source_fingerprints'].pop(name)
                else:
                    value['source_fingerprints'][name] = '0' * 64
                value['protocol_id'] = M.producer().protocol_identity(value)
            with self.subTest(remove=remove), self.mutate_json(self.protocol, change), self.assertRaisesRegex(ValueError, '[Ss]ource'):
                self.verify(self.protocol, sha(self.protocol))

    def test_root_failure_partial_missing_duplicate_cells_and_wrong_provenance_rejected(self):
        failure = self.run_root / 'failure.json'
        dump(failure, {'status': 'failed'})
        try:
            with self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            failure.unlink()
        for change in (
            lambda value: value.update(status='running'),
            lambda value: value['cells'].pop(),
            lambda value: value['cells'].__setitem__(1, value['cells'][0]),
            lambda value: value.update(protocol_sha256='0' * 64),
            lambda value: value['counts'].update(training_sessions=6047),
            lambda value: value.update(paper_claim_ready=True),
        ):
            with self.subTest(change=change), self.mutate_json(self.run_root / 'result.json', change), self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)

    def test_cell_result_mismatch_unknown_file_and_opaque_checkpoint_tampering_rejected(self):
        with self.mutate_json(self.home / 'result.json', lambda value: value.update(status='failed')), self.assertRaises(ValueError):
            self.verify(self.protocol, self.protocol_sha)
        unknown = self.home / 'extra.json'
        unknown.write_text('{}')
        try:
            with self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            unknown.unlink()
        checkpoint = self.home / 'ep504.pt'
        original = checkpoint.read_bytes()
        try:
            checkpoint.write_bytes(b'TAMPER')
            with self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            checkpoint.write_bytes(original)

    def test_cell_file_caps_and_symlink_refused_before_hash_or_parse(self):
        checkpoint = self.home / 'ep504.pt'
        original = checkpoint.read_bytes()
        try:
            with checkpoint.open('wb') as handle:
                handle.truncate(M.producer().LIMITS['checkpoint_bytes'] + 1)
            with self.assertRaisesRegex(ValueError, 'byte cap'):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            checkpoint.write_bytes(original)
        target = self.home / 'ep480.pt'
        try:
            checkpoint.unlink()
            checkpoint.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'non-symlink'):
                self.verify(self.protocol, self.protocol_sha)
        finally:
            checkpoint.unlink()
            checkpoint.write_bytes(original)

    def test_dev_rows_duplicate_missing_wrong_panels_foreign_seed_or_checkpoint_rejected(self):
        changes = (
            lambda values: values.__setitem__(1, values[0]),
            lambda values: values.pop(),
            lambda values: values[0].update(panels=['probe12']),
            lambda values: values[0].update(seed=17),
            lambda values: values[0].update(checkpoint_sha256='0' * 64),
            lambda values: values[48].update(case_id='dev_001'),
            lambda values: values[0].update(episode=True),
            lambda values: values[0].update(completed_requests=3),
            lambda values: values[0].update(request_count=0),
            lambda values: values[0]['model_cache'].pop('0'),
        )
        for change in changes:
            with self.subTest(change=change), self.mutate_rows('dev.jsonl', change), self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)

    def test_training_schedule_duplicate_missing_nonfinite_and_update_identity_rejected(self):
        for name, change in (
            ('training.jsonl', lambda values: values[0].update(case_id='train_071')),
            ('training.jsonl', lambda values: values.__setitem__(1, values[0])),
            ('training.jsonl', lambda values: values.pop()),
            ('training.jsonl', lambda values: values[0].update(steps=481)),
            ('updates.jsonl', lambda values: values[0].update(update=2)),
            ('updates.jsonl', lambda values: values.__setitem__(1, values[0])),
            ('updates.jsonl', lambda values: values.pop()),
        ):
            with self.subTest(name=name, change=change), self.mutate_rows(name, change), self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)

    def test_rng_isolation_every_point_and_checkpoint_identity_rejected(self):
        for change in (
            lambda values: values[0].update(rng_after='d' * 64),
            lambda values: values[0].update(learner_after='d' * 64),
            lambda values: values[0].update(rng_and_learner_unchanged=False),
            lambda values: values[0].update(unique_evaluation_cases=12),
            lambda values: values.__setitem__(1, values[0]),
            lambda values: values[0].update(completed_updates=1),
        ):
            with self.subTest(change=change), self.mutate_rows('rng_isolation.jsonl', change), self.assertRaises(ValueError):
                self.verify(self.protocol, self.protocol_sha)
        with self.mutate_json(self.home / 'ep504.identity.json', lambda value: value.update(primary=False), refresh_output=True), self.assertRaises(ValueError):
            self.verify(self.protocol, self.protocol_sha)

    def test_bound_command_and_decision_receipts_rejected(self):
        with self.mutate_json(self.run_root / 'run_started.json', lambda value: value.update(no_retry=False)), self.assertRaises(ValueError):
            self.verify(self.protocol, self.protocol_sha)
        decision = self.protocol.parent / 'synthetic_decision.json'
        with self.mutate_json(decision, lambda value: value.update(execution_authorized=False)), self.assertRaises(ValueError):
            self.verify(self.protocol, self.protocol_sha)
        for index, value in ((0, '/tmp/foreign.py'), (1, 'prepare'), (3, '/tmp/foreign_manifest.json'), (5, '/tmp/foreign_decision.json')):
            with self.subTest(index=index), self.mutate_json(self.run_root / 'run_started.json', lambda row: row['command'].__setitem__(index, value)), self.assertRaisesRegex(ValueError, '[Cc]ommand'):
                self.verify(self.protocol, self.protocol_sha)

    def test_inputs_cannot_change_between_parsing_and_final_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'receipt.json'
            dump(path, {'value': 1})
            budget = M.Budget()
            self.assertEqual(M.read_json(path, budget), {'value': 1})
            dump(path, {'value': 2})
            with self.assertRaisesRegex(ValueError, 'changed'):
                budget.unchanged()

    def test_known_nested_synthetic_markers_never_silently_pass(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rows.jsonl'
            jsonl(path, [dict(nested=[dict(synthetic_fixture=True)])])
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                list(M.rows(path, M.Budget(), 1, 1024))
            values = list(M.rows(path, M.Budget(allow_synthetic=True), 1, 1024))
            self.assertEqual(len(values), 1)

    def test_stream_line_byte_json_duplicate_key_nonfinite_and_count_guards(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rows.jsonl'
            for wire in (b'{}', b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b' ' * (M.MAX_LINE + 1) + b'\n'):
                path.write_bytes(wire)
                with self.subTest(wire=wire[:20]), self.assertRaises(ValueError):
                    list(M.rows(path, M.Budget(), 1, 2 * M.MAX_LINE))
            path.write_bytes(b'{}\n{}\n')
            with self.assertRaises(ValueError):
                list(M.rows(path, M.Budget(), 1, 100))
            with self.assertRaises(ValueError):
                list(M.rows(path, M.Budget(), 3, 100))
            with self.assertRaises(ValueError):
                list(M.rows(path, M.Budget(), 2, 1))

    def test_budget_hard_bounds_and_fresh_derived_root(self):
        with self.assertRaises(RuntimeError):
            M.Budget(rss=lambda: M.MAX_RSS + 1).check()
        times = iter([0., M.MAX_SECONDS + 1.])
        with self.assertRaises(RuntimeError):
            M.Budget(clock=lambda: next(times), rss=lambda: 1).check()
        with self.assertRaises(ValueError):
            M.render(self.protocol, self.protocol_sha, self.protocol.parent)


class StatisticsTests(unittest.TestCase):
    def test_pooled_cost_and_completion_are_ratios_of_sums(self):
        values = [dict(reward=-10, request_count=1, completed_requests=0, transferred_bytes=2048),
            dict(reward=-20, request_count=9, completed_requests=3, transferred_bytes=1024)]
        result = M.aggregate(values)
        self.assertEqual(result['reward'], -15)
        self.assertEqual(result['request_completion_percent'], 30)
        self.assertEqual(result['model_kib_per_completed_request'], 1)

    def test_zero_denominators_negative_rewards_and_low_plateau_not_learning_success(self):
        self.assertIsNone(M.relative_spread([0, 0, 0]))
        self.assertAlmostEqual(M.relative_spread([-100, -99, -101]), .02)
        criteria = M.producer().TAIL_STABILITY
        points = {episode: dict(reward=-100, request_completion_percent=0, model_kib_per_completed_request=None) for episode in criteria['episodes']}
        self.assertFalse(M.tail_check(points, criteria)['passed'])
        for value in points.values():
            value['model_kib_per_completed_request'] = 0
        self.assertFalse(M.tail_check(points, criteria)['passed'])
        for value in points.values():
            value['model_kib_per_completed_request'] = 4
        self.assertTrue(M.tail_check(points, criteria)['passed'])

    def test_tail_thresholds_report_every_failure_without_smoothing(self):
        criteria = M.producer().TAIL_STABILITY
        points = {episode: dict(reward=100, request_completion_percent=80, model_kib_per_completed_request=4) for episode in criteria['episodes']}
        for key, value, check in (('reward', 104, 'reward'), ('request_completion_percent', 82, 'completion'), ('model_kib_per_completed_request', 4.2, 'model_cost')):
            modified = copy.deepcopy(points)
            modified[504][key] = value
            result = M.tail_check(modified, criteria)
            self.assertFalse(result['passed'])
            self.assertFalse(result['checks'][check])

    def test_causal_trailing24_keeps_real_downward_rewards_no_padding(self):
        values = {episode: dict(reward=float(episode)) for episode in range(1, 26)}
        values[25]['reward'] = -100
        result = list(M.trailing(values))
        self.assertEqual([episode for episode, _ in result], [24, 25])
        self.assertEqual(result[0][1], 12.5)
        self.assertLess(result[1][1], result[0][1])


if __name__ == '__main__':
    unittest.main()
