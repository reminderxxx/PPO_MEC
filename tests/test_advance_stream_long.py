"""Pure conditional handoff tests; all Git, scientific gates, and spawn calls mocked."""
import ast
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/advance_stream_long.py'
SPEC = importlib.util.spec_from_file_location('advance_four_pure', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


class Fixture:
    def __init__(self, base):
        base = base.resolve()
        self.repo = base / 'runtime'
        self.repo.mkdir()
        self.binary = base / 'python_binary_target'
        self.binary.write_bytes(b'pure mock interpreter, never executed')
        self.binary.chmod(0o755)
        self.python = self.repo / '.venv/bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(self.binary)
        self.artifacts = base / 'artifacts'
        self.artifacts.mkdir()
        self.commit = 'a' * 40
        self.paths = {}
        for name in M.COMPONENTS[:-1]:
            self.paths[name] = self.artifacts / (name + '.json')
            value = {'output_root': str(self.artifacts / 'remaining')} if name == 'remaining_contract' else {'name': name}
            write(self.paths[name], value)
        self.paths['remaining_result'] = self.artifacts / 'remaining/result.json'
        self.summary = self.artifacts / 'composite/summary.json'
        self.review = write(self.artifacts / 'independent_policy_review.json', {'status': 'conditional_policy_review'})
        self.plan = dict(schema='joint_stream_long_four_v1', source_commit=self.commit,
                         methods=list(M.METHODS), seeds=list(M.SEEDS), episodes=504,
                         protocol_id='frozen_four_504', run_root=str(self.artifacts / 'long/run'))
        self.protocol = write(self.artifacts / 'long/manifest.json', self.plan)
        self.handoff = self.artifacts / 'handoff'
        self.authorization = dict(schema=M.SCHEMA, execution_authorized=True,
            authorization_type='conditional_after_complete_diagnostic', authorized_by='root_user_instruction',
            user_instruction='诊断通过后直接启动这四个方法的长训，不用再问我', authorized_at='2026-10-22T10:00:00Z',
            diagnostic_run_id='frozen_diagnostic', rationale='Bounded longer exposure after complete observer parity.',
            methods=list(M.METHODS), seeds=list(M.SEEDS), episodes=504, runtime_repo=str(self.repo),
            runtime_commit=self.commit, python_executable=str(self.python),
            python_runtime=dict(resolved_path=str(self.binary), sha256=M.sha_file(self.binary)),
            protocol=M.binding(self.protocol), independent_review=M.binding(self.review),
            diagnostic_result={'path': str(self.summary)}, handoff_root=str(self.handoff),
            diagnostic_components={name: ({'path': str(path)} if name == 'remaining_result' else M.binding(path))
                                   for name, path in self.paths.items()})
        self.path = self.artifacts / 'authorization.json'
        self.save()

    def save(self):
        write(self.path, self.authorization)

    def complete(self):
        write(self.paths['remaining_result'], {'status': 'complete_unexecuted_seed_observer_parity'})
        write(self.summary, {'status': M.COMPOSITE_STATUS})

    def qualify(self):
        """Opaque byte/metadata mock only; no interpreter or model executes."""
        scripts = ('train_stream_long.py', 'train_joint_stream_control.py',
                   'sa_stage5c_numerical_qualification.py', 'diagnose_stream_stability.py',
                   'qualify_stream_startup.py')
        for name in scripts:
            path = self.repo / 'scripts' / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('# software mock source ' + name + '\n')
        scientific = self.repo / 'src/example.py'
        scientific.parent.mkdir()
        scientific.write_text('# software mock algorithm source\n')
        fingerprints = {str(path.relative_to(self.repo)): M.sha_file(path)
                        for path in [scientific, *(self.repo / 'scripts' / name for name in scripts)]}
        source = dict(source_commit=self.commit, dirty_status=[], source_fingerprints=fingerprints)
        executable = dict(path=str(self.python), **self.authorization['python_runtime'])
        zero = dict(environment_calls=0, forward_calls=0, backward_calls=0,
                    adam_calls=0, autograd_grad_calls=0, scientific_training_sessions=0)
        home = self.artifacts / 'startup'
        home.mkdir()
        cells = []
        for method in M.METHODS:
            for seed in M.SEEDS:
                stem = method + '_seed' + str(seed)
                cold = home / (stem + '_cold')
                cold.mkdir()
                checkpoint = cold / 'initial.pt'
                checkpoint.write_bytes(b'opaque software mock checkpoint; never deserialize')
                saved = dict(M.binding(checkpoint), actual_bytes=checkpoint.stat().st_size,
                             write_bytes=checkpoint.stat().st_size, save_calls=1)
                state = dict(initial_learner_sha256='1' * 64,
                             initial_learned_state_sha256='2' * 64, initial_rng_sha256='3' * 64,
                             qualification_api_identities_restored=True)
                budget = dict.fromkeys(('learner_models', 'evaluation_models', 'predictor_models',
                    'training_sessions', 'evaluation_sessions', 'environments', 'training_frames',
                    'evaluation_frames', 'learner_updates', 'backward_calls', 'main_optimizer_steps',
                    'surrogate_optimizer_steps', 'categorical_sample_calls', 'evaluate_calls',
                    'checkpoints', 'output_bytes'), 0)
                budget.update(learner_models=1, evaluation_models=1, checkpoints=1, output_bytes=16 * 1024 ** 2)
                cold_evidence = dict(state, checkpoint=saved, budget_counts=budget, peak_rss_bytes=1024,
                    **dict.fromkeys(('loaded_matches', 'loaded_options_match', 'learner_unchanged',
                        'three_rng_restored', 'forbidden_guard_callables_unique', 'production_guard_stable',
                        'production_api_identities_restored', 'cold_dynamo_absent_before_runtime',
                        'cold_dynamo_absent_before_first_agent'), True))
                for mode in ('_cold', '_reference'):
                    folder = home / (stem + mode)
                    folder.mkdir(exist_ok=True)
                    row = dict(schema=M.STARTUP_SCHEMA, status='pass', mode=mode, method=method,
                        seed=seed, source_commit=self.commit, source=source, source_fingerprints=fingerprints,
                        runtime_executable=executable, cpu_threads=1, cpu_interop_threads=1,
                        versions=dict(python='mock Python', executable=str(self.python), torch='mock Torch', numpy='mock NumPy'),
                        counts=dict(zero, learner_instances=1, evaluation_copy_instances=int(mode == '_cold'),
                                    checkpoints=int(mode == '_cold')),
                        evidence=cold_evidence if mode == '_cold' else state)
                    write(folder / 'result.json', row)
                cells.append(dict(method=method, seed=seed, status='pass',
                    cold_result=M.binding(cold / 'result.json'),
                    reference_result=M.binding(home / (stem + '_reference') / 'result.json'),
                    checkpoint=saved, initial_learner_sha256=state['initial_learner_sha256'],
                    initial_rng_sha256=state['initial_rng_sha256']))
        self.startup = dict(schema=M.STARTUP_SCHEMA, status='pass', methods=list(M.METHODS),
            seeds=list(M.SEEDS), training_budget=504, source_commit=self.commit, source=source,
            source_fingerprints=fingerprints, runtime_executable=executable, cells=cells,
            counts=dict(zero, cold_learner_instances=12, evaluation_copy_instances=12,
                independent_reference_instances=12, cold_processes=12, reference_processes=12, checkpoints=12))
        self.startup_path = home / 'result.json'
        self.save_startup()

    def save_startup(self):
        write(self.startup_path, self.startup)
        self.authorization['startup_qualification'] = M.binding(self.startup_path)
        self.save()

    def mock_gate(self):
        gate = SimpleNamespace()
        gate.check_protocol = Mock(return_value=(self.plan, self.protocol, {}))
        gate.check_decision = Mock(side_effect=lambda path, protocol, plan: (M.binding(path), M.read_json(path)))
        return gate


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(M, 'ROOT', self.f.repo))
        self.stack.enter_context(patch.object(M, 'current_commit', return_value=self.f.commit))
        self.spawn = self.stack.enter_context(patch.object(M.subprocess, 'Popen', return_value=SimpleNamespace(pid=4321)))
        self.loader = self.stack.enter_context(patch.object(M, 'load_training_gate', side_effect=AssertionError('Unexpected gate import')))

    def allow_gate(self):
        gate = self.f.mock_gate()
        self.loader.side_effect = None
        self.loader.return_value = gate
        return gate

    def test_module_has_only_standard_library_top_level_imports(self):
        imports = []
        for node in ast.parse(SCRIPT.read_text()).body:
            if isinstance(node, ast.Import):
                imports.extend(item.name.split('.')[0] for item in node.names)
            if isinstance(node, ast.ImportFrom):
                imports.append((node.module or '').split('.')[0])
        self.assertFalse(set(imports) & {'torch', 'numpy', 'src', 'scripts'})

    def test_incomplete_check_and_run_wait_without_claim_gate_or_spawn(self):
        for launch in (False, True):
            result = M.advance(self.f.path, launch=launch)
            self.assertEqual(result['status'], 'waiting_no_launch')
            self.assertFalse(result['claimed'])
            self.assertFalse(self.f.handoff.exists())
        self.loader.assert_not_called()
        self.spawn.assert_not_called()

    def test_summary_alone_does_not_replace_remaining_result(self):
        write(self.f.summary, {'status': M.COMPOSITE_STATUS})
        result = M.advance(self.f.path, launch=True)
        self.assertEqual(result['missing'], [str(self.f.paths['remaining_result'])])
        self.assertFalse(self.f.handoff.exists())
        self.spawn.assert_not_called()

    def test_complete_check_still_does_not_write_decision_or_import_gate(self):
        self.f.complete()
        result = M.advance(self.f.path)
        self.assertEqual(result['status'], 'complete_inputs_present_no_launch')
        self.assertFalse(self.f.handoff.exists())
        self.loader.assert_not_called()
        self.spawn.assert_not_called()

    def test_launch_binds_all_current_components_and_calls_both_gates(self):
        self.f.complete()
        gate = self.allow_gate()
        result = M.advance(self.f.path, launch=True)
        self.assertEqual(result['status'], 'launched_pending_child_verification')
        self.assertEqual(result['pid'], 4321)
        self.assertFalse(result['child_run_started_verified'])
        gate.check_protocol.assert_called_once_with(str(self.f.protocol))
        gate.check_decision.assert_called_once()
        decision = M.read_json(self.f.handoff / 'decision_receipt.json')
        self.assertEqual(set(decision['diagnostic_components']), set(M.COMPONENTS))
        for name, path in self.f.paths.items():
            self.assertEqual(decision['diagnostic_components'][name], M.binding(path))
        self.assertEqual(decision['diagnostic_result'], M.binding(self.f.summary))
        self.assertEqual(decision['conditional_authorization'], M.binding(self.f.path))
        self.assertEqual(decision['intervention_type'], 'bounded_exposure_budget_test')
        self.assertEqual(decision['root_cause_status'], 'UNIDENTIFIED')
        self.assertIs(decision['algorithm_stability_fix_verified'], False)
        self.assertEqual(decision['decision_origin'], 'root_conditional_authorization_after_complete_diagnostic')
        self.assertEqual(M.read_json(self.f.handoff / 'launch.json')['methods'], M.METHODS)

    def test_spawn_command_is_fixed_detached_and_threads_are_bounded(self):
        self.f.complete()
        self.allow_gate()
        M.advance(self.f.path, launch=True)
        args, kwargs = self.spawn.call_args
        self.assertEqual(args[0], [str(self.f.python), '-u', str(self.f.repo / 'scripts/train_stream_long.py'),
                                  'run', '--protocol', str(self.f.protocol), '--decision_receipt',
                                  str(self.f.handoff / 'decision_receipt.json')])
        self.assertTrue(kwargs['start_new_session'])
        self.assertTrue(kwargs['close_fds'])
        self.assertEqual(kwargs['stdin'], M.subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], M.subprocess.STDOUT)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
            self.assertEqual(kwargs['env'][name], '1')
        self.assertEqual(kwargs['stdout'].name, str(self.f.handoff / 'long_training.log'))

    def test_existing_handoff_or_scientific_claim_refuses_duplicate(self):
        self.f.complete()
        self.allow_gate()
        M.advance(self.f.path, launch=True)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            M.advance(self.f.path, launch=True)
        self.spawn.assert_called_once()
        self.f.handoff.rename(self.f.artifacts / 'saved_handoff')
        Path(self.f.plan['run_root']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Scientific run already claimed'):
            M.advance(self.f.path, launch=True)
        self.spawn.assert_called_once()

    def test_remaining_failure_is_terminal_and_retains_failure_without_spawn(self):
        write(self.f.artifacts / 'remaining/failure.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'Remaining diagnostic failed'):
            M.advance(self.f.path, launch=True)
        failure = M.read_json(self.f.handoff / 'failure.json')
        self.assertEqual(failure['status'], 'failed_no_automatic_retry')
        self.assertIsNone(failure['child_pid'])
        self.loader.assert_not_called()
        self.spawn.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            M.advance(self.f.path, launch=True)

    def test_bad_completed_evidence_cannot_spawn_and_failure_is_preserved(self):
        self.f.complete()
        write(self.f.summary, {'status': 'incomplete'})
        with self.assertRaisesRegex(ValueError, 'Composite diagnostic'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'failure.json').is_file())
        self.spawn.assert_not_called()

    def test_gate_rejection_preserves_decision_and_failure_without_spawn(self):
        self.f.complete()
        gate = self.allow_gate()
        gate.check_decision.side_effect = ValueError('tampered raw diagnostic evidence')
        with self.assertRaisesRegex(ValueError, 'tampered raw'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'decision_receipt.json').is_file())
        self.assertTrue((self.f.handoff / 'failure.json').is_file())
        self.assertFalse((self.f.handoff / 'launch_intent.json').exists())
        self.spawn.assert_not_called()

    def test_spawn_exception_is_terminal_and_never_retried(self):
        self.f.complete()
        self.allow_gate()
        self.spawn.side_effect = OSError('mock spawn denied')
        with self.assertRaisesRegex(OSError, 'mock spawn denied'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'launch_intent.json').is_file())
        self.assertIsNone(M.read_json(self.f.handoff / 'failure.json')['child_pid'])
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            M.advance(self.f.path, launch=True)
        self.spawn.assert_called_once()

    def test_post_spawn_receipt_failure_records_pid_and_prohibits_relaunch(self):
        self.f.complete()
        self.allow_gate()
        original = M.write_exclusive
        def fail_launch(path, value):
            if Path(path).name == 'launch.json':
                raise OSError('mock receipt disk failure')
            return original(path, value)
        with patch.object(M, 'write_exclusive', side_effect=fail_launch):
            with self.assertRaisesRegex(OSError, 'mock receipt disk failure'):
                M.advance(self.f.path, launch=True)
        self.assertEqual(M.read_json(self.f.handoff / 'failure.json')['child_pid'], 4321)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            M.advance(self.f.path, launch=True)
        self.spawn.assert_called_once()

    def test_authorization_commit_protocol_and_existing_components_are_frozen(self):
        mutations = ('commit', 'protocol', 'component', 'methods', 'future_sha', 'rationale')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as location:
                    f = Fixture(Path(location))
                    if mutation == 'commit': f.authorization['runtime_commit'] = 'b' * 40
                    if mutation == 'protocol': write(f.protocol, dict(f.plan, episodes=505))
                    if mutation == 'component': write(f.paths['seed7_recovery_result'], {'changed': True})
                    if mutation == 'methods': f.authorization['methods'].append('cache_offload_drl')
                    if mutation == 'future_sha': f.authorization['diagnostic_result']['sha256'] = '0' * 64
                    if mutation == 'rationale': f.authorization['rationale'] = ''
                    f.save()
                    with patch.object(M, 'ROOT', f.repo), self.assertRaises(ValueError):
                        M.advance(f.path, launch=True)
                    self.assertFalse(f.handoff.exists())
        self.spawn.assert_not_called()

    def test_changed_authorization_after_gate_does_not_spawn(self):
        self.f.complete()
        gate = self.allow_gate()
        original = gate.check_decision.side_effect
        def mutate(path, protocol, plan):
            result = original(path, protocol, plan)
            self.f.authorization['rationale'] = 'changed after review'
            self.f.save()
            return result
        gate.check_decision.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'changed before launch'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'failure.json').exists())
        self.spawn.assert_not_called()

    def test_authorization_changed_during_initial_checks_is_rejected_without_claim(self):
        original = M.current_commit
        def mutate(repo):
            self.f.authorization['rationale'] = 'changed during initial binding checks'
            self.f.save()
            return original(repo)
        with patch.object(M, 'current_commit', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during checks'):
                M.advance(self.f.path, launch=True)
        self.assertFalse(self.f.handoff.exists())
        self.loader.assert_not_called()
        self.spawn.assert_not_called()

    def test_python_binary_changed_before_authorization_check_is_rejected(self):
        self.f.binary.write_bytes(b'replaced executable target')
        with self.assertRaisesRegex(ValueError, 'Python binary target/SHA differs'):
            M.advance(self.f.path, launch=True)
        self.assertFalse(self.f.handoff.exists())
        self.spawn.assert_not_called()

    def test_python_target_swap_after_gate_is_terminal_without_spawn(self):
        self.f.complete()
        gate = self.allow_gate()
        original = gate.check_decision.side_effect
        def mutate(path, protocol, plan):
            result = original(path, protocol, plan)
            target = self.f.binary.with_name('replacement_python')
            target.write_bytes(b'different interpreter')
            target.chmod(0o755)
            self.f.python.unlink()
            self.f.python.symlink_to(target)
            return result
        gate.check_decision.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'Python binary target/SHA differs'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'failure.json').exists())
        self.spawn.assert_not_called()

    def test_duplicate_json_and_nonfinite_authorization_are_rejected(self):
        for raw in ('{"schema":"a","schema":"b"}', '{"episodes":NaN}'):
            self.f.path.write_text(raw)
            with self.assertRaises(ValueError):
                M.advance(self.f.path, launch=True)
        self.spawn.assert_not_called()

    def test_complete_cold_matrix_is_required_bound_and_carried_into_decision(self):
        self.f.qualify()
        self.f.complete()
        self.allow_gate()
        M.advance(self.f.path, launch=True)
        decision = M.read_json(self.f.handoff / 'decision_receipt.json')
        self.assertEqual(decision['startup_qualification'], self.f.authorization['startup_qualification'])
        evidence = decision['validation_evidence']
        self.assertIn(decision['startup_qualification'], evidence)
        self.assertIn(self.f.startup['cells'][0]['cold_result'], evidence)
        self.assertIn({key: self.f.startup['cells'][0]['checkpoint'][key] for key in ('path', 'sha256')}, evidence)
        self.spawn.assert_called_once()

    def test_incomplete_foreign_nonzero_or_unverified_cold_qualification_rejected(self):
        for mode in ('missing_cell', 'source', 'science', 'guard', 'rng', 'checkpoint_sha'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as location:
                f = Fixture(Path(location))
                f.qualify()
                if mode == 'missing_cell':
                    f.startup['cells'].pop()
                elif mode == 'source':
                    f.startup['source_commit'] = 'b' * 40
                elif mode == 'science':
                    f.startup['counts']['forward_calls'] = 1
                elif mode == 'checkpoint_sha':
                    Path(f.startup['cells'][0]['checkpoint']['path']).write_bytes(b'tampered')
                else:
                    spec = f.startup['cells'][0]['cold_result']
                    child = M.read_json(spec['path'])
                    if mode == 'guard':
                        child['evidence']['loaded_options_match'] = False
                    else:
                        child['evidence']['initial_rng_sha256'] = '4' * 64
                    write(Path(spec['path']), child)
                    f.startup['cells'][0]['cold_result'] = M.binding(spec['path'])
                f.save_startup()
                with patch.object(M, 'ROOT', f.repo), self.assertRaises(ValueError):
                    M.advance(f.path, launch=True)
                self.assertFalse(f.handoff.exists())
        self.spawn.assert_not_called()

    def test_cold_evidence_changed_after_gate_cannot_spawn(self):
        self.f.qualify()
        self.f.complete()
        gate = self.allow_gate()
        original = gate.check_decision.side_effect
        def mutate(path, protocol, plan):
            value = original(path, protocol, plan)
            cold_path = Path(self.f.startup['cells'][0]['cold_result']['path'])
            cold_path.write_text('{}')
            return value
        gate.check_decision.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            M.advance(self.f.path, launch=True)
        self.assertTrue((self.f.handoff / 'failure.json').exists())
        self.spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
