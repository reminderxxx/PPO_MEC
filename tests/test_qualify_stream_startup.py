"""Pure orchestration/evidence regressions; the CLI owns real cold processes."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/qualify_stream_startup.py'
SPEC = importlib.util.spec_from_file_location('qualify_startup_test_helpers', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture_source():
    return dict(source_commit='a' * 40, dirty_status=[],
                source_fingerprints={'scripts/train_stream_long.py': 'b' * 64,
                                     'scripts/qualify_stream_startup.py': 'c' * 64})


def fixture_child(mode, method, seed, home):
    home.mkdir()
    evidence = dict(initial_learner_sha256='d' * 64,
                    initial_learned_state_sha256='e' * 64, initial_rng_sha256='f' * 64)
    if mode == '_cold':
        checkpoint = home / 'initial.pt'
        checkpoint.write_bytes(b'checkpoint-fixture')
        evidence.update({key: True for key in ('loaded_matches', 'learner_unchanged',
            'loaded_options_match',
            'three_rng_restored', 'forbidden_guard_callables_unique', 'production_guard_stable',
            'production_api_identities_restored', 'cold_dynamo_absent_before_runtime',
            'cold_dynamo_absent_before_first_agent')})
        evidence['checkpoint'] = dict(path=str(checkpoint), sha256=M.sha(checkpoint),
            actual_bytes=checkpoint.stat().st_size, write_bytes=checkpoint.stat().st_size, save_calls=1)
        evidence['budget_counts'] = dict(learner_models=1, evaluation_models=1,
            checkpoints=1, output_bytes=16 * 1024 ** 2, forward_calls=0)
    row = dict(schema=M.SCHEMA, status='pass', mode=mode, method=method, seed=seed,
        source=fixture_source(), counts=dict(M.ZERO_COUNTS, learner_instances=1,
        evaluation_copy_instances=int(mode == '_cold'), checkpoints=int(mode == '_cold')),
        versions=dict(torch='fixture'), cpu_threads=1, cpu_interop_threads=1, evidence=evidence)
    M.put(home / 'result.json', row)
    return row


def test_top_level_stdlib_and_no_dynamo_warm_import():
    tree = ast.parse(SCRIPT.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(value.name.split('.')[0] in ('torch', 'numpy', 'src', 'scripts') for value in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in ('torch', 'numpy', 'src', 'scripts')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [value.name for value in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not any('_dynamo' in name for name in names)


def test_commands_frozen_to_methods_and_seeds(tmp_path):
    assert len(M.METHODS) == 4 and M.SEEDS == (7, 17, 27)
    command = M.child_command('_cold', 'mappo', 17, tmp_path / 'cell')
    assert command[0] == sys.executable
    assert command[2] == '_cold'
    assert command[-4:] == ['--method', 'mappo', '--seed', '17']
    for mode, method, seed in (('_retry', 'ppo', 7), ('_cold', 'dqn', 7), ('_cold', 'ppo', 8)):
        with pytest.raises(ValueError):
            M.child_command(mode, method, seed, tmp_path)


def test_child_launch_is_bounded_separate_process_without_shell(tmp_path):
    home = tmp_path / 'cell'
    home.mkdir()
    M.put(home / 'result.json', {'status': 'fixture'})
    with patch.object(M.subprocess, 'run') as run:
        assert M.launch_child('_cold', 'ppo', 7, home) == {'status': 'fixture'}
    args, kwargs = run.call_args
    assert args[0][2] == '_cold'
    assert kwargs['timeout'] == 600 and kwargs['check'] is True
    assert not kwargs.get('shell')
    assert kwargs['env']['OMP_NUM_THREADS'] == '1'
    assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'


def test_matrix_runs_twelve_cold_then_independent_reference_processes(tmp_path):
    calls = []
    def launch(mode, method, seed, home):
        calls.append((mode, method, seed))
        return fixture_child(mode, method, seed, home)
    with patch.object(M, 'source_identity', return_value=fixture_source()):
        result = M.run_matrix(types.SimpleNamespace(output_root=tmp_path / 'fresh'), launch=launch)
    expected = [(mode, method, seed) for method in M.METHODS for seed in M.SEEDS
                for mode in ('_cold', '_reference')]
    assert calls == expected and len(result['cells']) == 12
    assert result['counts']['cold_processes'] == 12
    assert result['counts']['independent_reference_instances'] == 12
    assert all(result['counts'][key] == 0 for key in M.ZERO_COUNTS)
    assert result['source_commit'] == fixture_source()['source_commit']
    assert result['source_fingerprints'] == fixture_source()['source_fingerprints']
    assert len(result['runtime_executable']['sha256']) == 64
    assert not (tmp_path / 'fresh/failure.json').exists()
    for row in result['cells']:
        for key in ('cold_result', 'reference_result', 'checkpoint'):
            assert M.sha(row[key]['path']) == row[key]['sha256']


def test_existing_output_rejected_before_any_launch(tmp_path):
    calls = []
    with pytest.raises(FileExistsError):
        M.run_matrix(types.SimpleNamespace(output_root=tmp_path), launch=lambda *args: calls.append(args))
    assert not calls and not (tmp_path / 'run_started.json').exists()


def test_child_failure_stops_without_retry_or_other_cells(tmp_path):
    calls = []
    def launch(mode, method, seed, home):
        calls.append((mode, method, seed))
        if mode == '_reference':
            raise RuntimeError('fixture failure')
        return fixture_child(mode, method, seed, home)
    home = tmp_path / 'failure'
    with patch.object(M, 'source_identity', return_value=fixture_source()):
        with pytest.raises(RuntimeError, match='fixture failure'):
            M.run_matrix(types.SimpleNamespace(output_root=home), launch=launch)
    assert calls == [('_cold', 'sa_ghmappo', 7), ('_reference', 'sa_ghmappo', 7)]
    failure = M.get(home / 'failure.json')
    assert failure['completed_cells'] == 0 and failure['no_retry'] is True
    assert not (home / 'result.json').exists()


def test_child_stderr_is_preserved_with_a_hard_tail_cap(tmp_path):
    def launch(*args):
        raise M.subprocess.CalledProcessError(1, ['fixture'],
            output='x' * 20000, stderr='y' * 20000 + 'meaningful error')
    home = tmp_path / 'failure'
    with patch.object(M, 'source_identity', return_value=fixture_source()):
        with pytest.raises(M.subprocess.CalledProcessError):
            M.run_matrix(types.SimpleNamespace(output_root=home), launch=launch)
    failure = M.get(home / 'failure.json')
    assert len(failure['child_stderr_tail']) == M.MAX_CHILD_DIAGNOSTIC_CHARS
    assert len(failure['child_stdout_tail']) == M.MAX_CHILD_DIAGNOSTIC_CHARS
    assert failure['child_stderr_tail'].endswith('meaningful error')


@pytest.mark.parametrize('defect', ['wrong_method', 'nonzero_steps', 'rng_mismatch', 'warm_import',
                                  'guard_missing', 'loaded_option', 'checkpoint_changed', 'budget_forward', 'threads'])
def test_pair_rejects_incomplete_or_inconsistent_evidence(tmp_path, defect):
    cold = fixture_child('_cold', 'ppo', 7, tmp_path / 'cold')
    reference = fixture_child('_reference', 'ppo', 7, tmp_path / 'reference')
    if defect == 'wrong_method': cold['method'] = 'dqn'
    elif defect == 'nonzero_steps': cold['counts']['adam_calls'] = 1
    elif defect == 'rng_mismatch': reference['evidence']['initial_rng_sha256'] = '0' * 64
    elif defect == 'warm_import': cold['evidence']['cold_dynamo_absent_before_first_agent'] = False
    elif defect == 'guard_missing': cold['evidence']['production_guard_stable'] = False
    elif defect == 'loaded_option': cold['evidence']['loaded_options_match'] = False
    elif defect == 'checkpoint_changed': Path(cold['evidence']['checkpoint']['path']).write_bytes(b'changed')
    elif defect == 'budget_forward': cold['evidence']['budget_counts']['forward_calls'] = 1
    elif defect == 'threads': reference['cpu_threads'] = 2
    with pytest.raises(ValueError):
        M.validate_pair(cold, reference, 'ppo', 7)


def test_guard_identity_check_requires_exact_restoration():
    first = lambda: None
    second = lambda: None
    M.same_identities({'a': first}, {'a': first})
    with pytest.raises(AssertionError): M.same_identities({'a': first}, {'a': second})
    with pytest.raises(AssertionError): M.same_identities({'a': first}, {'b': first})


@pytest.mark.parametrize('defect', [None, 'aliased', 'accepts'])
def test_guard_probes_require_distinct_stable_callable_rejections(defect):
    def denial():
        def forbidden(*args):
            raise RuntimeError('Operation forbidden in fixture')
        return forbidden
    t = types.SimpleNamespace(cuda=types.SimpleNamespace(_lazy_init=denial()),
        Tensor=types.SimpleNamespace(cuda=denial(), backward=denial()),
        nn=types.SimpleNamespace(Module=types.SimpleNamespace(cuda=denial())),
        autograd=types.SimpleNamespace(grad=denial()),
        optim=types.SimpleNamespace(Adam=types.SimpleNamespace(step=denial())),
        distributions=types.SimpleNamespace(Categorical=types.SimpleNamespace(sample=denial())))
    scheduled = types.SimpleNamespace(_evaluate=denial())
    before = {key: denial() for key in M.guard_apis(t, scheduled)}
    if defect == 'aliased': t.Tensor.cuda = t.cuda._lazy_init
    if defect == 'accepts': t.autograd.grad = lambda *args: None
    installed = M.guard_apis(t, scheduled)
    if defect is None:
        M.check_guard(t, before, installed, scheduled)
    else:
        with pytest.raises(AssertionError): M.check_guard(t, before, installed, scheduled)


def test_warm_dynamo_check_fails_without_importing_it():
    with patch.dict(sys.modules, {'torch._dynamo': types.ModuleType('torch._dynamo')}):
        with pytest.raises(RuntimeError, match='warm'):
            M.cold_dynamo_check()


def test_production_runtime_scope_contains_cold_assertion_and_first_agent():
    tree = ast.parse(SCRIPT.read_text())
    scopes = [node for node in ast.walk(tree) if isinstance(node, ast.With)
              and any(isinstance(item.context_expr, ast.Call)
              and isinstance(item.context_expr.func, ast.Name)
              and item.context_expr.func.id == 'Runtime' for item in node.items)]
    assert len(scopes) == 1
    calls = [node for node in ast.walk(scopes[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    cold_lines = [node.lineno for node in calls if node.func.id == 'cold_dynamo_check']
    make_lines = [node.lineno for node in calls if node.func.id == 'make_agent']
    assert len(cold_lines) == len(make_lines) == 1 and cold_lines[0] < make_lines[0]
