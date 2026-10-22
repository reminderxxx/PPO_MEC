"""One-shot conditional handoff to the frozen four-method development run.

Only stdlib metadata is consumed here. Incomplete diagnostic evidence never
claims an output directory, imports numerical packages, or starts a process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'stream_four_long_conditional_authorization_v1'
DECISION_SCHEMA = 'stream_long_execution_decision_v1'
COMPOSITE_STATUS = 'COMPOSITE_SEED7_RECOVERED_SEEDS17_27_EXACT_OBSERVER_PARITY'
METHODS = ['sa_ghmappo', 'mappo', 'ppo', 'dag_offload_drl']
SEEDS = [7, 17, 27]
COMPONENTS = ('original_contract', 'original_failure', 'seed7_recovery_contract',
              'seed7_recovery_result', 'remaining_contract', 'remaining_result')
MAX_METADATA_BYTES = 32 * 1024 ** 2
MAX_PYTHON_BINARY_BYTES = 128 * 1024 ** 2


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_METADATA_BYTES, 'Oversized metadata')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical_path(value):
    require(type(value) is str and value, 'Absolute canonical path required')
    path = Path(value)
    require(path.is_absolute() and str(path.resolve()) == value, 'Absolute canonical path required')
    return path


def sha_file(path, *, max_bytes=MAX_METADATA_BYTES):
    path = Path(path)
    before = path.stat()
    require(path.is_file() and before.st_size <= max_bytes, 'Bounded file required')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed while hashing')
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha_file(path))


def check_spec(spec, *, future=False):
    require(type(spec) is dict and set(spec) == ({'path'} if future else {'path', 'sha256'}),
            'Exact path/SHA binding required; future evidence binds path only')
    path = canonical_path(spec['path'])
    if not future:
        require(type(spec['sha256']) is str and len(spec['sha256']) == 64 and
                sha_file(path) == spec['sha256'], 'Frozen evidence SHA differs')
    return path


def current_commit(repo):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def check_python_runtime(authorization):
    # Preserve the venv invocation path while binding its actual binary target.
    value = authorization['python_executable']
    require(type(value) is str, 'Explicit executable Python path required')
    python = Path(value)
    require(python.is_absolute() and str(python.absolute()) == value and
            python.is_file() and os.access(python, os.X_OK), 'Explicit executable Python path required')
    runtime = authorization['python_runtime']
    require(type(runtime) is dict and set(runtime) == {'resolved_path', 'sha256'}, 'Frozen Python binary binding required')
    target = canonical_path(runtime['resolved_path'])
    require(python.resolve() == target and type(runtime['sha256']) is str and len(runtime['sha256']) == 64 and
            sha_file(target, max_bytes=MAX_PYTHON_BINARY_BYTES) == runtime['sha256'], 'Python binary target/SHA differs')
    return python


def check_authorization(path):
    path = Path(path).resolve()
    initial_binding = binding(path)
    authorization = read_json(path)
    expected = dict(schema=SCHEMA, execution_authorized=True,
                    authorization_type='conditional_after_complete_diagnostic',
                    authorized_by='root_user_instruction', methods=METHODS, seeds=SEEDS, episodes=504)
    for key, value in expected.items():
        require(type(authorization.get(key)) is type(value) and authorization[key] == value,
                'Conditional authorization mismatch: ' + key)
    for key in ('user_instruction', 'authorized_at', 'diagnostic_run_id', 'rationale'):
        require(type(authorization.get(key)) is str and authorization[key].strip(),
                'Nonempty conditional authorization field required: ' + key)
    repo = canonical_path(authorization['runtime_repo'])
    require(repo == ROOT, 'Conditional runtime must be this frozen checkout')
    commit = authorization['runtime_commit']
    require(type(commit) is str and len(commit) == 40 and
            all(letter in '0123456789abcdef' for letter in commit), 'Full frozen commit required')
    require(current_commit(repo) == commit, 'Conditional runtime HEAD differs')
    check_python_runtime(authorization)
    protocol = check_spec(authorization['protocol'])
    review = check_spec(authorization['independent_review'])
    summary = check_spec(authorization['diagnostic_result'], future=True)
    components = authorization['diagnostic_components']
    require(type(components) is dict and set(components) == set(COMPONENTS), 'All six diagnostic component paths required')
    for name in COMPONENTS:
        check_spec(components[name], future=name == 'remaining_result')
    plan = read_json(protocol)
    require(plan.get('schema') == 'joint_stream_long_four_v1' and plan.get('methods') == METHODS and
            plan.get('seeds') == SEEDS and plan.get('episodes') == 504 and
            plan.get('source_commit') == commit, 'Only the frozen four-method504 plan is authorized')
    run_root = canonical_path(plan['run_root'])
    handoff = canonical_path(authorization['handoff_root'])
    require(handoff.parent.is_dir() and 'artifacts' in handoff.parts, 'Existing private artifact parent required')
    inputs = [path, protocol, review, summary, repo, run_root]
    inputs += [Path(spec['path']) for spec in components.values()]
    for protected in inputs:
        require(handoff != protected and handoff not in protected.parents and protected not in handoff.parents,
                'Handoff root must be separate from runtime, evidence, and scientific run')
    require(not handoff.exists(), 'Handoff already claimed; no overwrite/resume/retry')
    require(not run_root.exists(), 'Scientific run already claimed; no overwrite/resume/retry')
    require(binding(path) == initial_binding, 'Conditional authorization changed during checks')
    return authorization, plan, initial_binding


def inspect_completion(authorization):
    components = authorization['diagnostic_components']
    remaining = read_json(components['remaining_contract']['path'])
    remaining_root = canonical_path(remaining['output_root'])
    require(Path(components['remaining_result']['path']) == remaining_root / 'result.json',
            'Remaining result must belong to the frozen remaining contract')
    require(not (remaining_root / 'failure.json').exists(), 'Remaining diagnostic failed; automatic launch prohibited')
    summary = Path(authorization['diagnostic_result']['path'])
    missing = [str(path) for path in (Path(components['remaining_result']['path']), summary) if not path.is_file()]
    if missing:
        return dict(status='waiting_no_launch', missing=missing, claimed=False, launched=False)
    result = read_json(components['remaining_result']['path'])
    require(result.get('status') == 'complete_unexecuted_seed_observer_parity',
            'Remaining diagnostic did not complete exact observer parity')
    composite = read_json(summary)
    require(composite.get('status') == COMPOSITE_STATUS, 'Composite diagnostic did not complete required parity')
    return dict(status='complete_inputs_present_no_launch', claimed=False, launched=False)


def write_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def load_training_gate():
    spec = importlib.util.spec_from_file_location('conditional_four_long_gate', ROOT / 'scripts/train_stream_long.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_decision(authorization, plan, authorization_binding):
    components = {name: binding(authorization['diagnostic_components'][name]['path']) for name in COMPONENTS}
    diagnostic = binding(authorization['diagnostic_result']['path'])
    review = dict(authorization['independent_review'])
    return dict(schema=DECISION_SCHEMA, execution_authorized=True,
                decision='authorize_frozen_long_run', reviewer_role='root_independent_diagnostic_recovery',
                protocol_sha256=authorization['protocol']['sha256'], protocol_id=plan['protocol_id'],
                source_commit=authorization['runtime_commit'], intervention_type='bounded_exposure_budget_test',
                root_cause_status='UNIDENTIFIED', algorithm_stability_fix_verified=False,
                rationale=authorization['rationale'], diagnostic_result=diagnostic, independent_review=review,
                diagnostic_components=components, reviewed_at=utc(),
                diagnostic_run_id=authorization['diagnostic_run_id'],
                validation_evidence=list(components.values()) + [diagnostic, review, authorization_binding],
                conditional_authorization=authorization_binding,
                decision_origin='root_conditional_authorization_after_complete_diagnostic',
                evidence_boundary='Observer parity permits the authorized bounded exposure test; no verified algorithmic stability fix or convergence guarantee.')


def advance(authorization_path, *, launch=False):
    authorization, plan, authorization_binding = check_authorization(authorization_path)
    home = Path(authorization['handoff_root'])
    claimed = False
    child_pid = None
    try:
        state = inspect_completion(authorization)
        if state['status'] == 'waiting_no_launch' or not launch:
            return state
        home.mkdir(exist_ok=False)
        claimed = True
        write_exclusive(home / 'handoff_started.json', dict(schema=SCHEMA, started_at=utc(), pid=os.getpid(),
                        conditional_authorization=authorization_binding, no_resume=True, no_retry=True))
        decision = make_decision(authorization, plan, authorization_binding)
        decision_path = home / 'decision_receipt.json'
        write_exclusive(decision_path, decision)
        gate = load_training_gate()
        checked_plan, protocol_path, _ = gate.check_protocol(authorization['protocol']['path'])
        require(checked_plan == plan, 'Protocol changed between conditional checks')
        decision_spec, _ = gate.check_decision(decision_path, protocol_path, checked_plan)
        # A final bounded metadata recheck precedes process creation.
        require(binding(authorization_path) == authorization_binding and
                binding(protocol_path) == authorization['protocol'] and
                binding(authorization['independent_review']['path']) == authorization['independent_review'],
                'Conditional authorization/protocol/review changed before launch')
        require(not Path(plan['run_root']).exists(), 'Scientific run was claimed before launch')
        check_python_runtime(authorization)
        command = [authorization['python_executable'], '-u', str(ROOT / 'scripts/train_stream_long.py'),
                   'run', '--protocol', str(protocol_path), '--decision_receipt', str(decision_path)]
        environment = dict(os.environ)
        environment.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        log = home / 'long_training.log'
        intent = dict(schema=SCHEMA, status='launch_intent', created_at=utc(), command=command,
                      cwd=str(ROOT), stdout=str(log), decision=decision_spec,
                      conditional_authorization=authorization_binding, no_resume=True, no_retry=True)
        write_exclusive(home / 'launch_intent.json', intent)
        with log.open('xb') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
            child_pid = child.pid
        receipt = dict(intent, status='launched_pending_child_verification', launched_at=utc(), pid=child_pid,
                       child_run_started_verified=False, methods=METHODS, seeds=SEEDS, episodes=504,
                       sleep_guard='not_started_by_this_handoff')
        write_exclusive(home / 'launch.json', receipt)
        return receipt
    except BaseException as exc:
        # A failed handoff is terminal, including a lost post-spawn receipt.
        # Never issue a second Popen after any failed check or publication.
        if launch:
            if not claimed:
                home.mkdir(exist_ok=False)
                claimed = True
            failure = dict(schema=SCHEMA, status='failed_no_automatic_retry', failed_at=utc(),
                           exception_type=type(exc).__name__, message=str(exc)[:2048],
                           traceback=traceback.format_exc()[:32768], child_pid=child_pid,
                           conditional_authorization=authorization_binding, no_resume=True, no_retry=True)
            write_exclusive(home / 'failure.json', failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'run'):
        child = sub.add_parser(command)
        child.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args(argv)
    result = advance(args.authorization, launch=args.command == 'run')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
