"""Cold-process CPU startup/checkpoint qualification; no data or training.

Each cell enters the production Runtime before its first optimizer construction.
A separate process supplies the same-seed unguarded initialization reference.
Outputs are exclusive software evidence, never performance/convergence evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'stream_cold_startup_qualification_v1'
METHODS = ('sa_ghmappo', 'mappo', 'ppo', 'dag_offload_drl')
SEEDS = (7, 17, 27)
ZERO_COUNTS = dict(environment_calls=0, forward_calls=0, backward_calls=0,
                   adam_calls=0, autograd_grad_calls=0, scientific_training_sessions=0)
THREAD_ENV = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
CHILD_TIMEOUT_SECONDS = 600
MAX_CHILD_DIAGNOSTIC_CHARS = 8192


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def get(path):
    return json.loads(Path(path).read_text())


def source_identity():
    paths = sorted((ROOT / 'src').rglob('*.py')) + [
        ROOT / 'scripts/train_stream_long.py',
        ROOT / 'scripts/train_joint_stream_control.py',
        ROOT / 'scripts/sa_stage5c_numerical_qualification.py',
        ROOT / 'scripts/diagnose_stream_stability.py', Path(__file__).resolve()]
    return dict(source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_status=subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines(),
        source_fingerprints={str(path.relative_to(ROOT)): sha(path) for path in paths})


def cold_dynamo_check():
    if any(name == 'torch._dynamo' or name.startswith('torch._dynamo.') for name in sys.modules):
        raise RuntimeError('Cold qualification refuses a warm torch._dynamo import')


def versions(torch, np):
    return dict(python=sys.version, executable=sys.executable,
                torch=torch.__version__, numpy=np.__version__)


def executable_identity():
    path = Path(sys.executable)
    return dict(path=str(path), resolved_path=str(path.resolve()), sha256=sha(path.resolve()))


def guard_apis(torch, scheduled):
    return dict(cuda_lazy_init=torch.cuda._lazy_init, tensor_cuda=torch.Tensor.cuda,
                module_cuda=torch.nn.Module.cuda, autograd_grad=torch.autograd.grad,
                tensor_backward=torch.Tensor.backward, adam_step=torch.optim.Adam.step,
                categorical_sample=torch.distributions.Categorical.sample,
                evaluate=scheduled._evaluate)


def same_identities(left, right):
    if set(left) != set(right) or any(left[key] is not right[key] for key in left):
        raise AssertionError('Patched API identity changed or was not restored')


def check_guard(torch, before, installed, scheduled):
    current = guard_apis(torch, scheduled)
    same_identities(installed, current)
    if any(current[key] is before[key] for key in before):
        raise AssertionError('Production Runtime guard was not installed')
    names = ('cuda_lazy_init', 'tensor_cuda', 'module_cuda', 'autograd_grad')
    if len({id(current[key]) for key in names}) != len(names):
        raise AssertionError('Forbidden APIs require mutually distinct guard callables')
    for key, args in (('cuda_lazy_init', ()), ('tensor_cuda', (None,)),
                      ('module_cuda', (None,)), ('autograd_grad', (None, None))):
        try:
            current[key](*args)
        except RuntimeError as exc:
            if 'forbidden' not in str(exc).lower():
                raise AssertionError('Guard did not reject before the original: ' + key) from exc
        else:
            raise AssertionError('Forbidden operation was accepted: ' + key)


def check_fresh(agent):
    if (agent.training_budget != 504 or int(agent.completed_updates) != 0
            or agent.optimizer.state or agent.surrogate_optimizer.state):
        raise AssertionError('Fresh 504-budget learner with two empty Adam states required')
    if any(param.device.type != 'cpu' for param in agent.parameters()):
        raise AssertionError('CPU parameters required')


def qualify_child(args):
    """One process owns one cold cell or one independent no-Runtime reference."""
    home = args.output_root.resolve()
    home.mkdir(parents=False, exist_ok=False)
    source = source_identity()
    put(home / 'run_started.json', dict(schema=SCHEMA, started_at=utc(), pid=os.getpid(),
        mode=args.mode, method=args.method, seed=args.seed, source=source, command=sys.argv))
    counts = dict(ZERO_COUNTS, learner_instances=0, evaluation_copy_instances=0,
                  checkpoints=0)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        from scripts.train_joint_stream_control import make_agent
        from scripts.train_stream_long import (Runtime, Budget, save_checkpoint_bounded,
                                                preserve_rng, learner_digest)
        from scripts.diagnose_stream_stability import exact_digest
        from scripts.sa_stage5c_numerical_qualification import (
            Patches, learned_state, rng_state, compare_exact)
        from src.agents.ppo_agent import ScheduledControlPPO
        from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cold_dynamo_check()
        originals = guard_apis(torch, ScheduledControlPPO)
        extra_originals = dict(module_call_impl=torch.nn.Module._call_impl,
                               environment_init=MultiRequestStreamEnv.__init__, torch_save=torch.save)

        def deny(label):
            def forbidden(*unused_args, **unused_kwargs):
                raise RuntimeError('Startup qualification forbids ' + label)
            return forbidden

        with Patches() as patches:
            patches.set(torch.nn.Module, '_call_impl', deny('forward'))
            patches.set(MultiRequestStreamEnv, '__init__', deny('environment'))
            # The reference has no production Runtime; these qualification-only
            # guards still prevent accidentally running forward or optimization.
            if args.mode == '_reference':
                patches.set(torch.Tensor, 'backward', deny('backward'))
                patches.set(torch.optim.Adam, 'step', deny('optimizer'))
                patches.set(torch.autograd, 'grad', deny('autograd.grad'))
                random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                agent = make_agent(args.method, args.seed, 504)
                counts['learner_instances'] += 1
                check_fresh(agent)
                evidence = dict(initial_learner_sha256=learner_digest(agent),
                    initial_learned_state_sha256=exact_digest(learned_state(agent)),
                    initial_rng_sha256=exact_digest(rng_state(torch, np)))
            else:
                budget = Budget(home)
                with Runtime(budget, torch) as runtime:
                    installed = guard_apis(torch, ScheduledControlPPO)
                    # This assertion is immediately before the first real Adam
                    # construction; never warm-import Dynamo to hide the bug.
                    cold_dynamo_check()
                    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                    budget.charge('learner_models')
                    agent = make_agent(args.method, args.seed, 504)
                    counts['learner_instances'] += 1
                    runtime.learner = agent
                    check_fresh(agent)
                    check_guard(torch, originals, installed, ScheduledControlPPO)
                    evidence = dict(initial_learner_sha256=learner_digest(agent),
                        initial_learned_state_sha256=exact_digest(learned_state(agent)),
                        initial_rng_sha256=exact_digest(rng_state(torch, np)))
                    with preserve_rng(torch, np):
                        checkpoint = home / 'initial.pt'
                        saved = save_checkpoint_bounded(agent, checkpoint, budget, torch)
                        counts['checkpoints'] += 1
                        budget.charge('evaluation_models')
                        loaded = type(agent).from_checkpoint(checkpoint)
                        counts['evaluation_copy_instances'] += 1
                        check_fresh(loaded)
                        if loaded.prediction_enabled != agent.prediction_enabled:
                            raise AssertionError('Loaded prediction option differs')
                        compare_exact(learned_state(agent), learned_state(loaded), 'roundtrip')
                        check_guard(torch, originals, installed, ScheduledControlPPO)
                        evidence.update(checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint),
                            actual_bytes=saved['actual_bytes'], write_bytes=saved['write_bytes'],
                            save_calls=saved['save_calls']), loaded_matches=True, loaded_options_match=True)
                        del loaded
                    if (learner_digest(agent) != evidence['initial_learner_sha256']
                            or exact_digest(rng_state(torch, np)) != evidence['initial_rng_sha256']):
                        raise AssertionError('Checkpoint/copy disturbed learner or Python/NumPy/Torch RNG')
                    check_guard(torch, originals, installed, ScheduledControlPPO)
                    evidence.update(learner_unchanged=True, three_rng_restored=True,
                        forbidden_guard_callables_unique=True, production_guard_stable=True,
                        budget_counts=dict(budget.counts), peak_rss_bytes=budget.peak_rss,
                        cold_dynamo_absent_before_runtime=True,
                        cold_dynamo_absent_before_first_agent=True)
                same_identities(originals, guard_apis(torch, ScheduledControlPPO))
                evidence['production_api_identities_restored'] = True
        same_identities(originals, guard_apis(torch, ScheduledControlPPO))
        same_identities(extra_originals, dict(module_call_impl=torch.nn.Module._call_impl,
            environment_init=MultiRequestStreamEnv.__init__, torch_save=torch.save))
        evidence['qualification_api_identities_restored'] = True
        if source_identity() != source:
            raise RuntimeError('Source identity changed during qualification')
        result = dict(schema=SCHEMA, status='pass', completed_at=utc(), mode=args.mode,
            method=args.method, seed=args.seed, source=source, counts=counts,
            source_commit=source['source_commit'], source_fingerprints=source['source_fingerprints'],
            runtime_executable=executable_identity(),
            versions=versions(torch, np), cpu_threads=torch.get_num_threads(),
            cpu_interop_threads=torch.get_num_interop_threads(), evidence=evidence,
            scope='Startup/initial checkpoint interfaces only; no trained optimizer or performance coverage')
        put(home / 'result.json', result)
        print(json.dumps(dict(status='pass', mode=args.mode, method=args.method, seed=args.seed)))
        return result
    except BaseException as exc:
        put(home / 'failure.json', dict(schema=SCHEMA, status='failed', failed_at=utc(),
            mode=args.mode, method=args.method, seed=args.seed, counts=counts,
            exception=repr(exc), traceback=traceback.format_exc(), no_retry=True))
        raise


def child_command(mode, method, seed, home):
    if mode not in ('_cold', '_reference') or method not in METHODS or seed not in SEEDS:
        raise ValueError('Unfrozen qualification cell')
    return [sys.executable, str(Path(__file__).resolve()), mode, '--output-root', str(home),
            '--method', method, '--seed', str(seed)]


def launch_child(mode, method, seed, home):
    command = child_command(mode, method, seed, home)
    env = dict(os.environ, **THREAD_ENV)
    # No retry, shell, model/data input, or inherited long-run process.
    subprocess.run(command, cwd=ROOT, env=env, check=True,
        timeout=CHILD_TIMEOUT_SECONDS, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return get(home / 'result.json')


def validate_pair(cold, reference, method, seed):
    for row, mode in ((cold, '_cold'), (reference, '_reference')):
        if (row.get('schema') != SCHEMA or row.get('status') != 'pass'
                or row.get('mode') != mode or row.get('method') != method or row.get('seed') != seed):
            raise ValueError('Child qualification identity/status differs')
        expected = dict(ZERO_COUNTS, learner_instances=1,
                        evaluation_copy_instances=int(mode == '_cold'), checkpoints=int(mode == '_cold'))
        if row.get('counts') != expected or row.get('cpu_threads') != 1 or row.get('cpu_interop_threads') != 1:
            raise ValueError('Child operation/thread counts differ')
    if cold['source'] != reference['source'] or cold['versions'] != reference['versions']:
        raise ValueError('Reference source/runtime differs')
    for key in ('initial_learner_sha256', 'initial_learned_state_sha256', 'initial_rng_sha256'):
        if cold['evidence'][key] != reference['evidence'][key]:
            raise ValueError('Guarded initialization differs from independent no-Runtime reference: ' + key)
    evidence = cold['evidence']
    for key in ('loaded_matches', 'learner_unchanged', 'three_rng_restored',
                'loaded_options_match',
                'forbidden_guard_callables_unique', 'production_guard_stable',
                'production_api_identities_restored', 'cold_dynamo_absent_before_runtime',
                'cold_dynamo_absent_before_first_agent'):
        if evidence.get(key) is not True:
            raise ValueError('Incomplete cold evidence: ' + key)
    checkpoint = evidence['checkpoint']
    path = Path(checkpoint['path'])
    if (checkpoint['save_calls'] < 1 or path.stat().st_size != checkpoint['actual_bytes']
            or sha(path) != checkpoint['sha256'] or checkpoint['actual_bytes'] > 16 * 1024 ** 2
            or checkpoint['write_bytes'] > 16 * 1024 ** 2):
        raise ValueError('Bounded checkpoint evidence differs')
    budget = evidence.get('budget_counts', {})
    nonzero = {key: value for key, value in budget.items() if value}
    if nonzero != dict(learner_models=1, evaluation_models=1, checkpoints=1,
                       output_bytes=16 * 1024 ** 2):
        raise ValueError('Production budget contains unexpected numerical operations')


def run_matrix(args, *, launch=None):
    home = args.output_root.resolve()
    home.mkdir(parents=False, exist_ok=False)
    source = source_identity()
    put(home / 'run_started.json', dict(schema=SCHEMA, started_at=utc(), pid=os.getpid(),
        methods=METHODS, seeds=SEEDS, source=source, command=sys.argv, no_retry=True))
    rows = []
    try:
        for method in METHODS:
            for seed in SEEDS:
                stem = method + '_seed' + str(seed)
                cold_home = home / (stem + '_cold')
                reference_home = home / (stem + '_reference')
                cold = (launch or launch_child)('_cold', method, seed, cold_home)
                reference = (launch or launch_child)('_reference', method, seed, reference_home)
                validate_pair(cold, reference, method, seed)
                if cold['source'] != source:
                    raise RuntimeError('Matrix source identity changed')
                rows.append(dict(method=method, seed=seed, status='pass',
                    cold_result=dict(path=str(cold_home / 'result.json'), sha256=sha(cold_home / 'result.json')),
                    reference_result=dict(path=str(reference_home / 'result.json'), sha256=sha(reference_home / 'result.json')),
                    checkpoint=cold['evidence']['checkpoint'],
                    initial_learner_sha256=cold['evidence']['initial_learner_sha256'],
                    initial_rng_sha256=cold['evidence']['initial_rng_sha256']))
        if source_identity() != source:
            raise RuntimeError('Matrix source identity changed')
        result = dict(schema=SCHEMA, status='pass', completed_at=utc(), source=source,
            source_commit=source['source_commit'], source_fingerprints=source['source_fingerprints'],
            runtime_executable=executable_identity(),
            methods=list(METHODS), seeds=list(SEEDS), training_budget=504, cells=rows,
            counts=dict(ZERO_COUNTS, cold_learner_instances=12, evaluation_copy_instances=12,
                        independent_reference_instances=12, cold_processes=12, reference_processes=12,
                        checkpoints=12),
            scope='Twelve fresh cold initializations/save/load roundtrips, independently matched; no rollout, trained optimizer, learning or convergence claim')
        put(home / 'result.json', result)
        print(json.dumps(dict(status='pass', cells=len(rows), result=str(home / 'result.json'))))
        return result
    except BaseException as exc:
        failure = dict(schema=SCHEMA, status='failed', failed_at=utc(),
            completed_cells=len(rows), exception=repr(exc), traceback=traceback.format_exc(), no_retry=True)
        if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            for key in ('stdout', 'stderr'):
                value = getattr(exc, key, None)
                if isinstance(value, bytes):
                    value = value.decode(errors='replace')
                failure['child_' + key + '_tail'] = (value or '')[-MAX_CHILD_DIAGNOSTIC_CHARS:]
        put(home / 'failure.json', failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output-root', type=Path, required=True)
    for mode in ('_cold', '_reference'):
        child = sub.add_parser(mode)
        child.add_argument('--output-root', type=Path, required=True)
        child.add_argument('--method', choices=METHODS, required=True)
        child.add_argument('--seed', choices=SEEDS, type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'run':
        return run_matrix(args)
    return qualify_child(args)


if __name__ == '__main__':
    main()
