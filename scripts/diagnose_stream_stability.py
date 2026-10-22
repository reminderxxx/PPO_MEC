"""Exclusive, bounded observer parity on fresh stochastic stream learning.

``check`` uses only the standard library. ``run`` executes the frozen first four
training sessions for each seed, off then on, with one original update per side.
It changes no policy, reward, schedule or learning rule, and reads no holdout.
An audit chain remains unverified until independently checked against this run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import struct
import subprocess
import sys
import time
import traceback

SCHEMA = 'stream_stability_diagnostic_v1'
ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE_COMMIT = '4bc39b95ca55bf14b72625fbbe1e59b7ac509702'
GIB = 1024 ** 3
FIXED_LIMITS = dict(training_sessions=24, training_updates=6,
    frames_per_session=480, training_frames=11520, models=6, environments=24,
    audit_rows_per_seed=100000, audit_bytes_per_seed=6 * GIB,
    minimum_free_disk_bytes=30 * GIB, max_rss_bytes=8 * GIB,
    max_wall_seconds=7200, backward_calls=720, adam_calls=1440,
    categorical_sample_calls=276480, evaluate_calls=185040)
REQUIRED_SCRIPTS = ('scripts/diagnose_stream_stability.py',
    'scripts/train_joint_stream_control.py', 'scripts/sa_stage5c_credit_observer.py',
    'scripts/sa_stage5c_credit_audit.py', 'scripts/sa_stage5c_numerical_qualification.py',
    'scripts/train_joint_tail_control.py', 'scripts/launch_eight_vehicle_joint_experiment.py')
EXCLUDED_UID_KEY = 'credit_audit_uids'


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path, value):
    # Receipts are never silently overwritten or used as a resume marker.
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def exact_digest(value, *, exclude_audit_uids=False):
    """Type/order/shape/dtype/device/bit preserving, without cloning a rollout.

    Only the observer-added UID key may be omitted. Dict order and signed zero
    remain observable; no approximate numeric equality or fallback repr is used.
    Tensor/NumPy support is imported or accessed only when such a value exists.
    """
    result = hashlib.sha256()
    ancestors = set()
    def put(data):
        result.update(struct.pack('!Q', len(data)))
        result.update(data)
    def visit(item):
        typ = type(item)
        put((typ.__module__ + '.' + typ.__qualname__).encode())
        if item is None:
            return
        if typ is bool:
            put(b'1' if item else b'0'); return
        if typ is int:
            put(str(item).encode()); return
        if typ is float:
            if not math.isfinite(item):
                raise ValueError('Nonfinite digest scalar')
            put(struct.pack('!d', item)); return
        if typ is str:
            put(item.encode()); return
        if typ is bytes:
            put(item); return
        if hasattr(item, 'detach') and hasattr(item, 'cpu'):
            import torch
            if not bool(item.isfinite().all()):
                raise ValueError('Nonfinite digest tensor')
            visit((str(item.dtype), tuple(item.shape), str(item.device)))
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            put(raw.numpy().tobytes()); return
        if typ.__module__.startswith('numpy'):
            visit((str(item.dtype), tuple(item.shape)))
            visit(item.tolist())
            put(item.tobytes()); return
        if dataclasses.is_dataclass(item):
            visit(dataclasses.asdict(item)); return
        if isinstance(item, (dict, list, tuple, set, frozenset)):
            if id(item) in ancestors:
                raise ValueError('Cyclic digest container')
            ancestors.add(id(item))
            try:
                if isinstance(item, dict):
                    keys = [key for key in item if not (exclude_audit_uids and key == EXCLUDED_UID_KEY)]
                    visit(len(keys))
                    for key in keys:
                        visit(key); visit(item[key])
                elif isinstance(item, (set, frozenset)):
                    visit(sorted(exact_digest(child, exclude_audit_uids=exclude_audit_uids) for child in item))
                else:
                    visit(len(item))
                    for child in item:
                        visit(child)
            finally:
                ancestors.remove(id(item))
            return
        raise TypeError('Unsupported exact digest type: ' + typ.__name__)
    visit(value)
    return result.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def compare_episode_receipt(observed, archived, compare):
    """Compare original exact digests, not JSON-reordered container key order."""
    for field in ('episode', 'case_id', 'summary_sha256', 'rolls_sha256',
                  'receipts_sha256', 'rng_after_sha256'):
        compare(observed[field], archived[field], 'episode.' + field)


def contract_shape(contract):
    required = ('schema', 'status', 'repo_root', 'output_root', 'source_commit', 'baseline_source_commit',
        'source_fingerprints', 'protocol', 'seeds', 'sides', 'sessions_per_seed',
        'training_budget', 'epochs', 'limits')
    if any(key not in contract for key in required):
        raise ValueError('Incomplete diagnostic contract')
    if contract['schema'] != SCHEMA or contract['status'] != 'frozen':
        raise ValueError('Frozen diagnostic schema required')
    expected = dict(seeds=[7, 17, 27], sides=['off', 'on'], sessions_per_seed=4,
        baseline_source_commit=BASELINE_SOURCE_COMMIT,
        training_budget=64, epochs=4, limits=FIXED_LIMITS)
    # Exact types prevent True/1 and 4.0/4 from weakening a frozen count.
    for key, value in expected.items():
        if key == 'limits':
            observed = contract[key]
            if (type(observed) is not dict or set(observed) != set(value)
                or any(type(observed[name]) is not int or observed[name] != cap for name, cap in value.items())):
                raise ValueError('Frozen scope differs: limits')
            continue
        if exact_digest(contract[key]) != exact_digest(value):
            raise ValueError('Frozen scope differs: ' + key)
    if Path(contract['repo_root']).resolve() != ROOT:
        raise ValueError('Runner repository differs')
    if not isinstance(contract['source_fingerprints'], dict) or not contract['source_fingerprints']:
        raise ValueError('Source fingerprints required')


def check_contract(path):
    """Read-only preflight, before importing numerical libraries or claiming output."""
    path = Path(path).resolve(); contract = read_json(path); contract_shape(contract)
    root = Path(contract['repo_root']).resolve(); output = Path(contract['output_root']).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; claimed/failed runs cannot retry')
    if not output.parent.is_dir():
        raise ValueError('Output parent must already exist')
    if git(root, 'rev-parse', 'HEAD') != contract['source_commit'] or git(root, 'status', '--porcelain'):
        raise ValueError('Exact clean frozen source required')
    subprocess.check_call(['git', 'diff', '--quiet', BASELINE_SOURCE_COMMIT, 'HEAD', '--', 'src'], cwd=root)
    required = set(REQUIRED_SCRIPTS) | {str(p.relative_to(root)) for p in (root / 'src').rglob('*.py')}
    if not required.issubset(contract['source_fingerprints']):
        raise ValueError('Required runner/observer/algorithm source bindings missing')
    for name, expected in contract['source_fingerprints'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Source fingerprints require contained relative paths')
        if sha_file(root / relative) != expected:
            raise ValueError('Source SHA mismatch: ' + name)
    spec = contract['protocol']; protocol_path = Path(spec['path']).resolve()
    if sha_file(protocol_path) != spec['sha256']:
        raise ValueError('Protocol SHA mismatch')
    protocol = read_json(protocol_path)
    if (protocol.get('protocol') != 'joint_stream_learning_v2'
        or protocol.get('action_contract') != 'joint_stream_control_v2'
        or protocol.get('smoke') is not False or protocol.get('episodes') != 64
        or protocol.get('seeds') != [7, 17, 27]):
        raise ValueError('Original non-smoke stream v2/64-session protocol required')
    assets = protocol['assets']
    if 'protocol.pt' not in assets or 'predictor.pt' not in assets:
        raise ValueError('Both original protocol and predictor assets required')
    for name, expected in assets.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha_file(protocol_path.parent / relative) != expected:
            raise ValueError('Protocol asset SHA/path mismatch: ' + name)
    schedule = protocol['training_schedule']
    for seed in contract['seeds']:
        if type(schedule.get(str(seed))) is not list or len(schedule[str(seed)]) != 64:
            raise ValueError('Original full schedule required')
    return contract, protocol, path, protocol_path


def rss_bytes(platform=None, usage=None):
    platform = sys.platform if platform is None else platform
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if usage is None else usage
    if platform == 'darwin':
        return int(usage)
    if platform.startswith('linux'):
        return int(usage) * 1024
    raise RuntimeError('Unsupported ru_maxrss platform; cannot silently omit resource guard')


class Resources:
    def __init__(self, output, *, clock=time.monotonic, rss=rss_bytes, disk=shutil.disk_usage):
        self.output = Path(output); self.clock = clock; self.rss = rss; self.disk = disk
        self.started = clock(); self.counts = {key: 0 for key in
            ('training_sessions', 'training_updates', 'training_frames', 'models', 'environments',
             'backward_calls', 'adam_calls', 'categorical_sample_calls', 'evaluate_calls')}
        self.context = {}; self.peak_rss_bytes = 0

    def progress(self, phase, **identity):
        self.context.update(identity, phase=phase)
        self.check()
        value = dict(schema=SCHEMA, updated_at=datetime.now(timezone.utc).isoformat(),
            context=self.context, counts=self.counts, peak_rss_bytes=self.peak_rss_bytes,
            elapsed_seconds=self.clock() - self.started)
        temporary = self.output / 'progress.json.tmp'
        with temporary.open('w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
        temporary.replace(self.output / 'progress.json')

    def check(self, *, disk=False):
        used = self.rss(); self.peak_rss_bytes = max(self.peak_rss_bytes, used)
        if used > FIXED_LIMITS['max_rss_bytes']:
            raise RuntimeError('Frozen RSS limit exceeded; no automatic expansion/retry')
        if self.clock() - self.started > FIXED_LIMITS['max_wall_seconds']:
            raise RuntimeError('Frozen wall-time limit exceeded; no automatic expansion/retry')
        if disk and self.disk(self.output.parent).free < FIXED_LIMITS['minimum_free_disk_bytes']:
            raise RuntimeError('At least 30 GiB free disk required')

    def charge(self, name, amount=1):
        if type(amount) is not int or amount < 0 or name not in self.counts:
            raise ValueError('Invalid diagnostic charge')
        if self.counts[name] + amount > FIXED_LIMITS[name]:
            raise RuntimeError('Frozen call budget exceeded before original: ' + name)
        self.counts[name] += amount
        self.check()


@contextmanager
def guarded_runtime(agent, budget, torch):
    """External passive wrappers; no extra forward/backward/sample/Adam call."""
    from scripts.sa_stage5c_numerical_qualification import Patches
    counts = dict(idle_frames=0, decision_rows=0, active_heads=[0, 0, 0], forced_heads=[0, 0, 0],
        updates=[], main_adam_calls=0, surrogate_adam_calls=0)
    with Patches() as patches:
        original_act = agent.act_controls
        def act(decisions, frame, *, deterministic=False):
            if deterministic:
                raise ValueError('Diagnostic behavior must be stochastic')
            budget.check(); result = original_act(decisions, frame, deterministic=False)
            roll = result[1]
            if roll is None:
                counts['idle_frames'] += 1
            else:
                active = roll['active_heads'].detach().cpu()
                counts['decision_rows'] += len(active)
                for head in range(3):
                    n = int(active[:, head].sum()); counts['active_heads'][head] += n
                    counts['forced_heads'][head] += len(active) - n
            budget.check(); return result
        patches.set(agent, 'act_controls', act)
        original_eval = agent._evaluate
        def evaluate(*args, **kwargs):
            budget.charge('evaluate_calls'); result = original_eval(*args, **kwargs)
            budget.check(); return result
        patches.set(agent, '_evaluate', evaluate)
        original_update = agent.update
        def update(rollouts, *, epochs=4):
            if epochs != 4:
                raise ValueError('Original four epochs required')
            budget.charge('training_updates')
            before = dict(main_empty=not bool(agent.optimizer.state),
                surrogate_empty=not bool(agent.surrogate_optimizer.state),
                completed_updates=int(agent.completed_updates))
            result = original_update(rollouts, epochs=4)
            counts['updates'].append(dict(before=before,
                after=dict(main_empty=not bool(agent.optimizer.state),
                    surrogate_empty=not bool(agent.surrogate_optimizer.state),
                    completed_updates=int(agent.completed_updates)), stats=result))
            budget.check(); return result
        patches.set(agent, 'update', update)
        for label, optimizer in (('main', agent.optimizer), ('surrogate', agent.surrogate_optimizer)):
            original_step = optimizer.step
            def step(*args, _original=original_step, _label=label, **kwargs):
                budget.charge('adam_calls'); counts[_label + '_adam_calls'] += 1
                result = _original(*args, **kwargs); budget.check(); return result
            patches.set(optimizer, 'step', step)
        original_backward = torch.Tensor.backward
        def backward(tensor, *args, **kwargs):
            budget.charge('backward_calls'); result = original_backward(tensor, *args, **kwargs)
            budget.check(); return result
        patches.set(torch.Tensor, 'backward', backward)
        original_sample = torch.distributions.Categorical.sample
        def sample(distribution, *args, **kwargs):
            budget.charge('categorical_sample_calls')
            return original_sample(distribution, *args, **kwargs)
        patches.set(torch.distributions.Categorical, 'sample', sample)
        yield counts


@contextmanager
def guarded_consumer(budget):
    """Check each independent JSON event without retaining the producer history."""
    from scripts.sa_stage5c_credit_audit import _Joins
    from scripts.sa_stage5c_numerical_qualification import Patches
    original = _Joins.event
    def event(this, *args, **kwargs):
        budget.check(); result = original(this, *args, **kwargs); budget.check()
        return result
    with Patches() as patches:
        patches.set(_Joins, 'event', event)
        yield


def claim_output(output, payload):
    output = Path(output); output.mkdir(exist_ok=False)
    write_json(output / 'run_started.json', payload)


def execute(contract_path):
    contract, protocol, path, protocol_path = check_contract(contract_path)
    out = Path(contract['output_root']).resolve(); budget = Resources(out); budget.check(disk=True)
    contract_sha = sha_file(path)
    claim_output(out, dict(schema=SCHEMA, contract_sha256=contract_sha,
        source_commit=contract['source_commit'], command=sys.argv, no_retry=True,
        pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()))
    completed = []
    try:
        budget.progress('numerical_imports')
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        from scripts.train_joint_stream_control import make_agent, make_env, stream_episode
        from scripts.sa_stage5c_credit_audit import AuditSink, validate
        from scripts.sa_stage5c_credit_observer import credit_audit, detached
        from scripts.sa_stage5c_numerical_qualification import (compare_exact, learned_state,
            controller, extra_agent_state, rng_state, restore_rng)
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only diagnostic refuses initialized CUDA')
        data = torch.load(protocol_path.parent / 'protocol.pt', map_location='cpu', weights_only=False)
        if set(data['cases']) != {'train', 'dev'}:
            raise ValueError('Diagnostic input must contain train/dev only; holdout is forbidden')
        training = {case['case_id']: case for case in data['cases']['train']}
        if len(training) != len(data['cases']['train']):
            raise ValueError('Duplicate training case ID')
        class GuardedSink(AuditSink):
            def emit(self, *args, **kwargs):
                budget.check(); result = super().emit(*args, **kwargs); budget.check()
                return result
        for seed in contract['seeds']:
            budget.check(disk=True)
            seed_home = out / ('seed' + str(seed)); seed_home.mkdir()
            cases = protocol['training_schedule'][str(seed)][:4]
            if any(case_id not in training for case_id in cases):
                raise ValueError('Frozen first-four case absent from train')
            pair = []; common_rng = None
            for side in contract['sides']:
                home = seed_home / side; home.mkdir()
                budget.context = dict(seed=seed, side=side, phase='initialize')
                budget.progress('initialize')
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                budget.charge('models'); agent = make_agent('sa_ghmappo', seed, 64)
                if any(p.device.type != 'cpu' for p in agent.parameters()):
                    raise RuntimeError('CPU parameters required')
                if int(agent.completed_updates) != 0 or agent.optimizer.state or agent.surrogate_optimizer.state:
                    raise RuntimeError('Fresh zero-update/empty-Adam initialization required')
                initial = exact_digest(learned_state(agent)); start_rng = rng_state(torch, np)
                if side == 'off':
                    common_rng = start_rng; initial_hash = initial
                else:
                    if initial != initial_hash:
                        raise AssertionError('Fresh on/off initial learned states differ')
                    compare_exact(start_rng, common_rng, 'fresh RNG')
                restore_rng(common_rng, torch, np)
                enabled = side == 'on'
                sink = GuardedSink(home / 'audit', enable=enabled,
                    header=dict(run_uid=out.name + '_seed' + str(seed),
                        source_commit=contract['source_commit'], evidence_kind='unverified_actual_chain',
                        provenance=dict(source_run_id=out.name, protocol_path=str(protocol_path),
                            protocol_sha256=contract['protocol']['sha256'], contract_sha256=contract_sha,
                            seed=seed, side=side, source='fresh_original_stochastic_stream_sessions')),
                    max_rows=FIXED_LIMITS['audit_rows_per_seed'], max_bytes=FIXED_LIMITS['audit_bytes_per_seed'])
                rows = []; buffer = []; before_counts = dict(budget.counts)
                with sink:
                    with guarded_runtime(agent, budget, torch) as observed:
                        with credit_audit(agent, sink) as observer:
                            for episode, case_id in enumerate(cases, 1):
                                budget.context = dict(seed=seed, side=side, episode=episode, case_id=case_id, phase='rollout')
                                budget.progress('rollout')
                                budget.charge('training_sessions'); budget.charge('environments')
                                env = make_env(training[case_id], data, protocol)
                                original_step = env.step_controls; frames = [0]
                                def step(controls, _original=original_step, _frames=frames):
                                    if _frames[0] >= FIXED_LIMITS['frames_per_session']:
                                        raise RuntimeError('Session frame cap exceeded before original')
                                    budget.charge('training_frames'); _frames[0] += 1
                                    result = _original(controls); budget.check(); return result
                                env.step_controls = step
                                identity = dict(seed=seed, session_id=out.name + '_seed' + str(seed) + '_ep' + str(episode),
                                    episode=episode, case_id=case_id, phase='original_stochastic_train')
                                with observer.session(identity) if enabled else nullcontext():
                                    summary, rolls = stream_episode(env, agent, deterministic=False)
                                if summary['steps'] != frames[0] or not 0 < frames[0] <= 480:
                                    raise RuntimeError('Actual session/frame accounting differs')
                                row = dict(episode=episode, case_id=case_id, summary=detached(summary),
                                    summary_sha256=exact_digest(summary),
                                    rolls_sha256=exact_digest(rolls, exclude_audit_uids=True),
                                    receipts_sha256=exact_digest(env.frame_receipts),
                                    rng_after_sha256=exact_digest(rng_state(torch, np)))
                                rows.append(row); write_json(home / ('episode%d.json' % episode), row)
                                if side == 'on':
                                    expected_row = read_json(seed_home / 'off' / ('episode%d.json' % episode))
                                    compare_episode_receipt(row, expected_row, compare_exact)
                                buffer.extend(rolls); del env, rolls
                                budget.progress('episode_complete')
                            budget.progress('update')
                            stats = agent.update(buffer, epochs=4)
                            if int(agent.completed_updates) != 1:
                                raise RuntimeError('Exactly one original update required per side')
                            buffer.clear()
                            budget.progress('update_complete')
                        if enabled:
                            sink.finish(complete=True)
                final = dict(learned_state=learned_state(agent), controller=controller(agent),
                    extra_agent_state=extra_agent_state(agent), rng=rng_state(torch, np))
                checkpoint = home / 'checkpoint.pt'; agent.save_checkpoint(checkpoint)
                compare_exact(final['rng'], rng_state(torch, np), 'checkpoint RNG unchanged')
                final['checkpoint_payload'] = torch.load(checkpoint, map_location='cpu', weights_only=False)
                torch.save(final, home / 'final_evidence.pt')
                # Both observer._calls and sink.joins retain large detached inputs.
                # Drop them before the independent consumer builds its own joins.
                del observer, sink
                gc.collect(); budget.check()
                budget.progress('independent_consumer')
                with guarded_consumer(budget):
                    audit = validate(home / 'audit', max_file_bytes=FIXED_LIMITS['audit_bytes_per_seed']) if enabled else None
                if enabled and (audit['status'] != 'complete_unverified' or not audit['structural_complete']
                    or audit['actual_training_verified']):
                    raise RuntimeError('Observed chain structural completion failed')
                side_counts = {key: budget.counts[key] - before_counts[key] for key in before_counts}
                receipt = dict(side=side, seed=seed, initial_learned_state_sha256=initial, case_ids=cases,
                    original_call_counts=side_counts, coverage=detached(observed), update_stats=detached(stats),
                    audit=audit, final_evidence_sha256=sha_file(home / 'final_evidence.pt'),
                    checkpoint_sha256=sha_file(checkpoint), episode_count=len(rows))
                write_json(home / 'result.json', receipt); pair.append(receipt)
                if side == 'on':
                    off = torch.load(seed_home / 'off' / 'final_evidence.pt', map_location='cpu', weights_only=False)
                    for field in final:
                        compare_exact(final[field], off[field], 'final.' + field)
                    compare_exact(pair[0]['original_call_counts'], side_counts, 'original operation counts')
                    compare_exact(pair[0]['coverage'], detached(observed), 'coverage/update statistics')
                    del off
                del agent, final, checkpoint, observed, rows
                gc.collect(); budget.check()
            result = dict(seed=seed, status='exact_parity', sides=pair,
                only_excluded_rollout_field=EXCLUDED_UID_KEY,
                coverage_scope='actual observed paths only; absent free-cache/idle is not synthesized')
            write_json(seed_home / 'result.json', result); completed.append(result)
            budget.progress('seed_complete')
        # Recheck frozen inputs without treating this now-claimed output as new.
        if sha_file(path) != contract_sha or git(ROOT, 'rev-parse', 'HEAD') != contract['source_commit'] or git(ROOT, 'status', '--porcelain'):
            raise RuntimeError('Contract/source identity changed during run')
        for name, expected in contract['source_fingerprints'].items():
            if sha_file(ROOT / name) != expected:
                raise RuntimeError('Source changed during diagnostic: ' + name)
        if sha_file(protocol_path) != contract['protocol']['sha256']:
            raise RuntimeError('Original protocol manifest changed during diagnostic')
        for name, expected in protocol['assets'].items():
            if sha_file(protocol_path.parent / name) != expected:
                raise RuntimeError('Original protocol asset changed during diagnostic: ' + name)
        if budget.counts['training_sessions'] != 24 or budget.counts['training_updates'] != 6:
            raise RuntimeError('Incomplete frozen scope')
        result = dict(schema=SCHEMA, status='complete_exact_observer_parity', counts=budget.counts,
            limits=FIXED_LIMITS, peak_rss_bytes=budget.peak_rss_bytes,
            elapsed_seconds=budget.clock() - budget.started, seeds=completed,
            evidence_boundary='Fresh real stochastic training with original update. Observer parity on actual paths only; no stability repair, convergence or performance promotion claim.',
            formal_holdout='not_read', no_retry=True)
        write_json(out / 'result.json', result)
        budget.progress('complete')
        return 0
    except BaseException as exc:
        try:
            write_json(out / 'failure.json', dict(schema=SCHEMA, status='failed',
                exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                counts=budget.counts, context=budget.context, completed_seeds=len(completed),
                no_retry=True, possibly_partial_update=True,
                boundary='Original step may have completed; no rollback/resume/retry or automatic budget expansion.'))
        except BaseException:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'run':
        return execute(args.contract)
    contract, protocol, path, _ = check_contract(args.contract)
    budget = Resources(Path(contract['output_root'])); budget.check(disk=True)
    print(json.dumps(dict(status='preflight_pass_no_numerical_imports', schema=SCHEMA,
        contract_sha256=sha_file(path), limits=FIXED_LIMITS,
        first_four={str(seed): protocol['training_schedule'][str(seed)][:4] for seed in contract['seeds']})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
