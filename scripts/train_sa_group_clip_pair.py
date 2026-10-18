"""Frozen, single-arm SA group-clip experiment. Never resumes or chooses a peak.

The coordinator owns prepare/qualification and invokes this exactly once per arm.
GradientObserver is also importable for pure-tensor parity tests without any env.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = {'sa_legacy_clip': False, 'sa_group_clip': True}
SEEDS = [7, 17, 27]
POINTS = list(range(0, 65, 8))
VALUE_PREFIXES = ('critic.', 'prefix_values.', 'action_values.')
HEADS = ('execution', 'cache', 'prepare_priority')
TAG = 'sa_policy_value_clip_v1'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unserializable observation: ' + type(value).__name__)


def write_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def emit(handle, value):
    handle.write(json.dumps(jsonable(value), separators=(',', ':'), allow_nan=False) + '\n')
    handle.flush()


def object_hash(value):
    """Read-only, type/shape preserving hash; no sampling, backward or mutation."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            z = x.detach().cpu().contiguous()
            h.update(b'tensor'); h.update(str((z.dtype, tuple(z.shape))).encode()); h.update(z.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b'numpy'); h.update(str((x.dtype, x.shape)).encode()); h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k, v in sorted(x.items(), key=lambda item: repr(item[0])):
                visit(k); visit(v)
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for v in x: visit(v)
        else:
            h.update(type(x).__name__.encode()); h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def rng_hash():
    return object_hash(rng_state())


@contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def training_state_hash(agent):
    names = ('request_memory', 'target_memory', 'commitments', 'last_diagnostics',
             '_team_targets', '_calendar', '_cache_values', '_feature_opportunities',
             '_delivery_decisions', '_gradient_stats', '_separate_gradient_stats')
    return object_hash(dict(weights=agent.state_dict(), optimizer=agent.optimizer.state_dict(),
        surrogate_optimizer=agent.surrogate_optimizer.state_dict(),
        gradients={n: p.grad for n, p in agent.named_parameters()},
        context={n: getattr(agent, n, None) for n in names}, options=agent.options,
        train_mode=agent.training))


def parameter_groups(agent):
    groups = {'policy': [], 'value': [], 'surrogate': []}
    for name, param in agent.named_parameters():
        group = 'surrogate' if name.startswith('surrogates.') else ('value' if name.startswith(VALUE_PREFIXES) else 'policy')
        groups[group].append((name, param))
    ids = [id(p) for pairs in groups.values() for _, p in pairs]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in agent.parameters()}:
        raise RuntimeError('Gradient group partition is not disjoint/exhaustive')
    if any(not pairs for pairs in groups.values()):
        raise RuntimeError('Missing gradient group')
    return groups


def gradient_norm(pairs):
    squares = 0.; count = 0; parameters = 0
    for _, p in pairs:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError('Non-finite real training gradient')
        squares += float(g.double().square().sum()); count += g.numel(); parameters += 1
    return dict(norm=math.sqrt(squares), elements=count, parameters_with_grad=parameters,
                parameters_total=len(pairs), none_grad_parameters=len(pairs)-parameters)


def parameter_snapshot(groups):
    return {name: p.detach().clone() for pairs in groups.values() for name, p in pairs}


def parameter_delta(groups, before):
    answer = {}
    for group, pairs in groups.items():
        squares = 0.; changed = 0; maximum = 0.
        for name, param in pairs:
            delta = param.detach().double() - before[name].double()
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError('Non-finite Adam parameter delta')
            squares += float(delta.square().sum())
            changed += int(torch.count_nonzero(delta))
            if delta.numel(): maximum = max(maximum, float(delta.abs().max()))
        answer[group] = dict(l2=math.sqrt(squares), changed_elements=changed, max_abs=maximum)
    return answer


def optimizer_steps(optimizer):
    return {str(i): float(optimizer.state.get(p, {}).get('step', 0)) for i, p in enumerate(
        [p for group in optimizer.param_groups for p in group['params']])}


class GradientObserver:
    """Observe existing update calls, without adding backward/step/RNG calls.

    Patches exist only inside this context and are restored even on failure. All
    numerical observations use detach/no-grad. The original clip function return
    and original optimizer/forward return are passed through unchanged. A pure
    tensor test may call only _apply_control_gradients within the context.
    """
    def __init__(self, agent, *, identity=None, minibatch_file=None, forward_file=None):
        self.agent = agent; self.identity = dict(identity or {})
        self.minibatch_file = minibatch_file; self.forward_file = forward_file
        self.groups = parameter_groups(agent)
        self.minibatches = []; self.forwards = []; self.current = None
        self.adam_steps = Counter(); self._restore = []; self._choice_rows = None
        self.last_forward = None; self._installed = False

    def _patch(self, obj, name, call):
        had = name in obj.__dict__; old = obj.__dict__.get(name)
        self._restore.append((obj, name, had, old))
        setattr(obj, name, MethodType(call, obj))

    def __enter__(self):
        if self._installed: raise RuntimeError('Observer cannot be reused')
        self._installed = True; agent = self.agent
        original_apply = agent._apply_control_gradients
        original_clip = torch.nn.utils.clip_grad_norm_
        self._original_clip = original_clip
        expected = 3 if getattr(agent, 'separate_policy_value_clip', False) else 2
        expected_ids = {
            'policy': {id(p) for _, p in self.groups['policy']},
            'value': {id(p) for _, p in self.groups['value']},
            'surrogate': {id(p) for _, p in self.groups['surrogate']}}
        expected_ids['policy_value'] = expected_ids['policy'] | expected_ids['value']

        def clip(parameters, max_norm, *args, **kwargs):
            params = (parameters,) if isinstance(parameters, torch.Tensor) else tuple(parameters)
            if self.current is None: raise RuntimeError('clip outside an observed backward')
            ids = {id(p) for p in params}
            labels = [name for name, wanted in expected_ids.items() if ids == wanted]
            if len(labels) != 1 or float(max_norm) != .5:
                raise RuntimeError('Unexpected clip partition or threshold')
            label = labels[0]
            if self.current['preclip'] is None:
                self.current['preclip'] = {k: gradient_norm(v) for k, v in self.groups.items()}
            before_rng = rng_hash()
            norm = original_clip(params, max_norm, *args, **kwargs)
            if rng_hash() != before_rng: raise RuntimeError('Clipping consumed RNG')
            value = float(norm)
            if not math.isfinite(value): raise FloatingPointError('Non-finite returned clip norm')
            self.current['clip_calls'].append(dict(group=label, max_norm=float(max_norm), returned_pre_norm=value,
                coefficient=min(1., float(max_norm)/(value+1e-6)),
                group_norms_after={k: gradient_norm(v) for k, v in self.groups.items()}))
            return norm

        def apply(this, loss):
            if self.current is not None: raise RuntimeError('Nested gradient application')
            if not bool(torch.isfinite(loss.detach()).all()): raise FloatingPointError('Non-finite real training loss')
            row = dict(self.identity, minibatch=len(self.minibatches)+1,
                forward_id=self.last_forward, loss=float(loss.detach()),
                rng_before=rng_hash(), preclip=None, postclip=None, clip_calls=[], adam_steps=[])
            self.current = row
            try:
                answer = original_apply(loss)
                if len(row['clip_calls']) != expected: raise RuntimeError('Unexpected clip call count')
                wanted = ['policy', 'value', 'surrogate'] if expected == 3 else ['policy_value', 'surrogate']
                if [x['group'] for x in row['clip_calls']] != wanted: raise RuntimeError('Clip order mismatch')
                if [x['optimizer'] for x in row['adam_steps']] != ['main', 'surrogate']:
                    raise RuntimeError('Unexpected actual Adam call count/order')
                row['rng_after'] = rng_hash(); row['rng_unchanged'] = row['rng_after'] == row['rng_before']
                if not row['rng_unchanged']: raise RuntimeError('Observed backward/Adam consumed RNG')
                row['finite'] = True
                self.minibatches.append(row)
                if self.minibatch_file is not None: emit(self.minibatch_file, row)
                return answer
            finally:
                self.current = None

        for label, optimizer in [('main', agent.optimizer), ('surrogate', agent.surrogate_optimizer)]:
            original_step = optimizer.step
            def step(this, *args, _label=label, _original=original_step, **kwargs):
                if self.current is None: raise RuntimeError('Adam outside observed backward')
                before_rng = rng_hash(); before = parameter_snapshot(self.groups); steps_before = optimizer_steps(this)
                if _label == 'main':
                    self.current['postclip'] = {k: gradient_norm(v) for k, v in self.groups.items()}
                answer = _original(*args, **kwargs)
                if rng_hash() != before_rng: raise RuntimeError('Adam observation consumed RNG')
                self.adam_steps[_label] += 1
                self.current['adam_steps'].append(dict(optimizer=_label,
                    parameter_delta=parameter_delta(self.groups, before),
                    parameter_steps_before=steps_before, parameter_steps_after=optimizer_steps(this), rng_unchanged=True))
                return answer
            self._patch(optimizer, 'step', step)

        if hasattr(agent, '_choice_distribution') and hasattr(agent, '_evaluate'):
            original_choice = agent._choice_distribution; original_evaluate = agent._evaluate
            def choice(this, logits, mask, deterministic):
                if self._choice_rows is not None:
                    counts = mask.detach().sum(1).cpu()
                    self._choice_rows.append(dict(head=len(self._choice_rows), rows=len(counts),
                        forced_rows=int((counts == 1).sum()), multichoice_rows=int((counts > 1).sum()),
                        valid_candidate_min=int(counts.min()) if len(counts) else None,
                        valid_candidate_max=int(counts.max()) if len(counts) else None))
                return original_choice(logits, mask, deterministic)
            def evaluate(this, *args, **kwargs):
                if self._choice_rows is not None: raise RuntimeError('Nested training evaluate')
                self._choice_rows = []
                try:
                    result = original_evaluate(*args, **kwargs)
                    coverage = self._choice_rows
                finally:
                    self._choice_rows = None
                enabled = result[6].detach().cpu() if len(result) > 6 else None
                row = dict(self.identity, forward_id=len(self.forwards)+1,
                    choice_coverage=coverage, active_head_rows=(enabled.sum(0).tolist() if enabled is not None else None),
                    total_rows=(len(enabled) if enabled is not None else None))
                self.forwards.append(row); self.last_forward = row['forward_id']
                if self.forward_file is not None: emit(self.forward_file, row)
                return result
            self._patch(agent, '_choice_distribution', choice)
            self._patch(agent, '_evaluate', evaluate)
        self._patch(agent, '_apply_control_gradients', apply)
        torch.nn.utils.clip_grad_norm_ = clip
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.nn.utils.clip_grad_norm_ = self._original_clip
        for obj, name, had, old in reversed(self._restore):
            if had: setattr(obj, name, old)
            else: delattr(obj, name)
        return False

    def summary(self):
        norms = {}
        for group in self.groups:
            norms[group] = {}
            for phase in ('preclip', 'postclip'):
                values = [row[phase][group]['norm'] for row in self.minibatches]
                norms[group][phase] = dict(count=len(values), min=min(values) if values else None,
                    max=max(values) if values else None, mean=sum(values)/len(values) if values else None)
        return dict(actual_backwards=len(self.minibatches), actual_forward_calls=len(self.forwards),
            actual_adam_steps=dict(self.adam_steps), group_norms=norms,
            group_parameter_names={k: [n for n, _ in v] for k, v in self.groups.items()},
            group_partition_disjoint_exhaustive=True,
            all_backward_rng_unchanged=all(r['rng_unchanged'] for r in self.minibatches),
            observer_only=True, additional_backward_calls=0, additional_optimizer_steps=0)


def observe_update(agent, rollouts, *, identity=None, minibatch_file=None, forward_file=None):
    """Exactly one original update(rollouts), default epochs unchanged (4)."""
    before = parameter_snapshot(parameter_groups(agent)); updates = int(agent.completed_updates)
    before_rng = rng_hash()
    with GradientObserver(agent, identity=identity, minibatch_file=minibatch_file, forward_file=forward_file) as observer:
        result = agent.update(rollouts)
    if int(agent.completed_updates) != updates + 1: raise RuntimeError('Completed update count mismatch')
    report = observer.summary()
    if int(result['gradient_steps']) != report['actual_backwards']: raise RuntimeError('Gradient steps mismatch')
    if report['actual_adam_steps'].get('main', 0) != report['actual_backwards'] or report['actual_adam_steps'].get('surrogate', 0) != report['actual_backwards']:
        raise RuntimeError('Adam count mismatch')
    report.update(dict(identity or {}), completed_updates_before=updates, completed_updates_after=int(agent.completed_updates),
        rng_before=before_rng, rng_after=rng_hash(), parameter_delta=parameter_delta(parameter_groups(agent), before),
        losses=result, default_epochs=4)
    return result, report


def checked_path(spec, label, base=None):
    if isinstance(spec, str):
        path = Path(spec); expected = None
    else:
        path = Path(spec['path']); expected = spec.get('sha256')
    if not path.is_absolute() and base is not None: path = base / path
    path = path.resolve()
    if not path.is_file(): raise FileNotFoundError(label + ': ' + str(path))
    if expected is not None and sha(path) != expected: raise RuntimeError(label + ' SHA mismatch')
    return path


def verify_contract(path, *, root=ROOT):
    contract = json.loads(Path(path).read_text()); base = Path(path).resolve().parent
    if contract.get('schema') != 'sa_group_clip_pair_v1' or contract.get('status') != 'frozen':
        raise RuntimeError('Frozen paired contract is required')
    if Path(contract['repo_root']).resolve() != root.resolve() or Path(contract['output_root']).resolve() != base:
        raise RuntimeError('Frozen repository/output root mismatch')
    if contract.get('primary_checkpoint') != 64:
        raise RuntimeError('Primary checkpoint must remain fixed64')
    if contract['arms'] != ARMS: raise RuntimeError('Arm flags must be frozen booleans')
    if contract['seeds'] != SEEDS or contract['episodes'] != 64 or contract['checkpoint_episodes'] != POINTS:
        raise RuntimeError('Unexpected seed, training or checkpoint scope')
    if contract['per_arm_max_training_frames'] != 92160 or contract['per_arm_max_evaluation_frames'] != 622080:
        raise RuntimeError('Unexpected frame budget')
    expected_limits = dict(max_training_sessions=384, max_training_frames=184320,
        max_evaluation_sessions=2592, max_evaluation_frames=1244160,
        per_arm_training_sessions=192, per_arm_evaluation_sessions=1296,
        max_frames_per_session=480, torch_threads_per_arm=1)
    if any(contract.get(k) != v for k, v in expected_limits.items()):
        raise RuntimeError('Frozen paired limits changed')
    qualification = contract['qualification']
    receipt = checked_path(dict(path=qualification['receipt'], sha256=qualification['receipt_sha256']), 'qualification receipt', base)
    if qualification['max_sessions'] != 2 or qualification['max_frames'] != 960 or json.loads(receipt.read_text())['status'] != 'pass':
        raise RuntimeError('Bounded qualification has not passed')
    if len(contract['eval_panel_ids']) != 48 or len(set(contract['eval_panel_ids'])) != 48:
        raise RuntimeError('Expected exactly 48 unique evaluation conditions')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if commit != contract['source_commit']: raise RuntimeError('Frozen Git HEAD mismatch')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()
    if dirty: raise RuntimeError('Clean frozen Git source is required')
    if not contract['source_fingerprints']: raise RuntimeError('Missing frozen source fingerprints')
    for name, expected in contract['source_fingerprints'].items():
        if sha(root / name) != expected: raise RuntimeError('Source SHA mismatch: ' + name)
    protocol_spec = contract['source_protocol']
    if isinstance(protocol_spec, str):
        protocol_spec = dict(path=protocol_spec, sha256=contract['source_protocol_sha256'])
    protocol_path = checked_path(protocol_spec, 'source protocol', base)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('protocol') != 'joint_stream_learning_v2' or protocol.get('action_contract') != 'joint_stream_control_v2':
        raise RuntimeError('Wrong original protocol contract')
    data_path = checked_path(contract['protocol_data'], 'protocol data', base)
    if sha(data_path) != protocol['assets']['protocol.pt']: raise RuntimeError('Protocol data not bound to original asset')
    for name, expected in protocol['assets'].items():
        if sha(protocol_path.parent / name) != expected: raise RuntimeError('Original protocol asset mismatch: ' + name)
    for name, expected in protocol['source_fingerprints'].items():
        if name == 'src/agents/sa_ghmappo_agent.py': continue
        if sha(root / name) != expected: raise RuntimeError('Unauthorized change from original source: ' + name)
    if contract['training_schedule'] != protocol['training_schedule']:
        raise RuntimeError('Training condition schedule differs from original')
    if protocol['seeds'] != SEEDS or protocol['episodes'] != 64:
        raise RuntimeError('Original protocol budget mismatch')
    if any(len(contract['training_schedule'][str(seed)]) != 64 for seed in SEEDS):
        raise RuntimeError('Schedule length mismatch')
    if contract.get('preservation_plan') is None: raise RuntimeError('Missing preservation contract')
    checked_path(contract['preservation_plan'], 'preservation plan', base)
    return contract, protocol, data_path


def build_arm_agent(seed, flag):
    from src.agents.registry import build_agent
    return build_agent('sa_ghmappo', control_contract='joint_stream_control_v2', seed=seed,
        training_budget=64, prediction_enabled=True, return_gamma=1., separate_policy_value_clip=flag)


def finite_agent(agent):
    if any(not bool(torch.isfinite(t).all()) for t in agent.state_dict().values() if torch.is_floating_point(t)):
        raise FloatingPointError('Non-finite trained tensor')


def no_update(*args, **kwargs):
    raise RuntimeError('Deterministic evaluation forbids update and optimizer.step')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    # No environment or model is constructed before source/contract validation.
    contract, protocol, data_path = verify_contract(args.contract)
    contract_sha = sha(args.contract); output = args.output_root.resolve(); pair_root = args.contract.resolve().parent
    if output != pair_root / 'arms' / args.arm: raise RuntimeError('Arm output must be pairroot/arms/<canonical arm>')
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', started_at=utcnow(), arm=args.arm, agent='sa_ghmappo',
        separate_policy_value_clip=ARMS[args.arm], contract=str(args.contract.resolve()), contract_sha256=contract_sha,
        source_commit=contract['source_commit'], source_fingerprints=contract['source_fingerprints'], git_dirty=False,
        protocol=str(checked_path(contract['source_protocol'], 'protocol', pair_root)),
        protocol_sha256=(contract['source_protocol'].get('sha256') if isinstance(contract['source_protocol'], dict) else contract['source_protocol_sha256']),
        protocol_data_sha256=sha(data_path), checkpoints=[], training_episodes=0, training_sessions=0,
        training_frames=0, training_decisions=0, eval_cells=0, evaluation_frames=0,
        actual_updates=0, updates_by_seed={}, observer_adam_steps={'main': 0, 'surrogate': 0}, rng_isolation=[],
        guarded_training_step_attempts=0, guarded_evaluation_step_attempts=0)
    manifest_path = output / 'manifest.json'
    command = dict(argv=sys.argv, cwd=str(ROOT), started_at=utcnow(), contract_sha256=contract_sha,
                   script_path=str(Path(__file__).resolve()), script_sha256=sha(__file__))
    metadata['command'] = command
    write_json(output / 'command.json', command)
    with (output / 'run_started.json').open('x') as handle:
        emit(handle, dict(command=command, exclusive=True, no_resume=True))
    write_json(manifest_path, metadata)

    def safe():
        if (pair_root / 'failure.json').exists(): raise RuntimeError('Paired run failure flag exists; no further sessions allowed')
        if sha(args.contract) != contract_sha: raise RuntimeError('Contract changed during run')

    def progress(phase, seed=None, episode=None, case_id=None):
        write_json(output / 'progress.json', dict(status=metadata['status'], arm=args.arm, phase=phase,
            seed=seed, episode=episode, case_id=case_id, updated_at=utcnow(),
            training_sessions=metadata['training_sessions'], training_frames=metadata['training_frames'],
            eval_cells=metadata['eval_cells'], evaluation_frames=metadata['evaluation_frames'],
            actual_updates=metadata['actual_updates']))

    def before_session(phase, case, data):
        safe()
        session_frames = len(data['sessions'][case['session']]['frames'])
        if session_frames != 480: raise RuntimeError('Frozen session frame length changed')
        if phase == 'train':
            if metadata['training_sessions'] >= 192 or metadata['training_frames'] + session_frames > 92160:
                raise RuntimeError('Training scope/frame budget exceeded before session')
        else:
            if metadata['eval_cells'] >= 1296 or metadata['evaluation_frames'] + session_frames > 622080:
                raise RuntimeError('Evaluation scope/frame budget exceeded before session')

    try:
        from scripts.train_joint_stream_control import make_env, stream_episode, actor_hash
        from scripts.train_joint_tail_control import tensor_hash
        source_run = Path(contract['source_run']).resolve()
        source_manifest_path = checked_path(contract['original_sa_manifest'], 'original SA manifest', pair_root)
        if source_manifest_path != source_run / 'sa_ghmappo' / 'manifest.json':
            raise RuntimeError('Source run and frozen SA manifest disagree')
        source_manifest_sha = sha(source_manifest_path)
        original = json.loads(source_manifest_path.read_text())
        original_cp = {(c['seed'], c['episode']): c for c in original['checkpoints']}
        original_cp_hashes = {str(source_run/'sa_ghmappo'/c['path']): c['sha256'] for c in original['checkpoints']}
        for path, expected in original_cp_hashes.items():
            if sha(path) != expected: raise RuntimeError('Original checkpoint identity mismatch')
        data = torch.load(data_path, weights_only=False)

        def execute_episode(phase, case, model, *, deterministic=False, event_file=None, receipt_file=None, identity=None):
            before_session(phase, case, data)
            env = make_env(case, data, protocol)
            original_step = env.step_controls
            had_step = 'step_controls' in env.__dict__; old_step = env.__dict__.get('step_controls')
            calls = 0
            counter = 'guarded_training_step_attempts' if phase == 'train' else 'guarded_evaluation_step_attempts'
            maximum = 92160 if phase == 'train' else 622080
            def guarded_step(this, *step_args, **step_kwargs):
                nonlocal calls
                # A reservation is made before the session; this separate hard
                # guard also checks before EACH physical step, including failures.
                if calls >= 480 or metadata[counter] >= maximum:
                    raise RuntimeError('Physical step would exceed frozen frame budget')
                calls += 1; metadata[counter] += 1
                return original_step(*step_args, **step_kwargs)
            env.step_controls = MethodType(guarded_step, env)
            try:
                row, rolls = stream_episode(env, model, deterministic=deterministic,
                    event_file=event_file, receipt_file=receipt_file, identity=identity)
                if row['steps'] != calls or not 0 < calls <= 480:
                    raise RuntimeError('Reported physical frames differ from guarded step calls')
                return row, rolls
            finally:
                if had_step: env.step_controls = old_step
                else: delattr(env, 'step_controls')

        training = {c['case_id']: c for c in data['cases']['train']}
        development = {c['case_id']: c for c in data['cases']['dev']}
        if len(development) != 48 or set(contract['eval_panel_ids']) != set(development):
            raise RuntimeError('Evaluation panel does not equal original full 48 conditions')
        selected = [development[case_id] for case_id in contract['eval_panel_ids']]
        for schedule in contract['training_schedule'].values():
            if not set(schedule) <= set(training): raise RuntimeError('Training schedule contains unknown case')
        files = ['training.jsonl', 'dev.jsonl', 'gradient_minibatches.jsonl', 'gradient_forwards.jsonl', 'gradient_updates.jsonl', 'rng_isolation.jsonl']
        from contextlib import ExitStack
        with ExitStack() as stack:
            handles = {name: stack.enter_context((output / name).open('x')) for name in files}
            events = stack.enter_context(gzip.open(output / 'events.jsonl.gz', 'xt', compresslevel=1))
            receipts = stack.enter_context(gzip.open(output / 'receipts.jsonl.gz', 'xt', compresslevel=1))
            for seed in SEEDS:
                safe(); random.seed(seed); np.random.seed(seed); agent = build_arm_agent(seed, ARMS[args.arm])
                initial = tensor_hash(agent); initial_actor = actor_hash(agent)
                expected = contract['expected_initial'][str(seed)]
                if initial != expected['tensor_sha256'] or initial_actor != expected['actor_sha256']:
                    raise RuntimeError('Initial tensor/actor identity mismatch')
                if int(agent.completed_updates) != 0: raise RuntimeError('Nonzero initialized update count')
                buffer = []; metadata['updates_by_seed'][str(seed)] = 0

                def evaluate_point(episode):
                    safe(); finite_agent(agent)
                    if int(agent.completed_updates) != episode // 4: raise RuntimeError('Checkpoint update count mismatch')
                    before_rng = rng_hash(); before_state = training_state_hash(agent)
                    model_tensor = tensor_hash(agent); model_actor = actor_hash(agent)
                    if args.arm == 'sa_legacy_clip' and episode in (32, 64):
                        reference = original_cp[(seed, episode)]
                        if model_tensor != reference['trained_tensor_sha256'] or model_actor != reference['trained_actor_sha256']:
                            raise RuntimeError('Legacy does not exactly reproduce historical checkpoint; paired science stopped')
                    checkpoint = output / f'{args.arm}_seed{seed}_ep{episode}.pt'
                    if checkpoint.exists(): raise RuntimeError('Checkpoint overwrite forbidden')
                    agent.save_checkpoint(checkpoint)
                    saved = torch.load(checkpoint, weights_only=False)
                    flag = saved['options'].get('separate_policy_value_clip', False)
                    if type(flag) is not bool or flag != ARMS[args.arm]: raise RuntimeError('Checkpoint option flag mismatch')
                    tag = saved.get('training_gradient_contract')
                    if (flag and tag != TAG) or (not flag and tag is not None): raise RuntimeError('Checkpoint gradient contract tag mismatch')
                    record = dict(arm=args.arm, seed=seed, episode=episode, path=checkpoint.name, sha256=sha(checkpoint),
                        initial_tensor_sha256=initial, initial_actor_sha256=initial_actor,
                        trained_tensor_sha256=model_tensor, trained_actor_sha256=model_actor,
                        expected_updates=episode//4, completed_updates=int(agent.completed_updates),
                        options_flag=flag, gradient_tag=tag, rng_before=before_rng, training_state_before=before_state)
                    with preserved_rng():
                        evaluated = type(agent).from_checkpoint(checkpoint)
                        if tensor_hash(evaluated) != model_tensor or actor_hash(evaluated) != model_actor:
                            raise RuntimeError('Evaluation copy differs from saved checkpoint')
                        evaluated.update = no_update; evaluated.optimizer.step = no_update; evaluated.surrogate_optimizer.step = no_update
                        for case in selected:
                            identity = dict(arm=args.arm, agent='sa_ghmappo', rule=f'{args.arm}_seed{seed}',
                                seed=seed, episode=episode, case_id=case['case_id'])
                            row, _ = execute_episode('eval', case, evaluated, deterministic=True,
                                event_file=events, receipt_file=receipts, identity=identity)
                            if not 0 < row['steps'] <= 480: raise RuntimeError('Evaluation frame accounting mismatch')
                            if tensor_hash(evaluated) != model_tensor or actor_hash(evaluated) != model_actor:
                                raise RuntimeError('Evaluation changed model tensors')
                            emit(handles['dev.jsonl'], dict(identity, **row, tensor_sha256=model_tensor,
                                actor_sha256=model_actor, completed_updates=episode//4))
                            metadata['eval_cells'] += 1; metadata['evaluation_frames'] += row['steps']
                            progress('evaluation', seed, episode, case['case_id'])
                    record.update(rng_after=rng_hash(), training_state_after=training_state_hash(agent))
                    record['rng_unchanged'] = record['rng_before'] == record['rng_after']
                    record['training_agent_unchanged'] = record['training_state_before'] == record['training_state_after']
                    if not record['rng_unchanged'] or not record['training_agent_unchanged']:
                        raise RuntimeError('Evaluation disturbed training state/RNG')
                    emit(handles['rng_isolation.jsonl'], record)
                    metadata['rng_isolation'].append({k: record[k] for k in ('seed', 'episode', 'rng_unchanged', 'training_agent_unchanged')})
                    metadata['checkpoints'].append(record)
                    write_json(manifest_path, metadata)
                    progress('checkpoint_evaluation_complete', seed, episode)

                evaluate_point(0)
                for episode, case_id in enumerate(contract['training_schedule'][str(seed)], 1):
                    case = training[case_id]
                    row, rolls = execute_episode('train', case, agent)
                    if not 0 < row['steps'] <= 480: raise RuntimeError('Training frame accounting mismatch')
                    buffer.extend(rolls); losses = None
                    if episode % 4 == 0:
                        safe()
                        identity = dict(arm=args.arm, agent='sa_ghmappo', seed=seed, episode=episode,
                            update=episode//4, sample_sessions=list(range(episode-3, episode+1)))
                        losses, observation = observe_update(agent, buffer, identity=identity,
                            minibatch_file=handles['gradient_minibatches.jsonl'], forward_file=handles['gradient_forwards.jsonl'])
                        emit(handles['gradient_updates.jsonl'], observation)
                        metadata['actual_updates'] += 1; metadata['updates_by_seed'][str(seed)] += 1
                        for name in ('main', 'surrogate'):
                            metadata['observer_adam_steps'][name] += observation['actual_adam_steps'].get(name, 0)
                        buffer = []; finite_agent(agent)
                    emit(handles['training.jsonl'], dict(arm=args.arm, agent='sa_ghmappo', seed=seed,
                        episode=episode, case_id=case_id, losses=losses, completed_updates=int(agent.completed_updates), **row))
                    metadata['training_episodes'] += 1; metadata['training_sessions'] += 1
                    metadata['training_frames'] += row['steps']; metadata['training_decisions'] += row['decision_rows']
                    write_json(manifest_path, metadata); progress('training', seed, episode, case_id)
                    if episode in POINTS: evaluate_point(episode)
                if buffer or int(agent.completed_updates) != 16: raise RuntimeError('Incomplete fixed 64-session training')
            if metadata['training_sessions'] != 192 or metadata['eval_cells'] != 1296 or len(metadata['checkpoints']) != 27:
                raise RuntimeError('Incomplete frozen scientific scope')
            if metadata['actual_updates'] != 48 or set(metadata['updates_by_seed'].values()) != {16}:
                raise RuntimeError('Actual update count mismatch')
        if metadata['guarded_training_step_attempts'] != metadata['training_frames'] or metadata['guarded_evaluation_step_attempts'] != metadata['evaluation_frames']:
            raise RuntimeError('Physical step guard/final frame ledger mismatch')
        safe(); verify_contract(args.contract)
        if sha(source_manifest_path) != source_manifest_sha: raise RuntimeError('Original manifest changed')
        for path, expected in original_cp_hashes.items():
            if sha(path) != expected: raise RuntimeError('Original checkpoint changed')
        output_files = files + ['events.jsonl.gz', 'receipts.jsonl.gz', 'command.json', 'run_started.json']
        metadata.update(status='complete', completed_at=utcnow(), output_sha256={name: sha(output/name) for name in output_files},
            original_sa_manifest_and_checkpoints_unchanged=True, source_run_write_operations=0, actual_environment_sessions=metadata['training_sessions']+metadata['eval_cells'],
            additional_replay_sessions=0, no_peak_selection=True, final_episode=64)
        write_json(manifest_path, metadata); progress('complete')
        write_json(output/'result.json', dict(status='complete', arm=args.arm, manifest_sha256=sha(manifest_path),
            training_sessions=metadata['training_sessions'], training_frames=metadata['training_frames'],
            evaluation_sessions=metadata['eval_cells'], evaluation_frames=metadata['evaluation_frames'],
            actual_updates=metadata['actual_updates'], observer_adam_steps=metadata['observer_adam_steps'], completed_at=utcnow()))
        print(json.dumps(dict(status='complete', arm=args.arm, training_sessions=192, evaluation_sessions=1296)), flush=True)
    except BaseException as error:
        failure = dict(status='failed', arm=args.arm, failed_at=utcnow(), error_type=type(error).__name__,
            error=str(error), no_automatic_retry=True, last_progress=(json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else None))
        metadata.update(status='failed', failure=failure); write_json(manifest_path, metadata)
        write_json(output/'failure.json', failure)
        try:
            with (pair_root/'failure.json').open('x') as handle: emit(handle, failure)
        except FileExistsError:
            pass
        progress('failed'); raise


if __name__ == '__main__':
    main()
