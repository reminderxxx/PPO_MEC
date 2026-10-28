"""Frozen, from-scratch policy learning curves over complete physical sessions.

The fitted input predictor is shared and frozen. Evaluation is a separate loaded
policy, isolated from all training RNG state. Native rewards are never reshaped.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_stream_main_opportunity import select_train_cases
from scripts.train_joint_stream_control import make_agent, make_env, stream_episode
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent

CONTRACT = 'stream_reward_curve_v1'
METHODS = ['main', 'mappo', 'ppo', 'dqn']
SEEDS = [7, 17, 27, 37, 47]
CHECKPOINT_EPISODES = list(range(0, 97, 8))
_RESERVED_FAILURE_BYTES = 65536


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n'


def save_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json_text(value))


def validate_plan(plan, method, seed, output, *, root=ROOT, verify_files=True):
    """Reject scientific drift before a claim or any physical interaction."""
    if plan.get('contract') != CONTRACT:
        raise ValueError('unexpected reward curve contract')
    if plan.get('methods') != METHODS or plan.get('seeds') != SEEDS:
        raise ValueError('fixed method and seed pool required')
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('unknown cell')
    if plan.get('episodes_per_cell') != 96 or plan.get('update_group_size') != 4 or plan.get('update_epochs') != 4:
        raise ValueError('fixed episodes and native update schedule required')
    if plan.get('checkpoint_episodes') != CHECKPOINT_EPISODES:
        raise ValueError('fixed initial and every-eight checkpoint schedule required')
    ids = plan.get('training_case_ids', [])
    schedule = plan.get('training_schedule', [])
    probes = plan.get('probe_case_ids', [])
    if len(ids) != 12 or len(set(ids)) != 12 or len(schedule) != 96:
        raise ValueError('twelve cases and ninety-six episodes required')
    if any(sorted(schedule[i:i + 12]) != sorted(ids) for i in range(0, 96, 12)):
        raise ValueError('each input-selected twelve-case round must be balanced')
    if len(probes) != 3 or len(set(probes)) != 3 or not set(probes) <= set(ids):
        raise ValueError('three fixed train probes required')
    budget = plan.get('budgets', {})
    for key, value in [('maximum_frames_per_episode', 480),
                       ('maximum_training_frames_per_cell', 46080),
                       ('maximum_evaluation_frames_per_cell', 23040)]:
        if budget.get(key) != value:
            raise ValueError('fixed frame budget required: ' + key)
    for key, maximum in [('maximum_cell_wall_seconds', 172800),
                         ('maximum_cell_disk_bytes', 503316480)]:
        if not isinstance(budget.get(key), (int, float)) or not 0 < budget[key] <= maximum:
            raise ValueError('invalid cell hard budget: ' + key)
    if str(Path(output).resolve()) != plan.get('outputs', {}).get(method, {}).get(str(seed)):
        raise ValueError('cell output differs from frozen binding')
    if any(plan.get(k) for k in ('parent_checkpoint', 'policy_parent', 'resume_checkpoint')):
        raise ValueError('policy must start from random initialization')
    if verify_files:
        protocol = plan['protocol']
        for name, sha_key in [('path', 'sha256'), ('protocol_pt', 'protocol_pt_sha256')]:
            if file_hash(protocol[name]) != protocol[sha_key]:
                raise ValueError('input asset hash mismatch: ' + name)
        fingerprints = plan.get('source_fingerprints', {})
        required = {'scripts/run_stream_reward_curve.py', 'scripts/train_joint_stream_control.py',
                    'src/agents/sa_ghmappo_agent.py', 'src/agents/ppo_agent.py',
                    'src/agents/mappo_agent.py', 'src/agents/dqn_agent.py',
                    'src/envs/core/multi_request_stream_env.py'}
        if not required <= set(fingerprints):
            raise ValueError('incomplete frozen source inventory')
        for relative, sha in fingerprints.items():
            target = (Path(root) / relative).resolve()
            if Path(root).resolve() not in target.parents or file_hash(target) != sha:
                raise ValueError('source hash mismatch: ' + relative)
    return plan


def build_policy(method, seed, episodes=96):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if method == 'main':
        return IntentAwareStreamSAGHMAPPOAgent(seed=seed, training_budget=episodes,
                                             return_gamma=1., prediction_enabled=True)
    return make_agent(method, seed, episodes)


@contextmanager
def isolated_rng():
    """Policy loading and probe construction cannot advance training random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class CellBudget:
    def __init__(self, output, limits, *, deadline=None):
        self.output, self.limits = Path(output), limits
        self.started = time.monotonic()
        self.deadline = deadline
        self.frames = {'train': 0, 'evaluation': 0}

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())

    def check(self, extra_bytes=0):
        if time.monotonic() - self.started >= self.limits['maximum_cell_wall_seconds']:
            raise RuntimeError('cell wall budget exhausted')
        if self.deadline is not None and time.time() >= self.deadline:
            raise RuntimeError('campaign global deadline exhausted')
        if self.disk_bytes() + extra_bytes + _RESERVED_FAILURE_BYTES > self.limits['maximum_cell_disk_bytes']:
            raise RuntimeError('cell disk budget exhausted')

    def before_step(self, phase):
        if time.monotonic() - self.started >= self.limits['maximum_cell_wall_seconds']:
            raise RuntimeError('cell wall budget exhausted')
        if self.deadline is not None and time.time() >= self.deadline:
            raise RuntimeError('campaign global deadline exhausted')
        key = 'maximum_training_frames_per_cell' if phase == 'train' else 'maximum_evaluation_frames_per_cell'
        if self.frames[phase] >= self.limits[key]:
            raise RuntimeError(phase + ' frame budget exhausted')


class LimitedBinaryWriter:
    """Charge compressed writes before accepting bytes, including gzip trailers."""
    def __init__(self, path, budget):
        self.file = Path(path).open('xb', buffering=0)
        self.budget = budget

    def write(self, data):
        self.budget.check(len(data))
        return self.file.write(data)

    def flush(self):
        return self.file.flush()

    def tell(self):
        return self.file.tell()

    def close(self):
        return self.file.close()


class GzipRecords:
    def __init__(self, path, budget):
        self.raw = LimitedBinaryWriter(path, budget)
        self.stream = gzip.GzipFile(fileobj=self.raw, mode='wb', mtime=0)

    def write_record(self, row):
        self.stream.write(json_text(row).encode())

    def close(self):
        try:
            self.stream.close()
        finally:
            self.raw.close()


class GuardedEnv:
    """Preserve the native environment while enforcing caps before every step."""
    def __init__(self, env, budget, phase):
        self.env, self.budget, self.phase = env, budget, phase
        maximum = budget.limits['maximum_frames_per_episode']
        if env.frame_index != 0 or not 0 < len(env.frames) <= maximum:
            raise ValueError('complete fresh physical session must fit frame cap')
        self.receipt_count = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step_controls(self, controls):
        if self.env._control_episode_done():
            raise RuntimeError('post-terminal interaction rejected')
        self.budget.before_step(self.phase)
        before = self.env.frame_index
        result = self.env.step_controls(controls)
        self.budget.frames[self.phase] += 1
        self.receipt_count += 1
        if self.env.frame_index != before + 1:
            raise ValueError('physical frame did not advance exactly once')
        return result


class ReceiptSink:
    def __init__(self, trace, identity):
        self.trace, self.identity, self.count = trace, identity, 0

    def write(self, text):
        receipt = json.loads(text)
        parts = receipt['reward_components']
        reward = sum(parts.values())
        if not math.isfinite(reward):
            raise ValueError('nonfinite native reward')
        self.trace.write_record(dict(self.identity, record_kind='frame_receipt',
            frame_index=self.count, reward=reward, reward_components=parts, receipt=receipt))
        self.count += 1


class ControlSink:
    def __init__(self, trace, identity):
        self.trace, self.identity = trace, identity

    def write(self, text):
        event = json.loads(text)
        keys = ('frame_index', 'vehicle_id', 'request_id', 'workflow_id', 'status',
                'control', 'control_planning', 'cache_request', 'state_transfer_link_frames',
                'execution_rsu_id', 'service_target_changed', 'association_handoff')
        self.trace.write_record(dict(self.identity, record_kind='native_control',
            event={key: event[key] for key in keys if key in event}))


def physical_episode(env, agent, budget, phase, identity, receipts, controls, *, deterministic=False):
    wrapped = GuardedEnv(env, budget, phase)
    sink = ReceiptSink(receipts, identity)
    row, rolls = stream_episode(wrapped, agent, deterministic=deterministic,
                                 receipt_file=sink, event_file=ControlSink(controls, identity), identity=identity)
    if not env._control_episode_done() or sink.count != env.frame_index or row['steps'] != sink.count:
        raise ValueError('complete physical terminal and frame receipts required')
    if set(env._released) != {a.request_id for a in env.arrivals}:
        raise ValueError('scheduled late arrivals were not all released')
    if rolls and not rolls[-1]['episode_end']:
        raise ValueError('native terminal team tail missing')
    for roll in rolls:
        witness = dict(identity, record_kind='decision_rollout', frame_index=roll['frame_index'],
            vehicle_ids=roll['vehicle_ids'], request_ids=roll['request_ids'],
            episode_end=roll['episode_end'], initial_request_count=roll['initial_request_count'])
        for key in ('actions', 'head_logs', 'active_heads', 'values', 'returns',
                    'credit_targets', 'team_credit_targets', 'cost_targets'):
            if key in roll:
                witness[key] = roll[key].detach().cpu().tolist()
        controls.write_record(witness)
    row.update(physical_terminal=True, bootstrap_team_tail=0.,
               last_scheduled_release_frame=max(a.frame for a in env.arrivals),
               released_request_ids=sorted(env._released))
    return row, rolls


def checkpoint_policy(agent, path, identity, budget):
    """Bind the native serialized identity, class, options and all learned tensors."""
    budget.check(32 * 1024 * 1024)
    temporary = path.with_suffix('.native.tmp')
    agent.save_checkpoint(temporary)
    payload = torch.load(temporary, map_location='cpu', weights_only=False)
    expected = dict(identity, policy_class=type(agent).__name__, options=agent.options,
        native_method=agent.method, checkpoint_contract=agent.checkpoint_contract,
        policy_initialization='random_from_scratch_no_policy_parent',
        predictor_initialization='shared_frozen_previously_fitted_input_predictor')
    payload['reward_curve_identity'] = expected
    with path.open('xb') as stream:
        torch.save(payload, stream)
    temporary.unlink()
    record = dict(expected, path=str(path.resolve()), sha256=file_hash(path), tensor_sha256=tensor_hash(agent))
    budget.check()
    return record


def loaded_policy(path, record, policy_class):
    if file_hash(path) != record['sha256']:
        raise ValueError('checkpoint bytes changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = {k: v for k, v in record.items() if k not in ('path', 'sha256', 'tensor_sha256')}
    if payload.get('reward_curve_identity') != expected or payload.get('options') != record['options']:
        raise ValueError('checkpoint method, class or option identity mismatch')
    if record['policy_class'] != policy_class.__name__ or payload['contract'] != policy_class.checkpoint_contract:
        raise ValueError('checkpoint policy class mismatch')
    agent = policy_class.from_checkpoint(path)
    if agent.method != record['native_method'] or agent.options != record['options'] or tensor_hash(agent) != record['tensor_sha256']:
        raise ValueError('loaded policy identity or tensors differ')
    return agent


def native_update(agent, rollouts, epochs):
    """Observe exact optimizer calls without changing the native update algorithm."""
    counts, originals = {}, []
    for name in ('optimizer', 'surrogate_optimizer'):
        optimizer = getattr(agent, name, None)
        if optimizer is None or any(optimizer is obj for obj, _ in originals):
            continue
        original = optimizer.step
        counts[name] = 0
        def counted(*args, _name=name, _step=original, **kwargs):
            value = _step(*args, **kwargs)
            counts[_name] += 1
            return value
        originals.append((optimizer, original))
        optimizer.step = counted
    try:
        losses = agent.update(rollouts, epochs=epochs)
    finally:
        for optimizer, original in originals:
            optimizer.step = original
    return losses, counts


def evaluate_checkpoint(agent, record, case_ids, cases, data, source, budget, receipts, controls, output):
    rows = []
    training_tensor = tensor_hash(agent)
    with isolated_rng():
        evaluated = loaded_policy(Path(record['path']), record, type(agent))
        for case_id in case_ids:
            identity = dict(method=record['method'], seed=record['seed'], phase='evaluation',
                episode=record['episode'], checkpoint_episode=record['episode'], case_id=case_id,
                checkpoint_sha256=record['sha256'])
            row, _ = physical_episode(make_env(cases[case_id], data, source), evaluated, budget,
                'evaluation', identity, receipts, controls, deterministic=True)
            row = dict(identity, **row, cumulative_training_frames=record['cumulative_training_frames'],
                       cumulative_updates=record['cumulative_updates'])
            output.write(json_text(row)); output.flush(); rows.append(row)
    if tensor_hash(agent) != training_tensor:
        raise ValueError('evaluation mutated training policy tensors')
    return rows


def run_cell(plan, campaign_path, method, seed, output, *, verify=True):
    """One create-only cell; all failures retain their original artifacts."""
    output = Path(output)
    validate_plan(plan, method, seed, output, verify_files=verify)
    if verify:
        git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        if dirty or git != plan['source_commit']:
            raise ValueError('clean frozen campaign commit required')
    claim = Path(campaign_path).parent / ('cell_claim_%s_seed%s.json' % (method, seed))
    save_new(claim, dict(pid=os.getpid(), claimed_at=utc_now(), method=method, seed=seed,
        campaign_sha256=file_hash(campaign_path), output=str(output.resolve()), command=sys.argv))
    output.mkdir(parents=True, exist_ok=False)
    budget = CellBudget(output, plan['budgets'], deadline=plan.get('global_deadline_unix'))
    progress = dict(contract=CONTRACT, method=method, seed=seed, status='running',
        source_commit=plan['source_commit'], campaign_sha256=file_hash(campaign_path),
        policy_initialization='random_from_scratch_no_policy_parent',
        predictor_initialization='shared_frozen_previously_fitted_input_predictor',
        training_episodes=0, evaluation_episodes=0, training_frames=0, evaluation_frames=0,
        update_calls=0, gradient_steps=0, checkpoints=[])
    receipts = controls = None
    try:
        save_new(output / 'protocol.json', dict(plan=plan, **progress))
        source = json.loads(Path(plan['protocol']['path']).read_text())
        data = torch.load(plan['protocol']['protocol_pt'], weights_only=False)
        selected = select_train_cases(data)
        if [c['case_id'] for c in selected] != plan['training_case_ids']:
            raise ValueError('input-selected cases differ from prospective cases')
        cases = {c['case_id']: c for c in selected}
        if any(data['sessions'][c['session']]['split'] != 'train' for c in selected):
            raise ValueError('training and curve probes must be consumed train only')
        agent = build_policy(method, seed, plan['episodes_per_cell'])
        if int(agent.completed_updates) != 0 or agent.optimizer.state:
            raise ValueError('fresh policy and optimizer required')
        receipts = GzipRecords(output / 'receipts.jsonl.gz', budget)
        controls = GzipRecords(output / 'controls.jsonl.gz', budget)
        (output / 'checkpoints').mkdir()
        buffer = []
        with (output / 'training.jsonl').open('x') as training, \
             (output / 'evaluation.jsonl').open('x') as evaluation, \
             (output / 'updates.jsonl').open('x') as updates, \
             (output / 'checkpoints.jsonl').open('x') as checkpoints:
            def snapshot(episode):
                identity = dict(method=method, seed=seed, episode=episode,
                    cumulative_training_frames=progress['training_frames'],
                    cumulative_updates=progress['update_calls'])
                record = checkpoint_policy(agent, output / 'checkpoints' / ('ep%03d.pt' % episode), identity, budget)
                checkpoints.write(json_text(record)); checkpoints.flush()
                progress['checkpoints'].append(record)
                ids = plan['training_case_ids'] if episode == 96 else plan['probe_case_ids']
                # The final twelve include all three probes; no duplicate final probes.
                class ProbeFlagWriter:
                    def write(self, text):
                        value = json.loads(text)
                        value['is_curve_probe'] = value['case_id'] in plan['probe_case_ids']
                        value['is_fixed_final'] = episode == 96
                        return evaluation.write(json_text(value))
                    def flush(self):
                        evaluation.flush()
                rows = evaluate_checkpoint(agent, record, ids, cases, data, source, budget,
                                           receipts, controls, ProbeFlagWriter())
                progress['evaluation_episodes'] += len(rows)
                progress['evaluation_frames'] = budget.frames['evaluation']
            snapshot(0)
            for episode, case_id in enumerate(plan['training_schedule'], 1):
                identity = dict(method=method, seed=seed, phase='train', episode=episode, case_id=case_id)
                row, rolls = physical_episode(make_env(cases[case_id], data, source), agent, budget,
                    'train', identity, receipts, controls)
                buffer.extend(rolls)
                if episode % 4 == 0:
                    before = int(agent.completed_updates)
                    losses, optimizer_calls = native_update(agent, buffer, plan['update_epochs'])
                    if int(agent.completed_updates) != before + 1:
                        raise ValueError('native update counter mismatch')
                    buffer = []
                    progress['update_calls'] += 1
                    # Exact observed optimizer calls, with separate PPO auxiliary and DQN replay accounting.
                    progress['gradient_steps'] += sum(optimizer_calls.values())
                    updates.write(json_text(dict(identity, cumulative_training_frames=budget.frames['train'],
                        cumulative_updates=progress['update_calls'], native_losses=losses,
                        native_epochs=plan['update_epochs'], gradient_steps_reported=losses.get('gradient_steps'),
                        optimizer_step_calls=optimizer_calls, observed_gradient_steps=sum(optimizer_calls.values()))))
                    updates.flush()
                progress['training_episodes'] = episode
                progress['training_frames'] = budget.frames['train']
                training.write(json_text(dict(identity, **row, cumulative_training_frames=progress['training_frames'],
                    cumulative_updates=progress['update_calls']))); training.flush()
                if episode in plan['checkpoint_episodes']:
                    snapshot(episode)
                budget.check()
                (output / 'manifest.json').write_text(json_text(progress))
            if buffer or progress['training_episodes'] != 96 or progress['update_calls'] != 24 or progress['evaluation_episodes'] != 48:
                raise ValueError('fixed cell episode/update/evaluation counts differ')
        receipts.close(); receipts = None
        controls.close(); controls = None
        budget.check()
        progress.update(status='complete', completed_at=utc_now(), elapsed_seconds=time.monotonic() - budget.started,
            output_sha256={p.name: file_hash(p) for p in output.iterdir()
                           if p.is_file() and p.name not in ('manifest.json', 'complete.json')})
        (output / 'manifest.json').write_text(json_text(progress))
        save_new(output / 'complete.json', progress)
        return progress
    except BaseException as error:
        # Do not overwrite any parent artifacts or turn an error into a retried cell.
        progress.update(status='failed_preserved_do_not_retry', failed_at=utc_now(),
            training_frames=budget.frames['train'], evaluation_frames=budget.frames['evaluation'],
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - budget.started)
        save_new(output / 'failure.json', progress)
        raise
    finally:
        for stream in (receipts, controls):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass  # Preserve the first failure; partial gzip remains an explicit failed artifact.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_cell(json.loads(args.campaign.read_text()), args.campaign, args.method, args.seed, args.output)


if __name__ == '__main__':
    main()
