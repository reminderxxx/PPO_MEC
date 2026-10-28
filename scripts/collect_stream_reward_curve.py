"""Audit one frozen curve campaign, replay declared finals and export its figures."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stream_reward_curve import (METHODS, SEEDS, CHECKPOINT_EPISODES,
    file_hash, json_text, save_new, build_policy, loaded_policy, validate_plan)
from scripts.train_joint_stream_control import make_env
from src.envs.core.joint_stream_rules import StreamCacheRule
from scripts.plot_stream_reward_curve import fixed_policy

EVENT_KEYS = ('frame_index', 'vehicle_id', 'request_id', 'workflow_id', 'status',
    'control', 'control_planning', 'cache_request', 'state_transfer_link_frames',
    'execution_rsu_id', 'service_target_changed', 'association_handoff')


def read_rows(path):
    with Path(path).open() as stream: return [json.loads(line) for line in stream if line.strip()]


def key(row): return (row['phase'], row['episode'], row['case_id'])


def near(actual, expected):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected) > 1e-6:
        raise ValueError('native reward or MC label differs')


def audit_cell(plan, method, seed, campaign_sha=None):
    home = Path(plan['outputs'][method][str(seed)])
    validate_plan(plan, method, seed, home, verify_files=False)
    complete = json.loads((home/'complete.json').read_text())
    if (complete['status'] != 'complete' or complete['method'] != method or complete['seed'] != seed
            or complete['source_commit'] != plan['source_commit'] or (home/'failure.json').exists()
            or (campaign_sha is not None and complete['campaign_sha256'] != campaign_sha)):
        raise ValueError('missing, failed or wrong cell')
    required = {'protocol.json','training.jsonl','evaluation.jsonl','updates.jsonl',
                'checkpoints.jsonl','receipts.jsonl.gz','controls.jsonl.gz'}
    if set(complete['output_sha256']) != required:
        raise ValueError('cell mandatory file inventory differs')
    for name, sha in complete['output_sha256'].items():
        if Path(name).name != name or file_hash(home/name) != sha: raise ValueError('cell file hash mismatch')
    protocol = json.loads((home/'protocol.json').read_text())
    if protocol['plan'] != plan or protocol['campaign_sha256'] != complete['campaign_sha256']:
        raise ValueError('cell protocol or campaign identity differs')
    training = read_rows(home/'training.jsonl'); evaluation = read_rows(home/'evaluation.jsonl')
    updates = read_rows(home/'updates.jsonl'); checkpoints = read_rows(home/'checkpoints.jsonl')
    if ([r['episode'] for r in training] != list(range(1, 97))
            or [r['case_id'] for r in training] != plan['training_schedule']
            or [r['episode'] for r in updates] != list(range(4, 97, 4))
            or [r['episode'] for r in checkpoints] != CHECKPOINT_EPISODES):
        raise ValueError('omitted or reordered training/update/checkpoint')
    expected_eval = [(e, c) for e in CHECKPOINT_EPISODES for c in
                     (plan['training_case_ids'] if e == 96 else plan['probe_case_ids'])]
    if [(r['episode'], r['case_id']) for r in evaluation] != expected_eval:
        raise ValueError('evaluation case pool changes across checkpoints')
    cp_by = {r['episode']:r for r in checkpoints}; cumulative = 0
    for row in training:
        cumulative += row['steps']
        if row['cumulative_training_frames'] != cumulative or row['cumulative_updates'] != row['episode']//4:
            raise ValueError('actual training frame/update axis differs')
    expected_policy = build_policy(method, seed)
    fresh_tensor_sha = hashlib.sha256(b''.join(v.numpy().tobytes() for v in expected_policy.state_dict().values())).hexdigest()
    if checkpoints[0]['tensor_sha256'] != fresh_tensor_sha:
        raise ValueError('initial checkpoint is not the declared fresh random seed')
    expected_optimizers = {n for n in ('optimizer','surrogate_optimizer') if getattr(expected_policy,n,None) is not None}
    for row in updates:
        counts = row['optimizer_step_calls']; ep = row['episode']
        if (set(counts) != expected_optimizers or any(type(v) is not int or v < 0 for v in counts.values())
                or row['observed_gradient_steps'] != sum(counts.values()) or row['native_epochs'] != 4
                or row['cumulative_updates'] != ep//4 or row['method'] != method or row['seed'] != seed
                or row['cumulative_training_frames'] != training[ep-1]['cumulative_training_frames']
                or row['gradient_steps_reported'] != row['native_losses'].get('gradient_steps')):
            raise ValueError('observed optimizer/update resource ledger differs')
    if complete['gradient_steps'] != sum(r['observed_gradient_steps'] for r in updates):
        raise ValueError('terminal observed gradient count differs')
    for record in checkpoints:
        path = Path(record['path'])
        if path != home/'checkpoints'/('ep%03d.pt' % record['episode']) or file_hash(path) != record['sha256']:
            raise ValueError('checkpoint path/hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        identity = {k:v for k,v in record.items() if k not in ('path', 'sha256', 'tensor_sha256')}
        tensor_sha = hashlib.sha256(b''.join(v.numpy().tobytes() for v in payload['state_dict'].values())).hexdigest()
        if (payload.get('reward_curve_identity') != identity or payload['options'] != record['options']
                or record['method'] != method or record['seed'] != seed or tensor_sha != record['tensor_sha256']
                or record['policy_class'] != type(expected_policy).__name__
                or record['native_method'] != expected_policy.method or record['options'] != expected_policy.options
                or record['policy_initialization'] != 'random_from_scratch_no_policy_parent'
                or record['cumulative_updates'] != record['episode']//4):
            raise ValueError('checkpoint method/class/options/tensors differ')
        expected_frames = sum(r['steps'] for r in training[:record['episode']])
        if record['cumulative_training_frames'] != expected_frames: raise ValueError('checkpoint frame axis differs')
    records = {key(r):r for r in training+evaluation}
    if len(records) != 144: raise ValueError('duplicate native episode record')
    if any(r['method'] != method or r['seed'] != seed for r in records.values()):
        raise ValueError('native episode method/seed identity differs')
    rewards = defaultdict(list); frame_hashes = defaultdict(hashlib.sha256)
    with gzip.open(home/'receipts.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line); k = key(row)
            if (k not in records or row['method'] != method or row['seed'] != seed
                    or row['frame_index'] != len(rewards[k])): raise ValueError('frame identity/gap differs')
            if row['receipt']['reward_components'] != row['reward_components']:
                raise ValueError('flattened reward components differ from native receipt')
            near(row['reward'], sum(row['reward_components'].values()))
            rewards[k].append(row['reward'])
            frame_hashes[k].update(json_text({n:row[n] for n in ('frame_index','reward','reward_components')}).encode())
    suffixes = {}
    for k, record in records.items():
        if (len(rewards[k]) != record['steps'] or not record['physical_terminal']
                or record['bootstrap_team_tail'] != 0 or not 0 < record['steps'] <= 480):
            raise ValueError('native tail, terminal, idle or frame count differs')
        near(sum(rewards[k]), record['reward'])
        suffixes[k] = np.cumsum(np.asarray(rewards[k][::-1], dtype=float)/10)[::-1]
        if k[0] == 'evaluation':
            cp = cp_by[record['episode']]
            if (record['checkpoint_sha256'] != cp['sha256']
                    or record['cumulative_training_frames'] != cp['cumulative_training_frames']
                    or record['is_curve_probe'] != (record['case_id'] in plan['probe_case_ids'])
                    or record['is_fixed_final'] != (record['episode'] == 96)):
                raise ValueError('probe or final checkpoint binding differs')
    event_hashes = defaultdict(hashlib.sha256); decisions = 0
    rollout_frames = defaultdict(list); rollout_counts = defaultdict(int); rollout_ends = defaultdict(list)
    event_identities = defaultdict(lambda:defaultdict(set))
    with gzip.open(home/'controls.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line); k = key(row)
            if k not in records or row['method'] != method or row['seed'] != seed:
                raise ValueError('control outside declared episodes')
            if row['record_kind'] == 'native_control':
                event = row['event']; f = event['frame_index']; identity = (event['vehicle_id'],event['request_id'])
                if not 0 <= f < records[k]['steps'] or identity in event_identities[k][f]:
                    raise ValueError('native event frame/vehicle/request duplicated or outside episode')
                event_identities[k][f].add(identity)
                event_hashes[k].update(json_text(row['event']).encode())
            elif row['record_kind'] == 'decision_rollout':
                f = row['frame_index']; n = row['initial_request_count']
                if not 0 <= f < records[k]['steps'] or n <= 0: raise ValueError('invalid rollout frame/fleet')
                for field, target in [('returns', suffixes[k][f]), ('team_credit_targets', suffixes[k][f]/n)]:
                    if not np.allclose(row[field], target, rtol=2e-5, atol=1e-5): raise ValueError('complete MC suffix label differs')
                if not np.allclose(row['credit_targets'], suffixes[k][f]/n, rtol=2e-5, atol=1e-5):
                    raise ValueError('native credit target differs')
                size = len(row['vehicle_ids'])
                if not set(zip(row['vehicle_ids'],row['request_ids'])) <= event_identities[k][f]:
                    raise ValueError('decision rollout has no matching physical control event')
                if len(set(row['vehicle_ids'])) != size or any(len(row[field]) != size for field in
                        ('request_ids','actions','head_logs','active_heads','values','returns','credit_targets','team_credit_targets')):
                    raise ValueError('decision rollout vehicle/field dimensions differ')
                if any(not np.isfinite(np.asarray(row[field],dtype=float)).all() for field in
                       ('actions','head_logs','active_heads','values','returns','credit_targets','team_credit_targets')):
                    raise ValueError('nonfinite decision rollout')
                rollout_frames[k].append(f);rollout_counts[k]+=size;rollout_ends[k].append(row['episode_end'])
                decisions += len(row['vehicle_ids'])
            else: raise ValueError('unknown control record')
    for k, record in records.items():
        frames = rollout_frames[k]; ends = rollout_ends[k]
        if (rollout_counts[k] != record['decision_rows'] or frames != sorted(set(frames))
                or (ends and ends != [False]*(len(ends)-1)+[True])):
            raise ValueError('missing/duplicate/nonterminal decision rollouts')
    train_frames = sum(r['steps'] for r in training); eval_frames = sum(r['steps'] for r in evaluation)
    if (train_frames != complete['training_frames'] or eval_frames != complete['evaluation_frames']
            or train_frames > 46080 or eval_frames > 23040 or complete['update_calls'] != 24):
        raise ValueError('cell resource totals differ')
    return dict(method=method, seed=seed, training_frames=train_frames, evaluation_frames=eval_frames,
        update_calls=24, gradient_steps=complete['gradient_steps'], elapsed_seconds=complete['elapsed_seconds'],
        decision_rows=decisions, initial_parameter_count=sum(p.numel() for p in expected_policy.parameters()),
        evaluation=evaluation, checkpoints=checkpoints,
        frame_digests={'|'.join(map(str,k)):v.hexdigest() for k,v in frame_hashes.items()},
        control_digests={'|'.join(map(str,k)):v.hexdigest() for k,v in event_hashes.items()})


def evaluate_final(plan, data, source, case, agent, path, *, learned):
    env = make_env(case, data, source)
    if learned: agent.reset_controller()
    total = 0.; frames = hashlib.sha256(); controls_hash = hashlib.sha256()
    with gzip.open(path, 'xt') as stream:
        while not env._control_episode_done():
            if env.frame_index >= 480 or time.time() >= plan['global_deadline_unix']:
                raise RuntimeError('collector physical/time budget exhausted')
            f = env.frame_index; decisions = env.decision_states()
            commands = agent.act_controls(decisions, f, deterministic=True)[0] if learned else agent.act(decisions, f)
            _, reward, _, info = env.step_controls(commands); parts = info['frame_cost_receipt']['reward_components']
            near(reward, sum(parts.values())); total += reward
            packed = dict(frame_index=f, reward=sum(parts.values()), reward_components=parts)
            frames.update(json_text(packed).encode())
            for event in info['events']:
                if learned: event['control_planning'] = agent.last_diagnostics['by_vehicle'][event['vehicle_id']]
                controls_hash.update(json_text({k:event[k] for k in EVENT_KEYS if k in event}).encode())
            stream.write(json_text(dict(**packed, receipt=info['frame_cost_receipt'], events=info['events'])))
    return dict(reward=total, **env.summary(), frame_digest=frames.hexdigest(), control_digest=controls_hash.hexdigest())


def collect(plan_path, expected_sha=None):
    if expected_sha is not None and file_hash(plan_path) != expected_sha:
        raise ValueError('campaign SHA changed before collection')
    plan = json.loads(plan_path.read_text()); root = Path(plan['root']); output = root/'collection'
    for relative, sha in plan['source_fingerprints'].items():
        if file_hash(ROOT/relative) != sha: raise ValueError('frozen scientific source changed')
    output.mkdir(exist_ok=False)
    audited = [audit_cell(plan, m, s, file_hash(plan_path)) for m in METHODS for s in SEEDS]
    source = json.loads(Path(plan['protocol']['path']).read_text())
    data = torch.load(plan['protocol']['protocol_pt'], weights_only=False)
    cases = {c['case_id']:c for c in data['cases']['train'] if c['case_id'] in plan['training_case_ids']}
    rules = []
    for cid in plan['training_case_ids']:
        row = evaluate_final(plan, data, source, cases[cid], StreamCacheRule('gain_prefetch'),
                             output/('rule_'+cid+'.jsonl.gz'), learned=False)
        rules.append(dict(case_id=cid, **row))
    by = {(r['method'],r['seed']):r for r in audited}; replays = []
    for item in plan['replay_episodes']:
        cell = by[item['method'], item['seed']]; cp = cell['checkpoints'][-1]
        agent = loaded_policy(Path(cp['path']), cp, type(build_policy(item['method'], item['seed'])))
        row = evaluate_final(plan, data, source, cases[item['case_id']], agent,
            output/('replay_%s_%s_%s.jsonl.gz' % (item['method'], item['seed'], item['case_id'])), learned=True)
        original = next(r for r in cell['evaluation'] if r['episode']==96 and r['case_id']==item['case_id'])
        k = 'evaluation|96|'+item['case_id']
        near(row['reward'], original['reward'])
        if (row['steps'] != original['steps'] or row['completed_requests'] != original['completed_requests']
                or row['frame_digest'] != cell['frame_digests'][k] or row['control_digest'] != cell['control_digests'][k]):
            raise ValueError('independent final replay differs')
        replays.append(dict(**item, **row))
    train = sum(c['training_frames'] for c in audited); ev = sum(c['evaluation_frames'] for c in audited)
    auxiliary = sum(r['steps'] for r in rules+replays)
    if (sum(r['steps'] for r in rules) > plan['maximum_rule_frames']
            or sum(r['steps'] for r in replays) > plan['maximum_replay_frames']):
        raise ValueError('rule or replay physical cap exceeded')
    if train+ev+auxiliary > plan['maximum_total_physical_frames']: raise ValueError('campaign physical cap exceeded')
    duplicates = []
    for i,m in enumerate(METHODS):
        for other in METHODS[i+1:]:
            count = sum(by[m,s]['control_digests'][k] == by[other,s]['control_digests'].get(k)
                for s in SEEDS for k in by[m,s]['control_digests'] if k.startswith('evaluation|'))
            duplicates.append(dict(left=m,right=other,equal_evaluation_control_traces=count,total_pairs=240))
    audit = dict(status='curve_receipts_and_checkpoints_audited', campaign_sha256=file_hash(plan_path),
        source_commit=plan['source_commit'], cells=audited, rules=rules, replays=replays,
        probe_case_ids=plan['probe_case_ids'],training_case_ids=plan['training_case_ids'],
        physical_frames=dict(training=train,evaluation=ev,rules_and_replay=auxiliary,total=train+ev+auxiliary),
        behavior_trace_comparisons=duplicates, final_selection='episode96_fixed_no_peak',
        scope='Consumed train learning evidence only; no formal/hidden/generalization or external-paper-fidelity claim',
        paper_goal_complete=False)
    audit_path = output/'audit.json'; save_new(audit_path, audit)
    series = []
    for method in METHODS:
        values = []; frames = []
        for seed in SEEDS:
            cell = by[method,seed]
            values.append([float(np.mean([r['reward'] for r in cell['evaluation']
                if r['episode']==ep and r['case_id'] in plan['probe_case_ids']])) for ep in CHECKPOINT_EPISODES])
            frames.append([r['cumulative_training_frames'] for r in cell['checkpoints']])
        series.append(dict(method=method,seed_returns=values,seed_frames=frames))
    plot = dict(contract='stream_reward_curve_plot_v1', selection='fixed_final_no_peak_selection',
        audit_path=str(audit_path), audit_sha256=file_hash(audit_path), seeds=SEEDS, episodes=CHECKPOINT_EPISODES,
        series=series,gain_rule_probe_mean=float(np.mean([r['reward'] for r in rules if r['case_id'] in plan['probe_case_ids']])))
    plot_path = output/'plot_source.json'; save_new(plot_path, plot)
    figure = fixed_policy(plot_path, output/'figure')
    with (output/'resource_bill.csv').open('w',newline='') as stream:
        fields = ['method','seed','training_frames','evaluation_frames','update_calls','gradient_steps','elapsed_seconds','initial_parameter_count']
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:c[k] for k in fields} for c in audited)
    summary = dict(status='reward_curve_delivered', audit=str(audit_path), figure=str(figure),
        plot_data=str(plot_path),physical_frames=audit['physical_frames'],all_seeds_retained=True,
        manuscript_promoted=False,paper_goal_complete=False)
    save_new(output/'complete.json', summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--expected_sha',required=True)
    args=parser.parse_args();torch.set_num_threads(1);print(json.dumps(collect(args.campaign,args.expected_sha)))
