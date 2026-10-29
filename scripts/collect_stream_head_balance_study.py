"""Audit a new head-balanced candidate against a read-only frozen main reference.

No old audit, baseline training, rule evaluation, or reference replay is rerun.
All five candidate cells and the three prospective final replays are retained.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_stream_reward_curve import read_rows, key, near, EVENT_KEYS
from scripts.run_stream_head_balance_study import (METHOD, METHODS, SEEDS,
    CHECKPOINT_EPISODES, CONTRACT, file_hash, json_text, save_new, utc_now,
    build_policy, loaded_policy, validate_plan, read_reference, GzipRecords)
from scripts.train_joint_stream_control import make_env
from src.agents.sa_ghmappo_agent import HeadBalancedIntentStreamSAGHMAPPOAgent


class CollectionGuard:
    def __init__(self, plan):
        self.plan=plan;self.root=Path(plan['root']);self.count=0
        self.replay_frames=0;self.started=time.monotonic()

    def check(self, extra_bytes=0):
        if time.time() >= self.plan['global_deadline_unix']:
            raise RuntimeError('head-balance global deadline exhausted')
        if time.monotonic()-self.started >= self.plan['maximum_wall_seconds']:
            raise RuntimeError('head-balance collector wall cap exhausted')
        size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if size+extra_bytes+65536 > self.plan['maximum_disk_bytes']:
            raise RuntimeError('head-balance total disk cap exhausted')

    def tick(self):
        if time.time() >= self.plan['global_deadline_unix']:
            raise RuntimeError('head-balance global deadline exhausted')
        self.count+=1
        if self.count%5000==0:self.check()

    def before_step(self):
        self.tick()
        if self.replay_frames >= self.plan['maximum_replay_frames']:
            raise RuntimeError('prospective replay physical cap exhausted')


def save_guarded(path, value, guard):
    text=json_text(value)
    guard.check(len(text.encode()))
    with Path(path).open('x') as stream:stream.write(text)
    guard.check()


def native_scalar_near(actual, expected, *, maximum_ulps=2):
    """Bound recorded reducer scalars in the native declared floating dtype."""
    value=float(expected)
    if not np.isfinite(actual) or not np.isfinite(value):
        raise ValueError('nonfinite native head reduction scalar')
    if float(torch.tensor(actual,dtype=expected.dtype)) != actual:
        raise ValueError('reported head scalar is not representable in native dtype')
    neighbors=[float(torch.nextafter(expected,torch.full_like(expected,direction)))
               for direction in (float('inf'),-float('inf'))]
    spacing=max(abs(n-value) for n in neighbors if np.isfinite(n))
    if abs(actual-value)>maximum_ulps*spacing:
        raise ValueError('head reduction exceeds declared two-ULP arithmetic bound')


def audit_head_updates(plan, updates, active_rows):
    summaries=[]
    for update in updates:
        ep=update['episode'];loss=update['native_losses']
        if loss.get('policy_head_reduction') != HeadBalancedIntentStreamSAGHMAPPOAgent.head_reduction:
            raise ValueError('head reduction identity differs')
        rows=np.asarray([row for n in range(ep-3,ep+1) for row in
            active_rows['train',n,plan['training_schedule'][n-1]]],dtype=bool)
        if rows.shape != (loss['samples'],3):raise ValueError('head sample provenance differs')
        batches=loss.get('head_balance_batches',[])
        if (len(batches) != loss['gradient_steps']
                or len(batches) != update['optimizer_step_calls']['optimizer']
                or len(batches) != update['optimizer_step_calls']['surrogate_optimizer']
                or len(batches) > 4*int(np.ceil(len(rows)/512))):
            raise ValueError('head minibatch and optimizer count differs')
        epoch=[];completed=0;policy=[];entropy=[];total=np.zeros(3,dtype=int)
        for batch in batches:
            if batch.get('reduction_dtype') != 'torch.float32':
                raise ValueError('candidate head reduction must retain native float32 dtype')
            dtype=torch.float32
            indices=batch['batch_indices']
            if (not indices or len(indices)>512 or len(set(indices))!=len(indices)
                    or any(type(i) is not int or not 0<=i<len(rows) for i in indices)):
                raise ValueError('head minibatch row identity differs')
            expected=min(512,len(rows)-len(epoch))
            if len(indices)!=expected or set(indices)&set(epoch):
                raise ValueError('head minibatch permutation coverage differs')
            epoch.extend(indices)
            if len(epoch)==len(rows):completed+=1;epoch=[]
            counts=rows[indices].sum(0)
            enabled=torch.as_tensor(counts>0,dtype=dtype)
            denominator=enabled.sum().clamp(min=1)
            weights=enabled/denominator
            if (not np.array_equal(batch['active_counts'],counts)
                    or not torch.equal(torch.as_tensor(batch['objective_head_weights'],dtype=dtype),weights)):
                raise ValueError('head counts or objective weights differ from physical rollouts')
            means=np.asarray(batch['objective_head_means'],dtype=float)
            ent=np.asarray(batch['entropy_head_means'],dtype=float)
            if means.shape!=(3,) or ent.shape!=(3,) or not np.isfinite(means).all() or not np.isfinite(ent).all():
                raise ValueError('head objective telemetry nonfinite or wrong shape')
            native_scalar_near(batch['policy_loss'],
                -(torch.as_tensor(means,dtype=dtype)*enabled).sum()/denominator)
            native_scalar_near(batch['balanced_entropy'],
                (torch.as_tensor(ent,dtype=dtype)*enabled).sum()/denominator)
            policy.append(batch['policy_loss']);entropy.append(batch['balanced_entropy']);total+=counts
        if (not loss['kl_early_stop'] and (completed!=4 or epoch)):
            raise ValueError('head minibatches omit native four epochs')
        near(loss['policy_loss'],float(np.mean(policy)) if policy else 0.)
        summaries.append(dict(episode=ep,head_active_counts_once=rows.sum(0).tolist(),
            head_active_counts_across_optimized_minibatches=total.tolist(),
            optimized_minibatches=len(batches),complete_epochs=completed,
            final_epoch_rows=len(epoch),kl_early_stop=loss['kl_early_stop'],
            balanced_entropy_mean=float(np.mean(entropy)) if entropy else 0.))
    return summaries


def audit_cell(plan, method, seed, campaign_sha=None, *, guard=None, case_shapes=None):
    if guard: guard.check()
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
        if guard: guard.check()
        path = Path(record['path'])
        if path != home/'checkpoints'/('ep%03d.pt' % record['episode']) or file_hash(path) != record['sha256']:
            raise ValueError('checkpoint path/hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        identity = {k:v for k,v in record.items() if k not in ('path', 'sha256', 'tensor_sha256')}
        tensor_sha = hashlib.sha256(b''.join(v.numpy().tobytes() for v in payload['state_dict'].values())).hexdigest()
        if (payload.get('contract') != expected_policy.checkpoint_contract
                or payload.get('reward_curve_identity') != identity or payload['options'] != record['options']
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
    receipt_gate_frames = defaultdict(dict)
    with gzip.open(home/'receipts.jsonl.gz', 'rt') as stream:
        for line in stream:
            if guard: guard.tick()
            row = json.loads(line); k = key(row)
            if (k not in records or row['method'] != method or row['seed'] != seed
                    or row['frame_index'] != len(rewards[k])): raise ValueError('frame identity/gap differs')
            if row['receipt']['reward_components'] != row['reward_components']:
                raise ValueError('flattened reward components differ from native receipt')
            receipt_gate_frames[k][row['frame_index']] = {
                field:row['receipt'][field] for field in
                ('state_transfer_bytes','state_transfer_link_frames')}
            receipt_gate_frames[k][row['frame_index']]['completed_nodes']=row['reward_components']['completed_nodes']
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
                    or record.get('checkpoint_episode') != record['episode']
                    or record['cumulative_training_frames'] != cp['cumulative_training_frames']
                    or record.get('cumulative_updates') != record['episode']//4
                    or record['is_curve_probe'] != (record['case_id'] in plan['probe_case_ids'])
                    or record['is_fixed_final'] != (record['episode'] == 96)):
                raise ValueError('probe or final checkpoint binding differs')
    event_hashes = defaultdict(hashlib.sha256); decisions = 0
    rollout_frames = defaultdict(list); rollout_counts = defaultdict(int); rollout_ends = defaultdict(list)
    event_identities = defaultdict(lambda:defaultdict(set)); training_active = defaultdict(list)
    native_gate_frames = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    native_gate_totals = defaultdict(lambda:defaultdict(float))
    completed_requests = defaultdict(set)
    completed_node_ids = defaultdict(lambda:defaultdict(set))
    with gzip.open(home/'controls.jsonl.gz', 'rt') as stream:
        for line in stream:
            if guard: guard.tick()
            row = json.loads(line); k = key(row)
            if k not in records or row['method'] != method or row['seed'] != seed:
                raise ValueError('control outside declared episodes')
            if row['record_kind'] == 'native_control':
                event = row['event']; f = event['frame_index']; identity = (event['vehicle_id'],event['request_id'])
                if not 0 <= f < records[k]['steps'] or identity in event_identities[k][f]:
                    raise ValueError('native event frame/vehicle/request duplicated or outside episode')
                witness=row.get('gate_native_event',{})
                expected_fields={'frame_index','vehicle_id','request_id','node_id','workflow_completed',
                                 'handoff_failed','state_transfer_bytes','state_transfer_link_frames'}
                if set(witness)!=expected_fields or any(witness[n]!=event[n] for n in
                        ('frame_index','vehicle_id','request_id','state_transfer_link_frames')):
                    raise ValueError('native gate witness identity differs')
                if (type(witness['workflow_completed']) is not bool
                        or type(witness['handoff_failed']) is not bool
                        or type(event['service_target_changed']) is not bool
                        or not isinstance(witness['node_id'],str) or not witness['node_id']):
                    raise ValueError('native gate witness type differs')
                failed=bool(event['service_target_changed'] and event['status'] in
                            {'state_restore','state_link_queue'})
                if witness['handoff_failed'] != failed:
                    raise ValueError('native v3 handoff formula differs')
                rid=event['request_id'];nid=witness['node_id']
                if rid in completed_requests[k]:
                    raise ValueError('completed request emitted later controls')
                if case_shapes is not None and (rid not in case_shapes[k[2]] or nid not in case_shapes[k[2]][rid]):
                    raise ValueError('native request/node absent from frozen case')
                delivered=event['status'] in {'node_completed','vehicle_fallback_completed'}
                if delivered:
                    if nid in completed_node_ids[k][rid]:raise ValueError('node delivered twice')
                    completed_node_ids[k][rid].add(nid)
                if witness['workflow_completed']:
                    if not delivered:raise ValueError('workflow completed without node delivery')
                    if case_shapes is not None and completed_node_ids[k][rid]!=case_shapes[k[2]][rid]:
                        raise ValueError('workflow completed before all frozen DAG nodes delivered')
                    completed_requests[k].add(rid)
                native_gate_frames[k][f]['completed_nodes']+=int(delivered)
                native_gate_totals[k]['handoff_failed_count']+=int(failed)
                for field in ('state_transfer_bytes','state_transfer_link_frames'):
                    value=witness[field]
                    if type(value) not in (int,float) or not np.isfinite(value) or value<0:
                        raise ValueError('native state cost invalid')
                    native_gate_frames[k][f][field]+=value
                    native_gate_totals[k][field]+=value
                event_identities[k][f].add(identity)
                # Preserve exact comparability to the sealed original projection.
                event_hashes[k].update(json_text({n:event[n] for n in EVENT_KEYS if n in event}).encode())
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
                if any(len(values)!=3 or any(type(v) is not bool for v in values)
                       for values in row['active_heads']):
                    raise ValueError('active head witness must be three literal booleans')
                if k[0] == 'train': training_active[k].extend(row['active_heads'])
                rollout_frames[k].append(f);rollout_counts[k]+=size;rollout_ends[k].append(row['episode_end'])
                decisions += len(row['vehicle_ids'])
            else: raise ValueError('unknown control record')
    for k, record in records.items():
        frames = rollout_frames[k]; ends = rollout_ends[k]
        if (rollout_counts[k] != record['decision_rows'] or frames != sorted(set(frames))
                or (ends and ends != [False]*(len(ends)-1)+[True])):
            raise ValueError('missing/duplicate/nonterminal decision rollouts')
        for f, receipt in receipt_gate_frames[k].items():
            for field in ('state_transfer_bytes','state_transfer_link_frames','completed_nodes'):
                if receipt[field] != native_gate_frames[k][f][field]:
                    raise ValueError('native state events differ from frame receipt')
        for field in ('state_transfer_bytes','state_transfer_link_frames','handoff_failed_count'):
            if record[field] != native_gate_totals[k][field]:
                raise ValueError('summary gate cost differs from native ledger: '+field)
        if record['completed_requests'] != len(completed_requests[k]):
            raise ValueError('summary completed request gate differs from native ledger')
        request_rows={r['request_id']:r for r in record['per_request']}
        if (len(request_rows)!=len(record['per_request']) or len(request_rows)!=record['request_count']
                or (case_shapes is not None and set(request_rows)!=set(case_shapes[k[2]]))):
            raise ValueError('summary request identity pool differs')
        for rid,r in request_rows.items():
            if (r['completed'] != (rid in completed_requests[k])
                    or r['completed_nodes'] != len(completed_node_ids[k][rid])
                    or (case_shapes is not None and r['total_nodes']!=len(case_shapes[k[2]][rid]))):
                raise ValueError('summary completion differs from native delivered nodes')
        if (record['completed_nodes'] != sum(len(v) for v in completed_node_ids[k].values())
                or record['completed_nodes'] != sum(native_gate_frames[k][f]['completed_nodes']
                                                     for f in receipt_gate_frames[k])):
            raise ValueError('summary completed node count differs')
    train_frames = sum(r['steps'] for r in training); eval_frames = sum(r['steps'] for r in evaluation)
    if (train_frames != complete['training_frames'] or eval_frames != complete['evaluation_frames']
            or train_frames > 46080 or eval_frames > 23040 or complete['update_calls'] != 24):
        raise ValueError('cell resource totals differ')
    head_diagnostics = audit_head_updates(plan, updates, training_active)
    optimizer_calls = {name:sum(r['optimizer_step_calls'][name] for r in updates)
                       for name in expected_optimizers}
    gate_metrics={'|'.join(map(str,k)):dict(native_gate_totals[k],
        completed_requests=len(completed_requests[k]),
        completed_nodes=sum(len(v) for v in completed_node_ids[k].values())) for k in records}
    return dict(head_update_diagnostics=head_diagnostics, optimizer_step_calls=optimizer_calls,
        native_gate_metrics=gate_metrics,gate_metrics_audited=True,
        method=method, seed=seed, training_frames=train_frames, evaluation_frames=eval_frames,
        update_calls=24, gradient_steps=complete['gradient_steps'], elapsed_seconds=complete['elapsed_seconds'],
        decision_rows=decisions, initial_parameter_count=sum(p.numel() for p in expected_policy.parameters()),
        evaluation=evaluation, checkpoints=checkpoints,
        frame_digests={'|'.join(map(str,k)):v.hexdigest() for k,v in frame_hashes.items()},
        control_digests={'|'.join(map(str,k)):v.hexdigest() for k,v in event_hashes.items()})


def replay_final(plan, data, source, case, agent, path, guard):
    env=make_env(case,data,source);agent.reset_controller()
    total=0.;frames=hashlib.sha256();controls_hash=hashlib.sha256()
    trace=GzipRecords(path,guard)
    try:
        while not env._control_episode_done():
            if env.frame_index>=480:raise RuntimeError('replay episode exceeds physical cap')
            guard.before_step();f=env.frame_index
            commands=agent.act_controls(env.decision_states(),f,deterministic=True)[0]
            _,reward,_,info=env.step_controls(commands);guard.replay_frames+=1
            parts=info['frame_cost_receipt']['reward_components'];near(reward,sum(parts.values()));total+=reward
            packed=dict(frame_index=f,reward=sum(parts.values()),reward_components=parts)
            frames.update(json_text(packed).encode())
            for event in info['events']:
                event['control_planning']=agent.last_diagnostics['by_vehicle'][event['vehicle_id']]
                controls_hash.update(json_text({k:event[k] for k in EVENT_KEYS if k in event}).encode())
            trace.write_record(dict(**packed,receipt=info['frame_cost_receipt'],events=info['events']))
    finally:trace.close()
    guard.check()
    return dict(reward=total,**env.summary(),frame_digest=frames.hexdigest(),control_digest=controls_hash.hexdigest())


def validate_initial_reference(candidate, reference, probe_ids):
    """Compare already-planned initial probes, without any extra interaction."""
    if ([c['seed'] for c in candidate] != SEEDS
            or [c['seed'] for c in reference] != SEEDS):
        raise ValueError('initial comparison must retain the five declared seeds')
    for fresh, original in zip(candidate, reference):
        if (fresh['seed'] != original['seed']
                or fresh['checkpoints'][0]['tensor_sha256'] != original['checkpoints'][0]['tensor_sha256']):
            raise ValueError('candidate and reference initial policy tensors differ')
        new = {r['case_id']:r for r in fresh['evaluation'] if r['episode']==0}
        old = {r['case_id']:r for r in original['evaluation'] if r['episode']==0}
        if set(new) != set(probe_ids) or set(old) != set(probe_ids):
            raise ValueError('initial fixed probe identity differs')
        for cid in probe_ids:
            near(new[cid]['reward'],old[cid]['reward'])
            digest='evaluation|0|'+cid
            if (new[cid]['steps'] != old[cid]['steps']
                    or fresh['frame_digests'][digest] != original['frame_digests'][digest]
                    or fresh['control_digests'][digest] != original['control_digests'][digest]):
                raise ValueError('candidate initial forward/physical control differs from reference')


def paired_gate(candidate, reference, acceptance):
    fields=['reward','completed_requests','state_transfer_bytes','state_transfer_link_frames','handoff_failed_count']
    old={(c['seed'],r['case_id']):r for c in reference for r in c['evaluation'] if r['episode']==96}
    new={(c['seed'],r['case_id']):r for c in candidate for r in c['evaluation'] if r['episode']==96}
    if set(old)!=set(new) or len(new)!=60:raise ValueError('fixed final sixty-case comparison differs')
    rows=[dict(seed=s,case_id=cid,**{field+'_difference':new[s,cid][field]-old[s,cid][field] for field in fields}) for s,cid in sorted(new)]
    means={field:float(np.mean([r[field+'_difference'] for r in rows])) for field in fields}
    seed_diffs={str(s):float(np.mean([r['reward_difference'] for r in rows if r['seed']==s])) for s in SEEDS}
    tol=acceptance['numeric_tolerance'];nonnegative=sum(v>=-tol for v in seed_diffs.values())
    checks=dict(reward_strictly_positive=means['reward']>acceptance['minimum_paired_reward_gain'],
        completed_requests_noninferior=means['completed_requests']>=acceptance['minimum_completed_request_difference']-tol,
        at_least_four_nonregressing_seeds=nonnegative>=acceptance['minimum_nonregressing_seeds'])
    for field in fields[2:]:checks[field+'_nonincreasing']=means[field]<=acceptance['maximum_'+field+'_difference']+tol
    return dict(accepted=all(checks.values()),checks=checks,paired_mean_differences=means,
                reward_difference_by_seed=seed_diffs,nonregressing_seed_count=nonnegative,
                scope='Consumed-train development gate, not independent statistical significance',
                registry_promoted=False,canonical_promoted=False),rows


def export_curve(output,candidate,reference,guard):
    # Fixed probes, every declared seed, no smoothing or peak selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=[];fig,ax=plt.subplots(figsize=(6.7,3.9))
    for method,cells,color,style in [('original_main',reference,'#0072B2','--'),(METHOD,candidate,'#D55E00','-')]:
        seed_values=[]
        for cell in cells:
            values=[]
            for ep in CHECKPOINT_EPISODES:
                rows=[r for r in cell['evaluation'] if r['episode']==ep and r['is_curve_probe']]
                if len(rows)!=3:raise ValueError('curve three-probe identity differs')
                value=float(np.mean([r['reward'] for r in rows]));values.append(value)
                table.append(dict(method=method,seed=cell['seed'],episode=ep,reward=value,
                    actual_training_frames=next(c['cumulative_training_frames'] for c in cell['checkpoints'] if c['episode']==ep)))
            seed_values.append(values);ax.plot(CHECKPOINT_EPISODES,values,color=color,alpha=.19,lw=.7)
        a=np.asarray(seed_values);mean=a.mean(0);sd=a.std(0,ddof=1)
        ax.plot(CHECKPOINT_EPISODES,mean,color=color,ls=style,lw=1.8,label='Original main' if method=='original_main' else 'Head-balanced main')
        ax.fill_between(CHECKPOINT_EPISODES,mean-sd,mean+sd,color=color,alpha=.13,lw=0)
    ax.set(xlabel='Completed training episodes',ylabel='Greedy evaluation episode return',xlim=(0,96))
    ax.grid(alpha=.2,lw=.5);ax.legend(frameon=False,fontsize=9);fig.tight_layout()
    for suffix in ('png','pdf','svg'):
        guard.check(8*1024*1024);fig.savefig(output/('paired_reward_learning_curve.'+suffix),dpi=600)
    plt.close(fig)
    guard.check(1024*1024)
    with (output/'paired_curve_points.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    guard.check(4096)
    with (output/'caption.txt').open('x') as stream:
        stream.write('All five seeds, each point averages the same three consumed-train probes; mean plus/minus one sample SD, not CI. No smoothing. Original main is the frozen e00d5fa control and is not retrained. Policies start from random initialization; the shared GRU was fitted previously. Fixed ep96 selection; final12 results are separate. Actual physical frames and optimizer costs may differ. No independent/generalization/whole-paper claim.\n')
    guard.check()
    return table


def collect(plan_path,expected_sha):
    plan_path=Path(plan_path)
    if file_hash(plan_path)!=expected_sha:raise ValueError('candidate campaign hash differs')
    plan=json.loads(plan_path.read_text())
    validate_plan(plan,METHOD,SEEDS[0],plan['outputs'][METHOD][str(SEEDS[0])])
    _,anchor=read_reference(plan);root=Path(plan['root']);output=root/'collection'
    output.mkdir(exist_ok=False);guard=CollectionGuard(plan)
    try:
        guard.check()
        source=json.loads(Path(plan['protocol']['path']).read_text())
        data=torch.load(plan['protocol']['protocol_pt'],weights_only=False)
        cases={c['case_id']:c for c in data['cases']['train'] if c['case_id'] in plan['training_case_ids']}
        case_shapes={cid:{a.request_id:{n.node_id for n in a.workflow.nodes} for a in case['arrivals']}
                     for cid,case in cases.items()}
        cells=[audit_cell(plan,METHOD,s,expected_sha,guard=guard,case_shapes=case_shapes) for s in SEEDS]
        references=[c for s in SEEDS for c in anchor['cells'] if c['method']=='main' and c['seed']==s]
        if len(references)!=5:raise ValueError('reference five-seed main pool differs')
        validate_initial_reference(cells,references,plan['probe_case_ids'])
        replays=[]
        for item in plan['replay_episodes']:
            cell=next(c for c in cells if c['seed']==item['seed']);cp=cell['checkpoints'][-1]
            agent=loaded_policy(Path(cp['path']),cp,type(build_policy(METHOD,item['seed'])))
            row=replay_final(plan,data,source,cases[item['case_id']],agent,
                output/('replay_%s_%s.jsonl.gz'%(item['seed'],item['case_id'])),guard)
            original=next(r for r in cell['evaluation'] if r['episode']==96 and r['case_id']==item['case_id'])
            k='evaluation|96|'+item['case_id'];near(row['reward'],original['reward'])
            if (row['steps']!=original['steps'] or row['completed_requests']!=original['completed_requests']
                    or row['frame_digest']!=cell['frame_digests'][k]
                    or row['control_digest']!=cell['control_digests'][k]):
                raise ValueError('prospective final replay differs')
            for field in ('state_transfer_bytes','state_transfer_link_frames','handoff_failed_count'):
                if row[field]!=original[field]:raise ValueError('prospective replay gate cost differs: '+field)
            replays.append(dict(**item,**row))
        train=sum(c['training_frames'] for c in cells);ev=sum(c['evaluation_frames'] for c in cells)
        if (train>plan['maximum_training_frames'] or ev>plan['maximum_evaluation_frames']
                or train+ev+guard.replay_frames>plan['maximum_total_physical_frames']):
            raise ValueError('candidate total physical cap exceeded')
        gate,paired=paired_gate(cells,references,plan['acceptance'])
        audit=dict(status='head_balance_candidate_audited',campaign_sha256=expected_sha,
            source_commit=plan['source_commit'],reference=plan['reference'],cells=cells,replays=replays,
            reference_endpoint_qualification=plan['reference_endpoint_qualification'],
            physical_frames=dict(training=train,evaluation=ev,replays=guard.replay_frames,total=train+ev+guard.replay_frames),
            acceptance=gate,all_seeds_retained=True,final_selection='episode96_fixed_no_peak',
            manuscript_promoted=False,paper_goal_complete=False,
            telemetry_boundary='Head counts/weights/permutation and reported loss arithmetic independently recomputed; no counterfactual full gradient or actor forward replay is claimed')
        save_guarded(output/'audit.json',audit,guard)
        guard.check(1024*1024)
        with (output/'final60_paired.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        with (output/'resource_bill.csv').open('x',newline='') as stream:
            fields=['method','seed','training_frames','evaluation_frames','update_calls',
                    'optimizer_step_calls','surrogate_optimizer_step_calls','gradient_steps',
                    'elapsed_seconds','initial_parameter_count']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for c in cells:
                row={k:c[k] for k in fields if k not in ('optimizer_step_calls','surrogate_optimizer_step_calls')}
                row['optimizer_step_calls']=c['optimizer_step_calls']['optimizer']
                row['surrogate_optimizer_step_calls']=c['optimizer_step_calls']['surrogate_optimizer']
                writer.writerow(row)
        export_curve(output,cells,references,guard)
        figure_files=['paired_reward_learning_curve.png','paired_reward_learning_curve.pdf',
                      'paired_reward_learning_curve.svg','paired_curve_points.csv','caption.txt']
        save_guarded(output/'figure_manifest.json',dict(audit_path=str(output/'audit.json'),
            audit_sha256=file_hash(output/'audit.json'),all_seeds=SEEDS,
            checkpoint_episodes=CHECKPOINT_EPISODES,
            output_sha256={name:file_hash(output/name) for name in figure_files},
            smoothing='none',spread='one_sample_standard_deviation_not_CI'),guard)
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
        result=dict(status='head_balance_study_delivered',audit=str(output/'audit.json'),
            audit_sha256=file_hash(output/'audit.json'),acceptance=gate,physical_frames=audit['physical_frames'],
            output_sha256=files,all_seeds_retained=True,registry_promoted=False,
            canonical_promoted=False,paper_goal_complete=False)
        save_guarded(output/'complete.json',result,guard)
        return result
    except BaseException as error:
        save_new(output/'failure.json',dict(status='failed_preserved_do_not_retry',
            failed_at=utc_now(),error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),replay_frames=guard.replay_frames,
            elapsed_seconds=time.monotonic()-guard.started))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--expected_sha',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    print(json.dumps(collect(args.campaign,args.expected_sha)))
