"""One opt-in head-balanced main cell; all physical/RNG/checkpoint helpers are shared.

This independent plan does not modify or resume the sealed reward-curve run.
The parent launcher owns the single campaign claim and two-worker envelope.
"""
import argparse
import json
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
from scripts.run_stream_reward_curve import (CHECKPOINT_EPISODES, SEEDS, CellBudget,
    GzipRecords, file_hash, json_text, save_new, utc_now, GuardedEnv, ReceiptSink,
    checkpoint_policy, native_update, isolated_rng, loaded_policy, tensor_hash)
from scripts.probe_stream_main_opportunity import select_train_cases
from scripts.train_joint_stream_control import make_env, stream_episode
from src.agents.sa_ghmappo_agent import HeadBalancedIntentStreamSAGHMAPPOAgent

CONTRACT = 'stream_head_balance_study_v1'
METHODS = ['head_balanced_main']
METHOD = METHODS[0]
REFERENCE_CAMPAIGN_SHA = 'abd965ec2617cae544ed38f9cf607f9c9c87383e324dda8211783f46824357e1'
REFERENCE_AUDIT_SHA = 'd795a848637b7828214cd41582387075e4ad50a7269682a4d756674165f1a948'
REPLAYS = [dict(method=METHODS[0], seed=s, case_id=c) for s,c in
           [(7,'train_014'), (27,'train_031'), (47,'train_055')]]
ACCEPTANCE = dict(minimum_paired_reward_gain=0.0,
    minimum_completed_request_difference=0.0, minimum_nonregressing_seeds=4,
    maximum_state_transfer_bytes_difference=0.0,
    maximum_state_transfer_link_frames_difference=0.0,
    maximum_handoff_failed_count_difference=0.0, numeric_tolerance=1e-8)


def read_reference(plan):
    """Read the already-audited control, without re-auditing old frames or loading data."""
    reference = plan['reference']
    if (reference.get('campaign_sha256') != REFERENCE_CAMPAIGN_SHA
            or reference.get('audit_sha256') != REFERENCE_AUDIT_SHA):
        raise ValueError('reference is not the exact frozen scientific control')
    for name in ('campaign', 'audit'):
        if file_hash(reference[name+'_path']) != reference[name+'_sha256']:
            raise ValueError('reference '+name+' hash mismatch')
    original = json.loads(Path(reference['campaign_path']).read_text())
    audit = json.loads(Path(reference['audit_path']).read_text())
    if (original.get('contract') != 'stream_reward_curve_v1'
            or original.get('source_commit') != 'e00d5faf9c3f73a689174237245ed6f85394b2f2'
            or audit.get('status') != 'curve_receipts_and_checkpoints_audited'
            or audit.get('campaign_sha256') != reference['campaign_sha256']
            or audit.get('source_commit') != original['source_commit']
            or len(audit.get('cells',[])) != 20
            or {(c['method'],c['seed']) for c in audit.get('cells',[])} !=
               {(m,s) for m in ['main','mappo','ppo','dqn'] for s in SEEDS}):
        raise ValueError('old reference is not the frozen complete twenty-cell curve')
    for name in ('protocol','training_case_ids','training_schedule','probe_case_ids',
                 'episodes_per_cell','update_epochs','update_group_size','checkpoint_episodes'):
        if plan.get(name) != original.get(name):
            raise ValueError('candidate differs from frozen comparison input: '+name)
    endpoint_binding=plan.get('reference_endpoint_qualification',{})
    if set(endpoint_binding)!={'path','sha256'}:
        raise ValueError('native final sixty-endpoint reference qualification required')
    if file_hash(endpoint_binding['path'])!=endpoint_binding['sha256']:
        raise ValueError('reference endpoint qualification hash mismatch')
    endpoints=json.loads(Path(endpoint_binding['path']).read_text())
    if (endpoints.get('status')!='reference_final60_native_endpoints_qualified'
            or endpoints.get('campaign_sha256')!=REFERENCE_CAMPAIGN_SHA
            or endpoints.get('audit_sha256')!=REFERENCE_AUDIT_SHA
            or endpoints.get('source_commit')!=original['source_commit']):
        raise ValueError('reference native endpoint qualification identity differs')
    old={(c['seed'],r['case_id']):r for c in audit['cells'] if c['method']=='main'
         for r in c.get('evaluation',[]) if r['episode']==96}
    rows=endpoints.get('endpoints',[])
    if (len(rows)!=60 or len(old)!=60
            or {(r['seed'],r['case_id']) for r in rows}!=set(old)):
        raise ValueError('reference native endpoint qualification omits or duplicates cases')
    for row in rows:
        if row.get('episode')!=96 or any(row[field]!=old[row['seed'],row['case_id']][field]
                for field in ('reward','completed_requests','state_transfer_bytes',
                              'state_transfer_link_frames','handoff_failed_count')):
            raise ValueError('qualified native reference endpoints differ from frozen audit')
    return original, audit


def validate_plan(plan, method, seed, output, *, root=ROOT, verify_files=True):
    if (plan.get('contract') != CONTRACT or plan.get('methods') != METHODS
            or plan.get('seeds') != SEEDS or method != METHODS[0] or seed not in SEEDS):
        raise ValueError('fixed opt-in method and five seeds required')
    if (plan.get('episodes_per_cell') != 96 or plan.get('update_group_size') != 4
            or plan.get('update_epochs') != 4
            or plan.get('checkpoint_episodes') != CHECKPOINT_EPISODES):
        raise ValueError('fixed complete-session schedule required')
    ids=plan.get('training_case_ids',[])
    if (len(ids)!=12 or len(set(ids))!=12 or plan.get('training_schedule') != ids*8
            or len(plan.get('probe_case_ids',[]))!=3
            or len(set(plan.get('probe_case_ids',[])))!=3
            or not set(plan['probe_case_ids']) <= set(ids)):
        raise ValueError('same twelve-case order for all eight rounds required')
    if str(Path(output).resolve()) != plan.get('outputs',{}).get(method,{}).get(str(seed)):
        raise ValueError('cell output differs from frozen binding')
    paths=[v for values in plan.get('outputs',{}).values() for v in values.values()]
    if (set(plan.get('outputs',{})) != set(METHODS)
            or set(plan['outputs'][method]) != {str(s) for s in SEEDS}
            or len(paths)!=len(set(paths))):
        raise ValueError('fixed unique five-cell matrix required')
    if any(plan.get(k) for k in ('parent_checkpoint','policy_parent','resume_checkpoint')):
        raise ValueError('random initialized policy required')
    budget=plan.get('budgets',{})
    fixed={'maximum_frames_per_episode':480,'maximum_training_frames_per_cell':46080,
           'maximum_evaluation_frames_per_cell':23040}
    if any(budget.get(k)!=v for k,v in fixed.items()):
        raise ValueError('fixed cell physical budget required')
    for k,cap in [('maximum_cell_wall_seconds',86400),('maximum_cell_disk_bytes',503316480)]:
        if type(budget.get(k)) not in (int,float) or not 0<budget[k]<=cap:
            raise ValueError('invalid cell hard cap: '+k)
    caps={'maximum_training_frames':230400,'maximum_evaluation_frames':115200,
          'maximum_replay_frames':1440,'maximum_total_physical_frames':347040,
          'maximum_wall_seconds':86400,'maximum_disk_bytes':3221225472}
    if (any(plan.get(k)!=v for k,v in caps.items())
            or type(plan.get('workers')) is not int or plan['workers'] not in (1,2)):
        raise ValueError('fixed campaign envelope required')
    if plan.get('replay_episodes') != REPLAYS or plan.get('acceptance') != ACCEPTANCE:
        raise ValueError('fixed replay identities and acceptance gate required')
    if verify_files:
        original,_ = read_reference(plan)
        destination = Path(plan['root']).resolve()
        original_root = Path(original['root']).resolve()
        if (destination == original_root or original_root in destination.parents
                or destination in original_root.parents):
            raise ValueError('candidate cannot write inside or above sealed reference root')
        if any(destination not in Path(p).resolve().parents for p in paths):
            raise ValueError('candidate cells must remain inside new campaign root')
        protocol=plan['protocol']
        for name,sha_key in [('path','sha256'),('protocol_pt','protocol_pt_sha256')]:
            if file_hash(protocol[name]) != protocol[sha_key]:
                raise ValueError('input asset hash mismatch')
        fingerprints=plan.get('source_fingerprints',{})
        required={'scripts/run_stream_head_balance_study.py',
                  'scripts/collect_stream_head_balance_study.py',
                  'scripts/run_stream_reward_curve.py','scripts/collect_stream_reward_curve.py',
                  'scripts/train_joint_stream_control.py','src/agents/ppo_agent.py',
                  'src/agents/sa_ghmappo_agent.py'}
        if not required <= set(fingerprints):raise ValueError('incomplete source inventory')
        changed={'src/agents/ppo_agent.py','src/agents/sa_ghmappo_agent.py'}
        for relative,sha in original['source_fingerprints'].items():
            if relative not in changed and fingerprints.get(relative)!=sha:
                raise ValueError('unapproved original scientific source change: '+relative)
        for relative,sha in fingerprints.items():
            target=(Path(root)/relative).resolve()
            if Path(root).resolve() not in target.parents or file_hash(target)!=sha:
                raise ValueError('source hash mismatch: '+relative)
    return plan


def build_policy(method, seed, episodes=96):
    if method not in METHODS:raise ValueError('only opt-in head-balanced main permitted')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return HeadBalancedIntentStreamSAGHMAPPOAgent(seed=seed, training_budget=episodes,
                                               return_gamma=1.,prediction_enabled=True)


class ControlSink:
    """Preserve legacy control projection and separate native gate witnesses."""
    def __init__(self,trace,identity):self.trace,self.identity=trace,identity

    def write(self,text):
        event=json.loads(text)
        keys=('frame_index','vehicle_id','request_id','workflow_id','status',
              'control','control_planning','cache_request','state_transfer_link_frames',
              'execution_rsu_id','service_target_changed','association_handoff')
        gate_fields=('frame_index','vehicle_id','request_id','node_id','workflow_completed',
                     'handoff_failed','state_transfer_bytes','state_transfer_link_frames')
        self.trace.write_record(dict(self.identity,record_kind='native_control',
            event={key:event[key] for key in keys if key in event},
            gate_native_event={key:event[key] for key in gate_fields}))


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



class BudgetedTextWriter:
    """Charge telemetry and episode rows before accepting their UTF-8 bytes."""
    def __init__(self, stream, budget):
        self.stream,self.budget=stream,budget

    def write(self,text):
        self.budget.check(len(text.encode()))
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


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
        if verify:
            _, reference = read_reference(plan)
            original = next(c for c in reference['cells'] if c['method']=='main' and c['seed']==seed)
            if tensor_hash(agent) != original['checkpoints'][0]['tensor_sha256']:
                raise ValueError('candidate initial policy differs from frozen original main')
        receipts = GzipRecords(output / 'receipts.jsonl.gz', budget)
        controls = GzipRecords(output / 'controls.jsonl.gz', budget)
        (output / 'checkpoints').mkdir()
        buffer = []
        with (output / 'training.jsonl').open('x') as training, \
             (output / 'evaluation.jsonl').open('x') as evaluation, \
             (output / 'updates.jsonl').open('x') as updates, \
             (output / 'checkpoints.jsonl').open('x') as checkpoints:
            training,evaluation,updates,checkpoints = [BudgetedTextWriter(stream,budget)
                for stream in (training,evaluation,updates,checkpoints)]
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    run_cell(json.loads(args.campaign.read_text()),args.campaign,args.method,args.seed,args.output)


if __name__=='__main__':main()
