"""Read-only checks of the capped repair preflight; no performance promotion."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import run_fair_reward_training as fair
from preflight_sagh_execution_repair import AGENTS, SEEDS, STEPS_PER_CELL, CLASSES, write_json


def records(path):
    with path.open() as stream:
        for line in stream:yield json.loads(line)


def episode_key(meta):
    return tuple(meta[k] for k in ('phase','agent','seed','unit','training_steps','window_id','workflow_id'))


def audit(root):
    receipt=json.loads((root/'execution_receipt.json').read_text())
    completion=json.loads((root/'completion.json').read_text())
    if receipt['hard_total_interaction_cap']!=6144 or receipt['reward_changes'] or receipt['old_checkpoint_loads']:
        raise ValueError('Unauthorized preflight protocol')
    if completion['formal_training_started'] or completion['training_steps']!=6144 or completion['total_environment_steps']>10368:
        raise ValueError('Formal/global budget mismatch')
    source=Path(receipt['source_root'])
    for path,expected in receipt['source_sha256'].items():
        if fair.sha(source/path)!=expected:raise ValueError(f'Source changed: {path}')
    results=[]
    for name in AGENTS:
        for seed in SEEDS:
            cell=root/'cells'/name/f'seed{seed}';complete=json.loads((cell/'complete.json').read_text())
            for path,expected in complete['files_sha256'].items():
                if fair.sha(cell/path)!=expected:raise ValueError(f'Cell file changed: {name}/{seed}/{path}')
            phases=Counter();classes=defaultdict(set);episodes=defaultdict(lambda:[0,0.]);max_error=0.
            for row in records(cell/'steps.jsonl'):
                meta=row['metadata'];phases[meta['phase']]+=1;classes[meta['phase']].add(meta['window_class'])
                reward=row['reward_breakdown']
                expected=sum(reward[k] for k in ('positive_offset','service_reward','continuity_bonus','mechanism_exploration_bonus'))-sum(reward[k] for k in ('delay_penalty','cache_miss_penalty','migration_cost','constraint_penalty'))
                max_error=max(max_error,abs(expected-row['native_reward']))
                if max_error>1e-6 or reward['positive_offset']!=0:raise ValueError('Native reward mismatch')
                key=episode_key(meta);episodes[key][0]+=1;episodes[key][1]+=row['native_reward']
                action=row['action_info'];mask=row['action_mask']
                if not mask[row['actual_action']]:raise ValueError('Illegal executed action')
                if name=='sa_ghmappo':
                    if action['execution_mapping_contract']!='shared_train_eval_v1' or not row['stages']:
                        raise ValueError('Missing full gate diagnostics')
                    if action['probability_contract']!='sampled_base_and_option_v1':raise ValueError('Unverified sampled likelihood')
                    lp=action['sampled_base_log_prob']
                    option=action['option_gate']
                    if option.get('enabled'):lp+=option['option_log_prob']
                    if abs(lp-action['log_prob'])>2e-5:raise ValueError('Sampled joint mismatch')
            for row in records(cell/'episodes.jsonl'):
                count,total=episodes.pop(episode_key(row['metadata']))
                if count!=row['steps'] or abs(total-row['summary']['reward_breakdown']['total']['sum'])>1e-5:
                    raise ValueError('Episode/step identity mismatch')
            if episodes or phases['train']!=STEPS_PER_CELL or phases['validation']!=complete['validation_steps']:
                raise ValueError('Episode or budget mismatch')
            if any(classes[phase]!=set(CLASSES) for phase in ('train','validation')):raise ValueError('Missing frozen class')
            updates=list(records(cell/'updates.jsonl'))
            if len(updates)!=3 or [u['training_steps'] for u in updates]!=[256,512,768]:raise ValueError('Update opportunity mismatch')
            for row in updates:
                info=row['learn_info']
                if name in ('sa_ghmappo','ppo','mappo'):
                    if info['initial_log_prob_error_max']>2e-5:raise ValueError('Pre-update probability error')
                    if info['accepted_optimizer_steps']<=0 or info['accepted_policy_kl_max']>info['target_kl']+1e-7:
                        raise ValueError('No effective update or invalid local KL')
                if name=='sa_ghmappo':
                    diagnostic=info['objective_diagnostics']
                    if len(diagnostic['credit_rows'])!=256 or len(diagnostic['gradient_batches'])!=1:
                        raise ValueError('Incomplete objective/credit diagnostics')
                    batch=diagnostic['gradient_batches'][0]
                    if not math.isclose(batch['weighted_term_sum'],batch['total_loss'],rel_tol=1e-5,abs_tol=1e-5):
                        raise ValueError('Objective decomposition mismatch')
                    for term in batch['terms'].values():
                        if not all(math.isfinite(v) for v in term['gradient_norm'].values()):raise ValueError('Nonfinite diagnostic')
            checkpoints=sorted(cell.glob('*.pt'))
            if len(checkpoints)!=4:raise ValueError('Checkpoint count mismatch')
            for checkpoint in checkpoints:
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                expected_update=0 if checkpoint.stem=='initial' else int(checkpoint.stem.split('_')[1])
                if payload['update_count']!=expected_update:raise ValueError('Checkpoint counter mismatch')
                if any(not torch.isfinite(v).all() for v in payload['network_state_dict'].values()):raise ValueError('Nonfinite checkpoint')
            initial=torch.load(cell/'initial.pt',map_location='cpu',weights_only=False)
            final=torch.load(cell/'update_0003.pt',map_location='cpu',weights_only=False)
            if initial.get('optimizer_state_dict',{}).get('state',{}):raise ValueError('Initial optimizer not empty')
            if initial['update_count']!=0 or final['update_count']!=3:raise ValueError('Checkpoint update identity mismatch')
            changed=sum(not torch.equal(value,final['network_state_dict'][key]) for key,value in initial['network_state_dict'].items())
            if changed==0:raise ValueError('No network tensor updated')
            if any(not torch.isfinite(v).all() for v in final['network_state_dict'].values()):raise ValueError('Nonfinite weights')
            adam_steps=[int(v['step']) for v in final.get('optimizer_state_dict',{}).get('state',{}).values() if 'step' in v]
            if not adam_steps or max(adam_steps)<=0:raise ValueError('No effective Adam state')
            accepted=sum(u['learn_info'].get('accepted_optimizer_steps',0) for u in updates) if name!='qmix' else None
            if accepted is not None and max(adam_steps)!=accepted:raise ValueError('Effective Adam counter mismatch')
            probes=list(records(cell/'marginal_probes.jsonl'))
            if name=='sa_ghmappo':
                if len(probes)!=6:raise ValueError('Fixed marginal probe count mismatch')
                for probe in probes:
                    d=probe['diagnostic'];p=d['executed_action_probs']
                    if d['env_action_probs_scope']!='final_executed_action_marginal_at_probe_state' or not math.isclose(sum(p),1.,abs_tol=2e-5):
                        raise ValueError('Missing exact execution marginal')
            results.append(dict(agent=name,seed=seed,train_steps=phases['train'],validation_steps=phases['validation'],
                learner_opportunities=3,checkpoints=4,network_tensors_changed=changed,native_ledger_max_error=max_error,
                exact_marginal_probes=len(probes),accepted_optimizer_steps=accepted,
                adam_parameter_step_range=[min(adam_steps),max(adam_steps)]))
    return dict(status='PASS',reviewed_at=datetime.now().astimezone().isoformat(),scope='non_formal_preflight_only',
        source_commit=receipt['source_commit'],cells=results,total_train_interactions=6144,total_learner_opportunities=24,
        total_validation_interactions=completion['validation_steps'],total_environment_interactions=completion['total_environment_steps'],
        total_checkpoints=32,formal_training_started=False,performance_or_convergence_claims='UNVERIFIED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args();report=audit(args.run_root)
    write_json(args.run_root/'preflight_audit.json',report);print('PASS cells=8 checkpoints=32 interactions=6144 formal_training=False')


if __name__=='__main__':main()
