"""Capped non-formal preflight only. There is deliberately no full-training mode."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import run_fair_reward_training as fair
from sagh_execution_diagnostics import (enumerate_executed_distribution, install_gate_observer,
                                        isolated_rng, step_record)

SOURCE = Path(__file__).resolve().parents[1]
AGENTS = ("sa_ghmappo", "ppo", "mappo", "qmix")
SEEDS = (7, 13)
CLASSES = ("idle_or_sparse", "mechanism_activating", "active_non_mechanism")
STEPS_PER_CELL = 768
EXPECTED_PARENT_PLAN = '36f019938092eed36bb5bd34e23d88532156db37aae88c2704fa2f689dd4eef4'
RESOURCE_SOURCE = Path('/Users/howen/.codex/worktrees/fair-reward-resource-guard/PPO_MEC')
_owned_output_root = None


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n');stream.flush()


def policy_kl_limit(target):
    # Match the unchanged native _guarded_policy_step, including zero-target baselines.
    return target if target>0 else .01


def reuse_completed_sagh(root,plan):
    previous=json.loads((root/'execution_receipt.json').read_text())
    if previous['parent_plan_sha256']!=plan['plan_sha256'] or previous['hard_total_interaction_cap']!=6144:
        raise ValueError('Unmatched preflight inheritance')
    runtime_paths=[p for p in previous['source_sha256'] if p.startswith('src/') or p in (
        'scripts/run_fair_reward_training.py','scripts/diagnose_sagh_flat_reward.py','scripts/sagh_execution_diagnostics.py')]
    if any(fair.sha(SOURCE/p)!=previous['source_sha256'][p] for p in runtime_paths):
        raise ValueError('Scientific runtime changed; cannot inherit completed cells')
    complete_paths=sorted(root.glob('cells/*/seed*/complete.json'))
    if {str(p.parent.relative_to(root)) for p in complete_paths}!={'cells/sa_ghmappo/seed7','cells/sa_ghmappo/seed13'}:
        raise ValueError('Only the two complete, unselected SAGH cells can be inherited')
    for path in complete_paths:
        complete=json.loads(path.read_text())
        if complete['training_steps']!=768 or complete['learner_opportunities']!=3:
            raise ValueError('Incomplete inherited SAGH cell')
        if any(fair.sha(path.parent/name)!=expected for name,expected in complete['files_sha256'].items()):
            raise ValueError('Inherited file changed')
    partial=root/'cells/ppo/seed7/steps.jsonl'
    counts={'train':0,'validation':0}
    for line in partial.open():counts[json.loads(line)['metadata']['phase']]+=1
    if counts['train']!=256 or counts['validation']>132:raise ValueError('Unexpected partial preflight budget')
    inventory={str(p.relative_to(root)):fair.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    return dict(root=str(root),source_commit=previous['source_commit'],complete_cells=['sa_ghmappo/seed7','sa_ghmappo/seed13'],
        inventory_sha256=inventory,failed_partial_train_steps=counts['train'],failed_partial_validation_steps=counts['validation'],
        partial_policy='preserve_and_exclude; no weight or optimizer loading; fresh PPO retry')


def select_windows(plan, split):
    # Class/offset selection is fixed from data labels, never from outcomes.
    return [next(copy.deepcopy(w) for w in plan['windows']
                 if w['split']==split and w['window_class']==kind) for kind in CLASSES]


def plan_from_parent(parent):
    plan=json.loads((parent/'frozen_plan.json').read_text())
    expected=plan['plan_sha256'];payload=dict(plan);payload.pop('plan_sha256')
    if fair.semantic_hash(payload)!=expected or expected!=EXPECTED_PARENT_PLAN:
        raise ValueError('Parent scientific plan hash mismatch')
    if plan['external_reward_shaping'] or plan['reward_positive_offset']!=0:
        raise ValueError('Native reward only')
    if plan['steps_per_update']!=256 or plan['steps_per_training_unit']!=128 or len(plan['workflow_ids'])!=2 or plan['max_episode_steps']!=22:
        raise ValueError('Unmatched parent budgets/workflows')
    train, validation=select_windows(plan,'train'),select_windows(plan,'validation')
    fair.intervals_disjoint(train+validation)
    return plan,train,validation


def resource_guard():
    expected='71ca85dcac7f709bc27fc1c3e115f771f50a4cb4'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RESOURCE_SOURCE,text=True).strip()!=expected:
        raise ValueError('Pinned resource guard HEAD mismatch')
    path=RESOURCE_SOURCE/'scripts/continue_fair_reward_training.py'
    committed=subprocess.check_output(['git','show',expected+':scripts/continue_fair_reward_training.py'],cwd=RESOURCE_SOURCE)
    import hashlib
    if fair.sha(path)!=hashlib.sha256(committed).hexdigest():
        raise ValueError('Pinned resource guard bytes changed')
    spec=importlib.util.spec_from_file_location('preflight_resource_guard',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def configured_agent(plan,name,seed):
    config=copy.deepcopy(plan['agent_configs'][name][str(seed)])
    if name=='sa_ghmappo':
        config.update(execution_mapping_contract='shared_train_eval_v1',objective_diagnostics_enabled=True)
    return fair.fresh_agent(name,config,seed),config


def run_cell(out,plan,name,seed,train,validation,workflows,guard):
    cell=out/'cells'/name/f'seed{seed}';cell.mkdir(parents=True,exist_ok=False)
    agent,config=configured_agent(plan,name,seed)
    if getattr(agent,'_optimizer',None) is not None and agent._optimizer.state:
        raise ValueError('Fresh optimizer must be empty')
    if len(getattr(agent,'_replay',[])):
        raise ValueError('Fresh replay must be empty')
    agent.save(str(cell/'initial.pt'))
    write_json(cell/'identity.json',dict(agent=name,seed=seed,config=config,fresh=True,optimizer_initially_empty=True,
        execution_mapping_contract=config.get('execution_mapping_contract','legacy_deterministic_fallback_v1')))
    if name=='sa_ghmappo':install_gate_observer(agent)
    training_steps=0;updates=0;episodes=0;validation_rows=0;validation_episodes=0;pending=[]
    with (cell/'steps.jsonl').open('x') as trace,(cell/'episodes.jsonl').open('x') as receipts,\
         (cell/'updates.jsonl').open('x') as learning,(cell/'marginal_probes.jsonl').open('x') as probes:
        def record_episode(bundle,workflow,unit,max_steps,training):
            nonlocal training_steps,episodes,validation_rows,validation_episodes
            meta=dict(phase='train' if training else 'validation',agent=name,seed=seed,unit=unit,
                training_steps=training_steps,window_id=bundle.rsu_metadata['window_id'],workflow_id=workflow.workflow_id,
                window_class=bundle.rsu_metadata['window_class'],scope='non_formal_small_budget_preflight')
            summary,rollout=fair.make_episode(SOURCE,plan,agent,bundle,workflow,seed+unit,max_steps,meta,training)
            total=0.0
            for i,row in enumerate(rollout):
                record=step_record(row,dict(meta,step=i));json_line(trace,record);total+=record['native_reward']
            if not math.isclose(total,summary['reward_breakdown']['total']['sum'],abs_tol=1e-5):
                raise ValueError('Native episode reward mismatch')
            json_line(receipts,dict(metadata=meta,summary=summary,steps=len(rollout)))
            if training:training_steps+=len(rollout);episodes+=1
            else:validation_rows+=len(rollout);validation_episodes+=1
            if name=='sa_ghmappo' and meta['workflow_id']==plan['workflow_ids'][0] and rollout:
                # Exactly one first-state probe per class before any training and at final validation.
                if not training and training_steps in (0,STEPS_PER_CELL):
                    probe=enumerate_executed_distribution(agent,rollout[0]['decision_info'])
                    json_line(probes,dict(metadata=meta,diagnostic=probe))
            return rollout
        def evaluate():
            nonlocal agent
            guard.yield_resources(out)
            with isolated_rng():
                original=agent
                evaluated,_=configured_agent(plan,name,seed)
                evaluated._network.load_state_dict(original._network.state_dict())
                evaluated._update_count=original._update_count
                evaluated._deterministic_action=True;evaluated._network.eval()
                if name=='sa_ghmappo':install_gate_observer(evaluated)
                agent=evaluated
                try:
                    for i,bundle in enumerate(validation):
                        for j,workflow in enumerate(workflows):record_episode(bundle,workflow,10000+2*i+j,plan['max_episode_steps'],False)
                finally:agent=original
        evaluate()
        for j,workflow in enumerate(workflows):
            for i,bundle in enumerate(train):
                guard.yield_resources(out)
                unit=3*j+i;remaining=128
                while remaining:
                    rollout=record_episode(bundle,workflow,unit,min(remaining,plan['max_episode_steps']),True)
                    if not 0<len(rollout)<=remaining:raise ValueError('Unit budget mismatch')
                    remaining-=len(rollout);pending.extend(rollout)
                if len(pending)==256:
                    guard.yield_resources(out)
                    info=agent.learn(pending);pending=[];updates+=1
                    json_line(learning,dict(update=updates,training_steps=training_steps,learn_info=info))
                    if name in ('sa_ghmappo','ppo','mappo'):
                        if info['initial_log_prob_error_max']>2e-5 or info['accepted_policy_kl_max']>policy_kl_limit(config['target_kl'])+1e-7:
                            raise ValueError('Probability or local KL contract failed')
                    if name=='sa_ghmappo' and not info['objective_diagnostics']['gradient_batches']:
                        raise ValueError('Missing objective gradient diagnostic')
                    agent.save(str(cell/f'update_{updates:04d}.pt'))
                    evaluate()
                elif len(pending)>256:raise ValueError('Learner budget exceeded')
        if training_steps!=STEPS_PER_CELL or pending or updates!=3 or validation_episodes!=24:
            raise ValueError('Preflight hard cap/accounting failed')
    write_json(cell/'complete.json',dict(status='PREFLIGHT_COMPLETED',training_steps=training_steps,
        learner_opportunities=updates,training_episodes=episodes,validation_episodes=validation_episodes,
        validation_steps=validation_rows,checkpoints=4,
        files_sha256={p.name:fair.sha(p) for p in cell.iterdir() if p.is_file()}))
    print(f'PREFLIGHT {name} seed={seed} steps={training_steps} updates={updates}',flush=True)


def execute(parent,out,reuse_root=None):
    global _owned_output_root
    if out.exists():raise FileExistsError('Preflight output is create-only; no resume/restart')
    if out==parent or parent in out.parents:raise ValueError('Cannot write within parent artifact')
    if subprocess.check_output(['git','diff','HEAD','--','src','scripts','tests'],cwd=SOURCE):
        raise ValueError('Commit validated code before preflight')
    os.nice(max(0,19-os.nice(0)))
    plan,train_records,validation_records=plan_from_parent(parent)
    inherited=reuse_completed_sagh(reuse_root,plan) if reuse_root is not None else None
    for key,identity in plan['data_files'].items():
        data=Path(plan[key])
        if data.stat().st_size!=identity['size'] or fair.sha(data)!=identity['sha256']:
            raise ValueError('Original data hash mismatch')
    if subprocess.check_output(['git','diff','45c8c88','--','src/envs','src/metrics','configs'],cwd=SOURCE):
        raise ValueError('Reward/environment/configuration bytes changed')
    guard=resource_guard();out.mkdir(parents=True);_owned_output_root=out
    write_json(out/'execution_receipt.json',dict(scope='non_formal_preflight_only_no_formal_training_authorized',
        command=sys.argv,source_root=str(SOURCE),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        parent_plan_sha256=plan['plan_sha256'],agents=AGENTS,seeds=SEEDS,steps_per_cell=STEPS_PER_CELL,
        hard_total_interaction_cap=6144,learner_opportunities_per_cell=3,train_windows=train_records,validation_windows=validation_records,
        hard_validation_interaction_cap=4224,hard_total_environment_interaction_cap=10368,data_files=plan['data_files'],
        read_only_reuse=inherited,fresh_train_interaction_cap=4608 if inherited else 6144,
        failed_attempt_train_overhead=256 if inherited else 0,
        lifetime_train_interaction_cap=6400 if inherited else 6144,
        lifetime_validation_interaction_cap=4224+(inherited['failed_partial_validation_steps'] if inherited else 0),
        config_changes={'sa_ghmappo':{'execution_mapping_contract':'shared_train_eval_v1','objective_diagnostics_enabled':True}},
        reward_changes=False,old_checkpoint_loads=0,original_curve_changes=False,
        runtime=dict(cpu_threads=1,nice=19,concurrent_cells=1,yield_to_foreign_training=True),
        source_sha256={str(p.relative_to(SOURCE)):fair.sha(p) for p in sorted(SOURCE.rglob('*.py')) if p.parts[len(SOURCE.parts)] in ('src','scripts','tests')}))
    fair.load_api(SOURCE)
    train=[];validation=[]
    for records,bundles in [(train_records,train),(validation_records,validation)]:
        for record in records:
            bundle=fair.window_bundle(SOURCE,plan,record['frame_offset'])
            if fair.window_identity(bundle)!= {k:record[k] for k in fair.window_identity(bundle)}:
                raise ValueError('Frozen real window identity mismatch')
            bundle.rsu_metadata['window_class']=record['window_class'];bundles.append(bundle)
    workflows=fair.build_workflows(workflow_csv_path=plan['workflow_path'],max_workflows=plan['max_workflows'],
        workflow_selector='ordered',min_tasks=5,max_tasks=20,random_seed=7)
    if [w.workflow_id for w in workflows]!=plan['workflow_ids']:raise ValueError('Workflow identity mismatch')
    for name in AGENTS:
        for seed in SEEDS:
            if inherited and f'{name}/seed{seed}' in inherited['complete_cells']:
                target=out/'cells'/name/f'seed{seed}';target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(reuse_root/'cells'/name/f'seed{seed}',target_is_directory=True)
                print(f'REUSE_COMPLETED_READ_ONLY {name} seed={seed}',flush=True)
            else:run_cell(out,plan,name,seed,train,validation,workflows,guard)
    completed=[json.loads(p.read_text()) for p in (out/'cells').glob('*/seed*/complete.json')]
    if len(completed)!=8 or sum(x['training_steps'] for x in completed)!=6144:raise ValueError('Global hard cap failed')
    validation_steps=sum(x['validation_steps'] for x in completed)
    if validation_steps>4224:raise ValueError('Validation interaction cap exceeded')
    write_json(out/'completion.json',dict(status='PREFLIGHT_COMPLETED',cells=8,training_steps=6144,
        validation_steps=validation_steps,total_environment_steps=6144+validation_steps,
        inherited_cells=2 if inherited else 0,fresh_cells=6 if inherited else 8,
        failed_attempt_train_overhead=256 if inherited else 0,
        failed_attempt_validation_overhead=inherited['failed_partial_validation_steps'] if inherited else 0,
        learner_opportunities=24,checkpoints=32,formal_training_started=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--execute-preflight',action='store_true',required=True)
    parser.add_argument('--reuse-completed-sagh-from',type=Path)
    args=parser.parse_args()
    try:execute(args.parent_root.resolve(),args.output_root.resolve(),args.reuse_completed_sagh_from.resolve() if args.reuse_completed_sagh_from else None)
    except Exception as error:
        if _owned_output_root==args.output_root.resolve() and not (args.output_root/'failure.json').exists():
            write_json(args.output_root/'failure.json',dict(status='PREFLIGHT_FAILED',error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__=='__main__':main()
