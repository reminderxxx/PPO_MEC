"""Fixed-budget main-only warm-start intervention and retrained module ablations.

Every variant gets exactly 256 actual frames per input-selected train condition,
with identical parents and three update calls. At fixed-budget cuts, the common
central critic bootstraps the unobserved tail. This explicitly differs from the
old complete-episode Monte Carlo trainer. Native physics/reward are unchanged.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash,select_train_cases,run_episode
from scripts.train_joint_stream_control import make_env,actor_hash
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import RelocationAwareStreamSAGHMAPPOAgent as Main


VARIANTS={
    'original':dict(relocation_cost_mode='episode'),
    'relocation':dict(relocation_cost_mode='node'),
    'no_dag_edges':dict(relocation_cost_mode='node',dag_topology_enabled=False),
    'no_prediction':dict(relocation_cost_mode='node',prediction_enabled=False),
    'no_surrogate':dict(relocation_cost_mode='node',surrogate_enabled=False),
    'no_optional_cache':dict(relocation_cost_mode='node',optional_cache_enabled=False),
}


def train_unit(case,data,manifest,agent,budget,trace,identity):
    """Equal *actual* frame exposure per case; reset only on physical termination.

    All budget fragments and terminal/tail boundaries are logged. No lookahead
    transition is executed for bootstrapping and no native cost is discarded.
    """
    consumed=0;rollouts=[];fragments=[];total=0.;decisions_count=0
    while consumed<budget:
        env=make_env(case,data,manifest);agent.reset_controller()
        rewards=[];local=[];start=consumed
        while consumed<budget and not env._control_episode_done():
            frame=env.frame_index
            controls,roll=agent.act_controls(env.decision_states(),frame)
            _,reward,_,info=env.step_controls(controls)
            assert abs(reward-sum(info['frame_cost_receipt']['reward_components'].values()))<1e-8
            total+=reward;rewards.append(reward);consumed+=1
            if roll is not None:
                by={e['vehicle_id']:e for e in info['events']}
                roll.update(frame_index=frame,episode_end=False,initial_request_count=len(env.workflows))
                roll['cost_targets']=torch.tensor([[sum(np.ceil(env.model_cache.objects[k].transfer_size_bytes/env.model_cache.transfer_bytes_per_frame)
                    for k in by[v]['cache_request'].get('admitted_object_ids',[]))/10,
                    by[v]['state_transfer_link_frames']/10] for v in roll['vehicle_ids']],dtype=torch.float32)
                local.append(roll);decisions_count+=len(roll['actions'])
            trace.write(json.dumps(dict(**identity,unit_frame=consumed-1,fragment_start=start,
                frame_index=frame,reward=reward,receipt=info['frame_cost_receipt'],events=info['events']))+'\n')
        terminal=env._control_episode_done();bootstrap=0.
        if not terminal:
            rng=torch.get_rng_state()
            with torch.no_grad():
                _,next_roll=agent.act_controls(env.decision_states(),env.frame_index,deterministic=True)
                if next_roll is not None:bootstrap=float(next_roll['values'].mean())
            torch.set_rng_state(rng)
        suffix=bootstrap;targets={}
        for frame,value in reversed(list(enumerate(rewards))):
            suffix=value/10+suffix;targets[frame]=suffix
        for roll in local:
            roll['returns']=torch.full_like(roll['values'],targets[roll['frame_index']])
            roll['team_credit_targets']=roll['returns']/len(env.workflows)
            roll['credit_targets']=roll['team_credit_targets'].clone()
        if local:local[-1]['episode_end']=terminal
        rollouts.extend(local)
        fragments.append(dict(start=start,frames=len(rewards),physical_terminal=terminal,
            bootstrap_team_tail=bootstrap,observed_native_return=sum(rewards),summary=env.summary()))
    assert consumed==budget
    return dict(actual_frames=consumed,decision_rows=decisions_count,reward=total,fragments=fragments),rollouts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--checkpoint_root',type=Path,required=True)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not a.preflight:
        raise ValueError('clean frozen source required for the development study')
    source=json.loads(a.protocol.read_text())
    assert source['source_commit'].startswith('35975cb3')
    assert file_hash(a.protocol.parent/'protocol.pt')==source['assets']['protocol.pt']
    for name,expected in source['source_fingerprints'].items():
        if name.startswith('src/') and name!='src/agents/sa_ghmappo_agent.py':
            assert file_hash(ROOT/name)==expected,name
    diagnosis=json.loads(a.diagnosis.read_text())
    assert diagnosis['total_pairs']==36 and diagnosis['verified_frames']>0
    data=torch.load(a.protocol.parent/'protocol.pt',weights_only=False)
    cases=select_train_cases(data)
    eval_ids=['train_000','train_038','train_067']
    evaluations=[next(c for c in cases if c['case_id']==cid) for cid in eval_ids]
    variants=VARIANTS if not a.preflight else {k:VARIANTS[k] for k in ['original','relocation']}
    seeds=[7,17,27] if not a.preflight else [7]
    budget=256 if not a.preflight else 8
    if a.preflight:cases=cases[:1];evaluations=evaluations[:1]
    a.output.mkdir(parents=True,exist_ok=False)
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_commit=git,git_dirty=dirty,
        parent_protocol=str(a.protocol.resolve()),parent_protocol_sha256=file_hash(a.protocol),
        diagnosis_path=str(a.diagnosis.resolve()),diagnosis_sha256=file_hash(a.diagnosis),
        phase='main_only_fixed_frame_warm_start',variants=variants,seeds=seeds,
        training_cases=[c['case_id'] for c in cases],frames_per_training_case=budget,
        actual_training_frames_per_cell=budget*len(cases),update_group_size=4,
        update_calls_per_cell=(len(cases)+3)//4,epochs=4 if not a.preflight else 1,
        evaluation_cases=eval_ids if not a.preflight else [evaluations[0]['case_id']],
        evaluation_mode='greedy full native episode, before and fixed final, no checkpoint selection',
        training_return='observed native team reward /10 + central-critic mean tail bootstrap at budget cuts; team share /8',
        exposure='training partition only; repeated terminal fragments charged, cuts and bootstrap explicit',
        optimizer='fresh Adam for ALL variants; identical parent tensors per seed; no exact-resume claim',
        baseline_source_sha256=file_hash(ROOT/'src/envs/core/joint_stream_rules.py'),
        environment_source_sha256=file_hash(ROOT/'src/envs/core/multi_request_stream_env.py'),
        actor_source_sha256=file_hash(ROOT/'src/agents/ppo_agent.py'),
        main_source_sha256=file_hash(ROOT/'src/agents/sa_ghmappo_agent.py'),
        parameter_count=73940,preflight=a.preflight,command=sys.argv,
        formal_holdout='not read',paper_claim_ready=False,parents={})
    for seed in seeds:
        f=a.checkpoint_root/f'sa_ghmappo_seed{seed}_ep64.pt'
        plan['parents'][str(seed)]=dict(path=str(f.resolve()),sha256=file_hash(f))
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (a.output/'cells.jsonl').open('x') as cells:
        for seed in seeds:
            for name,options in variants.items():
                home=a.output/f'{name}_seed{seed}';home.mkdir()
                random.seed(seed);np.random.seed(seed)
                agent=Main.from_parent_checkpoint(plan['parents'][str(seed)]['path'],training_budget=len(cases),**options)
                assert sum(p.numel() for p in agent.parameters())==73940
                initial=tensor_hash(agent);initial_actor=actor_hash(agent)
                assert not agent.optimizer.state and not agent.surrogate_optimizer.state
                metadata=dict(seed=seed,variant=name,initial_tensor_sha256=initial,
                    initial_actor_sha256=initial_actor,parameter_count=73940,actual_frames=0,
                    decision_rows=0,updates=[],checkpoints=[],status='running')
                agent.save_checkpoint(home/'initial.pt')
                metadata['checkpoints'].append(dict(phase='initial',path='initial.pt',sha256=file_hash(home/'initial.pt')))
                with (home/'evaluation.jsonl').open('x') as ev:
                    for c in evaluations:
                        row=run_episode(make_env(c,data,source),agent,home/f'initial_{c["case_id"]}.jsonl',learned=True)
                        ev.write(json.dumps(dict(phase='initial',case_id=c['case_id'],**row))+'\n');ev.flush()
                    # Fixed common sampling stream; evaluation cannot advance training RNG.
                    torch.manual_seed(seed+10000)
                    buffer=[]
                    with (home/'training.jsonl').open('x') as log,(home/'training_trace.jsonl').open('x') as trace:
                        for i,c in enumerate(cases):
                            started=time.monotonic()
                            row,rolls=train_unit(c,data,source,agent,budget,trace,dict(seed=seed,variant=name,case_id=c['case_id']))
                            buffer.extend(rolls);metadata['actual_frames']+=row['actual_frames'];metadata['decision_rows']+=row['decision_rows']
                            update=None
                            if (i+1)%4==0 or i==len(cases)-1:
                                update=agent.update(buffer,epochs=plan['epochs']);buffer=[]
                                metadata['updates'].append(update)
                            log.write(json.dumps(dict(case_id=c['case_id'],**row,update=update))+'\n');log.flush();trace.flush()
                            (home/'progress.json').write_text(json.dumps(dict(case_index=i+1,**metadata),indent=2)+'\n')
                            print(json.dumps(dict(seed=seed,variant=name,case=c['case_id'],phase='train',frames=metadata['actual_frames'],seconds=round(time.monotonic()-started,2))),flush=True)
                    assert metadata['actual_frames']==plan['actual_training_frames_per_cell']
                    assert len(metadata['updates'])==plan['update_calls_per_cell']
                    agent.save_checkpoint(home/'final.pt')
                    metadata['checkpoints'].append(dict(phase='final',path='final.pt',sha256=file_hash(home/'final.pt')))
                    evaluated=Main.from_checkpoint(home/'final.pt')
                    assert tensor_hash(agent)==tensor_hash(evaluated)
                    for c in evaluations:
                        row=run_episode(make_env(c,data,source),evaluated,home/f'final_{c["case_id"]}.jsonl',learned=True)
                        ev.write(json.dumps(dict(phase='final',case_id=c['case_id'],**row))+'\n');ev.flush()
                metadata.update(status='complete',final_tensor_sha256=tensor_hash(agent),final_actor_sha256=actor_hash(agent),
                    output_sha256={f:file_hash(home/f) for f in ['training.jsonl','training_trace.jsonl','evaluation.jsonl']})
                (home/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
                cells.write(json.dumps(dict(path=home.name,**metadata))+'\n');cells.flush()
                print(json.dumps(dict(seed=seed,variant=name,phase='cell_complete')),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',cells=len(variants)*len(seeds),
        cells_sha256=file_hash(a.output/'cells.jsonl')),indent=2)+'\n')


if __name__=='__main__':main()
