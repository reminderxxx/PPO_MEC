"""Create-only qualification and bounded learning over a fixed FIFO batch."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.envs.core.persistent_workflow_env import PersistentWorkflowEnv, build_batch
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.specs import RSUState,WorkflowGraphState,WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.envs.core.predictor_manager import PredictorManager
from src.evaluators.main_results_support import load_window_bundle,build_selected_workflow_states
from src.agents.sa_ghmappo_agent import DirectWorkflowGraphPPOAgent
from src.agents.registry import build_agent
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def dump(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.to_dict())+'\n')


def workflows(data):
    return [WorkflowGraphState(workflow_id=x['workflow_id'],nodes=[WorkflowNode(**n) for n in x['nodes']],
        edges=[tuple(e) for e in x['edges']],execution_order=x['execution_order'],current_node_id=x['current_node_id']) for x in data]


def make_env(unit,wfs,sink,seed=7,horizon=95):
    core=PersistentWorkflowEnv(workflows=wfs,subject_id=unit['subject_id'],
        mobility_provider=ReplayProvider(deepcopy(unit['frames'])),rsu_states=[RSUState(**r) for r in unit['rsus']],
        adapter_catalog=AdapterCatalog.from_json(ROOT/'src/data/model_catalog/typed_model_cache_controlled.json'),
        predictor_manager=PredictorManager(horizon=32,random_seed=seed),max_steps=horizon,mobility_source='ngsim',
        cache_capacity_profile=dict(model_cache_profile_id='typed_base_adapter_state_v1',enabled=True,unit='mb',
            capacity_mb=288.,count_base_model_separately=True,eviction_policy='lru',
            typed_eviction_semantics='sequential_dependency_recompute_lru_v1'))
    recorder=EpisodeRecorder()
    return GymVecEnv(core,recorder,subject_boundary_receipt_path=str(sink),subject_boundary_exposure_id=str(sink)),recorder


def prepare(root,old_plan):
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    old=json.loads(Path(old_plan).read_text())
    wfs=build_selected_workflow_states(old['workflow_path'],2,'ordered',5,20,7)
    units=[]
    eligibility=[]
    for i in range(24):
        offset=i*192
        b=load_window_bundle(root_dir=ROOT,mobility_source='ngsim',mobility_csv_path=old['mobility_path'],
            max_mobility_rows=30000,rsu_layout='auto_dominant_tight',frame_offset=offset,window_length=96,random_seed=7)
        frames=[]
        for frame in b.frames:
            f=deepcopy(frame);f['vehicles']=[v.to_dict() if hasattr(v,'to_dict') else v for v in f['vehicles']];frames.append(f)
        common=set.intersection(*[{v['vehicle_id'] for v in f['vehicles']} for f in frames])
        eligibility.append(dict(candidate_index=i,offset=offset,frame_count=len(frames),persistent_candidates=len(common),eligible=len(frames)==96 and bool(common)))
        dump(root/'eligibility.json',eligibility)
        if len(frames)!=96 or not common:continue
        unit=dict(index=len(units),split='train' if len(units)<8 else 'validation',offset=offset,subject_id=sorted(common)[0],
            persistent_candidate_count=len(common),frames=frames,rsus=[r.to_dict() for r in b.rsu_states],metadata=b.rsu_metadata)
        units.append(unit)
        if len(units)==12:break
    if len(units)!=12:raise ValueError('Fewer than twelve eligible windows within frozen 24-candidate scan cap')
    payload=dict(contract='persistent_fifo_batch_v1',units=units,workflows=[x.to_dict() for x in wfs],
        source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        assignment='sha256(source workflow/node) modulo five compatible adapters; simulated AI labels',
        methods=['sa_direct_graph_ppo','ppo','mappo','qmix'],seeds=[7,13,29],interactions_per_cell=4096,
        technical_step_cap=12288,formal=False,holdout_access=False)
    dump(root/'frozen_inputs.json',payload)
    dump(root/'status.json',dict(status='PREPARED',technical_steps=0,learning_steps=0))
    print(json.dumps(dict(status='PREPARED',windows=len(units),nodes=len(build_batch(wfs)[0].nodes))))


def qualify(root):
    if (root/'qualification.json').exists():raise FileExistsError('Qualification is create-only')
    p=json.loads((root/'frozen_inputs.json').read_text());wfs=workflows(p['workflows']);rows=[];steps=0
    out=root/'qualification';out.mkdir()
    for unit in p['units']:
        for rule in ['reactive','prepare','prefetch','cycle','wait','local']:
            sink=out/f"boundary_{unit['index']}_{rule}.json"
            env,_=make_env(unit,wfs,sink)
            _,info=env.reset();actions=[]
            for t in range(95):
                mask=info['action_mask']
                proposal=dict(reactive=0,prepare=4,prefetch=1,cycle=[0,4,1,3][t%4],wait=3,local=2)[rule]
                action=proposal if mask[proposal] else next(i for i,x in enumerate(mask) if x)
                _,r,done,truncated,info=env.step(action);actions.append(action);steps+=1
                if steps>p['technical_step_cap']:raise RuntimeError('Technical budget exceeded')
                if done or truncated:break
            records=env.core_env.stream_rows
            later=[v for row in records for v in row['later_load_use']]
            rows.append(dict(window=unit['index'],split=unit['split'],rule=rule,steps=len(actions),
                reward=sum(x['reward'] for x in records),completed_requests=records[-1]['completed_requests'],
                pending_area_seconds=sum(x['pending_area_seconds'] for x in records),
                evictions=sum(int(x['cache_event'].get('eviction_count',0)) for x in records),
                delayed_load_uses=len(later),delayed_prepare_uses=sum('prepare' in str(x['strategy']) or 'prefetch' in str(x['strategy']) for x in later),
                transfer_mb=sum(x['transfer_mb'] for x in records),actions=actions))
            dump(out/f"trace_{unit['index']}_{rule}.json",records)
            dump(root/'qualification_progress.json',dict(technical_steps=steps,completed_exposures=len(rows)))
    differences=[u for u in range(12) if len({round(x['reward'],6) for x in rows if x['window']==u})>1]
    checks=dict(all_exposures_present=len(rows)==72,within_budget=steps<=p['technical_step_cap'],
        finite_cache_binds=any(x['evictions']>0 for x in rows),delayed_loading_observed=any(x['delayed_load_uses']>0 for x in rows),
        delayed_preparation_used=any(x['delayed_prepare_uses']>0 for x in rows),decision_outcomes_differ=bool(differences))
    dump(root/'qualification.json',dict(checks=checks,passed=all(checks.values()),technical_steps=steps,rows=rows,
        note='Qualification is technical opportunity evidence, not learned superiority. All fixed windows and scripted policies retained.'))
    print(json.dumps(dict(status='QUALIFIED' if all(checks.values()) else 'QUALIFICATION_FAILED',checks=checks,technical_steps=steps)))


def agent_for(method,seed,deterministic=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    common=dict(random_seed=seed,deterministic_action=deterministic,device='cpu')
    if method=='sa_direct_graph_ppo':return DirectWorkflowGraphPPOAgent(**common)
    return build_agent(method,**common)


def episode(root,p,unit,method,seed,agent,label,max_steps):
    sink=root/'boundaries'/f'{method}_{seed}_{label}.json'
    env,recorder=make_env(unit,workflows(p['workflows']),sink,seed)
    trainer=MARLOnPolicyTrainer(env=env,agent=agent,recorder=recorder,max_steps=max_steps,gamma=.99,gae_lambda=.95)
    summary,rollout=trainer.collect_episode(run_metadata=dict(phase=label,policy_evaluation_mode='raw_policy'),collect_model_targets=False)
    records=env.core_env.stream_rows
    if abs(sum(x['reward'] for x in records)-sum(x['reward'] for x in rollout))>1e-6:raise ValueError('Reward mismatch')
    result=dict(window=unit['index'],reward=sum(x['reward'] for x in records),steps=len(records),
        completed_requests=records[-1]['completed_requests'],pending_area_seconds=sum(x['pending_area_seconds'] for x in records),
        transfer_mb=sum(x['transfer_mb'] for x in records),unfinished_requests=records[-1]['pending_requests'],
        episode_terminal=bool(records[-1]['terminal_reason']))
    return result,rollout


def yield_to_foreign_training(root):
    from scripts.run_fair_reward_training import foreign_training
    while True:
        try:
            output=subprocess.check_output(['ps','-axo','pid=,comm=,args=']).decode('utf-8','backslashreplace')
            other=foreign_training(output,os.getpid())
        except (OSError,subprocess.SubprocessError,ValueError) as error:
            other=[dict(resource_check_unavailable=str(error))]
        if not other:return
        dump(root/'resource_wait.json',dict(status='YIELDING',observed_at=time.time(),processes=other))
        time.sleep(5)


def training_chunk_budget(remaining,buffered,batch_size=256):
    if remaining<1 or not 0<=buffered<batch_size:raise ValueError('Invalid on-policy collection boundary')
    return min(95,remaining,batch_size-buffered)


def train(root):
    p=json.loads((root/'frozen_inputs.json').read_text());q=json.loads((root/'qualification.json').read_text())
    if not q['passed']:raise ValueError('Qualification failed; no learning authorized by phase gate')
    out=root/'pilot';out.mkdir();(root/'boundaries').mkdir()
    source_files=[ROOT/'src/envs/core/persistent_workflow_env.py',ROOT/'src/agents/sa_ghmappo_agent.py',Path(__file__)]
    dump(out/'source_identity.json',dict(git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files={str(x.relative_to(ROOT)):sha(x) for x in source_files},input_sha256=sha(root/'frozen_inputs.json')))
    rows=[];totals=0
    for method in p['methods']:
        for seed in p['seeds']:
            yield_to_foreign_training(root)
            agent=agent_for(method,seed);cell=out/f'{method}_{seed}';cell.mkdir();collected=0;updates=0;train_rows=[]
            def evaluate():
                rng=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
                try:
                    ev=agent_for(method,seed,True);ev._network.load_state_dict(agent._network.state_dict())
                    ev._update_count=agent._update_count;ev._network.eval()
                    for unit in p['units'][8:]:
                        result,_=episode(root,p,unit,method,seed,ev,f'val_{collected}_{unit["index"]}',95)
                        rows.append(dict(method=method,seed=seed,training_steps=collected,**result))
                finally:random.setstate(rng[0]);np.random.set_state(rng[1]);torch.random.set_rng_state(rng[2])
            evaluate();buffer=[];next_eval=1024;episode_index=0
            while collected<p['interactions_per_cell']:
                unit=p['units'][episode_index%8]
                budget=training_chunk_budget(p['interactions_per_cell']-collected,len(buffer))
                result,rollout=episode(root,p,unit,method,seed,agent,f'train_{episode_index}',budget)
                if not rollout:raise RuntimeError('Empty episode')
                collected+=len(rollout);totals+=len(rollout);buffer.extend(rollout)
                train_rows.append(dict(episode=episode_index,training_steps=collected,**result))
                dump(cell/'train_episodes.json',train_rows)
                dump(cell/f'train_trace_{episode_index}.json',[dict(action=x['action'],reward=x['reward'],log_prob=x['log_prob'],terminated=x['terminated'],truncated=x['truncated'],persistent_stream=x['env_info']['persistent_stream']) for x in rollout])
                episode_index+=1
                while len(buffer)>=256:
                    yield_to_foreign_training(root)
                    stats=agent.learn(buffer[:256]);buffer=buffer[256:];updates+=1
                    dump(cell/f'update_{updates}.json',stats)
                if collected>=next_eval or collected==p['interactions_per_cell']:
                    evaluate();next_eval+=1024
                    dump(out/'validation.json',rows)
                    agent.save(str(cell/f'checkpoint_{collected}.pt'))
                dump(root/'status.json',dict(status='PILOT_RUNNING',method=method,seed=seed,cell_steps=collected,total_learning_steps=totals))
            if buffer or updates!=p['interactions_per_cell']//256:raise ValueError('Pilot update budget mismatch')
            agent.save(str(cell/'final.pt'))
            dump(cell/'train_episodes.json',train_rows)
            dump(cell/'completion.json',dict(method=method,seed=seed,interactions=collected,learn_calls=updates,
                agent_config=agent._checkpoint_config()))
            print(json.dumps(dict(method=method,seed=seed,interactions=collected,total=totals)),flush=True)
    dump(out/'validation.json',rows)
    dump(root/'status.json',dict(status='PILOT_COMPLETE',learning_steps=totals,cells=12))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','qualify','train']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--old-plan',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);os.nice(19)
    if args.phase=='prepare':prepare(args.root,args.old_plan)
    elif args.phase=='qualify':qualify(args.root)
    else:train(args.root)

if __name__=='__main__':main()
