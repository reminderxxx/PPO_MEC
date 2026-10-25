"""Bounded, opt-in recovery: causal preparation, matching features, actual learning."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_persistent_workflow_learning import workflows, dump, sha, training_chunk_budget, yield_to_foreign_training
from src.envs.wrappers.persistent_workflow_control import ReleasedPreparationEnv, PreparationGymEnv, candidates, CONTROL_CONTRACT
from src.encoders.persistent_workflow_encoder import PersistentWorkflowEncoder, OBSERVATION_CONTRACT
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.predictor_manager import PredictorManager
from src.envs.specs import RSUState
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
from src.agents.ppo_agent import PPOAgent
from src.agents.mappo_agent import MAPPOAgent
from src.agents.qmix_agent import QMIXAgent

METHODS=['graph_ppo','set_ppo','mappo','qmix']

def make_env(unit,wfs,sink,seed=7,count=8,capacity=384.,horizon=95):
    core=ReleasedPreparationEnv(workflows=wfs,subject_id=unit['subject_id'],request_count=count,
        mobility_provider=ReplayProvider(deepcopy(unit['frames'])),rsu_states=[RSUState(**r) for r in unit['rsus']],
        adapter_catalog=AdapterCatalog.from_json(ROOT/'src/data/model_catalog/typed_model_cache_controlled.json'),
        predictor_manager=PredictorManager(horizon=32,random_seed=seed),max_steps=horizon,mobility_source='ngsim',
        cache_capacity_profile=dict(model_cache_profile_id='typed_base_adapter_state_v1',enabled=True,unit='mb',capacity_mb=capacity,
            count_base_model_separately=True,eviction_policy='lru',typed_eviction_semantics='sequential_dependency_recompute_lru_v1'))
    recorder=EpisodeRecorder()
    return PreparationGymEnv(core,recorder,subject_boundary_receipt_path=str(sink),subject_boundary_exposure_id=str(sink)),recorder


def agent_for(method,seed,deterministic=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    common=dict(random_seed=seed,deterministic_action=deterministic,device='cpu',train_epochs=4,batch_size=64)
    if method in ['graph_ppo','set_ppo']:a=PPOAgent(**common)
    elif method=='mappo':a=MAPPOAgent(**common)
    elif method=='qmix':a=QMIXAgent(gamma=1.,**common)
    else:raise ValueError(method)
    a._network.encoder=PersistentWorkflowEncoder(hidden_dim=a._hidden_dim,use_graph=method=='graph_ppo').to(a._device)
    a._optimizer=torch.optim.Adam(a._network.parameters(),lr=a._learning_rate)
    if hasattr(a,'_target_network'):
        a._target_network=deepcopy(a._network);a._target_network.eval()
    a._recovery_method=method
    return a


def save_agent(a,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    a.save(str(path))
    checkpoint=torch.load(path,map_location='cpu')
    checkpoint['recovery_contract']=dict(action=CONTROL_CONTRACT,observation=OBSERVATION_CONTRACT,method=a._recovery_method,
        graph=a._recovery_method=='graph_ppo',gamma=1.,native_family=a.agent_name)
    checkpoint['config']['experimental_encoder']=OBSERVATION_CONTRACT
    torch.save(checkpoint,path)


def load_agent(path,method,seed,deterministic=True):
    checkpoint=torch.load(path,map_location='cpu')
    expected=dict(action=CONTROL_CONTRACT,observation=OBSERVATION_CONTRACT,method=method,graph=method=='graph_ppo',gamma=1.,native_family={'graph_ppo':'ppo','set_ppo':'ppo','mappo':'mappo','qmix':'qmix'}[method])
    if checkpoint.get('recovery_contract')!=expected:raise ValueError('Recovery checkpoint contract mismatch')
    a=agent_for(method,seed,deterministic);a.load(str(path));a._network.eval()
    return a


def expert(info,lookahead=True):
    s=info['semantic_state'];mask=info['action_mask'];opts=candidates(s)
    current=opts[0];rsu=next((r for r in s['rsus'] if r['rsu_id']==current['rsu_id']),{})
    resident=current['adapter_id'] in rsu.get('cached_adapter_ids',[])
    if mask[0] and not resident:return 0
    if lookahead and mask[1]:return 1
    return 3


def write_trace(path,rows):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False,allow_nan=False,default=str)


def result_from(env):
    rows=env.core_env.stream_rows
    return dict(steps=len(rows),reward=sum(x['reward'] for x in rows),completed_requests=rows[-1]['completed_requests'],
        waiting_seconds=sum(x['pending_area_seconds'] for x in rows),transfer_mb=sum(x['transfer_mb'] for x in rows),
        request_count=rows[-1]['request_count'],completed_nodes=len(env.core_env.workflow_state.completed_node_ids),terminal=bool(rows[-1]['terminal_reason']))


def qualification(out,p):
    out.mkdir();rows=[];steps=0
    for u in p['units']:
        for capacity in [288.,384.]:
            for count in [2,8]:
                for rule in ['reactive','lookahead']:
                    label=f"w{u['index']}_c{int(capacity)}_n{count}_{rule}"
                    env,_=make_env(u,workflows(p['workflows']),out/f'boundary_{label}.json',count=count,capacity=capacity)
                    _,info=env.reset(seed=7);traces=[]
                    for _ in range(95):
                        action=expert(info,rule=='lookahead')
                        before=info['semantic_state']
                        _,_,done,truncated,info=env.step(action);steps+=1
                        if steps>10000:raise RuntimeError('New-contract qualification cap exceeded')
                        traces.append(dict(action=action,state=before,stream=info['persistent_stream']))
                        if done or truncated:break
                    row=dict(window=u['index'],split=u['split'],capacity_mb=capacity,rule=rule,**result_from(env));rows.append(row)
                    write_trace(out/f'{label}.json.gz',traces)
        dump(out/'progress.json',dict(windows=u['index']+1,steps=steps))
        print(json.dumps(dict(qualification_window=u['index'],steps=steps)),flush=True)
    pairs=[]
    for u in p['units']:
        for capacity in [288.,384.]:
            r={x['rule']:x for x in rows if x['window']==u['index'] and x['capacity_mb']==capacity and x['request_count']==8}
            pairs.append(dict(window=u['index'],capacity_mb=capacity,delta_completed=r['lookahead']['completed_requests']-r['reactive']['completed_requests'],delta_reward=r['lookahead']['reward']-r['reactive']['reward']))
    dump(out/'results.json',dict(rows=rows,pairs=pairs,environment_steps=steps,formal=False))


def collect(out,p,unit,count,seed,agent,label,budget=95):
    env,recorder=make_env(unit,workflows(p['workflows']),out/f'boundary_{label}.json',seed,count)
    _,rollout=MARLOnPolicyTrainer(env,agent,recorder,max_steps=budget,gamma=1.,gae_lambda=.95).collect_episode(
        run_metadata=dict(policy_evaluation_mode='raw_policy',phase=label),collect_model_targets=False)
    result=result_from(env)
    if abs(result['reward']-sum(x['reward'] for x in rollout))>1e-8:raise ValueError('Reward-accounting mismatch')
    trace=[dict(action=x['action'],reward=x['reward'],log_prob=x['log_prob'],value=x['value'],terminated=x['terminated'],truncated=x['truncated'],
        state=x['decision_info']['semantic_state'],mask=x['decision_info']['action_mask'],stream=x['env_info']['persistent_stream']) for x in rollout]
    write_trace(out/f'{label}.json.gz',trace)
    return result,rollout


def evaluate(cell,p,a,method,seed,steps,rows,loads=(2,8)):
    rng=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
    try:
        ev=load_agent(cell/f'checkpoint_{steps}.pt',method,seed)
        for u in p['units'][8:]:
            for count in loads:
                result,_=collect(cell,p,u,count,seed,ev,f'validation_{steps}_w{u["index"]}_n{count}')
                rows.append(dict(method=method,seed=seed,training_steps=steps,window=u['index'],**result))
    finally:random.setstate(rng[0]);np.random.set_state(rng[1]);torch.random.set_rng_state(rng[2])


def curriculum_count(steps,episode):
    if steps<2048:return 1
    if steps<4096:return 2
    return [2,4,8][episode%3]


def train(out,p,methods=METHODS,seeds=(7,13,29),budget=8192,diagnostic=False):
    out.mkdir();validation=[];total=0
    for method in methods:
        for seed in seeds:
            yield_to_foreign_training(out)
            cell=out/f'{method}_{seed}';cell.mkdir();a=agent_for(method,seed);steps=0;episode=0;buffer=[];updates=[]
            save_agent(a,cell/'checkpoint_0.pt')
            evaluate(cell,p,a,method,seed,0,validation)
            while steps<budget:
                count=1 if diagnostic else curriculum_count(steps,episode)
                unit=p['units'][0 if diagnostic else episode%8]
                chunk=training_chunk_budget(budget-steps,len(buffer),256)
                result,rollout=collect(cell,p,unit,count,seed,a,f'train_{episode}',chunk)
                steps+=len(rollout);total+=len(rollout);buffer.extend(rollout);episode+=1
                if not rollout:raise RuntimeError('Empty rollout')
                if len(buffer)==256:
                    yield_to_foreign_training(out)
                    start=time.monotonic();stats=a.learn(buffer);buffer=[]
                    updates.append(dict(training_steps=steps,seconds=time.monotonic()-start,stats=stats))
                    dump(cell/'updates.json',updates)
                elif len(buffer)>256:raise RuntimeError('Mixed-policy buffer')
                if steps%2048==0:
                    save_agent(a,cell/f'checkpoint_{steps}.pt')
                    evaluate(cell,p,a,method,seed,steps,validation)
                    dump(out/'validation.json',validation)
                    print(json.dumps(dict(method=method,seed=seed,training_steps=steps,total=total)),flush=True)
                dump(out/'status.json',dict(method=method,seed=seed,training_steps=steps,total=total))
            if buffer:raise RuntimeError('Unconsumed on-policy samples')
            if not diagnostic:evaluate(cell,p,a,method,seed,steps,validation,loads=(4,))
            dump(cell/'completion.json',dict(method=method,seed=seed,training_steps=steps,learn_calls=len(updates),curriculum=not diagnostic,
                observation_contract=OBSERVATION_CONTRACT,action_contract=CONTROL_CONTRACT,teacher_steps=0))
    dump(out/'validation.json',validation)
    dump(out/'status.json',dict(status='COMPLETE',training_steps=total,cells=len(methods)*len(seeds)))


def source_identity(out,input_path):
    subprocess.run(['git','diff','--quiet'],cwd=ROOT,check=True)
    files=[ROOT/'src/envs/wrappers/persistent_workflow_control.py',ROOT/'src/encoders/persistent_workflow_encoder.py',Path(__file__)]
    dump(out/'source_identity.json',dict(git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=sha(input_path),files={str(x.relative_to(ROOT)):sha(x) for x in files}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['qualify','diagnostic','train'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);os.nice(19)
    p=json.loads(args.inputs.read_text());args.root.mkdir()
    source_identity(args.root,args.inputs)
    dump(args.root/'command.json',dict(argv=sys.argv,formal=False,holdout=False,methods=METHODS,gamma=1.,capacity_mb=384.,training_budget=8192,validation_loads=[2,8]))
    if args.phase=='qualify':qualification(args.root/'qualification',p)
    elif args.phase=='diagnostic':train(args.root/'learning',p,methods=['graph_ppo','set_ppo'],seeds=[7],budget=4096,diagnostic=True)
    else:train(args.root/'learning',p)

if __name__=='__main__':main()
