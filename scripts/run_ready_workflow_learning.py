"""Gated development experiments: dependency-safe scheduling before more RL."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_workflow_recovery import workflows, result_from, write_trace
from scripts.run_persistent_workflow_learning import dump, sha
from src.envs.wrappers.ready_workflow_control import ReadyWorkflowEnv, ready_node_ids
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.predictor_manager import PredictorManager
from src.envs.specs import RSUState


def make_env(unit, wfs, count=8, seed=7, horizon=95, capacity=384.):
    return ReadyWorkflowEnv(workflows=wfs, subject_id=unit['subject_id'], request_count=count,
        mobility_provider=ReplayProvider(deepcopy(unit['frames'])), rsu_states=[RSUState(**r) for r in unit['rsus']],
        adapter_catalog=AdapterCatalog.from_json(ROOT/'src/data/model_catalog/typed_model_cache_controlled.json'),
        predictor_manager=PredictorManager(horizon=32, random_seed=seed), max_steps=horizon, mobility_source='ngsim',
        cache_capacity_profile=dict(model_cache_profile_id='typed_base_adapter_state_v1', enabled=True, unit='mb',
            capacity_mb=capacity, count_base_model_separately=True, eviction_policy='lru',
            typed_eviction_semantics='sequential_dependency_recompute_lru_v1'))


def request_remaining(state):
    done = set(state['workflow']['completed_node_ids'])
    return {r['request_id']: len(set(r['node_ids'])-done) for r in state['request_stream']['requests']}


def node_runtime(state, node):
    vehicle = next(v for v in state['vehicles'] if v['vehicle_id'] == state['primary_vehicle_id'])
    rsu = vehicle.get('associated_rsu_id')
    resident = next((r for r in state['rsus'] if r['rsu_id'] == rsu), {})
    present = node['required_adapter'] in resident.get('cached_adapter_ids', [])
    loading = max((r['remaining_seconds'] for r in state['model_loading']
        if r['rsu_id'] == rsu and r['object_id'] in ['base:veh_base_v1', 'adapter:'+node['required_adapter'].removeprefix('adapter_')]), default=0.)
    return dict(present=present, ready=present and loading <= 1e-9, loading=loading)


def rule_action(state, rule):
    ready = set(ready_node_ids(state)); nodes = state['workflow']['nodes']; order = state['workflow']['execution_order']
    rank = {n:i for i,n in enumerate(order)}; remaining = request_remaining(state)
    options = [n for n in nodes if n['node_id'] in ready]
    def key(n):
        rt = node_runtime(state, n); req = n['node_id'].split(':')[0]
        if rule == 'fifo': return (rank[n['node_id']],)
        if rule == 'cache_first': return (not rt['ready'], not rt['present'], rt['loading'], remaining[req], rank[n['node_id']])
        if rule == 'shortest_request': return (remaining[req], not rt['ready'], not rt['present'], rank[n['node_id']])
        if rule == 'dependency_cache':
            # Public causal topology heuristic; no future mobility or simulator fork.
            by_id = {x['node_id']:x for x in nodes}; done = set(state['workflow']['completed_node_ids'])
            successors = [by_id[i] for i in n['successors'] if i not in done]
            unlock = sum(set(x['predecessors'])-done == {n['node_id']} for x in successors)
            warm_unlock = sum(node_runtime(state,x)['ready'] and set(x['predecessors'])-done == {n['node_id']} for x in successors)
            return (remaining[req], not rt['ready'], not rt['present'], -warm_unlock, -unlock, rank[n['node_id']])
        raise ValueError(rule)
    return min(options, key=key)['node_id']


def run_rule(unit, wfs, rule, count):
    env = make_env(unit, wfs, count); state,_ = env.reset(); trace=[]
    while True:
        node = rule_action(state, rule); before = state
        state,reward,done,truncated,info = env.step_node(node)
        trace.append(dict(state=before, node_id=node, reward=reward.total, stream=info['persistent_stream']))
        if done or truncated:break
    return result_from(type('Envelope', (), {'core_env':env})()),trace


def qualify(out, p):
    out.mkdir(); rows=[]; steps=0
    for u in p['units']:
        for count in [2,8]:
            for rule in ['fifo','cache_first','shortest_request','dependency_cache']:
                result,trace=run_rule(u,workflows(p['workflows']),rule,count)
                steps += len(trace)
                if steps>16000:raise RuntimeError('Qualification budget exceeded')
                rows.append(dict(window=u['index'],split=u['split'],rule=rule,**result))
                write_trace(out/f'w{u["index"]}_n{count}_{rule}.json.gz',trace)
        dump(out/'progress.json',dict(windows=u['index']+1,steps=steps))
        print(json.dumps(dict(window=u['index'],steps=steps)),flush=True)
    pairs=[]
    for u in p['units']:
        rr={x['rule']:x for x in rows if x['window']==u['index'] and x['request_count']==8}
        pairs.append(dict(window=u['index'],split=u['split'],**{rule:dict(reward_gain=rr[rule]['reward']-rr['fifo']['reward'],completion_gain=rr[rule]['completed_requests']-rr['fifo']['completed_requests']) for rule in rr}))
    train=[x for x in pairs if x['split']=='train']
    passed=any(x[r]['completion_gain']>0 or x[r]['reward_gain']>1 for x in train for r in ['cache_first','shortest_request','dependency_cache'])
    dump(out/'results.json',dict(rows=rows,pairs=pairs,environment_steps=steps,opportunity_gate_passed=passed,formal=False,holdout=False,rule='Gate uses training windows only; validation rows retained without selection.'))


def evaluate_policy(out, p, agent, method, seed, steps, units, loads=(8,)):
    import torch
    from src.encoders.ready_workflow_encoder import ready_features
    from src.agents.ppo_agent import ReadyWorkflowPPOAgent
    rng = torch.random.get_rng_state();checkpoint = out/f'checkpoint_{steps}.pt'
    try:
        frozen, payload = ReadyWorkflowPPOAgent.load(checkpoint); frozen.network.eval(); rows=[]
        for unit in units:
            for count in loads:
                env=make_env(unit,workflows(p['workflows']),count,seed);state,_=env.reset();trace=[]
                while True:
                    features=ready_features(state);action,_,_=frozen.act(features,True);node=features['ids'][action];before=state
                    state,reward,done,truncated,info=env.step_node(node)
                    trace.append(dict(state=before,node_id=node,reward=reward.total,stream=info['persistent_stream']))
                    if done or truncated:break
                label=f'validation_{steps}_w{unit["index"]}_n{count}.json.gz';write_trace(out/label,trace)
                result=result_from(type('Envelope',(),{'core_env':env})())
                rows.append(dict(method=method,seed=seed,training_steps=steps,window=unit['index'],split=unit['split'],
                    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trace=label,trace_sha256=sha(out/label),**result))
        return rows
    finally:torch.random.set_rng_state(rng)


def train_cell(out, p, method, seed, budget=2048, pilot=False):
    import torch
    from src.encoders.ready_workflow_encoder import ready_features
    from src.agents.ppo_agent import ReadyWorkflowPPOAgent
    torch.set_num_threads(1);out.mkdir();agent=ReadyWorkflowPPOAgent(seed,method=='graph_ppo')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=dict(source_git=source,method=method,seed=seed,budget=budget,expert_initialization=False,
        gamma=1.,gae_lambda=.95,training_reward_divisor=20.,evaluation_reward='original_business_reward',
        source_input_sha256=p['_input_sha256'],training_units=list(range(8)),pilot=pilot)
    dump(out/'command.json',dict(argv=sys.argv,cwd=str(ROOT),provenance=provenance))
    validation=[];history=[];updates=[];steps=0;episode=0;pending=[]
    agent.save(out/'checkpoint_0.pt',provenance)
    validation+=evaluate_policy(out,p,agent,method,seed,0,p['units'][:4] if pilot else p['units'][8:],(2,8) if pilot else (8,))
    dump(out/'validation.json',validation)
    points=[1024,2048] if pilot else [2048,4096,budget]
    for target in sorted(set(points)):
        while steps<target:
            unit=p['units'][episode%8]
            count=2 if steps<2048 else [2,4,8][episode%3]
            env=make_env(unit,workflows(p['workflows']),count,seed);state,_=env.reset();rows=[];trace=[]
            while steps<target:
                features=ready_features(state);action,logp,value=agent.act(features);node=features['ids'][action];before=state
                state,reward,done,truncated,info=env.step_node(node);steps+=1
                rows.append(dict(features=features,action=action,log_prob=logp,value=value,reward=reward.total/20.,done=bool(done or truncated)))
                trace.append(dict(state=before,node_id=node,reward=reward.total,stream=info['persistent_stream'],log_prob=logp,value=value))
                if done or truncated:break
            bootstrap=0. if rows[-1]['done'] else agent.act(ready_features(state),True)[2]
            gae=0.;next_value=bootstrap
            for row in reversed(rows):
                delta=row['reward']+(0. if row['done'] else next_value)-row['value']
                gae=delta+(0. if row['done'] else .95*gae)
                row['advantage']=gae;row['return']=gae+row['value'];next_value=row['value']
            pending+=rows;episode+=1
            write_trace(out/f'train_{episode}.json.gz',trace)
            history.append(dict(episode=episode,training_steps=steps,window=unit['index'],request_count=count,
                actual_terminal=rows[-1]['done'],reward=sum(r['reward'] for r in trace),completed=trace[-1]['stream']['completed_requests']))
            if len(pending)>=256 or steps==target:
                updates.append(agent.learn(pending));pending=[]
                dump(out/'updates.json',updates);dump(out/'training.json',history)
            dump(out/'status.json',dict(status='running',steps=steps,target=target,episodes=episode,updates=len(updates)))
        agent.save(out/f'checkpoint_{steps}.pt',provenance)
        validation+=evaluate_policy(out,p,agent,method,seed,steps,p['units'][:4] if pilot else p['units'][8:],(2,8) if pilot else (8,))
        dump(out/'validation.json',validation)
        print(json.dumps(dict(method=method,seed=seed,steps=steps,updates=len(updates))),flush=True)
    if not pilot:
        validation+=evaluate_policy(out,p,agent,method,seed,budget,p['units'][8:],(2,))
        dump(out/'validation.json',validation)
    dump(out/'completion.json',dict(status='complete',steps=steps,episodes=episode,updates=len(updates),
        validation_rows=len(validation),provenance=provenance))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['qualify','pilot','train'],default='qualify')
    parser.add_argument('--method',choices=['graph_ppo','set_ppo'],default='graph_ppo');parser.add_argument('--seed',type=int,default=7)
    a=parser.parse_args();p=json.loads(Path(a.inputs).read_text());p['_input_sha256']=sha(a.inputs);out=Path(a.output)
    if a.phase=='qualify':qualify(out,p)
    else:train_cell(out,p,a.method,a.seed,2048 if a.phase=='pilot' else 8192,a.phase=='pilot')

if __name__=='__main__':main()
