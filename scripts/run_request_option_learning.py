"""Fixed request-level PPO signal test; no graph or expert superiority claim."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_ready_workflow_learning import make_env,node_runtime,request_remaining,workflows,result_from
from scripts.run_workflow_recovery import write_trace
from scripts.run_persistent_workflow_learning import dump,sha
from src.encoders.request_option_encoder import option_features
from src.agents.ppo_agent import RequestOptionPPOAgent


def execute_request(env,state,request_id):
    start=state;trace=[]
    while True:
        by_id={n['node_id']:n for n in state['workflow']['nodes']};rank={n:i for i,n in enumerate(state['workflow']['execution_order'])}
        options=[by_id[n] for n in state['ready_node_ids'] if n.startswith(request_id+':')]
        if not options:raise ValueError('Active request has no dependency-ready task')
        def score(n):
            rt=node_runtime(state,n);return (not rt['ready'],not rt['present'],rt['loading'],rank[n['node_id']])
        node=min(options,key=score)['node_id'];before=state
        state,reward,done,truncated,info=env.step_node(node)
        trace.append(dict(state=before,node_id=node,request_id=request_id,reward=reward.total,stream=info['persistent_stream']))
        completed=request_remaining(state)[request_id]==0
        handoff=bool(info['persistent_stream']['cache_event'].get('handoff_event_count',0))
        if done or truncated or completed or handoff:
            reason='terminal' if done or truncated else 'request_completed' if completed else 'handoff'
            return state,sum(r['reward'] for r in trace),bool(done or truncated),trace,reason


def episode(unit,p,agent,seed,deterministic=False,rule=None):
    env=make_env(unit,workflows(p['workflows']),8);state,_=env.reset();rows=[];trace=[]
    torch.manual_seed(seed)
    while True:
        features=option_features(state)
        if rule=='shortest_request':action=min(range(len(features['ids'])),key=lambda i:request_remaining(state)[features['ids'][i]]);logp=value=0.
        elif rule=='uniform_request':action=int(torch.randint(len(features['ids']),(1,)));logp=value=0.
        else:action,logp,value=agent.act(features,deterministic)
        request=features['ids'][action];before=state
        state,reward,done,part,reason=execute_request(env,state,request)
        rows.append(dict(features=features,action=action,log_prob=logp,value=value,reward=reward/20.,done=done,
            holding_steps=len(part),raw_reward=reward,request_id=request,termination_reason=reason,state=before))
        trace+=part
        if done:break
    return result_from(type('Envelope',(),{'core_env':env})()),rows,trace


def evaluate(out,p,agent,seed,episodes,rule=None):
    rng=torch.random.get_rng_state();rows=[]
    try:
        for unit in p['units'][8:]:
            for rep in range(5):
                deterministic=rep==0
                result,options,trace=episode(unit,p,agent,8000+unit['index']*31+rep,deterministic,rule)
                name=f'eval_e{episodes}_w{unit["index"]}_r{rep}.json.gz';write_trace(out/name,trace)
                rows.append(dict(seed=seed,episodes=episodes,window=unit['index'],replicate=rep,mode='greedy' if deterministic else 'stochastic',rule=rule,
                    trace=name,trace_sha256=sha(out/name),option_count=len(options),**result))
        return rows
    finally:torch.random.set_rng_state(rng)


def train(out,p,seed):
    out.mkdir();torch.set_num_threads(1);agent=RequestOptionPPOAgent(seed);evaluation=[];training=[];updates=[];pending=[];steps=0
    provenance=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=p['_sha'],seed=seed,
        episodes=32,training_load=8,expert_steps=0,reward='original sum over primitive steps, training divisor20',gamma=1.,gae_lambda=1.,
        fixed_fast_executor='causal ready cache-first within selected request',interruption='request completion, actual handoff, finite horizon',
        optimizer='Adam1e-3,4epochs,batch128,clip.2,entropy.02,independent actor/value gradient clipping.5',graph=False)
    dump(out/'command.json',dict(argv=sys.argv,cwd=str(ROOT),provenance=provenance))
    agent.save(out/'checkpoint_0.pt',provenance);evaluation+=evaluate(out,p,agent,seed,0);dump(out/'validation.json',evaluation)
    for ep in range(32):
        unit=p['units'][ep%8];result,rows,trace=episode(unit,p,agent,seed*10000+ep)
        steps+=len(trace);value=0.
        # FiniteH, undiscounted semi-MDP: the exact native-reward sum per option.
        for row in reversed(rows):value+=row['reward'];row['return']=value;row['advantage']=value-row['value']
        pending+=rows;write_trace(out/f'train_{ep+1}.json.gz',trace)
        dump(out/f'options_{ep+1}.json',[{k:v for k,v in r.items() if k!='features'} for r in rows])
        training.append(dict(episode=ep+1,window=unit['index'],native_steps=steps,options=len(rows),**result))
        if (ep+1)%4==0:updates.append(agent.learn(pending));pending=[]
        dump(out/'training.json',training);dump(out/'updates.json',updates)
        if ep+1 in [16,32]:
            agent.save(out/f'checkpoint_{ep+1}.pt',provenance)
            evaluation+=evaluate(out,p,agent,seed,ep+1);dump(out/'validation.json',evaluation)
            print(json.dumps(dict(seed=seed,episodes=ep+1,native_steps=steps,updates=len(updates))),flush=True)
    dump(out/'completion.json',dict(status='complete',native_steps=steps,episodes=32,updates=len(updates),validation_rows=len(evaluation),provenance=provenance))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=7)
    a=parser.parse_args();p=json.loads(Path(a.inputs).read_text());p['_sha']=sha(a.inputs);train(Path(a.output),p,a.seed)

if __name__=='__main__':main()
