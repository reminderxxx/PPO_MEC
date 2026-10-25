"""Finite-horizon semi-MDP PPO over independent preparation opportunities."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_joint_workflow_probe import joint_env
from scripts.run_ready_workflow_learning import rule_action,node_runtime,workflows,result_from
from scripts.run_workflow_recovery import write_trace
from scripts.run_persistent_workflow_learning import dump,sha
from src.agents.ppo_agent import EventJointPPOAgent
from src.encoders.event_joint_encoder import event_features


def event_available(state):
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==state['primary_vehicle_id']);rsu=vehicle.get('associated_rsu_id')
    if not rsu or any(x['rsu_id']==rsu and x['remaining_seconds']>0 for x in state['model_loading']):return False
    by_id={n['node_id']:n for n in state['workflow']['nodes']}
    node=rule_action(state,'cache_first')
    return node_runtime(state,by_id[node])['ready'] and bool(event_features(state)['mask'][:-1].any())


def episode(unit,p,agent,seed,deterministic=False,rule=None,keep_states=True):
    env=joint_env(unit,workflows(unit.get('workflows',p['workflows'])));state,_=env.reset();rows=[];trace=[];torch.manual_seed(seed)
    while True:
        event=event_available(state);action=None
        if event:
            f=event_features(state)
            if rule=='uniform':action=int(torch.distributions.Categorical(probs=f['mask'].float()).sample());lp=v=0.
            elif rule=='no_prefetch':action=5;lp=v=0.
            elif rule=='shortest_frontier':
                action=min([i for i in range(5) if f['mask'][i]],key=lambda i:float(f['aux'][i,2]));lp=v=0.
            else:action,lp,v=agent.act(f,deterministic)
            rows.append(dict(features=f,action=action,log_prob=lp,value=v,reward=0.,raw_reward=0.,holding_steps=0))
            prep=f['ids'][action]
        else:
            prep=None
        node=rule_action(state,'cache_first');by_id={n['node_id']:n for n in state['workflow']['nodes']}
        vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==state['primary_vehicle_id']);rsu=vehicle.get('associated_rsu_id')
        loading=any(x['rsu_id']==rsu and x['remaining_seconds']>0 for x in state['model_loading'])
        # Fixed causal fast executor: work on warm tasks, preserve in-flight loads.
        if not event and rsu and not loading and not node_runtime(state,by_id[node])['present']:prep=by_id[node]['required_adapter']
        before=state;state,r,done,trunc,info=env.step_joint(node,prep)
        trace.append(dict(state=before if keep_states else None,node_id=node,prepare_adapter=prep,event=event,action=action,reward=r.total,stream=info['persistent_stream']))
        if rows:
            rows[-1]['reward']+=r.total/20.;rows[-1]['raw_reward']+=r.total;rows[-1]['holding_steps']+=1
        if done or trunc:break
    # Initial uncontrolled rewards precede the first decision and cannot affect its credit.
    value=0.
    for row in reversed(rows):value+=row['reward'];row['return']=value;row['advantage']=value-row['value']
    return result_from(type('Envelope',(),{'core_env':env})()),rows,trace


def evaluate(out,p,agent,seed,ep,rule=None):
    rng=torch.random.get_rng_state();rows=[]
    try:
        units=[u for u in p['units'] if u['split'] in ('validation','dev')]
        for unit in units:
            result,decisions,trace=episode(unit,p,agent,80000+unit['index'],True,rule)
            name=f'eval_e{ep}_w{unit["index"]}.json.gz';write_trace(out/name,trace)
            rows.append(dict(seed=seed,episode=ep,window=unit['index'],rule=rule,decisions=len(decisions),trace=name,trace_sha256=sha(out/name),**result))
        return rows
    finally:torch.random.set_rng_state(rng)


def train(out,p,seed,graph,episodes,cadence):
    out.mkdir();torch.set_num_threads(1);agent=EventJointPPOAgent(seed,graph);evaluation=[];training=[];updates=[];pending=[];steps=0
    units=[u for u in p['units'] if u['split']=='train']
    provenance=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=p['_sha'],seed=seed,
        episodes=episodes,use_graph=graph,expert_steps=0,reward='native business reward summed over holding time /20 for numerical scale only',gamma=1.,gae_lambda=1.,
        fast_executor='global cache-first service with no cancellation of in-flight loading',optimizer='Adam1e-3,4epochs,batch128,clip.2,entropy.02,separate actor/value norm clipping.5',
        evaluation='greedy on all fixed dev windows at episode0 and prespecified cadence; final checkpoint, no best selection',formal=False,holdout=False)
    dump(out/'command.json',dict(argv=sys.argv,cwd=str(ROOT),provenance=provenance));agent.save(out/'checkpoint_0.pt',provenance)
    evaluation+=evaluate(out,p,agent,seed,0);dump(out/'validation.json',evaluation)
    for ep in range(episodes):
        unit=units[ep%len(units)];result,rows,trace=episode(unit,p,agent,seed*100000+ep)
        steps+=len(trace);pending+=rows;write_trace(out/f'train_{ep+1}.json.gz',trace)
        training.append(dict(episode=ep+1,window=unit['index'],native_steps=steps,decisions=len(rows),**result))
        if (ep+1)%4==0 and pending:updates.append(agent.learn(pending));pending=[]
        dump(out/'training.json',training);dump(out/'updates.json',updates)
        if (ep+1)%cadence==0 or ep+1==episodes:
            agent.save(out/f'checkpoint_{ep+1}.pt',provenance);evaluation+=evaluate(out,p,agent,seed,ep+1);dump(out/'validation.json',evaluation)
            print(json.dumps(dict(seed=seed,graph=graph,episode=ep+1,native_steps=steps,dev_reward=sum(x['reward'] for x in evaluation if x['episode']==ep+1)/len([x for x in evaluation if x['episode']==ep+1]))),flush=True)
    dump(out/'completion.json',dict(status='complete',native_steps=steps,episodes=episodes,updates=len(updates),validation_rows=len(evaluation),provenance=provenance))


def main():
    a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=7);a.add_argument('--graph',type=int,default=1);a.add_argument('--episodes',type=int,default=32);a.add_argument('--cadence',type=int,default=16);args=a.parse_args()
    p=json.loads(Path(args.inputs).read_text());p['_sha']=sha(args.inputs)
    train(Path(args.output),p,args.seed,bool(args.graph),args.episodes,args.cadence)

if __name__=='__main__':main()
