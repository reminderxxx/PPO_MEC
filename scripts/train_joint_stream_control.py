"""Matched session learning with released FIFO controls and full physical team tails."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from scripts.launch_eight_vehicle_joint_experiment import _hash_file
from scripts.train_joint_tail_control import tensor_hash

NAMES=['sa_ghmappo','sa_no_prediction','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl']


def actor_hash(agent):
    return hashlib.sha256(b''.join(v.numpy().tobytes() for v in agent.heads.state_dict().values())).hexdigest()


def make_agent(name,seed,budget):
    method='sa_ghmappo' if name=='sa_no_prediction' else name
    return build_agent(method,control_contract='joint_stream_control_v1',seed=seed,training_budget=budget,
                       prediction_enabled=name!='sa_no_prediction',return_gamma=1.)


def make_env(case,data,manifest):
    session=data['sessions'][case['session']]
    return MultiRequestStreamEnv(frames=session['frames'],arrivals=case['arrivals'],catalog=case['catalog'],
        rsu_states=session['rsus'],costs=ReplayCosts(**manifest['costs']['replay_costs']),
        prediction_provider=LearnedMobilityPredictor(data['predictor']),**case['options'])


def stream_episode(env,agent,*,deterministic=False,event_file=None,receipt_file=None,identity=None):
    agent.reset_controller();total=0.;rolls=[];rewards=[];mechanisms=Counter();identity=identity or {}
    fleet=len(env.workflows)  # Public physical fleet, not future arrival/request count.
    while not env._control_episode_done():
        frame=env.frame_index;decisions=env.decision_states();controls,roll=agent.act_controls(decisions,frame,deterministic=deterministic)
        _,reward,done,info=env.step_controls(controls);receipt=info['frame_cost_receipt']
        assert abs(reward-sum(receipt['reward_components'].values()))<1e-9
        rewards.append(reward);total+=reward
        if roll is not None:
            roll.update(frame_index=frame,initial_request_count=fleet,episode_end=False)
            by={e['vehicle_id']:e for e in info['events']}
            roll['cost_targets']=torch.tensor([[sum(np.ceil(env.model_cache.objects[k].transfer_size_bytes/env.model_cache.transfer_bytes_per_frame) for k in by[v]['cache_request'].get('admitted_object_ids',[]))/10,
                by[v]['state_transfer_link_frames']/10] for v in roll['vehicle_ids']],dtype=torch.float32)
            rolls.append(roll);mechanisms['migration_opportunities']+=agent.last_diagnostics['migration_opportunities']
        for event in info['events']:
            event['control_planning']=agent.last_diagnostics['by_vehicle'][event['vehicle_id']]
            mechanisms['cache_commands']+=event['control']['cache_target'] is not None
            mechanisms['prepare_commands']+=event['control']['prepare_target'] is not None
            mechanisms['queued_cache_commands']+=event['control']['cache_request_id'] is not None and event['control']['cache_request_id']!=event['request_id']
            if event_file is not None:event_file.write(json.dumps({**identity,**event})+'\n')
        if receipt_file is not None:receipt_file.write(json.dumps({**identity,**receipt})+'\n')
    suffix=0.;returns={}
    for frame,value in reversed(list(enumerate(rewards))):suffix=value/10+suffix;returns[frame]=suffix
    for roll in rolls:
        roll['returns']=torch.full_like(roll['values'],returns[roll['frame_index']])
        roll['team_credit_targets']=roll['returns']/fleet
        roll['credit_targets']=roll['team_credit_targets'].clone()
    if rolls:rolls[-1]['episode_end']=True
    return dict(reward=total,**env.summary(),mechanisms=dict(mechanisms),decision_rows=sum(len(r['actions']) for r in rolls)),rolls


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output_root',type=Path,required=True);p.add_argument('--agent',choices=NAMES,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    manifest=json.loads(a.protocol.read_text());home=a.protocol.parent
    for name,sha in manifest['assets'].items():assert _hash_file(home/name)==sha
    for name,sha in manifest['source_fingerprints'].items():assert _hash_file(ROOT/name)==sha,name
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not manifest['smoke']:raise ValueError('clean frozen source required')
    data=torch.load(home/'protocol.pt',weights_only=False)
    output=a.output_root/a.agent;output.mkdir(parents=True,exist_ok=False)
    episodes=manifest['episodes'];metadata=dict(status='running',agent=a.agent,protocol=str(a.protocol.resolve()),protocol_sha256=_hash_file(a.protocol),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=dirty,
        checkpoints=[],command=sys.argv,training_episodes=0,training_frames=0,training_decisions=0,eval_cells=0)
    out_manifest=output/'manifest.json';out_manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'command.json').write_text(json.dumps(sys.argv,indent=2)+'\n')
    with (output/'training.jsonl').open('w') as train,(output/'dev.jsonl').open('w') as dev,(output/'events.jsonl').open('w') as events,(output/'receipts.jsonl').open('w') as receipts:
        for seed in manifest['seeds']:
            random.seed(seed);np.random.seed(seed);agent=make_agent(a.agent,seed,episodes);initial=tensor_hash(agent);initial_actor=actor_hash(agent);buffer=[]
            for ep,case_id in enumerate(manifest['training_schedule'][str(seed)],1):
                case=next(c for c in data['cases']['train'] if c['case_id']==case_id)
                row,rolls=stream_episode(make_env(case,data,manifest),agent);buffer+=rolls
                loss=agent.update(buffer) if ep%4==0 else None
                if loss is not None:buffer=[]
                train.write(json.dumps(dict(seed=seed,episode=ep,case_id=case_id,losses=loss,**row))+'\n');train.flush()
                metadata['training_episodes']+=1;metadata['training_frames']+=row['steps'];metadata['training_decisions']+=row['decision_rows']
                if ep not in manifest['checkpoint_episodes']:continue
                path=output/f'{a.agent}_seed{seed}_ep{ep}.pt';agent.save_checkpoint(path)
                metadata['checkpoints'].append(dict(seed=seed,episode=ep,path=path.name,sha256=_hash_file(path),initial_tensor_sha256=initial,trained_tensor_sha256=tensor_hash(agent),initial_actor_sha256=initial_actor,trained_actor_sha256=actor_hash(agent)))
                rng=torch.get_rng_state();evaluated=type(agent).from_checkpoint(path);assert tensor_hash(evaluated)==tensor_hash(agent)
                selected=data['cases']['dev'] if ep==episodes else [c for c in data['cases']['dev'] if c['case_id'] in manifest['curve_probe_ids']]
                for c in selected:
                    identity=dict(agent=a.agent,rule=f'{a.agent}_seed{seed}',seed=seed,episode=ep,case_id=c['case_id'])
                    row,_=stream_episode(make_env(c,data,manifest),evaluated,deterministic=True,
                        event_file=events if ep==episodes else None,receipt_file=receipts if ep==episodes else None,identity=identity)
                    dev.write(json.dumps(dict(identity,**row))+'\n');metadata['eval_cells']+=1
                torch.set_rng_state(rng);dev.flush();events.flush();receipts.flush();out_manifest.write_text(json.dumps(metadata,indent=2)+'\n')
                print(json.dumps(dict(agent=a.agent,seed=seed,episode=ep,phase='checkpoint_evaluation_complete')),flush=True)
    metadata.update(status='complete',output_sha256={f:_hash_file(output/f) for f in ['training.jsonl','dev.jsonl','events.jsonl','receipts.jsonl','command.json']})
    out_manifest.write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
