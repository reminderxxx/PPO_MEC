"""Team node-delivery training; fixed common scenarios, rewards, predictor and budgets."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_joint_parameterized_control import episode
from scripts.train_joint_tail_control import make_env,tensor_hash
from scripts.launch_eight_vehicle_joint_experiment import _hash_file
from src.agents.registry import build_agent
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv

NAMES=['sa_ghmappo','mappo','sa_no_prediction','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl',
       'sa_fixed_entropy','sa_no_surrogate']


def make_agent(name,seed,budget):
    method='sa_ghmappo' if name.startswith('sa_') else name
    return build_agent(method,control_contract='joint_parameterized_control_v3',delivery_controls=True,
        seed=seed,training_budget=budget,prediction_enabled=name!='sa_no_prediction',
        surrogate_enabled=name!='sa_no_surrogate',return_gamma=1.,entropy_anneal=name!='sa_fixed_entropy',
        **(dict(terminal_team_tail=True) if name=='dqn' else {}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output_root',type=Path,required=True)
    p.add_argument('--agent',choices=NAMES,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--eval_every',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(1)
    if a.episodes<4 or a.episodes%4 or a.episodes%a.eval_every:raise ValueError('matched multiple-of-four budgets required')
    protocol=json.loads(a.protocol.read_text());home=a.protocol.parent
    for filename,key in [('protocol.pt','protocol_sha256'),('predictor.pt','predictor_sha256')]:assert _hash_file(home/filename)==protocol[key]
    data=torch.load(home/'protocol.pt',weights_only=False);data['predictor']=torch.load(home/'predictor.pt',weights_only=False)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_development');output=a.output_root/run_id;output.mkdir(parents=True,exist_ok=False)
    metadata=dict(run_id=run_id,status='running',command=sys.argv,protocol='joint_team_node_delivery_v1',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        action_contract='joint_parameterized_control_v3',agent=a.agent,seeds=a.seeds,episodes=a.episodes,eval_every=a.eval_every,
        prepared_protocol=str(a.protocol.resolve()),prepared_manifest_sha256=_hash_file(a.protocol),
        environment_options=protocol['environment_options'],costs=protocol['costs'],
        prediction_sha256=protocol['predictor_sha256'],primary_panels=protocol['primary_panels'],
        return_gamma=1.,critic_units='reward_scale_0.1_per_initial_request',credit='team_mc_per_initial_request_including_terminal_tail',
        protection='two_pass_actual_execution_targets_reactive_model_input_output_before_idle_preparation',
        actor='cache_context_plus_zero_initialized_bounded_DAG_residual_for_SA',surrogate='team_Q_control_variate_separate_gradient_clip_no_logit_bonus',
        entropy_anneal=a.agent!='sa_fixed_entropy',
        primary_checkpoint=f'fixed_final{a.episodes}_no_peak_selection',train_probe_indices=[0,11,20,31],
        holdout_status='sealed_not_evaluated',paper_claim_ready=False,checkpoints=[])
    manifest=output/'manifest.json';manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.monotonic()
    with (output/'training.jsonl').open('w') as training,(output/'dev.jsonl').open('w') as dev,(output/'events.jsonl').open('w') as events,(output/'train_probe.jsonl').open('w') as probe:
        for seed in a.seeds:
            random.seed(seed);np.random.seed(seed);agent=make_agent(a.agent,seed,a.episodes);initial=tensor_hash(agent);buffer=[]
            for ep in range(1,a.episodes+1):
                w=data['scenarios']['train'][(ep-1)%len(data['scenarios']['train'])]
                row,rolls=episode(make_env(w,data,protocol),agent);buffer+=rolls
                losses=agent.update(buffer) if ep%4==0 else None
                if losses is not None:buffer=[]
                training.write(json.dumps(dict(agent=a.agent,seed=seed,episode=ep,scenario_id=w['scenario_id'],frame_offset=w['frame_offset'],panel=w['panel'],losses=losses,**row))+'\n');training.flush()
                if ep%a.eval_every:continue
                checkpoint=output/f'{a.agent}_seed{seed}_ep{ep}.pt';agent.save_checkpoint(checkpoint)
                metadata['checkpoints'].append(dict(seed=seed,episode=ep,path=checkpoint.name,sha256=_hash_file(checkpoint),initial_tensor_sha256=initial,trained_tensor_sha256=tensor_hash(agent)))
                rng=torch.get_rng_state();evaluated=type(agent).from_checkpoint(checkpoint);assert tensor_hash(evaluated)==tensor_hash(agent)
                values=[]
                for window in data['scenarios']['dev']:
                    identity=dict(agent=a.agent,seed=seed,episode=ep,scenario_id=window['scenario_id'],panel=window['panel'],frame_offset=window['frame_offset'])
                    row,_=episode(make_env(window,data,protocol),evaluated,deterministic=True,event_file=events if ep==a.episodes else None,identity=identity)
                    dev.write(json.dumps(dict(identity,**row))+'\n');values.append(row)
                for i in metadata['train_probe_indices']:
                    window=data['scenarios']['train'][i];row,_=episode(make_env(window,data,protocol),evaluated,deterministic=True)
                    probe.write(json.dumps(dict(agent=a.agent,seed=seed,episode=ep,scenario_id=window['scenario_id'],panel=window['panel'],reward=row['reward'],completed_requests=row['completed_requests']))+'\n')
                torch.set_rng_state(rng);dev.flush();probe.flush();events.flush()
                manifest.write_text(json.dumps(metadata,indent=2)+'\n')
                print(json.dumps(dict(agent=a.agent,seed=seed,episode=ep,reward=np.mean([r['reward'] for r in values]),completed=np.mean([r['completed_requests'] for r in values]),elapsed=round(time.monotonic()-started,1))),flush=True)
    metadata['status']='complete';metadata['completed_at']=datetime.now(timezone.utc).isoformat();manifest.write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
