"""Fixed-budget train/dev comparison for the opt-in v3 control contract."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import arguments, _hash_file, _load_protocol
from scripts.verify_joint_target_control import select_targets
from src.agents.registry import build_agent
from src.envs.core.joint_learned_prediction import fit_predictor, LearnedMobilityPredictor
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv, JointRequestControl
from src.envs.core.multi_request_replay_env import ReplayCosts


def episode(env, agent=None, *, deterministic=False, event_file=None, identity=None):
    if agent is not None:
        agent.reset_controller()
    assignments, reward, rolls, rewards, mechanisms = {}, 0., [], [], Counter()
    while True:
        states = env.decision_states()
        if agent is None:
            targets = select_targets(states, 'reachable_balanced_reactive', assignments)
            controls = {vid: JointRequestControl(targets[vid], targets[vid],
                d['semantic_state']['current_workflow_node']['node_id'] if targets[vid] else None) for vid,d in states.items()}
        else:
            controls, rollout = agent.act_controls(states, env.frame_index, deterministic=deterministic)
            rolls.append(rollout)
        _, value, done, info = env.step_controls(controls)
        assert abs(value-sum(info['events'][0]['frame_cost_receipt']['reward_components'].values())) < 1e-10
        reward += value; rewards.append(value)
        if agent is not None:
            by_vehicle = {e['vehicle_id']: e for e in info['events']}
            rollout['cost_targets'] = torch.tensor([[
                by_vehicle[vid]['cache_request']['link_slots_consumed']/10,
                by_vehicle[vid]['state_transfer_link_frames']/10] for vid in rollout['vehicle_ids']], dtype=torch.float32)
        for e in info['events']:
            control = e['control']
            mechanisms['cache_commands'] += int(control['cache_target'] is not None)
            mechanisms['future_cache_commands'] += int(control['cache_node_id'] is not None and control['cache_node_id'] != e['node_id'])
            mechanisms['base_only_commands'] += int(control['cache_target'] is not None and control['cache_kind']=='base_model')
            mechanisms['prepare_commands'] += int(control['prepare_target'] is not None)
            mechanisms['prepared_on_arrival'] += int(e['prepared_on_arrival'])
            if event_file is not None:
                event_file.write(json.dumps({**identity, **e})+'\n')
        if done:
            break
    # Every request receives the full team tail, including after its own completion.
    target = 0.
    for rollout, value in reversed(list(zip(rolls, rewards))):
        target = value/10 + .99*target
        rollout['returns'] = torch.full_like(rollout['values'], target)
    return dict(reward=reward, **env.summary(), mechanisms=dict(mechanisms)), rolls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_manifest',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--eval_every',type=int,default=32)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--agents',nargs='+',default=['sa_ghmappo','mappo','cache_offload_drl','sa_no_surrogate','sa_no_prediction'])
    p.add_argument('--output_root',type=Path,default=ROOT/'artifacts/joint_parameterized_learning')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--productive_controls',action='store_true')
    a = p.parse_args()
    if a.episodes < 4 or a.episodes % 4 or a.episodes % a.eval_every:
        raise ValueError('budget must be a positive multiple of update=4 and eval_every')
    allowed = {'sa_ghmappo','mappo','cache_offload_drl','sa_no_surrogate','sa_no_prediction'}
    if not set(a.agents) <= allowed:
        raise ValueError('unsupported comparison agent')
    torch.set_num_threads(a.threads)
    source = json.loads(a.source_manifest.read_text())
    if not source.get('development_only') or source.get('environment_revision') != 'v2':
        raise ValueError('requires frozen v2 development source')
    original = arguments(source['command'][1:])
    for name,path in [('ngsim',original.mobility_csv_path),('alibaba',original.workflow_csv_path)]:
        if _hash_file(path) != source['raw_data_sha256'][name]:
            raise ValueError('source hash differs')
    splits, workflows, rsus = _load_protocol(original)
    for split in ('train','dev'):
        public = [{k:v for k,v in w.items() if k!='frames'} for w in splits[split]]
        if public != source['splits'][split]:
            raise ValueError('split differs')
    options = {k:v for k,v in source['environment_options'].items() if k not in ('environment_revision','strict_cold_cache_mask')}
    options['model_cache_config'] = source['costs']['model_cache']
    if a.productive_controls:
        options['service_load_sharing'] = True
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_development')
    output = a.output_root/run_id; output.mkdir(parents=True,exist_ok=False)
    metadata = dict(run_id=run_id,status='running',command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        protocol='joint_parameterized_ppo_productive_v2' if a.productive_controls else 'joint_parameterized_ppo_development_v1',productive_controls=a.productive_controls,action_contract='joint_parameterized_control_v3',
        observation_contract='joint_request_observation_v3',development_only=True,paper_claim_ready=False,
        holdout_status='sealed_not_evaluated',source_manifest_sha256=_hash_file(a.source_manifest),
        raw_data_sha256=source['raw_data_sha256'],splits={k:source['splits'][k] for k in ('train','dev')},
        environment_options=options,costs=source['costs'],agents=a.agents,seeds=a.seeds,
        training=dict(episodes=a.episodes,update_every=4,epochs=4,batch_size=512,learning_rate=.0003,
            gamma=.99,reward_scale=.1,clip=.2,value_coef=.5,entropy_coef=.01,surrogate_coef=.2,
            cache_period=4,target_commitment='until_node_change_or_coverage_loss',
            checkpoint_selection='fixed_final_budget_no_dev_selection',eval_every=a.eval_every),
        limitations=['internal synthetic-cost simulation','first-party cache PPO, not external paper reproduction',
            'typed LRU fallback, no learned explicit eviction','surrogate predicts observational team return and per-command cache/state costs, not a causal counterfactual model',
            'serial DAG and single initial request batch','shared control bytes charged outside data link slots'])
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    bundle = fit_predictor(splits['train'])
    torch.save(bundle,output/'predictor.pt')
    metadata['predictor'] = {k:v for k,v in bundle.items() if k!='state_dict'}
    metadata['predictor']['sha256'] = _hash_file(output/'predictor.pt')
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    predictor = LearnedMobilityPredictor(bundle)
    def make_env(window):
        return MultiRequestTargetControlEnv(frames=window['frames'],
            vehicle_workflows=dict(zip(window['vehicle_ids'],workflows)),rsu_states=rsus,
            costs=ReplayCosts(**source['costs']['replay_costs']),prediction_provider=predictor,**options)
    print(json.dumps({'output':str(output),'predictor':metadata['predictor']}),flush=True)
    started = time.monotonic(); checkpoints = []; results = []
    with (output/'training.jsonl').open('w') as training, (output/'dev.jsonl').open('w') as dev, (output/'events.jsonl').open('w') as events:
        # Rule runs are a fixed comparator, never an actor teacher.
        for w in splits['dev']:
            row,_ = episode(make_env(w),event_file=events,identity=dict(agent='reachable_balanced_reactive',seed=None,episode=0,frame_offset=w['frame_offset']))
            record=dict(agent='reachable_balanced_reactive',seed=None,episode=0,frame_offset=w['frame_offset'],**row)
            dev.write(json.dumps(record)+'\n'); results.append(record)
        for name in a.agents:
            algorithm = 'sa_ghmappo' if name.startswith('sa_') else name
            for seed in a.seeds:
                random.seed(seed); np.random.seed(seed)
                agent = build_agent(algorithm,control_contract='joint_parameterized_control_v3',seed=seed,
                    surrogate_enabled=name!='sa_no_surrogate',prediction_enabled=name!='sa_no_prediction',productive_controls=a.productive_controls)
                initial_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in agent.state_dict().values())).hexdigest()
                rollout_buffer = []
                for ep in range(1,a.episodes+1):
                    window = splits['train'][(ep-1)%len(splits['train'])]
                    row, rolls = episode(make_env(window),agent)
                    rollout_buffer += rolls
                    losses = agent.update(rollout_buffer) if ep%4==0 else None
                    if losses is not None:
                        rollout_buffer = []
                    training.write(json.dumps(dict(agent=name,seed=seed,episode=ep,frame_offset=window['frame_offset'],losses=losses,**row))+'\n'); training.flush()
                    if ep%a.eval_every==0:
                        path = output/f'{name}_seed{seed}_ep{ep}.pt'; agent.save_checkpoint(path)
                        checkpoints.append(dict(agent=name,seed=seed,episode=ep,path=path.name,sha256=_hash_file(path),
                            parameters=sum(v.numel() for v in agent.parameters()),initial_tensor_sha256=initial_hash,
                            trained_tensor_sha256=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in agent.state_dict().values())).hexdigest()))
                        # Exact reload; evaluation must not advance training RNG.
                        rng = torch.get_rng_state(); evaluated = type(agent).from_checkpoint(path)
                        for key,value in agent.state_dict().items():
                            assert torch.equal(value,evaluated.state_dict()[key])
                        for w in splits['dev']:
                            identity=dict(agent=name,seed=seed,episode=ep,frame_offset=w['frame_offset'])
                            row,_ = episode(make_env(w),evaluated,deterministic=True,
                                event_file=events if ep==a.episodes else None,identity=identity)
                            record={**identity,**row}; dev.write(json.dumps(record)+'\n'); results.append(record)
                        torch.set_rng_state(rng); dev.flush(); events.flush()
                        final = [r for r in results if r['agent']==name and r['seed']==seed and r['episode']==ep]
                        print(json.dumps(dict(agent=name,seed=seed,episode=ep,reward=np.mean([r['reward'] for r in final]),
                            completed=np.mean([r['completed_requests'] for r in final]),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    summary = {}
    for name in ['reachable_balanced_reactive',*a.agents]:
        rows = [r for r in results if r['agent']==name and r['episode'] in (0,a.episodes)]
        summary[name] = dict(mean_reward=float(np.mean([r['reward'] for r in rows])),
            mean_completed_requests=float(np.mean([r['completed_requests'] for r in rows])),
            mean_completed_nodes=float(np.mean([r['completed_nodes'] for r in rows])),n=len(rows))
    metadata.update(status='complete',summary=summary,checkpoints=checkpoints,
                    finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(output),'summary':summary},indent=2),flush=True)


if __name__ == '__main__':
    main()
