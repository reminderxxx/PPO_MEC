"""Fixed-budget train/dev comparison for the opt-in v3 control contract."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from math import ceil
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
    initial_request_count=len(env.decision_states())
    while True:
        states = env.decision_states()
        if agent is None:
            targets = select_targets(states, 'reachable_balanced_reactive', assignments)
            controls = {vid: JointRequestControl(targets[vid], targets[vid],
                d['semantic_state']['current_workflow_node']['node_id'] if targets[vid] else None) for vid,d in states.items()}
        else:
            controls, rollout = agent.act_controls(states, env.frame_index, deterministic=deterministic)
            rollout['initial_request_count']=initial_request_count
            rolls.append(rollout)
        _, value, done, info = env.step_controls(controls)
        assert abs(value-sum(info['events'][0]['frame_cost_receipt']['reward_components'].values())) < 1e-10
        reward += value; rewards.append(value)
        if agent is not None:
            by_vehicle = {e['vehicle_id']: e for e in info['events']}
            receipt=info['events'][0]['frame_cost_receipt']
            if hasattr(agent,'credit_mode'):
                rollout['request_rewards'] = receipt['request_rewards']
                mechanisms['migration_opportunities'] += agent.last_diagnostics['migration_opportunities']
            scheduled=getattr(agent,'checkpoint_contract','').startswith('scheduled_')
            rollout['cost_targets'] = torch.tensor([[
                (sum(ceil(env.model_cache.objects[key].transfer_size_bytes/env.model_cache.transfer_bytes_per_frame) for key in by_vehicle[vid]['cache_request'].get('admitted_object_ids',[])) if scheduled else by_vehicle[vid]['cache_request']['link_slots_consumed'])/10,
                by_vehicle[vid]['state_transfer_link_frames']/10] for vid in rollout['vehicle_ids']], dtype=torch.float32)
        for e in info['events']:
            control = e['control']
            if agent is not None and hasattr(agent,'credit_mode'):
                e['control_planning']=agent.last_diagnostics['by_vehicle'][e['vehicle_id']]
                mechanisms['net_positive_prepare_commands'] += int(control['prepare_target'] in e['control_planning']['migration_candidates'])
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
    target = 0.; local_returns={}
    for rollout, value in reversed(list(zip(rolls, rewards))):
        target = value/10 + .99*target
        rollout['returns'] = torch.full_like(rollout['values'], target)
        if 'request_rewards' in rollout:
            for vid in set(local_returns)|set(rollout['request_rewards']):
                local_returns[vid] = rollout['request_rewards'].get(vid,0.)/10+.99*local_returns.get(vid,0.)
            local=torch.tensor([local_returns[vid] for vid in rollout['vehicle_ids']],dtype=torch.float32)
            rollout['request_returns']=local
            rollout['team_credit_targets']=torch.full_like(local,target/initial_request_count)
            rollout['credit_targets']=.5*local+.5*target/initial_request_count
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
    p.add_argument('--coordinated_controls',action='store_true')
    p.add_argument('--partial_preparation',action='store_true')
    p.add_argument('--scheduled_controls',action='store_true')
    a = p.parse_args()
    if a.episodes < 4 or a.episodes % 4 or a.episodes % a.eval_every:
        raise ValueError('budget must be a positive multiple of update=4 and eval_every')
    if a.scheduled_controls:a.partial_preparation=a.coordinated_controls=True
    if a.partial_preparation and not a.coordinated_controls:
        raise ValueError('--partial_preparation requires --coordinated_controls')
    if a.coordinated_controls:
        a.productive_controls=True
    allowed = {'sa_ghmappo','mappo','cache_offload_drl','sa_no_surrogate','sa_no_prediction','sa_no_coordination','sa_team_credit','sa_ungated_prediction'}
    if a.scheduled_controls:allowed.update({'ppo','dqn','dag_offload_drl','dt_handoff_drl','sa_constant_update'})
    if not a.coordinated_controls and set(a.agents)&{'sa_no_coordination','sa_team_credit','sa_ungated_prediction'}:
        raise ValueError('coordinated ablations require --coordinated_controls')
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
    if a.coordinated_controls:
        options['coordination_intent_enabled'] = True
    if a.scheduled_controls:options['link_reservation_enabled']=True
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_development')
    output = a.output_root/run_id; output.mkdir(parents=True,exist_ok=False)
    metadata = dict(run_id=run_id,status='running',command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        scheduled_controls=a.scheduled_controls,protocol='joint_parameterized_scheduled_v1' if a.scheduled_controls else 'joint_parameterized_ppo_coordinated_partial_v3' if a.partial_preparation else ('joint_parameterized_ppo_coordinated_v1' if a.coordinated_controls else ('joint_parameterized_ppo_productive_v2' if a.productive_controls else 'joint_parameterized_ppo_development_v1')),output_aware_cache=a.partial_preparation,partial_preparation=a.partial_preparation,coordinated_controls=a.coordinated_controls,productive_controls=a.productive_controls,action_contract='joint_parameterized_control_v3',
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
    if a.coordinated_controls:
        metadata['training'].update(target_commitment='8_frame_intent_then_hold_inflight_node_until_coverage_loss',actor_objective='conditional_head_clipped_ppo_prefix_baselines_v1',
            credit='half_local_ledger_half_team_per_initial_request',coordination='rotating_request_order_charged_intent_reservations',
            queue_prior='negative_reserved_service_plus_state_model_cost_over_remaining_horizon_per_rsu',
            migration_gate='partial_prefix_and_model_only_net_proxy_v2' if a.partial_preparation else 'forecast_execution_coverage_loss_positive_net_reward_proxy')
    if a.scheduled_controls:
        from src.agents.registry import ALGO_REGISTRY
        supported={'sa_ghmappo','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'}
        metadata['baseline_contract_inventory']={name:dict(scheduled_supported=name in supported,legacy_action_contract=spec['action_contract'],reason='parameterized implementation included' if name in supported else 'legacy action/checkpoint contract; not silently substituted') for name,spec in ALGO_REGISTRY.items()}
        metadata['training'].update(actor_objective='conditional_head_ppo_action_conditioned_policy_average_baseline',credit='half_local_half_team_mc_with_action_conditioned_baselines',link_calendar='eight_frame_public_io_model_state_intents_current_environment_grants',feature_scale='declared_service_frames_log1p_over_log121',stability='linear_lr_3e-4_to_3e-5_and_kl_limit_0.02; sa_constant_update_disables_both',entropy='per_head_divided_by_log_legal_candidate_count',train_probe_indices=[0,5,10,15],prediction_horizon=8,dqn='three_stage_bellman_replay4096_batch128_32_steps_per4episodes_target_sync2updates_epsilon0.3_to0.05; no_PPO_loss')
        metadata['limitations'].append('action-conditioned policy-average baseline is learned from on-policy returns, not an exact causal Q oracle; 3-frame future link reservations are estimates; existing model transfers nonpreemptive')
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    bundle = fit_predictor(splits['train'],**(dict(horizon=8) if a.scheduled_controls else {}))
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
                    surrogate_enabled=name!='sa_no_surrogate',prediction_enabled=name!='sa_no_prediction',productive_controls=a.productive_controls,
                    **(dict(coordinated_controls=True,coordination_enabled=name!='sa_no_coordination',
                            credit_mode='team' if name=='sa_team_credit' else 'request_team_blend',partial_preparation_enabled=a.partial_preparation,output_aware_cache=a.partial_preparation,
                            migration_gate_enabled=name!='sa_ungated_prediction',
                            **(dict(scheduled_controls=True,stability_enabled=name!='sa_constant_update',training_budget=a.episodes) if a.scheduled_controls else {})) if a.coordinated_controls else {}))
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
                        if a.scheduled_controls:
                            with (output/'train_probe.jsonl').open('a') as probe:
                                for i in (0,5,10,15):
                                    w=splits['train'][i];row,_=episode(make_env(w),evaluated,deterministic=True)
                                    probe.write(json.dumps(dict(agent=name,seed=seed,episode=ep,frame_offset=w['frame_offset'],reward=row['reward'],completed_requests=row['completed_requests']))+'\n')
                            env=make_env(splits['dev'][0]);graphs=[]
                            for decision in env.decision_states().values():
                                data,*_=evaluated._coordinated_features(decision,0,{})
                                graphs.append(torch.as_tensor(data['graph'])[torch.as_tensor(data['graph_mask'])])
                            with torch.no_grad():fraction=float((evaluated.graph_encoder(torch.cat(graphs)).abs()>.99).float().mean())
                            with (output/'representation_diagnostics.jsonl').open('a') as representation:
                                representation.write(json.dumps(dict(agent=name,seed=seed,episode=ep,first_dev_initial_graph_tanh_saturation=fraction))+'\n')
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
