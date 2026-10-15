"""Read-only artifact audit; never updates policy weights or chooses checkpoints."""
from collections import Counter,defaultdict
from math import hypot
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from scripts.train_joint_parameterized_control import episode
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol,_hash_file
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts

def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def close(a,b):assert abs(a-b)<1e-8,(a,b)
def identity(r):return r['agent'],r['seed'],r['episode'],r['frame_offset']
def main():
 torch.set_num_threads(1)
 experiment = sys.argv[1] if len(sys.argv)>1 else 'joint_coordinated_control'
 plan_path=ROOT/f'artifacts/{experiment}/comparison_plan.json'
 plan=json.loads(plan_path.read_text()) if plan_path.exists() else None
 locations=[(ROOT/f'artifacts/{experiment}'/group/f'seed{seed}',names,seed) for group,names in plan['groups'].items() for seed in plan['seeds']] if plan else [(ROOT/f'artifacts/{experiment}/seed{seed}',None,seed) for seed in (7,17,27)]
 candidates=[list(location.glob('*/manifest.json')) for location,names,seed in locations]
 assert all(len(paths)==1 for paths in candidates), 'exactly one prespecified run per seed and group required'
 manifests=[paths[0] for paths in candidates]
 sources=[json.loads(p.read_text()) for p in manifests]
 assert not any((p.parent/'run_abort.json').exists() for p in manifests), 'aborted pilot is not a complete comparison'
 assert all(s['status']=='complete' and not s['git_dirty'] and s.get('coordinated_controls') for s in sources)
 assert len({s['git_commit'] for s in sources})==1
 if plan:
  assert all(source['agents']==names and source['seeds']==[seed] and source.get('scheduled_controls') for source,(_,names,seed) in zip(sources,locations))
  assert all(source['git_commit'].startswith(plan['source_commit']) for source in sources)
  assert all(source['training']['episodes']==plan['episodes'] for source in sources)
 if not plan:assert all(source['agents']==sources[0]['agents'] for source in sources)
 all_agents=[name for names in plan['groups'].values() for name in names] if plan else sources[0]['agents']
 assert len(all_agents)==len(set(all_agents))
 for k in ('source_manifest_sha256','raw_data_sha256','splits','training','environment_options','costs','predictor'):
  assert all(s[k]==sources[0][k] for s in sources),k
 all_windows=sources[0]['splits']['train']+sources[0]['splits']['dev']
 for i,a in enumerate(all_windows):
  for b in all_windows[i+1:]:
   if a['source_segment_id']==b['source_segment_id']:
    assert a['time_end']<b['time_start'] or b['time_end']<a['time_start']
 rows=[];checkpoint_hashes=[];tensor_hashes=[];audit_counts=Counter(); traces={}
 for path,s in zip(manifests,sources):
  directory=path.parent; seed=s['seeds'][0];budget=s['training']['episodes']
  assert s['holdout_status']=='sealed_not_evaluated' and not s['paper_claim_ready']
  assert not (directory/'formal.jsonl').exists() and not (directory/'holdout.jsonl').exists()
  assert _hash_file(directory/'predictor.pt')==s['predictor']['sha256']
  bundle=torch.load(directory/'predictor.pt',weights_only=False)
  train_offsets={w['frame_offset'] for w in s['splits']['train']};dev_offsets={w['frame_offset'] for w in s['splits']['dev']}
  assert set(bundle['fit_offsets'])|set(bundle['validation_offsets'])==train_offsets
  assert not set(bundle['fit_offsets'])&set(bundle['validation_offsets'])
  training=read(directory/'training.jsonl');dev=read(directory/'dev.jsonl')
  if s.get('scheduled_controls'):
   probes=read(directory/'train_probe.jsonl');representations=read(directory/'representation_diagnostics.jsonl')
   assert len(probes)==4*len(s['agents'])*(budget//s['training']['eval_every'])
   assert len(representations)==len(s['agents'])*(budget//s['training']['eval_every'])
   assert {r['frame_offset'] for r in probes}=={s['splits']['train'][i]['frame_offset'] for i in (0,5,10,15)}
   assert all(0<=r['first_dev_initial_graph_tanh_saturation']<=1 for r in representations)
  assert len(training)==budget*len(s['agents'])
  assert len(dev)==8+8*len(s['agents'])*(budget//s['training']['eval_every'])
  for name in s['agents']:
   group=[r for r in training if r['agent']==name]
   assert [r['episode'] for r in group]==list(range(1,budget+1))
   assert [r['frame_offset'] for r in group]==[s['splits']['train'][(ep-1)%20]['frame_offset'] for ep in range(1,budget+1)]
   assert all(r['frame_offset'] in train_offsets for r in group)
   assert sum(r['losses'] is not None for r in group)==budget//4
   assert all(np.isfinite(list(r['losses'][k] for k in ('policy_loss','value_loss','surrogate_loss'))).all() for r in group if r['losses'])
  for c in s['checkpoints']:
   assert _hash_file(directory/c['path'])==c['sha256']
   assert c['initial_tensor_sha256']!=c['trained_tensor_sha256']
   saved=torch.load(directory/c['path'],weights_only=False)
   actual=hashlib.sha256(b''.join(v.numpy().tobytes() for v in saved['state_dict'].values())).hexdigest()
   assert actual==c['trained_tensor_sha256'];checkpoint_hashes.append(c['sha256']);tensor_hashes.append(actual)
   if s.get('scheduled_controls'):
    fresh=build_agent(saved['options']['method'],control_contract=s['action_contract'],scheduled_controls=True,**{k:v for k,v in saved['options'].items() if k!='method'})
    initial=fresh.state_dict()
    reconstructed=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    assert reconstructed==c['initial_tensor_sha256']
    assert any(not torch.equal(v,saved['state_dict'][k]) for k,v in initial.items() if k.startswith('heads.')), 'actor/Q head must learn, buffer counter changes are insufficient'
    if saved['options']['method']=='dqn':assert saved['training_resume_supported'] is False and 0.05<=saved['epsilon']<=0.3
  final={identity(r):r for r in dev if r['episode'] in (0,budget)}
  events=defaultdict(list)
  with (directory/'events.jsonl').open() as f:
   for line in f:
    e=json.loads(line);events[identity(e)].append(e)
  assert set(events)==set(final)
  for key,group in events.items():
   r=final[key]; receipts=[e['frame_cost_receipt'] for e in group if 'frame_cost_receipt' in e]
   assert len(receipts)==r['steps'] and len({x['frame_index'] for x in receipts})==r['steps']
   assert all(all(0<=v<=s['costs']['replay_costs']['link_slots_per_rsu'] for v in x['link_usage_by_rsu'].values()) for x in receipts)
   close(sum(sum(x['reward_components'].values()) for x in receipts),r['reward'])
   for receipt in receipts:
    close(sum(receipt['request_rewards'].values()),sum(receipt['reward_components'].values()))
    assert receipt['control_intent_transmitted_bytes']>0
    assert receipt['shared_state_transmitted_bytes']>=receipt['control_intent_transmitted_bytes']
   assert sum(x['control_intent_transmitted_bytes'] for x in receipts)==r['control_intent_transmitted_bytes']
   completed=sum(e['status'] in ('node_completed','vehicle_fallback_completed') for e in group)
   assert completed==r['completed_nodes']
   assert len({e['vehicle_id'] for e in group if e['workflow_completed']})==r['completed_requests']
   assert sum(e['state_transfer_link_frames'] for e in group)==r['state_transfer_link_frames']
   assert sum(x['model_transfer_link_frames'] for x in receipts)==r['model_transfer_link_frames']
   assert sum(x['shared_state_transmitted_bytes'] for x in receipts)==r['shared_state_transmitted_bytes']
   for k in ('input_transfer_bytes','output_transfer_bytes','input_transfer_link_frames','output_transfer_link_frames'):
    assert sum(e[k] for e in group)==r[k]
   assert all(cache['occupied_bytes']<=cache['capacity_bytes'] for cache in r['model_cache'].values())
   assert sum(r['status_counts'].values())==len(group)
   assert sum(x['completed_nodes'] for x in r['per_request'])==r['completed_nodes']
   assert sum(x['completed'] for x in r['per_request'])==r['completed_requests']
   mechanisms=Counter()
   for e in group:
    c=e['control'];mechanisms['cache_commands']+=int(c['cache_target'] is not None)
    mechanisms['future_cache_commands']+=int(c['cache_node_id'] is not None and c['cache_node_id']!=e['node_id'])
    mechanisms['base_only_commands']+=int(c['cache_target'] is not None and c['cache_kind']=='base_model')
    mechanisms['prepare_commands']+=int(c['prepare_target'] is not None)
    mechanisms['prepared_on_arrival']+=int(e['prepared_on_arrival'])
    if 'control_planning' in e:
     candidates=e['control_planning']['migration_candidates']
     mechanisms['migration_opportunities']+=len(candidates)
     mechanisms['net_positive_prepare_commands']+=int(c['prepare_target'] in candidates)
     if e['agent']!='sa_ungated_prediction' and c['prepare_target'] is not None:
      chosen=candidates[c['prepare_target']]
      assert chosen['net_reward_proxy']>0 and chosen['prepare_feasible']
      assert chosen.get('prepare_budget_frames',chosen['prepare_frames'])+chosen['model_prefetch_frames']<=chosen['eta_frames']
      if 'state_net_reward_proxy' in chosen:assert chosen['state_net_reward_proxy']>0
   if s.get('scheduled_controls'):
    for e in group:
     grants=e['link_reservation_grants'];vid=e['vehicle_id'];control=e['control']
     for target,kind in control['link_reservations']:
      assert target in grants
     for transfer in e['state_transfers']:
      if transfer['link_frames'] and transfer['target'] in grants:
       assert grants[transfer['target']]['vehicle_id']==vid and grants[transfer['target']]['kind']=='state'
     if e['cache_request']['link_slots_consumed'] and control['cache_target'] in grants:
      grant=grants[control['cache_target']];assert grant['vehicle_id']==vid and grant['kind']=='model'
     if e['output_transfer_link_frames'] and e['output_source_rsu_id'] in grants:
      grant=grants[e['output_source_rsu_id']];assert grant['vehicle_id']==vid and grant['kind']=='output'
     if e['input_transfer_link_frames'] and e['target_rsu_id'] in grants:
      grant=grants[e['target_rsu_id']];assert grant['vehicle_id']==vid and grant['kind']=='input'
   assert dict(mechanisms)==r['mechanisms']
   traces[key]=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
   audit_counts['events']+=len(group);audit_counts['receipts']+=len(receipts)
  rows+=dev
 assert len(set(checkpoint_hashes))==len(checkpoint_hashes)
 # Equal tensors are allowed for a zero-opportunity prediction ablation; record rather than invent distinct models.
 unique_tensor_count=len(set(tensor_hashes))
 # Raw-window aggregate only; same-seed rules are deduplicated, not 24 independent rule runs.
 budget=sources[0]['training']['episodes'];final=[r for r in rows if r['episode']==budget]
 final += [r for r in rows if r['agent']=='reachable_balanced_reactive' and r['seed'] is None][:8]
 summary={};curves={};paired={}
 metrics=['reward','completed_requests','completed_nodes','steps','cache_fetch_count','state_transfer_link_frames',
          'model_transfer_link_frames','association_handoff_count','prepared_on_arrival_count','handoff_failed_count','service_target_change_count']
 for name in ['reachable_balanced_reactive',*all_agents]:
  group=[r for r in final if r['agent']==name]
  summary[name]={k:float(np.mean([r[k] for r in group])) for k in metrics}
  summary[name]['mechanisms']={k:float(np.mean([r['mechanisms'][k] for r in group])) for k in group[0]['mechanisms']}
  by_seed=[float(np.mean([r['reward'] for r in group if r['seed']==seed])) for seed in (7,17,27)] if name!='reachable_balanced_reactive' else []
  summary[name].update(n=len(group),reward_by_seed=by_seed,reward_seed_std=float(np.std(by_seed,ddof=1)) if by_seed else None)
  curves[name]={str(ep):dict(reward=float(np.mean([r['reward'] for r in rows if r['agent']==name and r['episode']==ep])),
      completed=float(np.mean([r['completed_requests'] for r in rows if r['agent']==name and r['episode']==ep])))
      for ep in sorted({r['episode'] for r in rows if r['agent']==name})}
 sa={(r['seed'],r['frame_offset']):r for r in final if r['agent']=='sa_ghmappo'}
 for name in all_agents:
  if name=='sa_ghmappo':continue
  other={(r['seed'],r['frame_offset']):r for r in final if r['agent']==name}
  differences=[sa[k]['reward']-other[k]['reward'] for k in sa]
  completion=[sa[k]['completed_requests']-other[k]['completed_requests'] for k in sa]
  by_window={str(offset):float(np.mean([sa[(seed,offset)]['reward']-other[(seed,offset)]['reward'] for seed in (7,17,27)])) for offset in sorted(dev_offsets)}
  paired[name]=dict(reward_mean=float(np.mean(differences)),reward_min=min(differences),reward_max=max(differences),
      reward_wins=sum(v>1e-9 for v in differences),reward_losses=sum(v< -1e-9 for v in differences),
      completion_mean=float(np.mean(completion)),completion_wins=sum(v>0 for v in completion),
      completion_losses=sum(v<0 for v in completion),window_mean_differences=by_window)
 # Independently reload every final model and replay first dev window (all request events checked).
 source=json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
 assert _hash_file(ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json')==sources[0]['source_manifest_sha256']
 splits,workflows,rsus=_load_protocol(arguments(source['command'][1:]))
 for split in ('train','dev'):
  public=[{k:v for k,v in w.items() if k!='frames'} for w in splits[split]]
  assert public==sources[0]['splits'][split]
  for w in splits[split]:
   assert min(f['time_index'] for f in w['frames'])==w['time_start']
   assert max(f['time_index'] for f in w['frames'])==w['time_end']
 timeline={w['frame_offset']:[f['time_index'] for f in w['frames']] for w in splits['dev']}
 for path in manifests:
  with (path.parent/'events.jsonl').open() as f:
   for line in f:
    event=json.loads(line)
    assert event['time_index']==timeline[event['frame_offset']][event['frame_index']]
 physical={};concentrations={};window_map={w['frame_offset']:w for w in splits['dev']}
 for path in manifests:
  grouped=defaultdict(list)
  with (path.parent/'events.jsonl').open() as f:
   for line in f:
    event=json.loads(line);grouped[identity(event)].append(event)
  for key,group in grouped.items():
   counts,concentration=audit_events(group,window_map[key[3]],rsus)
   physical[str(key)]=counts;concentrations[str(key)]=concentration
 window=splits['dev'][0]
 reproduction=[]
 for path,s in zip(manifests,sources):
  provider=LearnedMobilityPredictor(torch.load(path.parent/'predictor.pt',weights_only=False))
  for c in s['checkpoints']:
   if c['episode']!=budget:continue
   saved=torch.load(path.parent/c['path'],weights_only=False)
   agent=build_agent(saved['options']['method'],control_contract=s['action_contract'],coordinated_controls=True,scheduled_controls=s.get('scheduled_controls',False),**{k:v for k,v in saved['options'].items() if k!='method'})
   agent=type(agent).from_checkpoint(path.parent/c['path'])
   env=MultiRequestTargetControlEnv(frames=window['frames'],vehicle_workflows=dict(zip(window['vehicle_ids'],workflows)),
      rsu_states=rsus,costs=ReplayCosts(**s['costs']['replay_costs']),prediction_provider=provider,**s['environment_options'])
   row,rolls=episode(env,agent,deterministic=True)
   surrogate_diagnostics=None
   if agent.surrogate_enabled:
    errors=[]
    with torch.no_grad():
     for roll in rolls:
      *_,prediction=agent.distributions(roll['data'],actions=roll['actions'])
      credit=roll['team_credit_targets'] if agent.credit_mode=='team' else roll['credit_targets']
      actual=torch.cat((credit[:,None],roll['cost_targets']),-1)
      errors.append((prediction-actual[:,None]).reshape(-1,3))
    error=torch.cat(errors)*10
    surrogate_diagnostics=dict(return_rmse_reward_units=float(error[:,0].square().mean().sqrt()),cache_mae_initial_link_slots=float(error[:,1].abs().mean()),state_mae_link_slots=float(error[:,2].abs().mean()))
   if surrogate_diagnostics and s.get('scheduled_controls'):surrogate_diagnostics['cache_mae_declared_admission_link_frames']=surrogate_diagnostics.pop('cache_mae_initial_link_slots')
   key=(c['agent'],c['seed'],budget,window['frame_offset']);close(row['reward'],next(r['reward'] for r in final if identity(r)==key))
   original=[]
   with (path.parent/'events.jsonl').open() as f:
    for line in f:
     e=json.loads(line)
     if identity(e)==key:original.append({k:v for k,v in e.items() if k not in ('agent','seed','episode','frame_offset')})
   assert json.loads(json.dumps(env.events))==original
   reproduction.append(dict(agent=c['agent'],seed=c['seed'],frame_offset=window['frame_offset'],reward=row['reward'],events_equal=True,surrogate_diagnostics=surrogate_diagnostics))
 report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-15',target_venue='IEEE TMC',
    policy_version='tmc_review_policy_v3_20260621',artifact_run_id=[str(p.parent.relative_to(ROOT)) for p in manifests],
    git_commit=sources[0]['git_commit'],experiment=experiment,evidence_level='E0_UNAVAILABLE_for_paper; development_raw_audited_and_first_window_reproduced',
    checks='complete manifests/source/protocol/budget/checkpoint hashes/predictor boundaries/raw receipts/completion/IO/cache caps/reproduction passed',
    physical_attribution=physical,service_concentrations=concentrations,checkpoint_count=len(checkpoint_hashes),unique_tensor_count=unique_tensor_count,counts=dict(audit_counts),summary=summary,curves=curves,paired=paired,reproductions=reproduction,
    statistical_scope='descriptive paired development comparison; 3 seeds, 8 adjacent intervals sharing same source/vehicles; no independence or significance claim',
    literature_novelty='not assessed in this engineering round',paper_claim_ready=False)
 if plan:report.update(comparison_plan_sha256=_hash_file(plan_path),configurations=all_agents)
 destination=ROOT/'artifacts/joint_coordinated_audit_20261015'/f'{experiment.replace("/","_")}_audit.json'
 destination.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'audit':str(destination),'summary':summary,'paired':paired},indent=2))
def audit_events(group,window,rsus):
    sites={r.rsu_id:r for r in rsus};locations={};explicit=set();counts=Counter();targets=Counter();compute=Counter()
    for e in group:
        vid=e['vehicle_id'];frame=e['frame_index'];target=e['target_rsu_id'];control=e['control']
        if target in sites:targets[target]+=1
        if e['status'] in ('rsu_compute','output_ready'):compute[(frame,target)]+=1
        if e['service_target_changed'] and e['prepared_on_arrival'] and (vid,target,e['state_version_before']) in explicit:
            counts['explicit_preparation_ready_arrivals']+=1
        prepare=control['prepare_target']
        if prepare is not None:
            counts['prepare_commands']+=1
            candidates=e.get('control_planning',{}).get('migration_candidates',{})
            opportunity=candidates.get(prepare)
            if opportunity:
                counts['positive_proxy_commands']+=1
                source=sites.get(locations.get(vid));eta=opportunity['eta_frames']
                if source is not None and frame+eta<len(window['frames']):
                    vehicle=next(v for v in window['frames'][frame+eta]['vehicles'] if v.vehicle_id==vid)
                    out=hypot(vehicle.position_x-source.position_x,vehicle.position_y-source.position_y)>source.coverage_radius
                    destination=sites[prepare]
                    reachable=hypot(vehicle.position_x-destination.position_x,vehicle.position_y-destination.position_y)<=destination.coverage_radius
                    counts['true_coverage_exit_and_destination_by_eta']+=int(out and reachable)
                    counts['false_or_unrealized_coverage_exit_by_eta']+=int(not(out and reachable))
            for t in e['state_transfers']:
                if t['target']==prepare and t['status']=='state_prepared':
                    explicit.add((vid,prepare,t['version']));counts['explicit_prepare_completions']+=1
        if e.get('output_source_rsu_id'):locations[vid]=e['output_source_rsu_id']
        elif e['status'] in ('rsu_compute','output_ready'):locations[vid]=target
        elif e['status'].startswith('vehicle_'):locations[vid]='vehicle:'+vid
    assert max(compute.values(),default=0)<=1
    return dict(counts),max(targets.values(),default=0)/max(sum(targets.values()),1)

if __name__=='__main__':main()
