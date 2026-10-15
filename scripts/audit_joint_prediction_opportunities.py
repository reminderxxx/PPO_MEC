"""Forecast physical coverage inventory, independent of the preparation gate."""
from pathlib import Path
from math import hypot,ceil
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts

def main():
 torch.set_num_threads(1)
 experiment=sys.argv[1] if len(sys.argv)>1 else 'joint_coordinated_control'
 key=experiment.replace('/','_')
 source=json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
 splits,workflows,rsus=_load_protocol(arguments(source['command'][1:]));sites={r.rsu_id:r for r in rsus}
 path=next((ROOT/f'artifacts/{experiment}/seed7').glob('*/manifest.json'));m=json.loads(path.read_text())
 provider=LearnedMobilityPredictor(torch.load(path.parent/'predictor.pt',weights_only=False))
 w=splits['dev'][0];mapper=MultiRequestTargetControlEnv(frames=w['frames'],vehicle_workflows=dict(zip(w['vehicle_ids'],workflows)),rsu_states=rsus,
      costs=ReplayCosts(**m['costs']['replay_costs']),**m['environment_options']).mapper
 predictions={};windows={w['frame_offset']:w for w in splits['dev']}
 for w in splits['dev']:
  provider.reset();predictions[w['frame_offset']]=[]
  for t,frame in enumerate(w['frames']):
   vehicles=frame['vehicles'];predictions[w['frame_offset']].append(provider.predict(vehicles,mapper.associate(vehicles),rsus,mapper,t))
 cases=[];true_cases=[]
 for seed in (7,17,27):
  path=next((ROOT/f'artifacts/{experiment}/seed{seed}').glob('*/events.jsonl'));locations={};payloads={}
  with path.open() as f:
   for line in f:
    e=json.loads(line)
    if e['agent']!='sa_ghmappo':continue
    vid=e['vehicle_id'];off=e['frame_offset'];t=e['frame_index'];key=(off,vid);src=sites.get(locations.get(key))
    p=predictions[off][t];points=p['predicted_positions_by_vehicle'][vid];margin=p['position_uncertainty_margin']
    if src is not None and e['state_version_before']>0 and p['prediction_confidence_by_vehicle'][vid]>=.5:
     vehicle=next(v for v in windows[off]['frames'][t]['vehicles'] if v.vehicle_id==vid)
     inside=hypot(vehicle.position_x-src.position_x,vehicle.position_y-src.position_y)<=src.coverage_radius
     exits=[i+1 for i,point in enumerate(points) if hypot(point[0]-src.position_x,point[1]-src.position_y)>src.coverage_radius+margin]
     actual=[]
     for delta in range(1,min(provider.horizon,len(windows[off]['frames'])-t-1)+1):
      future=next(v for v in windows[off]['frames'][t+delta]['vehicles'] if v.vehicle_id==vid)
      if hypot(future.position_x-src.position_x,future.position_y-src.position_y)>src.coverage_radius:
       destinations=[rid for rid,row in sites.items() if rid!=src.rsu_id and hypot(future.position_x-row.position_x,future.position_y-row.position_y)<=row.coverage_radius]
       if destinations:actual.append((delta,destinations))
     if inside and actual:
      true_cases.append(dict(seed=seed,frame_offset=off,vehicle_id=vid,frame_index=t,source=src.rsu_id,
       actual_eta=actual[0][0],actual_destinations=actual[0][1],forecast_exit=bool(exits),
       forecast_eta=exits[0] if exits else None,payload_bytes_before=payloads.get(key,0)))
     if inside and exits and t+exits[0]<len(windows[off]['frames']):
      eta=exits[0];point=points[eta-1]
      dest=[rid for rid,row in sites.items() if rid!=src.rsu_id and hypot(point[0]-row.position_x,point[1]-row.position_y)<=row.coverage_radius-margin]
      if dest:
       size=payloads.get(key,0);cases.append(dict(seed=seed,frame_offset=off,vehicle_id=vid,frame_index=t,source=src.rsu_id,eta=eta,
           possible_destination_ids=dest,payload_bytes_before=size,full_state_frames=ceil(max(1,size)/1024),
           gate_candidate_count=len(e['control_planning']['migration_candidates'])))
    if e.get('output_source_rsu_id'):locations[key]=e['output_source_rsu_id']
    elif e['status'] in ('rsu_compute','output_ready'):locations[key]=e['target_rsu_id']
    payloads[key]=e['state_payload_bytes']
 report=dict(experiment=experiment,scope='read-only actual/forecast coverage inventory at fitted horizon; actual future never enters policy; not net-benefit evidence',actual_candidate_request_frames=len(true_cases),actual_cases=true_cases,candidate_request_frames=len(cases),
       larger_full_copy_than_eta=sum(c['full_state_frames']>c['eta'] for c in cases),cases=cases)
 out=ROOT/f'artifacts/joint_coordinated_audit_20261015/{key}_opportunities.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('cases','actual_cases')},indent=2))
if __name__=='__main__':main()
