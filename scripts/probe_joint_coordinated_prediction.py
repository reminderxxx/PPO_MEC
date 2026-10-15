"""Prespecified fixed-weight prediction deployment ablation; no optimization."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol,_hash_file
from scripts.train_joint_parameterized_control import episode
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts

def main():
 torch.set_num_threads(1)
 destination=ROOT/'artifacts/joint_coordinated_audit_20261015'
 experiment=sys.argv[1] if len(sys.argv)>1 else 'joint_coordinated_control'
 partial=experiment!='joint_coordinated_control'
 key=experiment.replace('/','_')
 suffix='_'+key if partial else ''
 plan_path=destination/(f'support_plan_{key}.json' if partial else 'support_plan.json')
 plan=json.loads(plan_path.read_text())
 source=json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
 splits,workflows,rsus=_load_protocol(arguments(source['command'][1:]));rows=[]
 with (destination/f'frozen_prediction_off_events{suffix}.jsonl').open('w') as events:
  for seed in (7,17,27):
   paths=list((ROOT/f'artifacts/{experiment}/seed{seed}').glob('*/manifest.json'))
   assert len(paths)==1, 'exactly one prespecified run per seed required'
   manifest=paths[0]
   m=json.loads(manifest.read_text());directory=manifest.parent
   assert m['status'] in ('running','complete') and not (directory/'run_abort.json').exists()
   assert m['git_commit'].startswith(plan['source_commit']) and plan['seeds']==[7,17,27]
   path=directory/f'sa_ghmappo_seed{seed}_ep128.pt';saved=torch.load(path,weights_only=False)
   checkpoint_hash=_hash_file(path)
   cls=type(build_agent('sa_ghmappo',control_contract=m['action_contract'],coordinated_controls=True,scheduled_controls=m.get('scheduled_controls',False),**{k:v for k,v in saved['options'].items() if k!='method'}))
   agent=cls.from_checkpoint(path);agent.prediction_enabled=False
   trained_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in agent.state_dict().values())).hexdigest()
   provider=LearnedMobilityPredictor(torch.load(directory/'predictor.pt',weights_only=False))
   original={r['frame_offset']:r for r in (json.loads(s) for s in (directory/'dev.jsonl').read_text().splitlines()) if r['agent']=='sa_ghmappo' and r['episode']==128}
   assert set(original)=={w['frame_offset'] for w in splits['dev']}, 'all fixed-final full dev rows must exist'
   for window in splits['dev']:
    env=MultiRequestTargetControlEnv(frames=window['frames'],vehicle_workflows=dict(zip(window['vehicle_ids'],workflows)),
      rsu_states=rsus,costs=ReplayCosts(**m['costs']['replay_costs']),prediction_provider=provider,**m['environment_options'])
    row,_=episode(env,agent,deterministic=True,event_file=events,
      identity=dict(agent='frozen_full_prediction_off',seed=seed,episode=128,frame_offset=window['frame_offset']))
    on=original[window['frame_offset']]
    rows.append(dict(seed=seed,frame_offset=window['frame_offset'],checkpoint_sha256=checkpoint_hash,
      full_reward=on['reward'],prediction_off_reward=row['reward'],reward_difference=on['reward']-row['reward'],
      full_completed_requests=on['completed_requests'],prediction_off_completed_requests=row['completed_requests'],
      full_opportunities=on['mechanisms']['migration_opportunities'],full_prepare_commands=on['mechanisms']['prepare_commands']))
   assert trained_hash==hashlib.sha256(b''.join(v.numpy().tobytes() for v in agent.state_dict().values())).hexdigest()
   assert checkpoint_hash==_hash_file(path)
 report=dict(finished_at=datetime.now(timezone.utc).isoformat(),support_plan_sha256=_hash_file(plan_path),
    scope='fixed trained full weights; no parameter updates or checkpoint selection; all 24 dev cells',
    mean_reward_difference=float(np.mean([r['reward_difference'] for r in rows])),
    reward_wins=sum(r['reward_difference']>1e-9 for r in rows),reward_losses=sum(r['reward_difference']< -1e-9 for r in rows),
    mean_completed_difference=float(np.mean([r['full_completed_requests']-r['prediction_off_completed_requests'] for r in rows])),rows=rows)
 (destination/f'frozen_prediction_probe{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
