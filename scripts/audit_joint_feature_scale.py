"""Read-only initial-graph scale witness; never fits normalization or policy."""
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts

torch.set_num_threads(1)
experiment=sys.argv[1] if len(sys.argv)>1 else 'joint_coordinated_partial_control_v2'
source=json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
splits,workflows,rsus=_load_protocol(arguments(source['command'][1:]));w=splits['dev'][0];rows=[]
for seed in (7,17,27):
 p=next((ROOT/f'artifacts/{experiment}/seed{seed}').glob('*/manifest.json'));m=json.loads(p.read_text());path=p.parent/f'sa_ghmappo_seed{seed}_ep128.pt'
 saved=torch.load(path,weights_only=False);agent=build_agent('sa_ghmappo',control_contract=m['action_contract'],coordinated_controls=True,**{k:v for k,v in saved['options'].items() if k!='method'});agent=type(agent).from_checkpoint(path)
 env=MultiRequestTargetControlEnv(frames=w['frames'],vehicle_workflows=dict(zip(w['vehicle_ids'],workflows)),rsu_states=rsus,costs=ReplayCosts(**m['costs']['replay_costs']),**m['environment_options'])
 graphs=[];outputs=[]
 with torch.no_grad():
  for vid,decision in env.decision_states().items():
   data,*_=agent._coordinated_features(decision,0,{})
   graphs.append(torch.as_tensor(data['graph'])[torch.as_tensor(data['graph_mask'])]);outputs.append(agent.graph_encoder(torch.as_tensor(data['graph']))[torch.as_tensor(data['graph_mask'])])
 g=torch.cat(graphs);h=torch.cat(outputs)
 rows.append(dict(seed=seed,node_count=len(g),scaled_feature_min=g.min(0).values.tolist(),scaled_feature_max=g.max(0).values.tolist(),encoder_tanh_abs_over_099_fraction=float((h.abs()>.99).float().mean())))
out=dict(scope='first dev initial 70 known graph nodes only; descriptive saturation witness, not proven reward cause',source_commit=m['git_commit'],rows=rows)
(ROOT/f'artifacts/joint_coordinated_audit_20261015/{experiment}_feature_scale.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
