"""Read-only decomposition of a prespecified fixed-weight prediction intervention."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sys
root=Path(__file__).resolve().parents[1];base=root/'artifacts/joint_coordinated_audit_20261015';groups=defaultdict(list)
experiment=sys.argv[1] if len(sys.argv)>1 else 'joint_coordinated_partial_control_v2'
key=experiment.replace('/','_')
suffix='_'+key if experiment!='joint_coordinated_control' else ''
for seed in (7,17,27):
 p=next((root/f'artifacts/{experiment}/seed{seed}').glob('*/events.jsonl'))
 for line in p.open():
  e=json.loads(line)
  if e['agent']=='sa_ghmappo':groups[(seed,e['frame_offset'],'on')].append(e)
for line in (base/f'frozen_prediction_off_events{suffix}.jsonl').open():
 e=json.loads(line);groups[(e['seed'],e['frame_offset'],'off')].append(e)
assert len(groups)==48, 'all 24 on/off cells required'
report=[]
for seed in (7,17,27):
 for off in sorted({k[1] for k in groups}):
  on=groups[(seed,off,'on')];disabled=groups[(seed,off,'off')]
  strip=lambda e:{k:v for k,v in e.items() if k not in ('agent','seed','episode','frame_offset')}
  if [strip(e) for e in on]==[strip(e) for e in disabled]:continue
  def metrics(events):
   frame={e['frame_index']:e['frame_cost_receipt'] for e in events if 'frame_cost_receipt' in e}
   parts=Counter()
   for receipt in frame.values():parts.update(receipt['reward_components'])
   completions={e['vehicle_id']:e['frame_index'] for e in events if e['workflow_completed']}
   return dict(reward_components=dict(parts),request_completion_frames=completions,state_link_frames=sum(e['state_transfer_link_frames'] for e in events),model_link_frames=sum(r['model_transfer_link_frames'] for r in frame.values()),nodes=sum(e['status'] in ('node_completed','vehicle_fallback_completed') for e in events))
  changes=[];offmap={(e['frame_index'],e['vehicle_id']):e for e in disabled}
  for e in on:
   other=offmap.get((e['frame_index'],e['vehicle_id']))
   if other is None:continue
   if e['control']['prepare_target'] is not None or (e['control']!=other['control'] and not changes):
    changes.append(dict(frame=e['frame_index'],vehicle_id=e['vehicle_id'],on_control=e['control'],off_control=other['control'],on_status=e['status'],off_status=other['status'],on_state_transfers=e['state_transfers'],off_state_transfers=other['state_transfers']))
  report.append(dict(seed=seed,frame_offset=off,on=metrics(on),off=metrics(disabled),selected_changes=changes))
(base/f'{key}_prediction_effects.json').write_text(json.dumps(dict(scope='fixed-weight combined prediction features/mask intervention; not isolated causal state-action effect',different_event_cells=len(report),rows=report),indent=2)+'\n')
for r in report:print(r['seed'],r['frame_offset'],r['on'],r['off'])
